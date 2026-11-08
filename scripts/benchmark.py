import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import asyncio
import numpy as np
from app.dependencies import (
    get_ingestion_pipeline,
    get_retriever,
    get_reranker,
    get_vector_store,
    get_bm25_index,
    get_embedder,
)
from app.core.fusion import reciprocal_rank_fusion
from app.core.context_builder import ContextBuilder

BENCHMARK_DOCS = [
    (
        "auth_guide.txt",
        "Authentication Protocol Guide. If a user enters an expired JWT token, the server returns error code ERR_404_AUTH. "
        "To resolve ERR_404_AUTH, clear the local storage session and request a fresh OAuth bearer token from /oauth/token. "
        "Session hijacking can be mitigated by binding tokens to the client IP address."
    ),
    (
        "hardware_specs.txt",
        "Industrial Hardware Specifications. The high-performance centrifuge model X-500 operates at 15000 RPM. "
        "Model X-500 requires standard 220V power supply and maintenance every 6 months. "
        "In contrast, the budget model X-200 operates at 8000 RPM and is suitable for light laboratory assays."
    ),
    (
        "climate_study.txt",
        "Global Climate Change and Oceanography. Rising atmospheric greenhouse gas concentrations cause thermal expansion "
        "of ocean waters and melting of polar ice sheets, driving sea level rise across coastal environments worldwide."
    ),
    (
        "network_err.txt",
        "Network Failure Manual. Socket timeout exceptions trigger ERR_502_GATEWAY in proxy clusters. "
        "Check reverse proxy keepalive timeouts and upstream backend socket buffers."
    ),
]

BENCHMARK_QUERIES = [
    # Keyword-heavy queries (BM25 excels here)
    ("What does ERR_404_AUTH mean and how do I fix it?", "ERR_404_AUTH"),
    ("What are the RPM specs of centrifuge X-500?", "X-500"),
    ("Which error occurs on socket timeout in reverse proxies?", "ERR_502_GATEWAY"),
    # Conceptual/semantic queries (Dense vector excels here)
    ("How does atmospheric warming affect coastlines and ocean levels?", "greenhouse gas"),
    ("What precautions prevent unauthorized access using existing sessions?", "hijacking"),
]


async def run_benchmark():
    print("=" * 70)
    print("🚀 FusionRAG Performance & Accuracy Benchmark")
    print("=" * 70)

    pipeline = get_ingestion_pipeline()
    retriever = get_retriever()
    reranker = get_reranker()
    vector_store = get_vector_store()
    bm25 = get_bm25_index()
    embedder = get_embedder()

    # 1. Ingest benchmark corpus
    print("\n[1/3] Ingesting benchmark corpus...")
    for filename, text in BENCHMARK_DOCS:
        res = pipeline.ingest_bytes(
            file_bytes=text.encode("utf-8"),
            filename=filename,
            content_type="text/plain"
        )
        print(f"  ✓ Ingested '{filename}': {res.chunks_created} chunks")

    print(f"\nIndexes ready: ChromaDB={vector_store.count()} chunks, BM25={bm25.count()} chunks")

    # 2. Warm up models
    print("\n[2/3] Warming up models...")
    _ = embedder.embed_query("warmup")
    _ = reranker.model

    # 3. Run queries across modes
    print("\n[3/3] Evaluating Retrieval Methods across benchmark queries...")

    dense_latencies = []
    bm25_latencies = []
    hybrid_latencies = []

    dense_hits = 0
    bm25_hits = 0
    hybrid_hits = 0
    total_queries = len(BENCHMARK_QUERIES)

    for query_text, expected_keyword in BENCHMARK_QUERIES:
        print(f"\nQuery: '{query_text}' (Target: '{expected_keyword}')")

        # Dense only
        t0 = time.perf_counter()
        q_emb = embedder.embed_query(query_text)
        dense_res = vector_store.search(q_emb, top_k=5)
        dense_time = (time.perf_counter() - t0) * 1000
        dense_latencies.append(dense_time)
        dense_found = any(expected_keyword.lower() in c.chunk.text.lower() for c in dense_res)
        if dense_found:
            dense_hits += 1

        # BM25 only
        t0 = time.perf_counter()
        bm25_res = bm25.search(query_text, top_k=5)
        bm25_time = (time.perf_counter() - t0) * 1000
        bm25_latencies.append(bm25_time)
        bm25_found = any(expected_keyword.lower() in c.chunk.text.lower() for c in bm25_res)
        if bm25_found:
            bm25_hits += 1

        # Hybrid (Parallel Retrieval + RRF + Cross-Encoder)
        t0 = time.perf_counter()
        d_res, b_res, _ = await retriever.retrieve_parallel(query_text, dense_k=20, bm25_k=20)
        fused = reciprocal_rank_fusion([d_res, b_res], k=60, top_n=10)
        reranked = reranker.rerank(query_text, fused, top_n=5)
        hybrid_time = (time.perf_counter() - t0) * 1000
        hybrid_latencies.append(hybrid_time)
        hybrid_found = any(expected_keyword.lower() in c.chunk.text.lower() for c in reranked)
        if hybrid_found:
            hybrid_hits += 1

        print(f"  Dense:  found={dense_found} ({dense_time:.1f}ms)")
        print(f"  BM25:   found={bm25_found} ({bm25_time:.1f}ms)")
        print(f"  Hybrid: found={hybrid_found} ({hybrid_time:.1f}ms)")

    # Print summary table
    print("\n" + "=" * 70)
    print("📊 BENCHMARK SUMMARY RESULTS")
    print("=" * 70)
    print(f"{'Method':<15} | {'Recall Rate':<12} | {'P50 (ms)':<10} | {'P95 (ms)':<10} | {'Max (ms)':<10}")
    print("-" * 70)
    for name, hits, lats in [
        ("Dense (Chroma)", f"{dense_hits}/{total_queries} ({dense_hits/total_queries*100:.0f}%)", dense_latencies),
        ("BM25 (Sparse)", f"{bm25_hits}/{total_queries} ({bm25_hits/total_queries*100:.0f}%)", bm25_latencies),
        ("FusionRAG (All)", f"{hybrid_hits}/{total_queries} ({hybrid_hits/total_queries*100:.0f}%)", hybrid_latencies),
    ]:
        p50 = np.percentile(lats, 50)
        p95 = np.percentile(lats, 95)
        max_t = np.max(lats)
        print(f"{name:<15} | {hits:<12} | {p50:<10.1f} | {p95:<10.1f} | {max_t:<10.1f}")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(run_benchmark())

