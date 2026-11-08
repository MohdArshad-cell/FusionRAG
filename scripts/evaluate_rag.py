import os
import sys
import asyncio
import time
from typing import List, Dict, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.dependencies import (
    get_retriever,
    get_reranker,
    get_llm,
    get_groundedness_verifier,
    get_parent_store,
    warmup_services,
)
from app.core.fusion import reciprocal_rank_fusion
from app.core.context_builder import ContextBuilder

# Ground-truth evaluation dataset with expected answers and key facts
EVAL_DATASET = [
    {
        "query": "What is the operational ceiling and power requirement for centrifuge model X-500?",
        "expected_facts": ["15000 RPM", "220V"],
        "category": "Exact Technical Identifier"
    },
    {
        "query": "How do you resolve error code ERR_404_AUTH?",
        "expected_facts": ["session", "token", "oauth"],
        "category": "Error Code Remediation"
    },
    {
        "query": "What causes sea levels to rise according to oceanography studies?",
        "expected_facts": ["thermal expansion", "greenhouse gas", "ice"],
        "category": "Conceptual / Scientific Reasoning"
    },
    {
        "query": "Which HTTP error occurs when upstream reverse proxy sockets time out?",
        "expected_facts": ["ERR_502_GATEWAY", "proxy", "timeout"],
        "category": "Network Troubleshooting"
    },
    {
        "query": "What mathematical strategy mitigates thundering herd in distributed microservices retry?",
        "expected_facts": ["exponential backoff", "jitter"],
        "category": "System Architecture"
    }
]


def compute_context_relevance(context: str, expected_facts: List[str]) -> float:
    """Computes the percentage of ground-truth facts present in retrieved context."""
    if not context or not expected_facts:
        return 0.0
    context_lower = context.lower()
    hits = sum(1 for fact in expected_facts if fact.lower() in context_lower)
    return round(hits / len(expected_facts), 2)


def compute_answer_relevance(answer: str, expected_facts: List[str]) -> float:
    """Computes whether key expected facts are synthesized into the final answer."""
    if not answer or not expected_facts:
        return 0.0
    answer_lower = answer.lower()
    hits = sum(1 for fact in expected_facts if fact.lower() in answer_lower)
    return round(hits / len(expected_facts), 2)


async def run_evaluation():
    print("=" * 75)
    print("🔬 FusionRAG Automated Quality Evaluation Suite (RAG Triad)")
    print("   Metrics: Context Relevance | Groundedness | Answer Relevance")
    print("=" * 75)

    warmup_services()
    retriever = get_retriever()
    reranker = get_reranker()
    llm = get_llm()
    groundedness_verifier = get_groundedness_verifier()
    parent_store = get_parent_store()

    context_rel_scores = []
    groundedness_scores = []
    answer_rel_scores = []
    latencies = []

    print(f"\nEvaluating {len(EVAL_DATASET)} golden test queries...\n")
    print(f"{'Category':<32} | {'Context Rel':<11} | {'Grounded':<8} | {'Answer Rel':<10} | {'Latency':<8}")
    print("-" * 75)

    for item in EVAL_DATASET:
        query = item["query"]
        expected_facts = item["expected_facts"]
        category = item["category"]

        t0 = time.perf_counter()

        # 1. Dual Retrieval
        d_res, b_res, _ = await retriever.retrieve_parallel(query, dense_k=20, bm25_k=20)

        # 2. RRF Fusion
        fused = reciprocal_rank_fusion([d_res, b_res], k=60, top_n=10)

        # 3. Cross-Encoder Re-Ranking
        reranked = reranker.rerank(query, fused, top_n=5)

        # 4. Context assembly
        context_chunks = parent_store.resolve_to_parent_chunks(reranked)
        context_str = ContextBuilder.build_context_string(context_chunks, apply_lost_in_middle=True)

        # 5. LLM Answer Generation
        answer = await llm.generate(query, context_str)
        latency_ms = (time.perf_counter() - t0) * 1000

        # 6. Evaluation metrics computation
        c_rel = compute_context_relevance(context_str, expected_facts)
        g_eval = groundedness_verifier.verify(answer, context_str)
        g_score = g_eval["groundedness_score"]
        a_rel = compute_answer_relevance(answer, expected_facts)

        context_rel_scores.append(c_rel)
        groundedness_scores.append(g_score)
        answer_rel_scores.append(a_rel)
        latencies.append(latency_ms)

        print(f"{category:<32} | {c_rel * 100:>9.0f}% | {g_score * 100:>6.0f}% | {a_rel * 100:>8.0f}% | {latency_ms:>6.1f}ms")

    # Aggregate metrics
    mean_c_rel = sum(context_rel_scores) / len(context_rel_scores)
    mean_grounded = sum(groundedness_scores) / len(groundedness_scores)
    mean_a_rel = sum(answer_rel_scores) / len(answer_rel_scores)
    mean_lat = sum(latencies) / len(latencies)

    print("=" * 75)
    print("📊 AGGREGATE RAG QUALITY SCORES")
    print("=" * 75)
    print(f"  • Mean Context Relevance: {mean_c_rel * 100:.1f}%")
    print(f"  • Mean Groundedness / Faithfulness: {mean_grounded * 100:.1f}%")
    print(f"  • Mean Answer Relevance: {mean_a_rel * 100:.1f}%")
    print(f"  • Mean End-to-End Latency: {mean_lat:.1f}ms")
    print("=" * 75)


if __name__ == "__main__":
    asyncio.run(run_evaluation())
