import time
import json
import asyncio
from fastapi import APIRouter, Depends, HTTPException, Query
from starlette.responses import StreamingResponse
from app.config import Settings, get_settings
from app.api.schemas.query import QueryRequest, QueryResponse, ChunkSource, QueryTimings
from app.dependencies import (
    get_retriever,
    get_reranker,
    get_llm,
    get_query_processor,
    get_semantic_cache,
    get_crag_gate,
    get_groundedness_verifier,
    get_embedder,
    get_parent_store,
)
from app.core.retriever import DualRetriever
from app.core.fusion import reciprocal_rank_fusion
from app.core.reranker import CrossEncoderReranker
from app.core.context_builder import ContextBuilder
from app.core.generator import BaseLLMGenerator
from app.core.query_expansion import QueryProcessor
from app.core.semantic_cache import SemanticQueryCache
from app.core.guardrails import RetrievalConfidenceGate, GroundednessVerifier
from app.core.parent_store import ParentDocumentStore
from app.core.embedder import Embedder
from app.core.metrics import (
    QUERY_COUNTER,
    QUERY_LATENCY,
    CACHE_HITS,
    CACHE_MISSES,
    GROUNDEDNESS_HISTOGRAM,
)
from app.utils.logger import logger
from app.utils.timing import measure_ms

router = APIRouter(prefix="/query", tags=["Query"])


@router.post("", response_model=QueryResponse)
async def query_pipeline(
    request: QueryRequest,
    settings: Settings = Depends(get_settings),
    retriever: DualRetriever = Depends(get_retriever),
    reranker: CrossEncoderReranker = Depends(get_reranker),
    llm: BaseLLMGenerator = Depends(get_llm),
    query_processor: QueryProcessor = Depends(get_query_processor),
    semantic_cache: SemanticQueryCache = Depends(get_semantic_cache),
    crag_gate: RetrievalConfidenceGate = Depends(get_crag_gate),
    groundedness_verifier: GroundednessVerifier = Depends(get_groundedness_verifier),
    embedder: Embedder = Depends(get_embedder),
    parent_store: ParentDocumentStore = Depends(get_parent_store),
):
    """
    Executes the full production Hybrid-Search RAG pipeline:
    1. Semantic Query Caching check (sub-3ms response on cache hit)
    2. Query enhancement (HyDE / Multi-Query)
    3. Dual parallel retrieval: Dense (ChromaDB) + Sparse (BM25)
    4. Reciprocal Rank Fusion (RRF)
    5. Cross-Encoder re-ranking
    6. CRAG confidence gating (abstain if evidence is insufficient)
    7. Parent-Document resolution (Small-to-Big context expansion)
    8. 'Lost in the Middle' context assembly
    9. LLM answer synthesis
    10. Groundedness & faithfulness verification
    11. Prometheus metrics emission
    """
    total_start = time.perf_counter()
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    final_n = request.top_n or settings.FINAL_TOP_N

    # 1. Semantic Cache Check
    query_emb = await asyncio.to_thread(embedder.embed_query, query)
    if request.enable_cache:
        cached_result = semantic_cache.lookup(query_emb)
        if cached_result:
            cached_data, sim_score, orig_query = cached_result
            cached_resp = QueryResponse(**cached_data)
            cached_resp.cached = True
            cached_resp.timings.total_ms = round((time.perf_counter() - total_start) * 1000, 2)
            CACHE_HITS.inc()
            QUERY_COUNTER.labels(status="cached").inc()
            QUERY_LATENCY.observe(cached_resp.timings.total_ms / 1000.0)
            logger.info(f"Serving query '{query}' from semantic cache in {cached_resp.timings.total_ms}ms")
            return cached_resp
        else:
            CACHE_MISSES.inc()

    # 2. Query Enhancement (HyDE / Multi-Query)
    dense_query_text = query
    queries_to_search = [query]
    with measure_ms() as m_enhancement:
        if request.use_hyde:
            dense_query_text = await query_processor.generate_hypothetical_document(query)
        elif request.expand_query:
            queries_to_search = await query_processor.expand_query(query, num_variants=2)

    # 3. Parallel Dual Retrieval
    all_dense_results = []
    all_bm25_results = []
    retrieval_timings = {"dense_retrieval_ms": 0.0, "bm25_retrieval_ms": 0.0, "parallel_retrieval_ms": 0.0}

    for q_variant in queries_to_search:
        dense_q = dense_query_text if q_variant == query else q_variant
        d_res, b_res, r_timings = await retriever.retrieve_parallel(
            query=q_variant,
            dense_k=settings.DENSE_TOP_K,
            bm25_k=settings.BM25_TOP_K,
            dense_query_override=dense_q,
            doc_id_filter=request.doc_id_filter,
        )
        all_dense_results.append(d_res)
        all_bm25_results.append(b_res)
        retrieval_timings["dense_retrieval_ms"] += r_timings["dense_retrieval_ms"]
        retrieval_timings["bm25_retrieval_ms"] += r_timings["bm25_retrieval_ms"]
        retrieval_timings["parallel_retrieval_ms"] += r_timings["parallel_retrieval_ms"]

    # 4. Reciprocal Rank Fusion across all candidate lists
    candidate_lists = all_dense_results + all_bm25_results
    with measure_ms() as m_fusion:
        fused_candidates = reciprocal_rank_fusion(
            ranked_lists=candidate_lists,
            k=settings.RRF_K,
            top_n=settings.FUSION_TOP_N,
        )

    # 5. Cross-Encoder Re-Ranking
    with measure_ms() as m_rerank:
        reranked_chunks = await asyncio.to_thread(
            reranker.rerank,
            query=query,
            candidates=fused_candidates,
            top_n=final_n,
        )

    # 6. CRAG Confidence Gate
    crag_eval = crag_gate.evaluate_candidates(reranked_chunks)
    retrieval_confidence = crag_eval["decision"]

    if request.enforce_crag and retrieval_confidence == "INSUFFICIENT":
        answer = "Based on the verified knowledge base, no relevant documents were found to answer this question."
        groundedness = {"groundedness_score": 1.0, "status": "SAFE_ABSTENTION"}
        m_gen = {"elapsed_ms": 0.0}
        QUERY_COUNTER.labels(status="abstain").inc()
    else:
        # 7. Parent-Document Resolution (Small-to-Big Retrieval)
        context_chunks = parent_store.resolve_to_parent_chunks(reranked_chunks)

        # 8. Context Building with Lost-in-the-Middle Reordering
        context_str = ContextBuilder.build_context_string(context_chunks, apply_lost_in_middle=True)

        # 9. LLM Answer Generation
        with measure_ms() as m_gen:
            answer = await llm.generate(query=query, context=context_str)

        # 10. Groundedness Verification
        groundedness = groundedness_verifier.verify(answer=answer, context=context_str)
        QUERY_COUNTER.labels(status="success").inc()

    total_elapsed = round((time.perf_counter() - total_start) * 1000, 2)
    QUERY_LATENCY.observe(total_elapsed / 1000.0)
    GROUNDEDNESS_HISTOGRAM.observe(groundedness["groundedness_score"])

    # Format sources for response
    sources = [
        ChunkSource(
            chunk_id=sc.chunk.id,
            doc_id=sc.chunk.doc_id,
            source_file=sc.chunk.source_file,
            chunk_index=sc.chunk.chunk_index,
            text=sc.chunk.text,
            score=sc.score,
            retriever_origin=sc.retriever,
        )
        for sc in reranked_chunks
    ]

    timings = QueryTimings(
        query_enhancement_ms=m_enhancement["elapsed_ms"],
        dense_retrieval_ms=round(retrieval_timings["dense_retrieval_ms"], 2),
        bm25_retrieval_ms=round(retrieval_timings["bm25_retrieval_ms"], 2),
        parallel_retrieval_ms=round(retrieval_timings["parallel_retrieval_ms"], 2),
        fusion_ms=m_fusion["elapsed_ms"],
        rerank_ms=m_rerank["elapsed_ms"],
        generation_ms=m_gen["elapsed_ms"],
        total_ms=total_elapsed,
    )

    response = QueryResponse(
        query=query,
        answer=answer,
        sources=sources,
        timings=timings,
        cached=False,
        retrieval_confidence=retrieval_confidence,
        groundedness_score=groundedness["groundedness_score"],
        groundedness_status=groundedness["status"],
    )

    # Store in semantic cache if successful
    if request.enable_cache and retrieval_confidence != "INSUFFICIENT":
        semantic_cache.put(query=query, query_embedding=query_emb, response_data=response.model_dump())

    return response


@router.get("/stream")
async def stream_query_pipeline(
    q: str = Query(..., min_length=1, description="Query string to search and answer"),
    settings: Settings = Depends(get_settings),
    retriever: DualRetriever = Depends(get_retriever),
    reranker: CrossEncoderReranker = Depends(get_reranker),
    llm: BaseLLMGenerator = Depends(get_llm),
):
    """
    Streams the LLM response tokens in real-time via Server-Sent Events (SSE).
    Emits token events, followed by metadata and source attribution.
    """
    query = q.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    async def event_generator():
        try:
            # 1. Dual Retrieval
            dense_results, bm25_results, _ = await retriever.retrieve_parallel(
                query=query,
                dense_k=settings.DENSE_TOP_K,
                bm25_k=settings.BM25_TOP_K,
            )

            # 2. RRF Fusion
            fused_candidates = reciprocal_rank_fusion(
                ranked_lists=[dense_results, bm25_results],
                k=settings.RRF_K,
                top_n=settings.FUSION_TOP_N,
            )

            # 3. Cross-Encoder Re-Ranking
            reranked_chunks = await asyncio.to_thread(
                reranker.rerank,
                query=query,
                candidates=fused_candidates,
                top_n=settings.FINAL_TOP_N,
            )

            # 4. Context assembly
            context_str = ContextBuilder.build_context_string(reranked_chunks, apply_lost_in_middle=True)

            # 5. Stream tokens
            yield f"event: status\ndata: {json.dumps({'message': 'Generating answer...'})}\n\n"
            async for token in llm.generate_stream(query=query, context=context_str):
                yield f"event: token\ndata: {json.dumps({'token': token})}\n\n"

            # 6. Stream sources at the end
            sources_data = [
                {
                    "chunk_id": sc.chunk.id,
                    "source_file": sc.chunk.source_file,
                    "chunk_index": sc.chunk.chunk_index,
                    "score": sc.score,
                    "retriever": sc.retriever,
                    "preview": sc.chunk.text[:120] + "..." if len(sc.chunk.text) > 120 else sc.chunk.text
                }
                for sc in reranked_chunks
            ]
            yield f"event: sources\ndata: {json.dumps(sources_data)}\n\n"
            yield f"event: done\ndata: {json.dumps({'status': 'completed'})}\n\n"

        except Exception as e:
            logger.error(f"Streaming query failed: {e}", exc_info=True)
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )

