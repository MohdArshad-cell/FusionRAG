from typing import List, Dict, Tuple
from app.core.models import Chunk, ScoredChunk
from app.utils.logger import logger


def reciprocal_rank_fusion(
    ranked_lists: List[List[ScoredChunk]],
    k: int = 60,
    top_n: int = 10,
) -> List[ScoredChunk]:
    """
    Merges multiple ranked lists of ScoredChunks using Reciprocal Rank Fusion (RRF).
    
    Formula:
        RRF_Score(d) = sum_{m in M} (1 / (k + rank_m(d)))
    
    Args:
        ranked_lists: List of candidate chunk lists, each sorted by relevance descending.
        k: Smoothing constant (standard default is 60).
        top_n: Number of fused candidates to return.
    
    Returns:
        List of ScoredChunk with RRF scores, sorted descending.
    """
    # Accumulate scores and map chunk_id to Chunk object
    rrf_scores: Dict[str, float] = {}
    chunk_store: Dict[str, Chunk] = {}
    origins: Dict[str, List[str]] = {}

    for ranked_list in ranked_lists:
        for rank_idx, scored_chunk in enumerate(ranked_list, start=1):
            chunk = scored_chunk.chunk
            c_id = chunk.id

            if c_id not in chunk_store:
                chunk_store[c_id] = chunk
                rrf_scores[c_id] = 0.0
                origins[c_id] = []

            # RRF contribution
            score_contribution = 1.0 / (k + rank_idx)
            rrf_scores[c_id] += score_contribution
            origins[c_id].append(f"{scored_chunk.retriever}#rank{rank_idx}")

    # Sort chunks by descending RRF score
    sorted_items = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)

    fused_results: List[ScoredChunk] = []
    for rank_idx, (c_id, score) in enumerate(sorted_items[:top_n], start=1):
        fused_chunk = chunk_store[c_id]
        origin_str = "+".join(origins[c_id])
        fused_results.append(
            ScoredChunk(
                chunk=fused_chunk,
                score=round(score, 6),
                rank=rank_idx,
                retriever=f"rrf({origin_str})"
            )
        )

    logger.info(f"RRF fused {len(chunk_store)} unique items across {len(ranked_lists)} lists into top-{len(fused_results)}")
    return fused_results

