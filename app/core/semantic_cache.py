import time
import threading
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
from app.utils.logger import logger


class CachedResponse:
    def __init__(self, query: str, embedding: List[float], response_data: Dict[str, Any]):
        self.query = query
        self.embedding = np.array(embedding, dtype=np.float32)
        # Normalize to ensure dot product equals cosine similarity
        norm = np.linalg.norm(self.embedding)
        if norm > 0:
            self.embedding = self.embedding / norm
        self.response_data = response_data
        self.created_at = time.time()
        self.hits = 0


class SemanticQueryCache:
    """
    In-memory semantic query cache with cosine similarity matching.
    Returns sub-millisecond cached responses for semantically equivalent queries.
    """

    def __init__(self, similarity_threshold: float = 0.94, max_entries: int = 1000):
        self.similarity_threshold = similarity_threshold
        self.max_entries = max_entries
        self.entries: List[CachedResponse] = []
        self._lock = threading.Lock()
        self.total_hits = 0
        self.total_misses = 0

    def lookup(
        self,
        query_embedding: List[float],
        threshold: Optional[float] = None
    ) -> Optional[Tuple[Dict[str, Any], float, str]]:
        """
        Searches cache for a query with cosine similarity >= threshold.
        Returns: (cached_response_dict, similarity_score, original_cached_query) or None
        """
        target_thresh = threshold if threshold is not None else self.similarity_threshold

        with self._lock:
            if not self.entries:
                self.total_misses += 1
                return None

            q_vec = np.array(query_embedding, dtype=np.float32)
            norm = np.linalg.norm(q_vec)
            if norm > 0:
                q_vec = q_vec / norm

            # Stack cached embeddings into matrix (N, D)
            matrix = np.vstack([e.embedding for e in self.entries])
            similarities = np.dot(matrix, q_vec)

            best_idx = int(np.argmax(similarities))
            best_sim = float(similarities[best_idx])

            if best_sim >= target_thresh:
                matched_entry = self.entries[best_idx]
                matched_entry.hits += 1
                self.total_hits += 1
                logger.info(
                    f"🎯 Semantic Cache HIT! (Similarity: {best_sim:.4f} >= {target_thresh}) "
                    f"Matched with previous query: '{matched_entry.query}'"
                )
                return matched_entry.response_data, best_sim, matched_entry.query

            self.total_misses += 1
            return None

    def put(self, query: str, query_embedding: List[float], response_data: Dict[str, Any]) -> None:
        """Stores query embedding and response in cache with LRU eviction."""
        with self._lock:
            # Evict oldest if full
            if len(self.entries) >= self.max_entries:
                self.entries.pop(0)

            entry = CachedResponse(query=query, embedding=query_embedding, response_data=response_data)
            self.entries.append(entry)
            logger.info(f"Cached semantic entry for query: '{query}' (Total cached: {len(self.entries)})")

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            total_requests = self.total_hits + self.total_misses
            hit_rate = (self.total_hits / total_requests * 100) if total_requests > 0 else 0.0
            return {
                "cached_entries": len(self.entries),
                "total_hits": self.total_hits,
                "total_misses": self.total_misses,
                "hit_rate_pct": round(hit_rate, 2),
                "similarity_threshold": self.similarity_threshold,
            }

    def clear(self) -> None:
        with self._lock:
            self.entries.clear()
            self.total_hits = 0
            self.total_misses = 0

