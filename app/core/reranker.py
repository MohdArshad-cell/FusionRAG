import threading
from typing import List, Optional
import torch
from sentence_transformers import CrossEncoder
from app.core.models import ScoredChunk
from app.utils.logger import logger


class CrossEncoderReranker:
    """Re-ranks candidate chunks using a deep cross-encoder model."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model: Optional[CrossEncoder] = None
        self._lock = threading.Lock()
        logger.info(f"CrossEncoderReranker initialized with model '{self.model_name}' on device '{self.device}'")

    @property
    def model(self) -> CrossEncoder:
        if self._model is None:
            with self._lock:
                if self._model is None:
                    logger.info(f"Loading CrossEncoder: {self.model_name}...")
                    self._model = CrossEncoder(self.model_name, device=self.device)
                    logger.info(f"CrossEncoder '{self.model_name}' loaded successfully.")
        return self._model

    def rerank(
        self,
        query: str,
        candidates: List[ScoredChunk],
        top_n: int = 5
    ) -> List[ScoredChunk]:
        """
        Cross-scores (query, chunk.text) pairs and returns top_n candidates sorted by relevance.
        """
        if not candidates:
            return []

        # Form (query, document) pairs
        pairs = [(query, scored.chunk.text) for scored in candidates]

        try:
            raw_scores = self.model.predict(pairs, show_progress_bar=False)
            # raw_scores may be float or numpy array
            scores = [float(s) for s in raw_scores]
        except Exception as e:
            logger.error(f"CrossEncoder prediction failed: {e}. Falling back to previous ranking.")
            return candidates[:top_n]

        # Combine candidates with cross-encoder scores
        scored_pairs = list(zip(candidates, scores))
        scored_pairs.sort(key=lambda x: x[1], reverse=True)

        reranked_chunks: List[ScoredChunk] = []
        for rank_idx, (orig_scored, score) in enumerate(scored_pairs[:top_n], start=1):
            reranked_chunks.append(
                ScoredChunk(
                    chunk=orig_scored.chunk,
                    score=round(score, 4),
                    rank=rank_idx,
                    retriever="cross_encoder"
                )
            )

        logger.info(f"Cross-encoder reranked {len(candidates)} candidates down to top-{len(reranked_chunks)}")
        return reranked_chunks

