import threading
from typing import List, Optional
import torch
from sentence_transformers import SentenceTransformer
from app.utils.logger import logger


class Embedder:
    """Wrapper around SentenceTransformer with thread-safe lazy loading and batching."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: Optional[str] = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model: Optional[SentenceTransformer] = None
        self._lock = threading.Lock()
        logger.info(f"Embedder initialized with model '{self.model_name}' on device '{self.device}'")

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            with self._lock:
                if self._model is None:
                    logger.info(f"Loading SentenceTransformer: {self.model_name}...")
                    self._model = SentenceTransformer(self.model_name, device=self.device)
                    logger.info(f"SentenceTransformer '{self.model_name}' loaded successfully.")
        return self._model

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Generates embeddings for a list of texts in batches."""
        if not texts:
            return []
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        """Generates embedding for a single query string."""
        embedding = self.model.encode(
            query,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embedding.tolist()

