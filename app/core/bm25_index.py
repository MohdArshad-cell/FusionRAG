import os
import re
import pickle
import threading
from typing import List, Dict, Optional
from rank_bm25 import BM25Okapi
from app.core.models import Chunk, ScoredChunk
from app.utils.logger import logger


class BM25Index:
    """Manages an in-memory/persistent BM25Okapi index for lexical keyword search."""

    def __init__(self, persist_dir: str = "./data/bm25_index"):
        self.persist_dir = persist_dir
        self.index_file = os.path.join(self.persist_dir, "bm25_index.pkl")
        self._lock = threading.Lock()

        self.corpus_chunks: List[Chunk] = []
        self.tokenized_corpus: List[List[str]] = []
        self.bm25: Optional[BM25Okapi] = None
        self.chunk_map: Dict[str, Chunk] = {}

        os.makedirs(self.persist_dir, exist_ok=True)
        self._load_if_exists()

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """
        Tokenize text while keeping hyphenated terms, error codes (ERR_404_AUTH),
        and alphanumeric identifiers (X-500) intact.
        """
        if not text:
            return []
        text_lower = text.lower()
        # Match word characters, hyphens, and underscores together
        tokens = re.findall(r"[a-z0-9_\-]+", text_lower)
        return tokens

    def _load_if_exists(self) -> None:
        if os.path.exists(self.index_file):
            try:
                with open(self.index_file, "rb") as f:
                    data = pickle.load(f)
                    self.corpus_chunks = data.get("corpus_chunks", [])
                    self.tokenized_corpus = data.get("tokenized_corpus", [])
                    self.chunk_map = {c.id: c for c in self.corpus_chunks}
                    if self.tokenized_corpus:
                        self.bm25 = BM25Okapi(self.tokenized_corpus)
                logger.info(f"Loaded existing BM25 index with {len(self.corpus_chunks)} chunks.")
            except Exception as e:
                logger.warning(f"Could not load BM25 index: {e}. Starting fresh.")
                self.corpus_chunks = []
                self.tokenized_corpus = []
                self.bm25 = None
                self.chunk_map = {}

    def save(self) -> None:
        """Persists BM25 state to disk."""
        with self._lock:
            data = {
                "corpus_chunks": self.corpus_chunks,
                "tokenized_corpus": self.tokenized_corpus,
            }
            tmp_file = self.index_file + ".tmp"
            with open(tmp_file, "wb") as f:
                pickle.dump(data, f)
            os.replace(tmp_file, self.index_file)
            logger.info(f"BM25 index saved with {len(self.corpus_chunks)} chunks.")

    def add_chunks(self, chunks: List[Chunk]) -> None:
        """Adds new chunks to the BM25 index and updates the model."""
        if not chunks:
            return

        with self._lock:
            # Avoid duplicate chunk IDs
            existing_ids = {c.id for c in self.corpus_chunks}
            new_chunks = [c for c in chunks if c.id not in existing_ids]

            if not new_chunks:
                logger.info("No new unique chunks to add to BM25 index.")
                return

            for chunk in new_chunks:
                tokens = self.tokenize(chunk.text)
                self.corpus_chunks.append(chunk)
                self.tokenized_corpus.append(tokens)
                self.chunk_map[chunk.id] = chunk

            # Re-initialize BM25Okapi with updated tokenized corpus
            self.bm25 = BM25Okapi(self.tokenized_corpus)

        self.save()
        logger.info(f"Added {len(new_chunks)} chunks to BM25 index (total: {len(self.corpus_chunks)}).")

    def search(
        self,
        query: str,
        top_k: int = 20,
        doc_id_filter: Optional[str] = None
    ) -> List[ScoredChunk]:
        """Searches BM25 index and returns ScoredChunks sorted by BM25 score."""
        if not self.bm25 or not self.corpus_chunks:
            return []

        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)

        # Filter indices with score > 0 and matching doc_id_filter if provided
        indexed_scores = []
        for idx, score in enumerate(scores):
            if score > 0.0:
                if doc_id_filter and self.corpus_chunks[idx].doc_id != doc_id_filter:
                    continue
                indexed_scores.append((idx, score))

        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        effective_top_k = indexed_scores[:top_k]
        scored_chunks: List[ScoredChunk] = []

        for rank_idx, (idx, score) in enumerate(effective_top_k, start=1):
            chunk = self.corpus_chunks[idx]
            scored_chunks.append(
                ScoredChunk(
                    chunk=chunk,
                    score=float(score),
                    rank=rank_idx,
                    retriever="bm25"
                )
            )

        return scored_chunks

    def count(self) -> int:
        return len(self.corpus_chunks)

