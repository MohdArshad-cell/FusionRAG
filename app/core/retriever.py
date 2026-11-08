import asyncio
from typing import Tuple, List, Optional
from app.core.models import ScoredChunk
from app.core.embedder import Embedder
from app.core.vector_store import ChromaVectorStore
from app.core.bm25_index import BM25Index
from app.utils.logger import logger
from app.utils.timing import measure_ms


class DualRetriever:
    """Orchestrates parallel dense vector retrieval and sparse BM25 retrieval."""

    def __init__(
        self,
        embedder: Embedder,
        vector_store: ChromaVectorStore,
        bm25_index: BM25Index,
        dense_top_k: int = 20,
        bm25_top_k: int = 20,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.dense_top_k = dense_top_k
        self.bm25_top_k = bm25_top_k

    async def _search_dense(
        self,
        query: str,
        top_k: int,
        doc_id_filter: Optional[str] = None
    ) -> Tuple[List[ScoredChunk], float]:
        """Encodes query and queries ChromaDB."""
        with measure_ms() as m:
            query_embedding = await asyncio.to_thread(self.embedder.embed_query, query)
            where_filter = {"doc_id": doc_id_filter} if doc_id_filter else None
            results = await asyncio.to_thread(
                self.vector_store.search,
                query_embedding,
                top_k,
                where_filter
            )
        return results, m["elapsed_ms"]

    async def _search_bm25(
        self,
        query: str,
        top_k: int,
        doc_id_filter: Optional[str] = None
    ) -> Tuple[List[ScoredChunk], float]:
        """Queries BM25 index on threadpool."""
        with measure_ms() as m:
            results = await asyncio.to_thread(
                self.bm25_index.search,
                query,
                top_k,
                doc_id_filter
            )
        return results, m["elapsed_ms"]

    async def retrieve_parallel(
        self,
        query: str,
        dense_k: int = 20,
        bm25_k: int = 20,
        dense_query_override: Optional[str] = None,
        doc_id_filter: Optional[str] = None
    ) -> Tuple[List[ScoredChunk], List[ScoredChunk], dict]:
        """
        Executes dense semantic search and BM25 lexical search concurrently via asyncio.gather().
        Supports HyDE (dense_query_override) and metadata filtering (doc_id_filter).
        Returns (dense_results, bm25_results, timing_dict).
        """
        start_time = asyncio.get_event_loop().time()

        dense_search_text = dense_query_override or query
        dense_task = asyncio.create_task(self._search_dense(dense_search_text, dense_k, doc_id_filter))
        bm25_task = asyncio.create_task(self._search_bm25(query, bm25_k, doc_id_filter))

        (dense_results, dense_ms), (bm25_results, bm25_ms) = await asyncio.gather(
            dense_task, bm25_task
        )

        total_parallel_ms = round((asyncio.get_event_loop().time() - start_time) * 1000, 2)

        timings = {
            "dense_retrieval_ms": dense_ms,
            "bm25_retrieval_ms": bm25_ms,
            "parallel_retrieval_ms": total_parallel_ms,
        }

        logger.info(
            f"Dual retrieval finished in {total_parallel_ms}ms "
            f"(Dense: {len(dense_results)} in {dense_ms}ms, BM25: {len(bm25_results)} in {bm25_ms}ms)"
        )

        return dense_results, bm25_results, timings

