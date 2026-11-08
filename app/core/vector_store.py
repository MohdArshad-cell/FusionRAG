import os
from typing import List, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings
from app.core.models import Chunk, ScoredChunk
from app.utils.logger import logger


class ChromaVectorStore:
    """Wrapper around ChromaDB for dense vector storage and cosine similarity retrieval."""

    def __init__(self, persist_dir: str = "./data/chroma_db", collection_name: str = "fusionrag_docs"):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        os.makedirs(persist_dir, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"ChromaDB connected at '{self.persist_dir}' [Collection: {self.collection_name}, items: {self.collection.count()}]")

    def count(self) -> int:
        return self.collection.count()

    def add_chunks(self, chunks: List[Chunk], embeddings: List[List[float]]) -> None:
        """Batch upserts chunks and their embeddings into ChromaDB."""
        if not chunks:
            return

        ids = [c.id for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [
            {
                "doc_id": c.doc_id,
                "source_file": c.source_file,
                "chunk_index": c.chunk_index,
                "char_start": c.char_start,
                "char_end": c.char_end,
            }
            for c in chunks
        ]

        # ChromaDB accepts batches of up to ~5000 items
        batch_size = 500
        for i in range(0, len(chunks), batch_size):
            self.collection.upsert(
                ids=ids[i:i + batch_size],
                embeddings=embeddings[i:i + batch_size],
                documents=documents[i:i + batch_size],
                metadatas=metadatas[i:i + batch_size]
            )
        logger.info(f"Upserted {len(chunks)} chunks into ChromaDB.")

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 20,
        where_filter: Optional[dict] = None
    ) -> List[ScoredChunk]:
        """Performs cosine distance search and returns ScoredChunks sorted by similarity."""
        total_items = self.collection.count()
        if total_items == 0:
            return []

        effective_top_k = min(top_k, total_items)
        query_kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": effective_top_k,
            "include": ["documents", "metadatas", "distances"]
        }
        if where_filter:
            query_kwargs["where"] = where_filter

        results = self.collection.query(**query_kwargs)

        scored_chunks: List[ScoredChunk] = []
        if not results or not results["ids"] or not results["ids"][0]:
            return scored_chunks

        ids = results["ids"][0]
        docs = results["documents"][0] if results["documents"] else [""] * len(ids)
        metas = results["metadatas"][0] if results["metadatas"] else [{}] * len(ids)
        distances = results["distances"][0] if results["distances"] else [0.0] * len(ids)

        for rank_idx, (c_id, doc_text, meta, dist) in enumerate(zip(ids, docs, metas, distances), start=1):
            # ChromaDB cosine distance is (1 - cosine_similarity). Convert to similarity score in [0, 1]
            similarity_score = max(0.0, 1.0 - float(dist))

            chunk = Chunk(
                id=c_id,
                doc_id=meta.get("doc_id", ""),
                text=doc_text,
                source_file=meta.get("source_file", ""),
                chunk_index=meta.get("chunk_index", 0),
                char_start=meta.get("char_start", 0),
                char_end=meta.get("char_end", 0),
                metadata=meta
            )
            scored_chunks.append(
                ScoredChunk(
                    chunk=chunk,
                    score=similarity_score,
                    rank=rank_idx,
                    retriever="dense"
                )
            )

        return scored_chunks

    def delete_doc(self, doc_id: str) -> None:
        """Deletes all chunks associated with a doc_id."""
        self.collection.delete(where={"doc_id": doc_id})
        logger.info(f"Deleted chunks for doc_id '{doc_id}' from ChromaDB.")

