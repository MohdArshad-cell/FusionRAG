import uuid
from typing import Optional
from app.ingestion.extractors import get_extractor
from app.ingestion.preprocessor import Preprocessor
from app.core.chunker import RecursiveCharacterChunker
from app.core.embedder import Embedder
from app.core.vector_store import ChromaVectorStore
from app.core.bm25_index import BM25Index
from app.api.schemas.ingest import IngestResponse
from app.utils.logger import logger


class IngestionPipeline:
    """End-to-end ingestion orchestrator for dual dense & sparse indexing."""

    def __init__(
        self,
        chunker: RecursiveCharacterChunker,
        embedder: Embedder,
        vector_store: ChromaVectorStore,
        bm25_index: BM25Index,
    ):
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store
        self.bm25_index = bm25_index

    def ingest_bytes(
        self,
        file_bytes: bytes,
        filename: str,
        content_type: str = "",
        doc_id: Optional[str] = None
    ) -> IngestResponse:
        """Extracts, cleans, chunks, embeds, and stores a document in both ChromaDB and BM25."""
        if not file_bytes:
            raise ValueError("Uploaded file is empty.")

        document_id = doc_id or str(uuid.uuid4())[:8]

        # 1. Extract text
        extractor = get_extractor(filename, content_type)
        raw_text = extractor.extract(file_bytes, filename)

        if not raw_text.strip():
            logger.warning(f"Extracted empty text from file '{filename}'")
            return IngestResponse(
                doc_id=document_id,
                filename=filename,
                chunks_created=0,
                status="warning",
                message="Extracted text was empty."
            )

        # 2. Clean & Normalize
        cleaned_text = Preprocessor.clean(raw_text)

        # 3. Chunk text
        chunks = self.chunker.chunk(cleaned_text, doc_id=document_id, source_file=filename)

        if not chunks:
            return IngestResponse(
                doc_id=document_id,
                filename=filename,
                chunks_created=0,
                status="warning",
                message="No chunks could be produced."
            )

        # 4. Dense embeddings & store in ChromaDB
        chunk_texts = [c.text for c in chunks]
        embeddings = self.embedder.embed_texts(chunk_texts)
        self.vector_store.add_chunks(chunks, embeddings)

        # 5. Sparse lexical indexing in BM25
        self.bm25_index.add_chunks(chunks)

        logger.info(f"Successfully ingested '{filename}' with {len(chunks)} chunks (Doc ID: {document_id}).")

        return IngestResponse(
            doc_id=document_id,
            filename=filename,
            chunks_created=len(chunks),
            status="success",
            message=f"Indexed {len(chunks)} chunks into ChromaDB and BM25."
        )

