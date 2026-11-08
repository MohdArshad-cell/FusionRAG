from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, Form
from typing import Optional
from app.api.schemas.ingest import IngestResponse, IngestStatusResponse
from app.dependencies import get_ingestion_pipeline, get_vector_store, get_bm25_index
from app.ingestion.pipeline import IngestionPipeline
from app.core.vector_store import ChromaVectorStore
from app.core.bm25_index import BM25Index
from app.utils.logger import logger

router = APIRouter(prefix="/ingest", tags=["Ingestion"])


@router.post("", response_model=IngestResponse)
async def ingest_file(
    file: UploadFile = File(...),
    doc_id: Optional[str] = Form(None),
    pipeline: IngestionPipeline = Depends(get_ingestion_pipeline),
):
    """
    Ingest a document (.pdf, .docx, .txt, .md, .csv) into the hybrid store.
    Extracts text, splits semantically, creates dense embeddings in ChromaDB,
    and updates the BM25 sparse index.
    """
    try:
        content_bytes = await file.read()
        if not content_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        response = pipeline.ingest_bytes(
            file_bytes=content_bytes,
            filename=file.filename or "uploaded_document",
            content_type=file.content_type or "",
            doc_id=doc_id,
        )
        return response
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@router.post("/text", response_model=IngestResponse)
def ingest_raw_text(
    text: str = Form(...),
    filename: str = Form("raw_text_entry.txt"),
    doc_id: Optional[str] = Form(None),
    pipeline: IngestionPipeline = Depends(get_ingestion_pipeline),
):
    """Convenience endpoint to directly ingest raw text without uploading a file."""
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    try:
        response = pipeline.ingest_bytes(
            file_bytes=text.encode("utf-8"),
            filename=filename,
            content_type="text/plain",
            doc_id=doc_id,
        )
        return response
    except Exception as e:
        logger.error(f"Direct text ingestion failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Text ingestion failed: {str(e)}")


@router.get("/stats", response_model=IngestStatusResponse)
def get_ingest_stats(
    vector_store: ChromaVectorStore = Depends(get_vector_store),
    bm25_index: BM25Index = Depends(get_bm25_index),
):
    """Returns the current chunk counts from both dense and sparse indexes."""
    c_count = vector_store.count()
    b_count = bm25_index.count()
    return IngestStatusResponse(
        total_documents=0,  # document tracking can be enriched
        total_chunks=max(c_count, b_count),
        chroma_count=c_count,
        bm25_count=b_count,
    )

