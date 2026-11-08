from fastapi import APIRouter, Depends
from app.config import Settings, get_settings
from app.dependencies import get_vector_store, get_bm25_index
from app.core.vector_store import ChromaVectorStore
from app.core.bm25_index import BM25Index

router = APIRouter(tags=["Health"])


@router.get("/health")
def health_check(
    settings: Settings = Depends(get_settings),
    vector_store: ChromaVectorStore = Depends(get_vector_store),
    bm25_index: BM25Index = Depends(get_bm25_index),
):
    return {
        "status": "healthy",
        "app_name": settings.APP_NAME,
        "chroma_chunks_count": vector_store.count(),
        "bm25_chunks_count": bm25_index.count(),
        "embedding_model": settings.EMBEDDING_MODEL,
        "cross_encoder_model": settings.CROSS_ENCODER_MODEL,
        "llm_provider": settings.LLM_PROVIDER,
    }

