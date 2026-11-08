from functools import lru_cache
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # App
    APP_NAME: str = "FusionRAG"
    APP_ENV: str = "development"
    DEBUG: bool = True

    # Embedding & Re-Ranking Models
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    CROSS_ENCODER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Chunking
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50

    # Retrieval & Fusion Parameters
    DENSE_TOP_K: int = 20
    BM25_TOP_K: int = 20
    RRF_K: int = 60
    FUSION_TOP_N: int = 10
    FINAL_TOP_N: int = 5

    # Storage Paths
    CHROMA_PERSIST_DIR: str = "./data/chroma_db"
    CHROMA_COLLECTION: str = "fusionrag_docs"
    BM25_PERSIST_DIR: str = "./data/bm25_index"

    # LLM Settings
    LLM_PROVIDER: Literal["groq", "gemini", "mock"] = "mock"
    GROQ_API_KEY: str = ""
    GEMINI_API_KEY: str = ""
    LLM_MODEL: str = "llama-3.3-70b-versatile"
    LLM_TEMPERATURE: float = 0.2
    LLM_MAX_TOKENS: int = 1024

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


@lru_cache()
def get_settings() -> Settings:
    return Settings()

