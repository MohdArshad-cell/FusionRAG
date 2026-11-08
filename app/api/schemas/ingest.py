from typing import List, Optional
from pydantic import BaseModel, Field


class IngestResponse(BaseModel):
    doc_id: str
    filename: str
    chunks_created: int
    status: str
    message: str = ""


class IngestStatusResponse(BaseModel):
    total_documents: int
    total_chunks: int
    chroma_count: int
    bm25_count: int

