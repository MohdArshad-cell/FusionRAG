from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """Represents a text chunk created during ingestion."""
    id: str = Field(description="Unique chunk identifier, e.g. doc_id_chunk_idx")
    doc_id: str = Field(description="Parent document identifier")
    text: str = Field(description="Text content of the chunk")
    source_file: str = Field(default="", description="Original file name or source")
    chunk_index: int = Field(default=0, description="Sequential index of chunk in document")
    char_start: int = Field(default=0, description="Start character offset in source document")
    char_end: int = Field(default=0, description="End character offset in source document")
    parent_id: Optional[str] = Field(default=None, description="Identifier of enclosing parent chunk if small-to-big")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary extra metadata")


class ScoredChunk(BaseModel):
    """Represents a chunk with an associated relevance or ranking score."""
    chunk: Chunk
    score: float = Field(default=0.0, description="Score assigned by retrieval/ranking stage")
    rank: Optional[int] = Field(default=None, description="Rank within the candidate list")
    retriever: str = Field(default="unknown", description="Source retriever: dense, bm25, rrf, cross_encoder")

