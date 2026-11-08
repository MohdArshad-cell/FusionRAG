from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User question or query string")
    stream: bool = Field(default=False, description="Whether to stream response tokens")
    top_n: Optional[int] = Field(default=None, description="Override number of final context chunks")
    use_hyde: bool = Field(default=False, description="Enable Hypothetical Document Embeddings (HyDE)")
    expand_query: bool = Field(default=False, description="Enable Multi-Query Expansion via LLM")
    enable_cache: bool = Field(default=True, description="Enable Semantic Vector Query Caching")
    enforce_crag: bool = Field(default=True, description="Enable Corrective RAG confidence gating")
    doc_id_filter: Optional[str] = Field(default=None, description="Filter results to a specific document ID")


class ChunkSource(BaseModel):
    chunk_id: str
    doc_id: str
    source_file: str
    chunk_index: int
    text: str
    score: float
    retriever_origin: str


class QueryTimings(BaseModel):
    query_enhancement_ms: float = 0.0
    dense_retrieval_ms: float = 0.0
    bm25_retrieval_ms: float = 0.0
    parallel_retrieval_ms: float = 0.0
    fusion_ms: float = 0.0
    rerank_ms: float = 0.0
    generation_ms: float = 0.0
    total_ms: float = 0.0


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[ChunkSource]
    timings: QueryTimings
    cached: bool = Field(default=False, description="Whether the answer was served from the Semantic Cache")
    retrieval_confidence: str = Field(default="HIGH", description="CRAG confidence level: HIGH, AMBIGUOUS, INSUFFICIENT")
    groundedness_score: float = Field(default=1.0, description="Faithfulness / hallucination guardrail score (0.0 to 1.0)")
    groundedness_status: str = Field(default="HIGHLY_GROUNDED", description="Groundedness status: HIGHLY_GROUNDED, MODERATELY_GROUNDED, POTENTIAL_HALLUCINATION")

