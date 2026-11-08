from functools import lru_cache
from typing import Optional
from app.config import get_settings, Settings
from app.core.chunker import RecursiveCharacterChunker
from app.core.embedder import Embedder
from app.core.vector_store import ChromaVectorStore
from app.core.bm25_index import BM25Index
from app.core.reranker import CrossEncoderReranker
from app.core.retriever import DualRetriever
from app.core.generator import get_llm_generator, BaseLLMGenerator
from app.core.query_expansion import QueryProcessor
from app.core.semantic_cache import SemanticQueryCache
from app.core.guardrails import RetrievalConfidenceGate, GroundednessVerifier
from app.core.parent_store import ParentDocumentStore
from app.ingestion.pipeline import IngestionPipeline
from app.utils.logger import logger


class Container:
    """Singleton service container for models, stores, and pipelines."""
    settings: Settings = get_settings()
    chunker: Optional[RecursiveCharacterChunker] = None
    embedder: Optional[Embedder] = None
    vector_store: Optional[ChromaVectorStore] = None
    bm25_index: Optional[BM25Index] = None
    reranker: Optional[CrossEncoderReranker] = None
    retriever: Optional[DualRetriever] = None
    generator: Optional[BaseLLMGenerator] = None
    query_processor: Optional[QueryProcessor] = None
    semantic_cache: Optional[SemanticQueryCache] = None
    crag_gate: Optional[RetrievalConfidenceGate] = None
    groundedness_verifier: Optional[GroundednessVerifier] = None
    parent_store: Optional[ParentDocumentStore] = None
    pipeline: Optional[IngestionPipeline] = None


container = Container()


def get_chunker() -> RecursiveCharacterChunker:
    if container.chunker is None:
        container.chunker = RecursiveCharacterChunker(
            chunk_size=container.settings.CHUNK_SIZE,
            chunk_overlap=container.settings.CHUNK_OVERLAP,
        )
    return container.chunker


def get_embedder() -> Embedder:
    if container.embedder is None:
        container.embedder = Embedder(model_name=container.settings.EMBEDDING_MODEL)
    return container.embedder


def get_vector_store() -> ChromaVectorStore:
    if container.vector_store is None:
        container.vector_store = ChromaVectorStore(
            persist_dir=container.settings.CHROMA_PERSIST_DIR,
            collection_name=container.settings.CHROMA_COLLECTION,
        )
    return container.vector_store


def get_bm25_index() -> BM25Index:
    if container.bm25_index is None:
        container.bm25_index = BM25Index(persist_dir=container.settings.BM25_PERSIST_DIR)
    return container.bm25_index


def get_reranker() -> CrossEncoderReranker:
    if container.reranker is None:
        container.reranker = CrossEncoderReranker(model_name=container.settings.CROSS_ENCODER_MODEL)
    return container.reranker


def get_retriever() -> DualRetriever:
    if container.retriever is None:
        container.retriever = DualRetriever(
            embedder=get_embedder(),
            vector_store=get_vector_store(),
            bm25_index=get_bm25_index(),
            dense_top_k=container.settings.DENSE_TOP_K,
            bm25_top_k=container.settings.BM25_TOP_K,
        )
    return container.retriever


def get_llm() -> BaseLLMGenerator:
    if container.generator is None:
        container.generator = get_llm_generator(container.settings)
    return container.generator


def get_query_processor() -> QueryProcessor:
    if container.query_processor is None:
        container.query_processor = QueryProcessor(llm=get_llm())
    return container.query_processor


def get_semantic_cache() -> SemanticQueryCache:
    if container.semantic_cache is None:
        container.semantic_cache = SemanticQueryCache(similarity_threshold=0.94)
    return container.semantic_cache


def get_crag_gate() -> RetrievalConfidenceGate:
    if container.crag_gate is None:
        container.crag_gate = RetrievalConfidenceGate(
            high_confidence_threshold=0.50,
            low_confidence_threshold=-2.0
        )
    return container.crag_gate


def get_groundedness_verifier() -> GroundednessVerifier:
    if container.groundedness_verifier is None:
        container.groundedness_verifier = GroundednessVerifier()
    return container.groundedness_verifier


def get_parent_store() -> ParentDocumentStore:
    if container.parent_store is None:
        container.parent_store = ParentDocumentStore()
    return container.parent_store


def get_ingestion_pipeline() -> IngestionPipeline:
    if container.pipeline is None:
        container.pipeline = IngestionPipeline(
            chunker=get_chunker(),
            embedder=get_embedder(),
            vector_store=get_vector_store(),
            bm25_index=get_bm25_index(),
            parent_store=get_parent_store(),
        )
    return container.pipeline


def warmup_services() -> None:
    """Pre-warms models and stores to ensure instant first-request responsiveness."""
    logger.info("Initializing & pre-warming FusionRAG core components...")
    # Trigger instances
    _ = get_chunker()
    emb = get_embedder()
    _ = get_vector_store()
    _ = get_bm25_index()
    reranker = get_reranker()
    _ = get_llm()

    # Preload model weights into memory/GPU
    logger.info("Warming up SentenceTransformer embedding model...")
    _ = emb.embed_query("warmup query")

    logger.info("Warming up CrossEncoder reranker model...")
    _ = reranker.model  # Trigger model property to load weights into memory

    logger.info("All FusionRAG components initialized and ready!")

