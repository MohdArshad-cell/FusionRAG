import pytest
from app.core.models import Chunk, ScoredChunk
from app.core.semantic_cache import SemanticQueryCache
from app.core.guardrails import RetrievalConfidenceGate, GroundednessVerifier
from app.core.parent_store import ParentDocumentStore
from fastapi.testclient import TestClient
from app.main import app


def test_semantic_cache_hit_and_miss():
    cache = SemanticQueryCache(similarity_threshold=0.90)

    # 3D unit vectors
    q_emb1 = [1.0, 0.0, 0.0]
    q_emb_similar = [0.98, 0.1, 0.0]  # high similarity
    q_emb_dissimilar = [0.0, 1.0, 0.0]  # orthogonal, 0 similarity

    # Initially empty
    assert cache.lookup(q_emb1) is None

    # Put entry
    cache.put(query="What is X-500?", query_embedding=q_emb1, response_data={"answer": "X-500 is a centrifuge."})

    # Lookup with similar embedding
    hit = cache.lookup(q_emb_similar)
    assert hit is not None
    res, sim, orig_q = hit
    assert res["answer"] == "X-500 is a centrifuge."
    assert sim >= 0.90

    # Lookup with dissimilar embedding
    miss = cache.lookup(q_emb_dissimilar)
    assert miss is None

    stats = cache.stats()
    assert stats["total_hits"] == 1
    assert stats["total_misses"] == 2


def test_crag_confidence_gate():
    gate = RetrievalConfidenceGate(high_confidence_threshold=0.50, low_confidence_threshold=-1.0)

    # High score
    c_high = ScoredChunk(chunk=Chunk(id="c1", doc_id="d1", text="Text"), score=2.5)
    eval_high = gate.evaluate_candidates([c_high])
    assert eval_high["decision"] == "HIGH"

    # Borderline score
    c_mid = ScoredChunk(chunk=Chunk(id="c2", doc_id="d2", text="Text"), score=0.1)
    eval_mid = gate.evaluate_candidates([c_mid])
    assert eval_mid["decision"] == "AMBIGUOUS"

    # Low score
    c_low = ScoredChunk(chunk=Chunk(id="c3", doc_id="d3", text="Text"), score=-3.5)
    eval_low = gate.evaluate_candidates([c_low])
    assert eval_low["decision"] == "INSUFFICIENT"


def test_groundedness_verifier():
    context = (
        "Industrial centrifuge model X-500 operates at 15000 RPM with standard 220V power supply. "
        "Maintenance is required every 6 months."
    )

    # Well grounded answer
    grounded_answer = "Model X-500 centrifuge operates at 15000 RPM using 220V power."
    res_good = GroundednessVerifier.verify(grounded_answer, context)
    assert res_good["groundedness_score"] >= 0.70
    assert res_good["status"] == "HIGHLY_GROUNDED"

    # Safe abstention answer
    abstain_answer = "Based on the provided documents, there is no relevant context."
    res_abstain = GroundednessVerifier.verify(abstain_answer, context)
    assert res_abstain["status"] == "SAFE_ABSTENTION"


def test_parent_document_store(tmp_path):
    store = ParentDocumentStore(persist_dir=str(tmp_path))

    parent = Chunk(id="parent_doc_1", doc_id="d1", text="Full long section with complete context and details.")
    store.add_parents([parent])

    child1 = Chunk(id="child_1", doc_id="d1", text="Snippet 1", parent_id="parent_doc_1")
    child2 = Chunk(id="child_2", doc_id="d1", text="Snippet 2", parent_id="parent_doc_1")

    sc1 = ScoredChunk(chunk=child1, score=0.9, rank=1)
    sc2 = ScoredChunk(chunk=child2, score=0.8, rank=2)

    # Resolve children to parent chunks with deduplication
    resolved = store.resolve_to_parent_chunks([sc1, sc2])
    assert len(resolved) == 1
    assert resolved[0].chunk.id == "parent_doc_1"
    assert resolved[0].chunk.text == parent.text


def test_prometheus_metrics_endpoint():
    with TestClient(app) as client:
        response = client.get("/metrics")
        assert response.status_code == 200
        text = response.text
        assert "fusionrag_queries_total" in text
        assert "fusionrag_query_latency_seconds" in text

