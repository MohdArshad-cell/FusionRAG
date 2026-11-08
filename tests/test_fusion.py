from app.core.models import Chunk, ScoredChunk
from app.core.fusion import reciprocal_rank_fusion


def make_chunk(cid: str) -> Chunk:
    return Chunk(id=cid, doc_id="d1", text=f"Text for {cid}")


def test_rrf_mathematical_precision():
    c1 = make_chunk("c1")
    c2 = make_chunk("c2")
    c3 = make_chunk("c3")

    # List 1: c1 (rank 1), c2 (rank 2)
    list1 = [
        ScoredChunk(chunk=c1, score=0.9, rank=1, retriever="dense"),
        ScoredChunk(chunk=c2, score=0.8, rank=2, retriever="dense"),
    ]

    # List 2: c2 (rank 1), c3 (rank 2)
    list2 = [
        ScoredChunk(chunk=c2, score=15.0, rank=1, retriever="bm25"),
        ScoredChunk(chunk=c3, score=12.0, rank=2, retriever="bm25"),
    ]

    # Formula with k=60:
    # c2 appears at rank 2 in list1 (1 / 62) and rank 1 in list2 (1 / 61)
    # Expected c2 score = (1/62) + (1/61) = 0.016129 + 0.016393 = 0.032522
    # c1 appears only in list1 at rank 1: (1 / 61) = 0.016393
    # c3 appears only in list2 at rank 2: (1 / 62) = 0.016129

    fused = reciprocal_rank_fusion([list1, list2], k=60, top_n=3)

    assert len(fused) == 3
    # c2 must be top because it appeared high in BOTH lists
    assert fused[0].chunk.id == "c2"
    assert fused[1].chunk.id == "c1"
    assert fused[2].chunk.id == "c3"

    expected_c2_score = round((1.0 / 62) + (1.0 / 61), 6)
    assert abs(fused[0].score - expected_c2_score) < 1e-4


def test_rrf_empty_lists():
    fused = reciprocal_rank_fusion([], k=60, top_n=5)
    assert fused == []


def test_rrf_truncation():
    chunks = [make_chunk(f"c{i}") for i in range(20)]
    sc_list = [ScoredChunk(chunk=c, score=1.0, rank=i+1, retriever="dense") for i, c in enumerate(chunks)]

    fused = reciprocal_rank_fusion([sc_list], k=60, top_n=5)
    assert len(fused) == 5

