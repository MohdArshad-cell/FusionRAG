from app.core.models import Chunk, ScoredChunk
from app.core.context_builder import ContextBuilder


def make_scored(rank: int) -> ScoredChunk:
    chunk = Chunk(id=f"c{rank}", doc_id="d1", text=f"Content rank {rank}", chunk_index=rank)
    return ScoredChunk(chunk=chunk, score=1.0 / rank, rank=rank, retriever="test")


def test_lost_in_the_middle_reordering():
    # Input ranks: 1, 2, 3, 4, 5
    chunks = [make_scored(i) for i in range(1, 6)]
    reordered = ContextBuilder.reorder_lost_in_the_middle(chunks)

    result_ranks = [c.rank for c in reordered]
    # Expected: 1 at start, 2 at end, 3 next to 1, 4 before 2, 5 in middle
    # Odd indices (0, 2, 4 -> 1, 3, 5), Even reversed (3, 1 -> 4, 2)
    assert result_ranks == [1, 3, 5, 4, 2]

    # Highest ranked chunk must be first
    assert reordered[0].rank == 1
    # Second highest ranked chunk must be last
    assert reordered[-1].rank == 2


def test_lost_in_the_middle_small_list():
    chunks = [make_scored(1), make_scored(2)]
    reordered = ContextBuilder.reorder_lost_in_the_middle(chunks)
    assert [c.rank for c in reordered] == [1, 2]


def test_context_string_formatting():
    chunks = [make_scored(1), make_scored(2)]
    ctx = ContextBuilder.build_context_string(chunks)
    assert "[Source 1:" in ctx
    assert "Content rank 1" in ctx
    assert "---" in ctx

