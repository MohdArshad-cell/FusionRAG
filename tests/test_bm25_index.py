import os
from app.core.models import Chunk
from app.core.bm25_index import BM25Index


def test_bm25_tokenization_preserves_identifiers():
    text = "Encountered error code ERR_404_AUTH with device model X-500!"
    tokens = BM25Index.tokenize(text)
    assert "err_404_auth" in tokens
    assert "x-500" in tokens


def test_bm25_indexing_and_search(tmp_path):
    index_dir = str(tmp_path / "bm25")
    index = BM25Index(persist_dir=index_dir)

    chunks = [
        Chunk(id="c1", doc_id="d1", text="Centrifuge model X-500 runs at high speeds."),
        Chunk(id="c2", doc_id="d2", text="When authentication fails, inspect ERR_404_AUTH."),
        Chunk(id="c3", doc_id="d3", text="Database connection timeouts and memory issues."),
    ]

    index.add_chunks(chunks)

    # Search for exact keyword
    results_x500 = index.search("X-500", top_k=2)
    assert len(results_x500) > 0
    assert results_x500[0].chunk.id == "c1"

    # Search for error code
    results_auth = index.search("ERR_404_AUTH", top_k=2)
    assert len(results_auth) > 0
    assert results_auth[0].chunk.id == "c2"

    # Test persistence by creating a second instance from the same folder
    reloaded_index = BM25Index(persist_dir=index_dir)
    assert reloaded_index.count() == 3
    reloaded_results = reloaded_index.search("ERR_404_AUTH", top_k=1)
    assert reloaded_results[0].chunk.id == "c2"

