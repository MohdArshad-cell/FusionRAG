import os
import shutil
import pytest
from app.core.models import Chunk, ScoredChunk
from app.core.chunker import RecursiveCharacterChunker
from app.core.bm25_index import BM25Index
from app.config import Settings


@pytest.fixture
def temp_data_dir(tmp_path):
    data_dir = tmp_path / "test_data"
    data_dir.mkdir()
    yield str(data_dir)
    shutil.rmtree(str(data_dir), ignore_errors=True)


@pytest.fixture
def sample_corpus():
    return [
        (
            "doc_auth",
            "Authentication Protocol Guide. If a user enters an expired JWT token, the server returns error code ERR_404_AUTH. "
            "To resolve ERR_404_AUTH, clear the local storage session and request a fresh OAuth bearer token from /oauth/token."
        ),
        (
            "doc_hardware",
            "Industrial Hardware Specifications. The high-performance centrifuge model X-500 operates at 15000 RPM. "
            "Model X-500 requires standard 220V power supply and maintenance every 6 months. Model X-200 operates at 8000 RPM."
        ),
        (
            "doc_semantic",
            "Global Climate Change and Oceanography. Rising atmospheric greenhouse gas concentrations cause thermal expansion "
            "of ocean waters and melting of polar ice caps, resulting in significant sea level rise across coastal regions."
        ),
    ]


@pytest.fixture
def chunker():
    return RecursiveCharacterChunker(chunk_size=120, chunk_overlap=20)

