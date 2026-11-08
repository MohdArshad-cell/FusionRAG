import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "FusionRAG" in data["name"]


def test_ui_endpoint(client):
    response = client.get("/ui")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "FusionRAG" in response.text


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "chroma_chunks_count" in data
    assert "bm25_chunks_count" in data


def test_ingest_and_query_flow(client):
    # 1. Ingest text containing exact keyword and technical concept
    doc_text = (
        "Server Maintenance Manual. For permission denied errors, look for ERR_404_AUTH in the logs. "
        "The backup storage uses device model X-500 located in Rack 4."
    )
    ingest_res = client.post(
        "/ingest/text",
        data={"text": doc_text, "filename": "server_manual.txt", "doc_id": "test_manual"}
    )
    assert ingest_res.status_code == 200
    ingest_data = ingest_res.json()
    assert ingest_data["status"] == "success"
    assert ingest_data["chunks_created"] > 0

    # 2. Query for exact keyword
    query_res = client.post(
        "/query",
        json={"query": "What is the device model for backup storage?"}
    )
    assert query_res.status_code == 200
    query_data = query_res.json()
    assert "answer" in query_data
    assert len(query_data["sources"]) > 0
    assert query_data["timings"]["total_ms"] > 0

    # 3. Query for error code
    query_auth = client.post(
        "/query",
        json={"query": "What error code indicates permission denied?"}
    )
    assert query_auth.status_code == 200
    assert len(query_auth.json()["sources"]) > 0


def test_streaming_query(client):
    response = client.get("/query/stream?q=Tell me about ERR_404_AUTH")
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    text_content = response.text
    assert "event: token" in text_content
    assert "event: sources" in text_content

