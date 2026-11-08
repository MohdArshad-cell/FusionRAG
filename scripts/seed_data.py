import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.dependencies import get_ingestion_pipeline, get_vector_store, get_bm25_index

SEED_DOCUMENTS = [
    {
        "filename": "auth_security_guide.txt",
        "text": """Enterprise Authentication and Session Security Guide.
When a user attempts to authenticate with an invalid or expired JWT token, the identity provider returns error code ERR_404_AUTH.
To resolve ERR_404_AUTH:
1. Clear the client-side session storage and cookies.
2. Invalidate the stale refresh token via /api/v1/auth/revoke.
3. Request a fresh OAuth bearer token from /oauth/token using valid client credentials.
Session hijacking can be mitigated by binding tokens to the client IP address and enforcing short token lifetimes (15 minutes)."""
    },
    {
        "filename": "laboratory_centrifuge_manual.txt",
        "text": """Laboratory Equipment Manual — Centrifuge Series.
The ultra-high-speed centrifuge model X-500 operates at an operational ceiling of 15000 RPM (21000 x g).
Model X-500 requires standard 220V power supply and scheduled rotor balance checks every 6 months.
Emergency shutoff triggers automatically on vibration exceeding 0.5 mm/s.
In contrast, the entry-level centrifuge model X-200 operates at 8000 RPM (6000 x g) and is intended for routine blood tube separation."""
    },
    {
        "filename": "reverse_proxy_troubleshooting.txt",
        "text": """Network Failure & Gateway Troubleshooting Runbook.
When an upstream application server drops a socket connection or exhausts its thread pool, the reverse proxy returns error code ERR_502_GATEWAY.
Recommended triage steps for ERR_502_GATEWAY:
- Inspect reverse proxy keepalive timeouts (proxy_read_timeout 60s).
- Verify backend pool health checks in Consul or AWS ALB.
- Check upstream application socket buffers and OS file descriptor limits (ulimit -n 65535)."""
    },
    {
        "filename": "climate_oceanography_report.txt",
        "text": """Global Environmental Observations — Oceanography and Polar Ice Dynamics.
Sustained increases in atmospheric greenhouse gas concentrations accelerate ocean warming, leading to thermal expansion of ocean waters.
Simultaneously, accelerated meltwater discharge from polar ice sheets contributes significantly to global mean sea level rise.
Coastal ecosystems face increased risk of saltwater intrusion, tidal inundation, and storm surge amplification."""
    },
    {
        "filename": "microservices_retry_policy.txt",
        "text": """Distributed Systems Resiliency and Exponential Backoff.
Transient network partitions and database lock contentions should be handled using truncated exponential backoff with full jitter.
The retry formula is: sleep = min(cap, base * 2 ** attempt) * random(0, 1).
Idempotency keys (Idempotency-Key header) must be supplied with all mutating POST/PUT requests to prevent duplicate charges or operations."""
    }
]


def seed():
    print("=" * 60)
    print("🌱 Seeding FusionRAG with Technical Knowledge Base...")
    print("=" * 60)

    pipeline = get_ingestion_pipeline()
    vector_store = get_vector_store()
    bm25 = get_bm25_index()

    for doc in SEED_DOCUMENTS:
        filename = doc["filename"]
        text = doc["text"]
        res = pipeline.ingest_bytes(
            file_bytes=text.encode("utf-8"),
            filename=filename,
            content_type="text/plain"
        )
        print(f"  ✓ Ingested '{filename}': {res.chunks_created} chunks")

    print("\n" + "=" * 60)
    print(f"✨ Seeding complete! Current index counts:")
    print(f"   • ChromaDB Dense Chunks: {vector_store.count()}")
    print(f"   • BM25 Sparse Chunks:    {bm25.count()}")
    print("=" * 60)
    print("👉 You can now run 'uvicorn app.main:app' and query the engine!")


if __name__ == "__main__":
    seed()

