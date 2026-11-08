from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

QUERY_COUNTER = Counter(
    "fusionrag_queries_total",
    "Total RAG queries processed",
    ["status"]
)

QUERY_LATENCY = Histogram(
    "fusionrag_query_latency_seconds",
    "Total query execution duration in seconds",
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)

CACHE_HITS = Counter(
    "fusionrag_cache_hits_total",
    "Total semantic cache hits"
)

CACHE_MISSES = Counter(
    "fusionrag_cache_misses_total",
    "Total semantic cache misses"
)

GROUNDEDNESS_HISTOGRAM = Histogram(
    "fusionrag_groundedness_score",
    "Faithfulness / Groundedness score of generated answers",
    buckets=[0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0]
)

INDEXED_CHUNKS_GAUGE = Gauge(
    "fusionrag_indexed_chunks",
    "Total number of indexed chunks",
    ["store"]
)


def export_metrics() -> bytes:
    """Exports all Prometheus metrics in standard format."""
    return generate_latest()

