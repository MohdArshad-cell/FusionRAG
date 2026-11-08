import time
from contextlib import contextmanager
from typing import Generator, Dict, Any


@contextmanager
def measure_ms() -> Generator[Dict[str, float], None, None]:
    """Context manager to measure execution time in milliseconds."""
    result: Dict[str, float] = {"elapsed_ms": 0.0}
    start = time.perf_counter()
    try:
        yield result
    finally:
        result["elapsed_ms"] = round((time.perf_counter() - start) * 1000, 2)

