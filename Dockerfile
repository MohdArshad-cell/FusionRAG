# -------------------------------------------------------------
# Multi-stage Dockerfile for FusionRAG Engine
# -------------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /app

# Install system build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY pyproject.toml .

# Install dependencies into virtualenv
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir .

# -------------------------------------------------------------
# Runtime stage
# -------------------------------------------------------------
FROM python:3.11-slim AS runner

WORKDIR /app

# Install runtime tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy source code and UI assets
COPY app/ app/
COPY scripts/ scripts/
COPY pyproject.toml .
COPY README.md .
COPY IMPLEMENTATION_PLAN.md .

# Create persistent storage directories
RUN mkdir -p data/chroma_db data/bm25_index

# Environment defaults
ENV PYTHONUNBUFFERED=1 \
    PORT=8000 \
    HOST=0.0.0.0

EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start ASGI server
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

