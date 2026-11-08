import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse
from app.config import get_settings
from app.api.routes import api_router
from app.dependencies import warmup_services
from app.utils.logger import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: preload models and verify connections
    logger.info("Starting FusionRAG engine...")
    try:
        warmup_services()
    except Exception as e:
        logger.error(f"Warmup error (models will be lazy-loaded on demand): {e}")
    yield
    # Shutdown
    logger.info("Shutting down FusionRAG engine...")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="FusionRAG — Hybrid Search Engine",
        description=(
            "Production Hybrid-Search RAG Engine combining Dense Vector Retrieval (ChromaDB) "
            "and Sparse Lexical Retrieval (BM25) via Reciprocal Rank Fusion (RRF) and "
            "Cross-Encoder Re-Ranking."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routes
    app.include_router(api_router)

    ui_html_path = os.path.join(os.path.dirname(__file__), "ui", "dashboard.html")

    @app.get("/ui", response_class=HTMLResponse, tags=["UI"])
    def get_ui():
        if os.path.exists(ui_html_path):
            with open(ui_html_path, "r", encoding="utf-8") as f:
                return HTMLResponse(content=f.read())
        return HTMLResponse(content="<h2>Dashboard not found</h2>", status_code=404)

    @app.get("/", response_class=HTMLResponse, tags=["Root"])
    def root():
        # Default root serves the UI console
        if os.path.exists(ui_html_path):
            with open(ui_html_path, "r", encoding="utf-8") as f:
                return HTMLResponse(content=f.read())
        return {
            "name": settings.APP_NAME,
            "version": "0.1.0",
            "docs": "/docs",
            "health": "/health",
            "ui": "/ui"
        }

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

