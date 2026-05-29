from __future__ import annotations
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from .config import Settings
from .db.pool import create_pool, migrate
from .recall.embedder import STEmbedder
from .recall.reranker import CEReranker
from .llm.client import build_llm_client
from .repositories.turns import TurnRepository
from .repositories.memories import MemoryRepository
from .services.ingest import IngestService
from .services.recall_service import RecallService

logging.basicConfig(level=logging.INFO)


def build_app(settings: Settings | None = None, *, embedder=None, reranker=None,
              llm=None, pool=None) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.pool = pool or create_pool(settings.database_url)
        migrate(app.state.pool)
        app.state.embedder = embedder or STEmbedder(settings.embed_model)
        app.state.reranker = reranker or CEReranker(settings.rerank_model)
        app.state.llm = llm or build_llm_client(settings.openai_api_key, settings.openai_model)
        app.state.turns = TurnRepository(app.state.pool)
        app.state.memories = MemoryRepository(app.state.pool)
        app.state.ingest = IngestService(app.state.turns, app.state.memories,
                                         app.state.embedder, app.state.llm)
        app.state.recall = RecallService(app.state.memories, app.state.turns,
                                         app.state.embedder, app.state.reranker, app.state.llm)
        yield
        app.state.pool.close()

    app = FastAPI(title="memory-service", lifespan=lifespan)
    app.state.settings = settings

    @app.middleware("http")
    async def limit_body(request: Request, call_next):
        cl = request.headers.get("content-length")
        if cl and int(cl) > settings.max_body_bytes:
            return JSONResponse({"detail": "payload too large"}, status_code=413)
        return await call_next(request)

    from .routes import health, turns, recall, search, memories, admin
    app.include_router(health.router)
    app.include_router(turns.router)
    app.include_router(recall.router)
    app.include_router(search.router)
    app.include_router(memories.router)
    app.include_router(admin.router)
    return app
