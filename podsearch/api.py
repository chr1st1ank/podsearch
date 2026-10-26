"""FastAPI endpoint for the podcast search - attach-profiling target."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from podsearch.cli import common
from podsearch.core import index


def create_app(impl: str = "naive") -> FastAPI:
    """Build the app for a search impl; store+model load eagerly at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
        app.state.store = common.open_store(impl)
        app.state.model = index.load_model()
        yield

    app = FastAPI(title="podsearch", lifespan=lifespan)
    search_fn = common.search_fn(impl)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/search")
    def search(request: Request, q: str, k: int = 5) -> list[index.Hit]:
        return search_fn(request.app.state.store, request.app.state.model, q, k)

    @app.get("/search-async")
    async def search_async(request: Request, q: str, k: int = 5) -> list[index.Hit]:
        return search_fn(request.app.state.store, request.app.state.model, q, k)

    return app
