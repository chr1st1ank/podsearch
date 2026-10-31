"""FastAPI endpoint for the podcast search - attach-profiling target."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import cast

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, Response

from podsearch.cli import common
from podsearch.core import index


def create_app(impl: str = "naive", rerank: bool = True) -> FastAPI:
    """Build the app for a search impl; store+model load eagerly at startup.

    rerank=False skips loading the cross-encoder and returns stage-1 results.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
        app.state.store = common.open_store(impl)
        app.state.model = index.load_model()
        app.state.reranker = index.load_reranker() if rerank else None
        stats = common.stats(impl, app.state.store)
        app.state.info = {"impl": impl, "rerank": rerank, **stats}
        eps = f"{stats['episodes']:,}" if stats["episodes"] is not None else "?"
        print(
            f"podsearch backend: {impl} ({eps} episodes, {stats['chunks']:,} chunks, "
            f"rerank={'on' if rerank else 'off'})"
        )
        yield

    app = FastAPI(title="podsearch", lifespan=lifespan)
    search_fn = common.search_fn(impl)

    @app.get("/")
    def index_page() -> FileResponse:
        return FileResponse(Path(__file__).parent / "static" / "index.html")

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon() -> Response:
        svg = (
            "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'>"
            "<text y='.9em' font-size='90'>🎙</text></svg>"
        )
        return Response(svg, media_type="image/svg+xml")

    @app.get("/api/stats")
    def stats(request: Request, response: Response) -> dict[str, object]:
        response.headers["Cache-Control"] = "no-store"
        return cast("dict[str, object]", request.app.state.info)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/search")
    def search(request: Request, q: str, lang: str, k: int = 5) -> list[index.Hit]:
        return search_fn(
            request.app.state.store,
            request.app.state.model,
            request.app.state.reranker,
            q,
            lang,
            k,
        )

    @app.get("/search-async")
    async def search_async(request: Request, q: str, lang: str, k: int = 5) -> list[index.Hit]:
        return search_fn(
            request.app.state.store,
            request.app.state.model,
            request.app.state.reranker,
            q,
            lang,
            k,
        )

    return app
