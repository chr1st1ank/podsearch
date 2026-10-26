"""FastAPI endpoint for the podcast search - attach-profiling target."""

from fastapi import FastAPI

from podsearch.core import index

app = FastAPI(title="podsearch")

_index: index.Index | None = None
_model: "index.StaticModel | None" = None  # noqa: UP037 - StaticModel is a TYPE_CHECKING import


def _ensure_loaded() -> tuple[index.Index, index.StaticModel]:
    global _index, _model
    if _index is None or _model is None:
        _index = index.load()
        _model = index.load_model()
    return _index, _model


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/search")
def search(q: str, k: int = 5) -> list[index.Hit]:
    idx, model = _ensure_loaded()
    return index.search(idx, model, q, k)


@app.get("/search-async")
async def search_async(q: str, k: int = 5) -> list[index.Hit]:
    idx, model = _ensure_loaded()
    return index.search(idx, model, q, k)
