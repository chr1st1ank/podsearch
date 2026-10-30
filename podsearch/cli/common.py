"""Shared helpers for the query/bench entrypoints."""

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from podsearch.core import index, lance_store, qdrant_store
from podsearch.core.paths import LANCE_DIR, QDRANT_URL

QUERIES_PATH = Path(__file__).resolve().parent.parent / "queries.txt"
IMPLS = ["naive", "numpy", "lancedb", "qdrant"]


def fmt_hit(h: index.Hit) -> str:
    date = time.strftime("%Y-%m-%d", time.gmtime(h.pubdate)) if h.pubdate else "----------"
    return f"  {h.score:.4f}  {date}  {h.podcast} — {h.episode}"


def open_store(impl: str, lance_dir: Path | None = None, qdrant_url: str | None = None):
    """Open the search backend for impl: memmap Index, lancedb table or qdrant client."""
    match impl:
        case "lancedb":
            return lance_store.open_chunks(lance_dir or LANCE_DIR)
        case "qdrant":
            return qdrant_store.open_chunks(qdrant_url or QDRANT_URL)
        case "naive" | "numpy":
            return index.load()
        case _:
            raise ValueError(f"unknown impl: {impl}")


def search_fn(impl: str) -> Callable[[Any, Any, Any, str, str, int], list[index.Hit]]:
    match impl:
        case "naive":
            return index.search
        case "numpy":
            return index.search_numpy
        case "lancedb":
            return lance_store.search
        case "qdrant":
            return qdrant_store.search
        case _:
            raise ValueError(f"unknown impl: {impl}")


def stats(impl: str, store: Any) -> dict[str, int | list[str] | None]:
    """Chunk/episode counts and languages for a store, for display in the UI."""
    match impl:
        case "lancedb":
            col = store.search().select(["feed_url", "guid", "language"]).to_polars()
            eps = col.select(["feed_url", "guid"]).unique().height
            return {
                "episodes": eps,
                "chunks": store.count_rows(),
                "languages": sorted(col["language"].unique().to_list()),
            }
        case "qdrant":
            hits = store.facet(qdrant_store.CHUNKS_COLLECTION, key="language", limit=256).hits
            return {
                "episodes": qdrant_store.episode_count(store),
                "chunks": store.count(qdrant_store.CHUNKS_COLLECTION).count,
                "languages": sorted(h.value for h in hits),
            }
        case _:
            eps = store.conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
            langs = [
                r[0] for r in store.conn.execute("SELECT DISTINCT language FROM chunks ORDER BY 1")
            ]
            return {
                "episodes": eps,
                "chunks": int(store.vectors.shape[0]),
                "languages": langs,
            }
