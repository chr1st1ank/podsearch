"""Shared helpers for the query/bench entrypoints."""

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from podsearch.core import index, lance_store
from podsearch.core.paths import LANCE_DIR

QUERIES_PATH = Path(__file__).resolve().parent.parent / "queries.txt"
IMPLS = ["naive", "numpy", "lancedb"]


def fmt_hit(h: index.Hit) -> str:
    date = time.strftime("%Y-%m-%d", time.gmtime(h.pubdate)) if h.pubdate else "----------"
    return f"  {h.score:.4f}  {date}  {h.podcast} — {h.episode}"


def open_store(impl: str, lance_dir: Path | None = None):
    """Open the search backend for impl: memmap Index or lancedb chunks table."""
    match impl:
        case "lancedb":
            return lance_store.open_chunks(lance_dir or LANCE_DIR)
        case "naive" | "numpy":
            return index.load()
        case _:
            raise ValueError(f"unknown impl: {impl}")


def search_fn(impl: str) -> Callable[[Any, Any, str, int], list[index.Hit]]:
    match impl:
        case "naive":
            return index.search
        case "numpy":
            return index.search_numpy
        case "lancedb":
            return lance_store.search
        case _:
            raise ValueError(f"unknown impl: {impl}")
