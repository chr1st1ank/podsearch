"""Shared helpers for the query/bench entrypoints."""

import time
from pathlib import Path

from podsearch.core import index

QUERIES_PATH = Path(__file__).resolve().parent.parent / "queries.txt"


def fmt_hit(h: index.Hit) -> str:
    date = time.strftime("%Y-%m-%d", time.gmtime(h.pubdate)) if h.pubdate else "----------"
    return f"  {h.score:.4f}  {date}  {h.podcast} — {h.episode}"


def search_fn(impl: str):
    return index.search_numpy if impl == "numpy" else index.search
