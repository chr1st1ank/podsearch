"""Helpers shared by the fused ingest+index stores (lancedb, qdrant)."""

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from podsearch.core import corpus, fetch

EPISODE_COLUMNS = {
    "feed_url": pl.String,
    "guid": pl.String,
    "podcast": pl.String,
    "title": pl.String,
    "link": pl.String,
    "pubdate": pl.Int64,
    "description": pl.String,
}
CHUNK_COLUMNS = ["feed_url", "guid", "podcast", "episode", "link", "pubdate", "text"]


def latest_fetch_dates(rss_dir: Path) -> pl.DataFrame:
    """Newest fetch_date per feed url across all stored parts."""
    return (
        pl.scan_parquet(rss_dir / "part-*.parquet")
        .group_by("url")
        .agg(latest=pl.col("fetch_date").max())
        .collect()
    )


def parse_feed(feed: dict[str, Any], max_episodes: int, impl: str) -> list[dict[str, Any]]:
    return fetch.parse_feed(feed["rss"], feed["url"], feed["title"] or "", max_episodes, impl)


def chunk_episodes(episodes: list[dict[str, Any]], max_chunks: int) -> pl.DataFrame:
    """Episode dicts -> one row per text chunk, episode metadata broadcast."""

    def clean(expr: pl.Expr) -> pl.Expr:
        return expr.map_elements(corpus.strip_html, return_dtype=pl.String)

    return (
        pl.DataFrame(episodes, schema=EPISODE_COLUMNS)
        .with_columns(
            episode=clean(pl.col("title")),
            text=clean(pl.col("title") + "\n\n" + pl.col("description")),
        )
        .filter(pl.col("text") != "")
        .with_columns(
            pl.col("text").map_elements(
                lambda t: corpus.chunk(t)[:max_chunks], return_dtype=pl.List(pl.String)
            )
        )
        .explode("text")
        .drop_nulls("text")
        .select(CHUNK_COLUMNS)
    )


def embed(model, chunks: pl.DataFrame) -> pl.DataFrame:
    vecs = np.asarray(model.encode(chunks["text"].to_list()), dtype=np.float32)
    return chunks.with_columns(vector=pl.Series(vecs))
