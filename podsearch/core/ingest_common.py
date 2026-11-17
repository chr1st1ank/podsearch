"""Helpers shared by the fused ingest+index stores (lancedb, qdrant)."""

from typing import Any

import numpy as np
import polars as pl

from podsearch.core import corpus, parse

EPISODE_COLUMNS = {
    "feed_url": pl.String,
    "guid": pl.String,
    "podcast": pl.String,
    "title": pl.String,
    "link": pl.String,
    "pubdate": pl.Int64,
    "description": pl.String,
    "language": pl.String,
}
CHUNK_COLUMNS = ["feed_url", "guid", "podcast", "episode", "link", "pubdate", "text", "language"]


def parse_feed(
    feed: dict[str, Any],
    max_episodes: int,
    langs: tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    """Parse one feed; feeds whose normalized language is not in langs yield nothing."""
    if langs is not None and parse.lang_code(feed.get("language")) not in langs:
        return []
    return parse.parse_feed(
        feed["rss"],
        feed["url"],
        feed["title"] or "",
        max_episodes,
        feed.get("language"),
    )


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
        .unique(subset=["feed_url", "guid"], keep="last")
        .with_columns(
            pl.col("text").map_elements(
                lambda t: corpus.chunk(t)[:max_chunks], return_dtype=pl.List(pl.String)
            )
        )
        .explode("text")
        .drop_nulls("text")
        .unique(subset=["feed_url", "guid", "text"], keep="first")
        .select(CHUNK_COLUMNS)
    )


def embed(model, chunks: pl.DataFrame) -> pl.DataFrame:
    vecs = np.asarray(model.encode(chunks["text"].to_list()), dtype=np.float32)
    return chunks.with_columns(vector=pl.Series(vecs))
