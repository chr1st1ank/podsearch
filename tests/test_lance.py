"""Tests for podsearch.core.lance_store using synthetic parquet parts."""

import numpy as np
import pyarrow.parquet as pq
import pytest

from podsearch.core import fetch, lance_store

RSS = """<?xml version="1.0"?>
<rss version="2.0"><channel><title>Pod A</title>
<item><title>{title}</title><link>http://x/1</link><guid>g1</guid>
<pubDate>Mon, 01 Jan 2024 00:00:00 GMT</pubDate>
<description>{desc}</description></item>
</channel></rss>
"""


class FakeModel:
    """Encodes every text as the same fixed vector - no model download."""

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.tile(np.asarray([1.0, 0.0], dtype=np.float32), (len(texts), 1))


def _write_part(rss_dir, seq: int, feed_url: str, title: str, fetch_date: int) -> None:
    records = [
        {
            "url": feed_url,
            "title": "Pod A",
            "rss": RSS.format(title=title, desc="alpha beta gamma"),
            "fetch_date": fetch_date,
        }
    ]
    pq.write_table(fetch._part_table(records), rss_dir / f"part-{seq:04d}.parquet")


def test_ingest_and_search_roundtrip(tmp_path) -> None:
    rss_dir = tmp_path / "rss"
    rss_dir.mkdir()
    lance_dir = tmp_path / "lancedb"
    _write_part(rss_dir, 0, "u1", "Episode one", 1000)

    stats = lance_store.ingest_parts(FakeModel(), rss_dir=rss_dir, path=lance_dir)
    assert stats["parts"] == 1
    assert stats["chunks"] > 0

    tbl = lance_store.open_chunks(lance_dir)
    hits = lance_store.search(tbl, FakeModel(), "query", k=5)
    assert len(hits) == 1
    assert hits[0].score == pytest.approx(1.0)
    assert hits[0].podcast == "Pod A"
    assert hits[0].episode == "Episode one"
    assert hits[0].pubdate > 0


def test_ingest_idempotent_and_replace(tmp_path) -> None:
    rss_dir = tmp_path / "rss"
    rss_dir.mkdir()
    lance_dir = tmp_path / "lancedb"
    _write_part(rss_dir, 0, "u1", "Episode one", 1000)

    lance_store.ingest_parts(FakeModel(), rss_dir=rss_dir, path=lance_dir)
    stats = lance_store.ingest_parts(FakeModel(), rss_dir=rss_dir, path=lance_dir)
    assert stats["parts"] == 0 and stats["skipped"] == 1

    # refetched feed arrives as a new part: chunks are replaced, not duplicated
    _write_part(rss_dir, 1, "u1", "Episode one v2", 2000)
    stats = lance_store.ingest_parts(FakeModel(), rss_dir=rss_dir, path=lance_dir)
    assert stats["parts"] == 1
    tbl = lance_store.open_chunks(lance_dir)
    hits = lance_store.search(tbl, FakeModel(), "query", k=5)
    assert [h.episode for h in hits] == ["Episode one v2"]


def test_rebuild_reingests_everything(tmp_path) -> None:
    rss_dir = tmp_path / "rss"
    rss_dir.mkdir()
    lance_dir = tmp_path / "lancedb"
    _write_part(rss_dir, 0, "u1", "Episode one", 1000)

    lance_store.ingest_parts(FakeModel(), rss_dir=rss_dir, path=lance_dir)
    stats = lance_store.ingest_parts(FakeModel(), rss_dir=rss_dir, path=lance_dir, rebuild=True)
    assert stats["parts"] == 1
    hits = lance_store.search(lance_store.open_chunks(lance_dir), FakeModel(), "query")
    assert len(hits) == 1
