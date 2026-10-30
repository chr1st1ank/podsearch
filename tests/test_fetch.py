"""Tests for podsearch.core.fetch - parsing and fetch idempotency (no network)."""

import pyarrow as pa
import pyarrow.parquet as pq

from podsearch.core import fetch

RSS = b"""<?xml version="1.0"?>
<rss version="2.0">
  <channel>
    <title>Feed Title</title>
    <item>
      <title>Episode One</title>
      <link>https://example.com/1</link>
      <guid>guid-1</guid>
      <pubDate>Mon, 01 Jan 2024 00:00:00 +0000</pubDate>
      <description>&lt;p&gt;Shownotes &lt;b&gt;here&lt;/b&gt;&lt;/p&gt;</description>
      <enclosure url="https://example.com/1.mp3" type="audio/mpeg"/>
    </item>
    <item>
      <title>Episode Two</title>
      <guid>guid-2</guid>
    </item>
  </channel>
</rss>
"""


def test_parse_feed_entries() -> None:
    eps = fetch.parse_feed(RSS.decode(), "https://feed", "Feed Title")
    assert len(eps) == 2
    first = eps[0]
    assert first["feed_url"] == "https://feed"
    assert first["guid"] == "guid-1"
    assert first["title"] == "Episode One"
    assert first["link"] == "https://example.com/1"
    assert first["pubdate"] == 1_704_067_200
    assert "<p>" in first["description"]  # stripped later, at ingest


def test_parse_feed_max_episodes() -> None:
    assert len(fetch.parse_feed(RSS.decode(), "u", "p", max_episodes=1)) == 1


def test_parse_feed_rs_matches_feedparser() -> None:
    args = (RSS.decode(), "https://feed", "Feed Title")
    assert fetch.parse_feed(*args, impl="rs") == fetch.parse_feed(*args, impl="feedparser")


def _feed(url: str, last_update: int) -> dict[str, str | int]:
    return {"url": url, "lastUpdate": last_update}


def test_pending_feeds_skips_stored(tmp_path) -> None:
    rss_dir = tmp_path / "rss"
    rss_dir.mkdir()
    part = pa.table(
        {
            "url": ["https://a", "https://b"],
            "fetch_date": [100, 100],
        }
    )
    pq.write_table(part, rss_dir / "part-0000.parquet")

    feeds = [
        _feed("https://a", 50),  # stored, no update -> skip
        _feed("https://b", 150),  # stored but updated since -> refetch
        _feed("https://c", 0),  # never stored -> fetch
    ]
    pending = {f["url"] for f in fetch.pending_feeds(feeds, rss_dir)}
    assert pending == {"https://b", "https://c"}


def test_fetch_rss_writes_parts(tmp_path, monkeypatch) -> None:
    """fetch_rss consumes each completed future and writes bounded-size parts."""
    monkeypatch.setattr(fetch, "_fetch_one", lambda client, url: RSS)
    rss_dir = tmp_path / "rss"
    feeds = [_feed("https://a", 0), _feed("https://b", 0)]
    stats = fetch.fetch_rss(feeds, rss_dir, part_size=1)
    assert stats == {"selected": 2, "pending": 2, "ok": 2, "failed": 0}
    parts = sorted(rss_dir.glob("part-*.parquet"))
    assert len(parts) == 2  # part_size=1 -> one part per feed
    assert pq.read_table(parts[0])["rss"].to_pylist() == [RSS.decode()]


def test_fetch_rss_counts_failures(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(fetch, "_fetch_one", lambda client, url: None)
    rss_dir = tmp_path / "rss"
    stats = fetch.fetch_rss([_feed("https://a", 0)], rss_dir)
    assert stats == {"selected": 1, "pending": 1, "ok": 0, "failed": 1}
    assert list(rss_dir.glob("part-*.parquet")) == []


def test_fetch_rss_error_leaves_valid_partial_part(tmp_path, monkeypatch) -> None:
    """An exception mid-run must not leave a footer-less part behind."""

    def flaky(client, url: str) -> bytes | None:
        if url == "https://bad":
            raise RuntimeError("boom")
        return RSS

    monkeypatch.setattr(fetch, "_fetch_one", flaky)
    rss_dir = tmp_path / "rss"
    feeds = [_feed("https://a", 0), _feed("https://bad", 0)]
    import pytest

    with pytest.raises(RuntimeError, match="boom"):
        fetch.fetch_rss(feeds, rss_dir, concurrency=1)
    assert list(rss_dir.glob("*.tmp")) == []
    parts = sorted(rss_dir.glob("part-*.parquet"))
    assert len(parts) == 1  # rows written before the error are still valid
    assert pq.read_table(parts[0]).num_rows == 1


def test_prune_parts_deletes_old(tmp_path) -> None:
    import os
    import time

    rss_dir = tmp_path / "rss"
    rss_dir.mkdir()
    part = pa.table({"url": ["u"], "fetch_date": [1]})
    old = rss_dir / "part-0000.parquet"
    new = rss_dir / "part-0001.parquet"
    pq.write_table(part, old)
    pq.write_table(part, new)
    two_weeks_ago = time.time() - 15 * 86_400
    os.utime(old, (two_weeks_ago, two_weeks_ago))

    assert fetch.prune_parts(rss_dir, max_age_days=14) == 1
    assert not old.exists() and new.exists()
