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
