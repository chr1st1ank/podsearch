"""Tests for podsearch.core.corpus.ingest_parts - upsert and stale-version skip."""

import pyarrow as pa
import pyarrow.parquet as pq

from podsearch.core import corpus

RSS_TEMPLATE = """<?xml version="1.0"?>
<rss version="2.0"><channel><title>Pod</title>
  <item><title>Ep</title><guid>g1</guid>
    <description>{desc}</description></item>
</channel></rss>
"""


def _part(rss_dir, name, url, desc, fetch_date) -> None:
    table = pa.table(
        {
            "url": [url],
            "title": ["Pod"],
            "rss": [RSS_TEMPLATE.format(desc=desc)],
            "fetch_date": [fetch_date],
        }
    )
    pq.write_table(table, rss_dir / name)


def _episode_texts(db_path) -> list[str]:
    conn = corpus.open_corpus(db_path)
    texts = [str(r[0]) for r in conn.execute("SELECT text FROM episodes")]
    conn.close()
    return texts


def test_ingest_upserts_newer_feed_version(tmp_path) -> None:
    rss_dir = tmp_path / "rss"
    rss_dir.mkdir()
    db = tmp_path / "corpus.db"
    _part(rss_dir, "part-0000.parquet", "https://f", "old notes", 100)
    corpus.ingest_parts(rss_dir, db)
    assert _episode_texts(db) == ["Ep old notes"]

    # same feed refetched later with different description -> text updates
    _part(rss_dir, "part-0001.parquet", "https://f", "new notes", 200)
    corpus.ingest_parts(rss_dir, db)
    assert _episode_texts(db) == ["Ep new notes"]


def test_ingest_skips_stale_feed_version(tmp_path) -> None:
    rss_dir = tmp_path / "rss"
    rss_dir.mkdir()
    db = tmp_path / "corpus.db"
    # both parts uningested; only the newest fetch of the feed should parse
    _part(rss_dir, "part-0000.parquet", "https://f", "new notes", 200)
    _part(rss_dir, "part-0001.parquet", "https://f", "old notes", 100)
    corpus.ingest_parts(rss_dir, db)
    assert _episode_texts(db) == ["Ep new notes"]
