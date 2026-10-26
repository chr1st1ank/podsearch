"""Tests for podsearch.core.catalog.select_feeds - chash alias dedup."""

import sqlite3

from podsearch.core import catalog


def _catalog_db(db_path, rows) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE podcasts(url, title, episodeCount, newestItemPubdate, language, chash)"
    )
    conn.executemany("INSERT INTO podcasts VALUES (?,?,?,?,?,?)", rows)
    conn.commit()
    conn.close()


def test_select_feeds_dedups_chash_aliases(tmp_path) -> None:
    db = tmp_path / "catalog.db"
    _catalog_db(
        db,
        [
            ("u1", "Feed A", 100, 2100, "en", "hash1"),
            ("u2", "Feed A alias", 100, 2000, "en", "hash1"),
            ("u3", "Feed B", 50, 1000, "en", "hash2"),
            ("u4", "Feed C no hash", 50, 900, "en", ""),
            ("u5", "Feed D no hash", 50, 800, "en", ""),
        ],
    )
    feeds = catalog.select_feeds(db, langs=("en",), min_episodes=0)
    assert [f["url"] for f in feeds] == ["u1", "u3", "u4", "u5"]


def test_select_feeds_limit_counts_unique_feeds(tmp_path) -> None:
    db = tmp_path / "catalog.db"
    _catalog_db(
        db,
        [
            ("u1", "Feed A", 100, 2100, "en", "hash1"),
            ("u2", "Feed A alias", 100, 2000, "en", "hash1"),
            ("u3", "Feed B", 50, 1000, "en", "hash2"),
        ],
    )
    feeds = catalog.select_feeds(db, langs=("en",), min_episodes=0, limit=2)
    assert [f["url"] for f in feeds] == ["u1", "u3"]
