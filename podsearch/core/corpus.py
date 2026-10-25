"""Episode corpus: strip show-notes HTML, chunk text, and the corpus.db store."""

import html
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

from podsearch.core import fetch
from podsearch.core.paths import CORPUS_DB, RSS_DIR


@dataclass
class Episode:
    id: int
    feed_url: str
    guid: str
    podcast: str
    title: str
    link: str
    pubdate: int
    text: str


_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")


def strip_html(text: str) -> str:
    """Drop tags, unescape entities, collapse whitespace."""
    text = _TAG.sub(" ", text)
    return _WS.sub(" ", html.unescape(text)).strip()


def chunk(text: str, size: int = 200, overlap: int = 40) -> list[str]:
    """Split into word chunks with overlap."""
    words = text.split()
    return [" ".join(words[i : i + size]) for i in range(0, len(words), size - overlap)]


_SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes(
    id INTEGER PRIMARY KEY,
    feed_url TEXT NOT NULL,
    guid TEXT NOT NULL,
    podcast TEXT NOT NULL,
    title TEXT NOT NULL,
    link TEXT,
    pubdate INTEGER,
    text TEXT NOT NULL,
    UNIQUE(feed_url, guid)
);
CREATE TABLE IF NOT EXISTS chunks(
    id INTEGER PRIMARY KEY,
    vector_row INTEGER NOT NULL,
    episode_id INTEGER NOT NULL REFERENCES episodes(id),
    text TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS ingested_parts(
    path TEXT PRIMARY KEY,
    ingested_at INTEGER
);
CREATE TABLE IF NOT EXISTS meta(
    key TEXT PRIMARY KEY,
    value TEXT
);
CREATE INDEX IF NOT EXISTS idx_chunks_episode ON chunks(episode_id);

-- TODO: Following index is crucial for query performance
CREATE INDEX IF NOT EXISTS idx_chunks_vector_row ON chunks(vector_row);
"""


def open_corpus(path: Path = CORPUS_DB) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(_SCHEMA)
    return conn


def iter_episodes(conn: sqlite3.Connection):
    """Yield (episode_id, text) for indexing."""
    yield from conn.execute("SELECT id, text FROM episodes ORDER BY id")


def ingest_parts(
    rss_dir: Path = RSS_DIR,
    db_path: Path = CORPUS_DB,
    max_per_feed: int = 300,
    impl: str = "rs",
) -> dict[str, int]:
    """Parse not-yet-ingested parquet parts into the episodes table."""
    conn = open_corpus(db_path)
    try:
        done = {row[0] for row in conn.execute("SELECT path FROM ingested_parts")}
        parts = [p for p in sorted(rss_dir.glob("part-*.parquet")) if p.name not in done]
        latest = fetch.already_fetched(rss_dir)
        stats = {"parts": 0, "episodes": 0, "skipped": len(done), "stale": 0}
        for px, part in enumerate(parts):
            print(f"Processing part {px + 1}/{len(parts)}")
            table = pq.read_table(part, columns=["url", "title", "rss", "fetch_date"])
            rows = []
            for url, podcast, body, fetch_date in zip(
                table["url"].to_pylist(),
                table["title"].to_pylist(),
                table["rss"].to_pylist(),
                table["fetch_date"].to_pylist(),
                strict=True,
            ):
                if body is None or fetch_date < latest[url]:
                    stats["stale"] += body is not None
                    continue
                for ep in fetch.parse_feed(body, url, podcast or "", max_per_feed, impl):
                    text = strip_html(f"{ep['title']}\n\n{ep['description']}")
                    if not text:
                        continue
                    rows.append(
                        (
                            ep["feed_url"],
                            ep["guid"],
                            ep["podcast"],
                            strip_html(ep["title"]),
                            ep["link"],
                            ep["pubdate"],
                            text,
                        )
                    )
            conn.executemany(
                "INSERT INTO episodes"
                "(feed_url, guid, podcast, title, link, pubdate, text)"
                " VALUES (?,?,?,?,?,?,?)"
                " ON CONFLICT(feed_url, guid) DO UPDATE SET"
                " podcast=excluded.podcast, title=excluded.title,"
                " link=excluded.link, pubdate=excluded.pubdate, text=excluded.text",
                rows,
            )
            conn.execute(
                "INSERT INTO ingested_parts(path, ingested_at) VALUES (?,?)",
                (part.name, int(time.time())),
            )
            conn.commit()
            stats["parts"] += 1
            stats["episodes"] += len(rows)
        return stats
    finally:
        conn.close()
