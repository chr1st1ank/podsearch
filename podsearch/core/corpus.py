"""Episode corpus: strip show-notes HTML, chunk text, and the corpus.db store."""

import html
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from podsearch.core import parse, rss_store
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
    """Split into word chunks with overlap.

    A leftover tail of at most `overlap` words is folded into the last
    window instead of producing a sliver chunk.
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), size - overlap):
        if len(words) - (i + size) <= overlap:
            chunks.append(" ".join(words[i:]))
            break
        chunks.append(" ".join(words[i : i + size]))
    return chunks


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
    language TEXT NOT NULL DEFAULT 'und',
    UNIQUE(feed_url, guid)
);
CREATE TABLE IF NOT EXISTS chunks(
    id INTEGER PRIMARY KEY,
    vector_row INTEGER NOT NULL,
    episode_id INTEGER NOT NULL REFERENCES episodes(id),
    text TEXT NOT NULL,
    language TEXT NOT NULL DEFAULT 'und'
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
    # ADR-0001: language column was added after first corpora existed
    for table in ("episodes", "chunks"):
        cols = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
        if "language" not in cols:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN language TEXT NOT NULL DEFAULT 'und'")
    return conn


def iter_episodes(conn: sqlite3.Connection):
    """Yield (episode_id, text, language) for indexing."""
    yield from conn.execute("SELECT id, text, language FROM episodes ORDER BY id")


def ingest_parts(
    rss_dir: Path = RSS_DIR,
    db_path: Path = CORPUS_DB,
    max_per_feed: int = 300,
    impl: parse.Parser = "rs",
    max_parts: int | None = None,
    langs: tuple[str, ...] | None = None,
) -> dict[str, int]:
    """Parse not-yet-ingested parquet parts (at most max_parts) into the episodes table."""
    conn = open_corpus(db_path)
    try:
        done = {row[0] for row in conn.execute("SELECT path FROM ingested_parts")}
        parts = [p for p in rss_store.list_parts(rss_dir) if p.name not in done][:max_parts]
        latest = rss_store.already_fetched(rss_dir)
        stats = {"parts": 0, "episodes": 0, "skipped": len(done), "stale": 0}
        for px, part in enumerate(parts):
            print(f"Processing part {px + 1}/{len(parts)}")
            # parts written before ADR-0001 lack the language column -> 'und'
            cols = ["url", "title", "rss", "fetch_date"]
            if "language" in pl.read_parquet_schema(part):
                cols.append("language")
            table = pl.read_parquet(part, columns=cols)
            if "language" not in table.columns:
                table = table.with_columns(language=pl.lit(None, pl.String))
            rows = []
            for url, podcast, body, fetch_date, language in table.select(
                "url", "title", "rss", "fetch_date", "language"
            ).iter_rows():
                if body is None or fetch_date < latest[url]:
                    stats["stale"] += body is not None
                    continue
                if langs is not None and parse.lang_code(language) not in langs:
                    continue
                for ep in parse.parse_feed(body, url, podcast or "", max_per_feed, impl, language):
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
                            ep["language"],
                        )
                    )
            conn.executemany(
                "INSERT INTO episodes"
                "(feed_url, guid, podcast, title, link, pubdate, text, language)"
                " VALUES (?,?,?,?,?,?,?,?)"
                " ON CONFLICT(feed_url, guid) DO UPDATE SET"
                " podcast=excluded.podcast, title=excluded.title,"
                " link=excluded.link, pubdate=excluded.pubdate, text=excluded.text,"
                " language=excluded.language",
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
