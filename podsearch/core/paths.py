"""Shared data-directory paths."""

import os
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
CATALOG_DB = DATA_DIR / "podcastindex_feeds.db"
CATALOG_URL = "https://public.podcastindex.org/podcastindex_feeds.db.tgz"
RSS_DIR = DATA_DIR / "rss"
CORPUS_DB = DATA_DIR / "corpus.db"
VECTORS_PATH = DATA_DIR / "vectors.f32"
LANCE_DIR = DATA_DIR / "lancedb"
QDRANT_URL = os.environ.get("PODSEARCH_QDRANT_URL", "http://127.0.0.1:6333")
