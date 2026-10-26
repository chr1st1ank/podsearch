"""podsearch-catalog: download and extract the Podcast Index feed catalog."""

import argparse
import sqlite3
from pathlib import Path

from podsearch.core.catalog import ensure_db
from podsearch.core.paths import CATALOG_DB, CATALOG_URL


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="podsearch-catalog",
        description="Download and extract the Podcast Index feed catalog.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--db", type=Path, default=CATALOG_DB, help="catalog SQLite database path")
    parser.add_argument("--url", default=CATALOG_URL, help="catalog dump URL to download")
    args = parser.parse_args()
    db = ensure_db(args.db, args.url)
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    count = conn.execute("SELECT COUNT(*) FROM podcasts").fetchone()[0]
    conn.close()
    print(f"{count:,} feeds in catalog")


if __name__ == "__main__":
    main()
