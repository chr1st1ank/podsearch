"""Stage 3: parse stored RSS bodies into the episodes corpus."""

import argparse
import time
from pathlib import Path

from podsearch.core import corpus
from podsearch.core.paths import CORPUS_DB, RSS_DIR


def main() -> None:
    parser = argparse.ArgumentParser(prog="podsearch-ingest")
    parser.add_argument("--rss-dir", type=Path, default=RSS_DIR)
    parser.add_argument("--db", type=Path, default=CORPUS_DB)
    parser.add_argument("--max-per-feed", type=int, default=300)
    args = parser.parse_args()
    t0 = time.perf_counter()
    stats = corpus.ingest_parts(args.rss_dir, args.db, args.max_per_feed)
    print(
        f"ingested {stats['parts']} new parts ({stats['skipped']} already done), "
        f"{stats['episodes']:,} episodes ({time.perf_counter() - t0:.1f} s)"
    )


if __name__ == "__main__":
    main()
