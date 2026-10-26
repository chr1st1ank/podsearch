"""Stage 3: parse stored RSS bodies into the episodes corpus."""

import argparse
import time
from pathlib import Path

from podsearch.core import corpus, index, lance_store
from podsearch.core.paths import CORPUS_DB, LANCE_DIR, RSS_DIR


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="podsearch-ingest",
        description="Parse stored RSS bodies into the episodes corpus.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--rss-dir", type=Path, default=RSS_DIR, help="directory with fetched parquet parts"
    )
    parser.add_argument("--db", type=Path, default=CORPUS_DB, help="corpus SQLite database")
    parser.add_argument(
        "--impl",
        choices=["sqlite", "lancedb"],
        default="sqlite",
        help="storage backend: sqlite episodes (index stage follows) or fused lancedb ingest+index",
    )
    parser.add_argument(
        "--lance-dir", type=Path, default=LANCE_DIR, help="lancedb store directory (--impl lancedb)"
    )
    parser.add_argument(
        "--max-per-feed", type=int, default=300, help="maximum episodes ingested per feed"
    )
    parser.add_argument(
        "--parser", choices=["feedparser", "rs"], default="rs", help="feed parser implementation"
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="drop and recreate the lancedb store, re-ingesting every part present",
    )
    args = parser.parse_args()
    if args.rebuild and args.impl != "lancedb":
        parser.error("--rebuild only applies to --impl lancedb")
    t0 = time.perf_counter()
    if args.impl == "lancedb":
        stats = lance_store.ingest_parts(
            index.load_model(),
            rss_dir=args.rss_dir,
            path=args.lance_dir,
            max_per_feed=args.max_per_feed,
            parser_impl=args.parser,
            rebuild=args.rebuild,
        )
        print(
            f"ingested {stats['parts']} new parts ({stats['skipped']} already done), "
            f"{stats['chunks']:,} chunks ({time.perf_counter() - t0:.1f} s)"
        )
    else:
        stats = corpus.ingest_parts(args.rss_dir, args.db, args.max_per_feed, args.parser)
        print(
            f"ingested {stats['parts']} new parts ({stats['skipped']} already done), "
            f"{stats['episodes']:,} episodes ({time.perf_counter() - t0:.1f} s)"
        )


if __name__ == "__main__":
    main()
