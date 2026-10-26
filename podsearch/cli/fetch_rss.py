"""podsearch-fetch-rss: select catalog feeds and fetch their RSS into parquet parts."""

import argparse
from pathlib import Path

from podsearch.core import catalog
from podsearch.core.fetch import fetch_rss, prune_parts
from podsearch.core.paths import CATALOG_DB, RSS_DIR


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="podsearch-fetch-rss",
        description="Select catalog feeds and fetch their RSS into parquet parts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--db", type=Path, default=CATALOG_DB, help="catalog SQLite database")
    parser.add_argument(
        "--rss-dir", type=Path, default=RSS_DIR, help="output directory for parquet parts"
    )
    parser.add_argument(
        "--feeds", type=int, default=10_000, help="maximum number of feeds to select"
    )
    parser.add_argument(
        "--lang",
        action="append",
        default=None,
        help="language filter, repeatable or comma-separated; 'en,de' when omitted",
    )
    parser.add_argument(
        "--min-episodes", type=int, default=20, help="minimum episode count per selected feed"
    )
    parser.add_argument(
        "--updated-since",
        type=int,
        default=0,
        help="only feeds with newest episode at/after this unix timestamp",
    )
    parser.add_argument(
        "--order",
        choices=["recent", "episodes", "popularity"],
        default="recent",
        help="feed selection order",
    )
    parser.add_argument("--concurrency", type=int, default=32, help="parallel feed fetches")
    parser.add_argument(
        "--prune-days",
        type=int,
        default=14,
        help="delete parquet parts older than this many days (0 disables)",
    )
    args = parser.parse_args()
    feeds = catalog.select_feeds(
        args.db,
        langs=tuple(lang for arg in (args.lang or ["en", "de"]) for lang in arg.split(",") if lang),
        min_episodes=args.min_episodes,
        updated_since=args.updated_since,
        limit=args.feeds,
        order=args.order,
    )
    stats = fetch_rss(feeds, args.rss_dir, args.concurrency)
    print(
        f"selected {stats['selected']}, pending {stats['pending']}, "
        f"ok {stats['ok']}, failed {stats['failed']}"
    )
    if args.prune_days > 0:
        removed = prune_parts(args.rss_dir, args.prune_days)
        if removed:
            print(f"pruned {removed} parts older than {args.prune_days} days")


if __name__ == "__main__":
    main()
