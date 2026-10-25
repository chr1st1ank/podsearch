"""Stage 2: fetch RSS bodies for selected feeds into the parquet store."""

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import feedparser
import httpx
import pyarrow as pa
import pyarrow.parquet as pq

from podsearch.core import catalog
from podsearch.core.paths import CATALOG_DB, RSS_DIR

UA = {"User-Agent": "podsearch-demo/0.1 (profiling demo; one-off fetch)"}
PART_SIZE = 500
FETCH_TIMEOUT = 10

# podcastindex catalog columns that are INTEGER in sqlite (rest are TEXT).
_INT_COLS = {
    "id",
    "lastUpdate",
    "lastHttpStatus",
    "dead",
    "itunesId",
    "explicit",
    "newestItemPubdate",
    "oldestItemPubdate",
    "episodeCount",
    "popularityScore",
    "priority",
    "createdOn",
    "updateFrequency",
    "newestEnclosureDuration",
    "podcastId",
    "duplicateOf",
}


def _coerce(value: Any, is_int: bool) -> Any:
    if value is None or value == "":
        return None
    return int(value) if is_int else str(value)


def _part_table(records: list[dict[str, Any]]) -> pa.Table:
    cols: dict[str, Any] = {}
    for name in catalog.CATALOG_COLUMNS:
        is_int = name in _INT_COLS
        arrow_type = pa.int64() if is_int else pa.string()
        cols[name] = pa.array([_coerce(r.get(name), is_int) for r in records], type=arrow_type)
    cols["rss"] = pa.array([r["rss"] for r in records], type=pa.string())
    cols["fetch_date"] = pa.array([r["fetch_date"] for r in records], type=pa.int64())
    return pa.table(cols)


def already_fetched(rss_dir: Path = RSS_DIR) -> dict[str, int]:
    """Map feed url -> newest fetch_date across the stored parquet parts."""
    fetched: dict[str, int] = {}
    for part in sorted(rss_dir.glob("part-*.parquet")):
        table = pq.read_table(part, columns=["url", "fetch_date"])
        for url, ts in zip(table["url"].to_pylist(), table["fetch_date"].to_pylist(), strict=True):
            if ts > fetched.get(url, 0):
                fetched[url] = ts
    return fetched


def pending_feeds(feeds: list[dict[str, Any]], rss_dir: Path = RSS_DIR) -> list[dict[str, Any]]:
    """Feeds we have not stored yet, or whose catalog lastUpdate is newer."""
    fetched = already_fetched(rss_dir)
    return [
        f
        for f in feeds
        if f["url"] not in fetched or (f.get("lastUpdate") or 0) > fetched[f["url"]]
    ]


def _fetch_one(client: httpx.Client, url: str) -> bytes | None:
    try:
        r = client.get(url, timeout=FETCH_TIMEOUT)
        if r.status_code == 200 and r.content:
            return r.content
    except httpx.HTTPError:
        pass
    return None


def write_part(records: list[dict[str, Any]], rss_dir: Path, seq: int) -> Path:
    path = rss_dir / f"part-{seq:04d}.parquet"
    pq.write_table(_part_table(records), path)
    return path


def fetch_rss(
    feeds: list[dict[str, Any]],
    rss_dir: Path = RSS_DIR,
    concurrency: int = 32,
    part_size: int = PART_SIZE,
) -> dict[str, int]:
    """Fetch the pending feeds and append results to the parquet store."""
    rss_dir.mkdir(parents=True, exist_ok=True)
    pending = pending_feeds(feeds, rss_dir)
    stats = {"selected": len(feeds), "pending": len(pending), "ok": 0, "failed": 0}
    if not pending:
        return stats
    seq = int(time.time())
    n_part = 0
    batch: list[dict[str, Any]] = []
    client = httpx.Client(headers=UA, follow_redirects=True)
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(_fetch_one, client, f["url"]): f for f in pending}
        for i, fut in enumerate(as_completed(futures), 1):
            feed = futures[fut]
            body = fut.result()
            if body is None:
                stats["failed"] += 1
            else:
                stats["ok"] += 1
                batch.append(
                    {
                        **feed,
                        "rss": body.decode("utf-8", errors="replace"),
                        "fetch_date": int(time.time()),
                    }
                )
            if len(batch) >= part_size:
                write_part(batch, rss_dir, seq * 10_000 + n_part)
                n_part += 1
                batch = []
            if i % 500 == 0:
                print(f"  {i}/{len(pending)} fetched ({stats['failed']} failed)")
    if batch:
        write_part(batch, rss_dir, seq * 10_000 + n_part)
    client.close()
    return stats


def parse_feed(
    body: str, feed_url: str, podcast: str, max_episodes: int | None = None
) -> list[dict[str, Any]]:
    """Parse one RSS body into episode dicts. Used by the ingest stage."""
    import calendar

    parsed = feedparser.parse(body)
    feed_title = parsed.feed.get("title")
    podcast_name = podcast or (str(feed_title) if feed_title else "")
    episodes: list[dict[str, Any]] = []
    entries = parsed.entries[:max_episodes] if max_episodes is not None else parsed.entries
    for entry in entries:
        if entry.get("content"):
            description = entry.content[0].get("value", "")
        else:
            description = entry.get("summary") or entry.get("itunes_summary") or ""
        guid = entry.get("id") or entry.get("link") or entry.get("title") or ""
        parsed_date = entry.get("published_parsed") or entry.get("updated_parsed")
        episodes.append(
            {
                "feed_url": feed_url,
                "guid": guid,
                "podcast": podcast_name,
                "title": entry.get("title") or "(untitled)",
                "link": entry.get("link") or "",
                "pubdate": calendar.timegm(parsed_date) if parsed_date else 0,
                "description": description,
            }
        )
    return episodes


def prune_parts(rss_dir: Path = RSS_DIR, max_age_days: int = 14) -> int:
    """Delete parquet parts older than max_age_days; returns the count removed.

    Feeds whose only copies were in pruned parts look unfetched on the next
    run and get refetched - the store's TTL is the refresh policy. Episodes
    already ingested from them stay in corpus.db.
    """
    cutoff = time.time() - max_age_days * 86_400
    removed = 0
    for part in sorted(rss_dir.glob("part-*.parquet")):
        if part.stat().st_mtime < cutoff:
            part.unlink()
            removed += 1
    return removed


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
