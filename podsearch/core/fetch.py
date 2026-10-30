"""Stage 2: fetch RSS bodies for selected feeds into the parquet store.

Contract with every later stage: the stored `rss` text is valid Unicode and
never declares another encoding. Legacy encodings are resolved here, once, by
`decode_feed`.
"""

import asyncio
import codecs
import logging
import re
import time
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

from podsearch.core import rss_store
from podsearch.core.paths import RSS_DIR

log = logging.getLogger(__name__)

UA = {"User-Agent": "podsearch/0.1 (podcast feed indexer)"}
FETCH_TIMEOUT = 10  # per connect/read operation
FETCH_DEADLINE = 180  # whole request, so a slow-drip server cannot hold a slot forever
# Out-of-memory guard, not a truncation policy: ~3x the largest feed seen in practice.
MAX_FEED_BYTES = 100 << 20
PER_HOST = 4

_BOMS = (
    (codecs.BOM_UTF8, "utf-8-sig"),
    (codecs.BOM_UTF32_LE, "utf-32"),
    (codecs.BOM_UTF32_BE, "utf-32"),
    (codecs.BOM_UTF16_LE, "utf-16"),
    (codecs.BOM_UTF16_BE, "utf-16"),
)
_DECLARED_ENCODING = re.compile(rb"""^\s*<\?xml[^>]*?\sencoding\s*=\s*["']([\w.:-]+)["']""", re.I)
_DECLARATION = re.compile(r"""^(\s*<\?xml[^>]*?\sencoding\s*=\s*["'])[\w.:-]+(["'])""", re.I)


@dataclass(frozen=True)
class FetchStats:
    selected: int
    pending: int
    ok: int
    failed: int


def _decode(body: bytes, encodings: Iterable[str | None]) -> str:
    for encoding in filter(None, encodings):
        try:
            return body.decode(encoding)
        except LookupError, UnicodeDecodeError:
            pass
    return body.decode("utf-8", errors="replace")


def decode_feed(body: bytes, charset: str | None = None) -> str:
    """Decode an RSS body to str and rewrite its XML declaration to utf-8.

    Encoding precedence: BOM, XML declaration, HTTP charset, UTF-8; replacement
    characters only if all of those fail. The declaration is rewritten because
    parsers re-apply it to the text they are handed (feedparser_rs turns
    "Grüße" declared as ISO-8859-1 into mojibake otherwise).
    """
    bom = next((encoding for mark, encoding in _BOMS if body.startswith(mark)), None)
    if bom:
        encodings = [bom]
    else:
        declared = _DECLARED_ENCODING.match(body[:512])
        encodings = [declared and declared.group(1).decode("ascii"), charset, "utf-8"]
    return _DECLARATION.sub(r"\1utf-8\2", _decode(body, encodings), count=1)


def pending_feeds(feeds: list[dict[str, Any]], rss_dir: Path = RSS_DIR) -> list[dict[str, Any]]:
    """Feeds we have not stored yet, or whose catalog lastUpdate is newer (one per url)."""
    fetched = rss_store.already_fetched(rss_dir)
    due = {
        f["url"]: f
        for f in feeds
        if f["url"] not in fetched or (f.get("lastUpdate") or 0) > fetched[f["url"]]
    }
    return list(due.values())


def _host(url: str) -> str:
    try:
        return urlsplit(url).hostname or ""
    except ValueError:
        return ""


async def _fetch_one(client: httpx.AsyncClient, url: str, max_bytes: int) -> str | None:
    """GET url and return its decoded text; None on any failure (re-runs retry it)."""
    try:
        async with asyncio.timeout(FETCH_DEADLINE), client.stream("GET", url) as r:
            if r.status_code != 200:
                return None
            body = bytearray()
            async for chunk in r.aiter_bytes():
                body += chunk
                if len(body) > max_bytes:
                    log.warning("skipping %s: body exceeds %d MB", url, max_bytes >> 20)
                    return None
            return decode_feed(bytes(body), r.charset_encoding) if body else None
    except (httpx.HTTPError, httpx.InvalidURL, TimeoutError, UnicodeError) as e:
        log.debug("fetch failed for %s: %r", url, e)
        return None


async def _fetch_pending(
    pending: list[dict[str, Any]],
    rss_dir: Path,
    concurrency: int,
    part_size: int,
    max_feed_bytes: int,
    per_host: int,
    transport: httpx.AsyncBaseTransport | None,
) -> Counter[str]:
    counts: Counter[str] = Counter()
    slots = asyncio.Semaphore(concurrency)
    hosts: defaultdict[str, asyncio.Semaphore] = defaultdict(lambda: asyncio.Semaphore(per_host))
    write_lock = asyncio.Lock()

    async def process(feed: dict[str, Any]) -> None:
        # a slot is held until the body is written, so at most `concurrency` bodies are alive
        async with hosts[_host(feed["url"])], slots:
            body = await _fetch_one(client, feed["url"], max_feed_bytes)
            if body is None:
                counts["failed"] += 1
            else:
                counts["ok"] += 1
                record = {**feed, "rss": body, "fetch_date": int(time.time())}
                async with write_lock:
                    await asyncio.to_thread(writer.write, record)
        if counts.total() % 500 == 0:
            log.info("%d/%d fetched (%d failed)", counts.total(), len(pending), counts["failed"])

    with rss_store.PartWriter(rss_dir, part_size) as writer:
        async with httpx.AsyncClient(
            headers=UA,
            follow_redirects=True,
            timeout=httpx.Timeout(FETCH_TIMEOUT),
            limits=httpx.Limits(max_connections=concurrency),
            transport=transport,
        ) as client:
            tasks = [asyncio.create_task(process(f)) for f in pending]
            try:
                await asyncio.gather(*tasks)
            finally:
                # on error or Ctrl-C do not keep fetching the remaining feeds
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
    return counts


def fetch_rss(
    feeds: list[dict[str, Any]],
    rss_dir: Path = RSS_DIR,
    concurrency: int = 32,
    part_size: int = rss_store.PART_SIZE,
    max_feed_bytes: int = MAX_FEED_BYTES,
    per_host: int = PER_HOST,
    transport: httpx.AsyncBaseTransport | None = None,
) -> FetchStats:
    """Fetch the pending feeds and append results to the parquet store.

    Runs its own event loop, so it cannot be called from async code.
    """
    rss_dir.mkdir(parents=True, exist_ok=True)
    pending = pending_feeds(feeds, rss_dir)
    counts = (
        asyncio.run(
            _fetch_pending(
                pending, rss_dir, concurrency, part_size, max_feed_bytes, per_host, transport
            )
        )
        if pending
        else Counter()
    )
    return FetchStats(len(feeds), len(pending), counts["ok"], counts["failed"])
