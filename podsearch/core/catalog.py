"""Stage 1: fetch the Podcast Index feed catalog and select feeds to crawl."""

import sqlite3
import tarfile
import tempfile
from pathlib import Path
from typing import Any

import httpx

from podsearch.core.paths import CATALOG_DB, CATALOG_URL

CATALOG_COLUMNS = [
    "id",
    "url",
    "title",
    "lastUpdate",
    "link",
    "lastHttpStatus",
    "dead",
    "contentType",
    "itunesId",
    "originalUrl",
    "itunesAuthor",
    "itunesOwnerName",
    "explicit",
    "imageUrl",
    "itunesType",
    "generator",
    "newestItemPubdate",
    "language",
    "oldestItemPubdate",
    "episodeCount",
    "popularityScore",
    "priority",
    "createdOn",
    "updateFrequency",
    "chash",
    "host",
    "newestEnclosureUrl",
    "podcastGuid",
    "description",
    "category1",
    "category2",
    "category3",
    "category4",
    "category5",
    "category6",
    "category7",
    "category8",
    "category9",
    "category10",
    "newestEnclosureDuration",
    "podcastId",
    "duplicateOf",
]


def ensure_db(db_path: Path = CATALOG_DB, url: str = CATALOG_URL) -> Path:
    """Download and extract the feed catalog if it is not already there."""
    if db_path.exists():
        print(f"catalog already present: {db_path}")
        return db_path
    db_path.parent.mkdir(parents=True, exist_ok=True)
    tgz = db_path.with_name(db_path.name + ".tgz")
    cleanup = not tgz.exists()
    if cleanup:
        tgz = _download(url, db_path.parent)
    else:
        print(f"using local archive: {tgz}")
    try:
        with tarfile.open(tgz, "r:gz") as tar, db_path.open("wb") as out:
            member = next(m for m in tar.getmembers() if m.name.endswith(".db"))
            src = tar.extractfile(member)
            if src is None:
                raise RuntimeError(f"cannot extract {member.name} from {tgz}")
            while True:
                data = src.read(1 << 20)
                if not data:
                    break
                out.write(data)
    finally:
        if cleanup:
            tgz.unlink(missing_ok=True)
    print(f"extracted catalog to {db_path}")
    return db_path


def _download(url: str, dest_dir: Path) -> Path:
    # public.podcastindex.org 403s non-browser user agents (incl. the httpx default)
    ua = (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
        " (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
    )
    headers = {"User-Agent": ua}
    with tempfile.NamedTemporaryFile(dir=dest_dir, suffix=".tgz", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        try:
            with httpx.stream(
                "GET", url, timeout=None, follow_redirects=True, headers=headers
            ) as r:
                r.raise_for_status()
                for data in r.iter_bytes(1 << 20):
                    tmp.write(data)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise
    return tmp_path


def select_feeds(
    db_path: Path = CATALOG_DB,
    langs: tuple[str, ...] = ("en", "de"),
    min_episodes: int = 20,
    updated_since: int = 0,
    limit: int = 10_000,
    order: str = "recent",
) -> list[dict[str, Any]]:
    """Pick feeds from the catalog; returns one dict per feed (all columns).

    Feeds sharing a non-empty chash are the same feed content under alias
    URLs; keep only the first per chash in the chosen order.
    """
    order_sql = {
        "recent": "newestItemPubdate DESC",
        "episodes": "episodeCount DESC",
        "popularity": "popularityScore DESC, newestItemPubdate DESC",
    }[order]
    lang_sql = " OR ".join("language LIKE ?" for _ in langs)
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            f"SELECT * FROM podcasts WHERE episodeCount >= ?"
            f" AND newestItemPubdate >= ? AND ({lang_sql})"
            f" ORDER BY {order_sql}",
            (min_episodes, updated_since, *(f"{lang}%" for lang in langs)),
        )
        feeds: list[dict[str, Any]] = []
        seen: set[str] = set()
        for row in rows:
            if row["chash"] and row["chash"] in seen:
                continue
            seen.add(row["chash"])
            feeds.append(dict(row))
            if len(feeds) >= limit:
                break
    finally:
        conn.close()
    return feeds
