"""Parquet store for fetched RSS bodies: schema, atomic streaming writer, bookkeeping."""

import logging
import threading
import time
from pathlib import Path
from typing import Any, Self

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from podsearch.core import catalog
from podsearch.core.paths import RSS_DIR

log = logging.getLogger(__name__)

PART_SIZE = 500
ROW_GROUP_BYTES = 64 << 20

_INT_FIELDS = catalog.INT_COLUMNS | {"fetch_date"}
PART_SCHEMA = pa.schema(
    (name, pa.int64() if name in _INT_FIELDS else pa.string())
    for name in (*catalog.CATALOG_COLUMNS, "rss", "fetch_date")
)


def part_table(records: list[dict[str, Any]]) -> pa.Table:
    """Records -> Arrow table in PART_SCHEMA; empty strings become nulls."""

    def coerce(name: str, value: Any) -> Any:
        if value is None or value == "":
            return None
        return int(value) if name in _INT_FIELDS else str(value)

    rows = [{f.name: coerce(f.name, r.get(f.name)) for f in PART_SCHEMA} for r in records]
    return pa.Table.from_pylist(rows, schema=PART_SCHEMA)


class PartWriter:
    """Streams records into part-*.parquet files of at most part_size rows.

    Records are buffered only until row_group_bytes of RSS text have piled up,
    then written as one row group, so memory stays bounded however large the
    feeds are. Parts land under their final name only on close, so a killed
    run leaves no footer-less part-*.parquet behind. Safe to call from
    several threads: writes and close are serialized.
    """

    def __init__(
        self,
        rss_dir: Path,
        part_size: int = PART_SIZE,
        row_group_bytes: int = ROW_GROUP_BYTES,
    ) -> None:
        self._rss_dir, self._part_size, self._row_group_bytes = rss_dir, part_size, row_group_bytes
        self._run = time.time_ns()
        self._n_parts = 0
        self._part: tuple[pq.ParquetWriter, Path] | None = None
        self._rows = 0
        self._buffer: list[dict[str, Any]] = []
        self._buffer_chars = 0
        self._lock = threading.RLock()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def write(self, record: dict[str, Any]) -> None:
        with self._lock:
            if self._part is None:
                final = self._rss_dir / f"part-{self._run}-{self._n_parts:04d}.parquet"
                tmp = final.with_name(final.name + ".tmp")
                self._part = (pq.ParquetWriter(tmp, PART_SCHEMA), tmp)
                self._n_parts += 1
            self._buffer.append(record)
            self._buffer_chars += len(record["rss"])
            self._rows += 1
            if self._rows >= self._part_size:
                self.close()
            elif self._buffer_chars >= self._row_group_bytes:
                self._flush()

    def _flush(self) -> None:
        if self._part and self._buffer:
            self._part[0].write_table(part_table(self._buffer))
            self._buffer, self._buffer_chars = [], 0

    def close(self) -> None:
        with self._lock:
            if self._part is None:
                return
            self._flush()
            writer, tmp = self._part
            self._part, self._rows = None, 0
            writer.close()
            tmp.rename(tmp.with_suffix(""))


def list_parts(rss_dir: Path = RSS_DIR) -> list[Path]:
    """Readable part files in name order; unreadable ones are skipped with a warning."""
    parts = []
    for part in sorted(rss_dir.glob("part-*.parquet")):
        try:
            pq.read_metadata(part)
        except (pa.ArrowException, OSError) as e:
            log.warning("skipping unreadable part %s: %s", part.name, e)
            continue
        parts.append(part)
    return parts


def latest_fetch_dates(rss_dir: Path = RSS_DIR) -> pl.DataFrame:
    """Newest fetch_date per feed url (columns url, latest) across readable parts."""
    parts = list_parts(rss_dir)
    if not parts:
        return pl.DataFrame(schema={"url": pl.String, "latest": pl.Int64})
    return pl.scan_parquet(parts).group_by("url").agg(latest=pl.col("fetch_date").max()).collect()


def already_fetched(rss_dir: Path = RSS_DIR) -> dict[str, int]:
    """Map feed url -> newest fetch_date across the stored parquet parts."""
    return dict(latest_fetch_dates(rss_dir).iter_rows())


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
