"""Tests for podsearch.core.rss_store - schema, atomic part writer, bookkeeping."""

import logging
import os
import time

import polars as pl
import pyarrow.parquet as pq

from podsearch.core import rss_store


def _record(url: str, rss: str = "<rss/>", fetch_date: int = 1) -> dict[str, str | int]:
    return {"url": url, "rss": rss, "fetch_date": fetch_date}


def _write_dates(path, urls: list[str], fetch_date: int) -> None:
    pl.DataFrame({"url": urls, "fetch_date": [fetch_date] * len(urls)}).write_parquet(path)


def test_part_table_coerces_and_fills_missing() -> None:
    table = rss_store.part_table([{**_record("u", fetch_date=5), "lastUpdate": "7", "link": ""}])
    assert table.schema == rss_store.PART_SCHEMA
    row = table.to_pylist()[0]
    assert (row["lastUpdate"], row["link"], row["title"]) == (7, None, None)
    assert (row["url"], row["fetch_date"]) == ("u", 5)


def test_writer_splits_parts_by_row_count(tmp_path) -> None:
    with rss_store.PartWriter(tmp_path, part_size=2) as writer:
        for i in range(5):
            writer.write(_record(f"u{i}"))
    rows = [pq.read_table(p).num_rows for p in sorted(tmp_path.glob("part-*.parquet"))]
    assert rows == [2, 2, 1]


def test_writer_flushes_row_group_by_bytes(tmp_path) -> None:
    with rss_store.PartWriter(tmp_path, part_size=10, row_group_bytes=1) as writer:
        for i in range(3):
            writer.write(_record(f"u{i}"))
    (part,) = tmp_path.glob("part-*.parquet")
    assert pq.ParquetFile(part).num_row_groups == 3


def test_writer_part_appears_only_on_close(tmp_path) -> None:
    writer = rss_store.PartWriter(tmp_path)
    writer.write(_record("u"))
    assert list(tmp_path.glob("part-*.parquet")) == []
    assert len(list(tmp_path.glob("*.tmp"))) == 1
    writer.close()
    assert len(list(tmp_path.glob("part-*.parquet"))) == 1
    assert list(tmp_path.glob("*.tmp")) == []


def test_writers_never_overwrite_each_other(tmp_path) -> None:
    for _ in range(2):
        with rss_store.PartWriter(tmp_path) as writer:
            writer.write(_record("u"))
    assert len(list(tmp_path.glob("part-*.parquet"))) == 2


def test_list_parts_skips_unreadable(tmp_path, caplog) -> None:
    _write_dates(tmp_path / "part-0000.parquet", ["a"], 1)
    (tmp_path / "part-0001.parquet").write_bytes(b"not a parquet file at all")
    with caplog.at_level(logging.WARNING):
        parts = rss_store.list_parts(tmp_path)
    assert [p.name for p in parts] == ["part-0000.parquet"]
    assert "part-0001.parquet" in caplog.text


def test_already_fetched_takes_newest_across_parts(tmp_path) -> None:
    _write_dates(tmp_path / "part-0000.parquet", ["a", "b"], 100)
    _write_dates(tmp_path / "part-0001.parquet", ["a"], 200)
    (tmp_path / "part-0002.parquet").write_bytes(b"corrupt")
    assert rss_store.already_fetched(tmp_path) == {"a": 200, "b": 100}


def test_already_fetched_empty_dir(tmp_path) -> None:
    assert rss_store.already_fetched(tmp_path) == {}
    assert rss_store.latest_fetch_dates(tmp_path).columns == ["url", "latest"]


def test_prune_parts_deletes_old(tmp_path) -> None:
    old, new = tmp_path / "part-0000.parquet", tmp_path / "part-0001.parquet"
    _write_dates(old, ["u"], 1)
    _write_dates(new, ["u"], 1)
    two_weeks_ago = time.time() - 15 * 86_400
    os.utime(old, (two_weeks_ago, two_weeks_ago))

    assert rss_store.prune_parts(tmp_path, max_age_days=14) == 1
    assert not old.exists() and new.exists()
