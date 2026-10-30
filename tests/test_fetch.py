"""Tests for podsearch.core.fetch - decoding and fetch behavior (no network)."""

import asyncio
import time

import httpx
import polars as pl
import pytest

from podsearch.core import fetch, parse, rss_store
from podsearch.core.fetch import FetchStats

RSS = b"""<?xml version="1.0"?>
<rss version="2.0"><channel><title>Feed Title</title>
  <item><title>Episode One</title><guid>guid-1</guid></item>
</channel></rss>
"""

LATIN1_RSS = (
    '<?xml version="1.0" encoding="ISO-8859-1"?>'
    '<rss version="2.0"><channel><title>Pod</title>'
    "<item><title>Grüße</title><guid>g</guid></item></channel></rss>"
)


def _feed(url: str, last_update: int = 0) -> dict[str, str | int]:
    return {"url": url, "lastUpdate": last_update}


def _ok(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, content=RSS)


def _run(feeds, rss_dir, handler=_ok, **kwargs) -> FetchStats:
    return fetch.fetch_rss(feeds, rss_dir, transport=httpx.MockTransport(handler), **kwargs)


def _stored(rss_dir) -> pl.DataFrame:
    return pl.read_parquet(rss_store.list_parts(rss_dir))


def test_decode_declared_latin1_is_rewritten_and_parses() -> None:
    text = fetch.decode_feed(LATIN1_RSS.encode("latin-1"))
    assert 'encoding="utf-8"' in text
    for impl in ("rs", "feedparser"):
        assert parse.parse_feed(text, "u", "p", impl=impl)[0]["title"] == "Grüße"


def test_decode_uses_http_charset_without_declaration() -> None:
    body = "<rss><channel><title>Grüße</title></channel></rss>".encode("latin-1")
    assert "Grüße" in fetch.decode_feed(body, "iso-8859-1")


def test_decode_utf16_bom() -> None:
    text = fetch.decode_feed(
        '<?xml version="1.0" encoding="UTF-16"?><rss>Grüße</rss>'.encode("utf-16")
    )
    assert text == '<?xml version="1.0" encoding="utf-8"?><rss>Grüße</rss>'


def test_decode_ignores_unknown_codec() -> None:
    body = '<?xml version="1.0" encoding="no-such-codec"?><rss>Grüße</rss>'.encode()
    assert "Grüße" in fetch.decode_feed(body)


def test_decode_never_raises_on_invalid_bytes() -> None:
    assert "\ufffd" in fetch.decode_feed(b"<rss>\xff\xfe\xfa</rss>")


def test_pending_feeds_skips_stored(tmp_path) -> None:
    pl.DataFrame({"url": ["https://a", "https://b"], "fetch_date": [100, 100]}).write_parquet(
        tmp_path / "part-0000.parquet"
    )
    feeds = [
        _feed("https://a", 50),  # stored, no update -> skip
        _feed("https://b", 150),  # stored but updated since -> refetch
        _feed("https://c"),  # never stored -> fetch
        _feed("https://c"),  # duplicate url -> once
    ]
    pending = [f["url"] for f in fetch.pending_feeds(feeds, tmp_path)]
    assert sorted(pending) == ["https://b", "https://c"]


def test_fetch_rss_writes_bounded_parts(tmp_path) -> None:
    stats = _run([_feed("https://a"), _feed("https://b")], tmp_path, part_size=1)
    assert stats == FetchStats(selected=2, pending=2, ok=2, failed=0)
    assert len(list(tmp_path.glob("part-*.parquet"))) == 2  # part_size=1 -> one part per feed
    assert _stored(tmp_path)["rss"].to_list() == [RSS.decode()] * 2


def test_fetch_rss_is_idempotent_and_refetches_updated(tmp_path) -> None:
    _run([_feed("https://a")], tmp_path)
    assert _run([_feed("https://a")], tmp_path).pending == 0
    assert _run([_feed("https://a", int(time.time()) + 1_000)], tmp_path).ok == 1


@pytest.mark.parametrize(
    "handler",
    [
        lambda r: httpx.Response(404),
        lambda r: httpx.Response(500),
        lambda r: httpx.Response(200, content=b""),
        lambda r: (_ for _ in ()).throw(httpx.ConnectError("down")),
    ],
    ids=["404", "500", "empty", "connect-error"],
)
def test_fetch_rss_counts_failures(tmp_path, handler) -> None:
    stats = _run([_feed("https://a")], tmp_path, handler)
    assert stats == FetchStats(selected=1, pending=1, ok=0, failed=1)
    assert list(tmp_path.glob("part-*.parquet")) == []


def test_fetch_rss_invalid_url_is_a_failure_not_a_crash(tmp_path) -> None:
    stats = _run([_feed("http://exa mple.com:bad/"), _feed("https://ok")], tmp_path)
    assert (stats.ok, stats.failed) == (1, 1)


def test_fetch_rss_skips_oversized_body(tmp_path) -> None:
    stats = _run([_feed("https://a")], tmp_path, max_feed_bytes=len(RSS) - 1)
    assert (stats.ok, stats.failed) == (0, 1)
    assert _run([_feed("https://a")], tmp_path, max_feed_bytes=len(RSS)).ok == 1


def test_fetch_rss_decodes_with_http_charset(tmp_path) -> None:
    body = "<rss><channel><title>Grüße</title></channel></rss>".encode("latin-1")
    headers = {"content-type": "application/rss+xml; charset=iso-8859-1"}
    _run(
        [_feed("https://a")], tmp_path, lambda r: httpx.Response(200, content=body, headers=headers)
    )
    assert "Grüße" in _stored(tmp_path)["rss"][0]


def _tracking_handler(peak: dict[str, int]):
    inflight: dict[str, int] = {"all": 0}

    async def handler(request: httpx.Request) -> httpx.Response:
        host = request.url.host
        for key in ("all", host):
            inflight[key] = inflight.get(key, 0) + 1
            peak[key] = max(peak.get(key, 0), inflight[key])
        await asyncio.sleep(0.01)
        for key in ("all", host):
            inflight[key] -= 1
        return httpx.Response(200, content=RSS)

    return handler


def test_fetch_rss_bounds_total_concurrency(tmp_path) -> None:
    peak: dict[str, int] = {}
    feeds = [_feed(f"https://h{i}.example/feed") for i in range(20)]
    stats = _run(feeds, tmp_path, _tracking_handler(peak), concurrency=3)
    assert stats.ok == 20
    assert peak["all"] == 3


def test_fetch_rss_bounds_concurrency_per_host(tmp_path) -> None:
    peak: dict[str, int] = {}
    feeds = [_feed(f"https://same.example/{i}") for i in range(10)]
    stats = _run(feeds, tmp_path, _tracking_handler(peak), concurrency=10, per_host=2)
    assert stats.ok == 10
    assert peak["same.example"] == 2


def test_fetch_rss_error_stops_remaining_work_and_keeps_valid_part(tmp_path) -> None:
    requested: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(str(request.url))
        if request.url.host == "bad.example":
            raise RuntimeError("boom")
        return httpx.Response(200, content=RSS)

    feeds = [
        _feed("https://first.example/"),
        _feed("https://bad.example/"),
        *(_feed(f"https://h{i}.example/") for i in range(50)),
    ]
    with pytest.raises(RuntimeError, match="boom"):
        _run(feeds, tmp_path, handler, concurrency=1)
    assert len(requested) < 5  # the other 50 queued feeds were cancelled, not fetched
    assert list(tmp_path.glob("*.tmp")) == []
    assert _stored(tmp_path).height == 1  # the row written before the error is still valid
