"""Tests for podsearch.core.qdrant_store using synthetic parquet parts and in-memory qdrant."""

import pytest
from qdrant_client import QdrantClient

from podsearch.core import qdrant_store
from tests.test_lance import FakeModel, FakeReranker, _write_part


@pytest.fixture
def env(tmp_path):
    rss_dir = tmp_path / "rss"
    rss_dir.mkdir()
    return rss_dir, QdrantClient(":memory:")


def test_ingest_and_search_roundtrip(env) -> None:
    rss_dir, client = env
    _write_part(rss_dir, 0, "u1", "Episode one", 1000)
    stats = qdrant_store.ingest_parts(FakeModel(), rss_dir=rss_dir, client=client)
    assert stats["parts"] == 1 and stats["chunks"] > 0

    hits = qdrant_store.search(client, FakeModel(), FakeReranker(), "query", k=5)
    assert len(hits) == 1
    assert hits[0].score == pytest.approx(1.0)
    assert (hits[0].podcast, hits[0].episode) == ("Pod A", "Episode one")
    assert hits[0].pubdate > 0


def test_search_dedups_chunks_of_one_episode(env) -> None:
    rss_dir, client = env
    _write_part(rss_dir, 0, "u1", "Episode one", 1000, desc=" ".join(["word"] * 250))
    stats = qdrant_store.ingest_parts(FakeModel(), rss_dir=rss_dir, client=client)
    assert stats["chunks"] == 2

    hits = qdrant_store.search(client, FakeModel(), FakeReranker(), "query", k=5)
    assert [h.episode for h in hits] == ["Episode one"]


def test_ingest_idempotent_and_replace(env) -> None:
    rss_dir, client = env
    _write_part(rss_dir, 0, "u1", "Episode one", 1000)
    qdrant_store.ingest_parts(FakeModel(), rss_dir=rss_dir, client=client)
    stats = qdrant_store.ingest_parts(FakeModel(), rss_dir=rss_dir, client=client)
    assert stats["parts"] == 0 and stats["skipped"] == 1

    _write_part(rss_dir, 1, "u1", "Episode one v2", 2000)
    stats = qdrant_store.ingest_parts(FakeModel(), rss_dir=rss_dir, client=client)
    assert stats["parts"] == 1
    hits = qdrant_store.search(client, FakeModel(), FakeReranker(), "query", k=5)
    assert [h.episode for h in hits] == ["Episode one v2"]


def test_rebuild_reingests_everything(env) -> None:
    rss_dir, client = env
    _write_part(rss_dir, 0, "u1", "Episode one", 1000)
    qdrant_store.ingest_parts(FakeModel(), rss_dir=rss_dir, client=client)
    stats = qdrant_store.ingest_parts(FakeModel(), rss_dir=rss_dir, client=client, rebuild=True)
    assert stats["parts"] == 1
    assert len(qdrant_store.search(client, FakeModel(), FakeReranker(), "query")) == 1
