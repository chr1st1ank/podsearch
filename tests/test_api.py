"""Tests for podsearch.api that don't need the model or stores.

TestClient without a `with` block skips the lifespan, so app.state.store/model
are never touched - only routes that don't need them are covered here.
"""

from fastapi.testclient import TestClient

from podsearch import api


def test_index_page_served() -> None:
    client = TestClient(api.create_app("naive"))
    r = client.get("/")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert "podsearch" in r.text


def test_no_rerank_skips_reranker_load(monkeypatch) -> None:
    def boom() -> None:
        raise AssertionError("reranker must not load")

    monkeypatch.setattr(api.common, "open_store", lambda impl: object())
    monkeypatch.setattr(api.index, "load_model", lambda: object())
    monkeypatch.setattr(api.index, "load_reranker", boom)
    monkeypatch.setattr(api.common, "stats", lambda impl, store: {"episodes": 1, "chunks": 2})
    with TestClient(api.create_app("naive", rerank=False)) as client:
        assert client.get("/api/stats").json()["rerank"] is False


def test_stats_endpoint() -> None:
    app = api.create_app("naive")
    app.state.info = {"impl": "naive", "episodes": 12, "chunks": 34}
    client = TestClient(app)
    r = client.get("/api/stats")
    assert r.status_code == 200
    assert r.json() == {"impl": "naive", "episodes": 12, "chunks": 34}
