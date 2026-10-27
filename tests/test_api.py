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


def test_stats_endpoint() -> None:
    app = api.create_app("naive")
    app.state.info = {"impl": "naive", "episodes": 12, "chunks": 34}
    client = TestClient(app)
    r = client.get("/api/stats")
    assert r.status_code == 200
    assert r.json() == {"impl": "naive", "episodes": 12, "chunks": 34}
