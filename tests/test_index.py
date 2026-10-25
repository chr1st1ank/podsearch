"""Tests for podsearch.core.index helpers that don't need the real model."""

import numpy as np
import pytest

from podsearch.core import corpus, index


class FakeModel:
    """Encodes every text as the same fixed vector - no model download."""

    def __init__(self, vec: list[float]) -> None:
        self.vec = vec

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.tile(np.asarray(self.vec, dtype=np.float32), (len(texts), 1))


def test_cosine_basic() -> None:
    assert index._cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert index._cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert index._cosine([1.0, 1.0], [1.0, 0.0]) == pytest.approx(2**-0.5)


def test_cosine_zero_vector() -> None:
    assert index._cosine([0.0, 0.0], [1.0, 0.0]) == 0.0


def _tiny_corpus(db_path):
    conn = corpus.open_corpus(db_path)
    conn.executemany(
        "INSERT INTO episodes(feed_url, guid, podcast, title, link, pubdate, text)"
        " VALUES (?,?,?,?,?,?,?)",
        [
            ("u1", "g1", "Pod A", "Episode one", "l1", 0, "alpha beta gamma"),
            ("u1", "g2", "Pod B", "Episode two", "l2", 0, "delta epsilon"),
        ],
    )
    conn.commit()
    return conn


def test_build_and_search_roundtrip(tmp_path) -> None:
    db = tmp_path / "corpus.db"
    vectors = tmp_path / "vectors.f32"
    conn = _tiny_corpus(db)
    n = index.build_index(conn, FakeModel([1.0, 0.0]), vectors)
    assert n == 2
    conn.close()

    idx = index.load(vectors, db)
    hits = index.search_numpy(idx, FakeModel([1.0, 0.0]), "q")
    assert len(hits) == 2
    assert hits[0].score == pytest.approx(1.0)
    assert {h.podcast for h in hits} == {"Pod A", "Pod B"}

    hits = index.search(idx, FakeModel([1.0, 0.0]), "q", k=1)
    assert len(hits) == 1
    assert hits[0].episode in {"Episode one", "Episode two"}
    idx.conn.close()
