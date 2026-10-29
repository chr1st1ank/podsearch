"""Tests for podsearch.core.index helpers that don't need the real model."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from podsearch.core import corpus, index
from tests.test_lance import FakeReranker


class FakeModel:
    """Encodes every text as the same fixed vector - no model download."""

    def __init__(self, vec: list[float]) -> None:
        self.vec = vec

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.tile(np.asarray(self.vec, dtype=np.float32), (len(texts), 1))


class KeywordReranker:
    """Scores (query, passage) pairs by whether `keyword` appears in the passage."""

    def __init__(self, keyword: str) -> None:
        self.keyword = keyword
        self.last_pairs: list[list[str]] = []

    def predict(self, pairs: list[list[str]]) -> np.ndarray:
        self.last_pairs = pairs
        return np.asarray(
            [1.0 if self.keyword in text else 0.0 for _q, text in pairs], dtype=np.float32
        )


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
    hits = index.search_numpy(idx, FakeModel([1.0, 0.0]), FakeReranker(), "q")
    assert len(hits) == 2
    assert hits[0].score == pytest.approx(1.0)
    assert {h.podcast for h in hits} == {"Pod A", "Pod B"}

    hits = index.search(idx, FakeModel([1.0, 0.0]), FakeReranker(), "q", k=1)
    assert len(hits) == 1
    assert hits[0].episode in {"Episode one", "Episode two"}
    idx.conn.close()


def test_search_dedups_chunks_of_one_episode(tmp_path) -> None:
    db = tmp_path / "corpus.db"
    vectors = tmp_path / "vectors.f32"
    conn = _tiny_corpus(db)
    text = " ".join(["word"] * 250)
    conn.executemany(
        "INSERT INTO episodes(feed_url, guid, podcast, title, link, pubdate, text)"
        " VALUES (?,?,?,?,?,?,?)",
        [
            ("u2", "g3", "Pod C", "Episode three", "l3", 100, text),
            # same episode republished on a sibling feed -> text dedup
            ("u3", "g4", "Pod C HD", "Episode three", "l4", 100, text),
            # same title on another day is a different episode -> stays
            ("u4", "g5", "Pod D", "Episode three", "l5", 200, "delta epsilon"),
        ],
    )
    conn.commit()
    n = index.build_index(conn, FakeModel([1.0, 0.0]), vectors)
    assert n == 7
    conn.close()

    idx = index.load(vectors, db)
    for fn in (index.search, index.search_numpy):
        hits = fn(idx, FakeModel([1.0, 0.0]), FakeReranker(), "q", k=5)
        assert [h.episode for h in hits].count("Episode three") == 2
        kept = {(h.podcast, h.pubdate) for h in hits if h.episode == "Episode three"}
        assert ("Pod D", 200) in kept
        # one of the two republished copies survives - which one is arbitrary on ties
        assert kept - {("Pod D", 200)} <= {("Pod C", 100), ("Pod C HD", 100)}
    idx.conn.close()


class ZeroVecModel:
    """Zero vector for texts containing 'zword' - simulates all-OOV chunks."""

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            [[0.0, 0.0] if "zword" in t else [1.0, 0.0] for t in texts],
            dtype=np.float32,
        )


def test_search_numpy_zero_norm_rows(tmp_path) -> None:
    db = tmp_path / "corpus.db"
    vectors = tmp_path / "vectors.f32"
    conn = _tiny_corpus(db)
    conn.execute(
        "INSERT INTO episodes(feed_url, guid, podcast, title, link, pubdate, text)"
        " VALUES ('u2', 'g3', 'Pod C', 'Episode three', 'l3', 0, 'zword')"
    )
    conn.commit()
    index.build_index(conn, ZeroVecModel(), vectors)
    conn.close()

    idx = index.load(vectors, db)
    hits = index.search_numpy(idx, ZeroVecModel(), FakeReranker(), "q", k=3)
    assert all(h.score == h.score for h in hits)  # no NaN scores
    assert hits[-1].podcast == "Pod C"  # worst stage-1 candidate stays last
    assert hits[-1].score == pytest.approx(1.0)  # FakeReranker score replaces it
    idx.conn.close()


def test_rerank_reorders_candidates(tmp_path) -> None:
    """A lower-ranked stage-1 hit wins when the reranker scores it higher."""
    db = tmp_path / "corpus.db"
    vectors = tmp_path / "vectors.f32"
    conn = _tiny_corpus(db)  # Pod A "alpha beta gamma", Pod B "delta epsilon"
    index.build_index(conn, FakeModel([1.0, 0.0]), vectors)
    conn.close()

    idx = index.load(vectors, db)
    for fn in (index.search, index.search_numpy):
        hits = fn(idx, FakeModel([1.0, 0.0]), KeywordReranker("epsilon"), "q", k=1)
        assert [h.podcast for h in hits] == ["Pod B"]
        assert hits[0].score == pytest.approx(1.0)
    idx.conn.close()


def test_rerank_passage_formatting_and_snippet_truncate(tmp_path) -> None:
    """Passages get podcast+episode context; snippets end up <= 200 chars."""
    db = tmp_path / "corpus.db"
    vectors = tmp_path / "vectors.f32"
    conn = corpus.open_corpus(db)
    # real episodes store title + "\n\n" + body: chunk 1 repeats the title
    conn.execute(
        "INSERT INTO episodes(feed_url, guid, podcast, title, link, pubdate, text)"
        " VALUES ('u1', 'g1', 'Pod A', 'Episode one', 'l1', 0, ?)",
        ("Episode one\n\n" + " ".join(["word"] * 300),),
    )
    conn.commit()
    index.build_index(conn, FakeModel([1.0, 0.0]), vectors)
    conn.close()

    idx = index.load(vectors, db)
    reranker = KeywordReranker("word")
    hits = index.search_numpy(idx, FakeModel([1.0, 0.0]), reranker, "q", k=5)
    assert hits and all(len(h.snippet) <= 200 for h in hits)

    # every candidate passage is "<podcast> — <episode>\n\n<chunk text>"
    first = reranker.last_pairs[0][1]
    assert first.startswith("Pod A — Episode one\n\n")
    # chunk 1's leading title is not duplicated in the passage
    assert "Episode one\n\nEpisode one" not in first
    idx.conn.close()


def test_search_from_worker_thread(tmp_path) -> None:
    """FastAPI runs sync endpoints in a threadpool: load on main, search elsewhere."""
    db = tmp_path / "corpus.db"
    vectors = tmp_path / "vectors.f32"
    conn = _tiny_corpus(db)
    index.build_index(conn, FakeModel([1.0, 0.0]), vectors)
    conn.close()

    idx = index.load(vectors, db)
    model = FakeModel([1.0, 0.0])
    with ThreadPoolExecutor(1) as pool:
        for fn in (index.search, index.search_numpy):
            hits = pool.submit(fn, idx, model, FakeReranker(), "q").result()
            assert len(hits) == 2
    idx.conn.close()
