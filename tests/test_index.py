"""Tests for pepsearch.core.index helpers that don't need the model or data files."""

import pytest

from pepsearch.core import index


def test_cosine_basic() -> None:
    assert index._cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert index._cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert index._cosine([1.0, 1.0], [1.0, 0.0]) == pytest.approx(2**-0.5)


def test_cosine_zero_vector() -> None:
    assert index._cosine([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_index_roundtrip(tmp_path) -> None:
    idx = index.Index(
        pep_numbers=[8, 20],
        titles=["Style Guide", "Zen"],
        chunks=["chunk a", "chunk b"],
        vectors=[[1.0, 0.0], [0.0, 1.0]],
    )
    path = tmp_path / "index.json"
    index.save(idx, path)
    loaded = index.load(path)
    assert loaded == idx
