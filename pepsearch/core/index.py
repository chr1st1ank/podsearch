"""Embedding index + pure-python cosine similarity search."""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from pepsearch.core.corpus import DATA_DIR, chunk, clean

if TYPE_CHECKING:
    import numpy as np
    from model2vec import StaticModel

MODEL_NAME = "minishlab/potion-base-8M"
INDEX_PATH = DATA_DIR / "index.json"


def load_model() -> StaticModel:
    from model2vec import StaticModel

    return StaticModel.from_pretrained(MODEL_NAME)


@dataclass
class Index:
    pep_numbers: list[int]
    titles: list[str]
    chunks: list[str]
    vectors: list[list[float]]
    _matrix: np.ndarray | None = None


def build_index(peps, model) -> Index:
    pep_numbers, titles, chunks, vectors = [], [], [], []
    for pep in peps:
        for text in chunk(clean(pep.text)):
            vec = model.encode([text])[0].tolist()
            pep_numbers.append(pep.number)
            titles.append(pep.title)
            chunks.append(text)
            vectors.append(vec)
    return Index(pep_numbers, titles, chunks, vectors)


def save(index: Index, path: Path = INDEX_PATH) -> None:
    with path.open("w") as f:
        json.dump(
            {
                "pep_numbers": index.pep_numbers,
                "titles": index.titles,
                "chunks": index.chunks,
                "vectors": index.vectors,
            },
            f,
        )


def load(path: Path = INDEX_PATH) -> Index:
    data = json.loads(path.read_text())
    return Index(data["pep_numbers"], data["titles"], data["chunks"], data["vectors"])


@dataclass
class Hit:
    pep: int
    title: str
    score: float
    snippet: str


def _cosine(query: list[float], vec: list[float]) -> float:
    dot = sum(q * v for q, v in zip(query, vec, strict=False))
    nq = math.sqrt(sum(q * q for q in query))
    nv = math.sqrt(sum(v * v for v in vec))
    return dot / (nq * nv) if nq and nv else 0.0


def search(index: Index, model, query: str, k: int = 5) -> list[Hit]:
    """Score every chunk with cosine similarity and return the best k."""
    q = model.encode([query])[0].tolist()
    scored = [(_cosine(q, v), i) for i, v in enumerate(index.vectors)]
    scored.sort(reverse=True)
    return [
        Hit(index.pep_numbers[i], index.titles[i], s, index.chunks[i][:200]) for s, i in scored[:k]
    ]


def search_numpy(index: Index, model, query: str, k: int = 5) -> list[Hit]:
    """Vectorized version of search(): whole index as one matrix product."""
    import numpy as np

    matrix = index._matrix
    if matrix is None:
        m = np.asarray(index.vectors, dtype=np.float32)
        matrix = m / np.linalg.norm(m, axis=1, keepdims=True)
        index._matrix = matrix
    q = model.encode([query])[0].astype(np.float32)
    q /= np.linalg.norm(q)
    scores = matrix @ q
    top = np.argpartition(scores, -k)[-k:]
    top = top[np.argsort(scores[top])[::-1]]
    return [
        Hit(
            index.pep_numbers[i],
            index.titles[i],
            float(scores[i]),
            index.chunks[i][:200],
        )
        for i in top
    ]
