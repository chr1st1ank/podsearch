"""Embedding index: memmap'd float32 vectors + cosine similarity search."""

import math
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from podsearch.core import corpus
from podsearch.core.paths import CORPUS_DB, VECTORS_PATH

if TYPE_CHECKING:
    import numpy as np
    from model2vec import StaticModel
    from sentence_transformers import CrossEncoder

# ADR-0001: static model2vec embeddings + required language-scoped search.
# See: docs/decisions/0001-static-embeddings-language-scoped-search.md
MODEL_NAME = "minishlab/potion-multilingual-128M"
RERANK_MODEL_NAME = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
MAX_CHUNKS_PER_EPISODE = 3
ENCODE_BATCH = 512
# Hits repeat an episode for several reasons (chunks per episode, republished
# feeds). Fetch extra candidates so dedup_hits can still return k episodes.
SEARCH_OVERFETCH = 6
MIN_RERANK_CANDIDATES = 30


def rerank_limit(k: int) -> int:
    """Number of chunks to retrieve before reranking + deduping to k episodes."""
    return max(k * SEARCH_OVERFETCH, MIN_RERANK_CANDIDATES)


def load_model() -> StaticModel:
    from model2vec import StaticModel

    return StaticModel.from_pretrained(MODEL_NAME)


def load_reranker() -> CrossEncoder:
    """Load the cross-encoder reranker (sigmoid-normalized scores in [0, 1])."""
    import torch
    from sentence_transformers import CrossEncoder

    return CrossEncoder(RERANK_MODEL_NAME, activation_fn=torch.nn.Sigmoid())


@dataclass
class Index:
    vectors: np.ndarray  # memmap'd (n, dim) float32
    conn: sqlite3.Connection
    _norm: np.ndarray | None = None
    _lang_rows: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class Hit:
    podcast: str
    episode: str
    score: float
    snippet: str
    link: str
    pubdate: int
    feed_url: str
    guid: str
    language: str


def dedup_hits(hits: list[Hit], k: int) -> list[Hit]:
    """Keep the best-scoring hit per episode; input must arrive score-ordered.

    (feed_url, guid) drops repeated chunks of one episode; (title, pubdate)
    additionally drops the same episode republished on sibling feeds (media
    variants, syndication). Generic titles are exempt from the text key.
    """
    seen_ids: set[tuple[str, str]] = set()
    seen_text: set[tuple[str, int]] = set()
    out = []
    for h in hits:
        dup = (h.feed_url, h.guid) in seen_ids or (
            h.episode not in ("", "(untitled)") and (h.episode, h.pubdate) in seen_text
        )
        if dup:
            continue
        seen_ids.add((h.feed_url, h.guid))
        seen_text.add((h.episode, h.pubdate))
        out.append(h)
        if len(out) >= k:
            break
    return out


def build_index(
    conn: sqlite3.Connection,
    model,
    vectors_path: Path = VECTORS_PATH,
    max_chunks: int = MAX_CHUNKS_PER_EPISODE,
    batch: int = ENCODE_BATCH,
) -> int:
    """Stream episodes -> chunks -> embeddings; returns the vector count."""
    import numpy as np

    vectors_path.parent.mkdir(parents=True, exist_ok=True)
    texts: list[str] = []
    langs: list[str] = []
    episode_ids: list[int] = []
    n = 0
    dim = 0

    def flush(f) -> None:
        nonlocal n, dim
        if not texts:
            return
        vecs = np.asarray(model.encode(texts), dtype=np.float32)
        dim = vecs.shape[1]
        f.write(vecs.tobytes())
        conn.executemany(
            "INSERT INTO chunks(vector_row, episode_id, text, language) VALUES (?,?,?,?)",
            zip(range(n, n + len(texts)), episode_ids, texts, langs, strict=True),
        )
        n += len(texts)
        texts.clear()
        langs.clear()
        episode_ids.clear()

    conn.execute("DELETE FROM chunks")
    with vectors_path.open("wb") as f:
        for ep_id, text, language in corpus.iter_episodes(conn):
            for c in dict.fromkeys(corpus.chunk(text)[:max_chunks]):
                texts.append(c)
                langs.append(language)
                episode_ids.append(ep_id)
                if len(texts) >= batch:
                    flush(f)
        flush(f)
    conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES ('dim', ?)", (str(dim),))
    conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES ('n', ?)", (str(n),))
    conn.commit()
    return n


def load(vectors_path: Path = VECTORS_PATH, db_path: Path = CORPUS_DB) -> Index:
    import numpy as np

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, check_same_thread=False)
    meta = dict(conn.execute("SELECT key, value FROM meta"))
    dim, n = int(meta["dim"]), int(meta["n"])
    vectors = np.memmap(vectors_path, dtype=np.float32, mode="r").reshape(n, dim)
    return Index(vectors, conn)


def _cosine(query: list[float], vec) -> float:
    dot = sum(q * v for q, v in zip(query, vec, strict=False))
    nq = math.sqrt(sum(q * q for q in query))
    nv = math.sqrt(sum(v * v for v in vec))
    return float(dot / (nq * nv)) if nq and nv else 0.0


def _lang_rows(index: Index, lang: str) -> np.ndarray:
    """Vector rows belonging to one language; cached per language like _norm."""
    import numpy as np

    rows = index._lang_rows.get(lang)
    if rows is None:
        rows = np.asarray(
            [
                r[0]
                for r in index.conn.execute(
                    "SELECT vector_row FROM chunks WHERE language = ? ORDER BY vector_row",
                    (lang,),
                )
            ],
            dtype=np.int64,
        )
        index._lang_rows[lang] = rows
    return rows


def _hydrate(index: Index, scored: list[tuple[float, int]]) -> list[Hit]:
    """Fetch podcast/episode/snippet metadata for the top-k vector rows."""
    vector_rows = [i for _, i in scored]
    placeholders = ",".join("?" * len(vector_rows))
    rows = {
        r[0]: r
        for r in index.conn.execute(
            "SELECT c.vector_row, c.text, e.podcast, e.title, e.link, e.pubdate,"
            " e.feed_url, e.guid, c.language"
            " FROM chunks c JOIN episodes e ON e.id = c.episode_id"
            f" WHERE c.vector_row IN ({placeholders})",
            vector_rows,
        )
    }
    return [
        Hit(
            podcast=rows[i][2],
            episode=rows[i][3],
            score=float(s),
            snippet=rows[i][1],  # full chunk text; rerank() truncates to [:200]
            link=rows[i][4],
            pubdate=rows[i][5],
            feed_url=rows[i][6],
            guid=rows[i][7],
            language=rows[i][8],
        )
        for s, i in scored
        if i in rows
    ]


def rerank(reranker, query: str, hits: list[Hit], k: int) -> list[Hit]:
    """Score stage-1 candidates with the cross-encoder, dedupe, and keep top k.

    Each passage includes podcast + episode title + full chunk text so chunks 2+
    (which start mid-episode, without the title) still get episode context.

    With reranker=None the stage-1 scores (cosine similarity) are kept as-is.
    """
    if not hits:
        return []
    if reranker is None:
        for h in hits:
            h.snippet = h.snippet[:200]
        hits.sort(key=lambda h: h.score, reverse=True)
        return dedup_hits(hits, k)
    pairs = []
    for h in hits:
        text = h.snippet
        # chunk 1 of an episode repeats the title; don't send it twice
        if text == h.episode or text.startswith(h.episode + " "):
            text = text[len(h.episode) :].lstrip()
        pairs.append([query, f"{h.podcast} — {h.episode}\n\n{text}"])
    scores = reranker.predict(pairs)
    for h, s in zip(hits, scores, strict=True):
        h.score = float(s)
        h.snippet = h.snippet[:200]
    hits.sort(key=lambda h: h.score, reverse=True)
    return dedup_hits(hits, k)


def search(index: Index, model, reranker, query: str, lang: str, k: int = 5) -> list[Hit]:
    """Score every same-language chunk with cosine similarity; return best k."""
    rows = _lang_rows(index, lang)
    if not len(rows):
        return []
    q = model.encode([query])[0].tolist()
    scored = [(_cosine(q, index.vectors[i]), int(i)) for i in rows]
    scored.sort(reverse=True)
    candidates = _hydrate(index, scored[: rerank_limit(k)])
    return rerank(reranker, query, candidates, k)


def search_numpy(index: Index, model, reranker, query: str, lang: str, k: int = 5) -> list[Hit]:
    """Vectorized version of search(): one matrix product over the language shard."""
    import numpy as np

    rows = _lang_rows(index, lang)
    if not len(rows):
        return []
    matrix = index._norm
    if matrix is None:
        m = index.vectors
        norms = np.linalg.norm(m, axis=1, keepdims=True)
        matrix = np.divide(m, norms, out=np.zeros(m.shape, dtype=np.float32), where=norms > 0)
        index._norm = matrix
    q = model.encode([query])[0].astype(np.float32)
    nq = np.linalg.norm(q)
    if nq:
        q /= nq
    scores = matrix[rows] @ q
    limit = min(rerank_limit(k), len(scores))
    top = np.argpartition(scores, -limit)[-limit:]
    top = top[np.argsort(scores[top])[::-1]]
    candidates = _hydrate(index, [(float(scores[i]), int(rows[i])) for i in top])
    return rerank(reranker, query, candidates, k)
