"""Qdrant store: fused ingest+index with vectors and metadata in one collection.

Mirrors lance_store: the qdrant pipeline (--impl qdrant) writes one denormalized
`chunks` collection - vector plus all podcast/episode columns as payload - so
search needs no hydration step. Updates are delete-then-upsert per feed, and
ingest is incremental via our own `ingested_parts` collection. Point ids are
deterministic, so re-ingesting the same chunk overwrites instead of duplicating.

HNSW indexing is deferred during bulk load (indexing_threshold=0) and switched
on once at the end, so the graph is built after the final row count is known.
"""

import time
import uuid
from functools import partial
from pathlib import Path

import polars as pl
from qdrant_client import QdrantClient, models

from podsearch.core import index, ingest_common, rss_store
from podsearch.core.index import ENCODE_BATCH, MAX_CHUNKS_PER_EPISODE, Hit
from podsearch.core.parse import Parser
from podsearch.core.paths import QDRANT_URL, RSS_DIR

CHUNKS_COLLECTION = "chunks"
PARTS_COLLECTION = "ingested_parts"
UPSERT_BATCH = 1_000
INDEXING_THRESHOLD_KB = 20_000  # qdrant's default; restored after bulk load
_NS = uuid.NAMESPACE_URL


def connect(url: str = QDRANT_URL) -> QdrantClient:
    return QdrantClient(url=url, prefer_grpc=False, timeout=120)


def open_chunks(url: str = QDRANT_URL) -> QdrantClient:
    """Connect for searching; exits helpfully if server or collection is missing."""
    client = connect(url)
    try:
        exists = client.collection_exists(CHUNKS_COLLECTION)
    except Exception as e:
        raise SystemExit(f"cannot reach qdrant at {url} ({e}) - run: task qdrant-up") from e
    if not exists:
        raise SystemExit(f"no qdrant collection at {url} - run: podsearch-ingest --impl qdrant")
    return client


def _ensure_collections(client: QdrantClient, dim: int) -> None:
    if not client.collection_exists(CHUNKS_COLLECTION):
        client.create_collection(
            CHUNKS_COLLECTION,
            vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
            optimizers_config=models.OptimizersConfigDiff(indexing_threshold=0),
        )
        client.create_payload_index(CHUNKS_COLLECTION, "feed_url", models.PayloadSchemaType.KEYWORD)
        client.create_payload_index(CHUNKS_COLLECTION, "guid", models.PayloadSchemaType.KEYWORD)
        client.create_payload_index(CHUNKS_COLLECTION, "language", models.PayloadSchemaType.KEYWORD)


def _stats_id() -> str:
    return str(uuid.uuid5(_NS, "__stats__"))


def refresh_stats(client: QdrantClient) -> int:
    """Count distinct guids (expensive facet) and store it next to the chunk count.

    The facet enumerates every distinct guid, so this runs at ingest time only;
    readers use episode_count() which just compares the stored snapshot.
    """
    n = client.count(CHUNKS_COLLECTION).count
    episodes = len(client.facet(CHUNKS_COLLECTION, key="guid", limit=max(n, 1)).hits)
    client.upsert(
        PARTS_COLLECTION,
        [
            models.PointStruct(
                id=_stats_id(), vector={}, payload={"points_count": n, "episodes": episodes}
            )
        ],
        wait=True,
    )
    return episodes


def episode_count(client: QdrantClient) -> int | None:
    """Stored distinct-episode count, or None if missing or stale (chunk count moved)."""
    if not client.collection_exists(PARTS_COLLECTION):
        return None
    recs = client.retrieve(PARTS_COLLECTION, [_stats_id()], with_payload=True)
    if not recs or recs[0].payload is None:
        return None
    p = recs[0].payload
    if p["points_count"] != client.count(CHUNKS_COLLECTION).count:
        return None
    return int(p["episodes"])


def _ensure_parts(client: QdrantClient) -> None:
    if not client.collection_exists(PARTS_COLLECTION):
        client.create_collection(PARTS_COLLECTION, vectors_config={})


def _done_parts(client: QdrantClient) -> set[str]:
    done: set[str] = set()
    offset = None
    while True:
        points, offset = client.scroll(
            PARTS_COLLECTION, limit=1_000, offset=offset, with_payload=["path"]
        )
        done.update(
            p.payload["path"] for p in points if p.payload is not None and "path" in p.payload
        )
        if offset is None:
            return done


def _points(chunks: pl.DataFrame) -> list[models.PointStruct]:
    return [
        models.PointStruct(
            id=str(uuid.uuid5(_NS, f"{r['feed_url']}|{r['guid']}|{i}|{r['text'][:64]}")),
            vector=list(r.pop("vector")),
            payload=r,
        )
        for i, r in enumerate(chunks.iter_rows(named=True))
    ]


def ingest_parts(
    model,
    rss_dir: Path = RSS_DIR,
    client: QdrantClient | None = None,
    max_per_feed: int = 300,
    parser_impl: Parser = "rs",
    max_chunks: int = MAX_CHUNKS_PER_EPISODE,
    rebuild: bool = False,
    max_parts: int | None = None,
    langs: tuple[str, ...] | None = None,
) -> dict[str, int]:
    """Parse not-yet-ingested parts straight into the qdrant chunks collection."""
    client = client or connect()
    if rebuild:
        for name in (CHUNKS_COLLECTION, PARTS_COLLECTION):
            client.delete_collection(name)
    _ensure_parts(client)
    done = _done_parts(client)
    parts = [p for p in rss_store.list_parts(rss_dir) if p.name not in done][:max_parts]
    stats = {"parts": 0, "chunks": 0, "skipped": len(done), "stale": 0}
    if not parts:
        if client.collection_exists(CHUNKS_COLLECTION) and episode_count(client) is None:
            refresh_stats(client)
        return stats

    dim = len(model.encode(["dimension probe"])[0])
    _ensure_collections(client, dim)
    latest = rss_store.latest_fetch_dates(rss_dir)
    parse = partial(
        ingest_common.parse_feed, max_episodes=max_per_feed, impl=parser_impl, langs=langs
    )
    for px, part in enumerate(parts):
        t0 = time.perf_counter()
        # parts written before ADR-0001 lack the language column -> 'und'
        cols = ["url", "title", "rss", "fetch_date"]
        if "language" in pl.read_parquet_schema(part):
            cols.append("language")
        raw = pl.read_parquet(part, columns=cols)
        if "language" not in raw.columns:
            raw = raw.with_columns(pl.lit(None).cast(pl.String).alias("language"))
        feeds = raw.join(latest, on="url").filter(
            pl.col("rss").is_not_null() & (pl.col("fetch_date") >= pl.col("latest"))
        )
        stats["stale"] += raw.height - feeds.height
        print(
            f"part {px + 1}/{len(parts)} {part.name}: "
            f"{feeds.height:,} feeds ({raw.height - feeds.height:,} stale)"
        )
        episodes = [
            ep
            for feed in feeds.select("url", "title", "rss", "language").iter_rows(named=True)
            for ep in parse(feed)
        ]
        chunks = ingest_common.chunk_episodes(episodes, max_chunks)

        if feeds.height:
            client.delete(
                CHUNKS_COLLECTION,
                models.FilterSelector(
                    filter=models.Filter(
                        must=models.FieldCondition(
                            key="feed_url", match=models.MatchAny(any=feeds["url"].to_list())
                        )
                    )
                ),
            )
        for batch in chunks.iter_slices(ENCODE_BATCH):
            points = _points(ingest_common.embed(model, batch))
            for i in range(0, len(points), UPSERT_BATCH):
                client.upsert(CHUNKS_COLLECTION, points[i : i + UPSERT_BATCH], wait=True)
        print(
            f"  part done in {time.perf_counter() - t0:.1f}s: "
            f"{len(episodes):,} episodes, {chunks.height:,} chunks"
        )
        client.upsert(
            PARTS_COLLECTION,
            [
                models.PointStruct(
                    id=str(uuid.uuid5(_NS, part.name)),
                    vector={},
                    payload={"path": part.name, "ingested_at": int(time.time())},
                )
            ],
            wait=True,
        )
        stats["parts"] += 1
        stats["chunks"] += chunks.height

    print("enabling HNSW indexing (server-side build)")
    client.update_collection(
        CHUNKS_COLLECTION,
        optimizers_config=models.OptimizersConfigDiff(indexing_threshold=INDEXING_THRESHOLD_KB),
    )
    refresh_stats(client)
    return stats


def search(client: QdrantClient, model, reranker, q: str, lang: str, k: int = 5) -> list[Hit]:
    """Vector search scoped to one language; payload carries all metadata."""
    vec = [float(x) for x in model.encode([q])[0]]
    res = client.query_points(
        CHUNKS_COLLECTION,
        query=vec,
        limit=index.rerank_limit(k),
        query_filter=models.Filter(
            must=[models.FieldCondition(key="language", match=models.MatchValue(value=lang))]
        ),
        with_payload=True,
    )
    candidates = [
        Hit(
            podcast=p.payload["podcast"],
            episode=p.payload["episode"],
            score=p.score,
            snippet=p.payload["text"],
            link=p.payload["link"],
            pubdate=p.payload["pubdate"],
            feed_url=p.payload["feed_url"],
            guid=p.payload["guid"],
            language=p.payload.get("language", ""),
        )
        for p in res.points
        if p.payload is not None
    ]
    return index.rerank(reranker, q, candidates, k)
