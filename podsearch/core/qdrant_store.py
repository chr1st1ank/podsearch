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

from podsearch.core import index, ingest_common
from podsearch.core.index import ENCODE_BATCH, MAX_CHUNKS_PER_EPISODE, Hit
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
        done.update(p.payload["path"] for p in points if p.payload is not None)
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
    parser_impl: str = "rs",
    max_chunks: int = MAX_CHUNKS_PER_EPISODE,
    rebuild: bool = False,
    max_parts: int | None = None,
) -> dict[str, int]:
    """Parse not-yet-ingested parts straight into the qdrant chunks collection."""
    client = client or connect()
    if rebuild:
        for name in (CHUNKS_COLLECTION, PARTS_COLLECTION):
            client.delete_collection(name)
    _ensure_parts(client)
    done = _done_parts(client)
    parts = [p for p in sorted(rss_dir.glob("part-*.parquet")) if p.name not in done][:max_parts]
    stats = {"parts": 0, "chunks": 0, "skipped": len(done), "stale": 0}
    if not parts:
        return stats

    dim = len(model.encode(["dimension probe"])[0])
    _ensure_collections(client, dim)
    latest = ingest_common.latest_fetch_dates(rss_dir)
    parse = partial(ingest_common.parse_feed, max_episodes=max_per_feed, impl=parser_impl)
    for px, part in enumerate(parts):
        t0 = time.perf_counter()
        raw = pl.read_parquet(part, columns=["url", "title", "rss", "fetch_date"])
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
            for feed in feeds.select("url", "title", "rss").iter_rows(named=True)
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
    return stats


def search(client: QdrantClient, model, q: str, k: int = 5) -> list[Hit]:
    """Vector search; payload carries all metadata so hits need no hydration."""
    vec = [float(x) for x in model.encode([q])[0]]
    res = client.query_points(
        CHUNKS_COLLECTION, query=vec, limit=k * index.SEARCH_OVERFETCH, with_payload=True
    )
    return index.dedup_hits(
        [
            Hit(
                podcast=p.payload["podcast"],
                episode=p.payload["episode"],
                score=p.score,
                snippet=p.payload["text"][:200],
                link=p.payload["link"],
                pubdate=p.payload["pubdate"],
                feed_url=p.payload["feed_url"],
                guid=p.payload["guid"],
            )
            for p in res.points
            if p.payload is not None
        ],
        k,
    )
