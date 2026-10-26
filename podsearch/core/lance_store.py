"""LanceDB store: fused ingest+index with vectors and metadata in one table.

The lancedb pipeline (--impl lancedb) writes one denormalized `chunks` table -
vector plus all podcast/episode columns - so search needs no hydration step.
Updates are delete-then-add per feed: each parquet part carries the newest
complete version of its feeds. Incremental via our own `ingested_parts` table.

Index upkeep follows LanceDB's batch-ingest guidance: create each index once,
then let `optimize()` compact fragments and fold new rows into the indexes.
"""

import math
import time
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import polars as pl
import pyarrow as pa

from podsearch.core import index, ingest_common
from podsearch.core.index import ENCODE_BATCH, MAX_CHUNKS_PER_EPISODE, Hit
from podsearch.core.paths import LANCE_DIR, RSS_DIR

if TYPE_CHECKING:
    import lancedb
    from lancedb.query import LanceVectorQueryBuilder
    from lancedb.table import Table

CHUNKS_TABLE = "chunks"
PARTS_TABLE = "ingested_parts"
# Below this many rows, flat search in LanceDB's Rust core is fast and ANN
# training (IVF/HNSW) has too little data to be worthwhile.
MIN_ROWS_FOR_ANN = 50_000

PARTS_SCHEMA = pa.schema([("path", pa.string()), ("ingested_at", pa.int64())])


def _chunks_schema(dim: int) -> pa.Schema:
    return pa.schema(
        [
            ("feed_url", pa.string()),
            ("guid", pa.string()),
            ("podcast", pa.string()),
            ("episode", pa.string()),
            ("link", pa.string()),
            ("pubdate", pa.int64()),
            ("text", pa.string()),
            ("vector", pa.list_(pa.float32(), dim)),
        ]
    )


def _connect(path: Path) -> lancedb.DBConnection:
    import lancedb

    path.parent.mkdir(parents=True, exist_ok=True)
    return lancedb.connect(path)


def open_chunks(path: Path = LANCE_DIR) -> Table:
    """Open the chunks table for searching; exits helpfully if missing."""
    db = _connect(path)
    # list_tables() returns a paginated ListTablesResponse, not a plain list
    if CHUNKS_TABLE not in db.list_tables().tables:
        raise SystemExit(f"no lancedb store at {path} - run: podsearch-ingest --impl lancedb")
    return db.open_table(CHUNKS_TABLE)


def _sql_in(column: str, values: list[str]) -> str:
    quoted = ", ".join("'" + v.replace("'", "''") + "'" for v in values)
    return f"{column} IN ({quoted})"


def _ensure_btree(tbl: Table) -> None:
    """Scalar index on feed_url so per-feed deletes don't full-scan."""
    from lancedb.index import BTree

    indexed = {col for idx in tbl.list_indices() for col in idx.columns}
    if "feed_url" not in indexed:
        print("building BTree index on feed_url")
        tbl.create_index("feed_url", config=BTree())


def _ensure_ann(tbl: Table) -> None:
    """Build the ANN index once the store is big enough to warrant it.

    Run after ingest+optimize so it trains once on the final row count.
    Existing indexes are left alone: `create_index` would retrain them from
    scratch, whereas `optimize()` updates them incrementally.
    """
    from lancedb.index import HnswSq

    indexed = {col for idx in tbl.list_indices() for col in idx.columns}
    if "vector" not in indexed and (n := tbl.count_rows()) >= MIN_ROWS_FOR_ANN:
        print(f"building HNSW index on {n:,} vectors")
        tbl.create_index("vector", config=HnswSq(distance_type="cosine"))


def ingest_parts(
    model,
    rss_dir: Path = RSS_DIR,
    path: Path = LANCE_DIR,
    max_per_feed: int = 300,
    parser_impl: str = "rs",
    max_chunks: int = MAX_CHUNKS_PER_EPISODE,
    rebuild: bool = False,
    max_parts: int | None = None,
) -> dict[str, int]:
    """Parse not-yet-ingested parts straight into the lancedb chunks table."""
    db = _connect(path)
    if rebuild:
        for name in set(db.list_tables().tables) & {CHUNKS_TABLE, PARTS_TABLE}:
            db.drop_table(name)
    parts_tbl = db.create_table(PARTS_TABLE, schema=PARTS_SCHEMA, exist_ok=True)
    done = set(parts_tbl.to_arrow().column("path").to_pylist())
    parts = [p for p in sorted(rss_dir.glob("part-*.parquet")) if p.name not in done][:max_parts]
    stats = {"parts": 0, "chunks": 0, "skipped": len(done), "stale": 0}
    if not parts:
        return stats

    dim = len(model.encode(["dimension probe"])[0])
    tbl = db.create_table(CHUNKS_TABLE, schema=_chunks_schema(dim), exist_ok=True)
    _ensure_btree(tbl)
    latest = ingest_common.latest_fetch_dates(rss_dir)
    parse = partial(ingest_common.parse_feed, max_episodes=max_per_feed, impl=parser_impl)
    for px, part in enumerate(parts):
        t0 = time.perf_counter()
        raw = pl.read_parquet(part, columns=["url", "title", "rss", "fetch_date"])
        # stale (superseded by a newer part) or failed (no body): leave stored rows alone
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
            tbl.delete(_sql_in("feed_url", feeds["url"].to_list()))
        for batch in chunks.iter_slices(ENCODE_BATCH):
            tbl.add(ingest_common.embed(model, batch))
        print(
            f"  part done in {time.perf_counter() - t0:.1f}s: "
            f"{len(episodes):,} episodes, {chunks.height:,} chunks"
        )
        parts_tbl.add(
            pa.table({"path": [part.name], "ingested_at": [int(time.time())]}, PARTS_SCHEMA)
        )
        stats["parts"] += 1
        stats["chunks"] += chunks.height

    # stubs type stats() as TableStatistics; it returns a plain dict
    fragments = cast("dict[str, Any]", tbl.stats())["fragment_stats"]["num_fragments"]
    print(f"optimizing table ({fragments} fragments)")
    t0 = time.perf_counter()
    tbl.optimize()
    print(f"optimize done in {time.perf_counter() - t0:.1f}s")
    _ensure_ann(tbl)
    return stats


def search(tbl: Table, model, q: str, k: int = 5) -> list[Hit]:
    """Vector search; rows carry all metadata so hits need no hydration."""
    q_vec = np.asarray(model.encode([q])[0], dtype=np.float32)
    # stubs type search() as LanceQueryBuilder; a vector query yields the
    # vector builder that has .metric()
    builder = cast("LanceVectorQueryBuilder", tbl.search(q_vec))
    rows = builder.metric("cosine").limit(k * index.SEARCH_OVERFETCH).to_list()
    return index.dedup_hits(
        [
            Hit(
                podcast=r["podcast"],
                episode=r["episode"],
                score=0.0 if math.isnan(r["_distance"]) else 1.0 - r["_distance"],
                snippet=r["text"][:200],
                link=r["link"],
                pubdate=r["pubdate"],
                feed_url=r["feed_url"],
                guid=r["guid"],
            )
            for r in rows
        ],
        k,
    )
