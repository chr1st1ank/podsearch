import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def imports():
    import random
    import sqlite3

    import marimo as mo
    import numpy as np
    import polars as pl

    from podsearch.core import index, lance_store
    from podsearch.core.paths import (
        CATALOG_DB,
        CORPUS_DB,
        LANCE_DIR,
        RSS_DIR,
        VECTORS_PATH,
    )

    return (
        CATALOG_DB,
        CORPUS_DB,
        LANCE_DIR,
        RSS_DIR,
        VECTORS_PATH,
        index,
        lance_store,
        mo,
        np,
        pl,
        random,
        sqlite3,
    )


@app.cell
def header(mo):
    mo.md("""
    # podsearch — data sampler

    A random sample of each pipeline store: catalog → rss → corpus → vectors.
    Change `n` or the seed to resample.
    """)
    return


@app.cell
def controls(mo):
    n = mo.ui.number(start=1, stop=25, value=5, label="sample size n")
    seed = mo.ui.text(value="", label="seed (blank = random)")
    mo.vstack([n, seed])
    return n, seed


@app.cell
def make_rng(random, seed):
    seed_int = int(seed.value) if seed.value.strip().isdigit() else None
    rng = random.Random(seed_int)
    return rng, seed_int


@app.cell
def helpers(pl):
    def sql_sample(conn, table, columns, n, rng):
        """n random rows via sampled primary keys (no ORDER BY RANDOM() scan)."""
        total, max_id = conn.execute(f"SELECT COUNT(*), MAX(id) FROM {table}").fetchone()
        if not total:
            return total, pl.DataFrame()
        want = min(n, total)
        ids = rng.sample(range(1, max_id + 1), min(want * 3, max_id))
        df = pl.read_database(
            f"SELECT {columns} FROM {table} WHERE id IN ({','.join(map(str, ids))}) LIMIT {want}",
            conn,
        )
        return total, df

    def shorten(column, width=160):
        return pl.col(column).str.replace_all(r"\s+", " ").str.slice(0, width)

    return shorten, sql_sample


@app.cell
def catalog(CATALOG_DB, mo, pl, sqlite3):
    if not CATALOG_DB.exists():
        _out = mo.md("`data/podcastindex_feeds.db` not found — run `podsearch-catalog`")
        df = pl.DataFrame()
    else:
        _conn = sqlite3.connect(f"file:{CATALOG_DB}?mode=ro", uri=True)
        df = pl.read_database(
            "SELECT id, title, url, language, episodeCount, lastUpdate,"
            " newestItemPubdate, popularityScore FROM podcasts",
            _conn,
        )
        _conn.close()
        _out = mo.vstack(
            [
                mo.md(f"## catalog — {len(df)} feeds"),
                df.with_columns(
                    pl.from_epoch("lastUpdate").dt.strftime("%Y-%m-%d").alias("updated")
                ).drop("lastUpdate"),
            ]
        )
    _out
    return (df,)


@app.cell
def _(df, pl):
    df.group_by(pl.col("language")).len().sort(pl.col("len"), descending=True).head(10)
    return


@app.cell
def feeds_controls(mo):
    feed_order = mo.ui.dropdown(
        options=["recent", "episodes", "popularity"], value="recent", label="--order"
    )
    feed_min_eps = mo.ui.number(start=0, value=20, label="--min-episodes")
    feed_langs = mo.ui.text(value="en,de", label="--lang")
    mo.vstack([feed_order, feed_min_eps, feed_langs])
    return feed_langs, feed_min_eps, feed_order


@app.cell
def feeds_scaling(df, feed_langs, feed_min_eps, feed_order, mo, pl):
    """Cumulative episodes `catalog.select_feeds` would yield as --feeds grows."""
    _sort_by = {
        "recent": ["newestItemPubdate"],
        "episodes": ["episodeCount"],
        "popularity": ["popularityScore", "newestItemPubdate"],
    }[feed_order.value]
    _langs = [lang.strip().lower() for lang in feed_langs.value.split(",") if lang.strip()]
    _cum = (
        df.filter(
            pl.col("episodeCount") >= feed_min_eps.value,
            pl.any_horizontal(
                pl.lit(not _langs),
                *(pl.col("language").str.to_lowercase().str.starts_with(lang) for lang in _langs),
            ),
        )
        .sort(_sort_by, descending=True, nulls_last=True)["episodeCount"]
        .cum_sum()
    )
    mo.stop(_cum.is_empty(), mo.md("_no feeds match these filters_"))
    _steps = [s for s in (10, 100, 1_000, 10_000, 100_000, 1_000_000) if s < len(_cum)]
    _steps.append(len(_cum))
    mo.vstack(
        [
            mo.md(f"## episodes vs `--feeds` — {len(_cum):,} eligible feeds"),
            pl.DataFrame(
                {"feeds": _steps, "episodes": _cum.gather([s - 1 for s in _steps])}
            ).with_columns(
                (pl.col("episodes") / pl.col("feeds")).round(1).alias("episodes/feed"),
                (pl.col("episodes") / _cum[-1] * 100).round(1).alias("% of eligible"),
            ),
        ]
    )
    return


@app.cell
def rss(RSS_DIR, mo, n, pl, rng, seed_int):
    _parts = sorted(RSS_DIR.glob("part-*.parquet")) if RSS_DIR.exists() else []
    if not _parts:
        _out = mo.md("`data/rss/` not found — run `podsearch-fetch-rss`")
    else:
        _n_rows = pl.scan_parquet(RSS_DIR / "part-*.parquet").select(pl.len()).collect().item()
        _part = rng.choice(_parts)
        _table = pl.read_parquet(_part, columns=["url", "title", "rss", "fetch_date"])
        _out = mo.vstack(
            [
                mo.md(
                    f"## rss — {len(_parts)} parts, {_n_rows:,} feeds · sample from `{_part.name}`"
                ),
                _table.sample(n=min(n.value, _table.height), seed=seed_int)
                .with_columns(
                    pl.col("rss").str.len_chars().alias("rss_chars"),
                    pl.col("rss").str.replace_all(r"\s+", " ").str.slice(0, 200).alias("rss_head"),
                    pl.from_epoch("fetch_date").dt.strftime("%Y-%m-%d").alias("fetched"),
                )
                .drop("rss", "fetch_date"),
            ]
        )
    _out
    return


@app.cell
def corpus(CORPUS_DB, mo, n, pl, rng, shorten, sql_sample, sqlite3):
    if not CORPUS_DB.exists():
        _out = mo.md("`data/corpus.db` not found — run `podsearch-ingest`")
    else:
        _conn = sqlite3.connect(f"file:{CORPUS_DB}?mode=ro", uri=True)
        _counts = dict(
            _conn.execute(
                "SELECT 'episodes', COUNT(*) FROM episodes"
                " UNION ALL SELECT 'chunks', COUNT(*) FROM chunks"
                " UNION ALL SELECT 'parts', COUNT(*) FROM ingested_parts"
            ).fetchall()
        )
        _, _eps = sql_sample(
            _conn, "episodes", "id, podcast, title, pubdate, link, text", n.value, rng
        )
        _, _chunks = sql_sample(
            _conn, "chunks", "id, vector_row, episode_id, text", max(2, n.value // 2), rng
        )
        _conn.close()
        _out = mo.vstack(
            [
                mo.md(
                    f"## corpus — {_counts['episodes']:,} episodes · "
                    f"{_counts['chunks']:,} chunks · {_counts['parts']} ingested parts"
                ),
                mo.md("### episodes"),
                _eps.with_columns(
                    shorten("text"),
                    pl.from_epoch("pubdate").dt.strftime("%Y-%m-%d").alias("published"),
                ).drop("pubdate"),
                mo.md("### chunks"),
                _chunks.with_columns(shorten("text")),
            ]
        )
    _out
    return


@app.cell
def vectors(CORPUS_DB, VECTORS_PATH, index, mo, n, np, pl, rng, shorten):
    if not VECTORS_PATH.exists() or not CORPUS_DB.exists():
        _out = mo.md("`data/vectors.f32` not found — run `podsearch-index`")
    else:
        try:
            _idx = index.load(VECTORS_PATH, CORPUS_DB)
        except Exception as _e:
            _out = mo.md(f"cannot load index: {_e}")
        else:
            _n_vec, _dim = _idx.vectors.shape
            _rows = rng.sample(range(_n_vec), min(n.value, _n_vec))
            _meta = pl.read_database(
                "SELECT c.vector_row, c.text AS chunk, e.podcast, e.title FROM chunks c"
                " JOIN episodes e ON e.id = c.episode_id"
                f" WHERE c.vector_row IN ({','.join(map(str, _rows))})",
                _idx.conn,
            ).with_columns(shorten("chunk"))
            _idx.conn.close()
            _v = _idx.vectors[_rows]
            _out = mo.vstack(
                [
                    mo.md(f"## vectors — {_n_vec:,} x {_dim} float32"),
                    pl.DataFrame(
                        {
                            "vector_row": _rows,
                            "norm": np.linalg.norm(_v, axis=1).round(3),
                            "dims": [np.round(v[:4], 3).tolist() for v in _v],
                        }
                    )
                    .join(_meta, on="vector_row", how="left")
                    .sort("vector_row"),
                ]
            )
    _out
    return


@app.cell
def lancedb(LANCE_DIR, lance_store, mo, n, pl, rng, seed_int, shorten):
    if not LANCE_DIR.exists():
        _out = mo.md("`data/lancedb/` not found — run `podsearch-ingest --impl lancedb`")
        lance_tbl = None
    else:
        lance_tbl = lance_store.open_chunks(LANCE_DIR)
        _stats = lance_tbl.stats()
        _dim = lance_tbl.schema.field("vector").type.list_size
        _idx = lance_tbl.list_indices()
        _names = set(lance_store._connect(LANCE_DIR).list_tables().tables)
        _n_parts = (
            lance_store._connect(LANCE_DIR).open_table(lance_store.PARTS_TABLE).count_rows()
            if lance_store.PARTS_TABLE in _names
            else 0
        )
        _rows = rng.sample(range(_stats["num_rows"]), min(n.value, _stats["num_rows"]))
        _sample = (
            pl.from_arrow(
                lance_tbl.take_offsets(_rows)
                .select(["podcast", "episode", "pubdate", "text"])
                .to_arrow()
            )
            .with_columns(
                shorten("text"),
                pl.from_epoch("pubdate").dt.strftime("%Y-%m-%d").alias("published"),
            )
            .drop("pubdate")
        )
        _out = mo.vstack(
            [
                mo.md(
                    f"## lancedb — {_stats['num_rows']:,} chunks x {_dim} dims · "
                    f"{_stats['total_bytes'] / 2**30:.1f} GiB · "
                    f"{_stats['fragment_stats']['num_fragments']} fragments · "
                    f"{len(lance_tbl.list_versions())} versions · "
                    f"{_n_parts} ingested parts"
                ),
                mo.md(
                    "**indexes:** "
                    + ", ".join(
                        f"`{i.name}` ({i.index_type}, "
                        f"{i.num_indexed_rows:,} indexed / {i.num_unindexed_rows:,} unindexed)"
                        for i in _idx
                    )
                ),
                mo.md("### sample chunks"),
                _sample,
            ]
        )
    _out
    return (lance_tbl,)


@app.cell
def lance_model(index):
    """Cached embedding-model loader so repeated searches don't reload."""
    from functools import cache

    load = cache(index.load_model)
    load_reranker = cache(index.load_reranker)
    return load, load_reranker


@app.cell
def lance_query(mo):
    query = mo.ui.text(value="Home Assistant", label="query")
    run = mo.ui.run_button(label="search")
    mo.vstack([query, run])
    return query, run


@app.cell
def lance_search(lance_store, lance_tbl, load, load_reranker, mo, pl, query, run, shorten):
    mo.stop(lance_tbl is None, mo.md("_no lancedb store_"))
    mo.stop(not run.value, mo.md("_press search_"))
    _hits = lance_store.search(lance_tbl, load(), load_reranker(), query.value, k=10)
    mo.vstack(
        [
            mo.md(f"### hits for `{query.value}`"),
            pl.DataFrame([h.__dict__ for h in _hits])
            .with_columns(
                pl.col("score").round(3),
                shorten("snippet", 120),
                pl.from_epoch("pubdate").dt.strftime("%Y-%m-%d").alias("published"),
            )
            .drop("pubdate", "link")
            if _hits
            else mo.md("_no hits_"),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
