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

    from podsearch.core import index
    from podsearch.core.paths import CATALOG_DB, CORPUS_DB, RSS_DIR, VECTORS_PATH

    return CATALOG_DB, CORPUS_DB, RSS_DIR, VECTORS_PATH, index, mo, np, pl, random, sqlite3


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
def catalog(CATALOG_DB, mo, n, pl, rng, sqlite3, sql_sample):
    if not CATALOG_DB.exists():
        _out = mo.md("`data/podcastindex_feeds.db` not found — run `podsearch-catalog`")
    else:
        _conn = sqlite3.connect(f"file:{CATALOG_DB}?mode=ro", uri=True)
        _total, _df = sql_sample(
            _conn,
            "podcasts",
            "id, title, url, language, episodeCount, lastUpdate, popularityScore",
            n.value,
            rng,
        )
        _conn.close()
        _out = mo.vstack(
            [
                mo.md(f"## catalog — {_total:,} feeds"),
                _df.with_columns(
                    pl.from_epoch("lastUpdate").dt.strftime("%Y-%m-%d").alias("updated")
                ).drop("lastUpdate"),
            ]
        )
    _out
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
def corpus(CORPUS_DB, mo, n, pl, rng, shorten, sqlite3, sql_sample):
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


if __name__ == "__main__":
    app.run()
