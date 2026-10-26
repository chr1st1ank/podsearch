# podsearch — profiling demo

Semantic search over podcast episodes, built to be profiled. Three search implementations:

- `naive`: cosine similarity in a pure-Python loop (`index.search`) — the hot spot
- `numpy`: the same search as a single matrix product (`index.search_numpy`) — ~100x+ faster
- `lancedb`: embedded vector DB (`lance_store.search`) — vectors + metadata in one
  store, no hydration; the "production" impl, outside the profiling story
- `qdrant`: Qdrant server in Docker (`qdrant_store.search`) — same fused layout as lancedb,
  vectors + payload in one collection, server-side HNSW

Model: `model2vec` static embedding `minishlab/potion-base-8M` (256 dims, downloads 30 MB
from Hugging Face on first use, cached afterwards). Corpus: podcast episodes selected from
the Podcast Index catalog and fetched via RSS. Index: `data/vectors.f32` (raw float32 matrix,
memory-mapped) + `data/corpus.db` (SQLite: episodes, chunks, ingest bookkeeping).

## Data pipeline

Four independent stages — each is its own script (and `python -m` module).
Inputs on the left, data stores (all gitignored under `data/`) in the middle,
consumers on the right:

```mermaid
flowchart TD
    CAT(["Podcast Index<br/>catalog tgz"]) --> C1["podsearch-catalog"]
    C1 -->|"download + extract"| DB[("data/podcastindex_feeds.db<br/>feed metadata only")]
    DB --> C2["podsearch-fetch-rss"]
    WEB(["podcast hosts"]) --> C2
    C2 -->|"HTTP GET · append parts ·<br/>prune > 14 d"| PQ[("data/rss/part-*.parquet<br/>raw RSS")]
    PQ --> C3["podsearch-ingest"]
    C3 -->|"newest version per feed ·<br/>upsert episodes"| CORP[("data/corpus.db<br/>episodes + chunks")]
    CORP --> C4["podsearch-index"]
    C4 -->|"chunk + encode"| VEC[("data/vectors.f32")]
    VEC -->|"cosine: naive vs numpy"| S5["podsearch-query · -bench ·<br/>-serve · -load"]
    CORP -->|"hydrate top-k"| S5
```

```bash
# 1. download + extract the Podcast Index catalog (~1.8 GB tgz -> ~5 GB sqlite); skips if present
uv run podsearch-catalog

# 2. select feeds from the catalog and fetch their RSS into data/rss/*.parquet
#    idempotent: skips feeds already stored unless catalog lastUpdate is newer;
#    parts older than --prune-days (default 14) are deleted afterwards (0 disables)
uv run podsearch-fetch-rss                          # 10k most recently active en+de feeds
uv run podsearch-fetch-rss --feeds 100 --order episodes --lang en

# 3. parse stored RSS into the episodes corpus (incremental: only new parquet
#    parts; only the newest fetched version per feed is parsed, episodes upsert)
uv run podsearch-ingest                             # --max-per-feed 300 caps long feeds

# 4. embed episodes into chunks + vectors (full rebuild of derived data)
uv run podsearch-index

# lancedb alternative: stages 3+4 fused — parse + chunk + encode + insert in one
# pass into data/lancedb/ (vector + metadata columns in a single `chunks` table;
# delete-then-add per refetched feed; incremental via its own ingested_parts)
uv run podsearch-ingest --impl lancedb              # threaded parse, batched encode
uv run podsearch-ingest --impl lancedb --rebuild    # drop and recreate the store

# qdrant alternative: same fused ingest into a Qdrant server (docker compose)
task qdrant-up                                      # Qdrant on :6333, storage in data/qdrant/
uv run podsearch-ingest --impl qdrant               # --qdrant-url / PODSEARCH_QDRANT_URL to override
uv run podsearch-ingest --impl qdrant --rebuild     # drop and recreate the collections
```

### Orchestration

The four stages can also run as a [Dagu](https://dagu.cloud) DAG — a lightweight
local orchestrator (single binary, web UI with run history and per-step logs).
The DAG lives in `orchestration/podsearch.yaml` and just wraps the `uv run`
commands above, so the stages' own idempotency still decides what is skipped.
Dagu is a mise tool (`mise install`; the task entries call it via
`mise exec -- dagu`), no Python dependency. Steps run with the repo root as
working dir (the tasks export `PODSEARCH_DIR`; set it yourself if launching
dagu by hand):

The lancedb pipeline is a separate DAG — `orchestration/podsearch-lancedb.yaml`
(catalog → fetch_rss → ingest_lancedb; the fused stage replaces ingest+index).
Both DAGs share the catalog db and rss parquet store (fetch once, ingest into
each backend independently) and both appear in the same dagu UI.

```bash
task dagu               # web UI + scheduler on :8081 (no auth; override port via CLI args)
task pipeline           # one-shot run of the sqlite DAG
task pipeline-lancedb   # one-shot run of the lancedb DAG
task pipeline-qdrant    # one-shot run of the qdrant DAG (needs task qdrant-up)
```

Storage layout (all under gitignored `data/`):

- `podcastindex_feeds.db` — feed catalog (4.7M feeds; metadata only, no episode content)
- `rss/part-*.parquet` — raw RSS bodies + `fetch_date` + all catalog columns per feed;
  append-only history, parts pruned after 14 days (pruned feeds get refetched)
- `corpus.db` — `episodes` (feed_url, guid, podcast, title, link, pubdate, text;
  upserts on refetch so it mirrors the latest stored feed version),
  `chunks` (vector_row, episode_id, text), `ingested_parts`, `meta`
- `vectors.f32` — raw float32 matrix aligned with `chunks.vector_row`
- `lancedb/` — alternative store: one `chunks` table (vector + feed_url, guid,
  podcast, episode, link, pubdate, text) + `ingested_parts`; ANN index
  (HNSW-SQ cosine) is built once the table is large enough to train
- `qdrant/` — Qdrant server storage (compose volume): `chunks` collection (vector +
  same payload) and `ingested_parts`

## Exploring the data

`notebooks/sample.py` is a marimo notebook that prints a random sample of each
store above (run `marimo run` instead of `edit` for a read-only view):

```bash
uv run marimo edit notebooks/sample.py
```

(marimo is a dev-group dep, `>=0.25.2` because that release gates its `loro`
dep behind `python_version < "3.15"` — no cp315 wheels exist.)

## Search

```bash
uv run podsearch-query "startup fundraising advice"
uv run podsearch-query "startup fundraising advice" --impl numpy
uv run podsearch-query "startup fundraising advice" --impl lancedb
uv run podsearch-query "startup fundraising advice" --impl qdrant

# benchmark: queries from podsearch/queries.txt
uv run podsearch-bench --queries 2                 # naive: slow on purpose
uv run podsearch-bench --impl numpy --queries 20 --repeat 100
uv run podsearch-bench --impl lancedb --queries 20
uv run podsearch-bench --impl qdrant --queries 20

# web service (attach target)
uv run podsearch-serve                                  # serves :8000

# load generator (run in a second terminal)
uv run podsearch-load --duration 30 --concurrency 4
uv run podsearch-load --url http://127.0.0.1:8000/search-async --duration 30
```

The `-m` form also works (`uv run python -m podsearch.cli.bench`,
`uv run python -m podsearch.cli.ingest`).

## Profiling with Tachyon (Python 3.15, `profiling.sampling`)

`run` takes a script path or `-m module` — it does not search `PATH`, so console
scripts like `podsearch-ingest` won't resolve; use the `-m` form (the venv script
path `.venv/bin/podsearch-ingest` also works). Add `-d SECONDS` to cap the run.

The stages are idempotent, so the examples pass scratch outputs to force real
work without clobbering the actual stores: `--db data/corpus-prof.db` (ingest),
`--rss-dir data/rss-prof` (fetch), `--vectors data/vectors-prof.f32` (index).

```bash
# ingest: feedparser is CPU-bound — the classic flamegraph target
uv run python -m profiling.sampling run -m podsearch.cli.ingest --db data/corpus-prof.db
uv run python -m profiling.sampling run --flamegraph -o results/ingest.html -m podsearch.cli.ingest --db data/corpus-prof.db
uv run python -m profiling.sampling run --heatmap    -o results/ingest-heat -m podsearch.cli.ingest --db data/corpus-prof.db
uv run python -m profiling.sampling run --collapsed  -o results/ingest.txt  -m podsearch.cli.ingest --db data/corpus-prof.db
uv run python -m profiling.sampling run -r 10khz --flamegraph -o results/ingest-10khz.html -m podsearch.cli.ingest --db data/corpus-prof.db
uv run python -m profiling.sampling run --opcodes --flamegraph -o results/ingest-opcodes.html -m podsearch.cli.ingest --db data/corpus-prof.db
uv run python -m profiling.sampling run --mode=cpu       -m podsearch.cli.ingest --db data/corpus-prof.db
uv run python -m profiling.sampling run --mode=exception -m podsearch.cli.ingest --db data/corpus-prof.db

# fetch-rss: 32 threads waiting on the network — wall vs cpu diverge,
# gil shows the threads aren't GIL-bound
uv run python -m profiling.sampling run --mode=wall -m podsearch.cli.fetch_rss --rss-dir data/rss-prof --feeds 500
uv run python -m profiling.sampling run --mode=cpu  -m podsearch.cli.fetch_rss --rss-dir data/rss-prof --feeds 500
uv run python -m profiling.sampling run --mode=gil  -m podsearch.cli.fetch_rss --rss-dir data/rss-prof --feeds 500

# index: model2vec encode — native frames under the Python stack
uv run python -m profiling.sampling run --native --flamegraph -o results/index-native.html -m podsearch.cli.indexing --vectors data/vectors-prof.f32
uv run python -m profiling.sampling run --native --heatmap -o results/index-heat -m podsearch.cli.indexing --vectors data/vectors-prof.f32

# query: a single naive search — "why is this slow?"
uv run python -m profiling.sampling run --flamegraph -o results/query.html -m podsearch.cli.query "startup fundraising advice"

# record binary, replay to another format
uv run python -m profiling.sampling run --binary -o results/ingest.bin -m podsearch.cli.ingest --db data/corpus-prof.db
uv run python -m profiling.sampling replay --flamegraph -o results/ingest-replay.html results/ingest.bin
```

## Attach to the running server

**Requires ptrace permission.** `run` works out of the box (the profiler is the parent).
`attach` to an already-running process is blocked by `ptrace_scope=1` until you run once:

```bash
echo 0 | sudo tee /proc/sys/kernel/yama/ptrace_scope   # reset to 1 when done
```

Then, with the server running and load applied:

```bash
PID=$(pgrep -f "podsearch-serve" | head -1)
uv run python -m profiling.sampling dump $PID                       # one stack snapshot
uv run python -m profiling.sampling attach --live $PID              # top-like TUI, q quits
uv run python -m profiling.sampling attach -d 20 --flamegraph -o results/attach.html $PID
uv run python -m profiling.sampling attach -d 20 --mode=wall $PID   # includes idle wait
uv run python -m profiling.sampling attach -d 20 --mode=cpu  $PID   # only on-CPU samples
uv run python -m profiling.sampling attach --async-aware -d 20 $PID # with load on /search-async
```

## Comparison

```bash
# deterministic profiler (formerly cProfile)
uv run python -m profiling.tracing -o results/trace.pstats -m podsearch.cli.ingest --db data/corpus-prof.db
uv run python -m cProfile -s cumtime -m podsearch.cli.ingest --db data/corpus-prof.db | head -30

# py-spy (third-party; also attach-capable)
uvx py-spy record -o results/pyspy.svg -- uv run python -m podsearch.cli.ingest --db data/corpus-prof.db
uvx py-spy top --pid $PID
```

## Development

```bash
task check      # lint + typecheck + test (ruff, ty, pytest)
task catalog    # download the feed catalog
task fetch      # fetch RSS into data/rss/
task ingest     # parse RSS into corpus.db
task index      # build vectors.f32
task pipeline   # run the sqlite pipeline once via Dagu
task pipeline-lancedb  # run the lancedb pipeline once via Dagu
task dagu       # Dagu web UI on :8081
task bench      # = uv run podsearch-bench
task serve      # = uv run podsearch-serve
task load       # = uv run podsearch-load
```

## Files

- `podsearch/core/paths.py` — data/ file locations
- `podsearch/core/catalog.py` — catalog download + feed selection SQL
- `podsearch/core/fetch.py` — idempotent RSS fetch -> parquet parts, `parse_feed`
- `podsearch/core/corpus.py` — `strip_html`, `chunk`, corpus.db schema, `ingest_parts`
- `podsearch/core/index.py` — `build_index`, `search` (naive), `search_numpy`, hit hydration
- `podsearch/core/lance_store.py` — fused ingest+index + `search` for `--impl lancedb`
- `podsearch/core/qdrant_store.py` — fused ingest+index + `search` for `--impl qdrant`
- `podsearch/core/ingest_common.py` — chunking/embedding helpers shared by both fused stores
- `podsearch/api.py` — FastAPI `/search`, `/search-async` (blocks the event loop), `/health`
- `podsearch/cli/` — one entrypoint module per `podsearch-*` console script:
  `catalog`, `fetch_rss`, `ingest`, `indexing` (stages 1-4), `serve`, `load`, `query`, `bench`
- `podsearch/queries.txt` — 20 queries used by `bench` and `load.py`
- `tests/` — unit tests for corpus, fetch, and index helpers
