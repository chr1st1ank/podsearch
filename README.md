# pepsearch — profiling demo

Semantic search over the PEPs, built to be profiled. Two search implementations:

- `naive`: cosine similarity in a pure-Python loop (`index.search`) — the hot spot
- `numpy`: the same search as a single matrix product (`index.search_numpy`) — ~100x faster

Model: `model2vec` static embedding `minishlab/potion-base-8M` (256 dims, downloads ~30 MB
from Hugging Face on first use, cached afterwards). Corpus: `data/peps-main` (tarball of
github.com/python/peps, 745 `pep-*.rst` files, ~21 MB). Index: `data/index.json`
(12,588 chunks, ~84 MB).

## Setup

```bash
cd drafts/profiling-demo
uv sync                          # installs pepsearch editable into .venv (Python 3.15)
```

## Commands

```bash
# build the index (took ~17 s, mostly model encode)
uv run pepsearch-index                       # or: pepsearch ingest

# single query
uv run pepsearch query "how do I make a context manager"
uv run pepsearch query "how do I make a context manager" --impl numpy

# benchmark: 20 queries from queries.txt (naive ~0.32 s/query, numpy ~0.003 s/query)
uv run pepsearch bench                 # naive, ~6 s
uv run pepsearch bench --repeat 5      # ~30 s, better for sampling
uv run pepsearch bench --impl numpy --repeat 100

# web service (attach target)
uv run pepsearch-serve                       # serves :8000

# load generator (run in a second terminal)
uv run pepsearch-load --duration 30 --concurrency 4
uv run pepsearch-load --url http://127.0.0.1:8000/search-async --duration 30
```

The `-m` form also works (`uv run python -m pepsearch.cli bench`) and is what the
profiler examples below use. `python -m pepsearch.api` and `python -m pepsearch.indexing`
are equivalent to `pepsearch-serve` and `pepsearch-index`.

## Profiling with Tachyon (Python 3.15, `profiling.sampling`)

```bash
# run and profile a module — text table to stdout
uv run python -m profiling.sampling run -m pepsearch.cli bench --repeat 3

# flamegraph / heatmap / collapsed
uv run python -m profiling.sampling run --flamegraph -o results/flame.html  -m pepsearch.cli bench --repeat 3
uv run python -m profiling.sampling run --heatmap    -o results/heat       -m pepsearch.cli bench --repeat 3
uv run python -m profiling.sampling run --collapsed  -o results/stacks.txt -m pepsearch.cli bench --repeat 3

# faster rate, native frames, opcodes, modes
uv run python -m profiling.sampling run -r 10khz --flamegraph -o results/flame-10khz.html -m pepsearch.cli bench --repeat 3
uv run python -m profiling.sampling run --native --flamegraph -o results/flame-native.html -m pepsearch.cli bench --repeat 3
uv run python -m profiling.sampling run --opcodes --flamegraph -o results/flame-opcodes.html -m pepsearch.cli bench --repeat 3
uv run python -m profiling.sampling run --mode=cpu  -m pepsearch.cli bench --repeat 3
uv run python -m profiling.sampling run --mode=gil  -m pepsearch.cli bench --repeat 3
uv run python -m profiling.sampling run --mode=exception -m pepsearch.cli bench --repeat 3

# record binary, replay to another format
uv run python -m profiling.sampling run --binary -o results/naive.bin -m pepsearch.cli bench --repeat 3
uv run python -m profiling.sampling run --binary -o results/numpy.bin -m pepsearch.cli bench --impl numpy --repeat 100
uv run python -m profiling.sampling replay --flamegraph -o results/numpy.html results/numpy.bin

# differential flamegraph: baseline first
uv run python -m profiling.sampling run --diff-flamegraph results/naive.bin -o results/diff.html -m pepsearch.cli bench --impl numpy --repeat 100
```

## Attach to the running server

**Requires ptrace permission.** `run` works out of the box (the profiler is the parent).
`attach` to an already-running process is blocked by `ptrace_scope=1` until you run once:

```bash
echo 0 | sudo tee /proc/sys/kernel/yama/ptrace_scope   # reset to 1 when done
```

Then, with the server running and load applied:

```bash
PID=$(pgrep -f "\.venv/bin/python -m pepsearch.app" | head -1)
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
uv run python -m profiling.tracing -o results/trace.pstats -m pepsearch.cli bench
uv run python -m cProfile -s cumtime -m pepsearch.cli bench | head -30

# py-spy (third-party; also attach-capable)
uvx py-spy record -o results/pyspy.svg -- uv run python -m pepsearch.cli bench --repeat 3
uvx py-spy top --pid $PID
```

## Development

```bash
task check      # lint + typecheck + test (ruff, ty, pytest)
task bench      # = uv run pepsearch bench
task serve      # = uv run pepsearch-serve
task load       # = uv run pepsearch-load
```

## Files

- `pepsearch/core/corpus.py` — load PEPs, `clean()` RST, `chunk()` into word windows
- `pepsearch/core/index.py` — `build_index`, `search` (naive), `search_numpy`, JSON save/load
- `pepsearch/indexing.py` — indexing entrypoint (builds `data/index.json`)
- `pepsearch/api.py` — FastAPI `/search`, `/search-async` (blocks the event loop), `/health`
- `pepsearch/cli.py` — `ingest` / `query` / `bench`
- `pepsearch/load.py` — threaded load generator
- `pepsearch/queries.txt` — 20 questions used by `bench` and `load.py`
- `tests/` — unit tests for corpus and index helpers
