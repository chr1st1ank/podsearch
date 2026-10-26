# Profile says "waiting", CPU says "busy": `ingest --impl lancedb`

## Problem

Profiling the lancedb ingest shows a script that is almost entirely idle, yet the
machine's CPU is pegged while it runs.

```bash
uv run python -m profiling.sampling run -a --native --flamegraph -o results/ingest.html -m podsearch.cli.ingest --impl lancedb
uv run python -m profiling.sampling run -a --native --collapsed  -o results/ingest.txt  -m podsearch.cli.ingest --impl lancedb
```

![flamegraph](python-performance-flamegraph.svg)

## What the profile shows

Three stacks hold ~99% of samples (~56k each, one per thread):

| Thread                  | Stack                                                                                                          | Meaning                             |
|-------------------------|----------------------------------------------------------------------------------------------------------------|-------------------------------------|
| tqdm monitor            | `TMonitor.run` -> `Condition.wait`                                                                             | idle (only visible because of `-a`) |
| LanceDB background loop | `run_forever` -> `EpollSelector.select`                                                                        | idle (only visible because of `-a`) |
| **main**                | `ingest_parts:198` -> `create_indexes:128` -> `LanceTable.create_index` -> `Future.result` -> `Condition.wait` | **blocked waiting on LanceDB**      |

Nothing from the ingest loop (parse, encode, `tbl.add`) appears: the run had no new
parts, so it was just `create_indexes`.

## Why the CPU is hot anyway

`create_index` hands the HNSW-SQ build to LanceDB's Rust runtime (tokio/rayon threads)
and the Python main thread just waits on a `Future`. The sampler only walks Python
interpreter threads, so those Rust threads are invisible, even with `--native`
(`--native` adds native frames to Python threads; it does not discover foreign threads).

Takeaway: "mostly waiting" can mean "the Python thread is idle while native code works
elsewhere".

## Confirm

```bash
top -H -p $(pgrep -f podsearch.cli.ingest)   # many busy native threads
perf top -p $(pgrep -f podsearch.cli.ingest) # HNSW build in native code
```

`--mode=cpu` does not help: the main thread really is idle.

## Profile the native threads with perf

Start the ingest in the background, grab its PID with `$!`, and attach perf (it exits
when the process does):

```bash
python -m podsearch.cli.ingest --impl lancedb &
sudo perf record -F 99 -g --call-graph dwarf -p $! -o results/ingest.perf.data
```

Or, if it is already running: `-p $(pgrep -f podsearch.cli.ingest)`.

Read the result:

```bash
perf report -i results/ingest.perf.data --no-children --sort comm,dso   # which threads / libraries
perf report -i results/ingest.perf.data --no-children --sort symbol     # hottest functions
perf report -i results/ingest.perf.data --stdio --no-children | head -80
```

Note: `--call-graph dwarf` files get big (hundreds of MB), so keep them out of git.

### Result

The readable view is by thread and library, without call graphs:

```bash
perf report -i results/ingest.perf.data --no-children --sort comm,dso --stdio -g none | grep -v '^#' | grep -v '^$'
```

```
95.39%  lancedb-tokio-w  _lancedb.abi3.so
 3.06%  lancedb-tokio-w  [kernel.kallsyms]
 1.26%  lancedb-tokio-w  libc.so.6
 0.17%  python           python3.15
 ...
```

~99.7% of CPU is on `lancedb-tokio-w` threads, 95% of it inside the LanceDB native
library. The `python` thread is ~0.2%. This is the missing CPU time Tachyon cannot see.

`--sort symbol` shows only raw addresses (`0x875a609`, ...): the PyPI wheel is stripped
(`nm` reports no symbols), so function names are not available. The hot addresses cluster
in a few tight loops (`0x875a5xx`, `0x82e88xx`, `0x634c0xx`), consistent with distance
computation during the HNSW-SQ build, but that is an inference, not something the
profile names. To get names, build `lancedb` from source with debug symbols.

## Real bug found

`lance_store.ingest_parts` calls `create_indexes(tbl)` whenever the table exists, even
when `stats["parts"] == 0`. `create_index` replaces an existing index, so every no-op
re-run retrains HNSW on the whole table.

Fix options: only (re)index when parts were ingested, skip if the index already exists,
or let `tbl.optimize()` update existing indexes incrementally.
