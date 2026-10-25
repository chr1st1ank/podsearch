# Handoff: Tachyon profiling demo ("pepsearch")

## Goal

Build/blog about the new built-in sampling profiler **Tachyon** (`profiling.sampling`, Python 3.15,
PEP 799) for a blog post (Hugo blog at `~/code/blog`) and a 25-min talk for a mixed local Python
meetup (50-80 people, presentation first, then the post). Workflow skill: `blog-writing` — currently
between Phase 2 (research, mostly done) and Phase 3 (storyline). Drafts live in `~/code/blog/drafts/`.

## User decisions so far

- Example = hook, not an optimization exercise: **only ONE improvement round**.
- Example: "Ask the PEPs" mini-RAG — semantic search over the PEPs with **local** model2vec
  embeddings (`minishlab/potion-base-8M`, 256-dim). No network/LLM calls.
- cProfile mentioned but not detailed; py-spy as alternative; memory profiling only touched
  (link user's 2022 posts: `2022-09-04-locating-memory-leaks-in-services-part1/2.md`).
- Work stays inside `~/code/blog/drafts/profiling-demo/`.

## What exists (all verified working on Python 3.15.0)

`~/code/blog/drafts/profiling-demo/` — flat-layout uv/hatchling project (`pepsearch/` at root,
installed editable; **no PYTHONPATH needed**). Deps: model2vec, numpy, fastapi, uvicorn, httpx.
Tooling: `task check` = ruff + ty (all rules error) + pytest; scripts `pepsearch`,
`pepsearch-index`, `pepsearch-serve`, `pepsearch-load` (`-m` form still works, e.g.
`python -m pepsearch.cli bench`, `python -m pepsearch.api`).

- `pepsearch/core/corpus.py` — load 745 PEPs from `data/peps-main`, `clean()` RST, `chunk()`
- `pepsearch/core/index.py` — `search` (naive pure-Python cosine loop = hot spot), `search_numpy`
  (cached normalized float32 matrix, ~100x faster), JSON index `data/index.json` (12,588 chunks, 84 MB)
- `pepsearch/indexing.py` — indexing entrypoint; `pepsearch ingest` delegates to it
- `pepsearch/api.py` — FastAPI `/search` (sync), `/search-async` (blocks event loop), `/health`;
  `uv run pepsearch-serve`
- `pepsearch/cli.py` — `ingest` / `query` / `bench` (`--impl naive|numpy`, `--repeat`)
- `pepsearch/load.py` — threaded load generator
- `pepsearch/queries.txt` (20), `README.md` (full command list), `tests/`,
  `results/flame-bench-naive.html`

## Verified numbers

- ingest 17 s; naive ~0.32 s/query; numpy ~0.003 s/query; query "context manager" → PEP 343 (good)
- Tachyon `run` + `--flamegraph` works; flamegraph shows `_cosine` line. 20,809 samples in 20.8 s
  but **"Error rate: 25.93"** — worth investigating (rate? 3.15.0 bug?)
- `attach`/`dump` **blocked**: `ptrace_scope=1`. Needs user-run
  `echo 0 | sudo tee /proc/sys/kernel/yama/ptrace_scope` (restore to 1 after)
- Tachyon CLI matches docs: `run|attach|dump|replay`, `--mode {wall,cpu,gil,exception}`,
  `--async-aware`, `--native`, `--opcodes`, `--live`, `--diff-flamegraph BASELINE`, `--binary`

## Docs/research notes

`~/code/blog/drafts/phase2-profiling-python-research.md` — sources, flags, caveats (sampling =
estimates, not for <1 s runs/exact call counts/benchmarks; same-minor-version requirement; cProfile
→ `profiling.tracing`, `profile` removed in 3.17). Detailed execution plan:
`~/code/blog/drafts/profiling-demo-spike-plan.md` (paths in it are partially outdated — user edited
to point into drafts/).

## Next steps

1. Finish spike verification if needed: modes (wall vs cpu shows no diff on CPU-bound bench — use
   server under load), `--native`, `--opcodes`, `--async-aware` on `/search-async`,
   `--diff-flamegraph` for before/after, cProfile/py-spy overhead comparison, tracemalloc/memray
   check, report to `~/code/blog/drafts/phase2b-profiling-spike-results.md` (per plan §4–10).
2. Phase 3: storyline + headline options (read `references/writing-guide.md` in the blog-writing
   skill dir); save to `drafts/phase3-*.md`, confirm with user before drafting.

## Cautions

- Only ~2 GB RAM free — one heavy process at a time.
- Never run `sudo`; if attach is blocked, ask user to set ptrace_scope.
- Keep code "naturally naive" — no `sleep()`/fake load.
