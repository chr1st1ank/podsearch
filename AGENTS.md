# Handoff: Tachyon profiling demo ("podsearch")

## Goal

Build/blog about the new built-in sampling profiler **Tachyon** (`profiling.sampling`, Python 3.15,
PEP 799) for a blog post (Hugo blog at `~/code/blog`) and a 25-min talk for a mixed local Python
meetup (50-80 people, presentation first, then the post). Workflow skill: `blog-writing` — currently
between Phase 2 (research, mostly done) and Phase 3 (storyline). Drafts live in `~/code/blog/drafts/`.

## User decisions so far

- Example = hook, not an optimization exercise: **only ONE improvement round**.
- Example: podcast-episode semantic search over real RSS data with **local** model2vec
  embeddings (`minishlab/potion-base-8M`, 256-dim). No LLM calls. Replaced the earlier
  "Ask the PEPs" idea — that corpus was too small to produce a real performance problem.
- Full-size corpus is the point: hundreds of thousands of episodes so the naive search
  loop is genuinely slow (target ~15-25 s/query vs ~0.3 s at the old 12.6k-chunk scale).
- cProfile mentioned but not detailed; py-spy as alternative; memory profiling only touched
  (link user's 2022 posts: `2022-09-04-locating-memory-leaks-in-services-part1/2.md`).
- Work lives in `~/code/blog-assets/code/profiling-demo/` (copied out of drafts/).

## Pipeline (independent CLI stages, all under gitignored data/)

`podsearch-catalog` — download+extract the Podcast Index SQLite dump to
`data/podcastindex_feeds.db` (~5 GB; 4.7M feeds; **feed metadata only**, no episode content).
`podsearch-fetch-rss` — select feeds (`--feeds`, `--lang` default en+de, `--min-episodes`,
`--order recent|episodes|popularity`) → fetch RSS concurrently → append `data/rss/part-*.parquet`
(raw body + fetch_date + all catalog columns). Idempotent: refetches only missing feeds or
feeds whose catalog `lastUpdate` is newer than `fetch_date`.
`podsearch-ingest` — feedparser on stored parts → `data/corpus.db` episodes table
(incremental via `ingested_parts`; `--max-per-feed` 300 caps mega-feeds).
`podsearch-index` — episodes → chunks (≤3/episode) → batched encode → `data/vectors.f32`
raw float32 + `chunks.vector_row` alignment; `meta` table holds dim/n.

Then `podsearch query|bench` (`--impl naive|numpy`), `podsearch-serve` (:8000),
`podsearch-load`. `podsearch bench --queries` defaults to 5 — naive is seconds-to-tens-of-
seconds per query at scale.

## What exists (verified on Python 3.15.0)

- flat-layout uv/hatchling project (`podsearch/` at root, editable install). Deps: model2vec,
  fastapi, uvicorn, httpx, feedparser, pyarrow. `task check` = ruff + ty (all error) + pytest.
- `podsearch/core/{paths,catalog,fetch,corpus,index}.py`, `podsearch/{cli,ingest,indexing,api,load}.py`
- Index storage is scale-shaped: `vectors.f32` np.memmap + sqlite hydration of top-k hits
  (no `list[list[float]]`, no JSON index — would be GBs at this scale).
- Verified: catalog idempotent skip; fetch idempotent (`pending 0` on re-run); ingest
  incremental (0 dupes); smoke run of 20 feeds → 3,512 episodes → 5,660 chunks → 6 MB
  vectors; naive and numpy return identical hits; numpy ~0.01 s/query at that size.

## Verified numbers (from the old PEP corpus — refresh after full ingest)

- Old scale: 12,588 chunks; naive ~0.32 s/query; numpy ~0.003 s/query; ingest 17 s.
- Tachyon `run` + `--flamegraph` works; flamegraph shows `_cosine` line. 20,809 samples in
  20.8 s but **"Error rate: 25.93"** — worth investigating (rate? 3.15.0 bug?)
- `attach`/`dump` **blocked**: `ptrace_scope=1`. Needs user-run
  `echo 0 | sudo tee /proc/sys/kernel/yama/ptrace_scope` (restore to 1 after)
- Tachyon CLI matches docs: `run|attach|dump|replay`, `--mode {wall,cpu,gil,exception}`,
  `--async-aware`, `--native`, `--opcodes`, `--live`, `--diff-flamegraph BASELINE`, `--binary`

## Docs/research notes

`~/code/blog/drafts/phase2-profiling-python-research.md` — sources, flags, caveats (sampling =
estimates, not for <1 s runs/exact call counts/benchmarks; same-minor-version requirement; cProfile
→ `profiling.tracing`, `profile` removed in 3.17). Detailed execution plan:
`~/code/blog/drafts/profiling-demo-spike-plan.md` (paths partially outdated — user edited
to point into drafts/).

## Next steps

1. Finish spike verification if needed: modes (wall vs cpu shows no diff on CPU-bound bench — use
   server under load), `--native`, `--opcodes`, `--async-aware` on `/search-async`,
   `--diff-flamegraph` for before/after, cProfile/py-spy overhead comparison, tracemalloc/memray
   check, report to `~/code/blog/drafts/phase2b-profiling-spike-results.md` (per plan §4–10).
2. Refresh README/AGENTS numbers once the full fetch+ingest+index has run at scale.
3. Phase 3: storyline + headline options (read `references/writing-guide.md` in the blog-writing
   skill dir); save to `drafts/phase3-*.md`, confirm with user before drafting.

## Cautions

- Never run `sudo`; if attach is blocked, ask user to set ptrace_scope.
- Keep code "naturally naive" — no `sleep()`/fake load.
- Fetch politeness: bounded concurrency, timeouts, UA header; don't re-hammer dead feeds
  (idempotency in the parquet store handles this; failures are not stored, so a re-run
  retries them — intentional for transient errors).
- A copied `.venv` once had stale script shebangs pointing at the old drafts env;
  `uv sync --reinstall` fixes them.
