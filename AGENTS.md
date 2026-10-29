# Agent instructions

This file helps AI agents work effectively in the `podsearch` repository.

## Project purpose

Semantic search over podcast episodes. Feed metadata comes from the Podcast Index
catalog, episode text is fetched via RSS, embedded locally with
`intfloat/multilingual-e5-small` (384-dim, sentence-transformers), and searched through
four swappable backends: `naive`, `numpy`, `lancedb`, `qdrant`. A FastAPI service and a
CLI suite sit on top. Python 3.15, uv/hatchling, flat layout (`podsearch/` at root).

## Repository structure

- `podsearch/core/` — pure library: catalog, fetch, corpus, index, stores
- `podsearch/cli/` — one entrypoint module per `podsearch-*` console script
- `podsearch/api.py` — FastAPI app (`/search`, `/search-async`, `/health`)
- `podsearch/static/` — minimal web UI served by the API
- `podsearch/queries.txt` — 20 queries used by `bench` and `load`
- `tests/` — pytest suite; `test_retrieval.py` guards cross-language retrieval quality
- `notebooks/` — marimo data-sampler notebook
- `orchestration/` — Dagu DAGs wrapping the pipeline stages
- `compose.yaml` — Qdrant server for `--impl qdrant`
- `data/` — gitignored; all pipeline stores live here (see README storage layout)
- `local/` — gitignored scratch space. Put working files here.

## Key architectural rules

- `core/` is the pure library; `cli/` modules only parse args and call `core/`.
- `core/` must not import from `cli/` or `api`.
- `--impl` selects the storage/search backend (`naive|numpy|lancedb|qdrant`);
  `--parser` selects the feed parser (`feedparser|rs`).
- All search impls share the retrieve-then-rerank flow: overfetch `max(6k, 30)` chunks,
  cross-encoder `index.rerank` (`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`,
  sigmoid scores), then `dedup_hits`.
- Every pipeline stage is idempotent — incremental bookkeeping (`ingested_parts`,
  `fetch_date` vs catalog `lastUpdate`) decides what to skip. Preserve this.

## Gotchas

- lancedb `db.list_tables()` returns a paginated object — use `.tables` for names;
  `Table.search()` stubs type vector queries wrong — `cast` for ty.
- `torch` resolves via `[[tool.uv.index]] pytorch-cpu` — PyPI has no cp315 Linux
  wheels; `tool.uv.environments` pins resolution to Linux only.
- Index storage is scale-shaped: `vectors.f32` np.memmap + SQLite hydration of top-k
  hits. No `list[list[float]]` or JSON index — GBs at full-corpus scale.
- lancedb returns cosine *distance*, qdrant cosine *similarity* — scores not comparable.
- Fetch politeness: bounded concurrency, timeouts, UA header. Failed fetches are not
  stored, so re-runs retry them — intentional for transient errors.

## Commands

```bash
task check        # lint + typecheck + test — run before handing work back
task lint         # ruff check --fix + ruff format
task typecheck    # ty check (all rules error severity)
task test         # pytest with coverage
task catalog      # download Podcast Index feed catalog (idempotent)
task fetch        # fetch RSS into data/rss/*.parquet (idempotent)
task ingest       # parse RSS into corpus.db (incremental)
task index        # build vectors.f32
task qdrant-up    # start Qdrant on :6333 (needed for qdrant tests/impl)
task dagu         # Dagu web UI on :8081
task serve        # FastAPI on :8000
task bench        # search benchmark
task update-dependencies  # upgrade locks + hooks within cooldown, then check
```

Run from the repository root. Dagu runs via `mise exec -- dagu` (mise tool, not a
Python dep); tasks export `PODSEARCH_DIR` for the DAGs.

## House rules

- **Scratch files go in `local/`** (gitignored). Never commit working notes or dumps.
- Never run `sudo`; if profiler `attach` is blocked by `ptrace_scope`, ask the user.
- The `naive` implementation is intentionally unoptimized — it is the project's
  profiling target. Don't "fix" it.
- Never log or return secrets, tokens, or credentials.

## Code style

- Type hints required; `py.typed` is shipped. `ty` runs at error severity on all rules.
- Keep CLI modules thin; logic belongs in `core/` where it is testable.

## Pre-commit checklist

- `task check` passes
- New logic has tests; network and stores are faked/mocked in tests
- No secrets in code or log output
