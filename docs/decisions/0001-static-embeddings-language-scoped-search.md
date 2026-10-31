# ADR-0001: Static embeddings with language-scoped search

- Status: accepted
- Date: 2026-10-30

## Context

`intfloat/multilingual-e5-small` was adopted (commit `f7fdbc6`) to fix
cross-language retrieval: the previous static model2vec encoder split German
compounds ("Islandpferde" → "Island"+fragments), so island episodes outranked
Icelandic-horse episodes. e5 fixed that but made indexing impractically slow —
it is a 12-layer transformer; model2vec is a lookup table.

Measured on an AMD Ryzen 7 5700U, torch 2.14.1+cpu, 512 × ~200-word chunks
(`ENCODE_BATCH` shape):

| Encoder                              | chunks/s | `"Islandpferde"` ranks      |
|--------------------------------------|----------|-----------------------------|
| `minishlab/potion-multilingual-128M` | ~4,278   | islands doc #1 (bug)        |
| `intfloat/multilingual-e5-small`     | ~3.7     | Icelandic horses #1 (fixed) |
| e5 + 6 processes × 2 threads         | ~8.7     | fixed                       |
| e5 → `model2vec.distill()`           | ~2,532   | islands doc #1 (bug)        |

Rejected alternatives (all measured, not guessed):

- **Tuning e5**: batch_size 128 → 4.1/s; 16 threads → 3.9/s; dynamic int8
  quant → 3.7/s. Single-process encode uses only ~59% CPU; a 6-process pool
  reaches ~8.7/s (~2.4x) — still ~500x short of model2vec.
- **ONNX Runtime / Optimum / OpenVINO**: no cp315 wheels exist
  (`onnxruntime` ships up to cp314). Blocked on Python 3.15.
- **Static distillation of e5 itself**: `model2vec.distill(e5-small)` fails the
  same Islandpferde regression — contextualization is precisely what fixed the
  bug; no bag-of-subword model can pass it.

## Decision

Revert the passage/query encoder to `minishlab/potion-multilingual-128M`
(static model2vec) and make **language a first-class, required search
dimension**:

- `language` propagates feed → episode → chunk. The Podcast Index catalog
  already declares feed language and `fetch` already stores it in the RSS
  parquet parts; ingest normalizes it to the primary subtag
  (`en-us` → `en`, missing/empty → `und`).
- One index per impl (column + filter, not per-language stores). All four
  search impls (`naive`, `numpy`, `lancedb`, `qdrant`) take a required `lang`
  and filter candidates before reranking.
- `podsearch-bench`/`podsearch-load` require `--lang`;
  `/search` and `/search-async` require a `lang` query param; the web UI
  offers a language dropdown populated from `/api/stats`.

Rationale: cross-language ranking was the failure model2vec could not fix —
but scoping a query to one language removes the entire cross-language surface
instead of paying ~1,150x for a transformer that papers over it. Query
translation (translate the query, keep language-scoped corpus) can restore
multilingual reach later without touching embeddings.

## Consequences

- Indexing returns to ~4k chunks/s; full-corpus indexing is feasible again.
- **Cross-language retrieval is explicitly out of scope**: a German query sees
  German episodes only. This is the accepted tradeoff.
- The reranker (`mmarco-mMiniLMv2`) stays — it still fixes ranking within a
  language.
- Rebuild required: dim 384 → 256 and the new `language` column invalidate
  existing stores. `episodes.language` backfills by resetting
  `ingested_parts` and re-ingesting existing parts (the column is already in
  the parquet); `vectors.f32` is rebuilt by `podsearch-index`; lancedb/qdrant
  need `podsearch-ingest --impl … --rebuild`.
- Feeds without a declared language land in `und`, reachable only via
  `lang=und`.

## Non-goals

- Query translation / multilingual expansion of a query (future work; the
  `lang` contract leaves room for it).
- Episode- or chunk-level language detection; feed-declared language is the
  accepted granularity.
- A cross-language "search everything" mode.

## Implementation plan

- `podsearch/core/index.py`: `MODEL_NAME` → potion-multilingual-128M,
  `load_model()` returns `StaticModel`; `Hit.language`; `search`/`search_numpy`
  take `lang` and restrict candidates to that language's vector rows (lazy
  per-lang row cache on `Index`).
- `podsearch/core/corpus.py`: `language` column on `episodes`/`chunks` +
  `ALTER TABLE` migration; ingest reads `language` from parts; `iter_episodes`
  yields it; `build_index` writes `chunks.language`.
- `podsearch/core/fetch.py`: `parse_feed` takes `language`, emits it per
  episode.
- `podsearch/core/ingest_common.py`: `language` in `EPISODE_COLUMNS` /
  `CHUNK_COLUMNS`; shared `lang_code()` normalizer.
- `podsearch/core/lance_store.py`: `language` column in schema; `.where()`
  prefilter in `search`.
- `podsearch/core/qdrant_store.py`: KEYWORD payload index on `language`;
  `FieldCondition` filter in `search`.
- `podsearch/cli/{common,query,bench,load}.py`, `podsearch/api.py`,
  `podsearch/static/index.html`: required `lang` end-to-end; `stats` exposes
  `languages` for the UI dropdown.
- `pyproject.toml`: re-add `model2vec` (`uv add model2vec`).
- Tests: `test_retrieval.py` rewritten to same-language quality sanity;
  lang-filter plumbing tests per impl; fake parts carry `language`.

## Verification

- [ ] `task check` green (ruff, ty, pytest)
- [ ] searching "Pferde" with `lang=de` returns only German hits
- [ ] UI language dropdown filters results
- [ ] Indexing throughput is back to ~4k chunks/s order of magnitude
- [ ] `episodes.language`/`chunks.language` populated after re-ingest
