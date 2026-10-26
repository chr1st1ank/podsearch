# Talk Storyline: Find the Bottleneck, Then Profile It

## Goal

A 25-minute introduction to profiling with Python 3.15's built-in sampling
profiler, Tachyon (`profiling.sampling`).

Teach a repeatable investigation strategy through one real pipeline and one
improvement round. The example is podcast RSS ingestion and embedding, not a
comparison of databases or a tour of profiler features.

## Central Question

> This script turns podcast feeds into a searchable index. Which step do you
> think takes longest: parsing RSS, cleaning the text, or creating embeddings?

Ask the audience to vote before showing any timings. Let the measurements,
not the presenter, answer the question.

## Investigation Strategy

**Define -> Measure -> Locate -> Explain -> Change -> Verify**

1. Define the workload and what "fast enough" means.
2. Measure the whole pipeline without a profiler.
3. Locate the slowest stage, then profile that stage.
4. Explain the hot stack by reading the code and forming a hypothesis.
5. Make one targeted change.
6. Verify correctness and rerun the whole pipeline without profiling.

A benchmark tells us whether we improved. A profile helps explain where to act.
A wide frame tells us where time is spent, not necessarily why.

## Demo Scope

Use a fixed, local snapshot of real RSS feeds. Downloading the catalog and
fetching live feeds happen before the talk. Show these steps greyed out in the
pipeline diagram as "already downloaded".

The measured pipeline is:

```text
Stored RSS -> Parse and store episodes -> Clean, chunk and embed -> Searchable index
                     timed                         timed
```

The existing SQLite pipeline separates ingestion and indexing. Its CLIs accept
explicit input and output paths, so the demo needs only a small runner, not
another backend or orchestration framework.

Cleaning, chunking and embedding are initially one timed stage. The profile can
distinguish their contributions if that stage turns out to dominate.

## Demo Runner Plan

- Use the same bounded RSS snapshot for every variant.
- Create a fresh output directory for each run rather than deleting existing
  files or touching the main dataset.
- Time each stage and the total with a monotonic clock.
- Print a simple stage timing breakdown.
- Support baseline and optimized variants with identical inputs and limits.
- Keep the model available locally so the demo does not depend on downloads.
- Aim for a baseline around 20-40 seconds, subject to rehearsal measurements.

Keep startup and model-loading costs visible and consistent between variants.
Do not quietly compare a cold baseline against a warm optimized run.

## Narrative Arc

### 1. Predict

Introduce the podcast search use case and show the pipeline.

Ask the audience which component they expect to dominate. Embeddings are a
plausible suspect, but do not promise a surprise before measuring.

### 2. Measure

Run the bounded dataset and show the stage timings.

Explain why the investigation starts with coarse measurements: we first need
to know which component deserves a closer look.

### 3. Locate

Profile only the slowest stage with Tachyon. Open one flamegraph and explain
stack structure and width using the actual result.

Use a fresh output directory so incremental processing does not turn the
profile into a no-op.

### 4. Explain

Follow the hot stack into the code.

Ask:

> What is this doing repeatedly, and what change would reduce that work?

Distinguish the observation from the hypothesis. Do not treat the hottest
function as automatically defective.

### 5. Improve

Reveal one prepared change justified by the profile. Avoid live implementation
work that distracts from the investigation.

### 6. Verify

Check that the output is still correct, then rerun the complete pipeline
without profiling.

Show the before/after stage breakdown and total time. Optionally show a prepared
after-profile if it helps explain the change.

The payoff:

> Making a component faster only matters insofar as it improves the workflow.

Explain Amdahl's law in that sentence without introducing the formula. Show
the remaining bottleneck, but stop after one improvement round.

## Candidate Improvement: RSS Parser

The code already supports `--parser feedparser` and `--parser rs`. A possible
demo is an explicit `feedparser` baseline followed by the `rs` implementation.
The current CLI default is `rs`, so the baseline must select `feedparser`
explicitly.

This is a candidate, not yet a measured conclusion:

- First determine whether parsing dominates on a representative subset.
- Compare equivalent episode content, not just episode counts. Account for
  intentional parser differences and verify downstream output as appropriate.
- Hold the dataset, episode limits, chunk limits and model constant.
- Measure the end-to-end gain, not only parser speed.

No native-code profiling or implementation deep dive is needed. The lesson is
to replace a measured hotspot with a suitable library implementation.

If measurements do not support this improvement, choose the actual bottleneck
rather than forcing the data to fit the planned story.

## Timing

| Minutes | Story beat | What to show |
| --- | --- | --- |
| 0-4 | Problem and prediction | Use case, pipeline, audience vote |
| 4-8 | Baseline | Live run and stage timing breakdown |
| 8-15 | Investigation | Profile the slowest stage, read the flamegraph and code |
| 15-20 | One improvement | Prepared change, correctness check, full-pipeline rerun |
| 20-23 | Results | Before/after timings, total gain and remaining bottleneck |
| 23-25 | Takeaways and buffer | Recap the strategy and stopping condition |

## Presentation Guardrails

- Only one improvement round.
- No live network dependency.
- No LanceDB/Qdrant comparison, native-thread profiling or async detour.
- Introduce profiler options only when the investigation needs them.
- Prepare baseline and after-profiles plus timing results as fallbacks.
- Keep the dataset real and fixed; no artificial sleeps or fake load.
- Use current rehearsal results, not speedups recorded for an older corpus.
- Stop when the workflow is fast enough instead of chasing every hot frame.

## Next Rehearsal

1. Select and freeze a representative RSS subset.
2. Time ingestion and indexing in fresh output directories.
3. Adjust the subset size to fit the live-demo time budget.
4. Profile the slowest stage.
5. Test whether parser replacement is correct and gives a clear end-to-end gain.
6. Record the final commands, correctness checks, timings and fallback profiles.

## Closing Takeaway

> Start broad, measure the workflow, then zoom in. Change one thing, check the
> result, and measure the workflow again.
