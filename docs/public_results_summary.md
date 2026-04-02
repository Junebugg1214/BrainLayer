# BrainLayer Public Results Summary

## Summary

We tested a simple claim: agents may need more than "memory." They may need a layered cognitive state, a `BrainLayer`, that tracks working state, beliefs, autobiographical continuity, procedures, consolidation, and forgetting.

Our first frozen baseline, `study-v1-gemini-core`, supports that direction.

## Frozen Baseline

- baseline id: `study-v1-gemini-core`
- study bundle: `artifacts/study_runs/20260402T175429Z-study-v1-gemini-core`
- config: `examples/model_matrix.gemini.chat.core.live.json`

Models:

- `gemini-2.5-flash`
- `gemini-2.5-flash-lite`

## Headline Results

### `gemini-2.5-flash`

- overall: `43/47`
- contradiction: `16/16`
- natural: `27/31`
- held-out leader

### `gemini-2.5-flash-lite`

- overall: `39/47`
- contradiction: `14/16`
- natural: `25/31`
- fastest and cheapest model in the baseline

## Per-Pack Readout

### Standard

- `gemini-2.5-flash`: `19/19`
- `gemini-2.5-flash-lite`: `16/19`

### Hard

- `gemini-2.5-flash`: `14/14`
- `gemini-2.5-flash-lite`: `14/14`

### Held-Out

- `gemini-2.5-flash`: `10/14`
- `gemini-2.5-flash-lite`: `9/14`

## What This Means

Three things stand out.

1. BrainLayer is already strong on contradiction and revision.
   Both models handled the contradiction-heavy workloads very well, and both were perfect on the harder delayed/noisy authored pack.

2. Natural-memory extraction is workable, not just synthetic.
   The runtime can infer structured state from ordinary dialogue, not only from explicit `Record ...` prompts.

3. Held-out generalization is still the frontier.
   The biggest remaining weakness is alternate everyday wording, not the basic mechanics of retrieval, persistence, or revision.

## Why We Froze Here

We froze this baseline before tuning away the remaining held-out misses.

That matters because a benchmark should still be able to disagree with us. If we keep patching every held-out wording variant into the runtime, we stop measuring generalization and start optimizing to the test.

So this baseline is intentionally imperfect. That is a feature, not a bug.

## Bottom Line

The first BrainLayer baseline suggests that agent memory should be treated less like a note store and more like a lightweight cognitive state.

The architecture looks strong on:

- contradiction handling
- continuity
- natural extraction
- procedural reuse

The next step is not to relabel this as solved. The next step is to use this frozen baseline as the reference point for a clearly labeled `study-v2`.

## References

- internal baseline report: `docs/baseline_report.md`
- frozen protocol: `docs/study_protocol.md`
- experiment notes: `docs/experiments.md`
