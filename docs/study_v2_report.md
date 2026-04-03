# BrainLayer Study-V2 Report

## Official Post-Patch Reference

The current post-patch `study-v2` reference set is:

- `artifacts/study_runs/20260403T020152Z-study-v2-gemini-core-v5`
- `artifacts/study_runs/20260403T165058Z-study-v2-gemini-core-v5-repeat1`
- `artifacts/study_runs/20260403T174802Z-study-v2-gemini-core-v5-repeat2`

These three bundles should be treated together as the frozen post-patch `study-v2` baseline.

## Model Set

The post-patch `study-v2` runs use:

- `gemini-2.5-flash`
- `gemini-2.5-flash-lite`

The main comparison is between:

- `brainlayer_full`
- `structured_no_consolidation`
- `naive_retrieval`
- `summary_state`
- `context_only`

## Three-Run Aggregate Results

### `gemini-2.5-flash`

`brainlayer_full`

- per-run overall: `90/95`, `92/95`, `91/95`
- three-run average: `91.0/95`
- average estimated total cost: `$0.0384`
- average latency: `1977.2ms`

`structured_no_consolidation`

- per-run overall: `86/95`, `84/95`, `89/95`
- three-run average: `86.3/95`
- average estimated total cost: `$0.0385`
- average latency: `2062.3ms`

### `gemini-2.5-flash-lite`

`brainlayer_full`

- per-run overall: `85/95`, `87/95`, `85/95`
- three-run average: `85.7/95`
- average estimated total cost: `$0.0109`
- average latency: `836.0ms`

`structured_no_consolidation`

- per-run overall: `81/95`, `86/95`, `87/95`
- three-run average: `84.7/95`
- average estimated total cost: `$0.0110`
- average latency: `866.7ms`

## Pack-Level Pattern

Across all three post-patch runs:

- `standard` stayed at `19/19` for the top row
- `hard` stayed at `14/14` for the top row
- `held_out` stayed at `14/14` for the top row
- `external_dev` stayed at `24/24` for the top row
- `external_held_out` stayed strong but imperfect at `22/24`, `22/24`, and `21/24`

The remaining pressure is now concentrated in `external_held_out`, not in the authored packs.

## Exact Claim

The exact claim supported by the frozen post-patch `study-v2` reference set is:

`BrainLayer shows a strong and stable overall advantage over stronger non-BrainLayer baselines on gemini-2.5-flash, and a smaller, noisier, but still slightly favorable average result on gemini-2.5-flash-lite.`

More concretely:

- on `gemini-2.5-flash`, `brainlayer_full` beats `structured_no_consolidation` in all three full post-patch runs
- on `gemini-2.5-flash-lite`, the margin is much smaller and flips once, so the honest read is near-tie with a slight average edge for `brainlayer_full`

## What This Does Not Claim

This result does not justify claiming that BrainLayer is universally or decisively better on every model size.

It also does not justify claiming that the problem is solved. The main remaining difficulty is generalization on the hardest external held-out pack, not authored-pack regression.

## Interpretation

`study-v1` established that BrainLayer was plausible.

The frozen post-patch `study-v2` set goes further:

- the comparison is against stronger baselines
- the task mix includes external and held-out packs
- the strongest model shows a stable BrainLayer win
- the smaller model suggests the architecture still helps, but the gain is narrower and more sensitive to run noise

## Recommended Next Step

Treat this post-patch `study-v2` set as frozen.

From here:

1. use this report as the reference artifact for external communication
2. run planned ablations on the frozen setup
3. only then start a new labeled phase such as `study-v3`
