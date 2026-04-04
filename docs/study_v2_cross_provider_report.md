# BrainLayer Study-V2 Cross-Provider Report

## Scope

This report combines the frozen `study-v2` Gemini baseline and the Anthropic cross-provider replication into one artifact.

The goal is to answer a narrower question than `study-v2` itself:

`Do the main BrainLayer component-attribution results replicate across providers, and how strong is the full BrainLayer advantage over simpler structured state?`

## Frozen Reference Sets

### Gemini Post-Patch Study-V2 Baseline

- `artifacts/study_runs/20260403T020152Z-study-v2-gemini-core-v5`
- `artifacts/study_runs/20260403T165058Z-study-v2-gemini-core-v5-repeat1`
- `artifacts/study_runs/20260403T174802Z-study-v2-gemini-core-v5-repeat2`

Reference report:

- `docs/study_v2_report.md`

### Gemini Component-Attribution Baseline

- `artifacts/matrix_runs/20260403T232157Z-study-v2-gemini-core-ablations-standard-v1`
- `artifacts/matrix_runs/20260403T234052Z-study-v2-gemini-core-ablations-hard-v1`
- `artifacts/matrix_runs/20260403T235134Z-study-v2-gemini-core-ablations-heldout-v1`
- `artifacts/matrix_runs/20260404T002614Z-study-v2-gemini-core-ablations-external-dev-v1`
- `artifacts/matrix_runs/20260404T004713Z-study-v2-gemini-core-ablations-external-heldout-v1`

Reference report:

- `docs/study_v2_ablation_report.md`

### Anthropic Cross-Provider Replication

- `artifacts/matrix_runs/20260404T022325Z-study-v2-anthropic-ablations-standard-smoke-v2`
- `artifacts/matrix_runs/20260404T025654Z-study-v2-anthropic-ablations-hard-v1`
- `artifacts/matrix_runs/20260404T031608Z-study-v2-anthropic-ablations-heldout-v1`
- `artifacts/matrix_runs/20260404T151122Z-study-v2-anthropic-ablations-external-dev-v1`
- `artifacts/matrix_runs/20260404T154745Z-study-v2-anthropic-ablations-external-heldout-contradiction-v1`
- `artifacts/matrix_runs/20260404T160413Z-study-v2-anthropic-ablations-external-heldout-natural-v1`

Reference note:

- `docs/study_v2_cross_provider_note.md`

## Main Results

### Gemini Post-Patch Study-V2

#### `gemini-2.5-flash`

- `brainlayer_full`: `90/95`, `92/95`, `91/95`
- `structured_no_consolidation`: `86/95`, `84/95`, `89/95`
- three-run average gap: `+4.7`

#### `gemini-2.5-flash-lite`

- `brainlayer_full`: `85/95`, `87/95`, `85/95`
- `structured_no_consolidation`: `81/95`, `86/95`, `87/95`
- three-run average gap: `+1.0`

### Gemini Component Attribution

#### `gemini-2.5-flash`

- `brainlayer_full`: `91/95`
- `brainlayer_no_consolidation`: `90/95`
- `brainlayer_no_forgetting`: `92/95`
- `brainlayer_no_autobio`: `79/95`
- `brainlayer_no_working_state`: `78/95`

#### `gemini-2.5-flash-lite`

- `brainlayer_full`: `87/95`
- `brainlayer_no_consolidation`: `87/95`
- `brainlayer_no_forgetting`: `87/95`
- `brainlayer_no_autobio`: `75/95`
- `brainlayer_no_working_state`: `77/95`

### Anthropic Component Attribution

#### `claude-sonnet-4.5`

- `brainlayer_full`: `90/95`
- `brainlayer_no_consolidation`: `90/95`
- `structured_no_consolidation`: `90/95`
- `brainlayer_no_forgetting`: `89/95`
- `brainlayer_no_working_state`: `79/95`
- `brainlayer_no_autobio`: `78/95`

#### `claude-haiku-4.5`

- `brainlayer_full`: `86/95`
- `brainlayer_no_consolidation`: `88/95`
- `structured_no_consolidation`: `87/95`
- `brainlayer_no_forgetting`: `87/95`
- `brainlayer_no_working_state`: `77/95`
- `brainlayer_no_autobio`: `78/95`

## What Replicated

The strongest `study-v2` result replicated across providers:

- removing `autobio` hurts a lot
- removing `working_state` hurts a lot

That pattern held on:

- `gemini-2.5-flash`
- `gemini-2.5-flash-lite`
- `claude-sonnet-4.5`
- `claude-haiku-4.5`

This is the most credible cross-provider result in the project so far.

## What Did Not Strengthen

Two weaker findings also replicated:

- `consolidation` remains weak on this frozen setup
- `forgetting` remains mixed to near-neutral on this frozen setup

In other words, the current BrainLayer gains are not being carried by the parts that were supposed to make the architecture cognitively richer over long horizons. They are mainly being carried by `working_state` and `autobio`.

## Exact Claim

The exact claim supported by the combined `study-v2` evidence is:

`Across Gemini and Anthropic, BrainLayer's strongest and most stable component-level advantages come from autobiographical state and working state. That pattern replicates across providers. The broader full-architecture advantage over simpler structured state is real on gemini-2.5-flash, narrower on gemini-2.5-flash-lite, and not yet universal across Anthropic models.`

## What This Does Not Claim

This report does not justify claiming:

- that BrainLayer is universally better than simpler structured state
- that consolidation is already doing important work
- that forgetting is already doing important work
- that the architecture is finished

## Interpretation

`study-v2` succeeded in one important way: it made the project's strongest real signal easier to see.

That signal is not:

- "the whole BrainLayer stack clearly wins everywhere"

It is:

- "`autobio` and `working_state` are consistently useful across providers"

That is meaningful, but it also makes the next research step obvious. The next phase should stop treating consolidation and forgetting as assumed wins and start treating them as open problems that must earn their place.

## Recommended Next Step

Use this report as the handoff into `study-v3`.

The central `study-v3` question should be:

`Can we make consolidation and forgetting produce clear, measurable value on harder long-horizon tasks, and can that turn BrainLayer into a more consistent winner over simpler structured state?`
