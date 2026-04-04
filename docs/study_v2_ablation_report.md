# BrainLayer Study-V2 Ablation Report

## Scope

This report summarizes the first live ablation sweep on the frozen `study-v2` setup for:

- `gemini-2.5-flash`
- `gemini-2.5-flash-lite`

The sweep was run pack by pack across the same five frozen `study-v2` packs:

- `standard`
- `hard`
- `held_out`
- `external_dev`
- `external_held_out`

Source bundles:

- `artifacts/matrix_runs/20260403T232157Z-study-v2-gemini-core-ablations-standard-v1`
- `artifacts/matrix_runs/20260403T234052Z-study-v2-gemini-core-ablations-hard-v1`
- `artifacts/matrix_runs/20260403T235134Z-study-v2-gemini-core-ablations-heldout-v1`
- `artifacts/matrix_runs/20260404T002614Z-study-v2-gemini-core-ablations-external-dev-v1`
- `artifacts/matrix_runs/20260404T004713Z-study-v2-gemini-core-ablations-external-heldout-v1`

## Conditions

This sweep compared:

- `brainlayer_full`
- `brainlayer_no_consolidation`
- `brainlayer_no_forgetting`
- `brainlayer_no_autobio`
- `brainlayer_no_working_state`

For context, `structured_no_consolidation` is included as a non-BrainLayer structured baseline where useful.

## Aggregate Results

### `gemini-2.5-flash`

- `brainlayer_full`: `91/95`
- `brainlayer_no_consolidation`: `90/95`
- `brainlayer_no_forgetting`: `92/95`
- `brainlayer_no_autobio`: `79/95`
- `brainlayer_no_working_state`: `78/95`
- `structured_no_consolidation`: `87/95`

### `gemini-2.5-flash-lite`

- `brainlayer_full`: `87/95`
- `brainlayer_no_consolidation`: `87/95`
- `brainlayer_no_forgetting`: `87/95`
- `brainlayer_no_autobio`: `75/95`
- `brainlayer_no_working_state`: `77/95`
- `structured_no_consolidation`: `87/95`

## Pack-Level Pattern

### `gemini-2.5-flash`

- `standard`: full `19/19`, no consolidation `19/19`, no forgetting `19/19`, no autobio `17/19`, no working state `17/19`
- `hard`: full `13/14`, no consolidation `13/14`, no forgetting `13/14`, no autobio `12/14`, no working state `12/14`
- `held_out`: full `12/14`, no consolidation `12/14`, no forgetting `14/14`, no autobio `11/14`, no working state `11/14`
- `external_dev`: full `24/24`, no consolidation `24/24`, no forgetting `24/24`, no autobio `20/24`, no working state `22/24`
- `external_held_out`: full `23/24`, no consolidation `22/24`, no forgetting `22/24`, no autobio `19/24`, no working state `16/24`

### `gemini-2.5-flash-lite`

- `standard`: full `19/19`, no consolidation `19/19`, no forgetting `19/19`, no autobio `18/19`, no working state `17/19`
- `hard`: full `14/14`, no consolidation `14/14`, no forgetting `14/14`, no autobio `13/14`, no working state `13/14`
- `held_out`: full `14/14`, no consolidation `14/14`, no forgetting `14/14`, no autobio `12/14`, no working state `13/14`
- `external_dev`: full `20/24`, no consolidation `20/24`, no forgetting `20/24`, no autobio `16/24`, no working state `16/24`
- `external_held_out`: full `20/24`, no consolidation `20/24`, no forgetting `20/24`, no autobio `16/24`, no working state `18/24`

## Main Takeaway

The exact takeaway from this frozen `study-v2` ablation sweep is:

`Autobiographical state and working state matter a lot. Consolidation looks weak on this frozen setup. Forgetting looks neutral on this frozen setup.`

More concretely:

- removing `autobio` causes a large drop on both models: `-12` points overall on `gemini-2.5-flash` and `-12` on `gemini-2.5-flash-lite`
- removing `working_state` also causes a large drop: `-13` on `gemini-2.5-flash` and `-10` on `gemini-2.5-flash-lite`
- removing `consolidation` changes little in the aggregate: `-1` on `gemini-2.5-flash` and `0` on `gemini-2.5-flash-lite`
- removing `forgetting` does not hurt in this sweep: `+1` on `gemini-2.5-flash` and `0` on `gemini-2.5-flash-lite`

## Interpretation

The current frozen `study-v2` tasks appear to rely much more on:

- persistent self/user continuity
- active goal/state tracking

than on:

- explicit consolidation logic
- forgetting pressure

That does not mean consolidation or forgetting are unimportant in general. It means they are not yet the main source of gain on this frozen evaluation mix.

The current evidence supports a narrower claim:

`On the frozen study-v2 setup, BrainLayer gains are driven primarily by autobiographical continuity and working state, not by consolidation or forgetting.`

## What This Does Not Claim

This report does not justify claiming:

- that forgetting is useless in general
- that consolidation is never helpful
- that the current five-pack sweep fully isolates causal contribution under all task distributions

It only supports the claim above for the current frozen `study-v2` packs and models.

## Recommended Next Step

Treat this ablation sweep as the current component-attribution reference for `study-v2`.

From here:

1. freeze these results in the repo
2. run the same frozen ablations on a second model family or provider
3. only then decide whether `study-v3` should redesign consolidation/forgetting or simply expand the task mix to stress them more directly
