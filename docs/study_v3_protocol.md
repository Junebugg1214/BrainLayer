# BrainLayer Study-V3 Protocol

## Status

This file defines the intended frozen protocol for `study-v3`.

It should be treated as the comparison contract for the next phase of work:

- stress-test consolidation
- stress-test forgetting
- compare against the strongest simpler structured baseline
- freeze the first `study-v3` result only after the new packs are stable

Implementation should follow this protocol rather than changing the criteria midstream.

## Reference Inputs

`study-v3` is grounded in these frozen `study-v2` artifacts:

- `docs/study_v2_report.md`
- `docs/study_v2_ablation_report.md`
- `docs/study_v2_cross_provider_report.md`

Key carry-forward finding:

- `autobio` and `working_state` already look useful
- `consolidation` is still weak
- `forgetting` is still near-neutral
- `brainlayer_full` is not yet a universal winner over simpler structured state

## Study-V3 Question

Can stronger consolidation and forgetting produce a clear, repeatable advantage on long-horizon, noisy, revision-heavy tasks where simpler structured state should begin to fail?

## Primary Hypotheses

### Hypothesis 1: Consolidation Matters Under Delay

When durable beliefs, procedures, or lessons must be inferred from repeated weak signals across long gaps, `brainlayer_full` should outperform `brainlayer_no_consolidation` and `structured_no_consolidation`.

### Hypothesis 2: Forgetting Matters Under Noise

When stale, superseded, or irrelevant state accumulates, `brainlayer_full` should outperform `brainlayer_no_forgetting` and `structured_no_consolidation`.

### Hypothesis 3: Full BrainLayer Separates More Clearly

If consolidation and forgetting become meaningfully useful, `brainlayer_full` should gain a stronger margin over `structured_no_consolidation` than it achieved in `study-v2`.

## Provider Set

Primary provider pair for the first frozen `study-v3` pass:

- `gemini-2.5-flash`
- `claude-sonnet-4.5`

Optional smaller-model follow-up:

- `gemini-2.5-flash-lite`
- `claude-haiku-4.5`

The first freeze should not be blocked on the smaller-model pair.

## Required Runtime Conditions

The first frozen `study-v3` pass must include:

- `brainlayer_full`
- `brainlayer_no_consolidation`
- `brainlayer_no_forgetting`
- `structured_no_consolidation`

Secondary baselines may still exist in the codebase, but they are not required for the first frozen `study-v3` result.

## Evaluation Packs

### Regression Packs

Keep these packs for continuity:

- `standard`
- `hard`
- `held_out`
- `external_dev`
- `external_held_out`

### New Stress Packs

The first frozen `study-v3` pass must add:

- `consolidation_stress`
- `forgetting_stress`

These are the packs that determine whether `study-v3` succeeds.

## Consolidation-Stress Pack Requirements

`consolidation_stress` should include scenarios where:

- repeated weak hints must become one durable belief
- procedures must be inferred from multiple small failures
- lessons emerge only after several partial incidents
- late tasks depend on earlier signals that were never stated as explicit memory writes

Required properties:

- long gaps between cues and later use
- multiple small signals instead of one direct instruction
- at least some cases where simple append-only structured state should remain too literal or too fragmented

Minimum initial size:

- 8 contradiction-style checkpoints
- 8 natural-style checkpoints

## Forgetting-Stress Pack Requirements

`forgetting_stress` should include scenarios where:

- preferences reverse after many turns
- old goals should stop dominating current behavior
- noisy state should stop crowding out newer, relevant state
- contradicted beliefs should be downgraded or removed

Required properties:

- explicit stale state that should become less important
- enough irrelevant side information to make state bloat matter
- at least some cases where no-forgetting variants should retain the wrong emphasis

Minimum initial size:

- 8 contradiction-style checkpoints
- 8 natural-style checkpoints

## Scoring Rules

Use the same scoring stance as late `study-v2`:

- structural scoring for extraction checkpoints
- judge-backed semantic scoring for behavior checkpoints
- explicit logging of parse failures, empty answers, and provider errors

Do not weaken the held-out or stress-pack standards by patching wording after the packs are frozen.

## Primary Success Criteria

`study-v3` counts as successful only if both of these are true on the new stress packs:

1. `brainlayer_full` gains a clearer and repeatable margin over `structured_no_consolidation`
2. removing consolidation and/or forgetting causes a meaningful drop relative to `brainlayer_full`

Suggested threshold for the first freeze:

- at least one primary provider shows a repeatable `+3` or greater advantage for `brainlayer_full` over `structured_no_consolidation` on `consolidation_stress + forgetting_stress`
- `brainlayer_no_consolidation` and/or `brainlayer_no_forgetting` each lose at least `2` points relative to `brainlayer_full` on the same stress packs

## Failure Conditions

`study-v3` should be considered inconclusive or unsuccessful if:

- `consolidation_stress` does not expose a clear weakness in `brainlayer_no_consolidation`
- `forgetting_stress` does not expose a clear weakness in `brainlayer_no_forgetting`
- `brainlayer_full` still fails to separate meaningfully from `structured_no_consolidation`
- the apparent gains remain confined to regression packs rather than the new stress packs

## Freeze Rules

1. Freeze `consolidation_stress` and `forgetting_stress` before tuning against them repeatedly.
2. Do not patch held-out-like wording inside those packs after the first live freeze unless the issue is a clear general parsing or transport bug.
3. Treat the first successful `study-v3-*` bundle as the baseline only after at least one repeat on the primary provider pair.

## Recommended Build Order

1. implement `consolidation_stress`
2. implement `forgetting_stress`
3. run heuristic smoke passes on both packs
4. make targeted consolidation changes
5. make targeted forgetting changes
6. run the primary provider pair live
7. repeat once before freezing the first `study-v3` baseline

## Non-Goals

`study-v3` is not about:

- redesigning `autobio` or `working_state` from scratch
- broad provider/platform expansion
- benchmark wording cleanup as a substitute for harder tasks
- claiming universal BrainLayer superiority before consolidation and forgetting have shown measurable value
