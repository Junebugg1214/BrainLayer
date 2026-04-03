# BrainLayer Study-V2 Protocol

## Status

This file defines the intended frozen protocol for `study-v2`.

It should be treated as the comparison contract for the next phase of work:

- stronger baselines
- harder external tasks
- explicit promotion rules

Implementation should follow this protocol rather than changing the criteria midstream.

## Reference Baseline

All study-v2 results should be compared against the frozen reference baseline:

- baseline id: `study-v1-gemini-core`
- bundle: `artifacts/study_runs/20260402T175429Z-study-v1-gemini-core`
- report: `docs/baseline_report.md`
- config: `examples/model_matrix.gemini.chat.core.live.json`

Headline frozen baseline:

- `gemini-2.5-flash`: `43/47`
- `gemini-2.5-flash-lite`: `39/47`

## Current Post-Patch Freeze

As of April 3, 2026, the current post-patch study-v2 reference set is:

- primary bundle: `artifacts/study_runs/20260403T020152Z-study-v2-gemini-core-v5`
- repeat 1: `artifacts/study_runs/20260403T165058Z-study-v2-gemini-core-v5-repeat1`
- repeat 2: `artifacts/study_runs/20260403T174802Z-study-v2-gemini-core-v5-repeat2`

Aggregate post-patch results:

- `gemini-2.5-flash / brainlayer_full`: `90/95`, `92/95`, `91/95` across the three runs
- `gemini-2.5-flash / structured_no_consolidation`: `86/95`, `84/95`, `89/95`
- `gemini-2.5-flash-lite / brainlayer_full`: `85/95`, `87/95`, `85/95`
- `gemini-2.5-flash-lite / structured_no_consolidation`: `81/95`, `86/95`, `87/95`

Interpretation:

- `gemini-2.5-flash` shows a stable post-patch win for `brainlayer_full`
- `gemini-2.5-flash-lite` is closer to a near-tie, but remains slightly favorable to `brainlayer_full` on average across the three runs

These three bundles should be treated as the frozen post-patch study-v2 baseline until a clearly labeled successor such as `study-v2-*` is intentionally promoted.

## Study-V2 Question

Does BrainLayer still provide a meaningful advantage when compared against stronger non-BrainLayer baselines and evaluated on more external, less authored tasks?

## Model Set

Primary study-v2 models:

- `gemini-2.5-flash`
- `gemini-2.5-flash-lite`

These two models are the required first-pass model set. Additional models may be evaluated later, but they should not block the first frozen study-v2 run.

## Primary Conditions

Study-v2 should compare the following conditions.

### Condition 1: `brainlayer_full`

Full current BrainLayer runtime with:

- working state
- beliefs
- autobiographical state
- procedures
- consolidation
- forgetting

This is the main treatment condition.

### Condition 2: `context_only`

The model only sees the current task input and current turn history.

No durable state retrieval, no state writes, no consolidation.

### Condition 3: `naive_retrieval`

The model can retrieve prior notes/episodes by simple similarity or recency, but there is no explicit layered state and no consolidation.

### Condition 4: `structured_no_consolidation`

The model can read and write structured slots, but there is:

- no consolidation
- no revision policy
- no forgetting

State is explicit but effectively append-only.

### Condition 5: `summary_state`

The model maintains one evolving summary/profile instead of separate:

- working state
- beliefs
- autobiographical state
- procedures

### Optional Condition 6: `retrieval_plus_scratchpad`

Naive retrieval plus a transient task scratchpad, but no durable layered state.

This condition is useful if the team wants a simpler “working memory without BrainLayer” comparison.

## Required Comparison Grid

The first frozen study-v2 pass must include:

- `brainlayer_full`
- `context_only`
- `naive_retrieval`
- `structured_no_consolidation`
- `summary_state`

The optional `retrieval_plus_scratchpad` condition may be added, but it is not required for the first freeze.

## Evaluation Packs

Study-v2 should include both the existing authored packs and new external packs.

### Authored Packs

Keep the current authored packs for continuity:

- `standard`
- `hard`
- `held_out`

These remain the regression-tracking packs.

### External Packs

Add two external packs:

- `external_dev`
- `external_held_out`

These are the packs that determine whether BrainLayer still matters once tasks look less benchmark-authored.

## External Task Families

External tasks must be drawn from these four families.

### Family A: Collaboration Logs

Longer dialogue or collaboration fragments where:

- preferences emerge gradually
- collaboration framing shifts
- goals change after side turns

### Family B: Retrospectives and Failure Notes

Work-style retrospectives or incident-style notes where:

- a reusable lesson must be extracted
- a procedure must be inferred from ordinary language
- a future failure can be prevented if the lesson is retained

### Family C: Multi-Step Research Work

Research-style interaction logs where:

- the active goal shifts over time
- evidence changes the plan
- collaboration stance matters

### Family D: Realistic Contradiction Chains

Longer sequences where a user later corrects:

- style
- role
- goal
- factual assumptions

## External Slice Definition

The first frozen study-v2 pass should use:

- `external_dev`: 16 scenarios
- `external_held_out`: 16 scenarios

Recommended composition:

- 4 scenarios from Family A
- 4 scenarios from Family B
- 4 scenarios from Family C
- 4 scenarios from Family D

The held-out external slice should use different wording, different turn structure, and different surface details than the dev slice while targeting the same underlying capabilities.

## External Task Rules

1. Do not tune directly on `external_held_out`.
2. Keep wording natural rather than schema-shaped.
3. Prefer implicit memory signals over explicit “remember this” phrasing.
4. Preserve side turns, ambiguity, and realistic messiness.
5. If a dev task becomes too familiar, move a fresh example into held-out and relabel the study version.

## Scoring

Primary scoring mode:

- behavior scoring: `judge`
- extraction scoring: structural scoring from exported state

All study-v2 comparisons should use the same scoring mode across all conditions.

## Primary Metrics

Study-v2 should report these as the main decision metrics:

- `overall_pass_rate`
- `external_dev_overall_pass_rate`
- `external_held_out_overall_pass_rate`
- `external_held_out_natural_extraction`
- `external_held_out_natural_behavior`
- `contradiction_passed / contradiction_total`

## Secondary Metrics

Study-v2 should also report:

- `avg_score`
- `parse_failures`
- `empty_answers`
- `errors`
- `avg_latency_ms`
- `estimated_total_cost_usd`

## Required Win Conditions

Study-v2 should only be called a meaningful improvement if all of the following are true.

### Win Condition A: Stronger-Baseline Superiority

`brainlayer_full` must beat:

- `context_only`
- `naive_retrieval`
- `structured_no_consolidation`
- `summary_state`

on `external_held_out_overall_pass_rate` for at least one primary model.

### Win Condition B: External Generalization

For at least one primary model, `brainlayer_full` must achieve:

- `external_held_out_overall_pass_rate >= 0.75`
- `external_held_out_natural_extraction >= 0.70`
- `external_held_out_natural_behavior >= 0.70`

### Win Condition C: No Authored Collapse

Relative to `study-v1-gemini-core`, `brainlayer_full` must not regress by more than:

- `3` total passes on the authored aggregate

for either primary model.

### Win Condition D: Practical Cost Boundary

At least one primary model must remain a usable “best value” option, meaning:

- no catastrophic reliability failure
- no large latency blow-up relative to its own study-v1 baseline

## Failure Conditions

Study-v2 should be treated as mixed or negative if any of the following happen.

1. BrainLayer only beats weak baselines, but not `structured_no_consolidation` or `summary_state`.
2. External held-out performance falls below the thresholds above.
3. Improvements come mainly from wording-specific normalization patches rather than broader behavioral gains.
4. Reliability or cost regressions erase the practical value of the improvement.

## Ablation Scope

Do not run all ablations in the first protocol freeze.

The first ablation phase should only be run after the main study-v2 comparison grid is stable.

Priority ablations:

- no consolidation
- no forgetting
- no autobiographical state
- no working state

Run them on:

- the best quality model
- the best value model

## Deliverables

The first frozen study-v2 run should produce:

1. `study_v2_protocol.md`
2. `external_task_spec.md`
3. one results bundle for the full comparison grid
4. one external held-out report
5. one summary of what BrainLayer beats and what it does not beat

## Implementation Order

1. Implement the stronger baseline conditions.
2. Build the external task packs.
3. Run smoke tests on a small slice.
4. Freeze the study-v2 comparison config.
5. Run the first full study-v2 pass.
6. Only then run ablations.

## Promotion Rule

A result should only be promoted from experiment to new baseline if:

1. it satisfies the win conditions above
2. it is run under a fresh `study-v2-*` label
3. its external held-out slice was not tuned during implementation
4. the result is documented in a new baseline report rather than replacing `study-v1-gemini-core`

## Bottom Line

`study-v1` showed that BrainLayer is real.

`study-v2` must show that BrainLayer still matters when:

- the baselines are stronger
- the tasks are more external
- the language is messier
- and the comparison is fairer
