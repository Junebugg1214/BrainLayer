# BrainLayer Study-V3 Stress Report

## Scope

This report combines the current live `study-v3` stress results for:

- `forgetting_stress`
- `consolidation_stress`

It should be read alongside:

- [study_v3_protocol.md](/Users/marcsaint-jour/Documents/New%20project/docs/study_v3_protocol.md)
- [study_v3_forgetting_note.md](/Users/marcsaint-jour/Documents/New%20project/docs/study_v3_forgetting_note.md)
- [study_v3_forgetting_cross_provider_memo.md](/Users/marcsaint-jour/Documents/New%20project/docs/study_v3_forgetting_cross_provider_memo.md)

## Reference Bundles

### Forgetting

- Gemini natural-only live run:
  [20260404T230141Z-study-v3-gemini-forgetting-natural-live-v2](/Users/marcsaint-jour/Documents/New%20project/artifacts/matrix_runs/20260404T230141Z-study-v3-gemini-forgetting-natural-live-v2)
- Anthropic Haiku bounded repeat:
  [20260405T004441Z-study-v3-anthropic-haiku-forgetting-clean-repeat1](/Users/marcsaint-jour/Documents/New%20project/artifacts/natural_eval_repeat_runs/20260405T004441Z-study-v3-anthropic-haiku-forgetting-clean-repeat1)
- Anthropic Sonnet bounded scenario repeats:
  - [citation](/Users/marcsaint-jour/Documents/New%20project/artifacts/natural_eval_repeat_runs/20260405T014005Z-study-v3-anthropic-sonnet-forgetting-citation-repeat1)
  - [summary goal](/Users/marcsaint-jour/Documents/New%20project/artifacts/natural_eval_repeat_runs/20260405T014324Z-study-v3-anthropic-sonnet-forgetting-summary-goal-crowding-repeat1)
  - [report goal](/Users/marcsaint-jour/Documents/New%20project/artifacts/natural_eval_repeat_runs/20260405T014500Z-study-v3-anthropic-sonnet-forgetting-report-goal-crowding-repeat1)
  - [reasoning goal](/Users/marcsaint-jour/Documents/New%20project/artifacts/natural_eval_repeat_runs/20260405T014637Z-study-v3-anthropic-sonnet-forgetting-reasoning-goal-crowding-repeat1)

### Consolidation

- Gemini Flash bounded repeat:
  [20260405T020411Z-study-v3-gemini-flash-consolidation-clean-repeat1](/Users/marcsaint-jour/Documents/New%20project/artifacts/model_eval_repeat_runs/20260405T020411Z-study-v3-gemini-flash-consolidation-clean-repeat1)
- Anthropic Haiku bounded repeat:
  [20260405T022323Z-study-v3-anthropic-haiku-consolidation-clean-repeat1](/Users/marcsaint-jour/Documents/New%20project/artifacts/model_eval_repeat_runs/20260405T022323Z-study-v3-anthropic-haiku-consolidation-clean-repeat1)
- Anthropic Sonnet bounded scenario repeats:
  - [style hint](/Users/marcsaint-jour/Documents/New%20project/artifacts/model_eval_repeat_runs/20260405T022508Z-study-v3-anthropic-sonnet-consolidation-style-hint-stack-repeat1)
  - [goal hint](/Users/marcsaint-jour/Documents/New%20project/artifacts/model_eval_repeat_runs/20260405T022549Z-study-v3-anthropic-sonnet-consolidation-goal-hint-stack-repeat1)
  - [relationship hint](/Users/marcsaint-jour/Documents/New%20project/artifacts/model_eval_repeat_runs/20260405T022627Z-study-v3-anthropic-sonnet-consolidation-relationship-hint-stack-repeat1)
  - [lesson hint](/Users/marcsaint-jour/Documents/New%20project/artifacts/model_eval_repeat_runs/20260405T022705Z-study-v3-anthropic-sonnet-consolidation-lesson-hint-stack-repeat1)

## Headline Readout

### Forgetting

`gemini-2.5-flash`

- `brainlayer_full`: `8/8`
- `brainlayer_no_forgetting`: `6/8`
- `structured_no_consolidation`: `5/8`

`claude-haiku-4.5`

- `brainlayer_full`: `6/8`
- `brainlayer_no_forgetting`: `4/8`
- `structured_no_consolidation`: `5/8`

`claude-sonnet-4.5`

- `brainlayer_full`: `8/8`
- `brainlayer_no_forgetting`: `8/8`
- `structured_no_consolidation`: `7/8`

### Consolidation

`gemini-2.5-flash`

- `brainlayer_full`: `8/8`
- `brainlayer_no_consolidation`: `7/8`
- `structured_no_consolidation`: `7/8`

`claude-haiku-4.5`

- `brainlayer_full`: `8/8`
- `brainlayer_no_consolidation`: `8/8`
- `structured_no_consolidation`: `8/8`

`claude-sonnet-4.5`

- `brainlayer_full`: `8/8`
- `brainlayer_no_consolidation`: `8/8`
- `structured_no_consolidation`: `8/8`

## Exact Claim

The defensible `study-v3` claim right now is:

`Forgetting now has a real live stress-pack signal on gemini-2.5-flash and a positive bounded repeat on claude-haiku-4.5, but it is not yet universal because claude-sonnet-4.5 ties brainlayer_full and brainlayer_no_forgetting. Consolidation currently shows a bounded live positive signal on gemini-2.5-flash, but not yet on Anthropic Haiku or Sonnet.`

## What The Stress Results Actually Say

### Forgetting

The forgetting result is no longer just a deterministic smoke artifact.

On Gemini Flash, `brainlayer_full` now beats both `brainlayer_no_forgetting` and `structured_no_consolidation` on the longer natural-only pack. On Anthropic Haiku, the bounded repeat also favors `brainlayer_full` over `brainlayer_no_forgetting`, though the margin is smaller and noisier. On Anthropic Sonnet, the current pack is measurable and stable under the bounded runner, but forgetting still does not create separation: both `brainlayer_full` and `brainlayer_no_forgetting` land at `8/8`.

### Consolidation

Consolidation now has a bounded live positive signal on Gemini Flash, but it is narrow. The current margin is `8/8` versus `7/8`, and the misses in the weaker variants are partly parser-boundary failures rather than clean semantic misses. On Anthropic Haiku and Sonnet, the current bounded contradiction replication is flat: all three runtimes land at `8/8`.

## Boundary

This report does **not** justify claiming that forgetting or consolidation are now universally useful across providers or model tiers.

The current read is more specific:

- `working_state` and `autobio` already looked broadly useful in `study-v2`
- `forgetting` is now promising and partially replicated
- `consolidation` has one live positive provider result so far, but not a replicated cross-provider effect

## Recommended Next Step

Use this report as the current `study-v3` stress reference.

From here:

1. inspect the flat Anthropic cases against the Gemini wins
2. decide whether the next patch should increase retrieval pressure, delay, or ambiguity
3. rerun only the stress packs after a narrow change rather than jumping back to a full-suite phase
