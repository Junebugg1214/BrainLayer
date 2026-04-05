# Study-v3 Consolidation Note

## Current Frozen Consolidation Result

As of April 5, 2026, the current `study-v3` consolidation result is the strengthened bounded live replication on the redesigned `consolidation_stress` pack.

Reference bundles:

- Gemini Flash bounded repeat:
  [20260405T024937Z-study-v3-gemini-flash-consolidation-clean-repeat2-repeat1](/Users/marcsaint-jour/Documents/New%20project/artifacts/model_eval_repeat_runs/20260405T024937Z-study-v3-gemini-flash-consolidation-clean-repeat2-repeat1)
- Anthropic Haiku bounded repeat:
  [20260405T024911Z-study-v3-anthropic-haiku-consolidation-clean-repeat2-repeat1](/Users/marcsaint-jour/Documents/New%20project/artifacts/model_eval_repeat_runs/20260405T024911Z-study-v3-anthropic-haiku-consolidation-clean-repeat2-repeat1)
- Anthropic Sonnet bounded scenario repeats:
  - [style](/Users/marcsaint-jour/Documents/New%20project/artifacts/model_eval_repeat_runs/20260405T025121Z-study-v3-anthropic-sonnet-consolidation-style-hint-stack-repeat2-repeat1)
  - [goal](/Users/marcsaint-jour/Documents/New%20project/artifacts/model_eval_repeat_runs/20260405T025122Z-study-v3-anthropic-sonnet-consolidation-goal-hint-stack-repeat2-repeat1)
  - [relationship](/Users/marcsaint-jour/Documents/New%20project/artifacts/model_eval_repeat_runs/20260405T025121Z-study-v3-anthropic-sonnet-consolidation-relationship-hint-stack-repeat2-repeat1)
  - [lesson](/Users/marcsaint-jour/Documents/New%20project/artifacts/model_eval_repeat_runs/20260405T025110Z-study-v3-anthropic-sonnet-consolidation-lesson-hint-stack-repeat2-repeat1)

## Headline Readout

`gemini-2.5-flash`

- `brainlayer_full`: `5/8`
- `brainlayer_no_consolidation`: `3/8`
- `structured_no_consolidation`: `2/8`

`claude-haiku-4.5`

- `brainlayer_full`: `7/8`
- `brainlayer_no_consolidation`: `6/8`
- `structured_no_consolidation`: `6/8`

`claude-sonnet-4.5`

- `brainlayer_full`: `8/8`
- `brainlayer_no_consolidation`: `2/8`
- `structured_no_consolidation`: `2/8`

## Current Claim

The strengthened `consolidation_stress` pack now shows a real live consolidation signal.

The strongest result is Anthropic Sonnet: after the redesign, `brainlayer_full` cleanly separates from both `brainlayer_no_consolidation` and `structured_no_consolidation` at `8/8` versus `2/8`.

Anthropic Haiku shows a smaller positive signal, with `brainlayer_full` at `7/8` and both weaker variants at `6/8`.

Gemini Flash still separates in the expected direction, but that run is heavily confounded by parse failures, so it should be read as supportive rather than clean.

## Boundary

This note does **not** justify claiming that consolidation is now universally or cleanly solved across all providers.

The current honest read is:

- Anthropic Sonnet now gives a strong positive consolidation result
- Anthropic Haiku gives a weak positive consolidation result
- Gemini Flash is directionally positive but noisy because of fenced-JSON parse loss

## Why This Matters

Before the pack redesign, consolidation looked flat on Anthropic because recent episodes were enough to answer the queries without durable abstraction.

After the redesign, the same provider family now shows the intended pattern, especially on Sonnet. That means the earlier flat result was at least partly a pack-design problem, not just evidence that consolidation was useless.
