# Study-v3 Cross-Provider Forgetting Memo

## Scope

This memo summarizes the current live `study-v3` forgetting signal across Gemini and Anthropic on the longer natural-only `forgetting_stress` pack.

## Bundles

- Gemini: [artifacts/matrix_runs/20260404T230141Z-study-v3-gemini-forgetting-natural-live-v2](/Users/marcsaint-jour/Documents/New%20project/artifacts/matrix_runs/20260404T230141Z-study-v3-gemini-forgetting-natural-live-v2)
- Anthropic Sonnet baseline: [artifacts/matrix_runs/20260404T233923Z-study-v3-anthropic-forgetting-natural-live-v1](/Users/marcsaint-jour/Documents/New%20project/artifacts/matrix_runs/20260404T233923Z-study-v3-anthropic-forgetting-natural-live-v1)
- Anthropic Haiku baseline: [artifacts/matrix_runs/20260404T233046Z-study-v3-anthropic-haiku-forgetting-natural-live-v1](/Users/marcsaint-jour/Documents/New%20project/artifacts/matrix_runs/20260404T233046Z-study-v3-anthropic-haiku-forgetting-natural-live-v1)
- Anthropic Haiku clean repeat: [artifacts/natural_eval_repeat_runs/20260405T004441Z-study-v3-anthropic-haiku-forgetting-clean-repeat1](/Users/marcsaint-jour/Documents/New%20project/artifacts/natural_eval_repeat_runs/20260405T004441Z-study-v3-anthropic-haiku-forgetting-clean-repeat1)
- Anthropic Sonnet bounded scenario repeats:
  - [citation](/Users/marcsaint-jour/Documents/New%20project/artifacts/natural_eval_repeat_runs/20260405T014005Z-study-v3-anthropic-sonnet-forgetting-citation-repeat1)
  - [summary goal](/Users/marcsaint-jour/Documents/New%20project/artifacts/natural_eval_repeat_runs/20260405T014324Z-study-v3-anthropic-sonnet-forgetting-summary-goal-crowding-repeat1)
  - [report goal](/Users/marcsaint-jour/Documents/New%20project/artifacts/natural_eval_repeat_runs/20260405T014500Z-study-v3-anthropic-sonnet-forgetting-report-goal-crowding-repeat1)
  - [reasoning goal](/Users/marcsaint-jour/Documents/New%20project/artifacts/natural_eval_repeat_runs/20260405T014637Z-study-v3-anthropic-sonnet-forgetting-reasoning-goal-crowding-repeat1)

## Headline Readout

### Gemini

- `gemini-2.5-flash / brainlayer_full`: `8/8`
- `gemini-2.5-flash / brainlayer_no_forgetting`: `6/8`
- `gemini-2.5-flash / structured_no_consolidation`: `5/8`

- `gemini-2.5-flash-lite / brainlayer_full`: `6/8`
- `gemini-2.5-flash-lite / brainlayer_no_forgetting`: `6/8`
- `gemini-2.5-flash-lite / structured_no_consolidation`: `3/8`

### Anthropic

- `claude-sonnet-4.5 / brainlayer_full`: `8/8`
- `claude-sonnet-4.5 / brainlayer_no_forgetting`: `8/8`
- `claude-sonnet-4.5 / structured_no_consolidation`: `7/8`

- `claude-haiku-4.5 / brainlayer_full`: `6/8` in the clean bounded repeat
- `claude-haiku-4.5 / brainlayer_no_forgetting`: `4/8`
- `claude-haiku-4.5 / structured_no_consolidation`: `5/8`

## Current Read

The current live forgetting signal is real on Gemini `flash`: `brainlayer_full` beats both `brainlayer_no_forgetting` and `structured_no_consolidation` on the longer natural pack.

The Anthropic read is now sharper than it was in the first pass. Haiku shows a positive forgetting split in the bounded repeat, with `brainlayer_full` beating `brainlayer_no_forgetting` by `2` points. Sonnet is now measurable scenario-by-scenario under the bounded runner, but it still does not show a forgetting-specific split: `brainlayer_full` and `brainlayer_no_forgetting` both land at `8/8`.

## Defensible Claim

`study-v3` now has live evidence that forgetting can help on a realistic natural stress pack on Gemini `flash` and Anthropic Haiku, but the effect is not yet universal across provider/model tiers because Anthropic Sonnet still ties `brainlayer_full` and `brainlayer_no_forgetting`.

## Next Step

The next credible move is not to overclaim the forgetting result. It is to treat forgetting as promising but tier-sensitive, and then move to the other open `study-v3` question: whether consolidation can show the same kind of bounded live separation across providers.
