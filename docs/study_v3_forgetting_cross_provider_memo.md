# Study-v3 Cross-Provider Forgetting Memo

## Scope

This memo summarizes the current live `study-v3` forgetting signal across Gemini and Anthropic on the longer natural-only `forgetting_stress` pack.

## Bundles

- Gemini: [artifacts/matrix_runs/20260404T230141Z-study-v3-gemini-forgetting-natural-live-v2](/Users/marcsaint-jour/Documents/New%20project/artifacts/matrix_runs/20260404T230141Z-study-v3-gemini-forgetting-natural-live-v2)
- Anthropic Sonnet: [artifacts/matrix_runs/20260404T233923Z-study-v3-anthropic-forgetting-natural-live-v1](/Users/marcsaint-jour/Documents/New%20project/artifacts/matrix_runs/20260404T233923Z-study-v3-anthropic-forgetting-natural-live-v1)
- Anthropic Haiku: [artifacts/matrix_runs/20260404T233046Z-study-v3-anthropic-haiku-forgetting-natural-live-v1](/Users/marcsaint-jour/Documents/New%20project/artifacts/matrix_runs/20260404T233046Z-study-v3-anthropic-haiku-forgetting-natural-live-v1)

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

- `claude-haiku-4.5 / brainlayer_full`: `5/8`
- `claude-haiku-4.5 / brainlayer_no_forgetting`: `5/8`
- `claude-haiku-4.5 / structured_no_consolidation`: `4/8`

## Current Read

The current live forgetting signal is real on Gemini `flash`: `brainlayer_full` beats both `brainlayer_no_forgetting` and `structured_no_consolidation` on the longer natural pack.

That forgetting-specific separation does not yet cleanly replicate on Anthropic. Sonnet matches `brainlayer_full` and `brainlayer_no_forgetting` at `8/8`, while Haiku stays noisy and near-flat between those two conditions.

## Defensible Claim

`study-v3` now has first live evidence that forgetting can help on a realistic natural stress pack, but that effect is not yet cross-provider stable.

## Next Step

The next credible move is not to overclaim the forgetting result. It is to keep the longer natural forgetting pack, repeat the Anthropic comparison, and only call forgetting replicated once `brainlayer_full` separates from `brainlayer_no_forgetting` outside Gemini.
