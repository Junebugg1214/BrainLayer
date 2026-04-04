# Study-v3 Forgetting Note

## Current Frozen Forgetting Result

As of April 4, 2026, the current `study-v3` forgetting result is the Gemini natural-only live run:

- bundle: [artifacts/matrix_runs/20260404T230141Z-study-v3-gemini-forgetting-natural-live-v2](/Users/marcsaint-jour/Documents/New%20project/artifacts/matrix_runs/20260404T230141Z-study-v3-gemini-forgetting-natural-live-v2)
- pack: `forgetting_stress`
- suite: `natural`
- runtime profile: `study_v2`

## Headline Readout

- `gemini-2.5-flash / brainlayer_full`: `8/8`
- `gemini-2.5-flash / brainlayer_no_forgetting`: `6/8`
- `gemini-2.5-flash / structured_no_consolidation`: `5/8`
- `gemini-2.5-flash-lite / brainlayer_full`: `6/8`
- `gemini-2.5-flash-lite / brainlayer_no_forgetting`: `6/8`
- `gemini-2.5-flash-lite / structured_no_consolidation`: `3/8`

## Current Claim

This is the first clean live evidence in `study-v3` that forgetting can help on a realistic stress pack.

The result is strongest on `gemini-2.5-flash`, where `brainlayer_full` outperforms both `brainlayer_no_forgetting` and `structured_no_consolidation` on the longer natural forgetting sequences.

The result is weaker on `gemini-2.5-flash-lite`, where `brainlayer_full` still beats `structured_no_consolidation` but does not yet separate from `brainlayer_no_forgetting`.

## Boundary

This note freezes the current Gemini forgetting result only. Cross-provider comparison now exists, but the forgetting-specific signal is still mixed outside Gemini and should be read through the separate `study_v3` cross-provider forgetting memo rather than treated as a replicated result yet.
