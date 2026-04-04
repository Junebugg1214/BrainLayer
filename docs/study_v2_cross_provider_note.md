## Study-V2 Cross-Provider Note

As of April 4, 2026, the current cross-provider `study-v2` replication note is the frozen Anthropic ablation sweep run against the same 95-case setup used for the Gemini component-attribution baseline.

Config:

- `examples/model_matrix.anthropic.core.live.json`

Sweep bundles:

- `artifacts/matrix_runs/20260404T022325Z-study-v2-anthropic-ablations-standard-smoke-v2`
- `artifacts/matrix_runs/20260404T025654Z-study-v2-anthropic-ablations-hard-v1`
- `artifacts/matrix_runs/20260404T031608Z-study-v2-anthropic-ablations-heldout-v1`
- `artifacts/matrix_runs/20260404T151122Z-study-v2-anthropic-ablations-external-dev-v1`
- `artifacts/matrix_runs/20260404T154745Z-study-v2-anthropic-ablations-external-heldout-contradiction-v1`
- `artifacts/matrix_runs/20260404T160413Z-study-v2-anthropic-ablations-external-heldout-natural-v1`

Aggregate totals:

- `claude-sonnet-4.5 / brainlayer_full`: `90/95`
- `claude-sonnet-4.5 / brainlayer_no_consolidation`: `90/95`
- `claude-sonnet-4.5 / structured_no_consolidation`: `90/95`
- `claude-sonnet-4.5 / brainlayer_no_forgetting`: `89/95`
- `claude-sonnet-4.5 / brainlayer_no_working_state`: `79/95`
- `claude-sonnet-4.5 / brainlayer_no_autobio`: `78/95`

- `claude-haiku-4.5 / brainlayer_no_consolidation`: `88/95`
- `claude-haiku-4.5 / brainlayer_no_forgetting`: `87/95`
- `claude-haiku-4.5 / structured_no_consolidation`: `87/95`
- `claude-haiku-4.5 / brainlayer_full`: `86/95`
- `claude-haiku-4.5 / brainlayer_no_autobio`: `78/95`
- `claude-haiku-4.5 / brainlayer_no_working_state`: `77/95`

Current read:

- the strongest Gemini component pattern reproduces on Anthropic: autobiographical state and working state matter a lot
- consolidation is weak again on this frozen setup
- forgetting remains mixed to near-neutral on this frozen setup
- the broader BrainLayer win is not universal across providers yet: on Sonnet, `brainlayer_full` ties the simpler structured baselines; on Haiku, it trails slightly

Interpretation:

This Anthropic sweep should be treated as the current cross-provider replication note for `study-v2`. It increases confidence in the component-attribution story, especially around `autobio` and `working_state`, while also narrowing the claim: the full BrainLayer stack does not yet show a clear universal advantage over simpler structured state across every provider and model size.
