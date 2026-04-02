# BrainLayer Study Protocol

## Goal

Test whether a layered BrainLayer state improves agent continuity, revision, and natural-memory extraction beyond simpler memory baselines.

## Frozen Research Question

Do agents perform better when they maintain a layered cognitive state with working state, episodic traces, beliefs, autobiographical continuity, procedures, consolidation, and forgetting, rather than relying on context only or naive memory writes?

## Primary Hypotheses

1. Full BrainLayer improves overall pass rate across contradiction, natural, and held-out packs.
2. Natural-conversation extraction improves when BrainLayer state is explicit and normalized into canonical slots.
3. BrainLayer gains hold on held-out wording, not just tuned benchmark phrasing.
4. Smaller models benefit from BrainLayer support and runtime normalization, even if absolute scores remain below larger models.

## Secondary Hypotheses

1. Forgetting helps compactness more than raw accuracy.
2. Consolidation matters most on delayed and repeated-signal tasks.
3. Autobiographical and working-state layers matter most on continuity and goal-tracking tasks.

## Frozen Primary Model Set

The primary live study uses the entries defined in `examples/model_matrix.openai.chat.live.json`.

At the time this protocol was written, that file targets:

- `gpt-4o-mini`
- `gpt-4.1-mini`
- `gpt-4o`
- `gpt-4.1`

The study runner snapshots the exact config into each study bundle so later config edits do not rewrite history.

## Frozen Scenario Packs

Run all three packs:

- `standard`
- `hard`
- `held_out`

Use both suites in each pack:

- `contradiction`
- `natural`

## Scoring

- Primary behavior scoring mode: `judge`
- Extraction scoring: structural scoring from exported BrainLayer state
- Primary reporting unit: model-level leaderboard row per pack

## Primary Metrics

- `overall_pass_rate`
- `natural_extraction_passed / natural_extraction_total`
- `natural_behavior_passed / natural_behavior_total`
- `contradiction_passed / contradiction_total`

## Secondary Metrics

- `avg_score`
- `parse_failures`
- `empty_answers`
- `errors`
- `avg_latency_ms`
- `estimated_total_cost_usd`

## Primary Study Command

```bash
python3 scripts/run_study.py \
  --config examples/model_matrix.openai.chat.live.json \
  --label study-v1
```

This freezes and snapshots:

- the protocol
- the matrix config
- one matrix run per scenario pack
- one analysis export per scenario pack
- one aggregate study summary across packs

## Secondary Ablation Phase

Run ablations only after the primary frozen study completes.

Recommended approach:

- keep the main study on the full model set without ablations
- use a smaller config for the ablation phase
- focus ablations on the strongest model and the cheapest viable model

Example:

```bash
python3 scripts/run_study.py \
  --config path/to/smaller-ablation-config.json \
  --with-ablations \
  --label study-ablations-v1
```

## Stop Conditions

Treat the first frozen held-out run as the boundary between benchmark design and study execution.

After that point:

- do not edit held-out scenarios to rescue a result
- do not change score rules unless the change is clearly a bug fix and is documented
- do not change the model set mid-study without starting a new study label

## Decision Rule

Keep only changes that:

- improve at least one primary metric
- do not cause a meaningful held-out regression
- do not create a large reliability or cost regression without a clear reason

## Output Expectations

Each study bundle should contain:

- `study_protocol.md`
- `study_config.json`
- `study_summary.json`
- `study_summary.md`
- `aggregate_leaderboard.csv`
- `pack_summary.csv`
- `pareto_frontier.csv`
- `x_post.txt`
- `matrix_runs/`
- `matrix_analysis/`

## Notes

If a bug in scoring, parsing, or export invalidates a run, fix the bug, start a new study label, and rerun the full frozen protocol instead of patching the old bundle.

## Frozen Baseline Marker

As of April 2, 2026, the current frozen baseline is:

- baseline id: `study-v1-gemini-core`
- bundle: `artifacts/study_runs/20260402T175429Z-study-v1-gemini-core`
- config: `examples/model_matrix.gemini.chat.core.live.json`

Headline results:

- `gemini-2.5-flash`: `43/47`
- `gemini-2.5-flash-lite`: `39/47`

Interpretation:

- `standard` and `hard` are strong enough to treat the architecture as operational.
- `held_out` remains the real generalization check.
- Remaining held-out misses in the phone-briefing wording family are intentionally left unfixed for this baseline to avoid tuning on held-out phrasing.
