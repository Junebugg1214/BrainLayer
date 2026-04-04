# BrainLayer Experiments

## Goal

Test whether a structured BrainLayer improves agent behavior beyond plain context windows and naive memory systems.

## Baselines

Use at least three conditions:

1. `context-only`
   The agent only sees the current conversation or current task context.
2. `naive-memory`
   The agent can retrieve past notes or vector-matched snippets.
3. `brainlayer`
   The agent has working state, episodes, beliefs, autobiographical state, procedures, and salience-driven consolidation.

## Primary Hypotheses

### H1: Goal-Conditioned State Beats Similarity Retrieval

Claim:

Retrieval should be based on current goals and open questions, not just semantic similarity.

Expected gain:

- fewer irrelevant recalls
- better action selection

### H2: Consolidation Beats Append-Only Logging

Claim:

Agents perform better when repeated episodes are compressed into revisable beliefs.

Expected gain:

- less context bloat
- more stable behavior

### H3: Forgetting Can Improve Performance

Claim:

A controlled forgetting policy will outperform total retention on long-horizon tasks.

Expected gain:

- less distraction
- lower retrieval noise

### H4: Autobiographical State Improves Continuity

Claim:

A dedicated continuity layer will improve consistency across long interactions and recurring collaboration.

Expected gain:

- fewer broken commitments
- better user adaptation

### H5: Procedural Memory Improves Repeated Task Performance

Claim:

Agents that learn reusable procedures from successful episodes will improve faster on recurring tasks.

Expected gain:

- reduced trial-and-error
- more reliable execution

## Candidate Task Suite

### Task Family A: Preference Learning

The agent interacts with a user whose preferences emerge over time.

Measure:

- how quickly the agent adapts
- whether it stays consistent later

### Task Family B: Long-Horizon Project Work

The agent works through a multi-step project with prior failures, constraints, and changing goals.

Measure:

- whether it reuses lessons
- whether it avoids repeated mistakes

### Task Family C: Relationship Continuity

The agent must maintain role continuity, prior commitments, and collaboration style across sessions.

Measure:

- consistency of remembered commitments
- accuracy of collaboration framing

### Task Family D: Contradiction and Revision

The environment introduces evidence that contradicts previous beliefs.

Measure:

- whether the agent revises beliefs gracefully
- whether historical evidence remains inspectable

The next implementation step after the deterministic benchmark harness is to run these contradiction-heavy cases through the live BrainLayer runtime itself, not just the rule-based benchmark agents. That means evaluating full turns where a model:

- reads retrieved BrainLayer context
- emits a user-facing answer
- proposes structured memory updates
- gets scored later on whether revision and continuity still hold

### Task Family E: Skill Acquisition

The agent repeats similar workflows and should learn better procedures over time.

Measure:

- speedup on later attempts
- reduction in avoidable mistakes

## Metrics

Use both task and state metrics.

### Behavior Metrics

- task success rate
- steps to completion
- repeated error rate
- preference consistency
- planning quality

### State Metrics

- retrieval precision
- irrelevant recall rate
- belief revision accuracy
- memory growth over time
- consolidation rate

## Ablations

Test the contribution of each layer by removing one at a time:

- no working state
- no episodic memory
- no semantic beliefs
- no autobiographical state
- no procedural memory
- no forgetting

## Minimal Prototype Order

Build in this order:

1. working state
2. episodic store
3. semantic belief consolidation
4. simple forgetting policy
5. autobiographical summaries
6. procedural extraction

## First Experiment Recommendations

Start with three simple experiments:

1. Preference persistence across 20 to 50 interactions
2. Repeated task execution with lessons from prior failures
3. Contradictory evidence that forces belief revision

## Longer-Horizon Evaluation

After the seed experiments, add longer-horizon scenarios with:

- multiple noisy turns between signal and recall
- more than one query checkpoint inside the same scenario
- late contradictions that require belief revision
- repeated weak hints that must be consolidated before they help

These longer runs are where forgetting and autobiographical continuity should start to matter more clearly.

The repo now includes an initial contradiction-focused runtime suite for this layer in `scripts/run_model_evals.py`, covering:

- preference revision
- goal replacement
- relationship reframing
- hint consolidation followed by explicit correction

The next realism step is a natural-conversation suite where signals are implicit in ordinary dialogue instead of explicit `Record ...` prompts. The repo now includes that layer in `scripts/run_natural_model_evals.py`, with separate scoring for:

- extraction accuracy: whether the right BrainLayer state update was inferred
- behavior accuracy: whether later answers reflect that inferred state

Those runtime evals now use judge-backed semantic scoring for behavior checkpoints by default, while extraction checkpoints are scored structurally from the exported BrainLayer state. That makes the live suites less brittle than exact string matching while still preserving inspectable, deterministic scoring paths when needed.

The next comparison layer is a matrix runner in `scripts/run_model_matrix.py`, which executes both suites across multiple model/provider configs and exports:

- case-level CSV/JSON results
- per-case artifact bundles with prompt, retrieval, raw output, judge decision, and exported state
- per-suite summaries
- a cross-suite leaderboard
- score metadata such as `score`, `score_method`, and average score
- estimated cost columns when entry pricing is configured
- append-only history files for tracking progress over time
- an X-ready summary post for sharing results

The next reporting layer is a matrix-history analyzer in `scripts/analyze_matrix_history.py`, which reads `matrix_history.jsonl` and turns a selected run into:

- a publication-friendly Markdown/JSON report
- a Pareto-style cost/quality frontier
- a compact leaderboard and suite summary export
- a simple cost-vs-quality SVG for quick sharing

## Frozen Baseline

The current frozen baseline is `study-v1-gemini-core` at `artifacts/study_runs/20260402T175429Z-study-v1-gemini-core`, using `examples/model_matrix.gemini.chat.core.live.json`.

Headline scores:

- `gemini-2.5-flash`: `43/47`
- `gemini-2.5-flash-lite`: `39/47`

This baseline is intentionally frozen before mapping held-out phone-briefing wording like `punchy` onto `brief`, so the remaining held-out misses remain visible as a real generalization boundary rather than a tuned-away benchmark artifact.

The repo also includes a ready-to-run OpenAI live config in `examples/model_matrix.openai.live.json` with priced GPT-5.1, GPT-5 mini, and GPT-5 nano entries.

The repo now also includes a hard-mode eval pack across contradiction and natural suites. It is designed to stress:

- delayed recall after noisy intervening turns
- long-horizon revisions instead of one-step corrections
- implicit preference and collaboration updates under distraction
- procedure formation from repeated hints rather than one direct instruction

There is also a held-out generalization pack across both suites. It targets the same underlying BrainLayer capabilities with fresh scenario wording and alternate turn structure so you can check whether a change generalizes beyond the tuned benchmark text.

Those packs are available through the shared `scenario_pack` switch in the model eval, natural eval, and matrix CLIs with values `standard`, `hard`, `held_out`, or `all`.

For the actual frozen study workflow, use `scripts/run_study.py`. It snapshots the protocol and config, runs `standard`, `hard`, and `held_out` as separate matrix runs, exports a per-pack analysis for each, and writes one aggregate study summary bundle under `artifacts/study_runs/`.

## Frozen Study-V2 Post-Patch Baseline

The current post-patch study-v2 reference set is:

- `artifacts/study_runs/20260403T020152Z-study-v2-gemini-core-v5`
- `artifacts/study_runs/20260403T165058Z-study-v2-gemini-core-v5-repeat1`
- `artifacts/study_runs/20260403T174802Z-study-v2-gemini-core-v5-repeat2`

Across those three runs:

- `gemini-2.5-flash / brainlayer_full` scored `90/95`, `92/95`, and `91/95`
- `gemini-2.5-flash / structured_no_consolidation` scored `86/95`, `84/95`, and `89/95`
- `gemini-2.5-flash-lite / brainlayer_full` scored `85/95`, `87/95`, and `85/95`
- `gemini-2.5-flash-lite / structured_no_consolidation` scored `81/95`, `86/95`, and `87/95`

The current read is:

- strong and stable post-patch improvement for `gemini-2.5-flash`
- a smaller, noisier margin for `gemini-2.5-flash-lite`

Until a new `study-v2-*` result is explicitly promoted, treat these three bundles together as the frozen post-patch study-v2 baseline.

## Frozen Study-V2 Component-Attribution Baseline

The current `study-v2` component-attribution reference set is the first five-pack live ablation sweep:

- `artifacts/matrix_runs/20260403T232157Z-study-v2-gemini-core-ablations-standard-v1`
- `artifacts/matrix_runs/20260403T234052Z-study-v2-gemini-core-ablations-hard-v1`
- `artifacts/matrix_runs/20260403T235134Z-study-v2-gemini-core-ablations-heldout-v1`
- `artifacts/matrix_runs/20260404T002614Z-study-v2-gemini-core-ablations-external-dev-v1`
- `artifacts/matrix_runs/20260404T004713Z-study-v2-gemini-core-ablations-external-heldout-v1`

See [docs/study_v2_ablation_report.md](/Users/marcsaint-jour/Documents/New%20project/docs/study_v2_ablation_report.md) for the full writeup.

Aggregate totals:

- `gemini-2.5-flash / brainlayer_full`: `91/95`
- `gemini-2.5-flash / brainlayer_no_consolidation`: `90/95`
- `gemini-2.5-flash / brainlayer_no_forgetting`: `92/95`
- `gemini-2.5-flash / brainlayer_no_autobio`: `79/95`
- `gemini-2.5-flash / brainlayer_no_working_state`: `78/95`

- `gemini-2.5-flash-lite / brainlayer_full`: `87/95`
- `gemini-2.5-flash-lite / brainlayer_no_consolidation`: `87/95`
- `gemini-2.5-flash-lite / brainlayer_no_forgetting`: `87/95`
- `gemini-2.5-flash-lite / brainlayer_no_autobio`: `75/95`
- `gemini-2.5-flash-lite / brainlayer_no_working_state`: `77/95`

Current read:

- `autobio` and `working_state` matter a lot
- `consolidation` is weak on this frozen setup
- `forgetting` looks neutral on this frozen setup

## Current Study-V2 Cross-Provider Replication

The current cross-provider replication note for `study-v2` is the frozen Anthropic sweep recorded in [docs/study_v2_cross_provider_note.md](/Users/marcsaint-jour/Documents/New%20project/docs/study_v2_cross_provider_note.md).

Current read:

- the strongest Gemini component pattern reproduces on Anthropic
- `autobio` and `working_state` still matter a lot
- `consolidation` remains weak and `forgetting` remains mixed to near-neutral
- the broader `brainlayer_full` advantage over simpler structured state is not yet universal across providers

## What To Log

For every run, log:

- retrieved items
- selected plan
- actions taken
- outcome
- state updates
- promoted beliefs
- forgotten items

## Decision Rule

Keep only changes that measurably improve performance on at least one target behavior without causing a large regression elsewhere.
