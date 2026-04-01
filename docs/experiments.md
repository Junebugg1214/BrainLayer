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
- per-suite summaries
- a cross-suite leaderboard
- score metadata such as `score`, `score_method`, and average score
- append-only history files for tracking progress over time
- an X-ready summary post for sharing results

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
