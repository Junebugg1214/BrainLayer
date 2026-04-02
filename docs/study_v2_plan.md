# BrainLayer Study-V2 Plan

## Purpose

`study-v2` exists to make the BrainLayer result harder to dismiss.

`study-v1-gemini-core` showed that the architecture is already strong on authored and hard delayed/noisy packs, but it also showed the current limit clearly: held-out semantic generalization is still weaker than the authored benchmark suggests.

So `study-v2` should not focus on more infrastructure work or more benchmark polishing. It should focus on stronger baselines and harder external tasks.

## Study-V2 Question

Does BrainLayer still provide a meaningful advantage when compared against stronger non-BrainLayer baselines and evaluated on messier, less authored, more external tasks?

## Why Study-V2 Matters

`study-v1` established that BrainLayer is a credible systems result.

`study-v2` needs to establish something stronger:

1. BrainLayer beats more than weak baselines.
2. BrainLayer helps on tasks that were not designed around the current benchmark.
3. The gains hold when language is noisier, goals shift naturally, and state updates are less clean.

If `study-v2` succeeds, the project becomes more than “a good benchmarked memory prototype.” It becomes evidence that layered cognitive state is a necessary design pattern for agents that must generalize over time.

## Frozen Reference Point

The reference baseline for all `study-v2` comparisons is:

- baseline id: `study-v1-gemini-core`
- bundle: `artifacts/study_runs/20260402T175429Z-study-v1-gemini-core`
- report: `docs/baseline_report.md`

Study-v2 should compare against that frozen bundle rather than rewriting the baseline.

## Study-V2 Design Principles

1. No silent upgrades.
   Any new runtime, prompt, normalization rule, or retrieval change must run under a new `study-v2-*` label.

2. Harder baselines before harder claims.
   BrainLayer should be compared against stronger alternatives before we make stronger statements about its necessity.

3. External realism over benchmark comfort.
   New tasks should look less like authored eval prompts and more like messy real interaction traces, work logs, and project memory demands.

4. Protect the held-out boundary.
   If an external task source becomes part of tuning, move a fresh slice into a new held-out set.

## New Baselines

Study-v2 should add at least four comparison conditions beyond the current full BrainLayer runtime.

### Baseline A: Context-Only

The agent only sees the current task input and current turn history.

Purpose:

- establish the no-memory floor
- measure how much continuity comes from raw context alone

### Baseline B: Naive Retrieval Memory

The agent can retrieve prior notes, episodes, or text snippets by basic similarity, but there is no explicit layered state and no consolidation.

Purpose:

- test whether “memory as retrieval” is enough
- isolate the BrainLayer gain beyond simple recall

### Baseline C: Structured Memory Without Consolidation

The agent can write structured state slots, but there is no consolidation, revision policy, or forgetting. Memory is explicit but mostly append-only.

Purpose:

- test whether structure alone is enough
- isolate the value of consolidation and state revision

### Baseline D: Summary-State Baseline

The agent maintains a single evolving summary or profile instead of separate working state, beliefs, autobiographical state, and procedures.

Purpose:

- test whether the BrainLayer gain is really about layered separation
- compare layered state against the simpler “one summary state” alternative

### Optional Baseline E: Retrieval + Scratchpad

The agent gets naive retrieval plus a transient scratchpad for the current task, but no durable layered state.

Purpose:

- test whether working-state gains can be approximated by a simpler short-lived mechanism

## Harder External Task Families

Study-v2 should add tasks that were not authored around the current benchmark wording.

### External Family 1: Long Email or Chat Collaboration Logs

Use longer conversation fragments where preferences, roles, and goals emerge implicitly over time.

Target behaviors:

- preference extraction
- role continuity
- goal tracking under distraction
- revision after later clarification

### External Family 2: Project Retrospectives and Failure Notes

Use retrospective or project-update style text where lessons and procedures must be inferred from ordinary work language.

Target behaviors:

- procedural extraction
- lesson reuse
- failure avoidance
- trigger normalization from realistic phrasing

### External Family 3: Multi-Step Research Collaboration

Use research-style exchanges with evolving framing, plans, evidence, and changed priorities.

Target behaviors:

- collaboration continuity
- planning updates
- revision of active goals
- distinction between current goals and stable preferences

### External Family 4: Realistic Contradiction Chains

Use longer sequences where a user gradually changes or corrects:

- response style
- collaboration framing
- project goal
- key factual assumptions

Target behaviors:

- graceful revision
- historical evidence retention
- reduced stale-state bleedthrough

## External Task Sourcing Rules

Study-v2 should use a split between:

- `development external tasks`
- `held-out external tasks`

Rules:

1. Never tune directly on the held-out external slice.
2. Keep task wording natural rather than canonicalized to the current schema.
3. Prefer tasks where the signal is implicit instead of explicitly saying “remember this.”
4. Preserve messiness: real phrasing, partial corrections, and side turns are desirable.

## Evaluation Changes

Study-v2 should keep the current authored packs, but they should no longer be the whole story.

Recommended study-v2 evaluation mix:

1. Existing authored packs:
   - `standard`
   - `hard`
   - `held_out`

2. New external packs:
   - `external_dev`
   - `external_held_out`

The authored packs remain useful for continuity and regression tracking. The external packs determine whether the result still matters once the tasks feel more real.

## Primary Success Criteria

Study-v2 should count as a real improvement only if all of the following are true:

1. BrainLayer still beats the stronger baselines on overall pass rate.
2. BrainLayer still wins on natural extraction and later behavior on external tasks.
3. Gains are not limited to one model or one authored pack.
4. The held-out external set does not collapse relative to the authored sets.

## Secondary Success Criteria

1. BrainLayer keeps its contradiction and revision strength.
2. Cost and latency remain within a practical range for at least one “best value” model.
3. Improvements can be explained by specific BrainLayer components rather than vague prompt drift.

## Failure Conditions

Study-v2 should be treated as a negative or mixed result if:

1. BrainLayer only beats weak baselines but not structured or summary-state baselines.
2. External held-out tasks erase most of the authored benchmark gains.
3. Gains depend mainly on benchmark-specific normalization rather than broader behavioral improvement.
4. Performance gains come with unacceptable reliability or cost regressions.

## Model Plan

Keep the primary model set narrow at first.

Recommended study-v2 primary models:

- `gemini-2.5-flash`
- `gemini-2.5-flash-lite`

Optional secondary comparison:

- add one stronger/slower model only after the external task pipeline is stable

This keeps the first study-v2 pass focused on research quality rather than provider sprawl.

## Ablation Plan

Do not begin with full ablations across every baseline and every model.

Recommended order:

1. full BrainLayer vs stronger baselines on both primary models
2. once the external task result stabilizes, run component ablations on:
   - the best quality model
   - the best value model

Priority ablations:

- no consolidation
- no forgetting
- no autobiographical state
- no working state

## Deliverables

Study-v2 should produce:

1. a frozen `study-v2` protocol
2. one external-task specification doc
3. one baseline-comparison results bundle
4. one external held-out report
5. one concise summary of what BrainLayer beats and what it does not beat

## Recommended Implementation Order

1. Write the `study-v2` protocol.
2. Implement the stronger baselines.
3. Add external task packs.
4. Run small smoke comparisons.
5. Freeze the study-v2 protocol.
6. Run the first full study-v2 pass.

## Immediate Next Build

The best next artifact after this plan is:

`docs/study_v2_protocol.md`

That file should specify:

- exact baseline conditions
- exact external task slices
- model set
- metrics
- promotion rules
- what counts as a valid improvement over `study-v1-gemini-core`

## Bottom Line

`study-v1` answered: “Is BrainLayer a real systems result?”

`study-v2` must answer: “Does BrainLayer still matter when the comparison is fairer and the tasks are more real?”
