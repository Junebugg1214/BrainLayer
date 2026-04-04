# BrainLayer Study-V3 Plan

## Purpose

`study-v3` is the next phase after the frozen `study-v2` cross-provider result.

Its purpose is not to add more general scaffolding. Its purpose is to attack the specific weakness exposed by `study-v2`:

- `autobio` and `working_state` already look useful
- `consolidation` does not yet clearly matter
- `forgetting` does not yet clearly matter
- `brainlayer_full` does not yet beat simpler structured state consistently enough across providers

So `study-v3` should focus on making consolidation and forgetting earn their keep.

## Core Question

`Can better consolidation and forgetting produce a clear advantage on long-horizon, noisy, revision-heavy tasks where simpler structured state should start to fail?`

## Primary Hypotheses

### Hypothesis 1: Consolidation Helps Under Delay

If the system must infer durable lessons, procedures, or stable beliefs from repeated weak signals across many turns, a stronger consolidation policy should outperform append-only structured state.

### Hypothesis 2: Forgetting Helps Under Noise

If the system must keep performing after accumulating irrelevant, stale, or superseded state, a stronger forgetting policy should outperform no-forgetting variants on both quality and compactness.

### Hypothesis 3: Full BrainLayer Can Beat Simpler Structured State

If consolidation and forgetting become meaningfully useful, `brainlayer_full` should separate more clearly from `structured_no_consolidation` across providers, not just from weaker baselines.

## What Changes In Study-V3

`study-v3` should keep the validated strengths of the current system:

- working state
- autobiographical state
- existing provider integrations
- current standard regression packs

But it should explicitly redesign and retest:

- consolidation logic
- forgetting logic
- task packs that make those mechanisms necessary

## Planned System Work

### Consolidation Track

Improve consolidation so it is not just passive structure filling.

Targeted changes:

- stronger repeated-signal promotion rules
- better conversion of episodes into durable beliefs and procedures
- clearer revision/supersession of old beliefs by newer evidence
- provenance-aware promotion so consolidated state keeps links to its supporting episodes

### Forgetting Track

Improve forgetting so it is not just record count reduction.

Targeted changes:

- stronger stale-state pruning
- better demotion of outdated goals and preferences
- contradiction-aware removal or downgrade of superseded state
- retrieval-time penalties for stale or low-salience items

### Evaluation-Hardening Track

Make the tasks force these mechanisms to matter.

Targeted changes:

- longer delayed recall windows
- more irrelevant side information
- more superseded preferences and goals
- more repeated hints that must become durable only after accumulation

## Study-V3 Packs

`study-v3` should keep the current packs for regression continuity:

- `standard`
- `hard`
- `held_out`
- `external_dev`
- `external_held_out`

But it should add at least two new stress packs.

### Pack 1: `consolidation_stress`

Design goals:

- repeated weak hints instead of direct "remember this" turns
- long gaps between hints and later use
- procedure extraction from multiple small failures
- lesson formation from retrospectives and near-miss incidents

### Pack 2: `forgetting_stress`

Design goals:

- stale preferences that later reverse
- obsolete goals that should stop dominating behavior
- noisy irrelevant context that should not accumulate forever
- conflicting evidence where old state should be downgraded or removed

## Required Comparisons

The main `study-v3` comparisons should be:

- `brainlayer_full`
- `brainlayer_no_consolidation`
- `brainlayer_no_forgetting`
- `structured_no_consolidation`

Keep the weaker baselines available, but they should become secondary in `study-v3`. The key battle now is not against `context_only`. It is against the strongest simpler structured alternative.

## Provider Strategy

`study-v3` should use at least:

- one Gemini model
- one Anthropic model

Recommended primary pair:

- `gemini-2.5-flash`
- `claude-sonnet-4.5`

Recommended smaller-model pair:

- `gemini-2.5-flash-lite`
- `claude-haiku-4.5`

The first pass should focus on the primary pair. The smaller pair can be added after the first freeze if cost becomes an issue.

## Success Criteria

`study-v3` should count as successful only if it shows both of these:

1. `brainlayer_full` gains a clearer and more consistent margin over `structured_no_consolidation` on the new stress packs.
2. Removing consolidation or forgetting now causes a meaningful and repeatable drop on those stress packs.

Suggested minimum success threshold:

- at least one strong-model pair shows a repeatable `+3` or greater overall margin for `brainlayer_full` over `structured_no_consolidation` on the new stress packs
- `brainlayer_no_consolidation` and/or `brainlayer_no_forgetting` each lose at least `2` points on the same stress packs relative to `brainlayer_full`

## Failure Conditions

`study-v3` should be considered inconclusive or unsuccessful if:

- consolidation still looks effectively neutral
- forgetting still looks effectively neutral
- full BrainLayer still does not separate meaningfully from simpler structured state
- gains only appear on tuned authored tasks and not on the new stress packs

## Non-Goals

`study-v3` is not about:

- adding more general provider plumbing
- redesigning `autobio` or `working_state` from scratch
- chasing small benchmark wording fixes
- claiming universal superiority before the stronger mechanisms work

## Recommended Order

1. write a frozen `study_v3_protocol.md`
2. implement the new `consolidation_stress` and `forgetting_stress` packs
3. improve consolidation and forgetting in isolated, targeted patches
4. run smoke comparisons on the primary provider pair
5. freeze the first `study-v3` baseline only after the new stress packs are stable

## Short Version

`study-v2` showed what already works.

`study-v3` should focus on what has not earned its place yet: consolidation and forgetting.
