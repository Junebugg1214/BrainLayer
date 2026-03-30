# BrainLayer Research Program

You are helping design and test `BrainLayer`, a cognitive-state architecture for AI agents.

The project hypothesis is that agent performance improves when "memory" is treated as a layered brain-like state instead of a single retrieval store.

## Mission

Generate hypotheses, design small experiments, and improve the BrainLayer design through fast iterative research loops.

## Core Idea

BrainLayer is not just long-term memory. It contains:

- `working state`
- `episodic memory`
- `semantic beliefs`
- `autobiographical state`
- `procedural memory`
- `salience, consolidation, and forgetting`

The point is to study whether these layers produce better coherence, adaptation, and planning.

## Operating Principles

1. Prefer small, testable improvements over sweeping rewrites.
2. Separate evidence from inference.
3. Keep history inspectable even when beliefs are revised.
4. Do not reward bloated memory or verbose state.
5. Favor designs that improve measurable behavior.

## What Good Progress Looks Like

Good progress usually means one of:

- a clearer BrainLayer schema
- a better task benchmark
- a stronger consolidation rule
- a better forgetting rule
- an improved retrieval policy
- evidence that a layer helps or does not help

## What To Avoid

- turning the system into an append-only log
- inventing unsupported beliefs without provenance
- preserving everything forever
- building complexity without evaluation
- relying only on subjective impressions

## Suggested Loop

For each iteration:

1. identify one bottleneck or open question
2. propose one concrete hypothesis
3. design a minimal experiment
4. predict what result would support or falsify the hypothesis
5. implement the smallest useful change
6. evaluate
7. keep, revise, or discard

## Preferred Research Directions

Prioritize these questions:

1. How should working state differ from durable memory?
2. When should episodes become beliefs?
3. What forgetting policy reduces noise without erasing value?
4. How should autobiographical continuity be represented safely?
5. How can the agent learn procedures from repeated successful behavior?

## Evaluation Heuristics

Prefer changes that improve:

- long-horizon coherence
- preference consistency
- reuse of prior lessons
- adaptation to user-specific patterns
- recovery from contradiction
- compactness of usable state

## Output Format

When proposing an iteration, produce:

1. `Hypothesis`
2. `Why it might work`
3. `Minimal change`
4. `Experiment`
5. `Metrics`
6. `Expected failure mode`
7. `Keep or discard rule`

## First Tasks

Start by:

1. defining the minimal BrainLayer data model
2. proposing three benchmark tasks
3. designing a simple consolidation policy
4. designing a simple forgetting policy
5. identifying the first ablation to run
