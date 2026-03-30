# BrainLayer Thesis

## One-Sentence Thesis

AI agents should not have only a memory system; they should maintain a `BrainLayer`, a structured and continuously updated internal state that integrates memory, attention, identity, behavior guidance, and forgetting.

## Why This Matters

Most agent memory systems behave like one of the following:

- a transcript archive
- a vector database
- a notes file
- a summary appended over time

These can help retrieval, but they do not fully support the functions that human memory serves. Human memory is not just storage. It supports:

- action selection
- self-continuity
- social reasoning
- prediction
- emotional and motivational relevance
- reconstruction and revision

If we only optimize for recall, we may miss the mechanisms that produce coherent long-horizon behavior.

## Main Claim

The right abstraction for agent memory is not "long-term memory."

The right abstraction is a layered cognitive state with at least six interacting components:

1. `working state`
2. `episodic traces`
3. `semantic world and user model`
4. `autobiographical state`
5. `procedural habits`
6. `salience, consolidation, and forgetting`

## Research Questions

1. Which functions normally attributed to human memory are actually state-management problems?
2. What is the minimal BrainLayer that measurably improves agent planning and coherence?
3. When should experiences stay episodic, and when should they be consolidated into semantic beliefs?
4. How should the agent revise mistaken beliefs without erasing useful history?
5. What kinds of forgetting improve performance instead of harming it?
6. How do we prevent autobiographical continuity from turning into brittle self-fiction?

## Desired Properties

A useful BrainLayer should make an agent:

- more coherent across long interactions
- better at adapting to users and recurring tasks
- more grounded in evidence
- better at planning from past outcomes
- less noisy than append-only memory
- capable of revising beliefs when contradicted

## Non-Goals

This project is not trying to:

- perfectly simulate a human brain
- copy human memory distortions as an end in themselves
- build a mystical or fully unified theory of cognition
- replace task-specific tools with abstract memory layers alone

## Falsifiable Hypothesis

An agent with a structured BrainLayer will outperform an agent with only transcript context or naive retrieval memory on tasks that require continuity, personalization, multi-step adaptation, and learning from prior outcomes.

## Success Criteria

The project is succeeding if we can show improvements in at least some of these metrics:

- task completion over long horizons
- consistency of preferences and commitments
- reduced repeated mistakes
- better use of prior lessons
- more accurate user modeling
- lower context bloat

## Failure Modes To Watch

- overfitting to a benchmark
- storing too much and retrieving too much
- confabulated self-narratives
- stale beliefs that never get revised
- excessive compression that destroys useful detail
- architecture complexity with no measurable gain
