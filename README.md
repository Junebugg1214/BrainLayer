# BrainLayer

BrainLayer is a research project about making AI agent "memory" closer to a human-like cognitive layer instead of a simple storage layer.

The core idea is that an effective agent does not just need to remember facts. It needs a living internal state that supports:

- active goals and attention
- episodic recall of past interactions
- semantic understanding of the user and world
- autobiographical continuity
- procedural habits and routines
- salience, consolidation, and forgetting

This repo is a starter pack for studying that idea with Codex and Karpathy-style `autoresearch` loops.

## Core Question

How do we make an AI agent's internal state behave more like a brain layer than a memory database?

More specifically:

- What should be stored, updated, forgotten, or consolidated?
- Which parts of state should guide planning in real time?
- How should the agent build a stable sense of self, user, and context without becoming rigid or delusional?
- What evaluation setup tells us whether a richer internal state actually improves agent behavior?

## Working Thesis

Agent memory should be modeled as a multi-part cognitive state:

- `working state`: what matters now
- `episodic memory`: what happened
- `semantic memory`: what is generally true
- `autobiographical state`: what defines continuity of self and relationship
- `procedural memory`: what the agent tends to do
- `salience and forgetting`: what should persist, decay, or be revised

The main hypothesis is that agents become more coherent, adaptive, and useful when these layers are separated and allowed to interact, instead of being collapsed into one retrieval store.

## Repo Layout

- `docs/thesis.md`: project thesis and research framing
- `docs/architecture.md`: proposed BrainLayer architecture
- `docs/experiments.md`: hypotheses, tasks, and evaluation plan
- `program.md`: Codex/autoresearch research prompt scaffold
- `schemas/`: JSON Schemas for the core BrainLayer layers
- `brainlayer/`: minimal Python prototype and benchmark harness
- `examples/brainlayer_state.sample.json`: sample serialized BrainLayer state

## First Outcomes To Target

- A BrainLayer schema that is simple enough to implement
- A repeatable benchmark for measuring behavioral gains
- A set of ablations that compare:
  - plain context window
  - naive memory/RAG
  - structured multi-layer BrainLayer
- A loop where Codex can generate, test, and refine hypotheses

## Design Principles

- Keep evidence separate from inference.
- Allow memory to change through consolidation, not only append.
- Give the agent a notion of importance, uncertainty, and recency.
- Treat identity and user modeling as first-class state.
- Prefer simple modules with measurable effects over grand unified abstractions.

## Suggested Next Step

Use `program.md` as the working brief for Codex and start with a minimal prototype:

1. implement a BrainLayer state schema
2. build a tiny simulation task suite
3. compare it against a baseline memory agent
4. keep only changes that improve measurable agent behavior

## Minimal Prototype

The repo now includes a dependency-light prototype with:

- JSON Schema contracts for `working_state`, `episodes`, `beliefs`, `autobiographical_state`, and `procedures`
- a tiny Python harness with three deterministic benchmark scenarios
- three agents:
  - `context_only`
  - `naive_memory`
  - `brainlayer`

Run the seed benchmark suite with:

```bash
python3 scripts/run_benchmarks.py
```

To also dump serialized agent state for inspection:

```bash
python3 scripts/run_benchmarks.py --dump-states artifacts/states
```
