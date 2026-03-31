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
- a tiny Python harness with short-form and long-horizon deterministic benchmark scenarios
- three core agents:
  - `context_only`
  - `naive_memory`
  - `brainlayer`
- four BrainLayer ablations:
  - `brainlayer_no_consolidation`
  - `brainlayer_no_forgetting`
  - `brainlayer_no_autobio`
  - `brainlayer_no_working_state`
- state validation and load/save helpers for persistent BrainLayer JSON files
- a small `BrainLayerSession` wrapper for real agent loops
- a consolidation/forgetting engine for promoting repeated signals and pruning low-value noise
- a model-backed BrainLayer runtime with an adapter interface for live LLM turns
- contradiction-heavy model-loop evals for revision, continuity, and consolidation

Run the full benchmark suite with ablations:

```bash
python3 scripts/run_benchmarks.py
```

Run only the core baseline set:

```bash
python3 scripts/run_benchmarks.py --core-only
```

Export a timestamped benchmark run with CSV, JSON, append-only history, and an X-ready post:

```bash
python3 scripts/run_benchmarks.py --export-results artifacts/benchmark_runs --label baseline-v1
```

The benchmark report now includes:

- short scenarios for isolated capability checks
- long-horizon scenarios with multiple checkpoints
- ablation results for `no_consolidation`, `no_forgetting`, `no_autobio`, and `no_working_state`
- compactness signals like average retained records and average retained episodes
- optional run exports:
  - `results.json`
  - `results.csv`
  - `summary.csv`
  - `x_post.txt`
  - append-only `history.csv` and `history.jsonl` for cross-run tracking

To also dump serialized agent state for inspection:

```bash
python3 scripts/run_benchmarks.py --dump-states artifacts/states
```

Validate a saved BrainLayer state file:

```bash
python3 scripts/validate_state.py examples/brainlayer_state.sample.json
```

Run consolidation and forgetting on a saved state file:

```bash
python3 scripts/consolidate_state.py examples/brainlayer_state.sample.json --output artifacts/consolidated_state.json
```

Use the persistence helpers in a real loop:

```python
from brainlayer import BrainLayerSession

session = BrainLayerSession()
session.observe(
    text="Primary goal for this task: preserve source citations in every answer.",
    memory_type="goal",
    payload={
        "key": "primary_goal",
        "value": "preserve source citations",
        "summary": "The current primary goal is to preserve source citations in every answer.",
    },
    salience=0.9,
)
session.consolidate()
session.save("artifacts/live_state.json")
```

Run a model-backed BrainLayer turn with a chat-completions-compatible provider:

```bash
OPENAI_API_KEY=... python3 scripts/run_model_loop.py \
  --prompt "Draft the reply for the user." \
  --observe-file examples/live_turn_observations.sample.json \
  --state artifacts/live_state.json
```

Run the same loop without network access using a static dry-run response:

```bash
python3 scripts/run_model_loop.py \
  --prompt "Draft the reply for the user." \
  --observe-file examples/live_turn_observations.sample.json \
  --state artifacts/live_state.json \
  --dry-run-response '{"assistant_response":"I will keep the reply concise.","episodic_summary":"The assistant committed to a concise reply.","memory_observations":[]}'
```

The model-backed loop now does a full BrainLayer turn:

- ingests any explicit observations you pass in
- consolidates before retrieval so the prompt sees current beliefs and goals
- retrieves relevant working state, beliefs, autobiographical notes, procedures, and episodes
- prompts the model with a BrainLayer snapshot
- records the reply as an episode
- optionally applies model-suggested memory observations back into the BrainLayer state
- saves the updated state for the next turn

Run the contradiction-heavy eval suite for the model-backed loop:

```bash
python3 scripts/run_model_evals.py
```

Run only the full model-backed runtime without ablations:

```bash
python3 scripts/run_model_evals.py --core-only
```

Export a timestamped model-loop eval run with CSV, JSON, append-only history, and an X-ready post:

```bash
python3 scripts/run_model_evals.py --export-results artifacts/model_eval_runs --label contradiction-v1
```

These model-loop evals focus on the cases where BrainLayer should matter most:

- explicit preference revision after contradiction
- goal replacement when the active task changes
- relationship reframing across turns
- repeated weak hints that must consolidate before a later correction

They use a deterministic adapter over the real runtime path, which means the eval exercises:

- retrieval from the layered BrainLayer state
- runtime prompt construction
- model-style JSON output parsing
- memory write-back into BrainLayer
- later retrieval after revision and consolidation

Model-loop eval exports produce:

- `results.json`
- `results.csv`
- `summary.csv`
- `x_post.txt`
- append-only `model_eval_history.csv` and `model_eval_history.jsonl` for cross-run tracking
