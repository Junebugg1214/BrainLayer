# BrainLayer Architecture

## Overview

BrainLayer is a layered internal state for an AI agent. The layers are separated because they play different roles in cognition and should update at different speeds.

The proposed architecture is intentionally simple enough to prototype.

## Layer 1: Working State

This is the agent's live cognitive workspace.

It should contain:

- current task
- active subgoals
- open questions
- current hypotheses
- recent observations
- current constraints
- temporary scratch conclusions

Properties:

- small
- fast-changing
- tightly coupled to planning
- not all content should persist

Analogy:

This is closer to attention plus working memory than durable memory.

## Layer 2: Episodic Memory

This stores concrete events and experiences.

Examples:

- "The user said they prefer concise answers on March 30."
- "The agent previously tried approach B and it failed because the API timed out."
- "A review comment was addressed and then reopened."

Each episode should ideally capture:

- timestamp
- participants
- task context
- action taken
- observed outcome
- confidence
- salience

Properties:

- event-specific
- time-indexed
- useful for recalling cases, precedents, and failed attempts

## Layer 3: Semantic World and User Model

This stores generalized beliefs distilled from episodes and evidence.

Examples:

- "This user prefers direct communication."
- "This codebase uses `pnpm` and Vitest."
- "This tool often fails when auth is missing."

Each semantic item should include:

- proposition
- supporting evidence links or episode references
- confidence
- scope
- last updated timestamp

Properties:

- more stable than episodes
- should be revisable
- should not be mere copies of raw events

## Layer 4: Autobiographical State

This captures continuity of self and relationship over time.

Examples:

- ongoing collaboration themes
- prior commitments
- shared project narratives
- enduring user-agent interaction patterns

This is the layer that helps the agent answer questions like:

- "What kind of collaboration are we in?"
- "What have I been trying to accomplish across sessions?"
- "What promises or directions define this relationship?"

Properties:

- narrative and identity-oriented
- must be grounded in evidence
- should be editable and revisable

## Layer 5: Procedural Memory

This stores reusable action patterns.

Examples:

- how to approach code review in this environment
- common debugging workflows
- user-preferred formatting patterns
- recurring research routines

Properties:

- action-oriented
- often represented as plans, templates, heuristics, or policies
- should strengthen with success and weaken with failure

## Layer 6: Salience, Consolidation, and Forgetting

This is not a separate memory store as much as a control system over all the others.

It decides:

- what gets stored
- what gets promoted
- what gets merged
- what decays
- what gets revised
- what should stay uncertain

Useful signals:

- recency
- repetition
- task relevance
- emotional or goal significance
- outcome impact
- contradiction with existing beliefs

## Core Flows

### Experience Ingestion

1. Observe event
2. Write lightweight episode
3. Estimate salience
4. Update working state immediately if relevant

### Consolidation

1. Review clusters of related episodes
2. Extract stable patterns
3. Update semantic beliefs
4. Update autobiographical state if the pattern changes continuity or identity
5. Record provenance for later revision

### Retrieval For Action

1. Read current working state
2. Retrieve relevant episodes
3. Retrieve supporting semantic beliefs
4. Retrieve any procedural patterns that match the situation
5. Compose action context for planning

### Revision

1. Detect contradiction or failure
2. Lower confidence on affected beliefs
3. Attach new evidence
4. Rewrite semantic or autobiographical state without deleting historical episodes

## Minimal Data Model

The first prototype can use five simple record types:

- `working_item`
- `episode`
- `belief`
- `autobio_note`
- `procedure`

Recommended metadata fields:

- `id`
- `type`
- `content`
- `created_at`
- `updated_at`
- `source_refs`
- `confidence`
- `salience`
- `scope`

## Key Design Tensions

- compression vs fidelity
- stability vs revisability
- continuity vs confabulation
- personalization vs overfitting
- retention vs forgetting

## Implementation Advice

Start with explicit symbolic structures first.

Do not begin with one giant latent memory object. A symbolic first pass will make it easier to evaluate:

- what was stored
- why it was retrieved
- what changed
- which layer produced the benefit
