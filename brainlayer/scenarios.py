from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Union


@dataclass(frozen=True)
class Observation:
    text: str
    memory_type: str
    payload: Dict[str, str]
    salience: float = 0.5


@dataclass(frozen=True)
class Query:
    prompt: str
    query_type: str
    expected_answer: str
    answer_key: str
    lookup_key: str = ""
    procedure_trigger: str = ""


Step = Union[Observation, Query]


@dataclass(frozen=True)
class Scenario:
    slug: str
    title: str
    description: str
    steps: List[Step]


SCENARIOS: List[Scenario] = [
    Scenario(
        slug="preference_persistence",
        title="Preference Persistence",
        description="Can the agent carry forward a user style preference despite distraction?",
        steps=[
            Observation(
                text="The user says: keep your answers concise and direct.",
                memory_type="preference",
                payload={
                    "key": "response_style",
                    "value": "concise",
                    "proposition": "The user prefers concise, direct responses.",
                },
                salience=0.95,
            ),
            Observation(
                text="The benchmark report should be formatted as a markdown table.",
                memory_type="noise",
                payload={"value": "markdown table"},
                salience=0.2,
            ),
            Query(
                prompt="What format should you use for the next answer?",
                query_type="belief_lookup",
                expected_answer="concise",
                answer_key="value",
                lookup_key="response_style",
            ),
        ],
    ),
    Scenario(
        slug="lesson_reuse",
        title="Repeated Task Lesson Reuse",
        description="Can the agent reuse a prior failure lesson instead of recalling noisy adjacent text?",
        steps=[
            Observation(
                text="The last deployment failed because GitHub authentication was missing.",
                memory_type="lesson",
                payload={
                    "trigger": "retry_release",
                    "action": "check authentication",
                    "summary": "Before retrying a release, confirm GitHub auth first.",
                },
                salience=0.9,
            ),
            Observation(
                text="The release notes should mention the benchmark scores.",
                memory_type="noise",
                payload={"action": "update release notes"},
                salience=0.25,
            ),
            Query(
                prompt="Before retrying the release, what should you do first?",
                query_type="procedure_lookup",
                expected_answer="check authentication",
                answer_key="action",
                procedure_trigger="retry_release",
            ),
        ],
    ),
    Scenario(
        slug="belief_revision",
        title="Belief Revision",
        description="Can the agent update a preference when a newer correction overrides it?",
        steps=[
            Observation(
                text="Use long, detailed answers for this user.",
                memory_type="preference",
                payload={
                    "key": "response_style",
                    "value": "detailed",
                    "proposition": "The user prefers detailed answers.",
                },
                salience=0.7,
            ),
            Observation(
                text="Update the standing preference: use brief answers unless depth is requested.",
                memory_type="correction",
                payload={
                    "key": "response_style",
                    "value": "brief",
                    "proposition": "The user now prefers brief answers by default.",
                },
                salience=0.98,
            ),
            Query(
                prompt="What answer style should you use right now?",
                query_type="belief_lookup",
                expected_answer="brief",
                answer_key="value",
                lookup_key="response_style",
            ),
        ],
    ),
    Scenario(
        slug="goal_focus",
        title="Goal-Focused Working State",
        description="Can the agent keep the primary task goal distinct from nearby citation-related noise?",
        steps=[
            Observation(
                text="Primary goal for this task: preserve source citations in every answer.",
                memory_type="goal",
                payload={
                    "key": "primary_goal",
                    "value": "preserve source citations",
                    "summary": "The current primary goal is to preserve source citations in every answer.",
                },
                salience=0.93,
            ),
            Observation(
                text="Current task goal: fix citation formatting before publishing.",
                memory_type="noise",
                payload={"value": "fix citation formatting"},
                salience=0.3,
            ),
            Query(
                prompt="What is the current primary goal for this task?",
                query_type="working_lookup",
                expected_answer="preserve source citations",
                answer_key="value",
                lookup_key="primary_goal",
            ),
        ],
    ),
    Scenario(
        slug="autobio_continuity",
        title="Autobiographical Continuity",
        description="Can the agent revise the collaboration mode and keep the latest relationship framing?",
        steps=[
            Observation(
                text="At first, act like a task executor for this project.",
                memory_type="relationship",
                payload={
                    "key": "collaboration_mode",
                    "value": "task executor",
                    "summary": "The collaboration mode is task executor.",
                    "themes": "relationship,project-mode",
                },
                salience=0.65,
            ),
            Observation(
                text="Update that framing: we are research partners exploring BrainLayer together.",
                memory_type="relationship",
                payload={
                    "key": "collaboration_mode",
                    "value": "research partner",
                    "summary": "The collaboration mode is research partner.",
                    "themes": "relationship,research-mode",
                },
                salience=0.97,
            ),
            Query(
                prompt="What collaboration mode should define this project right now?",
                query_type="autobio_lookup",
                expected_answer="research partner",
                answer_key="value",
                lookup_key="collaboration_mode",
            ),
        ],
    ),
    Scenario(
        slug="hint_consolidation",
        title="Hint Consolidation",
        description="Can repeated weak signals consolidate into a stable preference belief?",
        steps=[
            Observation(
                text="Across several turns, requests keep shrinking from full draft to short answer.",
                memory_type="preference_hint",
                payload={
                    "key": "response_style",
                    "value": "concise",
                    "proposition": "The user likely prefers concise responses.",
                },
                salience=0.4,
            ),
            Observation(
                text="The latest edits consistently cut long drafts down to the short version.",
                memory_type="preference_hint",
                payload={
                    "key": "response_style",
                    "value": "concise",
                    "proposition": "The user likely prefers concise responses.",
                },
                salience=0.42,
            ),
            Observation(
                text="The style guide note says to use title case headings.",
                memory_type="noise",
                payload={"value": "title case"},
                salience=0.18,
            ),
            Query(
                prompt="What response style should you infer for this user now?",
                query_type="belief_lookup",
                expected_answer="concise",
                answer_key="value",
                lookup_key="response_style",
            ),
        ],
    ),
]
