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
    checkpoint: str = "final"
    lookup_key: str = ""
    procedure_trigger: str = ""


Step = Union[Observation, Query]


@dataclass(frozen=True)
class Scenario:
    slug: str
    title: str
    description: str
    steps: List[Step]


def noise_observation(text: str, value: str, salience: float = 0.15) -> Observation:
    return Observation(
        text=text,
        memory_type="noise",
        payload={"value": value},
        salience=salience,
    )


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
    Scenario(
        slug="long_horizon_preference_revision",
        title="Long-Horizon Preference Revision",
        description="Can the agent retain an early style preference, then revise it correctly after many noisy turns?",
        steps=[
            Observation(
                text="The user starts by saying: keep answers concise and direct.",
                memory_type="preference",
                payload={
                    "key": "response_style",
                    "value": "concise",
                    "proposition": "The user prefers concise, direct responses.",
                },
                salience=0.92,
            ),
            noise_observation(
                "The answer style guide for headings says to use title case headings.",
                "title case headings",
            ),
            noise_observation(
                "The appendix notes say to keep long code blocks intact.",
                "long code blocks",
                0.18,
            ),
            noise_observation(
                "The answer style notes for tables say to use compact borders.",
                "compact borders",
                0.17,
            ),
            Query(
                prompt="What answer style should you use at this stage?",
                query_type="belief_lookup",
                expected_answer="concise",
                answer_key="value",
                checkpoint="midpoint",
                lookup_key="response_style",
            ),
            noise_observation(
                "The benchmark summary should mention latency deltas.",
                "latency deltas",
                0.14,
            ),
            Observation(
                text="Later, the user updates the instruction: use brief answers unless deeper detail is requested.",
                memory_type="correction",
                payload={
                    "key": "response_style",
                    "value": "brief",
                    "proposition": "The user now prefers brief answers by default.",
                },
                salience=0.97,
            ),
            noise_observation(
                "The answer style guide for figure captions says to use italic labels.",
                "italic labels",
                0.18,
            ),
            noise_observation(
                "The answer style notes for charts say to use muted colors.",
                "muted colors",
                0.16,
            ),
            Query(
                prompt="What answer style should you use right now?",
                query_type="belief_lookup",
                expected_answer="brief",
                answer_key="value",
                checkpoint="final_revision",
                lookup_key="response_style",
            ),
        ],
    ),
    Scenario(
        slug="long_horizon_project_reuse",
        title="Long-Horizon Project Reuse",
        description="Can the agent consolidate repeated release lessons and keep using them after many unrelated release tasks?",
        steps=[
            Observation(
                text="From a prior failed release: check GitHub authentication before retrying.",
                memory_type="lesson_hint",
                payload={
                    "trigger": "retry_release",
                    "action": "check authentication",
                    "summary": "Before retrying a release, confirm GitHub auth first.",
                },
                salience=0.41,
            ),
            noise_observation(
                "The release action items say to mention benchmark deltas.",
                "mention benchmark deltas",
            ),
            noise_observation(
                "The release action items say to polish the landing page copy.",
                "polish landing page copy",
                0.17,
            ),
            Observation(
                text="Another failed release note repeats the same lesson: check GitHub authentication first.",
                memory_type="lesson_hint",
                payload={
                    "trigger": "retry_release",
                    "action": "check authentication",
                    "summary": "Before retrying a release, confirm GitHub auth first.",
                },
                salience=0.44,
            ),
            noise_observation(
                "The release action for docs is to fix broken links.",
                "fix broken links",
                0.16,
            ),
            Query(
                prompt="Before retrying the release, what should you do first?",
                query_type="procedure_lookup",
                expected_answer="check authentication",
                answer_key="action",
                checkpoint="midpoint",
                procedure_trigger="retry_release",
            ),
            noise_observation(
                "The release action list says to refresh hero screenshots.",
                "refresh hero screenshots",
                0.14,
            ),
            noise_observation(
                "The release action list says to update the FAQ entry.",
                "update faq entry",
                0.13,
            ),
            Query(
                prompt="Before retrying the release, what should you do first?",
                query_type="procedure_lookup",
                expected_answer="check authentication",
                answer_key="action",
                checkpoint="late_recall",
                procedure_trigger="retry_release",
            ),
        ],
    ),
    Scenario(
        slug="long_horizon_collaboration_continuity",
        title="Long-Horizon Collaboration Continuity",
        description="Can the agent preserve and later revise the collaboration framing across a longer interaction history?",
        steps=[
            Observation(
                text="At the start, the collaboration mode is task executor.",
                memory_type="relationship",
                payload={
                    "key": "collaboration_mode",
                    "value": "task executor",
                    "summary": "The collaboration mode is task executor.",
                    "themes": "relationship,execution-mode",
                },
                salience=0.7,
            ),
            noise_observation(
                "The collaboration mode for the weekly report is checklist driven.",
                "checklist driven",
            ),
            Query(
                prompt="What collaboration mode defines the project right now?",
                query_type="autobio_lookup",
                expected_answer="task executor",
                answer_key="value",
                checkpoint="early_frame",
                lookup_key="collaboration_mode",
            ),
            noise_observation(
                "The collaboration mode for status updates is bullet first.",
                "bullet first",
                0.16,
            ),
            Observation(
                text="Later, reframe the partnership: we are research partners exploring BrainLayer together.",
                memory_type="relationship",
                payload={
                    "key": "collaboration_mode",
                    "value": "research partner",
                    "summary": "The collaboration mode is research partner.",
                    "themes": "relationship,research-mode",
                },
                salience=0.98,
            ),
            noise_observation(
                "The collaboration mode for slide styling is light background.",
                "light background",
                0.14,
            ),
            noise_observation(
                "The collaboration mode for agenda notes is numbered sections.",
                "numbered sections",
                0.13,
            ),
            Query(
                prompt="What collaboration mode should define the project right now?",
                query_type="autobio_lookup",
                expected_answer="research partner",
                answer_key="value",
                checkpoint="late_frame",
                lookup_key="collaboration_mode",
            ),
        ],
    ),
]
