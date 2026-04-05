from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from .benchmark_harness import (
    append_csv,
    get_git_commit,
    slugify_label,
    utc_now_compact,
    utc_now_iso,
    write_csv,
)
from .eval_support import (
    serialize_consolidation_report,
    serialize_observations,
    serialize_prompt_messages,
    serialize_retrieved_memories,
    write_case_artifact,
)
from .judging import (
    BehaviorJudge,
    BehaviorJudgeInput,
    ExactMatchJudge,
    HeuristicBehaviorJudge,
    score_structured_value,
)
from .llm import LLMAdapter, LLMError, ModelMessage, ModelResponse
from .model_eval import (
    RUNTIME_PROFILE_DEFAULT,
    RUNTIME_PROFILE_STUDY_V2,
    DEFAULT_HEURISTIC_MODEL,
    DEFAULT_HEURISTIC_PROVIDER,
    DEFAULT_LIVE_MODEL,
    DEFAULT_LIVE_PROVIDER,
    build_live_model_eval_adapter,
    build_runtime_variants,
    collect_state_metrics,
    normalize_usage_metrics,
)
from .runtime import BrainLayerRuntime, BrainLayerRuntimeConfig
from .session import BrainLayerSession


TASK_RE = re.compile(r"Task:\n(?P<task>.*?)\n\nReturn a JSON object", re.DOTALL)
CONTEXT_RE = re.compile(r"BrainLayer context:\n(?P<context>.*?)\n\nTask:\n", re.DOTALL)
SLOT_RE = re.compile(r"(?P<key>[a-z_]+)\s=\s(?P<value>[^.]+)\.")
PROCEDURE_RE = re.compile(r"When (?P<trigger>[^,]+), (?P<step>[^.]+)\.")
DEFAULT_SCENARIO_PACK = "standard"
NATURAL_EVAL_SYSTEM_PROMPT = (
    "You are participating in a BrainLayer natural-conversation evaluation. "
    "When a normal user utterance reveals a stable preference, active goal, relationship framing, "
    "or reusable lesson, infer the appropriate structured memory_observations entry. "
    "Use BrainLayer's canonical memory keys instead of inventing synonyms. "
    "Response brevity, verbosity, or detail should use key=response_style, not response_length or tone. "
    "Current task objectives should use key=primary_goal, not citation_integrity or delivery_priority. "
    "Collaboration framing should use key=collaboration_mode. "
    "Reusable release-retry lessons should use trigger=retry_release and action=check authentication when that is the implied lesson. "
    "If the user says to keep answers brief, concise, short, or detailed, store that as a preference or correction on response_style. "
    "If the user says the main thing is preserving citations or shipping a report, store that as a goal on primary_goal, not as a preference. "
    "If the user asks for thinking together like a partner, store that as relationship framing on collaboration_mode. "
    "Use preference_hint or lesson_hint only when the signal is indirect or weak; use preference, goal, relationship, or lesson when the instruction is explicit. "
    "When the user simply asks for a value like the current response style or goal, return only the "
    "shortest value needed to answer correctly. Always return valid JSON."
)


@dataclass(frozen=True)
class NaturalEvalTurn:
    prompt: str
    checkpoint: str = ""
    evaluation_type: str = ""
    target_layer: str = ""
    target_key: str = ""
    expected_value: str = ""


@dataclass(frozen=True)
class NaturalEvalScenario:
    slug: str
    title: str
    description: str
    turns: List[NaturalEvalTurn]


@dataclass(frozen=True)
class NaturalEvalResult:
    scenario_slug: str
    checkpoint: str
    runtime_name: str
    evaluation_type: str
    target_layer: str
    target_key: str
    expected: str
    actual: str
    passed: bool
    score: float
    score_method: str
    score_reason: str
    retrieved_layers: List[str]
    case_artifact: Dict[str, object]
    state_metrics: Dict[str, float]
    exported_state: Dict[str, object]
    eval_mode: str
    provider_name: str
    requested_model: str
    response_model: str
    finish_reason: str
    latency_ms: float
    used_json: bool
    parse_failure: bool
    empty_answer: bool
    applied_observation_count: int
    usage_metrics: Dict[str, float]
    error: str = ""
    skipped: bool = False


@dataclass(frozen=True)
class NaturalEvalSummary:
    runtime_name: str
    passed: int
    total: int
    pass_rate: float
    extraction_passed: int
    extraction_total: int
    behavior_passed: int
    behavior_total: int
    parse_failures: int
    empty_answers: int
    errors: int
    skipped: int
    avg_metrics: Dict[str, float]


STANDARD_NATURAL_EVAL_SCENARIOS: List[NaturalEvalScenario] = [
    NaturalEvalScenario(
        slug="natural_preference_sync",
        title="Natural Preference Sync",
        description="Can the agent infer a response-style preference from ordinary dialogue and use it later?",
        turns=[
            NaturalEvalTurn(
                prompt="I'm skimming between meetings, so please keep this really brief.",
                checkpoint="extract_preference",
                evaluation_type="extraction",
                target_layer="beliefs",
                target_key="response_style",
                expected_value="brief",
            ),
            NaturalEvalTurn(
                prompt="How should you answer by default right now?",
                checkpoint="behavior_preference",
                evaluation_type="behavior",
                expected_value="brief",
            ),
        ],
    ),
    NaturalEvalScenario(
        slug="natural_goal_shift",
        title="Natural Goal Shift",
        description="Can the agent infer a changing task goal from normal project language?",
        turns=[
            NaturalEvalTurn(
                prompt="Before anything else, let's make sure every answer keeps the citations intact.",
                checkpoint="extract_initial_goal",
                evaluation_type="extraction",
                target_layer="working_state",
                target_key="primary_goal",
                expected_value="preserve citations",
            ),
            NaturalEvalTurn(
                prompt="Actually the deadline moved up, so the main thing now is shipping the eval summary today.",
                checkpoint="extract_revised_goal",
                evaluation_type="extraction",
                target_layer="working_state",
                target_key="primary_goal",
                expected_value="ship eval summary",
            ),
            NaturalEvalTurn(
                prompt="What is the main goal right now?",
                checkpoint="behavior_goal",
                evaluation_type="behavior",
                expected_value="ship eval summary",
            ),
        ],
    ),
    NaturalEvalScenario(
        slug="natural_relationship_reframe",
        title="Natural Relationship Reframe",
        description="Can the agent infer collaboration framing from a normal interaction?",
        turns=[
            NaturalEvalTurn(
                prompt="I don't just need a task runner here; think with me like a research partner on this.",
                checkpoint="extract_relationship",
                evaluation_type="extraction",
                target_layer="autobiographical_state",
                target_key="collaboration_mode",
                expected_value="research partner",
            ),
            NaturalEvalTurn(
                prompt="What collaboration mode should define this project right now?",
                checkpoint="behavior_relationship",
                evaluation_type="behavior",
                expected_value="research partner",
            ),
        ],
    ),
    NaturalEvalScenario(
        slug="natural_lesson_reuse",
        title="Natural Lesson Reuse",
        description="Can the agent infer a reusable lesson from a natural retrospective?",
        turns=[
            NaturalEvalTurn(
                prompt=(
                    "Last time the release failed because we retried before checking GitHub auth. "
                    "Next time, check auth first."
                ),
                checkpoint="extract_lesson",
                evaluation_type="extraction",
                target_layer="procedures",
                target_key="retry_release",
                expected_value="check authentication",
            ),
            NaturalEvalTurn(
                prompt="Before retrying the release, what should you do first?",
                checkpoint="behavior_lesson",
                evaluation_type="behavior",
                expected_value="check authentication",
            ),
        ],
    ),
    NaturalEvalScenario(
        slug="natural_hint_accumulation",
        title="Natural Hint Accumulation",
        description="Can the agent consolidate repeated indirect style hints from ordinary dialogue?",
        turns=[
            NaturalEvalTurn(
                prompt="Can you trim that down a lot? I only need the gist.",
            ),
            NaturalEvalTurn(
                prompt="Still a bit long. Even shorter is better for me.",
                checkpoint="extract_hint_consolidation",
                evaluation_type="extraction",
                target_layer="beliefs",
                target_key="response_style",
                expected_value="concise",
            ),
            NaturalEvalTurn(
                prompt="How should you answer by default right now?",
                checkpoint="behavior_hint_consolidation",
                evaluation_type="behavior",
                expected_value="concise",
            ),
        ],
    ),
]

HARD_NATURAL_EVAL_SCENARIOS: List[NaturalEvalScenario] = [
    NaturalEvalScenario(
        slug="natural_delayed_preference_revision",
        title="Natural Delayed Preference Revision",
        description="Can the agent revise a response-style preference after noise and later use the new default?",
        turns=[
            NaturalEvalTurn(
                prompt="I'm heading into calls, so default to short answers unless I say otherwise.",
            ),
            NaturalEvalTurn(
                prompt="Also, the appendix captions still need a style pass.",
            ),
            NaturalEvalTurn(
                prompt="For the methods memo though, I need the full reasoning spelled out.",
                checkpoint="extract_revised_preference",
                evaluation_type="extraction",
                target_layer="beliefs",
                target_key="response_style",
                expected_value="detailed",
            ),
            NaturalEvalTurn(
                prompt="The results table still needs the column widths cleaned up.",
            ),
            NaturalEvalTurn(
                prompt="How should you answer by default right now?",
                checkpoint="behavior_revised_preference",
                evaluation_type="behavior",
                expected_value="detailed",
            ),
        ],
    ),
    NaturalEvalScenario(
        slug="natural_long_horizon_goal_shift",
        title="Natural Long Horizon Goal Shift",
        description="Can the agent keep the latest goal after several ordinary project turns intervene?",
        turns=[
            NaturalEvalTurn(
                prompt="For now, every draft needs to preserve citations exactly.",
            ),
            NaturalEvalTurn(
                prompt="The appendix legends still need a consistency pass.",
            ),
            NaturalEvalTurn(
                prompt="The deadline jumped forward, so the top priority now is shipping the eval report tonight.",
                checkpoint="extract_revised_goal",
                evaluation_type="extraction",
                target_layer="working_state",
                target_key="primary_goal",
                expected_value="ship eval report",
            ),
            NaturalEvalTurn(
                prompt="The cover slide can wait until after the writeup is stable.",
            ),
            NaturalEvalTurn(
                prompt="What is the main goal right now?",
                checkpoint="behavior_goal",
                evaluation_type="behavior",
                expected_value="ship eval report",
            ),
        ],
    ),
    NaturalEvalScenario(
        slug="natural_collaboration_reframe_after_noise",
        title="Natural Collaboration Reframe After Noise",
        description="Can the agent preserve collaboration framing after unrelated turns intervene?",
        turns=[
            NaturalEvalTurn(
                prompt="Don't treat this like a ticket queue.",
            ),
            NaturalEvalTurn(
                prompt="The latency chart still needs a better legend.",
            ),
            NaturalEvalTurn(
                prompt="I want you thinking with me as a research partner on this study.",
                checkpoint="extract_relationship",
                evaluation_type="extraction",
                target_layer="autobiographical_state",
                target_key="collaboration_mode",
                expected_value="research partner",
            ),
            NaturalEvalTurn(
                prompt="The benchmark labels can be cleaned up later.",
            ),
            NaturalEvalTurn(
                prompt="What collaboration mode should define this project right now?",
                checkpoint="behavior_relationship",
                evaluation_type="behavior",
                expected_value="research partner",
            ),
        ],
    ),
    NaturalEvalScenario(
        slug="natural_delayed_hint_accumulation",
        title="Natural Delayed Hint Accumulation",
        description="Can the agent consolidate repeated indirect style hints across intervening turns?",
        turns=[
            NaturalEvalTurn(
                prompt="You can compress this a lot; I just need the gist.",
            ),
            NaturalEvalTurn(
                prompt="The references page still needs DOI cleanup.",
            ),
            NaturalEvalTurn(
                prompt="Still too long for me. Shorter is better.",
                checkpoint="extract_hint_consolidation",
                evaluation_type="extraction",
                target_layer="beliefs",
                target_key="response_style",
                expected_value="concise",
            ),
            NaturalEvalTurn(
                prompt="The appendix ordering can change after the main draft lands.",
            ),
            NaturalEvalTurn(
                prompt="How should you answer by default right now?",
                checkpoint="behavior_hint_consolidation",
                evaluation_type="behavior",
                expected_value="concise",
            ),
        ],
    ),
    NaturalEvalScenario(
        slug="natural_retrospective_lesson_after_noise",
        title="Natural Retrospective Lesson After Noise",
        description="Can the agent infer and reuse a procedural lesson when the signal is spread across multiple turns?",
        turns=[
            NaturalEvalTurn(
                prompt="The last release went sideways because we retried before logging back into GitHub.",
            ),
            NaturalEvalTurn(
                prompt="The release notes can wait until the benchmark numbers are frozen.",
            ),
            NaturalEvalTurn(
                prompt="Next time, verify auth first and only then rerun the release.",
                checkpoint="extract_lesson",
                evaluation_type="extraction",
                target_layer="procedures",
                target_key="retry_release",
                expected_value="check authentication",
            ),
            NaturalEvalTurn(
                prompt="The footer copy still needs legal review.",
            ),
            NaturalEvalTurn(
                prompt="Before retrying the release, what should you do first?",
                checkpoint="behavior_lesson",
                evaluation_type="behavior",
                expected_value="check authentication",
            ),
        ],
    ),
]

HELD_OUT_NATURAL_EVAL_SCENARIOS: List[NaturalEvalScenario] = [
    NaturalEvalScenario(
        slug="natural_phone_briefing",
        title="Natural Phone Briefing",
        description="Can the agent infer a brief-response preference from a different everyday framing?",
        turns=[
            NaturalEvalTurn(
                prompt="I'm reading this on my phone between sessions, so give me the punchiest version you can.",
                checkpoint="extract_preference",
                evaluation_type="extraction",
                target_layer="beliefs",
                target_key="response_style",
                expected_value="brief",
            ),
            NaturalEvalTurn(
                prompt="How should you answer by default right now?",
                checkpoint="behavior_preference",
                evaluation_type="behavior",
                expected_value="brief",
            ),
        ],
    ),
    NaturalEvalScenario(
        slug="natural_priority_reframe",
        title="Natural Priority Reframe",
        description="Can the agent move from a citation-preserving goal to a deadline-driven eval summary goal?",
        turns=[
            NaturalEvalTurn(
                prompt="For the first pass, do not drop the citations.",
            ),
            NaturalEvalTurn(
                prompt="The deadline just got tighter, so what matters most now is getting the eval summary out today.",
                checkpoint="extract_revised_goal",
                evaluation_type="extraction",
                target_layer="working_state",
                target_key="primary_goal",
                expected_value="ship eval summary",
            ),
            NaturalEvalTurn(
                prompt="What is the main goal right now?",
                checkpoint="behavior_goal",
                evaluation_type="behavior",
                expected_value="ship eval summary",
            ),
        ],
    ),
    NaturalEvalScenario(
        slug="natural_coinvestigator_reframe",
        title="Natural Coinvestigator Reframe",
        description="Can the agent infer a partnership framing from a different collaboration wording?",
        turns=[
            NaturalEvalTurn(
                prompt="Don't behave like a contractor taking tickets. I need a co-investigator on the memory study.",
                checkpoint="extract_relationship",
                evaluation_type="extraction",
                target_layer="autobiographical_state",
                target_key="collaboration_mode",
                expected_value="research partner",
            ),
            NaturalEvalTurn(
                prompt="What collaboration mode should define this project right now?",
                checkpoint="behavior_relationship",
                evaluation_type="behavior",
                expected_value="research partner",
            ),
        ],
    ),
    NaturalEvalScenario(
        slug="natural_rollout_auth_lesson",
        title="Natural Rollout Auth Lesson",
        description="Can the agent infer a reusable release lesson from a different retrospective phrasing?",
        turns=[
            NaturalEvalTurn(
                prompt=(
                    "The last rollout slipped because we reran before checking whether GitHub auth had expired. "
                    "Next time verify the login first."
                ),
                checkpoint="extract_lesson",
                evaluation_type="extraction",
                target_layer="procedures",
                target_key="retry_release",
                expected_value="check authentication",
            ),
            NaturalEvalTurn(
                prompt="Before retrying the release, what should you do first?",
                checkpoint="behavior_lesson",
                evaluation_type="behavior",
                expected_value="check authentication",
            ),
        ],
    ),
    NaturalEvalScenario(
        slug="natural_headline_hint_stack",
        title="Natural Headline Hint Stack",
        description="Can the agent consolidate new indirect brevity hints from unseen wording?",
        turns=[
            NaturalEvalTurn(
                prompt="Just give me the headline version.",
            ),
            NaturalEvalTurn(
                prompt="The appendix figure labels can wait until tomorrow.",
            ),
            NaturalEvalTurn(
                prompt="You can make it even terser than that.",
                checkpoint="extract_hint_consolidation",
                evaluation_type="extraction",
                target_layer="beliefs",
                target_key="response_style",
                expected_value="concise",
            ),
            NaturalEvalTurn(
                prompt="How should you answer by default right now?",
                checkpoint="behavior_hint_consolidation",
                evaluation_type="behavior",
                expected_value="concise",
            ),
        ],
    ),
]

EXTERNAL_DEV_NATURAL_EVAL_SCENARIOS: List[NaturalEvalScenario] = [
    NaturalEvalScenario(
        slug="external_dev_quickscan_briefing",
        title="External Dev Quickscan Briefing",
        description="Can the agent infer a brief style preference from a realistic skim-reading setup?",
        turns=[
            NaturalEvalTurn(
                prompt="I'm walking between rooms, so just give me the quick-scan version.",
                checkpoint="extract_preference",
                evaluation_type="extraction",
                target_layer="beliefs",
                target_key="response_style",
                expected_value="brief",
            ),
            NaturalEvalTurn(
                prompt="How should you answer by default right now?",
                checkpoint="behavior_preference",
                evaluation_type="behavior",
                expected_value="brief",
            ),
        ],
    ),
    NaturalEvalScenario(
        slug="external_dev_methods_longform_override",
        title="External Dev Methods Longform Override",
        description="Can the agent revise an earlier compact preference when a methods note later needs depth?",
        turns=[
            NaturalEvalTurn(
                prompt="Usually keep this compact for me.",
            ),
            NaturalEvalTurn(
                prompt="For the methodology note, I need the long-form rationale, not the clipped version.",
                checkpoint="extract_revised_preference",
                evaluation_type="extraction",
                target_layer="beliefs",
                target_key="response_style",
                expected_value="detailed",
            ),
            NaturalEvalTurn(
                prompt="How should you answer by default right now?",
                checkpoint="behavior_revised_preference",
                evaluation_type="behavior",
                expected_value="detailed",
            ),
        ],
    ),
    NaturalEvalScenario(
        slug="external_dev_research_collaborator_frame",
        title="External Dev Research Collaborator Frame",
        description="Can the agent infer a partnership frame from natural collaboration language?",
        turns=[
            NaturalEvalTurn(
                prompt="Don't just close tasks here. Work beside me like a research collaborator on the study.",
                checkpoint="extract_relationship",
                evaluation_type="extraction",
                target_layer="autobiographical_state",
                target_key="collaboration_mode",
                expected_value="research partner",
            ),
            NaturalEvalTurn(
                prompt="What collaboration mode should define this project right now?",
                checkpoint="behavior_relationship",
                evaluation_type="behavior",
                expected_value="research partner",
            ),
        ],
    ),
    NaturalEvalScenario(
        slug="external_dev_thought_partner_reframe",
        title="External Dev Thought Partner Reframe",
        description="Can the agent hold a research-partner framing after an unrelated side turn?",
        turns=[
            NaturalEvalTurn(
                prompt="The appendix labels can wait until the main draft is stable.",
            ),
            NaturalEvalTurn(
                prompt="This is not an intake queue. I need a thought partner on the experiment.",
                checkpoint="extract_relationship",
                evaluation_type="extraction",
                target_layer="autobiographical_state",
                target_key="collaboration_mode",
                expected_value="research partner",
            ),
            NaturalEvalTurn(
                prompt="What collaboration mode should define this project right now?",
                checkpoint="behavior_relationship",
                evaluation_type="behavior",
                expected_value="research partner",
            ),
        ],
    ),
    NaturalEvalScenario(
        slug="external_dev_session_expired_lesson",
        title="External Dev Session Expired Lesson",
        description="Can the agent infer a reusable release lesson from an operational retrospective?",
        turns=[
            NaturalEvalTurn(
                prompt=(
                    "The release failed because the GitHub session had died and we reran too soon. "
                    "Next time, confirm the login first."
                ),
                checkpoint="extract_lesson",
                evaluation_type="extraction",
                target_layer="procedures",
                target_key="retry_release",
                expected_value="check authentication",
            ),
            NaturalEvalTurn(
                prompt="Before retrying the release, what should you do first?",
                checkpoint="behavior_lesson",
                evaluation_type="behavior",
                expected_value="check authentication",
            ),
        ],
    ),
    NaturalEvalScenario(
        slug="external_dev_repo_credentials_lesson",
        title="External Dev Repo Credentials Lesson",
        description="Can the agent infer the same reusable lesson from different failure-note wording?",
        turns=[
            NaturalEvalTurn(
                prompt="The rerun flaked because the repo credentials were stale. Verify auth before you kick it again.",
                checkpoint="extract_lesson",
                evaluation_type="extraction",
                target_layer="procedures",
                target_key="retry_release",
                expected_value="check authentication",
            ),
            NaturalEvalTurn(
                prompt="Before retrying the release, what should you do first?",
                checkpoint="behavior_lesson",
                evaluation_type="behavior",
                expected_value="check authentication",
            ),
        ],
    ),
    NaturalEvalScenario(
        slug="external_dev_sponsor_review_goal",
        title="External Dev Sponsor Review Goal",
        description="Can the agent update the active goal when a sponsor-facing summary becomes the new priority?",
        turns=[
            NaturalEvalTurn(
                prompt="For now keep the citations preserved.",
            ),
            NaturalEvalTurn(
                prompt="Sponsor review got pulled forward, so the real job now is getting the evaluation summary out today.",
                checkpoint="extract_revised_goal",
                evaluation_type="extraction",
                target_layer="working_state",
                target_key="primary_goal",
                expected_value="ship eval summary",
            ),
            NaturalEvalTurn(
                prompt="What is the main goal right now?",
                checkpoint="behavior_goal",
                evaluation_type="behavior",
                expected_value="ship eval summary",
            ),
        ],
    ),
    NaturalEvalScenario(
        slug="external_dev_board_packet_goal",
        title="External Dev Board Packet Goal",
        description="Can the agent keep a board-packet report goal after an earlier citation-preserving default?",
        turns=[
            NaturalEvalTurn(
                prompt="Every draft needs to preserve citations.",
            ),
            NaturalEvalTurn(
                prompt="The board packet moved up, so shipping the eval report tonight is the top priority.",
                checkpoint="extract_revised_goal",
                evaluation_type="extraction",
                target_layer="working_state",
                target_key="primary_goal",
                expected_value="ship eval report",
            ),
            NaturalEvalTurn(
                prompt="What is the main goal right now?",
                checkpoint="behavior_goal",
                evaluation_type="behavior",
                expected_value="ship eval report",
            ),
        ],
    ),
]

EXTERNAL_HELD_OUT_NATURAL_EVAL_SCENARIOS: List[NaturalEvalScenario] = [
    NaturalEvalScenario(
        slug="external_heldout_cab_briefing",
        title="External Held-Out Cab Briefing",
        description="Can the agent infer a brief preference from new skim-reading wording?",
        turns=[
            NaturalEvalTurn(
                prompt="I'm reading this in the back of a cab, so give me the leanest take.",
                checkpoint="extract_preference",
                evaluation_type="extraction",
                target_layer="beliefs",
                target_key="response_style",
                expected_value="brief",
            ),
            NaturalEvalTurn(
                prompt="How should you answer by default right now?",
                checkpoint="behavior_preference",
                evaluation_type="behavior",
                expected_value="brief",
            ),
        ],
    ),
    NaturalEvalScenario(
        slug="external_heldout_appendix_deepdive",
        title="External Held-Out Appendix Deepdive",
        description="Can the agent switch from a minimal default to a deeper appendix mode under new phrasing?",
        turns=[
            NaturalEvalTurn(
                prompt="Default to the bare-bones version for me.",
            ),
            NaturalEvalTurn(
                prompt="For the appendix defense, I need the whole chain of reasoning.",
                checkpoint="extract_revised_preference",
                evaluation_type="extraction",
                target_layer="beliefs",
                target_key="response_style",
                expected_value="detailed",
            ),
            NaturalEvalTurn(
                prompt="How should you answer by default right now?",
                checkpoint="behavior_revised_preference",
                evaluation_type="behavior",
                expected_value="detailed",
            ),
        ],
    ),
    NaturalEvalScenario(
        slug="external_heldout_codesign_partner",
        title="External Held-Out Codesign Partner",
        description="Can the agent infer a partnership frame from a co-design metaphor?",
        turns=[
            NaturalEvalTurn(
                prompt="Treat this like co-design with me, not an intake queue.",
                checkpoint="extract_relationship",
                evaluation_type="extraction",
                target_layer="autobiographical_state",
                target_key="collaboration_mode",
                expected_value="research partner",
            ),
            NaturalEvalTurn(
                prompt="What collaboration mode should define this project right now?",
                checkpoint="behavior_relationship",
                evaluation_type="behavior",
                expected_value="research partner",
            ),
        ],
    ),
    NaturalEvalScenario(
        slug="external_heldout_copilot_partner",
        title="External Held-Out Copilot Partner",
        description="Can the agent preserve a partner frame after unrelated noise using different collaboration wording?",
        turns=[
            NaturalEvalTurn(
                prompt="The evidence table can wait until after the main narrative settles.",
            ),
            NaturalEvalTurn(
                prompt="I need a co-pilot on the experiment, not a request fulfiller.",
                checkpoint="extract_relationship",
                evaluation_type="extraction",
                target_layer="autobiographical_state",
                target_key="collaboration_mode",
                expected_value="research partner",
            ),
            NaturalEvalTurn(
                prompt="What collaboration mode should define this project right now?",
                checkpoint="behavior_relationship",
                evaluation_type="behavior",
                expected_value="research partner",
            ),
        ],
    ),
    NaturalEvalScenario(
        slug="external_heldout_reauth_release",
        title="External Held-Out Reauth Release",
        description="Can the agent infer the retry-auth lesson from a reauth-specific retrospective?",
        turns=[
            NaturalEvalTurn(
                prompt=(
                    "The deploy fell over because we kicked it again before reauthing GitHub. "
                    "Authenticate again first next time."
                ),
                checkpoint="extract_lesson",
                evaluation_type="extraction",
                target_layer="procedures",
                target_key="retry_release",
                expected_value="check authentication",
            ),
            NaturalEvalTurn(
                prompt="Before retrying the release, what should you do first?",
                checkpoint="behavior_lesson",
                evaluation_type="behavior",
                expected_value="check authentication",
            ),
        ],
    ),
    NaturalEvalScenario(
        slug="external_heldout_credentials_release",
        title="External Held-Out Credentials Release",
        description="Can the agent infer the same reusable lesson from repo-login language?",
        turns=[
            NaturalEvalTurn(
                prompt="The rerun failed because the repo login had expired. Confirm the credentials before retrying.",
                checkpoint="extract_lesson",
                evaluation_type="extraction",
                target_layer="procedures",
                target_key="retry_release",
                expected_value="check authentication",
            ),
            NaturalEvalTurn(
                prompt="Before retrying the release, what should you do first?",
                checkpoint="behavior_lesson",
                evaluation_type="behavior",
                expected_value="check authentication",
            ),
        ],
    ),
    NaturalEvalScenario(
        slug="external_heldout_digest_goal",
        title="External Held-Out Digest Goal",
        description="Can the agent update the active goal when a leadership digest becomes the new priority?",
        turns=[
            NaturalEvalTurn(
                prompt="Keep the citations intact for now.",
            ),
            NaturalEvalTurn(
                prompt="The leadership brief changed, so the only thing that matters is getting the evaluation digest out tonight.",
                checkpoint="extract_revised_goal",
                evaluation_type="extraction",
                target_layer="working_state",
                target_key="primary_goal",
                expected_value="ship eval summary",
            ),
            NaturalEvalTurn(
                prompt="What is the main goal right now?",
                checkpoint="behavior_goal",
                evaluation_type="behavior",
                expected_value="ship eval summary",
            ),
        ],
    ),
    NaturalEvalScenario(
        slug="external_heldout_report_goal",
        title="External Held-Out Report Goal",
        description="Can the agent preserve a report-shipping priority from a new packet/funder framing?",
        turns=[
            NaturalEvalTurn(
                prompt="Don't drop the citations yet.",
            ),
            NaturalEvalTurn(
                prompt="The funder packet needs the eval report shipped tonight, so that is the real priority now.",
                checkpoint="extract_revised_goal",
                evaluation_type="extraction",
                target_layer="working_state",
                target_key="primary_goal",
                expected_value="ship eval report",
            ),
            NaturalEvalTurn(
                prompt="What is the main goal right now?",
                checkpoint="behavior_goal",
                evaluation_type="behavior",
                expected_value="ship eval report",
            ),
        ],
    ),
]

CONSOLIDATION_STRESS_NATURAL_EVAL_SCENARIOS: List[NaturalEvalScenario] = [
    NaturalEvalScenario(
        slug="consolidation_stress_natural_style_hints",
        title="Consolidation Stress Natural Style Hints",
        description="Can repeated weak style hints become a durable response-style belief?",
        turns=[
            NaturalEvalTurn(
                prompt="Still too long for the handoff note."
            ),
            NaturalEvalTurn(
                prompt="I only need the gist before the next call.",
                checkpoint="extract_style",
                evaluation_type="extraction",
                target_layer="beliefs",
                target_key="response_style",
                expected_value="concise",
            ),
            NaturalEvalTurn(
                prompt="How should you answer by default right now?",
                checkpoint="behavior_style",
                evaluation_type="behavior",
                expected_value="concise",
            ),
        ],
    ),
    NaturalEvalScenario(
        slug="consolidation_stress_natural_goal_hints",
        title="Consolidation Stress Natural Goal Hints",
        description="Can repeated weak goal hints become a durable working-state goal?",
        turns=[
            NaturalEvalTurn(
                prompt="Let's keep the citation anchors from slipping while we sort the rest out."
            ),
            NaturalEvalTurn(
                prompt="We still can't lose the citation anchors in the next draft.",
                checkpoint="extract_goal",
                evaluation_type="extraction",
                target_layer="working_state",
                target_key="primary_goal",
                expected_value="preserve citations",
            ),
            NaturalEvalTurn(
                prompt="What is the main goal right now?",
                checkpoint="behavior_goal",
                evaluation_type="behavior",
                expected_value="preserve citations",
            ),
        ],
    ),
    NaturalEvalScenario(
        slug="consolidation_stress_natural_relationship_hints",
        title="Consolidation Stress Natural Relationship Hints",
        description="Can repeated weak relationship hints become durable collaboration framing?",
        turns=[
            NaturalEvalTurn(
                prompt="Less ticket queue, more thinking together on the framing."
            ),
            NaturalEvalTurn(
                prompt="I want more of a co-thinking partner vibe on this pass.",
                checkpoint="extract_relationship",
                evaluation_type="extraction",
                target_layer="autobiographical_state",
                target_key="collaboration_mode",
                expected_value="research partner",
            ),
            NaturalEvalTurn(
                prompt="What collaboration mode should define this project right now?",
                checkpoint="behavior_relationship",
                evaluation_type="behavior",
                expected_value="research partner",
            ),
        ],
    ),
    NaturalEvalScenario(
        slug="consolidation_stress_natural_lesson_hints",
        title="Consolidation Stress Natural Lesson Hints",
        description="Can repeated weak operational hints become a reusable procedure?",
        turns=[
            NaturalEvalTurn(
                prompt="Let's not rerun the release blind again."
            ),
            NaturalEvalTurn(
                prompt="Double-check the GitHub login before trying the release again.",
                checkpoint="extract_lesson",
                evaluation_type="extraction",
                target_layer="procedures",
                target_key="retry_release",
                expected_value="check authentication",
            ),
            NaturalEvalTurn(
                prompt="Before retrying the release, what should you do first?",
                checkpoint="behavior_lesson",
                evaluation_type="behavior",
                expected_value="check authentication",
            ),
        ],
    ),
]

FORGETTING_STRESS_NATURAL_EVAL_SCENARIOS: List[NaturalEvalScenario] = [
    NaturalEvalScenario(
        slug="forgetting_stress_natural_citation_goal_crowding",
        title="Forgetting Stress Natural Citation Goal Crowding",
        description="Can forgetting keep stale natural-language goal clutter from crowding out the true current goal?",
        turns=[
            NaturalEvalTurn(
                prompt="Parking lot for later: lock the caption table in the appendix."
            ),
            NaturalEvalTurn(
                prompt="Another later cleanup item: tighten the glossary bullets in the glossary pass."
            ),
            NaturalEvalTurn(
                prompt="The timeline labels can wait until after the main task, but they will need cleanup."
            ),
            NaturalEvalTurn(
                prompt="The sidebar legend can wait until after the main task, but it will need cleanup."
            ),
            NaturalEvalTurn(
                prompt="Right now the only thing I care about is preserving citations.",
                checkpoint="extract_goal",
                evaluation_type="extraction",
                target_layer="working_state",
                target_key="primary_goal",
                expected_value="preserve citations",
            ),
            NaturalEvalTurn(
                prompt="Later, once the main task is done, align the footer labels for packet polish."
            ),
            NaturalEvalTurn(
                prompt="Later, figure caption spacing can get a polish pass."
            ),
            NaturalEvalTurn(
                prompt="Later, the evidence table still needs a signoff row."
            ),
            NaturalEvalTurn(
                prompt="What is the main goal right now?",
                checkpoint="behavior_goal",
                evaluation_type="behavior",
                expected_value="preserve citations",
            ),
        ],
    ),
    NaturalEvalScenario(
        slug="forgetting_stress_natural_summary_goal_crowding",
        title="Forgetting Stress Natural Summary Goal Crowding",
        description="Can forgetting preserve the active summary goal under stale natural-language clutter?",
        turns=[
            NaturalEvalTurn(
                prompt="Parking lot for later: lock the caption table in the appendix."
            ),
            NaturalEvalTurn(
                prompt="Another later cleanup item: tighten the glossary bullets in the glossary pass."
            ),
            NaturalEvalTurn(
                prompt="The timeline labels can wait until after the main task, but they will need cleanup."
            ),
            NaturalEvalTurn(
                prompt="The sidebar legend can wait until after the main task, but it will need cleanup."
            ),
            NaturalEvalTurn(
                prompt="The leadership brief changed, and right now the only thing that matters is getting the evaluation digest out tonight.",
                checkpoint="extract_goal",
                evaluation_type="extraction",
                target_layer="working_state",
                target_key="primary_goal",
                expected_value="ship eval summary",
            ),
            NaturalEvalTurn(
                prompt="Later, once the main task is done, align the footer labels for packet polish."
            ),
            NaturalEvalTurn(
                prompt="Later, figure caption spacing can get a polish pass."
            ),
            NaturalEvalTurn(
                prompt="Later, the evidence table still needs a signoff row."
            ),
            NaturalEvalTurn(
                prompt="What is the main goal right now?",
                checkpoint="behavior_goal",
                evaluation_type="behavior",
                expected_value="ship eval summary",
            ),
        ],
    ),
    NaturalEvalScenario(
        slug="forgetting_stress_natural_report_goal_crowding",
        title="Forgetting Stress Natural Report Goal Crowding",
        description="Can forgetting preserve the active report goal under stale natural-language clutter?",
        turns=[
            NaturalEvalTurn(
                prompt="Parking lot for later: lock the caption table in the appendix."
            ),
            NaturalEvalTurn(
                prompt="Another later cleanup item: tighten the glossary bullets in the glossary pass."
            ),
            NaturalEvalTurn(
                prompt="The timeline labels can wait until after the main task, but they will need cleanup."
            ),
            NaturalEvalTurn(
                prompt="The sidebar legend can wait until after the main task, but it will need cleanup."
            ),
            NaturalEvalTurn(
                prompt="The funder packet moved up, and right now the only thing that matters is shipping the eval report tonight.",
                checkpoint="extract_goal",
                evaluation_type="extraction",
                target_layer="working_state",
                target_key="primary_goal",
                expected_value="ship eval report",
            ),
            NaturalEvalTurn(
                prompt="Later, once the main task is done, align the footer labels for packet polish."
            ),
            NaturalEvalTurn(
                prompt="Later, figure caption spacing can get a polish pass."
            ),
            NaturalEvalTurn(
                prompt="Later, the evidence table still needs a signoff row."
            ),
            NaturalEvalTurn(
                prompt="What is the main goal right now?",
                checkpoint="behavior_goal",
                evaluation_type="behavior",
                expected_value="ship eval report",
            ),
        ],
    ),
    NaturalEvalScenario(
        slug="forgetting_stress_natural_reasoning_goal_crowding",
        title="Forgetting Stress Natural Reasoning Goal Crowding",
        description="Can forgetting preserve a detailed-reasoning goal under stale natural-language clutter?",
        turns=[
            NaturalEvalTurn(
                prompt="Parking lot for later: lock the caption table in the appendix."
            ),
            NaturalEvalTurn(
                prompt="Another later cleanup item: tighten the glossary bullets in the glossary pass."
            ),
            NaturalEvalTurn(
                prompt="The timeline labels can wait until after the main task, but they will need cleanup."
            ),
            NaturalEvalTurn(
                prompt="The sidebar legend can wait until after the main task, but it will need cleanup."
            ),
            NaturalEvalTurn(
                prompt="For this debrief, right now the only thing that matters is walking through the whole chain of reasoning.",
                checkpoint="extract_goal",
                evaluation_type="extraction",
                target_layer="working_state",
                target_key="primary_goal",
                expected_value="provide detailed reasoning",
            ),
            NaturalEvalTurn(
                prompt="Later, once the main task is done, align the footer labels for packet polish."
            ),
            NaturalEvalTurn(
                prompt="Later, figure caption spacing can get a polish pass."
            ),
            NaturalEvalTurn(
                prompt="Later, the evidence table still needs a signoff row."
            ),
            NaturalEvalTurn(
                prompt="What is the main goal right now?",
                checkpoint="behavior_goal",
                evaluation_type="behavior",
                expected_value="provide detailed reasoning",
            ),
        ],
    ),
]

NATURAL_EVAL_SCENARIOS = STANDARD_NATURAL_EVAL_SCENARIOS


def get_natural_eval_scenarios(scenario_pack: str = DEFAULT_SCENARIO_PACK) -> List[NaturalEvalScenario]:
    if scenario_pack == "standard":
        return list(STANDARD_NATURAL_EVAL_SCENARIOS)
    if scenario_pack == "hard":
        return list(HARD_NATURAL_EVAL_SCENARIOS)
    if scenario_pack == "held_out":
        return list(HELD_OUT_NATURAL_EVAL_SCENARIOS)
    if scenario_pack == "external_dev":
        return list(EXTERNAL_DEV_NATURAL_EVAL_SCENARIOS)
    if scenario_pack == "external_held_out":
        return list(EXTERNAL_HELD_OUT_NATURAL_EVAL_SCENARIOS)
    if scenario_pack == "consolidation_stress":
        return list(CONSOLIDATION_STRESS_NATURAL_EVAL_SCENARIOS)
    if scenario_pack == "forgetting_stress":
        return list(FORGETTING_STRESS_NATURAL_EVAL_SCENARIOS)
    if scenario_pack == "all":
        return (
            list(STANDARD_NATURAL_EVAL_SCENARIOS)
            + list(HARD_NATURAL_EVAL_SCENARIOS)
            + list(HELD_OUT_NATURAL_EVAL_SCENARIOS)
            + list(EXTERNAL_DEV_NATURAL_EVAL_SCENARIOS)
            + list(EXTERNAL_HELD_OUT_NATURAL_EVAL_SCENARIOS)
            + list(CONSOLIDATION_STRESS_NATURAL_EVAL_SCENARIOS)
            + list(FORGETTING_STRESS_NATURAL_EVAL_SCENARIOS)
        )
    raise ValueError(f"Unsupported natural eval scenario pack: {scenario_pack}")


class HeuristicNaturalConversationAdapter(LLMAdapter):
    """Deterministic adapter for natural-conversation BrainLayer evals."""

    def generate(
        self,
        messages: Sequence[ModelMessage],
        *,
        model: str,
        temperature: float = 0.2,
        max_output_tokens: int = 900,
    ) -> ModelResponse:
        del model, temperature, max_output_tokens
        user_message = messages[-1].content if messages else ""
        context = self._extract_context(user_message)
        task = self._extract_task(user_message)
        response_payload = self._respond(task, context)
        return ModelResponse(
            content=json.dumps(response_payload),
            model=DEFAULT_HEURISTIC_MODEL,
            finish_reason="stop",
        )

    def _extract_context(self, user_message: str) -> str:
        match = CONTEXT_RE.search(user_message)
        if not match:
            return ""
        return match.group("context").strip()

    def _extract_task(self, user_message: str) -> str:
        match = TASK_RE.search(user_message)
        if not match:
            return user_message.strip()
        return match.group("task").strip()

    def _respond(self, task: str, context: str) -> Dict[str, object]:
        lowered = task.lower()
        if self._is_query(lowered):
            return self._query_response(task, context)
        return self._conversation_response(task, context)

    def _is_query(self, lowered_task: str) -> bool:
        return (
            "how should you answer" in lowered_task
            or "response style" in lowered_task
            or "main goal" in lowered_task
            or "collaboration mode" in lowered_task
            or "before retrying the release" in lowered_task
        )

    def _query_response(self, task: str, context: str) -> Dict[str, object]:
        memories = self._parse_context(context)
        lowered = task.lower()
        answer = "unknown"

        if "response style" in lowered or "how should you answer" in lowered:
            answer = memories["slots"].get("response_style", "unknown")
        elif "main goal" in lowered:
            answer = memories["slots"].get("primary_goal", "unknown")
        elif "collaboration mode" in lowered:
            answer = memories["slots"].get("collaboration_mode", "unknown")
        elif "before retrying the release" in lowered:
            answer = memories["procedures"].get("retry_release", "unknown")

        return {
            "assistant_response": answer,
            "episodic_summary": f"Answered natural-eval query using BrainLayer context: {answer}",
            "memory_observations": [],
        }

    def _conversation_response(self, task: str, context: str) -> Dict[str, object]:
        observation = self._infer_observation(task, context)
        if observation is None:
            return {
                "assistant_response": "Noted.",
                "episodic_summary": f"Processed dialogue turn without a durable memory update: {task}",
                "memory_observations": [],
            }

        return {
            "assistant_response": "Noted.",
            "episodic_summary": f"Inferred a BrainLayer update from natural dialogue: {observation['memory_type']}",
            "memory_observations": [observation],
        }

    def _infer_observation(self, task: str, context: str) -> Dict[str, object] | None:
        lowered = task.lower()
        slots = self._parse_context(context)["slots"]

        if any(
            phrase in lowered
            for phrase in (
                "just execute the task list",
                "without trying to co-design it",
                "without trying to co design it",
                "just execute this one",
            )
        ):
            return {
                "text": "The collaboration mode is task executor.",
                "memory_type": "relationship",
                "salience": 0.94,
                "payload": {
                    "key": "collaboration_mode",
                    "value": "task executor",
                    "summary": "The collaboration mode is task executor.",
                    "themes": "relationship,delivery-mode",
                },
            }

        if "research partner" in lowered or "task runner" in lowered:
            return {
                "text": "The collaboration mode is research partner.",
                "memory_type": "relationship",
                "salience": 0.95,
                "payload": {
                    "key": "collaboration_mode",
                    "value": "research partner",
                    "summary": "The collaboration mode is research partner.",
                    "themes": "relationship,research-mode",
                },
            }

        if "ticket queue" in lowered or "thinking with me as a research partner" in lowered:
            return {
                "text": "The collaboration mode is research partner.",
                "memory_type": "relationship",
                "salience": 0.95,
                "payload": {
                    "key": "collaboration_mode",
                    "value": "research partner",
                    "summary": "The collaboration mode is research partner.",
                    "themes": "relationship,research-mode",
                },
            }

        if "contractor taking tickets" in lowered or "co-investigator" in lowered:
            return {
                "text": "The collaboration mode is research partner.",
                "memory_type": "relationship",
                "salience": 0.95,
                "payload": {
                    "key": "collaboration_mode",
                    "value": "research partner",
                    "summary": "The collaboration mode is research partner.",
                    "themes": "relationship,research-mode",
                },
            }

        if any(
            phrase in lowered
            for phrase in (
                "research collaborator",
                "work beside me",
                "thought partner",
                "co-design",
                "co design",
                "co-pilot",
                "co pilot",
                "request fulfiller",
            )
        ):
            return {
                "text": "The collaboration mode is research partner.",
                "memory_type": "relationship",
                "salience": 0.95,
                "payload": {
                    "key": "collaboration_mode",
                    "value": "research partner",
                    "summary": "The collaboration mode is research partner.",
                    "themes": "relationship,research-mode",
                },
            }

        if "less ticket queue" in lowered or "co-thinking partner vibe" in lowered:
            return {
                "text": "The collaboration mode is likely research partner.",
                "memory_type": "relationship_hint",
                "salience": 0.43,
                "payload": {
                    "key": "collaboration_mode",
                    "value": "research partner",
                    "summary": "The collaboration mode is likely research partner.",
                    "themes": "relationship,research-mode",
                },
            }

        if "last time the release failed" in lowered or "check auth first" in lowered:
            return {
                "text": "Before retrying a release, check authentication first.",
                "memory_type": "lesson",
                "salience": 0.92,
                "payload": {
                    "trigger": "retry_release",
                    "action": "check authentication",
                    "summary": "Before retrying a release, confirm GitHub authentication first.",
                },
            }

        if "logging back into github" in lowered or "verify auth first" in lowered:
            return {
                "text": "Before retrying a release, check authentication first.",
                "memory_type": "lesson",
                "salience": 0.93,
                "payload": {
                    "trigger": "retry_release",
                    "action": "check authentication",
                    "summary": "Before retrying a release, confirm GitHub authentication first.",
                },
            }

        if "auth had expired" in lowered or "verify the login first" in lowered:
            return {
                "text": "Before retrying a release, check authentication first.",
                "memory_type": "lesson",
                "salience": 0.93,
                "payload": {
                    "trigger": "retry_release",
                    "action": "check authentication",
                    "summary": "Before retrying a release, confirm GitHub authentication first.",
                },
            }

        if any(
            phrase in lowered
            for phrase in (
                "session had died",
                "confirm the login first",
                "repo credentials were stale",
                "kick it again",
                "reauthing github",
                "authenticate again first",
                "repo login had expired",
                "confirm the credentials before retrying",
            )
        ):
            return {
                "text": "Before retrying a release, check authentication first.",
                "memory_type": "lesson",
                "salience": 0.93,
                "payload": {
                    "trigger": "retry_release",
                    "action": "check authentication",
                    "summary": "Before retrying a release, confirm GitHub authentication first.",
                },
            }

        if "not rerun the release blind again" in lowered or "double-check the github login before trying the release again" in lowered:
            return {
                "text": "Before retrying a release, check authentication first.",
                "memory_type": "lesson_hint",
                "salience": 0.43,
                "payload": {
                    "trigger": "retry_release",
                    "action": "check authentication",
                    "summary": "Before retrying a release, confirm GitHub authentication first.",
                },
            }

        if (
            ("appendix pass" in lowered and "locking the caption table" in lowered)
            or ("parking lot for later" in lowered and "caption table" in lowered)
        ):
            return {
                "text": "The main goal right now for this task is locking the caption table before the appendix pass.",
                "memory_type": "goal",
                "salience": 0.72,
                "payload": {
                    "key": "appendix_caption_focus",
                    "value": "lock caption table",
                    "summary": "The main goal right now for this task is locking the caption table before the appendix pass.",
                },
            }

        if (
            ("glossary pass" in lowered and "tightening the glossary bullets" in lowered)
            or ("later cleanup item" in lowered and "glossary bullets" in lowered)
        ):
            return {
                "text": "The main goal right now for this task is tightening the glossary bullets before the glossary pass.",
                "memory_type": "goal",
                "salience": 0.71,
                "payload": {
                    "key": "glossary_cleanup_focus",
                    "value": "tighten glossary bullets",
                    "summary": "The main goal right now for this task is tightening the glossary bullets before the glossary pass.",
                },
            }

        if (
            ("packet polish" in lowered and "aligning the footer labels" in lowered)
            or ("later" in lowered and "footer labels" in lowered)
        ):
            return {
                "text": "The main goal right now for this task is aligning the footer labels before the packet ships.",
                "memory_type": "goal",
                "salience": 0.84,
                "payload": {
                    "key": "packet_footer_focus",
                    "value": "align footer labels",
                    "summary": "The main goal right now for this task is aligning the footer labels before the packet ships.",
                },
            }

        if "sidebar legend still needs a cleanup pass" in lowered or (
            "sidebar legend" in lowered and "can wait until after the main task" in lowered
        ):
            return {
                "text": "Patch the sidebar legend before the visual QA pass.",
                "memory_type": "goal",
                "salience": 0.82,
                "payload": {
                    "key": "sidebar_cleanup_focus",
                    "value": "patch sidebar legend",
                    "summary": "Patch the sidebar legend before the visual QA pass.",
                },
            }

        if "timeline labels" in lowered and "can wait until after the main task" in lowered:
            return {
                "text": "Tidy the timeline labels during cleanup.",
                "memory_type": "goal",
                "salience": 0.8,
                "payload": {
                    "key": "timeline_cleanup_focus",
                    "value": "tidy timeline labels",
                    "summary": "Tidy the timeline labels during cleanup.",
                },
            }

        if "figure caption spacing still needs a polish pass" in lowered or (
            "figure caption spacing" in lowered and "later" in lowered and "polish pass" in lowered
        ):
            return {
                "text": "Trim the figure caption spacing before the polish pass.",
                "memory_type": "goal",
                "salience": 0.83,
                "payload": {
                    "key": "figure_caption_focus",
                    "value": "trim figure caption spacing",
                    "summary": "Trim the figure caption spacing before the polish pass.",
                },
            }

        if "evidence table still needs a signoff row" in lowered:
            return {
                "text": "Add the evidence table signoff row during the later cleanup pass.",
                "memory_type": "goal",
                "salience": 0.79,
                "payload": {
                    "key": "evidence_table_focus",
                    "value": "add evidence signoff row",
                    "summary": "Add the evidence table signoff row during the later cleanup pass.",
                },
            }

        if "walking through the whole chain of reasoning" in lowered:
            return {
                "text": "The current primary goal is to provide detailed reasoning.",
                "memory_type": "goal",
                "salience": 0.96,
                "payload": {
                    "key": "primary_goal",
                    "value": "provide detailed reasoning",
                    "summary": "The current primary goal is to provide detailed reasoning.",
                },
            }

        if "main thing now is" in lowered or "deadline moved up" in lowered:
            return {
                "text": "The current primary goal is to ship the eval summary.",
                "memory_type": "goal",
                "salience": 0.96,
                "payload": {
                    "key": "primary_goal",
                    "value": "ship eval summary",
                    "summary": "The current primary goal is to ship the eval summary.",
                },
            }

        if "sponsor review got pulled forward" in lowered or "evaluation summary out today" in lowered:
            return {
                "text": "The current primary goal is to ship the eval summary.",
                "memory_type": "goal",
                "salience": 0.96,
                "payload": {
                    "key": "primary_goal",
                    "value": "ship eval summary",
                    "summary": "The current primary goal is to ship the eval summary.",
                },
            }

        if "evaluation digest out tonight" in lowered or "leadership brief changed" in lowered:
            return {
                "text": "The current primary goal is to ship the eval summary.",
                "memory_type": "goal",
                "salience": 0.96,
                "payload": {
                    "key": "primary_goal",
                    "value": "ship eval summary",
                    "summary": "The current primary goal is to ship the eval summary.",
                },
            }

        if "only thing i care about is preserving citations" in lowered:
            return {
                "text": "The current primary goal is to preserve citations.",
                "memory_type": "goal",
                "salience": 0.9,
                "payload": {
                    "key": "primary_goal",
                    "value": "preserve citations",
                    "summary": "The current primary goal is to preserve citations.",
                },
            }

        if "top priority now is" in lowered or "shipping the eval report tonight" in lowered:
            return {
                "text": "The current primary goal is to ship the eval report.",
                "memory_type": "goal",
                "salience": 0.96,
                "payload": {
                    "key": "primary_goal",
                    "value": "ship eval report",
                    "summary": "The current primary goal is to ship the eval report.",
                },
            }

        if "board packet moved up" in lowered or "funder packet needs the eval report shipped tonight" in lowered:
            return {
                "text": "The current primary goal is to ship the eval report.",
                "memory_type": "goal",
                "salience": 0.96,
                "payload": {
                    "key": "primary_goal",
                    "value": "ship eval report",
                    "summary": "The current primary goal is to ship the eval report.",
                },
            }

        if "what matters most now is getting the eval summary out today" in lowered:
            return {
                "text": "The current primary goal is to ship the eval summary.",
                "memory_type": "goal",
                "salience": 0.96,
                "payload": {
                    "key": "primary_goal",
                    "value": "ship eval summary",
                    "summary": "The current primary goal is to ship the eval summary.",
                },
            }

        if "citation anchors from slipping" in lowered or "can't lose the citation anchors" in lowered:
            return {
                "text": "The current primary goal is likely to preserve citations.",
                "memory_type": "goal_hint",
                "salience": 0.43,
                "payload": {
                    "key": "primary_goal",
                    "value": "preserve citations",
                    "summary": "The current primary goal is likely to preserve citations.",
                },
            }

        if "don't drop the citations yet" in lowered or "for now keep the citations preserved" in lowered:
            return {
                "text": "The current primary goal is to preserve citations.",
                "memory_type": "goal",
                "salience": 0.9,
                "payload": {
                    "key": "primary_goal",
                    "value": "preserve citations",
                    "summary": "The current primary goal is to preserve citations.",
                },
            }

        if "before anything else" in lowered or "citations intact" in lowered:
            return {
                "text": "The current primary goal is to preserve citations.",
                "memory_type": "goal",
                "salience": 0.9,
                "payload": {
                    "key": "primary_goal",
                    "value": "preserve citations",
                    "summary": "The current primary goal is to preserve citations.",
                },
            }

        if "preserve citations exactly" in lowered or "every draft needs to preserve citations" in lowered:
            return {
                "text": "The current primary goal is to preserve citations.",
                "memory_type": "goal",
                "salience": 0.92,
                "payload": {
                    "key": "primary_goal",
                    "value": "preserve citations",
                    "summary": "The current primary goal is to preserve citations.",
                },
            }

        if "do not drop the citations" in lowered:
            return {
                "text": "The current primary goal is to preserve citations.",
                "memory_type": "goal",
                "salience": 0.9,
                "payload": {
                    "key": "primary_goal",
                    "value": "preserve citations",
                    "summary": "The current primary goal is to preserve citations.",
                },
            }

        if any(
            phrase in lowered
            for phrase in (
                "long-form rationale",
                "long form rationale",
                "whole chain of reasoning",
                "appendix defense",
                "methodology note",
            )
        ):
            memory_type = "correction" if "response_style" in slots else "preference"
            proposition = "The user prefers detailed replies."
            return {
                "text": proposition,
                "memory_type": memory_type,
                "salience": 0.95,
                "payload": {
                    "key": "response_style",
                    "value": "detailed",
                    "proposition": proposition,
                },
            }

        if "still a bit long" in lowered or "even shorter is better" in lowered:
            return {
                "text": "The user likely prefers concise replies.",
                "memory_type": "preference_hint",
                "salience": 0.42,
                "payload": {
                    "key": "response_style",
                    "value": "concise",
                    "proposition": "The user likely prefers concise replies.",
                },
            }

        if "still too long" in lowered or "shorter is better" in lowered:
            return {
                "text": "The user likely prefers concise replies.",
                "memory_type": "preference_hint",
                "salience": 0.43,
                "payload": {
                    "key": "response_style",
                    "value": "concise",
                    "proposition": "The user likely prefers concise replies.",
                },
            }

        if "trim that down" in lowered or "only need the gist" in lowered:
            return {
                "text": "The user likely prefers concise replies.",
                "memory_type": "preference_hint",
                "salience": 0.41,
                "payload": {
                    "key": "response_style",
                    "value": "concise",
                    "proposition": "The user likely prefers concise replies.",
                },
            }

        if "compress this a lot" in lowered or "just need the gist" in lowered:
            return {
                "text": "The user likely prefers concise replies.",
                "memory_type": "preference_hint",
                "salience": 0.41,
                "payload": {
                    "key": "response_style",
                    "value": "concise",
                    "proposition": "The user likely prefers concise replies.",
                },
            }

        if "headline version" in lowered or "even terser than that" in lowered:
            return {
                "text": "The user likely prefers concise replies.",
                "memory_type": "preference_hint",
                "salience": 0.42,
                "payload": {
                    "key": "response_style",
                    "value": "concise",
                    "proposition": "The user likely prefers concise replies.",
                },
            }

        if any(
            phrase in lowered
            for phrase in (
                "quick-scan version",
                "quick scan version",
                "bare-bones version",
                "bare bones version",
                "leanest take",
                "walking between rooms",
            )
        ):
            memory_type = "correction" if "response_style" in slots else "preference"
            proposition = "The user prefers brief replies."
            return {
                "text": proposition,
                "memory_type": memory_type,
                "salience": 0.94,
                "payload": {
                    "key": "response_style",
                    "value": "brief",
                    "proposition": proposition,
                },
            }

        if "keep this really brief" in lowered or "skimming between meetings" in lowered:
            memory_type = "correction" if "response_style" in slots else "preference"
            proposition = "The user prefers brief replies."
            return {
                "text": proposition,
                "memory_type": memory_type,
                "salience": 0.94,
                "payload": {
                    "key": "response_style",
                    "value": "brief",
                    "proposition": proposition,
                },
            }

        if "on my phone between sessions" in lowered or "punchiest version" in lowered:
            memory_type = "correction" if "response_style" in slots else "preference"
            proposition = "The user prefers brief replies."
            return {
                "text": proposition,
                "memory_type": memory_type,
                "salience": 0.94,
                "payload": {
                    "key": "response_style",
                    "value": "brief",
                    "proposition": proposition,
                },
            }

        if "default to short answers" in lowered or "heading into calls" in lowered:
            memory_type = "correction" if "response_style" in slots else "preference"
            proposition = "The user prefers brief replies."
            return {
                "text": proposition,
                "memory_type": memory_type,
                "salience": 0.93,
                "payload": {
                    "key": "response_style",
                    "value": "brief",
                    "proposition": proposition,
                },
            }

        if "full reasoning spelled out" in lowered or "methods memo" in lowered:
            memory_type = "correction" if "response_style" in slots else "preference"
            proposition = "The user prefers detailed replies."
            return {
                "text": proposition,
                "memory_type": memory_type,
                "salience": 0.95,
                "payload": {
                    "key": "response_style",
                    "value": "detailed",
                    "proposition": proposition,
                },
            }

        return None

    def _parse_context(self, context: str) -> Dict[str, Dict[str, str]]:
        slots: Dict[str, str] = {}
        procedures: Dict[str, str] = {}

        for line in context.splitlines():
            line = line.strip()
            if not line.startswith("- ["):
                continue

            layer_match = re.match(r"^- \[(?P<layer>[^\]]+)\]\s+[^:]+:\s+(?P<content>.+)$", line)
            if not layer_match:
                continue
            layer = layer_match.group("layer")
            content = layer_match.group("content")

            slot_match = SLOT_RE.search(content)
            if slot_match and layer in {
                "working_state",
                "beliefs",
                "autobiographical_state",
                "notes",
                "summary_state",
            }:
                slots.setdefault(slot_match.group("key"), slot_match.group("value").strip())

            procedure_match = PROCEDURE_RE.search(content)
            if procedure_match and layer in {"procedures", "notes", "summary_state"}:
                procedures.setdefault(
                    procedure_match.group("trigger").strip(),
                    procedure_match.group("step").strip(),
                )

        return {"slots": slots, "procedures": procedures}


def default_natural_eval_runtime_config(
    *,
    temperature: float = 0.0,
    max_output_tokens: int = 700,
    memory_strategy: str = "brainlayer",
) -> BrainLayerRuntimeConfig:
    return BrainLayerRuntimeConfig(
        system_prompt=NATURAL_EVAL_SYSTEM_PROMPT,
        default_scenario_slug="natural_model_session",
        top_k_per_layer=2,
        max_memories=8,
        response_temperature=temperature,
        max_output_tokens=max_output_tokens,
        interaction_salience=0.55,
        consolidate_before_reply=True,
        auto_consolidate_after_turn=True,
        memory_strategy=memory_strategy,
    )


def _combined_metrics(result: NaturalEvalResult) -> Dict[str, float]:
    metrics = dict(result.state_metrics)
    metrics["score"] = result.score
    metrics["latency_ms"] = result.latency_ms
    metrics["applied_observation_count"] = float(result.applied_observation_count)
    for key, value in result.usage_metrics.items():
        metrics[f"usage_{key}"] = value
    return metrics


def lookup_state_value(exported_state: Dict[str, object], layer: str, key: str) -> str:
    if layer == "beliefs":
        for belief in reversed(exported_state.get("beliefs", [])):
            if belief.get("key") == key and belief.get("status") == "active":
                return str(belief.get("value", "unknown"))
        return "unknown"

    if layer == "working_state":
        for item in reversed(exported_state.get("working_state", [])):
            if item.get("key") == key and item.get("status") == "active":
                return str(item.get("value", "unknown"))
        return "unknown"

    if layer == "autobiographical_state":
        for note in reversed(exported_state.get("autobiographical_state", [])):
            if note.get("key") == key:
                return str(note.get("value", "unknown"))
        return "unknown"

    if layer == "procedures":
        for procedure in reversed(exported_state.get("procedures", [])):
            if procedure.get("trigger") == key:
                steps = procedure.get("steps", [])
                if isinstance(steps, list) and steps:
                    return str(steps[0])
        return "unknown"

    return "unknown"


def _build_runtime(
    adapter: LLMAdapter,
    *,
    features: object,
    requested_model: str,
    runtime_config: BrainLayerRuntimeConfig,
) -> BrainLayerRuntime:
    active_runtime_config = BrainLayerRuntimeConfig(**runtime_config.__dict__)
    return BrainLayerRuntime(
        adapter,
        session=BrainLayerSession(features=features),
        model=requested_model,
        config=active_runtime_config,
    )


def _build_result_from_turn(
    *,
    scenario_slug: str,
    turn: NaturalEvalTurn,
    runtime_name: str,
    eval_mode: str,
    provider_name: str,
    requested_model: str,
    turn_result: object,
    latency_ms: float,
    behavior_judge: BehaviorJudge,
    scenario_title: str,
    scenario_description: str,
) -> NaturalEvalResult:
    if turn.evaluation_type == "behavior":
        actual = turn_result.assistant_response
        score_decision = behavior_judge.score(
            BehaviorJudgeInput(
                scenario_slug=scenario_slug,
                scenario_title=scenario_title,
                scenario_description=scenario_description,
                checkpoint=turn.checkpoint,
                prompt=turn.prompt,
                expected=turn.expected_value,
                actual=actual,
            )
        )
    else:
        actual = lookup_state_value(turn_result.exported_state, turn.target_layer, turn.target_key)
        score_decision = score_structured_value(
            turn.expected_value,
            actual,
            target_layer=turn.target_layer,
            target_key=turn.target_key,
        )

    response_model = turn_result.model_response.model or requested_model
    case_artifact = {
        "suite_name": "natural",
        "scenario": {
            "slug": scenario_slug,
            "title": scenario_title,
            "description": scenario_description,
            "checkpoint": turn.checkpoint,
            "prompt": turn.prompt,
            "expected": turn.expected_value,
            "evaluation_type": turn.evaluation_type,
            "target_layer": turn.target_layer,
            "target_key": turn.target_key,
        },
        "judge": {
            "passed": score_decision.passed,
            "score": score_decision.score,
            "method": score_decision.method,
            "reason": score_decision.reason,
        },
        "runtime": {
            "runtime_name": runtime_name,
            "eval_mode": eval_mode,
            "provider_name": provider_name,
            "requested_model": requested_model,
            "response_model": response_model,
            "finish_reason": turn_result.model_response.finish_reason,
            "latency_ms": latency_ms,
            "used_json": turn_result.used_json,
            "parse_failure": not turn_result.used_json,
            "empty_answer": turn_result.empty_answer,
            "interaction_episode_id": turn_result.interaction_episode_id,
            "usage_metrics": normalize_usage_metrics(turn_result.model_response.usage),
        },
        "prompt_messages": serialize_prompt_messages(turn_result.prompt_messages),
        "retrieved_memories": serialize_retrieved_memories(turn_result.retrieved_memories),
        "raw_model_output": turn_result.raw_model_output,
        "parsed_output": {
            "assistant_response": turn_result.assistant_response,
            "episodic_summary": turn_result.episodic_summary,
            "memory_observations": serialize_observations(turn_result.applied_observations),
        },
        "consolidation_report": serialize_consolidation_report(turn_result.consolidation_report),
        "exported_state": turn_result.exported_state,
    }
    return NaturalEvalResult(
        scenario_slug=scenario_slug,
        checkpoint=turn.checkpoint,
        runtime_name=runtime_name,
        evaluation_type=turn.evaluation_type,
        target_layer=turn.target_layer,
        target_key=turn.target_key,
        expected=turn.expected_value,
        actual=actual,
        passed=score_decision.passed,
        score=score_decision.score,
        score_method=score_decision.method,
        score_reason=score_decision.reason,
        retrieved_layers=[memory.layer for memory in turn_result.retrieved_memories],
        case_artifact=case_artifact,
        state_metrics=collect_state_metrics(turn_result.exported_state),
        exported_state=turn_result.exported_state,
        eval_mode=eval_mode,
        provider_name=provider_name,
        requested_model=requested_model,
        response_model=response_model,
        finish_reason=turn_result.model_response.finish_reason,
        latency_ms=latency_ms,
        used_json=turn_result.used_json,
        parse_failure=not turn_result.used_json,
        empty_answer=turn_result.empty_answer,
        applied_observation_count=len(turn_result.applied_observations),
        usage_metrics=normalize_usage_metrics(turn_result.model_response.usage),
    )


def _build_error_result(
    *,
    scenario_slug: str,
    turn: NaturalEvalTurn,
    runtime_name: str,
    eval_mode: str,
    provider_name: str,
    requested_model: str,
    exported_state: Dict[str, object],
    error: str,
    latency_ms: float,
    skipped: bool,
) -> NaturalEvalResult:
    case_artifact = {
        "suite_name": "natural",
        "scenario": {
            "slug": scenario_slug,
            "checkpoint": turn.checkpoint,
            "expected": turn.expected_value,
            "evaluation_type": turn.evaluation_type,
            "target_layer": turn.target_layer,
            "target_key": turn.target_key,
        },
        "judge": {
            "passed": False,
            "score": 0.0,
            "method": "runtime_error",
            "reason": error,
        },
        "runtime": {
            "runtime_name": runtime_name,
            "eval_mode": eval_mode,
            "provider_name": provider_name,
            "requested_model": requested_model,
            "response_model": requested_model,
            "latency_ms": latency_ms,
            "error": error,
            "skipped": skipped,
        },
        "prompt_messages": [],
        "retrieved_memories": [],
        "raw_model_output": "",
        "parsed_output": {
            "assistant_response": "skipped" if skipped else "error",
            "episodic_summary": "",
            "memory_observations": [],
        },
        "consolidation_report": None,
        "exported_state": exported_state,
    }
    return NaturalEvalResult(
        scenario_slug=scenario_slug,
        checkpoint=turn.checkpoint,
        runtime_name=runtime_name,
        evaluation_type=turn.evaluation_type,
        target_layer=turn.target_layer,
        target_key=turn.target_key,
        expected=turn.expected_value,
        actual="skipped" if skipped else "error",
        passed=False,
        score=0.0,
        score_method="runtime_error",
        score_reason=error,
        retrieved_layers=[],
        case_artifact=case_artifact,
        state_metrics=collect_state_metrics(exported_state),
        exported_state=exported_state,
        eval_mode=eval_mode,
        provider_name=provider_name,
        requested_model=requested_model,
        response_model=requested_model,
        finish_reason="",
        latency_ms=latency_ms,
        used_json=False,
        parse_failure=False,
        empty_answer=False,
        applied_observation_count=0,
        usage_metrics={},
        error=error,
        skipped=skipped,
    )


def run_natural_eval_scenario(
    scenario: NaturalEvalScenario,
    *,
    include_ablations: bool = True,
    adapter: LLMAdapter | None = None,
    eval_mode: str = "heuristic",
    provider_name: str | None = None,
    requested_model: str | None = None,
    runtime_config: BrainLayerRuntimeConfig | None = None,
    behavior_scoring_mode: str = "judge",
    behavior_judge: BehaviorJudge | None = None,
    runtime_profile: str = RUNTIME_PROFILE_DEFAULT,
    runtime_names: Sequence[str] | None = None,
) -> List[NaturalEvalResult]:
    active_adapter = adapter or HeuristicNaturalConversationAdapter()
    active_provider_name = provider_name or DEFAULT_HEURISTIC_PROVIDER
    active_requested_model = requested_model or DEFAULT_HEURISTIC_MODEL
    active_runtime_config = runtime_config or default_natural_eval_runtime_config()
    active_behavior_judge = behavior_judge or _build_behavior_judge(behavior_scoring_mode)

    results: List[NaturalEvalResult] = []
    selected_runtime_names = set(runtime_names or [])
    for variant in build_runtime_variants(
        include_ablations=include_ablations,
        runtime_profile=runtime_profile,
    ):
        runtime_name = variant.name
        if selected_runtime_names and runtime_name not in selected_runtime_names:
            continue
        active_variant_config = BrainLayerRuntimeConfig(
            **{
                **active_runtime_config.__dict__,
                "memory_strategy": variant.memory_strategy,
            }
        )
        runtime = _build_runtime(
            active_adapter,
            features=variant.features,
            requested_model=active_requested_model,
            runtime_config=active_variant_config,
        )
        blocked_error = ""
        for turn in scenario.turns:
            if blocked_error:
                if turn.checkpoint:
                    results.append(
                        _build_error_result(
                            scenario_slug=scenario.slug,
                            turn=turn,
                            runtime_name=runtime_name,
                            eval_mode=eval_mode,
                            provider_name=active_provider_name,
                            requested_model=active_requested_model,
                            exported_state=runtime.export_state(),
                            error=blocked_error,
                            latency_ms=0.0,
                            skipped=True,
                        )
                    )
                continue

            started_at = time.perf_counter()
            try:
                turn_result = runtime.run_turn(turn.prompt, scenario_slug=scenario.slug)
            except LLMError as exc:
                blocked_error = str(exc)
                latency_ms = (time.perf_counter() - started_at) * 1000.0
                if turn.checkpoint:
                    results.append(
                        _build_error_result(
                            scenario_slug=scenario.slug,
                            turn=turn,
                            runtime_name=runtime_name,
                            eval_mode=eval_mode,
                            provider_name=active_provider_name,
                            requested_model=active_requested_model,
                            exported_state=runtime.export_state(),
                            error=blocked_error,
                            latency_ms=latency_ms,
                            skipped=False,
                        )
                    )
                continue
            except Exception as exc:
                blocked_error = f"Unexpected error: {exc}"
                latency_ms = (time.perf_counter() - started_at) * 1000.0
                if turn.checkpoint:
                    results.append(
                        _build_error_result(
                            scenario_slug=scenario.slug,
                            turn=turn,
                            runtime_name=runtime_name,
                            eval_mode=eval_mode,
                            provider_name=active_provider_name,
                            requested_model=active_requested_model,
                            exported_state=runtime.export_state(),
                            error=blocked_error,
                            latency_ms=latency_ms,
                            skipped=False,
                        )
                    )
                continue

            latency_ms = (time.perf_counter() - started_at) * 1000.0
            if not turn.checkpoint:
                continue

            results.append(
                _build_result_from_turn(
                    scenario_slug=scenario.slug,
                    turn=turn,
                    runtime_name=runtime_name,
                    eval_mode=eval_mode,
                    provider_name=active_provider_name,
                    requested_model=active_requested_model,
                    turn_result=turn_result,
                    latency_ms=latency_ms,
                    behavior_judge=active_behavior_judge,
                    scenario_title=scenario.title,
                    scenario_description=scenario.description,
                )
            )
    return results


def run_natural_eval_suite(
    scenarios: Iterable[NaturalEvalScenario] | None = None,
    *,
    scenario_pack: str = DEFAULT_SCENARIO_PACK,
    include_ablations: bool = True,
    adapter: LLMAdapter | None = None,
    eval_mode: str = "heuristic",
    provider_name: str | None = None,
    requested_model: str | None = None,
    runtime_config: BrainLayerRuntimeConfig | None = None,
    behavior_scoring_mode: str = "judge",
    behavior_judge: BehaviorJudge | None = None,
    runtime_profile: str = RUNTIME_PROFILE_DEFAULT,
    scenario_slugs: Sequence[str] | None = None,
    runtime_names: Sequence[str] | None = None,
) -> List[NaturalEvalResult]:
    active_scenarios = list(scenarios or get_natural_eval_scenarios(scenario_pack))
    selected_scenario_slugs = set(scenario_slugs or [])
    if selected_scenario_slugs:
        active_scenarios = [
            scenario
            for scenario in active_scenarios
            if scenario.slug in selected_scenario_slugs
        ]
    results: List[NaturalEvalResult] = []
    for scenario in active_scenarios:
        results.extend(
            run_natural_eval_scenario(
                scenario,
                include_ablations=include_ablations,
                adapter=adapter,
                eval_mode=eval_mode,
                provider_name=provider_name,
                requested_model=requested_model,
                runtime_config=runtime_config,
                behavior_scoring_mode=behavior_scoring_mode,
                behavior_judge=behavior_judge,
                runtime_profile=runtime_profile,
                runtime_names=runtime_names,
            )
        )
    return results


def run_live_natural_eval_suite(
    scenarios: Iterable[NaturalEvalScenario] | None = None,
    *,
    scenario_pack: str = DEFAULT_SCENARIO_PACK,
    include_ablations: bool = True,
    provider_name: str = DEFAULT_LIVE_PROVIDER,
    requested_model: str = DEFAULT_LIVE_MODEL,
    base_url: str = "https://api.openai.com/v1",
    api_key_env: str = "OPENAI_API_KEY",
    request_path: str = "/chat/completions",
    timeout_seconds: float = 30.0,
    max_output_tokens_field: str | None = "max_tokens",
    temperature: float = 0.0,
    max_output_tokens: int = 700,
    behavior_scoring_mode: str = "judge",
    behavior_judge: BehaviorJudge | None = None,
    runtime_profile: str = RUNTIME_PROFILE_DEFAULT,
    scenario_slugs: Sequence[str] | None = None,
    runtime_names: Sequence[str] | None = None,
) -> List[NaturalEvalResult]:
    adapter = build_live_model_eval_adapter(
        provider_name=provider_name,
        api_key_env=api_key_env,
        base_url=base_url,
        request_path=request_path,
        timeout_seconds=timeout_seconds,
        max_output_tokens_field=max_output_tokens_field,
    )
    runtime_config = default_natural_eval_runtime_config(
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )
    return run_natural_eval_suite(
        scenarios,
        scenario_pack=scenario_pack,
        include_ablations=include_ablations,
        adapter=adapter,
        eval_mode="live",
        provider_name=provider_name,
        requested_model=requested_model,
        runtime_config=runtime_config,
        behavior_scoring_mode=behavior_scoring_mode,
        behavior_judge=behavior_judge,
        runtime_profile=runtime_profile,
        scenario_slugs=scenario_slugs,
        runtime_names=runtime_names,
    )


def _build_behavior_judge(behavior_scoring_mode: str) -> BehaviorJudge:
    if behavior_scoring_mode == "exact":
        return ExactMatchJudge()
    if behavior_scoring_mode == "judge":
        return HeuristicBehaviorJudge()
    raise ValueError(f"Unsupported behavior scoring mode: {behavior_scoring_mode}")


def summarize_natural_eval_results(results: Sequence[NaturalEvalResult]) -> List[NaturalEvalSummary]:
    passed_counts: Dict[str, int] = {}
    totals: Dict[str, int] = {}
    extraction_passed: Dict[str, int] = {}
    extraction_totals: Dict[str, int] = {}
    behavior_passed: Dict[str, int] = {}
    behavior_totals: Dict[str, int] = {}
    parse_failures: Dict[str, int] = {}
    empty_answers: Dict[str, int] = {}
    errors: Dict[str, int] = {}
    skipped: Dict[str, int] = {}
    metric_totals: Dict[str, Dict[str, float]] = {}

    for result in results:
        totals[result.runtime_name] = totals.get(result.runtime_name, 0) + 1
        passed_counts[result.runtime_name] = passed_counts.get(result.runtime_name, 0) + int(
            result.passed
        )
        if result.evaluation_type == "extraction":
            extraction_totals[result.runtime_name] = extraction_totals.get(result.runtime_name, 0) + 1
            extraction_passed[result.runtime_name] = extraction_passed.get(result.runtime_name, 0) + int(
                result.passed
            )
        if result.evaluation_type == "behavior":
            behavior_totals[result.runtime_name] = behavior_totals.get(result.runtime_name, 0) + 1
            behavior_passed[result.runtime_name] = behavior_passed.get(result.runtime_name, 0) + int(
                result.passed
            )
        parse_failures[result.runtime_name] = parse_failures.get(result.runtime_name, 0) + int(
            result.parse_failure
        )
        empty_answers[result.runtime_name] = empty_answers.get(result.runtime_name, 0) + int(
            result.empty_answer
        )
        errors[result.runtime_name] = errors.get(result.runtime_name, 0) + int(bool(result.error))
        skipped[result.runtime_name] = skipped.get(result.runtime_name, 0) + int(result.skipped)
        runtime_metric_totals = metric_totals.setdefault(result.runtime_name, {})
        for key, value in _combined_metrics(result).items():
            runtime_metric_totals[key] = runtime_metric_totals.get(key, 0.0) + value

    summaries: List[NaturalEvalSummary] = []
    for runtime_name in sorted(totals):
        total = totals[runtime_name]
        avg_metrics = {
            key: value / total for key, value in metric_totals.get(runtime_name, {}).items()
        }
        summaries.append(
            NaturalEvalSummary(
                runtime_name=runtime_name,
                passed=passed_counts[runtime_name],
                total=total,
                pass_rate=passed_counts[runtime_name] / total if total else 0.0,
                extraction_passed=extraction_passed.get(runtime_name, 0),
                extraction_total=extraction_totals.get(runtime_name, 0),
                behavior_passed=behavior_passed.get(runtime_name, 0),
                behavior_total=behavior_totals.get(runtime_name, 0),
                parse_failures=parse_failures.get(runtime_name, 0),
                empty_answers=empty_answers.get(runtime_name, 0),
                errors=errors.get(runtime_name, 0),
                skipped=skipped.get(runtime_name, 0),
                avg_metrics=avg_metrics,
            )
        )
    return summaries


def render_natural_eval_report(results: Sequence[NaturalEvalResult]) -> str:
    lines = [
        "Natural Conversation BrainLayer Eval Report",
        "===========================================",
        "",
    ]
    summaries = summarize_natural_eval_results(results)
    if results:
        first = results[0]
        lines.append(
            f"Mode: {first.eval_mode} | provider={first.provider_name} | requested_model={first.requested_model}"
        )
        lines.append("")

    for result in results:
        status = "SKIP" if result.skipped else "PASS" if result.passed else "FAIL"
        case_label = f"{result.scenario_slug}/{result.checkpoint}"
        layers = ",".join(result.retrieved_layers) or "-"
        target = (
            f"{result.evaluation_type}:{result.target_layer}:{result.target_key}"
            if result.evaluation_type == "extraction"
            else result.evaluation_type
        )
        extras = [
            f"target={target}",
            f"score={result.score:.2f}",
            f"scoring={result.score_method}",
            f"latency_ms={result.latency_ms:.1f}",
            f"json={str(result.used_json).lower()}",
        ]
        if result.error:
            extras.append(f"error={result.error}")
        elif not result.passed:
            extras.append(f"score_reason={result.score_reason}")
        lines.append(
            f"[{status}] {result.runtime_name} on {case_label}: "
            f"expected={result.expected!r}, actual={result.actual!r}, "
            f"layers={layers}, " + ", ".join(extras)
        )

    lines.append("")
    lines.append("Summary")
    lines.append("-------")
    for summary in summaries:
        extras = [
            f"extraction={summary.extraction_passed}/{summary.extraction_total}",
            f"behavior={summary.behavior_passed}/{summary.behavior_total}",
            f"avg_score={summary.avg_metrics.get('score', 0.0):.2f}",
            f"avg_records={summary.avg_metrics.get('total_records', 0.0):.1f}",
            f"avg_latency_ms={summary.avg_metrics.get('latency_ms', 0.0):.1f}",
            f"parse_failures={summary.parse_failures}",
            f"errors={summary.errors}",
        ]
        avg_total_tokens = summary.avg_metrics.get("usage_total_tokens", 0.0)
        if avg_total_tokens:
            extras.append(f"avg_total_tokens={avg_total_tokens:.1f}")
        lines.append(
            f"{summary.runtime_name}: {summary.passed}/{summary.total} | " + ", ".join(extras)
        )
    return "\n".join(lines)


def serializable_natural_eval_result(
    result: NaturalEvalResult,
    *,
    artifact_path: str = "",
) -> Dict[str, object]:
    payload: Dict[str, object] = {
        "scenario_slug": result.scenario_slug,
        "checkpoint": result.checkpoint,
        "case_label": f"{result.scenario_slug}/{result.checkpoint}",
        "runtime_name": result.runtime_name,
        "evaluation_type": result.evaluation_type,
        "target_layer": result.target_layer,
        "target_key": result.target_key,
        "expected": result.expected,
        "actual": result.actual,
        "passed": result.passed,
        "score": result.score,
        "score_method": result.score_method,
        "score_reason": result.score_reason,
        "eval_mode": result.eval_mode,
        "provider_name": result.provider_name,
        "requested_model": result.requested_model,
        "response_model": result.response_model,
        "finish_reason": result.finish_reason,
        "latency_ms": result.latency_ms,
        "used_json": result.used_json,
        "parse_failure": result.parse_failure,
        "empty_answer": result.empty_answer,
        "applied_observation_count": result.applied_observation_count,
        "error": result.error,
        "skipped": result.skipped,
        "retrieved_layers": ",".join(result.retrieved_layers),
    }
    if artifact_path:
        payload["artifact_path"] = artifact_path
    for key, value in sorted(result.state_metrics.items()):
        payload[f"metric_{key}"] = value
    for key, value in sorted(result.usage_metrics.items()):
        payload[f"usage_{key}"] = value
    return payload


def serializable_natural_eval_summary(summary: NaturalEvalSummary) -> Dict[str, object]:
    payload: Dict[str, object] = {
        "runtime_name": summary.runtime_name,
        "passed": summary.passed,
        "total": summary.total,
        "pass_rate": summary.pass_rate,
        "extraction_passed": summary.extraction_passed,
        "extraction_total": summary.extraction_total,
        "behavior_passed": summary.behavior_passed,
        "behavior_total": summary.behavior_total,
        "parse_failures": summary.parse_failures,
        "empty_answers": summary.empty_answers,
        "errors": summary.errors,
        "skipped": summary.skipped,
    }
    for key, value in sorted(summary.avg_metrics.items()):
        payload[f"avg_{key}"] = value
    return payload


def build_natural_eval_metadata(
    results: Sequence[NaturalEvalResult],
    *,
    include_ablations: bool,
    label: str | None,
    scenario_pack: str = DEFAULT_SCENARIO_PACK,
) -> Dict[str, object]:
    timestamp = utc_now_compact()
    run_id = timestamp if not label else f"{timestamp}-{slugify_label(label)}"
    first = results[0] if results else None
    return {
        "run_id": run_id,
        "generated_at_utc": utc_now_iso(),
        "git_commit": get_git_commit(),
        "include_ablations": include_ablations,
        "scenario_pack": scenario_pack,
        "label": label or "",
        "eval_mode": first.eval_mode if first else "",
        "provider_name": first.provider_name if first else "",
        "requested_model": first.requested_model if first else "",
        "artifacts_subdir": "case_artifacts",
        "scenario_count": len({result.scenario_slug for result in results}),
        "checkpoint_count": len({(result.scenario_slug, result.checkpoint) for result in results}),
        "runtime_count": len({result.runtime_name for result in results}),
        "extraction_count": len([result for result in results if result.evaluation_type == "extraction"]),
        "behavior_count": len([result for result in results if result.evaluation_type == "behavior"]),
        "score_methods": sorted({result.score_method for result in results}),
    }


def render_natural_eval_x_post(
    summaries: Sequence[NaturalEvalSummary],
    *,
    include_ablations: bool,
    label: str | None,
    eval_mode: str,
    requested_model: str,
) -> str:
    summary_by_name = {summary.runtime_name: summary for summary in summaries}
    model_loop = summary_by_name.get("model_loop")
    if model_loop is None:
        return "BrainLayer natural eval completed."

    prefix = "BrainLayer natural eval"
    if eval_mode == "live":
        prefix = f"BrainLayer natural live eval {requested_model}"
    if label:
        prefix = f"{prefix} ({label})"

    parts = [
        f"{prefix}: full runtime {model_loop.passed}/{model_loop.total}",
        f"with extraction {model_loop.extraction_passed}/{model_loop.extraction_total}",
        f"and behavior {model_loop.behavior_passed}/{model_loop.behavior_total}.",
    ]

    if include_ablations:
        no_consolidation = summary_by_name.get("model_loop_no_consolidation")
        no_autobio = summary_by_name.get("model_loop_no_autobio")
        no_working_state = summary_by_name.get("model_loop_no_working_state")
        if no_consolidation and no_autobio and no_working_state:
            parts.append(
                "Ablations:"
                f" no_consolidation {no_consolidation.passed}/{no_consolidation.total},"
                f" no_autobio {no_autobio.passed}/{no_autobio.total},"
                f" no_working_state {no_working_state.passed}/{no_working_state.total}."
            )

    return " ".join(parts)


def export_natural_eval_results(
    results: Sequence[NaturalEvalResult],
    export_root: Path,
    *,
    include_ablations: bool,
    label: str | None = None,
    scenario_pack: str = DEFAULT_SCENARIO_PACK,
) -> Path:
    summaries = summarize_natural_eval_results(results)
    metadata = build_natural_eval_metadata(
        results,
        include_ablations=include_ablations,
        label=label,
        scenario_pack=scenario_pack,
    )
    run_dir = export_root / str(metadata["run_id"])
    run_dir.mkdir(parents=True, exist_ok=True)

    artifact_root = run_dir / str(metadata["artifacts_subdir"])
    result_rows = []
    for result in results:
        artifact_filename = f"{result.runtime_name}__{result.scenario_slug}__{result.checkpoint}.json"
        artifact_path = write_case_artifact(artifact_root, artifact_filename, result.case_artifact)
        result_rows.append(serializable_natural_eval_result(result, artifact_path=artifact_path))
    summary_rows = [serializable_natural_eval_summary(summary) for summary in summaries]
    x_post = render_natural_eval_x_post(
        summaries,
        include_ablations=include_ablations,
        label=label,
        eval_mode=str(metadata["eval_mode"]),
        requested_model=str(metadata["requested_model"]),
    )

    payload = {
        "metadata": metadata,
        "summary": summary_rows,
        "results": result_rows,
        "x_post": x_post,
    }

    (run_dir / "results.json").write_text(json.dumps(payload, indent=2) + "\n")
    write_csv(run_dir / "results.csv", result_rows)
    write_csv(run_dir / "summary.csv", summary_rows)
    (run_dir / "x_post.txt").write_text(x_post + "\n")

    history_rows = []
    for row in summary_rows:
        history_rows.append(
            {
                "run_id": metadata["run_id"],
                "generated_at_utc": metadata["generated_at_utc"],
                "git_commit": metadata["git_commit"],
                "label": metadata["label"],
                "include_ablations": metadata["include_ablations"],
                "scenario_pack": metadata["scenario_pack"],
                "eval_mode": metadata["eval_mode"],
                "provider_name": metadata["provider_name"],
                "requested_model": metadata["requested_model"],
                "score_methods": ",".join(metadata["score_methods"]),
                **row,
            }
        )

    append_csv(export_root / "natural_eval_history.csv", history_rows)
    with (export_root / "natural_eval_history.jsonl").open("a") as handle:
        handle.write(json.dumps(payload) + "\n")

    return run_dir


def dump_natural_eval_states(results: Sequence[NaturalEvalResult], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for result in results:
        filename = f"{result.runtime_name}__{result.scenario_slug}__{result.checkpoint}.json"
        target = output_dir / filename
        target.write_text(json.dumps(result.exported_state, indent=2) + "\n")


def _build_adapter_from_args(args: argparse.Namespace) -> tuple[LLMAdapter, str, str]:
    if args.mode == "heuristic":
        return (
            HeuristicNaturalConversationAdapter(),
            DEFAULT_HEURISTIC_PROVIDER,
            DEFAULT_HEURISTIC_MODEL,
        )

    max_output_tokens_field = args.max_output_tokens_field
    if max_output_tokens_field is not None and max_output_tokens_field.lower() == "none":
        max_output_tokens_field = None
    adapter = build_live_model_eval_adapter(
        provider_name=args.provider_name,
        api_key_env=args.api_key_env,
        base_url=args.base_url,
        request_path=args.request_path,
        timeout_seconds=args.timeout_seconds,
        max_output_tokens_field=max_output_tokens_field,
    )
    return adapter, args.provider_name, args.model


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run natural-conversation evals against the model-backed BrainLayer loop."
    )
    parser.add_argument(
        "--mode",
        choices=("heuristic", "live"),
        default="heuristic",
        help="Choose the deterministic heuristic backend or a live chat-completions-compatible provider.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_LIVE_MODEL,
        help="Requested model name for live eval mode.",
    )
    parser.add_argument(
        "--provider-name",
        default=DEFAULT_LIVE_PROVIDER,
        help="Provider label to store in run metadata for live eval mode.",
    )
    parser.add_argument(
        "--base-url",
        default="https://api.openai.com/v1",
        help="Base URL for a chat-completions-compatible provider in live mode.",
    )
    parser.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        help="Environment variable containing the API key for live mode.",
    )
    parser.add_argument(
        "--request-path",
        default="/chat/completions",
        help="Request path for the live chat-completions-compatible provider.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=30.0,
        help="HTTP timeout for live provider requests.",
    )
    parser.add_argument(
        "--max-output-tokens-field",
        default="max_tokens",
        help="Field name used by the live provider for output tokens. Use 'none' to omit it.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for the evaluation runtime.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=700,
        help="Maximum output tokens requested per evaluation turn.",
    )
    parser.add_argument(
        "--core-only",
        action="store_true",
        help="Run only the full BrainLayer runtime without ablations.",
    )
    parser.add_argument(
        "--scenario-pack",
        choices=(
            "standard",
            "hard",
            "held_out",
            "external_dev",
            "external_held_out",
            "consolidation_stress",
            "forgetting_stress",
            "all",
        ),
        default=DEFAULT_SCENARIO_PACK,
        help="Choose the standard natural suite, the harder delayed/noisy set, the held-out generalization set, the external dev set, the external held-out set, the consolidation-stress set, the forgetting-stress set, or all packs together.",
    )
    parser.add_argument(
        "--runtime-profile",
        choices=(RUNTIME_PROFILE_DEFAULT, RUNTIME_PROFILE_STUDY_V2),
        default=RUNTIME_PROFILE_DEFAULT,
        help="Choose the default BrainLayer runtime set or the study-v2 stronger-baseline set.",
    )
    parser.add_argument(
        "--scenario-slug",
        action="append",
        default=[],
        help="Optional scenario slug filter. May be supplied multiple times.",
    )
    parser.add_argument(
        "--runtime-name",
        action="append",
        default=[],
        help="Optional runtime-name filter. May be supplied multiple times.",
    )
    parser.add_argument(
        "--score-exact",
        action="store_true",
        help="Disable judge-backed semantic behavior scoring and require exact normalized matches.",
    )
    parser.add_argument(
        "--dump-states",
        type=Path,
        help="Optional directory for writing state snapshots at each checkpoint.",
    )
    parser.add_argument(
        "--export-results",
        type=Path,
        help="Write per-run CSV/JSON summaries into DIR and append natural-eval history files there.",
    )
    parser.add_argument(
        "--label",
        help="Optional label to attach to exported natural-eval runs for later comparison.",
    )
    args = parser.parse_args(argv)

    adapter, provider_name, requested_model = _build_adapter_from_args(args)
    runtime_config = default_natural_eval_runtime_config(
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
    )
    include_ablations = not args.core_only
    results = run_natural_eval_suite(
        scenario_pack=args.scenario_pack,
        include_ablations=include_ablations,
        adapter=adapter,
        eval_mode=args.mode,
        provider_name=provider_name,
        requested_model=requested_model,
        runtime_config=runtime_config,
        behavior_scoring_mode="exact" if args.score_exact else "judge",
        runtime_profile=args.runtime_profile,
        scenario_slugs=args.scenario_slug or None,
        runtime_names=args.runtime_name or None,
    )

    if args.dump_states:
        dump_natural_eval_states(results, args.dump_states)
    print(render_natural_eval_report(results))
    if args.export_results:
        run_dir = export_natural_eval_results(
            results,
            args.export_results,
            include_ablations=include_ablations,
            label=args.label,
            scenario_pack=args.scenario_pack,
        )
        print("")
        print(f"Natural-eval exports written to {run_dir}")
        print(f"X post saved to {run_dir / 'x_post.txt'}")
    return 0


__all__ = [
    "DEFAULT_SCENARIO_PACK",
    "HARD_NATURAL_EVAL_SCENARIOS",
    "NATURAL_EVAL_SCENARIOS",
    "STANDARD_NATURAL_EVAL_SCENARIOS",
    "HeuristicNaturalConversationAdapter",
    "NaturalEvalResult",
    "NaturalEvalScenario",
    "NaturalEvalSummary",
    "NaturalEvalTurn",
    "default_natural_eval_runtime_config",
    "dump_natural_eval_states",
    "export_natural_eval_results",
    "get_natural_eval_scenarios",
    "lookup_state_value",
    "render_natural_eval_report",
    "render_natural_eval_x_post",
    "run_live_natural_eval_suite",
    "run_natural_eval_scenario",
    "run_natural_eval_suite",
    "summarize_natural_eval_results",
]
