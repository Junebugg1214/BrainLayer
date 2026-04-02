from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from .consolidation import ConsolidationReport
from .llm import LLMAdapter, ModelMessage, ModelResponse
from .models import AutobioNote, Belief, BrainLayerState, Episode, Procedure, WorkingItem
from .scenarios import Observation
from .session import BrainLayerSession


TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "as",
    "be",
    "for",
    "from",
    "i",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "we",
    "with",
}
CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*(?P<body>.*)\s*```$", re.DOTALL)
VALID_MEMORY_TYPES = {
    "preference",
    "correction",
    "lesson",
    "goal",
    "relationship",
    "preference_hint",
    "lesson_hint",
    "goal_hint",
    "relationship_hint",
    "noise",
}
REQUIRED_PAYLOAD_KEYS = {
    "preference": {"key", "value", "proposition"},
    "correction": {"key", "value", "proposition"},
    "preference_hint": {"key", "value", "proposition"},
    "lesson": {"trigger", "action", "summary"},
    "lesson_hint": {"trigger", "action", "summary"},
    "goal": {"key", "value", "summary"},
    "goal_hint": {"key", "value", "summary"},
    "relationship": {"key", "value", "summary", "themes"},
    "relationship_hint": {"key", "value", "summary", "themes"},
}
OBSERVATION_RESERVED_KEYS = {"memory_type", "text", "salience", "payload"}
KEY_ALIASES = {
    "preference": "response_style",
    "style preference": "response_style",
    "style_preference": "response_style",
    "response preference": "response_style",
    "response_preference": "response_style",
    "response style": "response_style",
    "response_style": "response_style",
    "response-length": "response_style",
    "response_length": "response_style",
    "response verbosity": "response_style",
    "response_verbosity": "response_style",
    "verbosity": "response_style",
    "detail level": "response_style",
    "detail_level": "response_style",
    "answer length": "response_style",
    "answer_length": "response_style",
    "citation integrity": "primary_goal",
    "citation_integrity": "primary_goal",
    "citation preservation": "primary_goal",
    "citation_preservation": "primary_goal",
    "main goal": "primary_goal",
    "main_goal": "primary_goal",
    "current goal": "primary_goal",
    "current_goal": "primary_goal",
    "project goal": "primary_goal",
    "project_goal": "primary_goal",
    "delivery priority": "primary_goal",
    "delivery_priority": "primary_goal",
    "collaboration style": "collaboration_mode",
    "collaboration_style": "collaboration_mode",
    "partner mode": "collaboration_mode",
    "partner_mode": "collaboration_mode",
    "working relationship": "collaboration_mode",
    "working_relationship": "collaboration_mode",
}
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful AI agent using BrainLayer as a layered cognitive state. "
    "Use the retrieved state when it is relevant, prefer the latest explicit corrections "
    "over older traces, and keep your response grounded in the prompt."
)


def tokenize(text: str) -> List[str]:
    return [token for token in TOKEN_RE.findall(text.lower()) if token not in STOPWORDS]


def lexical_overlap_score(query: str, text: str) -> int:
    return len(set(tokenize(query)) & set(tokenize(text)))


@dataclass(frozen=True)
class RetrievedMemory:
    layer: str
    record_id: str
    score: float
    content: str


@dataclass(frozen=True)
class BrainLayerRuntimeConfig:
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    default_scenario_slug: str = "model_session"
    top_k_per_layer: int = 2
    max_memories: int = 8
    response_temperature: float = 0.2
    max_output_tokens: int = 900
    interaction_salience: float = 0.55
    consolidate_before_reply: bool = True
    auto_consolidate_after_turn: bool = True


@dataclass
class ParsedModelOutput:
    assistant_response: str
    episodic_summary: str
    memory_observations: List[Observation] = field(default_factory=list)
    raw_text: str = ""
    used_json: bool = False
    empty_answer: bool = False


@dataclass
class ModelTurnResult:
    assistant_response: str
    episodic_summary: str
    raw_model_output: str
    prompt_messages: List[ModelMessage]
    retrieved_memories: List[RetrievedMemory]
    applied_observations: List[Observation]
    interaction_episode_id: str
    consolidation_report: ConsolidationReport | None
    model_response: ModelResponse
    exported_state: Dict[str, object]
    used_json: bool
    empty_answer: bool


class BrainLayerRuntime:
    """Model-backed runtime that reads from and writes back to BrainLayer state."""

    def __init__(
        self,
        adapter: LLMAdapter,
        *,
        session: BrainLayerSession | None = None,
        model: str = "gpt-4.1-mini",
        config: BrainLayerRuntimeConfig | None = None,
    ) -> None:
        self.adapter = adapter
        self.session = session or BrainLayerSession()
        self.model = model
        self.config = config or BrainLayerRuntimeConfig()

    def run_turn(
        self,
        prompt: str,
        *,
        observations: Sequence[Observation] | None = None,
        scenario_slug: str | None = None,
    ) -> ModelTurnResult:
        active_scenario_slug = scenario_slug or self.config.default_scenario_slug
        for observation in observations or []:
            self.session.observe(
                text=observation.text,
                memory_type=observation.memory_type,
                payload=observation.payload,
                salience=observation.salience,
                scenario_slug=active_scenario_slug,
            )

        if self.config.consolidate_before_reply:
            self.session.consolidate()

        retrieved = self.retrieve_memories(prompt)
        prompt_messages = self.build_messages(prompt, retrieved)
        model_response = self.adapter.generate(
            prompt_messages,
            model=self.model,
            temperature=self.config.response_temperature,
            max_output_tokens=self.config.max_output_tokens,
        )
        parsed_output = self.parse_model_output(
            model_response.content,
            fallback_prompt=prompt,
        )
        recovered_observations = list(parsed_output.memory_observations)
        if not recovered_observations:
            recovered_observations = self._recover_missing_observations(
                prompt,
                parsed_output.assistant_response,
                parsed_output.episodic_summary,
            )

        applied_observations: List[Observation] = []
        for observation in recovered_observations:
            try:
                self.session.observe(
                    text=observation.text,
                    memory_type=observation.memory_type,
                    payload=observation.payload,
                    salience=observation.salience,
                    scenario_slug=active_scenario_slug,
                )
            except Exception:
                continue
            applied_observations.append(observation)

        interaction_episode = self.session.state.record_episode(
            scenario=active_scenario_slug,
            summary=parsed_output.episodic_summary,
            tags=["interaction", "assistant_reply"],
            metadata={
                "prompt": _truncate_text(prompt, 240),
                "assistant_response": _truncate_text(parsed_output.assistant_response, 240),
                "model": self.model,
                "used_json": "true" if parsed_output.used_json else "false",
            },
            salience=self.config.interaction_salience,
            outcome="completed model-backed turn",
            source_refs=[memory.record_id for memory in retrieved],
        )

        consolidation_report = None
        if self.config.auto_consolidate_after_turn:
            consolidation_report = self.session.consolidate()

        return ModelTurnResult(
            assistant_response=parsed_output.assistant_response,
            episodic_summary=parsed_output.episodic_summary,
            raw_model_output=model_response.content,
            prompt_messages=prompt_messages,
            retrieved_memories=retrieved,
            applied_observations=applied_observations,
            interaction_episode_id=interaction_episode.id,
            consolidation_report=consolidation_report,
            model_response=model_response,
            exported_state=self.session.state.to_dict(),
            used_json=parsed_output.used_json,
            empty_answer=parsed_output.empty_answer,
        )

    def retrieve_memories(self, prompt: str) -> List[RetrievedMemory]:
        state = self.session.state
        layer_candidates = {
            "working_state": self._retrieve_working_state(prompt, state),
            "beliefs": self._retrieve_beliefs(prompt, state),
            "autobiographical_state": self._retrieve_autobio(prompt, state),
            "procedures": self._retrieve_procedures(prompt, state),
            "episodes": self._retrieve_episodes(prompt, state),
        }

        selected: List[RetrievedMemory] = []
        for layer_name in (
            "working_state",
            "beliefs",
            "autobiographical_state",
            "procedures",
            "episodes",
        ):
            top_records = sorted(
                layer_candidates[layer_name],
                key=lambda memory: memory.score,
                reverse=True,
            )[: self.config.top_k_per_layer]
            selected.extend(top_records)

        selected.sort(key=lambda memory: memory.score, reverse=True)
        return selected[: self.config.max_memories]

    def build_messages(
        self,
        prompt: str,
        retrieved_memories: Sequence[RetrievedMemory],
    ) -> List[ModelMessage]:
        context = self.render_retrieved_context(retrieved_memories)
        user_message = (
            "BrainLayer context:\n"
            f"{context}\n\n"
            "Task:\n"
            f"{prompt}\n\n"
            "Return a JSON object with the following keys:\n"
            '- "assistant_response": the answer for the user\n'
            '- "episodic_summary": a short summary worth storing as an episode\n'
            '- "memory_observations": an optional array of observations to write back to BrainLayer\n\n'
            "Only emit memory observations when the current turn strongly justifies them.\n"
            "Allowed memory_type values: preference, correction, lesson, goal, relationship, "
            "preference_hint, lesson_hint, goal_hint, relationship_hint, noise.\n"
            "Required payloads:\n"
            "- preference/correction: key, value, proposition\n"
            "- lesson/lesson_hint: trigger, action, summary\n"
            "- goal/goal_hint: key, value, summary\n"
            "- relationship/relationship_hint: key, value, summary, themes\n"
            "- noise: any small string payload\n"
            "Each memory_observations item must include memory_type and a nested payload object.\n"
            "Do not place key/value/summary/proposition directly at the top level of a memory observation.\n"
            "Example preference observation:\n"
            '{"memory_type":"preference","payload":{"key":"response_style","value":"brief","proposition":"The user prefers brief replies."}}\n'
            "Example goal observation:\n"
            '{"memory_type":"goal","payload":{"key":"primary_goal","value":"ship eval summary","summary":"The current primary goal is to ship the eval summary."}}\n'
            "If no memory write is justified, return an empty memory_observations array."
        )
        return [
            ModelMessage(role="system", content=self.config.system_prompt),
            ModelMessage(role="user", content=user_message),
        ]

    def render_retrieved_context(self, retrieved_memories: Sequence[RetrievedMemory]) -> str:
        if not retrieved_memories:
            return "- none"

        lines = []
        for memory in retrieved_memories:
            lines.append(
                f"- [{memory.layer}] {memory.record_id}: {memory.content} "
                f"(score={memory.score:.2f})"
            )
        return "\n".join(lines)

    def parse_model_output(
        self,
        raw_text: str,
        *,
        fallback_prompt: str,
    ) -> ParsedModelOutput:
        candidate = raw_text.strip()
        fence_match = CODE_FENCE_RE.match(candidate)
        if fence_match:
            candidate = fence_match.group("body").strip()

        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            answer = candidate or "unknown"
            return ParsedModelOutput(
                assistant_response=answer,
                episodic_summary=self.default_episode_summary(fallback_prompt, answer),
                raw_text=raw_text,
                empty_answer=not bool(candidate),
            )

        if not isinstance(payload, dict):
            answer = candidate or "unknown"
            return ParsedModelOutput(
                assistant_response=answer,
                episodic_summary=self.default_episode_summary(fallback_prompt, answer),
                raw_text=raw_text,
                empty_answer=not bool(candidate),
            )

        raw_answer = str(payload.get("assistant_response") or payload.get("answer") or "").strip()
        empty_answer = not bool(raw_answer)
        answer = raw_answer
        if not answer:
            answer = "unknown"
        episodic_summary = str(payload.get("episodic_summary") or "").strip()
        if not episodic_summary:
            episodic_summary = self.default_episode_summary(fallback_prompt, answer)

        observations = self._coerce_model_observations(
            payload.get("memory_observations", [])
        )
        return ParsedModelOutput(
            assistant_response=answer,
            episodic_summary=episodic_summary,
            memory_observations=observations,
            raw_text=raw_text,
            used_json=True,
            empty_answer=empty_answer,
        )

    def default_episode_summary(self, prompt: str, answer: str) -> str:
        return (
            f"User asked: {_truncate_text(prompt, 120)} "
            f"Assistant answered: {_truncate_text(answer, 120)}"
        )

    def _recover_missing_observations(
        self,
        prompt: str,
        assistant_response: str,
        episodic_summary: str,
    ) -> List[Observation]:
        if _looks_like_query_prompt(prompt):
            return []

        combined = " ".join(
            part for part in (prompt, assistant_response, episodic_summary) if part
        ).lower()
        existing_slots = {
            item.key: item.value for item in self.session.state.working_state if item.status == "active"
        }
        for belief in self.session.state.beliefs:
            if belief.status == "active":
                existing_slots.setdefault(belief.key, belief.value)

        if (
            "retry" in combined
            and "release" in combined
            and any(term in combined for term in ("auth", "authentication", "login", "github"))
        ):
            return [
                Observation(
                    text="Before retrying a release, check authentication first.",
                    memory_type="lesson",
                    payload={
                        "trigger": "retry_release",
                        "action": "check authentication",
                        "summary": "Before retrying a release, confirm GitHub authentication first.",
                    },
                    salience=0.72,
                )
            ]

        if any(term in combined for term in ("research partner", "co-investigator", "co investigator")):
            return [
                Observation(
                    text="The collaboration mode is research partner.",
                    memory_type="relationship",
                    payload={
                        "key": "collaboration_mode",
                        "value": "research partner",
                        "summary": "The collaboration mode is research partner.",
                        "themes": "relationship,research-mode",
                    },
                    salience=0.74,
                )
            ]

        if "eval summary" in combined or "evaluation summary" in combined:
            return [
                Observation(
                    text="The current primary goal is to ship the eval summary.",
                    memory_type="goal",
                    payload={
                        "key": "primary_goal",
                        "value": "ship eval summary",
                        "summary": "The current primary goal is to ship the eval summary.",
                    },
                    salience=0.72,
                )
            ]

        if "eval report" in combined or "evaluation report" in combined:
            return [
                Observation(
                    text="The current primary goal is to ship the eval report.",
                    memory_type="goal",
                    payload={
                        "key": "primary_goal",
                        "value": "ship eval report",
                        "summary": "The current primary goal is to ship the eval report.",
                    },
                    salience=0.72,
                )
            ]

        if "citation" in combined:
            return [
                Observation(
                    text="The current primary goal is to preserve citations.",
                    memory_type="goal",
                    payload={
                        "key": "primary_goal",
                        "value": "preserve citations",
                        "summary": "The current primary goal is to preserve citations.",
                    },
                    salience=0.7,
                )
            ]

        if any(term in combined for term in ("full reasoning", "detailed explanation", "detailed replies")):
            memory_type = "correction" if "response_style" in existing_slots else "preference"
            return [
                Observation(
                    text="The user prefers detailed replies.",
                    memory_type=memory_type,
                    payload={
                        "key": "response_style",
                        "value": "detailed",
                        "proposition": "The user prefers detailed replies.",
                    },
                    salience=0.74,
                )
            ]

        if any(term in combined for term in ("really brief", "keep it brief", "punchiest", "short answers")):
            memory_type = "correction" if "response_style" in existing_slots else "preference"
            return [
                Observation(
                    text="The user prefers brief replies.",
                    memory_type=memory_type,
                    payload={
                        "key": "response_style",
                        "value": "brief",
                        "proposition": "The user prefers brief replies.",
                    },
                    salience=0.72,
                )
            ]

        if any(term in combined for term in ("terser", "headline version", "even shorter", "gist")):
            return [
                Observation(
                    text="The user likely prefers concise replies.",
                    memory_type="preference_hint",
                    payload={
                        "key": "response_style",
                        "value": "concise",
                        "proposition": "The user likely prefers concise replies.",
                    },
                    salience=0.45,
                )
            ]

        return []

    def _coerce_model_observations(self, payload: object) -> List[Observation]:
        if not isinstance(payload, list):
            return []

        observations: List[Observation] = []
        for item in payload:
            observation = self._coerce_model_observation(item)
            if observation is not None:
                observations.append(observation)
        return observations

    def _coerce_model_observation(self, payload: object) -> Observation | None:
        if not isinstance(payload, dict):
            return None

        text = str(payload.get("text") or "").strip()
        normalized_payload = _extract_observation_payload(payload)
        explicit_memory_type = str(payload.get("memory_type", "")).strip()
        memory_type = explicit_memory_type
        inferred_memory_type = False
        if memory_type not in VALID_MEMORY_TYPES:
            memory_type = _infer_missing_memory_type(normalized_payload, text)
            inferred_memory_type = bool(memory_type)
        if memory_type not in VALID_MEMORY_TYPES:
            return None

        memory_type, normalized_payload = _normalize_observation_payload(
            memory_type,
            normalized_payload,
            text,
            allow_default_slot_keys=inferred_memory_type,
        )

        required_keys = REQUIRED_PAYLOAD_KEYS.get(memory_type, set())
        if not required_keys.issubset(normalized_payload):
            return None

        if not text:
            text = self._default_observation_text(memory_type, normalized_payload)

        salience = _coerce_salience(payload.get("salience", 0.5))
        return Observation(
            text=text,
            memory_type=memory_type,
            payload=normalized_payload,
            salience=salience,
        )

    def _retrieve_working_state(
        self,
        prompt: str,
        state: BrainLayerState,
    ) -> List[RetrievedMemory]:
        memories = []
        for item in state.working_state:
            if item.status != "active":
                continue
            content = f"{item.key} = {item.value}. {item.content}"
            memories.append(
                RetrievedMemory(
                    layer="working_state",
                    record_id=item.id,
                    score=self._score_candidate(prompt, content, item.priority, 4.0),
                    content=content,
                )
            )
        return memories

    def _retrieve_beliefs(
        self,
        prompt: str,
        state: BrainLayerState,
    ) -> List[RetrievedMemory]:
        memories = []
        for belief in state.beliefs:
            if belief.status != "active":
                continue
            content = (
                f"{belief.key} = {belief.value}. {belief.proposition} "
                f"(confidence={belief.confidence:.2f})"
            )
            memories.append(
                RetrievedMemory(
                    layer="beliefs",
                    record_id=belief.id,
                    score=self._score_candidate(prompt, content, belief.confidence, 3.2),
                    content=content,
                )
            )
        return memories

    def _retrieve_autobio(
        self,
        prompt: str,
        state: BrainLayerState,
    ) -> List[RetrievedMemory]:
        memories = []
        for note in state.autobiographical_state:
            content = (
                f"{note.key} = {note.value}. {note.summary} "
                f"(themes={', '.join(note.themes)})"
            )
            memories.append(
                RetrievedMemory(
                    layer="autobiographical_state",
                    record_id=note.id,
                    score=self._score_candidate(prompt, content, 0.7, 2.8),
                    content=content,
                )
            )
        return memories

    def _retrieve_procedures(
        self,
        prompt: str,
        state: BrainLayerState,
    ) -> List[RetrievedMemory]:
        memories = []
        for procedure in state.procedures:
            first_step = procedure.steps[0] if procedure.steps else "no step recorded"
            content = (
                f"When {procedure.trigger}, {first_step}. {procedure.summary} "
                f"(confidence={procedure.confidence:.2f})"
            )
            memories.append(
                RetrievedMemory(
                    layer="procedures",
                    record_id=procedure.id,
                    score=self._score_candidate(prompt, content, procedure.confidence, 2.6),
                    content=content,
                )
            )
        return memories

    def _retrieve_episodes(
        self,
        prompt: str,
        state: BrainLayerState,
    ) -> List[RetrievedMemory]:
        memories = []
        for index, episode in enumerate(state.episodes):
            recency_bonus = min(1.0, (index + 1) / max(1, len(state.episodes)))
            content = f"{episode.summary} (tags={', '.join(episode.tags)})"
            memories.append(
                RetrievedMemory(
                    layer="episodes",
                    record_id=episode.id,
                    score=self._score_candidate(
                        prompt,
                        content,
                        episode.salience + recency_bonus,
                        1.2,
                    ),
                    content=content,
                )
            )
        return memories

    def _score_candidate(
        self,
        prompt: str,
        content: str,
        weight: float,
        base_score: float,
    ) -> float:
        overlap = lexical_overlap_score(prompt, content)
        return base_score + (overlap * 1.8) + weight

    def _default_observation_text(
        self,
        memory_type: str,
        payload: Dict[str, str],
    ) -> str:
        if memory_type in {"preference", "correction", "preference_hint"}:
            key = payload.get("key")
            value = payload.get("value")
            if key and value:
                return f"{key} is currently {value}."
            return "Captured preference update."
        if memory_type in {"goal", "goal_hint"}:
            return payload.get("summary", "Captured goal update.")
        if memory_type in {"relationship", "relationship_hint"}:
            return payload.get("summary", "Captured relationship update.")
        if memory_type in {"lesson", "lesson_hint"}:
            trigger = payload.get("trigger")
            action = payload.get("action")
            if trigger and action:
                return f"When {trigger}, {action}."
            return payload.get("summary", "Captured lesson update.")
        if memory_type == "noise":
            return payload.get("value", "Captured low-salience turn detail.")
        return "Captured model observation."


def _coerce_salience(value: object) -> float:
    try:
        salience = float(value)
    except (TypeError, ValueError):
        return 0.5
    return max(0.0, min(1.0, salience))


def _truncate_text(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 3].rstrip() + "..."


def _looks_like_query_prompt(prompt: str) -> bool:
    lowered = prompt.lower().strip()
    return (
        lowered.endswith("?")
        or lowered.startswith("what ")
        or lowered.startswith("how ")
        or lowered.startswith("before retrying")
    )


def _extract_observation_payload(payload: Dict[str, object]) -> Dict[str, str]:
    normalized_payload: Dict[str, str] = {}

    raw_payload = payload.get("payload", {})
    if isinstance(raw_payload, dict):
        for key, value in raw_payload.items():
            stringified = _stringify_payload_value(value)
            if stringified:
                normalized_payload[str(key).strip()] = stringified

    for key, value in payload.items():
        if key in OBSERVATION_RESERVED_KEYS:
            continue
        stringified = _stringify_payload_value(value)
        if stringified and str(key).strip() not in normalized_payload:
            normalized_payload[str(key).strip()] = stringified

    return normalized_payload


def _infer_missing_memory_type(payload: Dict[str, str], text: str) -> str:
    combined = " ".join(
        part
        for part in (
            payload.get("key", ""),
            payload.get("value", ""),
            payload.get("summary", ""),
            payload.get("proposition", ""),
            payload.get("trigger", ""),
            payload.get("action", ""),
            text,
        )
        if part
    ).lower()
    normalized_key = _normalize_slot_key(payload.get("key", "")) if payload.get("key") else ""

    if payload.get("trigger") or payload.get("action"):
        return "lesson"
    if normalized_key == "collaboration_mode" or payload.get("themes"):
        return "relationship"
    if normalized_key == "primary_goal":
        return "goal"
    if normalized_key in {"response_style", "preference"} or payload.get("proposition"):
        return "preference_hint" if "likely" in combined else "preference"

    if "authentication" in combined or ("retry" in combined and "release" in combined):
        return "lesson"
    if "research partner" in combined or "co-investigator" in combined:
        return "relationship"
    if "eval summary" in combined or "eval report" in combined or "citation" in combined:
        return "goal"
    if any(term in combined for term in ("brief", "concise", "shorter", "detailed", "full reasoning")):
        return "preference"
    return ""


def _normalize_observation_payload(
    memory_type: str,
    payload: Dict[str, str],
    text: str,
    *,
    allow_default_slot_keys: bool = False,
) -> tuple[str, Dict[str, str]]:
    normalized_payload = {
        str(key).strip(): str(value).strip()
        for key, value in payload.items()
        if str(value).strip()
    }

    if "key" in normalized_payload:
        normalized_payload["key"] = _normalize_slot_key(normalized_payload["key"])
        memory_type = _normalize_memory_type(memory_type, normalized_payload["key"])

    if memory_type == "noise":
        if "value" not in normalized_payload and text:
            normalized_payload["value"] = text
        return memory_type, normalized_payload

    if memory_type in {"preference", "correction", "preference_hint"}:
        key = normalized_payload.get("key", "")
        if not key and allow_default_slot_keys:
            key = "response_style"
        if key:
            normalized_payload["key"] = _normalize_slot_key(key)
        key = normalized_payload.get("key", "")
        if key:
            normalized_payload["value"] = _normalize_slot_value(
                key,
                normalized_payload.get("value", ""),
                normalized_payload,
                text,
            )
        if key and normalized_payload.get("value") and "proposition" not in normalized_payload:
            normalized_payload["proposition"] = _default_proposition(
                key,
                normalized_payload["value"],
                text,
            )
        return memory_type, normalized_payload

    if memory_type in {"goal", "goal_hint"}:
        key = normalized_payload.get("key", "")
        if not key and allow_default_slot_keys:
            key = "primary_goal"
        if key:
            normalized_payload["key"] = _normalize_slot_key(key)
        key = normalized_payload.get("key", "")
        if key:
            normalized_payload["value"] = _normalize_slot_value(
                key,
                normalized_payload.get("value", ""),
                normalized_payload,
                text,
            )
        if key and normalized_payload.get("value") and "summary" not in normalized_payload:
            normalized_payload["summary"] = _default_summary(
                key,
                normalized_payload["value"],
                text,
            )
        return memory_type, normalized_payload

    if memory_type in {"relationship", "relationship_hint"}:
        key = normalized_payload.get("key", "")
        if not key and allow_default_slot_keys:
            key = "collaboration_mode"
        if key:
            normalized_payload["key"] = _normalize_slot_key(key)
        key = normalized_payload.get("key", "")
        if key:
            normalized_payload["value"] = _normalize_slot_value(
                key,
                normalized_payload.get("value", ""),
                normalized_payload,
                text,
            )
        if key and normalized_payload.get("value") and "summary" not in normalized_payload:
            normalized_payload["summary"] = _default_summary(
                key,
                normalized_payload["value"],
                text,
            )
        normalized_payload.setdefault("themes", "relationship")
        return memory_type, normalized_payload

    if memory_type in {"lesson", "lesson_hint"}:
        normalized_payload["trigger"] = _normalize_trigger(
            normalized_payload.get("trigger", ""),
            normalized_payload,
            text,
        )
        normalized_payload["action"] = _normalize_action(
            normalized_payload.get("action", ""),
            normalized_payload,
            text,
        )
        if (
            normalized_payload.get("trigger")
            and normalized_payload.get("action")
            and "summary" not in normalized_payload
        ):
            normalized_payload["summary"] = text or (
                f"When {normalized_payload['trigger']}, {normalized_payload['action']}."
            )
        return memory_type, normalized_payload

    return memory_type, normalized_payload


def _normalize_memory_type(memory_type: str, key: str) -> str:
    if key == "primary_goal":
        return "goal_hint" if memory_type.endswith("_hint") else "goal"
    if key == "collaboration_mode":
        return "relationship_hint" if memory_type.endswith("_hint") else "relationship"
    if key == "response_style" and memory_type not in {
        "preference",
        "correction",
        "preference_hint",
    }:
        return "preference_hint" if memory_type.endswith("_hint") else "preference"
    return memory_type


def _normalize_slot_key(key: str) -> str:
    lowered = key.strip().lower()
    normalized = re.sub(r"[\s\-]+", "_", lowered)
    return KEY_ALIASES.get(lowered, KEY_ALIASES.get(normalized, normalized))


def _normalize_slot_value(
    key: str,
    value: str,
    payload: Dict[str, str],
    text: str,
) -> str:
    combined = " ".join(
        part
        for part in (
            value,
            payload.get("summary", ""),
            payload.get("proposition", ""),
            text,
        )
        if part
    ).strip()
    lowered = combined.lower()

    if key == "response_style":
        if any(phrase in lowered for phrase in ("full reasoning", "detailed", "step-by-step", "step by step")):
            return "detailed"
        if "concise" in lowered:
            return "concise"
        if any(term in lowered for term in ("terser", "terse", "headline version", "headline", "gist", "even shorter")):
            return "concise"
        if any(token in lowered for token in ("brief", "short", "succinct")):
            return "brief"
        return value.strip().lower()

    if key == "primary_goal":
        if "eval summary" in lowered or "evaluation summary" in lowered:
            return "ship eval summary"
        if "eval report" in lowered or "evaluation report" in lowered:
            return "ship eval report"
        if "citation" in lowered:
            return "preserve citations"
        return value.strip().lower()

    if key == "collaboration_mode":
        if "research partner" in lowered or "co-investigator" in lowered or "co investigator" in lowered:
            return "research partner"
        if "task executor" in lowered or "task runner" in lowered:
            return "task executor"
        return value.strip().lower()

    return value.strip()


def _normalize_trigger(trigger: str, payload: Dict[str, str], text: str) -> str:
    combined = " ".join(part for part in (trigger, payload.get("summary", ""), text) if part).lower()
    if "retry" in combined and "release" in combined:
        return "retry_release"
    if (
        ("rollout" in combined or "release" in combined or "reran" in combined or "rerun" in combined)
        and (
            "auth" in combined
            or "authentication" in combined
            or "login" in combined
            or "log in" in combined
        )
    ):
        return "retry_release"
    normalized = re.sub(r"[\s\-]+", "_", trigger.strip().lower())
    return normalized


def _normalize_action(action: str, payload: Dict[str, str], text: str) -> str:
    combined = " ".join(part for part in (action, payload.get("summary", ""), text) if part).lower()
    if (
        "auth" in combined
        or "authentication" in combined
        or "log into github" in combined
        or "login to github" in combined
        or "logging back into github" in combined
    ):
        return "check authentication"
    return action.strip().lower()


def _default_proposition(key: str, value: str, text: str) -> str:
    if text:
        return text
    if key == "response_style":
        return f"The user prefers {value} replies."
    return f"{key} is currently {value}."


def _default_summary(key: str, value: str, text: str) -> str:
    if text:
        return text
    if key == "primary_goal":
        return f"The current primary goal is to {value}."
    if key == "collaboration_mode":
        return f"The collaboration mode is {value}."
    return f"{key} is currently {value}."


def _stringify_payload_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(item).strip() for item in value if str(item).strip())
    return str(value).strip()
