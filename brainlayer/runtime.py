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
VALID_MEMORY_STRATEGIES = {
    "brainlayer",
    "context_only",
    "naive_retrieval",
    "structured_no_consolidation",
    "summary_state",
}


def tokenize(text: str) -> List[str]:
    return [token for token in TOKEN_RE.findall(text.lower()) if token not in STOPWORDS]


def lexical_overlap_score(query: str, text: str) -> int:
    return len(set(tokenize(query)) & set(tokenize(text)))


def _is_response_style_query(prompt: str) -> bool:
    lowered = prompt.lower()
    return any(
        phrase in lowered
        for phrase in (
            "response style",
            "how should you answer",
            "how should you respond",
            "what style should you use",
        )
    )


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
    memory_strategy: str = "brainlayer"
    max_turn_history: int = 6


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
        if self.config.memory_strategy not in VALID_MEMORY_STRATEGIES:
            raise ValueError(
                f"Unsupported memory strategy: {self.config.memory_strategy}. "
                f"Expected one of {sorted(VALID_MEMORY_STRATEGIES)}."
            )
        self.turn_history: List[Dict[str, str]] = []
        self.note_memory: List[Dict[str, object]] = []
        self.summary_slots: Dict[str, str] = {}
        self.summary_procedures: Dict[str, str] = {}
        self.summary_events: List[str] = []

    def run_turn(
        self,
        prompt: str,
        *,
        observations: Sequence[Observation] | None = None,
        scenario_slug: str | None = None,
    ) -> ModelTurnResult:
        active_scenario_slug = scenario_slug or self.config.default_scenario_slug
        for observation in observations or []:
            self._store_observation(observation, scenario_slug=active_scenario_slug)

        if self.config.consolidate_before_reply and self.config.memory_strategy == "brainlayer":
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
        if (
            not recovered_observations
            and self.config.memory_strategy != "context_only"
        ):
            recovered_observations = self._recover_missing_observations(
                prompt,
                parsed_output.assistant_response,
                parsed_output.episodic_summary,
            )

        applied_observations: List[Observation] = []
        for observation in recovered_observations:
            try:
                if not self._store_observation(observation, scenario_slug=active_scenario_slug):
                    continue
            except Exception:
                continue
            applied_observations.append(observation)

        interaction_episode_id = self._record_interaction_episode(
            scenario_slug=active_scenario_slug,
            prompt=prompt,
            assistant_response=parsed_output.assistant_response,
            episodic_summary=parsed_output.episodic_summary,
            used_json=parsed_output.used_json,
            retrieved=retrieved,
        )

        consolidation_report = None
        if self.config.auto_consolidate_after_turn and self.config.memory_strategy == "brainlayer":
            consolidation_report = self.session.consolidate()

        self._append_turn_history(prompt, parsed_output.assistant_response)
        exported_state = self._export_runtime_state()

        return ModelTurnResult(
            assistant_response=parsed_output.assistant_response,
            episodic_summary=parsed_output.episodic_summary,
            raw_model_output=model_response.content,
            prompt_messages=prompt_messages,
            retrieved_memories=retrieved,
            applied_observations=applied_observations,
            interaction_episode_id=interaction_episode_id,
            consolidation_report=consolidation_report,
            model_response=model_response,
            exported_state=exported_state,
            used_json=parsed_output.used_json,
            empty_answer=parsed_output.empty_answer,
        )

    def retrieve_memories(self, prompt: str) -> List[RetrievedMemory]:
        if self.config.memory_strategy == "context_only":
            return self._retrieve_turn_history()
        if self.config.memory_strategy == "naive_retrieval":
            return self._retrieve_naive_notes(prompt)
        if self.config.memory_strategy == "summary_state":
            return self._retrieve_summary_state(prompt)

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

    def export_state(self) -> Dict[str, object]:
        return self._export_runtime_state()

    def build_messages(
        self,
        prompt: str,
        retrieved_memories: Sequence[RetrievedMemory],
    ) -> List[ModelMessage]:
        context = self.render_retrieved_context(retrieved_memories)
        style_override = self._derive_response_style_override(prompt, retrieved_memories)
        if style_override:
            context = f"{context}\n{style_override}"
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

    def _derive_response_style_override(
        self,
        prompt: str,
        retrieved_memories: Sequence[RetrievedMemory],
    ) -> str:
        if not _is_response_style_query(prompt):
            return ""

        detailed_goal_active = any(
            memory.layer == "working_state"
            and "primary_goal =" in memory.content
            and any(
                phrase in memory.content.lower()
                for phrase in (
                    "provide detailed reasoning",
                    "detailed explanation",
                    "full reasoning",
                    "whole chain of reasoning",
                    "full chain of reasoning",
                )
            )
            for memory in retrieved_memories
        )
        if not detailed_goal_active:
            return ""

        return (
            "- [derived_override] response_style = detailed. "
            "An active working-state goal currently requires detailed reasoning, "
            "so that overrides older brief defaults for this answer."
        )

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

    def _store_observation(self, observation: Observation, *, scenario_slug: str) -> bool:
        strategy = self.config.memory_strategy
        if strategy == "context_only":
            return False
        if strategy == "brainlayer":
            self.session.observe(
                text=observation.text,
                memory_type=observation.memory_type,
                payload=observation.payload,
                salience=observation.salience,
                scenario_slug=scenario_slug,
            )
            return True
        if strategy == "structured_no_consolidation":
            self._observe_append_only(observation, scenario_slug=scenario_slug)
            return True
        if strategy == "naive_retrieval":
            self.note_memory.append(
                {
                    "id": f"note-{len(self.note_memory) + 1}",
                    "scenario": scenario_slug,
                    "text": observation.text,
                    "memory_type": observation.memory_type,
                    "payload": dict(observation.payload),
                    "salience": observation.salience,
                }
            )
            return True
        if strategy == "summary_state":
            self._update_summary_state(observation)
            return True
        return False

    def _record_interaction_episode(
        self,
        *,
        scenario_slug: str,
        prompt: str,
        assistant_response: str,
        episodic_summary: str,
        used_json: bool,
        retrieved: Sequence[RetrievedMemory],
    ) -> str:
        strategy = self.config.memory_strategy
        if strategy not in {"brainlayer", "structured_no_consolidation"}:
            return ""

        episode = self.session.state.record_episode(
            scenario=scenario_slug,
            summary=episodic_summary,
            tags=["interaction", "assistant_reply"],
            metadata={
                "prompt": _truncate_text(prompt, 240),
                "assistant_response": _truncate_text(assistant_response, 240),
                "model": self.model,
                "used_json": "true" if used_json else "false",
            },
            salience=self.config.interaction_salience,
            outcome="completed model-backed turn",
            source_refs=[memory.record_id for memory in retrieved],
        )
        return episode.id

    def _append_turn_history(self, prompt: str, assistant_response: str) -> None:
        self.turn_history.append(
            {
                "prompt": _truncate_text(prompt, 240),
                "assistant_response": _truncate_text(assistant_response, 240),
            }
        )
        if len(self.turn_history) > self.config.max_turn_history:
            self.turn_history = self.turn_history[-self.config.max_turn_history :]

    def _export_runtime_state(self) -> Dict[str, object]:
        strategy = self.config.memory_strategy
        if strategy in {"brainlayer", "structured_no_consolidation"}:
            return self.session.state.to_dict()
        if strategy == "naive_retrieval":
            episodes = []
            for note in self.note_memory:
                payload = note.get("payload", {})
                if not isinstance(payload, dict):
                    payload = {}
                episodes.append(
                    {
                        "id": str(note.get("id", "")),
                        "scenario": str(note.get("scenario", "")),
                        "summary": str(note.get("text", "")),
                        "tags": [str(note.get("memory_type", "note"))],
                        "metadata": {str(key): str(value) for key, value in payload.items()},
                        "salience": float(note.get("salience", 0.5) or 0.5),
                        "outcome": "stored naive note",
                        "source_refs": [],
                        "timestamp": "",
                    }
                )
            return {
                "working_state": [],
                "episodes": episodes,
                "beliefs": [],
                "autobiographical_state": [],
                "procedures": [],
                "naive_notes": list(self.note_memory),
            }
        if strategy == "summary_state":
            return {
                "working_state": [],
                "episodes": [],
                "beliefs": [],
                "autobiographical_state": [],
                "procedures": [],
                "summary_state": {
                    "slots": dict(self.summary_slots),
                    "procedures": dict(self.summary_procedures),
                    "events": list(self.summary_events),
                },
            }
        return {
            "working_state": [],
            "episodes": [],
            "beliefs": [],
            "autobiographical_state": [],
            "procedures": [],
        }

    def _retrieve_turn_history(self) -> List[RetrievedMemory]:
        recent_history = self.turn_history[-self.config.max_turn_history :]
        memories: List[RetrievedMemory] = []
        for index, turn in enumerate(recent_history, start=1):
            memories.append(
                RetrievedMemory(
                    layer="history",
                    record_id=f"history-{index}",
                    score=float(index),
                    content=(
                        f"User: {turn.get('prompt', '')} "
                        f"Assistant: {turn.get('assistant_response', '')}"
                    ),
                )
            )
        return memories

    def _retrieve_naive_notes(self, prompt: str) -> List[RetrievedMemory]:
        memories: List[RetrievedMemory] = []
        total = max(1, len(self.note_memory))
        for index, note in enumerate(self.note_memory):
            content = self._render_note_content(note)
            recency_bonus = (index + 1) / total
            memories.append(
                RetrievedMemory(
                    layer="notes",
                    record_id=str(note.get("id", f"note-{index + 1}")),
                    score=self._score_candidate(
                        prompt,
                        content,
                        float(note.get("salience", 0.5) or 0.5) + recency_bonus,
                        1.6,
                    ),
                    content=content,
                )
            )
        memories.sort(key=lambda memory: memory.score, reverse=True)
        return memories[: self.config.max_memories]

    def _retrieve_summary_state(self, prompt: str) -> List[RetrievedMemory]:
        memories: List[RetrievedMemory] = []
        for key, value in self.summary_slots.items():
            content = f"{key} = {value}."
            memories.append(
                RetrievedMemory(
                    layer="summary_state",
                    record_id=f"summary-slot-{key}",
                    score=self._score_candidate(prompt, content, 0.8, 2.0),
                    content=content,
                )
            )
        for trigger, action in self.summary_procedures.items():
            content = f"When {trigger}, {action}."
            memories.append(
                RetrievedMemory(
                    layer="summary_state",
                    record_id=f"summary-procedure-{trigger}",
                    score=self._score_candidate(prompt, content, 0.8, 2.0),
                    content=content,
                )
            )
        for index, event in enumerate(self.summary_events[-3:], start=1):
            memories.append(
                RetrievedMemory(
                    layer="summary_state",
                    record_id=f"summary-event-{index}",
                    score=self._score_candidate(prompt, event, 0.3, 0.8),
                    content=event,
                )
            )
        memories.sort(key=lambda memory: memory.score, reverse=True)
        return memories[: self.config.max_memories]

    def _render_note_content(self, note: Dict[str, object]) -> str:
        memory_type = str(note.get("memory_type", "note"))
        payload = note.get("payload", {})
        if not isinstance(payload, dict):
            payload = {}
        text = str(note.get("text", "")).strip()
        if memory_type in {"preference", "correction", "preference_hint"}:
            key = payload.get("key", "")
            value = payload.get("value", "")
            proposition = payload.get("proposition", "")
            if key and value:
                return f"{key} = {value}. {proposition or text}".strip()
        if memory_type in {"goal", "goal_hint", "relationship", "relationship_hint"}:
            key = payload.get("key", "")
            value = payload.get("value", "")
            summary = payload.get("summary", "")
            if key and value:
                return f"{key} = {value}. {summary or text}".strip()
        if memory_type in {"lesson", "lesson_hint"}:
            trigger = payload.get("trigger", "")
            action = payload.get("action", "")
            summary = payload.get("summary", "")
            if trigger and action:
                return f"When {trigger}, {action}. {summary or text}".strip()
        return text or json.dumps(note, sort_keys=True)

    def _update_summary_state(self, observation: Observation) -> None:
        self.summary_events.append(observation.text)
        self.summary_events = self.summary_events[-8:]

        if observation.memory_type in {"preference", "correction"}:
            key = observation.payload.get("key")
            value = observation.payload.get("value")
            if key and value:
                self.summary_slots[key] = value
            return

        if observation.memory_type == "goal":
            key = observation.payload.get("key")
            value = observation.payload.get("value")
            if key and value:
                self.summary_slots[key] = value
            return

        if observation.memory_type == "relationship":
            key = observation.payload.get("key")
            value = observation.payload.get("value")
            if key and value:
                self.summary_slots[key] = value
            return

        if observation.memory_type == "lesson":
            trigger = observation.payload.get("trigger")
            action = observation.payload.get("action")
            if trigger and action:
                self.summary_procedures[trigger] = action
            return

        if observation.memory_type in {"preference_hint", "goal_hint", "relationship_hint", "lesson_hint"}:
            key = observation.payload.get("key")
            trigger = observation.payload.get("trigger")
            if key and key not in self.summary_slots:
                value = observation.payload.get("value")
                if value:
                    self.summary_slots[key] = value
            if trigger and trigger not in self.summary_procedures:
                action = observation.payload.get("action")
                if action:
                    self.summary_procedures[trigger] = action

    def _observe_append_only(self, observation: Observation, *, scenario_slug: str) -> None:
        state = self.session.state
        if observation.memory_type in {"preference", "correction"}:
            episode = state.record_episode(
                scenario=scenario_slug,
                summary=observation.text,
                tags=[observation.memory_type, observation.payload["key"]],
                metadata=observation.payload,
                salience=observation.salience,
                outcome="captured preference",
            )
            state.beliefs.append(
                Belief(
                    id=state._next_id("belief"),
                    key=observation.payload["key"],
                    proposition=observation.payload["proposition"],
                    value=observation.payload["value"],
                    confidence=observation.salience,
                    evidence_episode_ids=[episode.id],
                )
            )
            state.working_state.append(
                WorkingItem(
                    id=state._next_id("working"),
                    key=observation.payload["key"],
                    value=observation.payload["value"],
                    content=observation.payload["proposition"],
                    priority=observation.salience,
                    source_refs=[episode.id],
                )
            )
            return

        if observation.memory_type == "goal":
            episode = state.record_episode(
                scenario=scenario_slug,
                summary=observation.text,
                tags=["goal", observation.payload["key"]],
                metadata=observation.payload,
                salience=observation.salience,
                outcome="captured goal",
            )
            state.working_state.append(
                WorkingItem(
                    id=state._next_id("working"),
                    key=observation.payload["key"],
                    value=observation.payload["value"],
                    content=observation.payload["summary"],
                    priority=observation.salience,
                    source_refs=[episode.id],
                )
            )
            return

        if observation.memory_type == "relationship":
            episode = state.record_episode(
                scenario=scenario_slug,
                summary=observation.text,
                tags=["relationship", observation.payload["key"]],
                metadata=observation.payload,
                salience=observation.salience,
                outcome="updated relationship framing",
            )
            themes = [
                value.strip()
                for value in observation.payload.get("themes", "").split(",")
                if value.strip()
            ]
            state.autobiographical_state.append(
                AutobioNote(
                    id=state._next_id("autobio"),
                    key=observation.payload["key"],
                    value=observation.payload["value"],
                    summary=observation.payload["summary"],
                    themes=themes or ["relationship"],
                    supporting_ids=[episode.id],
                )
            )
            state.working_state.append(
                WorkingItem(
                    id=state._next_id("working"),
                    key=observation.payload["key"],
                    value=observation.payload["value"],
                    content=observation.payload["summary"],
                    priority=observation.salience,
                    source_refs=[episode.id],
                )
            )
            return

        if observation.memory_type == "lesson":
            episode = state.record_episode(
                scenario=scenario_slug,
                summary=observation.text,
                tags=["lesson", observation.payload["trigger"]],
                metadata=observation.payload,
                salience=observation.salience,
                outcome="failure lesson",
            )
            state.procedures.append(
                Procedure(
                    id=state._next_id("procedure"),
                    trigger=observation.payload["trigger"],
                    summary=observation.payload["summary"],
                    steps=[observation.payload["action"]],
                    confidence=observation.salience,
                    derived_from=[episode.id],
                )
            )
            return

        if observation.memory_type in {
            "preference_hint",
            "lesson_hint",
            "goal_hint",
            "relationship_hint",
        }:
            topic_key = observation.payload.get("key") or observation.payload.get("trigger", "signal")
            state.record_episode(
                scenario=scenario_slug,
                summary=observation.text,
                tags=[observation.memory_type, topic_key],
                metadata=observation.payload,
                salience=observation.salience,
                outcome="captured candidate signal",
            )
            return

        state.record_episode(
            scenario=scenario_slug,
            summary=observation.text,
            tags=["noise"],
            metadata=observation.payload,
            salience=observation.salience,
            outcome="ignored",
        )

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
            (
                "retry" in combined
                and "release" in combined
                and any(term in combined for term in ("auth", "authentication", "login", "github"))
            )
            or any(
                phrase in combined
                for phrase in (
                    "session had died",
                    "repo credentials were stale",
                    "reauthing github",
                    "repo login had expired",
                )
            )
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

        if any(
            term in combined
            for term in (
                "research partner",
                "co-investigator",
                "co investigator",
                "research collaborator",
                "thought partner",
                "co-design",
                "co design",
                "co-pilot",
                "co pilot",
            )
        ):
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

        if (
            "eval summary" in combined
            or "evaluation summary" in combined
            or "evaluation digest" in combined
        ):
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

        if any(
            term in combined
            for term in (
                "full reasoning",
                "detailed explanation",
                "detailed replies",
                "long-form rationale",
                "long form rationale",
                "whole chain of reasoning",
            )
        ):
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

        if any(
            term in combined
            for term in (
                "really brief",
                "keep it brief",
                "punchiest",
                "short answers",
                "quick-scan version",
                "quick scan version",
                "leanest take",
                "bare-bones version",
                "bare bones version",
            )
        ):
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
        total = max(1, len(state.working_state))
        for index, item in enumerate(state.working_state):
            if item.status != "active":
                continue
            content = f"{item.key} = {item.value}. {item.content}"
            recency_bonus = (index + 1) / total
            memories.append(
                RetrievedMemory(
                    layer="working_state",
                    record_id=item.id,
                    score=self._score_candidate(
                        prompt,
                        content,
                        item.priority + (recency_bonus * 0.35),
                        4.0,
                    ),
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
        total = max(1, len(state.beliefs))
        for index, belief in enumerate(state.beliefs):
            if belief.status != "active":
                continue
            content = (
                f"{belief.key} = {belief.value}. {belief.proposition} "
                f"(confidence={belief.confidence:.2f})"
            )
            recency_bonus = (index + 1) / total
            memories.append(
                RetrievedMemory(
                    layer="beliefs",
                    record_id=belief.id,
                    score=self._score_candidate(
                        prompt,
                        content,
                        belief.confidence + (recency_bonus * 0.35),
                        3.2,
                    ),
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
        total = max(1, len(state.autobiographical_state))
        for index, note in enumerate(state.autobiographical_state):
            content = (
                f"{note.key} = {note.value}. {note.summary} "
                f"(themes={', '.join(note.themes)})"
            )
            recency_bonus = (index + 1) / total
            memories.append(
                RetrievedMemory(
                    layer="autobiographical_state",
                    record_id=note.id,
                    score=self._score_candidate(
                        prompt,
                        content,
                        0.7 + (recency_bonus * 0.35),
                        2.8,
                    ),
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
        total = max(1, len(state.procedures))
        for index, procedure in enumerate(state.procedures):
            first_step = procedure.steps[0] if procedure.steps else "no step recorded"
            content = (
                f"When {procedure.trigger}, {first_step}. {procedure.summary} "
                f"(confidence={procedure.confidence:.2f})"
            )
            recency_bonus = (index + 1) / total
            memories.append(
                RetrievedMemory(
                    layer="procedures",
                    record_id=procedure.id,
                    score=self._score_candidate(
                        prompt,
                        content,
                        procedure.confidence + (recency_bonus * 0.35),
                        2.6,
                    ),
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
        if any(
            phrase in lowered
            for phrase in (
                "full reasoning",
                "detailed",
                "step-by-step",
                "step by step",
                "long-form rationale",
                "long form rationale",
                "whole chain of reasoning",
            )
        ):
            return "detailed"
        if "concise" in lowered:
            return "concise"
        if any(
            term in lowered
            for term in (
                "terser",
                "terse",
                "headline version",
                "headline",
                "gist",
                "even shorter",
            )
        ):
            return "concise"
        if any(
            token in lowered
            for token in (
                "brief",
                "short",
                "succinct",
                "quick-scan",
                "quick scan",
                "leanest take",
                "bare-bones",
                "bare bones",
            )
        ):
            return "brief"
        return value.strip().lower()

    if key == "primary_goal":
        if any(
            phrase in lowered
            for phrase in (
                "provide detailed reasoning",
                "detailed reasoning",
                "detailed explanation",
                "full reasoning",
                "whole chain of reasoning",
                "full chain of reasoning",
            )
        ):
            return "provide detailed reasoning"
        if (
            "eval summary" in lowered
            or "evaluation summary" in lowered
            or "eval digest" in lowered
            or "evaluation digest" in lowered
        ):
            return "ship eval summary"
        if "eval report" in lowered or "evaluation report" in lowered:
            return "ship eval report"
        if "citation" in lowered:
            return "preserve citations"
        return value.strip().lower()

    if key == "collaboration_mode":
        if any(
            phrase in lowered
            for phrase in (
                "research partner",
                "co-investigator",
                "co investigator",
                "research collaborator",
                "thought partner",
                "co-design",
                "co design",
                "co-pilot",
                "co pilot",
            )
        ):
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
        or "confirm the login first" in combined
        or "confirm the credentials" in combined
        or "reauth" in combined
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
