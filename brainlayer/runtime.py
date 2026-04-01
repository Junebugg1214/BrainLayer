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
    "lesson": {"trigger", "action", "summary"},
    "lesson_hint": {"trigger", "action", "summary"},
    "goal": {"key", "value", "summary"},
    "goal_hint": {"key", "value", "summary"},
    "relationship": {"key", "value", "summary", "themes"},
    "relationship_hint": {"key", "value", "summary", "themes"},
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

        applied_observations: List[Observation] = []
        for observation in parsed_output.memory_observations:
            self.session.observe(
                text=observation.text,
                memory_type=observation.memory_type,
                payload=observation.payload,
                salience=observation.salience,
                scenario_slug=active_scenario_slug,
            )
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

        memory_type = str(payload.get("memory_type", "")).strip()
        if memory_type not in VALID_MEMORY_TYPES:
            return None

        raw_payload = payload.get("payload", {})
        if not isinstance(raw_payload, dict):
            return None
        normalized_payload = {
            str(key): str(value)
            for key, value in raw_payload.items()
            if value is not None
        }

        required_keys = REQUIRED_PAYLOAD_KEYS.get(memory_type, set())
        if not required_keys.issubset(normalized_payload):
            return None

        text = str(payload.get("text") or "").strip()
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
            return f"{payload['key']} is currently {payload['value']}."
        if memory_type in {"goal", "goal_hint"}:
            return payload["summary"]
        if memory_type in {"relationship", "relationship_hint"}:
            return payload["summary"]
        if memory_type in {"lesson", "lesson_hint"}:
            return f"When {payload['trigger']}, {payload['action']}."
        if memory_type == "noise":
            return "Captured low-salience turn detail."
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
