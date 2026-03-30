from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from .consolidation import ConsolidationConfig, ConsolidationEngine, ConsolidationReport
from .models import BrainLayerState
from .scenarios import Observation, Query


TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "as",
    "be",
    "before",
    "do",
    "for",
    "is",
    "of",
    "should",
    "the",
    "to",
    "use",
    "what",
    "you",
}


def tokenize(text: str) -> List[str]:
    return [token for token in TOKEN_RE.findall(text.lower()) if token not in STOPWORDS]


def lexical_overlap_score(query: str, note: str) -> int:
    return len(set(tokenize(query)) & set(tokenize(note)))


@dataclass
class AnswerRecord:
    answer: str
    evidence: str


class BaseAgent:
    name = "base"

    def reset(self) -> None:
        raise NotImplementedError

    def observe(self, scenario_slug: str, observation: Observation) -> None:
        raise NotImplementedError

    def answer(self, query: Query) -> AnswerRecord:
        raise NotImplementedError

    def export_state(self) -> Dict[str, object]:
        return {}


class ContextOnlyAgent(BaseAgent):
    name = "context_only"

    def reset(self) -> None:
        self.current_observation: Optional[Observation] = None

    def observe(self, scenario_slug: str, observation: Observation) -> None:
        self.current_observation = observation

    def answer(self, query: Query) -> AnswerRecord:
        if self.current_observation and lexical_overlap_score(
            query.prompt, self.current_observation.text
        ):
            value = self.current_observation.payload.get(query.answer_key, "unknown")
            return AnswerRecord(answer=value, evidence=self.current_observation.text)
        return AnswerRecord(answer="unknown", evidence="No persistent state.")

    def export_state(self) -> Dict[str, object]:
        if not self.current_observation:
            return {"current_observation": None}
        return {
            "current_observation": {
                "text": self.current_observation.text,
                "memory_type": self.current_observation.memory_type,
            }
        }


class NaiveMemoryAgent(BaseAgent):
    name = "naive_memory"

    def reset(self) -> None:
        self.notes: List[Observation] = []

    def observe(self, scenario_slug: str, observation: Observation) -> None:
        self.notes.append(observation)

    def answer(self, query: Query) -> AnswerRecord:
        if not self.notes:
            return AnswerRecord(answer="unknown", evidence="No notes stored.")

        best_note = max(
            self.notes,
            key=lambda note: lexical_overlap_score(query.prompt, note.text),
        )
        return AnswerRecord(
            answer=best_note.payload.get(query.answer_key, "unknown"),
            evidence=best_note.text,
        )

    def export_state(self) -> Dict[str, object]:
        return {
            "notes": [
                {
                    "text": note.text,
                    "memory_type": note.memory_type,
                    "payload": note.payload,
                }
                for note in self.notes
            ]
        }


class BrainLayerAgent(BaseAgent):
    name = "brainlayer"

    def __init__(
        self,
        state: BrainLayerState | None = None,
        *,
        auto_consolidate: bool = True,
        consolidation_config: ConsolidationConfig | None = None,
    ) -> None:
        self.state = state or BrainLayerState()
        self.auto_consolidate = auto_consolidate
        self.consolidation_engine = ConsolidationEngine(consolidation_config)

    def reset(self) -> None:
        self.state = BrainLayerState()

    def observe(self, scenario_slug: str, observation: Observation) -> None:
        if observation.memory_type in {"preference", "correction"}:
            episode = self.state.record_episode(
                scenario=scenario_slug,
                summary=observation.text,
                tags=[observation.memory_type, observation.payload["key"]],
                metadata=observation.payload,
                salience=observation.salience,
                outcome="captured preference",
            )
            belief = self.state.upsert_belief(
                key=observation.payload["key"],
                proposition=observation.payload["proposition"],
                value=observation.payload["value"],
                confidence=observation.salience,
                evidence_episode_ids=[episode.id],
            )
            self.state.upsert_working_item(
                key=belief.key,
                value=belief.value,
                content=f"Current {belief.key}: {belief.value}",
                priority=observation.salience,
                source_refs=[episode.id, belief.id],
            )
            if belief.key == "response_style":
                self.state.upsert_autobio_note(
                    key="collaboration_tone",
                    value=belief.value,
                    summary=(
                        "Collaboration is strongest when replies stay aligned with the "
                        f"user's preferred response style: {belief.value}."
                    ),
                    themes=["communication-style", "user-model"],
                    supporting_ids=[episode.id, belief.id],
                )
            return

        if observation.memory_type == "lesson":
            episode = self.state.record_episode(
                scenario=scenario_slug,
                summary=observation.text,
                tags=["lesson", observation.payload["trigger"]],
                metadata=observation.payload,
                salience=observation.salience,
                outcome="failure lesson",
            )
            procedure = self.state.learn_procedure(
                trigger=observation.payload["trigger"],
                summary=observation.payload["summary"],
                steps=[observation.payload["action"]],
                confidence=observation.salience,
                derived_from=[episode.id],
            )
            self.state.upsert_working_item(
                key=f"procedure:{procedure.trigger}",
                value=procedure.steps[0],
                content=f"When {procedure.trigger}, {procedure.steps[0]}",
                priority=observation.salience,
                source_refs=[episode.id, procedure.id],
            )
            return

        if observation.memory_type == "goal":
            episode = self.state.record_episode(
                scenario=scenario_slug,
                summary=observation.text,
                tags=["goal", observation.payload["key"]],
                metadata=observation.payload,
                salience=observation.salience,
                outcome="captured goal",
            )
            self.state.upsert_working_item(
                key=observation.payload["key"],
                value=observation.payload["value"],
                content=observation.payload["summary"],
                priority=observation.salience,
                source_refs=[episode.id],
            )
            return

        if observation.memory_type == "relationship":
            episode = self.state.record_episode(
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
            autobio = self.state.upsert_autobio_note(
                key=observation.payload["key"],
                value=observation.payload["value"],
                summary=observation.payload["summary"],
                themes=themes or ["relationship"],
                supporting_ids=[episode.id],
            )
            self.state.upsert_working_item(
                key=autobio.key,
                value=autobio.value,
                content=autobio.summary,
                priority=observation.salience,
                source_refs=[episode.id, autobio.id],
            )
            return

        if observation.memory_type in {
            "preference_hint",
            "lesson_hint",
            "goal_hint",
            "relationship_hint",
        }:
            topic_key = observation.payload.get("key") or observation.payload.get("trigger", "signal")
            self.state.record_episode(
                scenario=scenario_slug,
                summary=observation.text,
                tags=[observation.memory_type, topic_key],
                metadata=observation.payload,
                salience=observation.salience,
                outcome="captured candidate signal",
            )
            return

        self.state.record_episode(
            scenario=scenario_slug,
            summary=observation.text,
            tags=["noise"],
            metadata=observation.payload,
            salience=observation.salience,
            outcome="ignored",
        )

    def consolidate(self) -> ConsolidationReport:
        return self.consolidation_engine.run(self.state)

    def answer(self, query: Query) -> AnswerRecord:
        if self.auto_consolidate:
            self.consolidate()

        if query.query_type == "belief_lookup":
            for belief in reversed(self.state.beliefs):
                if belief.key == query.lookup_key and belief.status == "active":
                    return AnswerRecord(
                        answer=belief.value,
                        evidence=f"{belief.key} belief from {belief.evidence_episode_ids}",
                    )
            return AnswerRecord(answer="unknown", evidence="No active belief found.")

        if query.query_type == "procedure_lookup":
            for procedure in reversed(self.state.procedures):
                if procedure.trigger == query.procedure_trigger and procedure.steps:
                    return AnswerRecord(
                        answer=procedure.steps[0],
                        evidence=f"Procedure {procedure.id} from {procedure.derived_from}",
                    )
            return AnswerRecord(answer="unknown", evidence="No matching procedure found.")

        if query.query_type == "working_lookup":
            for item in reversed(self.state.working_state):
                if item.key == query.lookup_key and item.status == "active":
                    return AnswerRecord(
                        answer=item.value,
                        evidence=f"working item {item.id} from {item.source_refs}",
                    )
            return AnswerRecord(answer="unknown", evidence="No active working item found.")

        if query.query_type == "autobio_lookup":
            for note in reversed(self.state.autobiographical_state):
                if note.key == query.lookup_key:
                    return AnswerRecord(
                        answer=note.value,
                        evidence=f"autobio note {note.id} from {note.supporting_ids}",
                    )
            return AnswerRecord(answer="unknown", evidence="No autobiographical note found.")

        return AnswerRecord(answer="unknown", evidence="Unsupported query type.")

    def export_state(self) -> Dict[str, object]:
        return self.state.to_dict()
