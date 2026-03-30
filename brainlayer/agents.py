from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional

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

    def reset(self) -> None:
        self.state = BrainLayerState()

    def observe(self, scenario_slug: str, observation: Observation) -> None:
        if observation.memory_type in {"preference", "correction"}:
            episode = self.state.record_episode(
                scenario=scenario_slug,
                summary=observation.text,
                tags=[observation.memory_type, observation.payload["key"]],
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
            self.state.add_working_item(
                content=f"Current {belief.key}: {belief.value}",
                priority=observation.salience,
                source_refs=[episode.id, belief.id],
            )
            if belief.key == "response_style":
                self.state.upsert_autobio_note(
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
            self.state.add_working_item(
                content=f"When {procedure.trigger}, {procedure.steps[0]}",
                priority=observation.salience,
                source_refs=[episode.id, procedure.id],
            )
            return

        self.state.record_episode(
            scenario=scenario_slug,
            summary=observation.text,
            tags=["noise"],
            salience=observation.salience,
            outcome="ignored",
        )

    def answer(self, query: Query) -> AnswerRecord:
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

        return AnswerRecord(answer="unknown", evidence="Unsupported query type.")

    def export_state(self) -> Dict[str, object]:
        return self.state.to_dict()
