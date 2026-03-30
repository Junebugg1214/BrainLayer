from __future__ import annotations

from pathlib import Path

from .agents import AnswerRecord, BrainLayerAgent
from .consolidation import ConsolidationConfig, ConsolidationReport
from .models import BrainLayerState
from .scenarios import Observation, Query
from .storage import load_state, save_state


class BrainLayerSession:
    """Small stateful wrapper for using BrainLayer in a persistent agent loop."""

    def __init__(
        self,
        state: BrainLayerState | None = None,
        *,
        auto_consolidate: bool = True,
        consolidation_config: ConsolidationConfig | None = None,
    ) -> None:
        self.agent = BrainLayerAgent(
            state=state or BrainLayerState(),
            auto_consolidate=auto_consolidate,
            consolidation_config=consolidation_config,
        )

    @property
    def state(self) -> BrainLayerState:
        return self.agent.state

    @classmethod
    def from_file(cls, path: str | Path, validate: bool = True) -> "BrainLayerSession":
        return cls(state=load_state(path, validate=validate))

    def observe(
        self,
        *,
        text: str,
        memory_type: str,
        payload: dict[str, str],
        salience: float = 0.5,
        scenario_slug: str = "live_session",
    ) -> None:
        self.agent.observe(
            scenario_slug,
            Observation(
                text=text,
                memory_type=memory_type,
                payload=payload,
                salience=salience,
            ),
        )

    def answer(
        self,
        *,
        prompt: str,
        query_type: str,
        answer_key: str = "value",
        lookup_key: str = "",
        procedure_trigger: str = "",
    ) -> AnswerRecord:
        return self.agent.answer(
            Query(
                prompt=prompt,
                query_type=query_type,
                expected_answer="",
                answer_key=answer_key,
                lookup_key=lookup_key,
                procedure_trigger=procedure_trigger,
            )
        )

    def consolidate(self) -> ConsolidationReport:
        return self.agent.consolidate()

    def save(self, path: str | Path, validate: bool = True) -> Path:
        return save_state(self.state, path, validate=validate)
