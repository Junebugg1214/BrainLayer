from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

from .consolidation import ConsolidationReport
from .llm import ModelMessage
from .runtime import RetrievedMemory
from .scenarios import Observation


def serialize_prompt_messages(messages: Sequence[ModelMessage]) -> list[dict[str, str]]:
    return [{"role": message.role, "content": message.content} for message in messages]


def serialize_retrieved_memories(memories: Sequence[RetrievedMemory]) -> list[dict[str, object]]:
    return [
        {
            "layer": memory.layer,
            "record_id": memory.record_id,
            "score": memory.score,
            "content": memory.content,
        }
        for memory in memories
    ]


def serialize_observations(observations: Sequence[Observation]) -> list[dict[str, object]]:
    return [
        {
            "text": observation.text,
            "memory_type": observation.memory_type,
            "payload": dict(observation.payload),
            "salience": observation.salience,
        }
        for observation in observations
    ]


def serialize_consolidation_report(
    report: ConsolidationReport | None,
) -> dict[str, list[str]] | None:
    if report is None:
        return None
    return {
        "promoted_belief_keys": list(report.promoted_belief_keys),
        "promoted_procedure_triggers": list(report.promoted_procedure_triggers),
        "updated_working_keys": list(report.updated_working_keys),
        "updated_autobio_keys": list(report.updated_autobio_keys),
        "forgotten_episode_ids": list(report.forgotten_episode_ids),
        "paused_working_item_ids": list(report.paused_working_item_ids),
    }


def estimate_usage_cost_usd(
    usage_metrics: Mapping[str, float],
    *,
    input_cost_per_1k_tokens: float = 0.0,
    output_cost_per_1k_tokens: float = 0.0,
    total_cost_per_1k_tokens: float = 0.0,
) -> float:
    prompt_tokens = float(usage_metrics.get("prompt_tokens", 0.0) or 0.0)
    completion_tokens = float(usage_metrics.get("completion_tokens", 0.0) or 0.0)
    total_tokens = float(usage_metrics.get("total_tokens", 0.0) or 0.0)

    if prompt_tokens or completion_tokens:
        cost = (
            (prompt_tokens / 1000.0) * input_cost_per_1k_tokens
            + (completion_tokens / 1000.0) * output_cost_per_1k_tokens
        )
        if cost > 0.0:
            return round(cost, 8)

    if total_tokens and total_cost_per_1k_tokens:
        return round((total_tokens / 1000.0) * total_cost_per_1k_tokens, 8)

    return 0.0


def write_case_artifact(
    artifact_root: Path,
    filename: str,
    payload: Mapping[str, object],
) -> str:
    artifact_root.mkdir(parents=True, exist_ok=True)
    target = artifact_root / filename
    target.write_text(json.dumps(payload, indent=2) + "\n")
    return str(Path("case_artifacts") / filename)


__all__ = [
    "estimate_usage_cost_usd",
    "serialize_consolidation_report",
    "serialize_observations",
    "serialize_prompt_messages",
    "serialize_retrieved_memories",
    "write_case_artifact",
]
