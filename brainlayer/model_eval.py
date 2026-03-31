from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from .agents import BrainLayerFeatureConfig
from .llm import LLMAdapter, ModelMessage, ModelResponse
from .runtime import BrainLayerRuntime
from .session import BrainLayerSession


ROOT = Path(__file__).resolve().parent.parent
SLOT_RE = re.compile(r"(?P<key>[a-z_]+)\s=\s(?P<value>[^.]+)\.")
PROCEDURE_RE = re.compile(r"When (?P<trigger>[^,]+), (?P<step>[^.]+)\.")
TASK_RE = re.compile(
    r"Task:\n(?P<task>.*?)\n\nReturn a JSON object",
    re.DOTALL,
)
CONTEXT_RE = re.compile(
    r"BrainLayer context:\n(?P<context>.*?)\n\nTask:\n",
    re.DOTALL,
)


@dataclass(frozen=True)
class ModelEvalTurn:
    prompt: str
    expected_answer: str = ""
    checkpoint: str = ""


@dataclass(frozen=True)
class ModelEvalScenario:
    slug: str
    title: str
    description: str
    turns: List[ModelEvalTurn]


@dataclass(frozen=True)
class ModelEvalResult:
    scenario_slug: str
    checkpoint: str
    runtime_name: str
    expected: str
    actual: str
    passed: bool
    retrieved_layers: List[str]
    state_metrics: Dict[str, float]
    exported_state: Dict[str, object]


@dataclass(frozen=True)
class ModelEvalSummary:
    runtime_name: str
    passed: int
    total: int
    pass_rate: float
    avg_metrics: Dict[str, float]


MODEL_EVAL_SCENARIOS: List[ModelEvalScenario] = [
    ModelEvalScenario(
        slug="model_preference_revision",
        title="Model Preference Revision",
        description="Can the model-backed loop revise a stored preference after an explicit correction?",
        turns=[
            ModelEvalTurn(
                prompt=(
                    "Record preference: key=response_style; value=detailed; "
                    "proposition=The user prefers detailed replies."
                )
            ),
            ModelEvalTurn(
                prompt="Record noise: value=Use markdown tables in the benchmark appendix."
            ),
            ModelEvalTurn(
                prompt="What response style should you use right now?",
                expected_answer="detailed",
                checkpoint="midpoint",
            ),
            ModelEvalTurn(
                prompt=(
                    "Record correction: key=response_style; value=brief; "
                    "proposition=The user now prefers brief replies by default."
                )
            ),
            ModelEvalTurn(
                prompt="What response style should you use right now?",
                expected_answer="brief",
                checkpoint="final_revision",
            ),
        ],
    ),
    ModelEvalScenario(
        slug="model_goal_revision",
        title="Model Goal Revision",
        description="Can the runtime keep the latest active goal after a contradictory update?",
        turns=[
            ModelEvalTurn(
                prompt=(
                    "Record goal: key=primary_goal; value=preserve citations; "
                    "summary=The current primary goal is to preserve citations in every answer."
                )
            ),
            ModelEvalTurn(
                prompt="Record noise: value=The release note headline should use title case."
            ),
            ModelEvalTurn(
                prompt="What is the current primary goal for this task?",
                expected_answer="preserve citations",
                checkpoint="initial_goal",
            ),
            ModelEvalTurn(
                prompt=(
                    "Record goal: key=primary_goal; value=ship eval report; "
                    "summary=The current primary goal is to ship the evaluation report."
                )
            ),
            ModelEvalTurn(
                prompt="What is the current primary goal for this task?",
                expected_answer="ship eval report",
                checkpoint="revised_goal",
            ),
        ],
    ),
    ModelEvalScenario(
        slug="model_relationship_revision",
        title="Model Relationship Revision",
        description="Can the runtime update collaboration framing and keep the latest relationship state?",
        turns=[
            ModelEvalTurn(
                prompt=(
                    "Record relationship: key=collaboration_mode; value=task executor; "
                    "summary=The collaboration mode is task executor.; "
                    "themes=relationship,project-mode"
                )
            ),
            ModelEvalTurn(
                prompt="What collaboration mode should define this project right now?",
                expected_answer="task executor",
                checkpoint="initial_frame",
            ),
            ModelEvalTurn(
                prompt=(
                    "Record relationship: key=collaboration_mode; value=research partner; "
                    "summary=The collaboration mode is research partner.; "
                    "themes=relationship,research-mode"
                )
            ),
            ModelEvalTurn(
                prompt="What collaboration mode should define this project right now?",
                expected_answer="research partner",
                checkpoint="revised_frame",
            ),
        ],
    ),
    ModelEvalScenario(
        slug="model_hint_then_correction",
        title="Model Hint Consolidation Then Correction",
        description="Can repeated weak signals consolidate before a later explicit contradiction revises them?",
        turns=[
            ModelEvalTurn(
                prompt=(
                    "Record preference hint: key=response_style; value=concise; "
                    "proposition=The user likely prefers concise replies."
                )
            ),
            ModelEvalTurn(
                prompt="Record noise: value=The references section should use hanging indents."
            ),
            ModelEvalTurn(
                prompt=(
                    "Record preference hint: key=response_style; value=concise; "
                    "proposition=The user likely prefers concise replies."
                )
            ),
            ModelEvalTurn(
                prompt="What response style should you use right now?",
                expected_answer="concise",
                checkpoint="hint_consolidated",
            ),
            ModelEvalTurn(
                prompt=(
                    "Record correction: key=response_style; value=detailed; "
                    "proposition=The user now wants detailed replies for this study."
                )
            ),
            ModelEvalTurn(
                prompt="What response style should you use right now?",
                expected_answer="detailed",
                checkpoint="post_correction",
            ),
        ],
    ),
]


class HeuristicBrainLayerEvalAdapter(LLMAdapter):
    """Deterministic adapter that behaves like a simple model over retrieved BrainLayer context."""

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
        return ModelResponse(content=json.dumps(response_payload))

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
        if lowered.startswith("record "):
            return self._record_response(task)
        return self._query_response(task, context)

    def _record_response(self, task: str) -> Dict[str, object]:
        match = re.match(r"^Record (?P<label>[a-z ]+): (?P<body>.+)$", task)
        if not match:
            return {
                "assistant_response": "Unable to parse the record request.",
                "episodic_summary": f"Failed to parse record request: {task}",
                "memory_observations": [],
            }

        label = match.group("label").strip().replace(" ", "_")
        fields = self._parse_fields(match.group("body"))
        memory_type = self._normalize_memory_type(label)
        payload = self._build_payload(memory_type, fields)
        observation_text = self._build_observation_text(memory_type, payload)
        return {
            "assistant_response": self._build_store_ack(memory_type, payload),
            "episodic_summary": f"Stored {memory_type} signal for future turns.",
            "memory_observations": [
                {
                    "text": observation_text,
                    "memory_type": memory_type,
                    "salience": self._default_salience(memory_type),
                    "payload": payload,
                }
            ],
        }

    def _query_response(self, task: str, context: str) -> Dict[str, object]:
        memories = self._parse_context(context)
        answer = "unknown"
        lowered = task.lower()

        if "response style" in lowered:
            answer = memories["slots"].get("response_style", "unknown")
        elif "primary goal" in lowered:
            answer = memories["slots"].get("primary_goal", "unknown")
        elif "collaboration mode" in lowered:
            answer = memories["slots"].get("collaboration_mode", "unknown")
        elif "before retrying" in lowered or "do first" in lowered:
            answer = memories["procedures"].get("retry_release", "unknown")

        return {
            "assistant_response": answer,
            "episodic_summary": f"Answered query using BrainLayer context: {answer}",
            "memory_observations": [],
        }

    def _parse_fields(self, body: str) -> Dict[str, str]:
        fields: Dict[str, str] = {}
        for chunk in body.split(";"):
            piece = chunk.strip()
            if not piece or "=" not in piece:
                continue
            key, value = piece.split("=", 1)
            fields[key.strip()] = value.strip()
        return fields

    def _normalize_memory_type(self, label: str) -> str:
        normalized = label.strip().replace(" ", "_")
        allowed = {
            "preference",
            "correction",
            "goal",
            "relationship",
            "preference_hint",
            "noise",
        }
        if normalized not in allowed:
            raise ValueError(f"Unsupported eval memory label: {label}")
        return normalized

    def _build_payload(self, memory_type: str, fields: Dict[str, str]) -> Dict[str, str]:
        if memory_type in {"preference", "correction", "preference_hint"}:
            return {
                "key": fields["key"],
                "value": fields["value"],
                "proposition": fields["proposition"],
            }
        if memory_type == "goal":
            return {
                "key": fields["key"],
                "value": fields["value"],
                "summary": fields["summary"],
            }
        if memory_type == "relationship":
            return {
                "key": fields["key"],
                "value": fields["value"],
                "summary": fields["summary"],
                "themes": fields["themes"],
            }
        if memory_type == "noise":
            return {"value": fields["value"]}
        raise ValueError(f"Unsupported memory type in eval adapter: {memory_type}")

    def _build_observation_text(self, memory_type: str, payload: Dict[str, str]) -> str:
        if memory_type in {"preference", "correction", "preference_hint"}:
            return payload["proposition"]
        if memory_type == "goal":
            return payload["summary"]
        if memory_type == "relationship":
            return payload["summary"]
        if memory_type == "noise":
            return payload["value"]
        return "Captured eval memory."

    def _build_store_ack(self, memory_type: str, payload: Dict[str, str]) -> str:
        if "key" in payload and "value" in payload:
            return f"Stored {memory_type} for {payload['key']} = {payload['value']}."
        return f"Stored {memory_type} detail."

    def _default_salience(self, memory_type: str) -> float:
        if memory_type == "noise":
            return 0.18
        if memory_type.endswith("_hint"):
            return 0.42
        if memory_type in {"correction", "goal", "relationship"}:
            return 0.95
        return 0.9

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
            }:
                slots.setdefault(slot_match.group("key"), slot_match.group("value").strip())

            procedure_match = PROCEDURE_RE.search(content)
            if procedure_match and layer == "procedures":
                procedures.setdefault(
                    procedure_match.group("trigger").strip(),
                    procedure_match.group("step").strip(),
                )

        return {"slots": slots, "procedures": procedures}


def build_runtime_variants(include_ablations: bool = True) -> List[tuple[str, BrainLayerFeatureConfig]]:
    variants = [("model_loop", BrainLayerFeatureConfig())]
    if not include_ablations:
        return variants

    variants.extend(
        [
            (
                "model_loop_no_consolidation",
                BrainLayerFeatureConfig(enable_consolidation=False),
            ),
            (
                "model_loop_no_forgetting",
                BrainLayerFeatureConfig(enable_forgetting=False),
            ),
            (
                "model_loop_no_autobio",
                BrainLayerFeatureConfig(enable_autobio=False),
            ),
            (
                "model_loop_no_working_state",
                BrainLayerFeatureConfig(enable_working_state=False),
            ),
        ]
    )
    return variants


def collect_state_metrics(exported_state: Dict[str, object]) -> Dict[str, float]:
    working_state = exported_state.get("working_state", [])
    episodes = exported_state.get("episodes", [])
    beliefs = exported_state.get("beliefs", [])
    autobiographical_state = exported_state.get("autobiographical_state", [])
    procedures = exported_state.get("procedures", [])
    active_working = [
        item for item in working_state if isinstance(item, dict) and item.get("status") == "active"
    ]
    total_records = (
        len(working_state)
        + len(episodes)
        + len(beliefs)
        + len(autobiographical_state)
        + len(procedures)
    )
    return {
        "total_records": float(total_records),
        "episodes": float(len(episodes)),
        "beliefs": float(len(beliefs)),
        "procedures": float(len(procedures)),
        "autobio_notes": float(len(autobiographical_state)),
        "active_working_items": float(len(active_working)),
    }


def normalize_answer(value: str) -> str:
    return " ".join(value.strip().lower().split())


def run_model_eval_scenario(
    scenario: ModelEvalScenario,
    *,
    include_ablations: bool = True,
) -> List[ModelEvalResult]:
    results: List[ModelEvalResult] = []
    for runtime_name, features in build_runtime_variants(include_ablations=include_ablations):
        runtime = BrainLayerRuntime(
            HeuristicBrainLayerEvalAdapter(),
            session=BrainLayerSession(features=features),
            model="heuristic-brainlayer-eval",
        )
        for turn in scenario.turns:
            turn_result = runtime.run_turn(turn.prompt, scenario_slug=scenario.slug)
            if not turn.checkpoint:
                continue

            results.append(
                ModelEvalResult(
                    scenario_slug=scenario.slug,
                    checkpoint=turn.checkpoint,
                    runtime_name=runtime_name,
                    expected=turn.expected_answer,
                    actual=turn_result.assistant_response,
                    passed=normalize_answer(turn_result.assistant_response)
                    == normalize_answer(turn.expected_answer),
                    retrieved_layers=[
                        memory.layer for memory in turn_result.retrieved_memories
                    ],
                    state_metrics=collect_state_metrics(turn_result.exported_state),
                    exported_state=turn_result.exported_state,
                )
            )
    return results


def run_model_eval_suite(
    scenarios: Iterable[ModelEvalScenario] | None = None,
    *,
    include_ablations: bool = True,
) -> List[ModelEvalResult]:
    active_scenarios = list(scenarios or MODEL_EVAL_SCENARIOS)
    results: List[ModelEvalResult] = []
    for scenario in active_scenarios:
        results.extend(
            run_model_eval_scenario(scenario, include_ablations=include_ablations)
        )
    return results


def summarize_model_eval_results(results: Sequence[ModelEvalResult]) -> List[ModelEvalSummary]:
    passed_counts: Dict[str, int] = {}
    totals: Dict[str, int] = {}
    metric_totals: Dict[str, Dict[str, float]] = {}

    for result in results:
        totals[result.runtime_name] = totals.get(result.runtime_name, 0) + 1
        passed_counts[result.runtime_name] = (
            passed_counts.get(result.runtime_name, 0) + int(result.passed)
        )
        runtime_metric_totals = metric_totals.setdefault(result.runtime_name, {})
        for key, value in result.state_metrics.items():
            runtime_metric_totals[key] = runtime_metric_totals.get(key, 0.0) + value

    summaries: List[ModelEvalSummary] = []
    for runtime_name in sorted(totals):
        total = totals[runtime_name]
        avg_metrics = {
            key: value / total for key, value in metric_totals.get(runtime_name, {}).items()
        }
        summaries.append(
            ModelEvalSummary(
                runtime_name=runtime_name,
                passed=passed_counts[runtime_name],
                total=total,
                pass_rate=passed_counts[runtime_name] / total if total else 0.0,
                avg_metrics=avg_metrics,
            )
        )
    return summaries


def render_model_eval_report(results: Sequence[ModelEvalResult]) -> str:
    lines = [
        "Model-Backed BrainLayer Eval Report",
        "===================================",
        "",
    ]
    summaries = summarize_model_eval_results(results)

    for result in results:
        status = "PASS" if result.passed else "FAIL"
        case_label = f"{result.scenario_slug}/{result.checkpoint}"
        layers = ",".join(result.retrieved_layers)
        lines.append(
            f"[{status}] {result.runtime_name} on {case_label}: "
            f"expected={result.expected!r}, actual={result.actual!r}, "
            f"layers={layers}"
        )

    lines.append("")
    lines.append("Summary")
    lines.append("-------")
    for summary in summaries:
        avg_records = summary.avg_metrics.get("total_records", 0.0)
        avg_episodes = summary.avg_metrics.get("episodes", 0.0)
        lines.append(
            f"{summary.runtime_name}: {summary.passed}/{summary.total} | "
            f"avg_records={avg_records:.1f}, avg_episodes={avg_episodes:.1f}"
        )
    return "\n".join(lines)


def dump_model_eval_states(results: Sequence[ModelEvalResult], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for result in results:
        filename = f"{result.runtime_name}__{result.scenario_slug}__{result.checkpoint}.json"
        target = output_dir / filename
        target.write_text(json.dumps(result.exported_state, indent=2) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run contradiction-heavy evals against the model-backed BrainLayer loop."
    )
    parser.add_argument(
        "--core-only",
        action="store_true",
        help="Run only the full model-backed BrainLayer runtime without ablations.",
    )
    parser.add_argument(
        "--dump-states",
        type=Path,
        help="Optional directory for writing state snapshots at each checkpoint.",
    )
    args = parser.parse_args(argv)

    results = run_model_eval_suite(include_ablations=not args.core_only)
    if args.dump_states:
        dump_model_eval_states(results, args.dump_states)
    print(render_model_eval_report(results))
    return 0


__all__ = [
    "MODEL_EVAL_SCENARIOS",
    "HeuristicBrainLayerEvalAdapter",
    "ModelEvalResult",
    "ModelEvalScenario",
    "ModelEvalSummary",
    "ModelEvalTurn",
    "dump_model_eval_states",
    "render_model_eval_report",
    "run_model_eval_scenario",
    "run_model_eval_suite",
    "summarize_model_eval_results",
]
