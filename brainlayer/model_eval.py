from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from .agents import BrainLayerFeatureConfig
from .benchmark_harness import (
    append_csv,
    get_git_commit,
    slugify_label,
    utc_now_compact,
    utc_now_iso,
    write_csv,
)
from .judging import (
    BehaviorJudge,
    BehaviorJudgeInput,
    ExactMatchJudge,
    HeuristicBehaviorJudge,
    ScoreDecision,
    normalize_answer_text,
)
from .llm import (
    LLMAdapter,
    LLMError,
    ModelMessage,
    ModelResponse,
    OpenAICompatibleChatAdapter,
)
from .runtime import BrainLayerRuntime, BrainLayerRuntimeConfig
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
TOKEN_RE = re.compile(r"[a-z0-9]+")
DEFAULT_HEURISTIC_PROVIDER = "heuristic"
DEFAULT_HEURISTIC_MODEL = "heuristic-brainlayer-eval"
DEFAULT_LIVE_PROVIDER = "openai_compatible"
DEFAULT_LIVE_MODEL = os.environ.get("BRAINLAYER_MODEL", "gpt-4.1-mini")
MODEL_EVAL_SYSTEM_PROMPT = (
    "You are participating in a BrainLayer evaluation. "
    "Follow the task literally. "
    "If the task begins with 'Record', convert it into the appropriate structured "
    "memory_observations entry and keep assistant_response as a short acknowledgement. "
    "If the task asks about the current response style, primary goal, collaboration mode, "
    "or another stored value, return only the shortest value needed to answer correctly, "
    "not a full sentence. "
    "Always return valid JSON matching the requested schema."
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
    score: float
    score_method: str
    score_reason: str
    retrieved_layers: List[str]
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
class ModelEvalSummary:
    runtime_name: str
    passed: int
    total: int
    pass_rate: float
    parse_failures: int
    empty_answers: int
    errors: int
    skipped: int
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
        if memory_type in {"goal", "relationship"}:
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


def default_model_eval_runtime_config(
    *,
    temperature: float = 0.0,
    max_output_tokens: int = 700,
) -> BrainLayerRuntimeConfig:
    return BrainLayerRuntimeConfig(
        system_prompt=MODEL_EVAL_SYSTEM_PROMPT,
        response_temperature=temperature,
        max_output_tokens=max_output_tokens,
    )


def build_live_model_eval_adapter(
    *,
    api_key_env: str = "OPENAI_API_KEY",
    base_url: str = "https://api.openai.com/v1",
    request_path: str = "/chat/completions",
    timeout_seconds: float = 30.0,
    max_output_tokens_field: str | None = "max_tokens",
) -> OpenAICompatibleChatAdapter:
    return OpenAICompatibleChatAdapter(
        api_key=os.environ.get(api_key_env),
        base_url=base_url,
        request_path=request_path,
        timeout_seconds=timeout_seconds,
        max_output_tokens_field=max_output_tokens_field,
    )


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


def answers_match(expected: str, actual: str) -> bool:
    normalized_expected = normalize_answer_text(expected)
    normalized_actual = normalize_answer_text(actual)
    if not normalized_expected:
        return normalized_actual == normalized_expected
    if normalized_actual == normalized_expected:
        return True
    return f" {normalized_expected} " in f" {normalized_actual} "


def normalize_usage_metrics(payload: Dict[str, object]) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for key, value in payload.items():
        try:
            metrics[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return metrics


def _combined_metrics(result: ModelEvalResult) -> Dict[str, float]:
    metrics = dict(result.state_metrics)
    metrics["score"] = result.score
    metrics["latency_ms"] = result.latency_ms
    metrics["applied_observation_count"] = float(result.applied_observation_count)
    for key, value in result.usage_metrics.items():
        metrics[f"usage_{key}"] = value
    return metrics


def _build_runtime(
    adapter: LLMAdapter,
    *,
    features: BrainLayerFeatureConfig,
    requested_model: str,
    runtime_config: BrainLayerRuntimeConfig,
) -> BrainLayerRuntime:
    return BrainLayerRuntime(
        adapter,
        session=BrainLayerSession(features=features),
        model=requested_model,
        config=runtime_config,
    )


def _build_result_from_turn(
    *,
    scenario_slug: str,
    checkpoint: str,
    runtime_name: str,
    expected: str,
    eval_mode: str,
    provider_name: str,
    requested_model: str,
    turn_result: object,
    latency_ms: float,
    score_decision: ScoreDecision,
) -> ModelEvalResult:
    actual = turn_result.assistant_response
    response_model = turn_result.model_response.model or requested_model
    return ModelEvalResult(
        scenario_slug=scenario_slug,
        checkpoint=checkpoint,
        runtime_name=runtime_name,
        expected=expected,
        actual=actual,
        passed=score_decision.passed,
        score=score_decision.score,
        score_method=score_decision.method,
        score_reason=score_decision.reason,
        retrieved_layers=[memory.layer for memory in turn_result.retrieved_memories],
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
    checkpoint: str,
    runtime_name: str,
    expected: str,
    eval_mode: str,
    provider_name: str,
    requested_model: str,
    exported_state: Dict[str, object],
    error: str,
    latency_ms: float,
    skipped: bool,
) -> ModelEvalResult:
    return ModelEvalResult(
        scenario_slug=scenario_slug,
        checkpoint=checkpoint,
        runtime_name=runtime_name,
        expected=expected,
        actual="skipped" if skipped else "error",
        passed=False,
        score=0.0,
        score_method="runtime_error",
        score_reason=error,
        retrieved_layers=[],
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


def run_model_eval_scenario(
    scenario: ModelEvalScenario,
    *,
    include_ablations: bool = True,
    adapter: LLMAdapter | None = None,
    eval_mode: str = "heuristic",
    provider_name: str | None = None,
    requested_model: str | None = None,
    runtime_config: BrainLayerRuntimeConfig | None = None,
    behavior_scoring_mode: str = "judge",
    behavior_judge: BehaviorJudge | None = None,
) -> List[ModelEvalResult]:
    active_adapter = adapter or HeuristicBrainLayerEvalAdapter()
    active_eval_mode = eval_mode
    active_provider_name = provider_name or DEFAULT_HEURISTIC_PROVIDER
    active_requested_model = requested_model or DEFAULT_HEURISTIC_MODEL
    active_runtime_config = runtime_config or default_model_eval_runtime_config()
    active_behavior_judge = behavior_judge or _build_behavior_judge(behavior_scoring_mode)

    results: List[ModelEvalResult] = []
    for runtime_name, features in build_runtime_variants(include_ablations=include_ablations):
        runtime = _build_runtime(
            active_adapter,
            features=features,
            requested_model=active_requested_model,
            runtime_config=active_runtime_config,
        )
        blocked_error = ""
        for turn in scenario.turns:
            if blocked_error:
                if turn.checkpoint:
                    results.append(
                        _build_error_result(
                            scenario_slug=scenario.slug,
                            checkpoint=turn.checkpoint,
                            runtime_name=runtime_name,
                            expected=turn.expected_answer,
                            eval_mode=active_eval_mode,
                            provider_name=active_provider_name,
                            requested_model=active_requested_model,
                            exported_state=runtime.session.state.to_dict(),
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
                            checkpoint=turn.checkpoint,
                            runtime_name=runtime_name,
                            expected=turn.expected_answer,
                            eval_mode=active_eval_mode,
                            provider_name=active_provider_name,
                            requested_model=active_requested_model,
                            exported_state=runtime.session.state.to_dict(),
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
                            checkpoint=turn.checkpoint,
                            runtime_name=runtime_name,
                            expected=turn.expected_answer,
                            eval_mode=active_eval_mode,
                            provider_name=active_provider_name,
                            requested_model=active_requested_model,
                            exported_state=runtime.session.state.to_dict(),
                            error=blocked_error,
                            latency_ms=latency_ms,
                            skipped=False,
                        )
                    )
                continue

            latency_ms = (time.perf_counter() - started_at) * 1000.0
            if not turn.checkpoint:
                continue

            score_decision = active_behavior_judge.score(
                BehaviorJudgeInput(
                    scenario_slug=scenario.slug,
                    scenario_title=scenario.title,
                    scenario_description=scenario.description,
                    checkpoint=turn.checkpoint,
                    prompt=turn.prompt,
                    expected=turn.expected_answer,
                    actual=turn_result.assistant_response,
                )
            )

            results.append(
                _build_result_from_turn(
                    scenario_slug=scenario.slug,
                    checkpoint=turn.checkpoint,
                    runtime_name=runtime_name,
                    expected=turn.expected_answer,
                    eval_mode=active_eval_mode,
                    provider_name=active_provider_name,
                    requested_model=active_requested_model,
                    turn_result=turn_result,
                    latency_ms=latency_ms,
                    score_decision=score_decision,
                )
            )
    return results


def run_model_eval_suite(
    scenarios: Iterable[ModelEvalScenario] | None = None,
    *,
    include_ablations: bool = True,
    adapter: LLMAdapter | None = None,
    eval_mode: str = "heuristic",
    provider_name: str | None = None,
    requested_model: str | None = None,
    runtime_config: BrainLayerRuntimeConfig | None = None,
    behavior_scoring_mode: str = "judge",
    behavior_judge: BehaviorJudge | None = None,
) -> List[ModelEvalResult]:
    active_scenarios = list(scenarios or MODEL_EVAL_SCENARIOS)
    results: List[ModelEvalResult] = []
    for scenario in active_scenarios:
        results.extend(
            run_model_eval_scenario(
                scenario,
                include_ablations=include_ablations,
                adapter=adapter,
                eval_mode=eval_mode,
                provider_name=provider_name,
                requested_model=requested_model,
                runtime_config=runtime_config,
                behavior_scoring_mode=behavior_scoring_mode,
                behavior_judge=behavior_judge,
            )
        )
    return results


def run_live_model_eval_suite(
    scenarios: Iterable[ModelEvalScenario] | None = None,
    *,
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
) -> List[ModelEvalResult]:
    adapter = build_live_model_eval_adapter(
        api_key_env=api_key_env,
        base_url=base_url,
        request_path=request_path,
        timeout_seconds=timeout_seconds,
        max_output_tokens_field=max_output_tokens_field,
    )
    runtime_config = default_model_eval_runtime_config(
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )
    return run_model_eval_suite(
        scenarios,
        include_ablations=include_ablations,
        adapter=adapter,
        eval_mode="live",
        provider_name=provider_name,
        requested_model=requested_model,
        runtime_config=runtime_config,
        behavior_scoring_mode=behavior_scoring_mode,
        behavior_judge=behavior_judge,
    )


def _build_behavior_judge(behavior_scoring_mode: str) -> BehaviorJudge:
    if behavior_scoring_mode == "exact":
        return ExactMatchJudge()
    if behavior_scoring_mode == "judge":
        return HeuristicBehaviorJudge()
    raise ValueError(f"Unsupported behavior scoring mode: {behavior_scoring_mode}")


def summarize_model_eval_results(results: Sequence[ModelEvalResult]) -> List[ModelEvalSummary]:
    passed_counts: Dict[str, int] = {}
    totals: Dict[str, int] = {}
    parse_failures: Dict[str, int] = {}
    empty_answers: Dict[str, int] = {}
    errors: Dict[str, int] = {}
    skipped: Dict[str, int] = {}
    metric_totals: Dict[str, Dict[str, float]] = {}

    for result in results:
        totals[result.runtime_name] = totals.get(result.runtime_name, 0) + 1
        passed_counts[result.runtime_name] = (
            passed_counts.get(result.runtime_name, 0) + int(result.passed)
        )
        parse_failures[result.runtime_name] = (
            parse_failures.get(result.runtime_name, 0) + int(result.parse_failure)
        )
        empty_answers[result.runtime_name] = (
            empty_answers.get(result.runtime_name, 0) + int(result.empty_answer)
        )
        errors[result.runtime_name] = errors.get(result.runtime_name, 0) + int(bool(result.error))
        skipped[result.runtime_name] = skipped.get(result.runtime_name, 0) + int(result.skipped)
        runtime_metric_totals = metric_totals.setdefault(result.runtime_name, {})
        for key, value in _combined_metrics(result).items():
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
                parse_failures=parse_failures.get(runtime_name, 0),
                empty_answers=empty_answers.get(runtime_name, 0),
                errors=errors.get(runtime_name, 0),
                skipped=skipped.get(runtime_name, 0),
                avg_metrics=avg_metrics,
            )
        )
    return summaries


def _describe_backend(results: Sequence[ModelEvalResult]) -> str:
    if not results:
        return ""
    first = results[0]
    same_backend = all(
        result.eval_mode == first.eval_mode
        and result.provider_name == first.provider_name
        and result.requested_model == first.requested_model
        for result in results
    )
    if not same_backend:
        return ""
    return (
        f"Mode: {first.eval_mode} | provider={first.provider_name} "
        f"| requested_model={first.requested_model}"
    )


def render_model_eval_report(results: Sequence[ModelEvalResult]) -> str:
    lines = [
        "Model-Backed BrainLayer Eval Report",
        "===================================",
    ]
    backend_line = _describe_backend(results)
    if backend_line:
        lines.extend(["", backend_line])
    lines.append("")
    summaries = summarize_model_eval_results(results)

    for result in results:
        status = "SKIP" if result.skipped else "PASS" if result.passed else "FAIL"
        case_label = f"{result.scenario_slug}/{result.checkpoint}"
        layers = ",".join(result.retrieved_layers) or "-"
        extras = [
            f"score={result.score:.2f}",
            f"scoring={result.score_method}",
            f"latency_ms={result.latency_ms:.1f}",
            f"json={str(result.used_json).lower()}",
        ]
        if result.finish_reason:
            extras.append(f"finish={result.finish_reason}")
        if result.empty_answer:
            extras.append("empty_answer=true")
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
        avg_records = summary.avg_metrics.get("total_records", 0.0)
        avg_episodes = summary.avg_metrics.get("episodes", 0.0)
        avg_latency = summary.avg_metrics.get("latency_ms", 0.0)
        extras = [
            f"avg_records={avg_records:.1f}",
            f"avg_episodes={avg_episodes:.1f}",
            f"avg_score={summary.avg_metrics.get('score', 0.0):.2f}",
            f"avg_latency_ms={avg_latency:.1f}",
            f"parse_failures={summary.parse_failures}",
            f"empty_answers={summary.empty_answers}",
            f"errors={summary.errors}",
        ]
        avg_total_tokens = summary.avg_metrics.get("usage_total_tokens", 0.0)
        if avg_total_tokens:
            extras.append(f"avg_total_tokens={avg_total_tokens:.1f}")
        lines.append(
            f"{summary.runtime_name}: {summary.passed}/{summary.total} | "
            + ", ".join(extras)
        )
    return "\n".join(lines)


def serializable_model_eval_result(result: ModelEvalResult) -> Dict[str, object]:
    payload: Dict[str, object] = {
        "scenario_slug": result.scenario_slug,
        "checkpoint": result.checkpoint,
        "case_label": f"{result.scenario_slug}/{result.checkpoint}",
        "runtime_name": result.runtime_name,
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
    for key, value in sorted(result.state_metrics.items()):
        payload[f"metric_{key}"] = value
    for key, value in sorted(result.usage_metrics.items()):
        payload[f"usage_{key}"] = value
    return payload


def serializable_model_eval_summary(summary: ModelEvalSummary) -> Dict[str, object]:
    payload: Dict[str, object] = {
        "runtime_name": summary.runtime_name,
        "passed": summary.passed,
        "total": summary.total,
        "pass_rate": summary.pass_rate,
        "parse_failures": summary.parse_failures,
        "empty_answers": summary.empty_answers,
        "errors": summary.errors,
        "skipped": summary.skipped,
    }
    for key, value in sorted(summary.avg_metrics.items()):
        payload[f"avg_{key}"] = value
    return payload


def build_model_eval_metadata(
    results: Sequence[ModelEvalResult],
    *,
    include_ablations: bool,
    label: str | None,
) -> Dict[str, object]:
    timestamp = utc_now_compact()
    run_id = timestamp if not label else f"{timestamp}-{slugify_label(label)}"
    first = results[0] if results else None
    return {
        "run_id": run_id,
        "generated_at_utc": utc_now_iso(),
        "git_commit": get_git_commit(),
        "include_ablations": include_ablations,
        "label": label or "",
        "eval_mode": first.eval_mode if first else "",
        "provider_name": first.provider_name if first else "",
        "requested_model": first.requested_model if first else "",
        "scenario_count": len({result.scenario_slug for result in results}),
        "checkpoint_count": len({(result.scenario_slug, result.checkpoint) for result in results}),
        "runtime_count": len({result.runtime_name for result in results}),
        "response_model_count": len({result.response_model for result in results if result.response_model}),
        "score_methods": sorted({result.score_method for result in results}),
    }


def render_model_eval_x_post(
    summaries: Sequence[ModelEvalSummary],
    *,
    include_ablations: bool,
    label: str | None,
    eval_mode: str,
    requested_model: str,
) -> str:
    summary_by_name = {summary.runtime_name: summary for summary in summaries}
    model_loop = summary_by_name.get("model_loop")
    if model_loop is None:
        return "BrainLayer model-loop eval completed."

    if eval_mode == "live":
        prefix = f"BrainLayer live eval {requested_model}"
    else:
        prefix = "BrainLayer model-loop eval"
    if label:
        prefix = f"{prefix} ({label})"

    parts = [f"{prefix}: full runtime {model_loop.passed}/{model_loop.total}."]
    if model_loop.parse_failures or model_loop.empty_answers or model_loop.errors:
        parts.append(
            "Reliability:"
            f" parse_failures {model_loop.parse_failures},"
            f" empty_answers {model_loop.empty_answers},"
            f" errors {model_loop.errors}."
        )

    if include_ablations:
        no_consolidation = summary_by_name.get("model_loop_no_consolidation")
        no_autobio = summary_by_name.get("model_loop_no_autobio")
        no_working_state = summary_by_name.get("model_loop_no_working_state")
        no_forgetting = summary_by_name.get("model_loop_no_forgetting")
        if no_consolidation and no_autobio and no_working_state and no_forgetting:
            parts.append(
                "Ablations:"
                f" no_consolidation {no_consolidation.passed}/{no_consolidation.total},"
                f" no_autobio {no_autobio.passed}/{no_autobio.total},"
                f" no_working_state {no_working_state.passed}/{no_working_state.total}."
            )
            parts.append(
                "No-forgetting stayed"
                f" {no_forgetting.passed}/{no_forgetting.total}"
                f" but retained more state"
                f" ({no_forgetting.avg_metrics.get('total_records', 0.0):.1f}"
                f" vs {model_loop.avg_metrics.get('total_records', 0.0):.1f} avg records)."
            )

    return " ".join(parts)


def export_model_eval_results(
    results: Sequence[ModelEvalResult],
    export_root: Path,
    *,
    include_ablations: bool,
    label: str | None = None,
) -> Path:
    summaries = summarize_model_eval_results(results)
    metadata = build_model_eval_metadata(
        results,
        include_ablations=include_ablations,
        label=label,
    )
    run_dir = export_root / str(metadata["run_id"])
    run_dir.mkdir(parents=True, exist_ok=True)

    result_rows = [serializable_model_eval_result(result) for result in results]
    summary_rows = [serializable_model_eval_summary(summary) for summary in summaries]
    x_post = render_model_eval_x_post(
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
                "eval_mode": metadata["eval_mode"],
                "provider_name": metadata["provider_name"],
                "requested_model": metadata["requested_model"],
                "score_methods": ",".join(metadata["score_methods"]),
                **row,
            }
        )

    append_csv(export_root / "model_eval_history.csv", history_rows)
    with (export_root / "model_eval_history.jsonl").open("a") as handle:
        handle.write(json.dumps(payload) + "\n")

    return run_dir


def dump_model_eval_states(results: Sequence[ModelEvalResult], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for result in results:
        filename = f"{result.runtime_name}__{result.scenario_slug}__{result.checkpoint}.json"
        target = output_dir / filename
        target.write_text(json.dumps(result.exported_state, indent=2) + "\n")


def _build_adapter_from_args(args: argparse.Namespace) -> tuple[LLMAdapter, str, str]:
    if args.mode == "heuristic":
        return HeuristicBrainLayerEvalAdapter(), DEFAULT_HEURISTIC_PROVIDER, DEFAULT_HEURISTIC_MODEL

    max_output_tokens_field = args.max_output_tokens_field
    if max_output_tokens_field is not None and max_output_tokens_field.lower() == "none":
        max_output_tokens_field = None
    adapter = build_live_model_eval_adapter(
        api_key_env=args.api_key_env,
        base_url=args.base_url,
        request_path=args.request_path,
        timeout_seconds=args.timeout_seconds,
        max_output_tokens_field=max_output_tokens_field,
    )
    return adapter, args.provider_name, args.model


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run contradiction-heavy evals against the model-backed BrainLayer loop."
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
        default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
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
        help="Maximum output tokens requested per live evaluation turn.",
    )
    parser.add_argument(
        "--core-only",
        action="store_true",
        help="Run only the full model-backed BrainLayer runtime without ablations.",
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
        help="Write per-run CSV/JSON summaries into DIR and append model-loop history files there.",
    )
    parser.add_argument(
        "--label",
        help="Optional label to attach to exported model-loop eval runs for later comparison.",
    )
    args = parser.parse_args(argv)

    adapter, provider_name, requested_model = _build_adapter_from_args(args)
    runtime_config = default_model_eval_runtime_config(
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
    )
    include_ablations = not args.core_only
    results = run_model_eval_suite(
        include_ablations=include_ablations,
        adapter=adapter,
        eval_mode=args.mode,
        provider_name=provider_name,
        requested_model=requested_model,
        runtime_config=runtime_config,
        behavior_scoring_mode="exact" if args.score_exact else "judge",
    )
    if args.dump_states:
        dump_model_eval_states(results, args.dump_states)
    print(render_model_eval_report(results))
    if args.export_results:
        run_dir = export_model_eval_results(
            results,
            args.export_results,
            include_ablations=include_ablations,
            label=args.label,
        )
        print("")
        print(f"Model-loop exports written to {run_dir}")
        print(f"X post saved to {run_dir / 'x_post.txt'}")
    return 0


__all__ = [
    "MODEL_EVAL_SCENARIOS",
    "HeuristicBrainLayerEvalAdapter",
    "ModelEvalResult",
    "ModelEvalScenario",
    "ModelEvalSummary",
    "ModelEvalTurn",
    "build_live_model_eval_adapter",
    "default_model_eval_runtime_config",
    "dump_model_eval_states",
    "export_model_eval_results",
    "render_model_eval_report",
    "render_model_eval_x_post",
    "run_live_model_eval_suite",
    "run_model_eval_scenario",
    "run_model_eval_suite",
    "summarize_model_eval_results",
]
