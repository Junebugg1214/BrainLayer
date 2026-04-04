from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

from .benchmark_harness import (
    append_csv,
    get_git_commit,
    slugify_label,
    utc_now_compact,
    utc_now_iso,
    write_csv,
)
from .eval_support import estimate_usage_cost_usd, write_case_artifact
from .llm import LLMAdapter
from .model_eval import (
    RUNTIME_PROFILE_DEFAULT,
    RUNTIME_PROFILE_STUDY_V2,
    DEFAULT_SCENARIO_PACK as DEFAULT_MODEL_SCENARIO_PACK,
    DEFAULT_HEURISTIC_MODEL,
    DEFAULT_HEURISTIC_PROVIDER,
    DEFAULT_LIVE_MODEL,
    DEFAULT_LIVE_PROVIDER,
    HeuristicBrainLayerEvalAdapter,
    ModelEvalResult,
    build_live_model_eval_adapter,
    default_model_eval_runtime_config,
    run_model_eval_suite,
)
from .natural_eval import (
    HeuristicNaturalConversationAdapter,
    NaturalEvalResult,
    default_natural_eval_runtime_config,
    run_natural_eval_suite,
)


SUITE_NAMES = ("contradiction", "natural")
DEFAULT_MATRIX_CONFIG = Path("examples/model_matrix.sample.json")


@dataclass(frozen=True)
class ModelMatrixEntry:
    name: str
    mode: str = "heuristic"
    provider_name: str = DEFAULT_LIVE_PROVIDER
    requested_model: str = DEFAULT_LIVE_MODEL
    base_url: str = "https://api.openai.com/v1"
    api_key_env: str = "OPENAI_API_KEY"
    request_path: str = "/chat/completions"
    timeout_seconds: float = 30.0
    max_output_tokens_field: str | None = "max_tokens"
    temperature: float = 0.0
    max_output_tokens: int = 700
    input_cost_per_1k_tokens: float = 0.0
    output_cost_per_1k_tokens: float = 0.0
    total_cost_per_1k_tokens: float = 0.0
    enabled: bool = True

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ModelMatrixEntry":
        name = str(payload["name"]).strip()
        mode = str(payload.get("mode", "heuristic")).strip()
        if mode not in {"heuristic", "live"}:
            raise ValueError(f"Unsupported matrix entry mode: {mode}")

        provider_name = str(
            payload.get(
                "provider_name",
                DEFAULT_HEURISTIC_PROVIDER if mode == "heuristic" else DEFAULT_LIVE_PROVIDER,
            )
        ).strip()
        requested_model = str(
            payload.get(
                "model",
                DEFAULT_HEURISTIC_MODEL if mode == "heuristic" else DEFAULT_LIVE_MODEL,
            )
        ).strip()
        max_output_tokens_field = payload.get("max_output_tokens_field", "max_tokens")
        if isinstance(max_output_tokens_field, str) and max_output_tokens_field.lower() == "none":
            max_output_tokens_field = None
        elif max_output_tokens_field is not None:
            max_output_tokens_field = str(max_output_tokens_field)

        return cls(
            name=name,
            mode=mode,
            provider_name=provider_name,
            requested_model=requested_model,
            base_url=str(payload.get("base_url", "https://api.openai.com/v1")),
            api_key_env=str(payload.get("api_key_env", "OPENAI_API_KEY")),
            request_path=str(payload.get("request_path", "/chat/completions")),
            timeout_seconds=float(payload.get("timeout_seconds", 30.0)),
            max_output_tokens_field=max_output_tokens_field,
            temperature=float(payload.get("temperature", 0.0)),
            max_output_tokens=int(payload.get("max_output_tokens", 700)),
            input_cost_per_1k_tokens=float(payload.get("input_cost_per_1k_tokens", 0.0)),
            output_cost_per_1k_tokens=float(payload.get("output_cost_per_1k_tokens", 0.0)),
            total_cost_per_1k_tokens=float(payload.get("total_cost_per_1k_tokens", 0.0)),
            enabled=bool(payload.get("enabled", True)),
        )


@dataclass(frozen=True)
class MatrixCaseResult:
    entry_name: str
    suite_name: str
    runtime_name: str
    scenario_slug: str
    checkpoint: str
    case_label: str
    evaluation_type: str
    target_layer: str
    target_key: str
    expected: str
    actual: str
    passed: bool
    score: float
    score_method: str
    score_reason: str
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
    estimated_cost_usd: float
    error: str
    skipped: bool
    retrieved_layers: List[str]
    case_artifact: Dict[str, object]
    exported_state: Dict[str, object]
    state_metrics: Dict[str, float]
    usage_metrics: Dict[str, float]


@dataclass(frozen=True)
class MatrixSuiteSummary:
    entry_name: str
    suite_name: str
    runtime_name: str
    eval_mode: str
    provider_name: str
    requested_model: str
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
    estimated_total_cost_usd: float
    avg_metrics: Dict[str, float]


@dataclass(frozen=True)
class MatrixLeaderboardRow:
    entry_name: str
    runtime_name: str
    eval_mode: str
    provider_name: str
    requested_model: str
    overall_passed: int
    overall_total: int
    overall_pass_rate: float
    contradiction_passed: int
    contradiction_total: int
    natural_passed: int
    natural_total: int
    natural_extraction_passed: int
    natural_extraction_total: int
    natural_behavior_passed: int
    natural_behavior_total: int
    parse_failures: int
    empty_answers: int
    errors: int
    skipped: int
    estimated_total_cost_usd: float
    avg_metrics: Dict[str, float]


def load_model_matrix_entries(path: str | Path) -> List[ModelMatrixEntry]:
    payload = json.loads(Path(path).read_text())
    if isinstance(payload, dict):
        raw_entries = payload.get("entries", [])
    elif isinstance(payload, list):
        raw_entries = payload
    else:
        raise ValueError("Matrix config must be a JSON object with an 'entries' array or a JSON array.")

    if not isinstance(raw_entries, list):
        raise ValueError("Matrix config 'entries' must be a JSON array.")

    entries = [ModelMatrixEntry.from_dict(item) for item in raw_entries if isinstance(item, dict)]
    return [entry for entry in entries if entry.enabled]


def _suite_runtime_config(entry: ModelMatrixEntry, suite_name: str):
    if suite_name == "contradiction":
        return default_model_eval_runtime_config(
            temperature=entry.temperature,
            max_output_tokens=entry.max_output_tokens,
        )
    if suite_name == "natural":
        return default_natural_eval_runtime_config(
            temperature=entry.temperature,
            max_output_tokens=entry.max_output_tokens,
        )
    raise ValueError(f"Unsupported suite name: {suite_name}")


def _build_suite_adapter(entry: ModelMatrixEntry, suite_name: str) -> LLMAdapter:
    if entry.mode == "heuristic":
        if suite_name == "contradiction":
            return HeuristicBrainLayerEvalAdapter()
        if suite_name == "natural":
            return HeuristicNaturalConversationAdapter()
        raise ValueError(f"Unsupported suite name: {suite_name}")

    return build_live_model_eval_adapter(
        provider_name=entry.provider_name,
        api_key_env=entry.api_key_env,
        base_url=entry.base_url,
        request_path=entry.request_path,
        timeout_seconds=entry.timeout_seconds,
        max_output_tokens_field=entry.max_output_tokens_field,
    )


def _convert_model_eval_result(entry: ModelMatrixEntry, result: ModelEvalResult) -> MatrixCaseResult:
    estimated_cost_usd = estimate_usage_cost_usd(
        result.usage_metrics,
        input_cost_per_1k_tokens=entry.input_cost_per_1k_tokens,
        output_cost_per_1k_tokens=entry.output_cost_per_1k_tokens,
        total_cost_per_1k_tokens=entry.total_cost_per_1k_tokens,
    )
    case_artifact = dict(result.case_artifact)
    case_artifact["matrix_entry"] = {
        "name": entry.name,
        "mode": entry.mode,
        "provider_name": entry.provider_name,
        "requested_model": entry.requested_model,
        "estimated_cost_usd": estimated_cost_usd,
    }
    return MatrixCaseResult(
        entry_name=entry.name,
        suite_name="contradiction",
        runtime_name=result.runtime_name,
        scenario_slug=result.scenario_slug,
        checkpoint=result.checkpoint,
        case_label=f"{result.scenario_slug}/{result.checkpoint}",
        evaluation_type="behavior",
        target_layer="",
        target_key="",
        expected=result.expected,
        actual=result.actual,
        passed=result.passed,
        score=result.score,
        score_method=result.score_method,
        score_reason=result.score_reason,
        eval_mode=result.eval_mode,
        provider_name=result.provider_name,
        requested_model=result.requested_model,
        response_model=result.response_model,
        finish_reason=result.finish_reason,
        latency_ms=result.latency_ms,
        used_json=result.used_json,
        parse_failure=result.parse_failure,
        empty_answer=result.empty_answer,
        applied_observation_count=result.applied_observation_count,
        estimated_cost_usd=estimated_cost_usd,
        error=result.error,
        skipped=result.skipped,
        retrieved_layers=list(result.retrieved_layers),
        case_artifact=case_artifact,
        exported_state=dict(result.exported_state),
        state_metrics=dict(result.state_metrics),
        usage_metrics=dict(result.usage_metrics),
    )


def _convert_natural_eval_result(entry: ModelMatrixEntry, result: NaturalEvalResult) -> MatrixCaseResult:
    estimated_cost_usd = estimate_usage_cost_usd(
        result.usage_metrics,
        input_cost_per_1k_tokens=entry.input_cost_per_1k_tokens,
        output_cost_per_1k_tokens=entry.output_cost_per_1k_tokens,
        total_cost_per_1k_tokens=entry.total_cost_per_1k_tokens,
    )
    case_artifact = dict(result.case_artifact)
    case_artifact["matrix_entry"] = {
        "name": entry.name,
        "mode": entry.mode,
        "provider_name": entry.provider_name,
        "requested_model": entry.requested_model,
        "estimated_cost_usd": estimated_cost_usd,
    }
    return MatrixCaseResult(
        entry_name=entry.name,
        suite_name="natural",
        runtime_name=result.runtime_name,
        scenario_slug=result.scenario_slug,
        checkpoint=result.checkpoint,
        case_label=f"{result.scenario_slug}/{result.checkpoint}",
        evaluation_type=result.evaluation_type,
        target_layer=result.target_layer,
        target_key=result.target_key,
        expected=result.expected,
        actual=result.actual,
        passed=result.passed,
        score=result.score,
        score_method=result.score_method,
        score_reason=result.score_reason,
        eval_mode=result.eval_mode,
        provider_name=result.provider_name,
        requested_model=result.requested_model,
        response_model=result.response_model,
        finish_reason=result.finish_reason,
        latency_ms=result.latency_ms,
        used_json=result.used_json,
        parse_failure=result.parse_failure,
        empty_answer=result.empty_answer,
        applied_observation_count=result.applied_observation_count,
        estimated_cost_usd=estimated_cost_usd,
        error=result.error,
        skipped=result.skipped,
        retrieved_layers=list(result.retrieved_layers),
        case_artifact=case_artifact,
        exported_state=dict(result.exported_state),
        state_metrics=dict(result.state_metrics),
        usage_metrics=dict(result.usage_metrics),
    )


def _case_metrics(result: MatrixCaseResult) -> Dict[str, float]:
    metrics = dict(result.state_metrics)
    metrics["score"] = result.score
    metrics["estimated_cost_usd"] = result.estimated_cost_usd
    metrics["latency_ms"] = result.latency_ms
    metrics["applied_observation_count"] = float(result.applied_observation_count)
    for key, value in result.usage_metrics.items():
        metrics[f"usage_{key}"] = value
    return metrics


def run_model_matrix(
    entries: Sequence[ModelMatrixEntry],
    *,
    scenario_pack: str = DEFAULT_MODEL_SCENARIO_PACK,
    include_ablations: bool = False,
    suites: Sequence[str] = SUITE_NAMES,
    adapter_overrides: Mapping[tuple[str, str], LLMAdapter] | None = None,
    behavior_scoring_mode: str = "judge",
    runtime_profile: str = RUNTIME_PROFILE_DEFAULT,
) -> List[MatrixCaseResult]:
    results: List[MatrixCaseResult] = []
    overrides = dict(adapter_overrides or {})

    for entry in entries:
        for suite_name in suites:
            adapter = overrides.get((entry.name, suite_name)) or _build_suite_adapter(entry, suite_name)
            runtime_config = _suite_runtime_config(entry, suite_name)
            if suite_name == "contradiction":
                suite_results = run_model_eval_suite(
                    scenario_pack=scenario_pack,
                    include_ablations=include_ablations,
                    adapter=adapter,
                    eval_mode=entry.mode,
                    provider_name=entry.provider_name,
                    requested_model=entry.requested_model,
                    runtime_config=runtime_config,
                    behavior_scoring_mode=behavior_scoring_mode,
                    runtime_profile=runtime_profile,
                )
                results.extend(
                    _convert_model_eval_result(entry, result) for result in suite_results
                )
                continue

            if suite_name == "natural":
                suite_results = run_natural_eval_suite(
                    scenario_pack=scenario_pack,
                    include_ablations=include_ablations,
                    adapter=adapter,
                    eval_mode=entry.mode,
                    provider_name=entry.provider_name,
                    requested_model=entry.requested_model,
                    runtime_config=runtime_config,
                    behavior_scoring_mode=behavior_scoring_mode,
                    runtime_profile=runtime_profile,
                )
                results.extend(
                    _convert_natural_eval_result(entry, result) for result in suite_results
                )
                continue

            raise ValueError(f"Unsupported suite name: {suite_name}")

    return results


def summarize_matrix_results_by_suite(
    results: Sequence[MatrixCaseResult],
) -> List[MatrixSuiteSummary]:
    groups: Dict[tuple[str, str, str, str, str, str], Dict[str, object]] = {}

    for result in results:
        key = (
            result.entry_name,
            result.suite_name,
            result.runtime_name,
            result.eval_mode,
            result.provider_name,
            result.requested_model,
        )
        group = groups.setdefault(
            key,
            {
                "passed": 0,
                "total": 0,
                "extraction_passed": 0,
                "extraction_total": 0,
                "behavior_passed": 0,
                "behavior_total": 0,
                "parse_failures": 0,
                "empty_answers": 0,
                "errors": 0,
                "skipped": 0,
                "estimated_total_cost_usd": 0.0,
                "metric_totals": {},
            },
        )
        group["passed"] = int(group["passed"]) + int(result.passed)
        group["total"] = int(group["total"]) + 1
        if result.evaluation_type == "extraction":
            group["extraction_passed"] = int(group["extraction_passed"]) + int(result.passed)
            group["extraction_total"] = int(group["extraction_total"]) + 1
        else:
            group["behavior_passed"] = int(group["behavior_passed"]) + int(result.passed)
            group["behavior_total"] = int(group["behavior_total"]) + 1
        group["parse_failures"] = int(group["parse_failures"]) + int(result.parse_failure)
        group["empty_answers"] = int(group["empty_answers"]) + int(result.empty_answer)
        group["errors"] = int(group["errors"]) + int(bool(result.error))
        group["skipped"] = int(group["skipped"]) + int(result.skipped)
        group["estimated_total_cost_usd"] = float(group["estimated_total_cost_usd"]) + result.estimated_cost_usd

        metric_totals = group["metric_totals"]
        assert isinstance(metric_totals, dict)
        for metric_key, metric_value in _case_metrics(result).items():
            metric_totals[metric_key] = float(metric_totals.get(metric_key, 0.0)) + metric_value

    summaries: List[MatrixSuiteSummary] = []
    for key in sorted(groups):
        entry_name, suite_name, runtime_name, eval_mode, provider_name, requested_model = key
        group = groups[key]
        total = int(group["total"])
        metric_totals = group["metric_totals"]
        assert isinstance(metric_totals, dict)
        avg_metrics = {
            metric_key: float(metric_value) / total for metric_key, metric_value in metric_totals.items()
        }
        summaries.append(
            MatrixSuiteSummary(
                entry_name=entry_name,
                suite_name=suite_name,
                runtime_name=runtime_name,
                eval_mode=eval_mode,
                provider_name=provider_name,
                requested_model=requested_model,
                passed=int(group["passed"]),
                total=total,
                pass_rate=(int(group["passed"]) / total) if total else 0.0,
                extraction_passed=int(group["extraction_passed"]),
                extraction_total=int(group["extraction_total"]),
                behavior_passed=int(group["behavior_passed"]),
                behavior_total=int(group["behavior_total"]),
                parse_failures=int(group["parse_failures"]),
                empty_answers=int(group["empty_answers"]),
                errors=int(group["errors"]),
                skipped=int(group["skipped"]),
                estimated_total_cost_usd=float(group["estimated_total_cost_usd"]),
                avg_metrics=avg_metrics,
            )
        )
    return summaries


def build_matrix_leaderboard(results: Sequence[MatrixCaseResult]) -> List[MatrixLeaderboardRow]:
    suite_summaries = summarize_matrix_results_by_suite(results)
    by_entry_runtime: Dict[tuple[str, str, str, str, str], Dict[str, object]] = {}

    for summary in suite_summaries:
        key = (
            summary.entry_name,
            summary.runtime_name,
            summary.eval_mode,
            summary.provider_name,
            summary.requested_model,
        )
        row = by_entry_runtime.setdefault(
            key,
            {
                "overall_passed": 0,
                "overall_total": 0,
                "contradiction_passed": 0,
                "contradiction_total": 0,
                "natural_passed": 0,
                "natural_total": 0,
                "natural_extraction_passed": 0,
                "natural_extraction_total": 0,
                "natural_behavior_passed": 0,
                "natural_behavior_total": 0,
                "parse_failures": 0,
                "empty_answers": 0,
                "errors": 0,
                "skipped": 0,
                "estimated_total_cost_usd": 0.0,
                "weighted_metric_totals": {},
            },
        )
        row["overall_passed"] = int(row["overall_passed"]) + summary.passed
        row["overall_total"] = int(row["overall_total"]) + summary.total
        row["parse_failures"] = int(row["parse_failures"]) + summary.parse_failures
        row["empty_answers"] = int(row["empty_answers"]) + summary.empty_answers
        row["errors"] = int(row["errors"]) + summary.errors
        row["skipped"] = int(row["skipped"]) + summary.skipped
        row["estimated_total_cost_usd"] = float(row["estimated_total_cost_usd"]) + summary.estimated_total_cost_usd
        if summary.suite_name == "contradiction":
            row["contradiction_passed"] = summary.passed
            row["contradiction_total"] = summary.total
        elif summary.suite_name == "natural":
            row["natural_passed"] = summary.passed
            row["natural_total"] = summary.total
            row["natural_extraction_passed"] = summary.extraction_passed
            row["natural_extraction_total"] = summary.extraction_total
            row["natural_behavior_passed"] = summary.behavior_passed
            row["natural_behavior_total"] = summary.behavior_total

        weighted_metric_totals = row["weighted_metric_totals"]
        assert isinstance(weighted_metric_totals, dict)
        for metric_key, metric_value in summary.avg_metrics.items():
            weighted_metric_totals[metric_key] = float(
                weighted_metric_totals.get(metric_key, 0.0)
            ) + (metric_value * summary.total)

    leaderboard: List[MatrixLeaderboardRow] = []
    for key in sorted(by_entry_runtime):
        entry_name, runtime_name, eval_mode, provider_name, requested_model = key
        row = by_entry_runtime[key]
        overall_total = int(row["overall_total"])
        weighted_metric_totals = row["weighted_metric_totals"]
        assert isinstance(weighted_metric_totals, dict)
        avg_metrics = {
            metric_key: float(metric_value) / overall_total
            for metric_key, metric_value in weighted_metric_totals.items()
        }
        overall_passed = int(row["overall_passed"])
        leaderboard.append(
            MatrixLeaderboardRow(
                entry_name=entry_name,
                runtime_name=runtime_name,
                eval_mode=eval_mode,
                provider_name=provider_name,
                requested_model=requested_model,
                overall_passed=overall_passed,
                overall_total=overall_total,
                overall_pass_rate=(overall_passed / overall_total) if overall_total else 0.0,
                contradiction_passed=int(row["contradiction_passed"]),
                contradiction_total=int(row["contradiction_total"]),
                natural_passed=int(row["natural_passed"]),
                natural_total=int(row["natural_total"]),
                natural_extraction_passed=int(row["natural_extraction_passed"]),
                natural_extraction_total=int(row["natural_extraction_total"]),
                natural_behavior_passed=int(row["natural_behavior_passed"]),
                natural_behavior_total=int(row["natural_behavior_total"]),
                parse_failures=int(row["parse_failures"]),
                empty_answers=int(row["empty_answers"]),
                errors=int(row["errors"]),
                skipped=int(row["skipped"]),
                estimated_total_cost_usd=float(row["estimated_total_cost_usd"]),
                avg_metrics=avg_metrics,
            )
        )

    leaderboard.sort(
        key=lambda row: (
            -row.overall_pass_rate,
            -row.overall_passed,
            row.avg_metrics.get("latency_ms", 0.0),
            row.entry_name,
            row.runtime_name,
        )
    )
    return leaderboard


def render_model_matrix_report(results: Sequence[MatrixCaseResult]) -> str:
    suite_summaries = summarize_matrix_results_by_suite(results)
    leaderboard = build_matrix_leaderboard(results)
    lines = [
        "BrainLayer Model Matrix Report",
        "==============================",
        "",
    ]

    lines.append(
        f"Entries: {len({result.entry_name for result in results})} | "
        f"Suites: {len({result.suite_name for result in results})} | "
        f"Rows: {len(results)}"
    )
    lines.append("")
    lines.append("Suite Summary")
    lines.append("-------------")
    for summary in suite_summaries:
        extras = [
            f"{summary.passed}/{summary.total}",
            f"avg_score={summary.avg_metrics.get('score', 0.0):.2f}",
            f"avg_latency_ms={summary.avg_metrics.get('latency_ms', 0.0):.1f}",
            f"parse_failures={summary.parse_failures}",
            f"errors={summary.errors}",
        ]
        if summary.estimated_total_cost_usd:
            extras.append(f"est_cost_usd={summary.estimated_total_cost_usd:.4f}")
        if summary.suite_name == "natural":
            extras.insert(
                1,
                f"extraction={summary.extraction_passed}/{summary.extraction_total}",
            )
            extras.insert(
                2,
                f"behavior={summary.behavior_passed}/{summary.behavior_total}",
            )
        lines.append(
            f"{summary.entry_name} / {summary.runtime_name} / {summary.suite_name}: "
            + ", ".join(extras)
        )

    lines.append("")
    lines.append("Leaderboard")
    lines.append("-----------")
    for row in leaderboard:
        extras = [
            f"overall={row.overall_passed}/{row.overall_total}",
            f"contradiction={row.contradiction_passed}/{row.contradiction_total}",
            f"natural={row.natural_passed}/{row.natural_total}",
            (
                f"natural_extraction={row.natural_extraction_passed}/{row.natural_extraction_total}"
            ),
            f"natural_behavior={row.natural_behavior_passed}/{row.natural_behavior_total}",
            f"avg_score={row.avg_metrics.get('score', 0.0):.2f}",
            f"avg_latency_ms={row.avg_metrics.get('latency_ms', 0.0):.1f}",
            f"errors={row.errors}",
        ]
        if row.estimated_total_cost_usd:
            extras.append(f"est_cost_usd={row.estimated_total_cost_usd:.4f}")
        avg_total_tokens = row.avg_metrics.get("usage_total_tokens", 0.0)
        if avg_total_tokens:
            extras.append(f"avg_total_tokens={avg_total_tokens:.1f}")
        lines.append(f"{row.entry_name} / {row.runtime_name}: " + ", ".join(extras))

    return "\n".join(lines)


def serializable_matrix_case_result(
    result: MatrixCaseResult,
    *,
    artifact_path: str = "",
) -> Dict[str, object]:
    payload: Dict[str, object] = {
        "entry_name": result.entry_name,
        "suite_name": result.suite_name,
        "runtime_name": result.runtime_name,
        "scenario_slug": result.scenario_slug,
        "checkpoint": result.checkpoint,
        "case_label": result.case_label,
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
        "estimated_cost_usd": result.estimated_cost_usd,
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


def serializable_matrix_suite_summary(summary: MatrixSuiteSummary) -> Dict[str, object]:
    payload: Dict[str, object] = {
        "entry_name": summary.entry_name,
        "suite_name": summary.suite_name,
        "runtime_name": summary.runtime_name,
        "eval_mode": summary.eval_mode,
        "provider_name": summary.provider_name,
        "requested_model": summary.requested_model,
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
        "estimated_total_cost_usd": summary.estimated_total_cost_usd,
    }
    for key, value in sorted(summary.avg_metrics.items()):
        payload[f"avg_{key}"] = value
    return payload


def serializable_matrix_leaderboard_row(row: MatrixLeaderboardRow) -> Dict[str, object]:
    payload: Dict[str, object] = {
        "entry_name": row.entry_name,
        "runtime_name": row.runtime_name,
        "eval_mode": row.eval_mode,
        "provider_name": row.provider_name,
        "requested_model": row.requested_model,
        "overall_passed": row.overall_passed,
        "overall_total": row.overall_total,
        "overall_pass_rate": row.overall_pass_rate,
        "contradiction_passed": row.contradiction_passed,
        "contradiction_total": row.contradiction_total,
        "natural_passed": row.natural_passed,
        "natural_total": row.natural_total,
        "natural_extraction_passed": row.natural_extraction_passed,
        "natural_extraction_total": row.natural_extraction_total,
        "natural_behavior_passed": row.natural_behavior_passed,
        "natural_behavior_total": row.natural_behavior_total,
        "parse_failures": row.parse_failures,
        "empty_answers": row.empty_answers,
        "errors": row.errors,
        "skipped": row.skipped,
        "estimated_total_cost_usd": row.estimated_total_cost_usd,
    }
    for key, value in sorted(row.avg_metrics.items()):
        payload[f"avg_{key}"] = value
    return payload


def build_model_matrix_metadata(
    results: Sequence[MatrixCaseResult],
    *,
    scenario_pack: str = DEFAULT_MODEL_SCENARIO_PACK,
    include_ablations: bool,
    label: str | None,
    runtime_profile: str = RUNTIME_PROFILE_DEFAULT,
) -> Dict[str, object]:
    timestamp = utc_now_compact()
    run_id = timestamp if not label else f"{timestamp}-{slugify_label(label)}"
    return {
        "run_id": run_id,
        "generated_at_utc": utc_now_iso(),
        "git_commit": get_git_commit(),
        "scenario_pack": scenario_pack,
        "include_ablations": include_ablations,
        "runtime_profile": runtime_profile,
        "label": label or "",
        "entry_count": len({result.entry_name for result in results}),
        "suite_count": len({result.suite_name for result in results}),
        "runtime_count": len({(result.entry_name, result.runtime_name) for result in results}),
        "case_count": len(results),
        "artifacts_subdir": "case_artifacts",
        "score_methods": sorted({result.score_method for result in results}),
    }


def render_model_matrix_x_post(
    leaderboard: Sequence[MatrixLeaderboardRow],
    *,
    label: str | None,
) -> str:
    full_runtime_rows = [row for row in leaderboard if row.runtime_name == "model_loop"]
    if not full_runtime_rows:
        return "BrainLayer matrix run completed."

    prefix = "BrainLayer matrix"
    if label:
        prefix = f"BrainLayer matrix ({label})"

    top_rows = full_runtime_rows[:3]
    top_bits = [
        f"{row.entry_name} {row.overall_passed}/{row.overall_total}" for row in top_rows
    ]
    fastest = min(
        full_runtime_rows,
        key=lambda row: (
            row.avg_metrics.get("latency_ms", float("inf")),
            row.entry_name,
        ),
    )
    best_extraction = max(
        full_runtime_rows,
        key=lambda row: (
            row.natural_extraction_passed / max(1, row.natural_extraction_total),
            row.natural_extraction_passed,
            -row.avg_metrics.get("latency_ms", 0.0),
        ),
    )
    parts = [
        f"{prefix}: {len(full_runtime_rows)} configs across contradiction + natural suites. "
        f"Top full runtimes: {', '.join(top_bits)}. "
        f"Best natural extraction: {best_extraction.entry_name} "
        f"{best_extraction.natural_extraction_passed}/{best_extraction.natural_extraction_total}. "
        f"Fastest avg latency: {fastest.entry_name} "
        f"{fastest.avg_metrics.get('latency_ms', 0.0):.1f}ms."
    ]
    priced_rows = [row for row in full_runtime_rows if row.estimated_total_cost_usd > 0.0]
    if priced_rows:
        cheapest = min(
            priced_rows,
            key=lambda row: (row.estimated_total_cost_usd, row.entry_name),
        )
        parts.append(
            f"Cheapest estimated run cost: {cheapest.entry_name} ${cheapest.estimated_total_cost_usd:.4f}."
        )
    return " ".join(parts)


def export_model_matrix_results(
    results: Sequence[MatrixCaseResult],
    export_root: Path,
    *,
    scenario_pack: str = DEFAULT_MODEL_SCENARIO_PACK,
    include_ablations: bool,
    label: str | None = None,
    runtime_profile: str = RUNTIME_PROFILE_DEFAULT,
) -> Path:
    suite_summaries = summarize_matrix_results_by_suite(results)
    leaderboard = build_matrix_leaderboard(results)
    metadata = build_model_matrix_metadata(
        results,
        scenario_pack=scenario_pack,
        include_ablations=include_ablations,
        label=label,
        runtime_profile=runtime_profile,
    )
    run_dir = export_root / str(metadata["run_id"])
    run_dir.mkdir(parents=True, exist_ok=True)

    artifact_root = run_dir / str(metadata["artifacts_subdir"])
    result_rows = []
    for result in results:
        artifact_filename = (
            f"{result.entry_name}__{result.suite_name}__{result.runtime_name}"
            f"__{result.scenario_slug}__{result.checkpoint}.json"
        )
        artifact_path = write_case_artifact(artifact_root, artifact_filename, result.case_artifact)
        result_rows.append(serializable_matrix_case_result(result, artifact_path=artifact_path))
    summary_rows = [serializable_matrix_suite_summary(summary) for summary in suite_summaries]
    leaderboard_rows = [
        serializable_matrix_leaderboard_row(row) for row in leaderboard
    ]
    x_post = render_model_matrix_x_post(leaderboard, label=label)

    payload = {
        "metadata": metadata,
        "summary": summary_rows,
        "leaderboard": leaderboard_rows,
        "results": result_rows,
        "x_post": x_post,
    }

    (run_dir / "results.json").write_text(json.dumps(payload, indent=2) + "\n")
    write_csv(run_dir / "results.csv", result_rows)
    write_csv(run_dir / "summary.csv", summary_rows)
    write_csv(run_dir / "leaderboard.csv", leaderboard_rows)
    (run_dir / "x_post.txt").write_text(x_post + "\n")

    history_rows = []
    for row in leaderboard_rows:
        history_rows.append(
            {
                "run_id": metadata["run_id"],
                "generated_at_utc": metadata["generated_at_utc"],
                "git_commit": metadata["git_commit"],
                "label": metadata["label"],
                "scenario_pack": metadata["scenario_pack"],
                "include_ablations": metadata["include_ablations"],
                "runtime_profile": metadata["runtime_profile"],
                "score_methods": ",".join(metadata["score_methods"]),
                **row,
            }
        )

    append_csv(export_root / "matrix_history.csv", history_rows)
    with (export_root / "matrix_history.jsonl").open("a") as handle:
        handle.write(json.dumps(payload) + "\n")

    return run_dir


def dump_model_matrix_states(results: Sequence[MatrixCaseResult], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for result in results:
        filename = (
            f"{result.entry_name}__{result.suite_name}__{result.runtime_name}"
            f"__{result.scenario_slug}__{result.checkpoint}.json"
        )
        (output_dir / filename).write_text(json.dumps(result.exported_state, indent=2) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run both BrainLayer eval suites across a matrix of model/provider configs."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_MATRIX_CONFIG,
        help="Path to a JSON matrix config file.",
    )
    parser.add_argument(
        "--with-ablations",
        action="store_true",
        help="Include BrainLayer ablation variants for every matrix entry.",
    )
    parser.add_argument(
        "--scenario-pack",
        choices=("standard", "hard", "held_out", "external_dev", "external_held_out", "all"),
        default=DEFAULT_MODEL_SCENARIO_PACK,
        help="Choose the standard eval suites, the harder delayed/noisy suites, the held-out generalization suites, the external dev suites, the external held-out suites, or all packs together.",
    )
    parser.add_argument(
        "--runtime-profile",
        choices=(RUNTIME_PROFILE_DEFAULT, RUNTIME_PROFILE_STUDY_V2),
        default=RUNTIME_PROFILE_DEFAULT,
        help="Choose the default BrainLayer runtime set or the study-v2 stronger-baseline set.",
    )
    parser.add_argument(
        "--suite",
        choices=("all", "contradiction", "natural"),
        default="all",
        help="Run both eval suites or restrict the matrix to one suite.",
    )
    parser.add_argument(
        "--score-exact",
        action="store_true",
        help="Disable judge-backed semantic behavior scoring and require exact normalized matches.",
    )
    parser.add_argument(
        "--dump-states",
        type=Path,
        help="Optional directory for writing exported state snapshots for each case.",
    )
    parser.add_argument(
        "--export-results",
        type=Path,
        help="Write matrix CSV/JSON summaries into DIR and append history files there.",
    )
    parser.add_argument(
        "--label",
        help="Optional label to attach to exported matrix runs for later comparison.",
    )
    args = parser.parse_args(argv)

    entries = load_model_matrix_entries(args.config)
    suites = SUITE_NAMES if args.suite == "all" else (args.suite,)
    results = run_model_matrix(
        entries,
        scenario_pack=args.scenario_pack,
        include_ablations=args.with_ablations,
        suites=suites,
        behavior_scoring_mode="exact" if args.score_exact else "judge",
        runtime_profile=args.runtime_profile,
    )
    print(render_model_matrix_report(results))
    if args.dump_states:
        dump_model_matrix_states(results, args.dump_states)
        print("")
        print(f"Matrix state dumps written to {args.dump_states}")
    if args.export_results:
        run_dir = export_model_matrix_results(
            results,
            args.export_results,
            scenario_pack=args.scenario_pack,
            include_ablations=args.with_ablations,
            label=args.label,
            runtime_profile=args.runtime_profile,
        )
        print("")
        print(f"Matrix exports written to {run_dir}")
        print(f"X post saved to {run_dir / 'x_post.txt'}")
    return 0


__all__ = [
    "DEFAULT_MATRIX_CONFIG",
    "MatrixCaseResult",
    "MatrixLeaderboardRow",
    "MatrixSuiteSummary",
    "ModelMatrixEntry",
    "build_matrix_leaderboard",
    "export_model_matrix_results",
    "load_model_matrix_entries",
    "render_model_matrix_report",
    "render_model_matrix_x_post",
    "run_model_matrix",
    "summarize_matrix_results_by_suite",
]
