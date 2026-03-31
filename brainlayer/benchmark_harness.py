from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re
import subprocess
from typing import Dict, Iterable, List, Sequence

from .agents import (
    AnswerRecord,
    BaseAgent,
    BrainLayerAgent,
    BrainLayerFeatureConfig,
    ContextOnlyAgent,
    NaiveMemoryAgent,
)
from .scenarios import Observation, Query, SCENARIOS, Scenario
from .validation import validate_state_dict

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class ScenarioResult:
    scenario_slug: str
    checkpoint: str
    agent_name: str
    expected: str
    actual: str
    passed: bool
    evidence: str
    exported_state: Dict[str, object]
    state_metrics: Dict[str, float]


@dataclass(frozen=True)
class AgentSummary:
    agent_name: str
    passed: int
    total: int
    pass_rate: float
    avg_metrics: Dict[str, float]


def utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def slugify_label(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "run"


def get_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def format_case_label(scenario_slug: str, checkpoint: str) -> str:
    if checkpoint == "final":
        return scenario_slug
    return f"{scenario_slug}/{checkpoint}"


def normalize_answer(value: str) -> str:
    return " ".join(value.strip().lower().split())


def build_agents(include_ablations: bool = True) -> List[BaseAgent]:
    agents: List[BaseAgent] = [ContextOnlyAgent(), NaiveMemoryAgent(), BrainLayerAgent()]
    if not include_ablations:
        return agents

    agents.extend(
        [
            BrainLayerAgent(
                agent_name="brainlayer_no_consolidation",
                features=BrainLayerFeatureConfig(enable_consolidation=False),
            ),
            BrainLayerAgent(
                agent_name="brainlayer_no_forgetting",
                features=BrainLayerFeatureConfig(enable_forgetting=False),
            ),
            BrainLayerAgent(
                agent_name="brainlayer_no_autobio",
                features=BrainLayerFeatureConfig(enable_autobio=False),
            ),
            BrainLayerAgent(
                agent_name="brainlayer_no_working_state",
                features=BrainLayerFeatureConfig(enable_working_state=False),
            ),
        ]
    )
    return agents


def collect_state_metrics(agent_name: str, exported_state: Dict[str, object]) -> Dict[str, float]:
    if agent_name.startswith("brainlayer"):
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

    if agent_name == "naive_memory":
        notes = exported_state.get("notes", [])
        return {"total_records": float(len(notes)), "notes": float(len(notes))}

    current_observation = exported_state.get("current_observation")
    return {
        "total_records": 1.0 if current_observation else 0.0,
        "current_observation": 1.0 if current_observation else 0.0,
    }


def run_scenario(scenario: Scenario, agents: Sequence[BaseAgent]) -> List[ScenarioResult]:
    results: List[ScenarioResult] = []
    for agent in agents:
        agent.reset()

        for step in scenario.steps:
            if isinstance(step, Observation):
                agent.observe(scenario.slug, step)
                continue

            answer = agent.answer(step)
            exported_state = agent.export_state()
            results.append(
                ScenarioResult(
                    scenario_slug=scenario.slug,
                    checkpoint=step.checkpoint,
                    agent_name=agent.name,
                    expected=step.expected_answer,
                    actual=answer.answer,
                    passed=normalize_answer(answer.answer)
                    == normalize_answer(step.expected_answer),
                    evidence=answer.evidence,
                    exported_state=exported_state,
                    state_metrics=collect_state_metrics(agent.name, exported_state),
                )
            )
    if not results:
        raise ValueError(f"Scenario {scenario.slug} is missing a query step.")
    return results


def run_suite(
    scenarios: Iterable[Scenario] | None = None,
    *,
    include_ablations: bool = True,
) -> List[ScenarioResult]:
    active_scenarios = list(scenarios or SCENARIOS)
    all_results: List[ScenarioResult] = []
    for scenario in active_scenarios:
        all_results.extend(run_scenario(scenario, build_agents(include_ablations=include_ablations)))
    return all_results


def summarize_results(results: Sequence[ScenarioResult]) -> List[AgentSummary]:
    passed_counts: Dict[str, int] = {}
    totals: Dict[str, int] = {}
    metric_totals: Dict[str, Dict[str, float]] = {}

    for result in results:
        totals[result.agent_name] = totals.get(result.agent_name, 0) + 1
        passed_counts[result.agent_name] = passed_counts.get(result.agent_name, 0) + int(result.passed)
        agent_metric_totals = metric_totals.setdefault(result.agent_name, {})
        for key, value in result.state_metrics.items():
            agent_metric_totals[key] = agent_metric_totals.get(key, 0.0) + value

    summaries: List[AgentSummary] = []
    for agent_name in sorted(totals):
        total = totals[agent_name]
        avg_metrics = {
            key: value / total for key, value in metric_totals.get(agent_name, {}).items()
        }
        summaries.append(
            AgentSummary(
                agent_name=agent_name,
                passed=passed_counts[agent_name],
                total=total,
                pass_rate=passed_counts[agent_name] / total if total else 0.0,
                avg_metrics=avg_metrics,
            )
        )
    return summaries


def render_report(results: Sequence[ScenarioResult]) -> str:
    lines = [
        "BrainLayer Benchmark Report",
        "===========================",
        "",
    ]
    summaries = summarize_results(results)

    for result in results:
        status = "PASS" if result.passed else "FAIL"
        case_label = format_case_label(result.scenario_slug, result.checkpoint)
        lines.append(
            f"[{status}] {result.agent_name} on {case_label}: "
            f"expected={result.expected!r}, actual={result.actual!r}"
        )
        lines.append(f"        evidence: {result.evidence}")

    lines.append("")
    lines.append("Summary")
    lines.append("-------")
    for summary in summaries:
        avg_total_records = summary.avg_metrics.get("total_records", 0.0)
        extras = [f"avg_records={avg_total_records:.1f}"]
        if summary.agent_name.startswith("brainlayer"):
            avg_episodes = summary.avg_metrics.get("episodes", 0.0)
            avg_active_working = summary.avg_metrics.get("active_working_items", 0.0)
            extras.append(f"avg_episodes={avg_episodes:.1f}")
            extras.append(f"avg_active_working={avg_active_working:.1f}")
        lines.append(
            f"{summary.agent_name}: {summary.passed}/{summary.total} | " + ", ".join(extras)
        )
    return "\n".join(lines)


def dump_states(results: Sequence[ScenarioResult], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for result in results:
        if result.agent_name.startswith("brainlayer"):
            validate_state_dict(result.exported_state)
        filename = output_dir / f"{result.scenario_slug}.{result.checkpoint}.{result.agent_name}.json"
        filename.write_text(json.dumps(result.exported_state, indent=2) + "\n")


def serializable_result(result: ScenarioResult) -> Dict[str, object]:
    payload: Dict[str, object] = {
        "scenario_slug": result.scenario_slug,
        "checkpoint": result.checkpoint,
        "case_label": format_case_label(result.scenario_slug, result.checkpoint),
        "agent_name": result.agent_name,
        "expected": result.expected,
        "actual": result.actual,
        "passed": result.passed,
        "evidence": result.evidence,
    }
    for key, value in sorted(result.state_metrics.items()):
        payload[f"metric_{key}"] = value
    return payload


def serializable_summary(summary: AgentSummary) -> Dict[str, object]:
    payload: Dict[str, object] = {
        "agent_name": summary.agent_name,
        "passed": summary.passed,
        "total": summary.total,
        "pass_rate": summary.pass_rate,
    }
    for key, value in sorted(summary.avg_metrics.items()):
        payload[f"avg_{key}"] = value
    return payload


def build_run_metadata(
    results: Sequence[ScenarioResult],
    *,
    include_ablations: bool,
    label: str | None,
) -> Dict[str, object]:
    timestamp = utc_now_compact()
    run_id = timestamp if not label else f"{timestamp}-{slugify_label(label)}"
    return {
        "run_id": run_id,
        "generated_at_utc": utc_now_iso(),
        "git_commit": get_git_commit(),
        "include_ablations": include_ablations,
        "label": label or "",
        "scenario_count": len({result.scenario_slug for result in results}),
        "checkpoint_count": len({(result.scenario_slug, result.checkpoint) for result in results}),
        "agent_count": len({result.agent_name for result in results}),
    }


def render_x_post(
    summaries: Sequence[AgentSummary],
    *,
    include_ablations: bool,
    label: str | None,
) -> str:
    summary_by_name = {summary.agent_name: summary for summary in summaries}
    brainlayer = summary_by_name.get("brainlayer")
    naive_memory = summary_by_name.get("naive_memory")
    context_only = summary_by_name.get("context_only")
    if brainlayer is None or naive_memory is None or context_only is None:
        return "BrainLayer benchmark run completed."

    prefix = "BrainLayer eval"
    if label:
        prefix = f"BrainLayer eval ({label})"

    parts = [
        f"{prefix}: full model {brainlayer.passed}/{brainlayer.total}",
        f"vs naive memory {naive_memory.passed}/{naive_memory.total}",
        f"and context-only {context_only.passed}/{context_only.total}.",
    ]

    if include_ablations:
        no_consolidation = summary_by_name.get("brainlayer_no_consolidation")
        no_autobio = summary_by_name.get("brainlayer_no_autobio")
        no_working_state = summary_by_name.get("brainlayer_no_working_state")
        no_forgetting = summary_by_name.get("brainlayer_no_forgetting")
        if no_consolidation and no_autobio and no_working_state and no_forgetting:
            parts.append(
                "Ablations:"
                f" no_consolidation {no_consolidation.passed}/{no_consolidation.total},"
                f" no_autobio {no_autobio.passed}/{no_autobio.total},"
                f" no_working_state {no_working_state.passed}/{no_working_state.total}."
            )
            parts.append(
                "No-forgetting kept accuracy"
                f" ({no_forgetting.passed}/{no_forgetting.total})"
                f" but retained more state"
                f" ({no_forgetting.avg_metrics.get('total_records', 0.0):.1f}"
                f" vs {brainlayer.avg_metrics.get('total_records', 0.0):.1f} avg records)."
            )

    return " ".join(parts)


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def append_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    file_exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def export_results(
    results: Sequence[ScenarioResult],
    export_root: Path,
    *,
    include_ablations: bool,
    label: str | None = None,
) -> Path:
    summaries = summarize_results(results)
    metadata = build_run_metadata(
        results,
        include_ablations=include_ablations,
        label=label,
    )
    run_dir = export_root / str(metadata["run_id"])
    run_dir.mkdir(parents=True, exist_ok=True)

    result_rows = [serializable_result(result) for result in results]
    summary_rows = [serializable_summary(summary) for summary in summaries]
    x_post = render_x_post(summaries, include_ablations=include_ablations, label=label)

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
        history_row = {
            "run_id": metadata["run_id"],
            "generated_at_utc": metadata["generated_at_utc"],
            "git_commit": metadata["git_commit"],
            "label": metadata["label"],
            "include_ablations": metadata["include_ablations"],
            **row,
        }
        history_rows.append(history_row)

    append_csv(export_root / "history.csv", history_rows)
    with (export_root / "history.jsonl").open("a") as handle:
        handle.write(json.dumps(payload) + "\n")

    return run_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the minimal BrainLayer benchmark suite.")
    parser.add_argument(
        "--dump-states",
        metavar="DIR",
        help="Write exported agent states for each scenario into DIR.",
    )
    parser.add_argument(
        "--core-only",
        action="store_true",
        help="Run only context_only, naive_memory, and full brainlayer without ablation variants.",
    )
    parser.add_argument(
        "--export-results",
        metavar="DIR",
        help="Write per-run CSV/JSON summaries into DIR and append to history files there.",
    )
    parser.add_argument(
        "--label",
        help="Optional label to attach to exported benchmark runs for later comparison.",
    )
    args = parser.parse_args()

    include_ablations = not args.core_only
    results = run_suite(include_ablations=include_ablations)
    print(render_report(results))

    if args.dump_states:
        dump_states(results, Path(args.dump_states))
        print("")
        print(f"State dumps written to {args.dump_states}")

    if args.export_results:
        run_dir = export_results(
            results,
            Path(args.export_results),
            include_ablations=include_ablations,
            label=args.label,
        )
        print("")
        print(f"Benchmark exports written to {run_dir}")
        print(f"X post saved to {run_dir / 'x_post.txt'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
