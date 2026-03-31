from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
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


def render_report(results: Sequence[ScenarioResult]) -> str:
    lines = [
        "BrainLayer Benchmark Report",
        "===========================",
        "",
    ]
    summary: Dict[str, int] = {}
    totals: Dict[str, int] = {}
    metric_totals: Dict[str, Dict[str, float]] = {}

    for result in results:
        totals[result.agent_name] = totals.get(result.agent_name, 0) + 1
        summary[result.agent_name] = summary.get(result.agent_name, 0) + int(result.passed)
        agent_metric_totals = metric_totals.setdefault(result.agent_name, {})
        for key, value in result.state_metrics.items():
            agent_metric_totals[key] = agent_metric_totals.get(key, 0.0) + value
        status = "PASS" if result.passed else "FAIL"
        case_label = result.scenario_slug
        if result.checkpoint != "final":
            case_label = f"{case_label}/{result.checkpoint}"
        lines.append(
            f"[{status}] {result.agent_name} on {case_label}: "
            f"expected={result.expected!r}, actual={result.actual!r}"
        )
        lines.append(f"        evidence: {result.evidence}")

    lines.append("")
    lines.append("Summary")
    lines.append("-------")
    for agent_name in sorted(summary):
        avg_total_records = metric_totals[agent_name].get("total_records", 0.0) / totals[agent_name]
        extras = [f"avg_records={avg_total_records:.1f}"]
        if agent_name.startswith("brainlayer"):
            avg_episodes = metric_totals[agent_name].get("episodes", 0.0) / totals[agent_name]
            avg_active_working = (
                metric_totals[agent_name].get("active_working_items", 0.0) / totals[agent_name]
            )
            extras.append(f"avg_episodes={avg_episodes:.1f}")
            extras.append(f"avg_active_working={avg_active_working:.1f}")
        lines.append(
            f"{agent_name}: {summary[agent_name]}/{totals[agent_name]} | " + ", ".join(extras)
        )
    return "\n".join(lines)


def dump_states(results: Sequence[ScenarioResult], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for result in results:
        if result.agent_name.startswith("brainlayer"):
            validate_state_dict(result.exported_state)
        filename = output_dir / f"{result.scenario_slug}.{result.checkpoint}.{result.agent_name}.json"
        filename.write_text(json.dumps(result.exported_state, indent=2) + "\n")


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
    args = parser.parse_args()

    results = run_suite(include_ablations=not args.core_only)
    print(render_report(results))

    if args.dump_states:
        dump_states(results, Path(args.dump_states))
        print("")
        print(f"State dumps written to {args.dump_states}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
