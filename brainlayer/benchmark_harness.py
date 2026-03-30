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
    ContextOnlyAgent,
    NaiveMemoryAgent,
)
from .scenarios import Observation, Query, SCENARIOS, Scenario


@dataclass
class ScenarioResult:
    scenario_slug: str
    agent_name: str
    expected: str
    actual: str
    passed: bool
    evidence: str
    exported_state: Dict[str, object]


def normalize_answer(value: str) -> str:
    return " ".join(value.strip().lower().split())


def build_agents() -> List[BaseAgent]:
    return [ContextOnlyAgent(), NaiveMemoryAgent(), BrainLayerAgent()]


def run_scenario(scenario: Scenario, agents: Sequence[BaseAgent]) -> List[ScenarioResult]:
    results: List[ScenarioResult] = []
    for agent in agents:
        agent.reset()
        answer: AnswerRecord | None = None
        query_step: Query | None = None

        for step in scenario.steps:
            if isinstance(step, Observation):
                agent.observe(scenario.slug, step)
            else:
                query_step = step
                answer = agent.answer(step)

        if not answer or not query_step:
            raise ValueError(f"Scenario {scenario.slug} is missing a query step.")

        results.append(
            ScenarioResult(
                scenario_slug=scenario.slug,
                agent_name=agent.name,
                expected=query_step.expected_answer,
                actual=answer.answer,
                passed=normalize_answer(answer.answer)
                == normalize_answer(query_step.expected_answer),
                evidence=answer.evidence,
                exported_state=agent.export_state(),
            )
        )
    return results


def run_suite(scenarios: Iterable[Scenario] | None = None) -> List[ScenarioResult]:
    active_scenarios = list(scenarios or SCENARIOS)
    all_results: List[ScenarioResult] = []
    for scenario in active_scenarios:
        all_results.extend(run_scenario(scenario, build_agents()))
    return all_results


def render_report(results: Sequence[ScenarioResult]) -> str:
    lines = [
        "BrainLayer Benchmark Report",
        "===========================",
        "",
    ]
    summary: Dict[str, int] = {}
    totals: Dict[str, int] = {}

    for result in results:
        totals[result.agent_name] = totals.get(result.agent_name, 0) + 1
        summary[result.agent_name] = summary.get(result.agent_name, 0) + int(result.passed)
        status = "PASS" if result.passed else "FAIL"
        lines.append(
            f"[{status}] {result.agent_name} on {result.scenario_slug}: "
            f"expected={result.expected!r}, actual={result.actual!r}"
        )
        lines.append(f"        evidence: {result.evidence}")

    lines.append("")
    lines.append("Summary")
    lines.append("-------")
    for agent_name in sorted(summary):
        lines.append(f"{agent_name}: {summary[agent_name]}/{totals[agent_name]}")
    return "\n".join(lines)


def dump_states(results: Sequence[ScenarioResult], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for result in results:
        filename = output_dir / f"{result.scenario_slug}.{result.agent_name}.json"
        filename.write_text(json.dumps(result.exported_state, indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the minimal BrainLayer benchmark suite.")
    parser.add_argument(
        "--dump-states",
        metavar="DIR",
        help="Write exported agent states for each scenario into DIR.",
    )
    args = parser.parse_args()

    results = run_suite()
    print(render_report(results))

    if args.dump_states:
        dump_states(results, Path(args.dump_states))
        print("")
        print(f"State dumps written to {args.dump_states}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
