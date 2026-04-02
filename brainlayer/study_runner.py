from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

from .benchmark_harness import get_git_commit, slugify_label, utc_now_compact, utc_now_iso, write_csv
from .matrix_analysis import (
    build_cost_quality_frontier,
    build_matrix_analysis_highlights,
    export_matrix_analysis,
)
from .model_matrix import SUITE_NAMES, export_model_matrix_results, load_model_matrix_entries, run_model_matrix
from .runtime_variants import RUNTIME_PROFILE_DEFAULT, RUNTIME_PROFILE_STUDY_V2


DEFAULT_STUDY_CONFIG = Path("examples/model_matrix.openai.chat.live.json")
DEFAULT_STUDY_PROTOCOL = Path("docs/study_protocol.md")
DEFAULT_STUDY_EXPORT_ROOT = Path("artifacts/study_runs")
DEFAULT_STUDY_SCENARIO_PACKS = ("standard", "hard", "held_out")
VALID_STUDY_SCENARIO_PACKS = set(DEFAULT_STUDY_SCENARIO_PACKS)


def parse_study_scenario_packs(raw: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(raw, str):
        parts = [part.strip() for part in raw.split(",") if part.strip()]
    else:
        parts = [str(part).strip() for part in raw if str(part).strip()]

    if not parts or parts == ["all"]:
        return DEFAULT_STUDY_SCENARIO_PACKS

    resolved: List[str] = []
    for part in parts:
        if part == "all":
            for default_pack in DEFAULT_STUDY_SCENARIO_PACKS:
                if default_pack not in resolved:
                    resolved.append(default_pack)
            continue
        if part not in VALID_STUDY_SCENARIO_PACKS:
            raise ValueError(
                "Unsupported study scenario pack: "
                f"{part}. Expected one of {sorted(VALID_STUDY_SCENARIO_PACKS)} or 'all'."
            )
        if part not in resolved:
            resolved.append(part)
    return tuple(resolved)


def run_study(
    *,
    config_path: str | Path = DEFAULT_STUDY_CONFIG,
    protocol_path: str | Path = DEFAULT_STUDY_PROTOCOL,
    export_root: str | Path = DEFAULT_STUDY_EXPORT_ROOT,
    scenario_packs: Sequence[str] = DEFAULT_STUDY_SCENARIO_PACKS,
    include_ablations: bool = False,
    suites: Sequence[str] = SUITE_NAMES,
    behavior_scoring_mode: str = "judge",
    label: str | None = None,
    runtime_profile: str = RUNTIME_PROFILE_DEFAULT,
) -> Path:
    config_path = Path(config_path)
    protocol_path = Path(protocol_path)
    export_root = Path(export_root)

    if not config_path.exists():
        raise FileNotFoundError(f"Study config not found: {config_path}")
    if not protocol_path.exists():
        raise FileNotFoundError(f"Study protocol not found: {protocol_path}")

    packs = parse_study_scenario_packs(scenario_packs)
    entries = load_model_matrix_entries(config_path)
    if not entries:
        raise ValueError("Study config did not produce any enabled matrix entries.")

    suite_names = tuple(suites) if suites else SUITE_NAMES
    study_id = _study_id(label)
    study_dir = export_root / study_id
    study_dir.mkdir(parents=True, exist_ok=True)

    protocol_snapshot_path = study_dir / "study_protocol.md"
    protocol_snapshot_path.write_text(protocol_path.read_text())
    config_snapshot_path = study_dir / "study_config.json"
    config_snapshot_path.write_text(config_path.read_text())

    matrix_root = study_dir / "matrix_runs"
    analysis_root = study_dir / "matrix_analysis"

    pack_exports: List[Dict[str, object]] = []
    pack_summaries: List[Dict[str, object]] = []
    for pack in packs:
        pack_label = _pack_label(label, pack)
        results = run_model_matrix(
            entries,
            scenario_pack=pack,
            include_ablations=include_ablations,
            suites=suite_names,
            behavior_scoring_mode=behavior_scoring_mode,
            runtime_profile=runtime_profile,
        )
        run_dir = export_model_matrix_results(
            results,
            matrix_root,
            scenario_pack=pack,
            include_ablations=include_ablations,
            label=pack_label,
            runtime_profile=runtime_profile,
        )
        run_payload = json.loads((run_dir / "results.json").read_text())
        run_id = str(run_payload.get("metadata", {}).get("run_id", ""))
        analysis_dir = export_matrix_analysis(
            matrix_root / "matrix_history.jsonl",
            analysis_root,
            run_id=run_id,
            label=pack_label,
        )
        analysis_payload = json.loads((analysis_dir / "report.json").read_text())
        leaderboard_rows = [
            dict(row)
            for row in run_payload.get("leaderboard", [])
            if isinstance(row, dict)
        ]
        top_row = leaderboard_rows[0] if leaderboard_rows else {}
        pack_exports.append(
            {
                "scenario_pack": pack,
                "run_payload": run_payload,
                "analysis_payload": analysis_payload,
                "leaderboard": leaderboard_rows,
                "run_dir": str(run_dir),
                "analysis_dir": str(analysis_dir),
            }
        )
        pack_summaries.append(
            {
                "scenario_pack": pack,
                "run_id": run_id,
                "label": str(run_payload.get("metadata", {}).get("label", "")),
                "generated_at_utc": str(run_payload.get("metadata", {}).get("generated_at_utc", "")),
                "case_count": int(run_payload.get("metadata", {}).get("case_count", 0) or 0),
                "matrix_run_dir": str(run_dir.relative_to(study_dir)),
                "analysis_dir": str(analysis_dir.relative_to(study_dir)),
                "top_entry_name": str(top_row.get("entry_name", "")),
                "top_requested_model": str(top_row.get("requested_model", "")),
                "top_overall_passed": int(top_row.get("overall_passed", 0) or 0),
                "top_overall_total": int(top_row.get("overall_total", 0) or 0),
                "top_overall_pass_rate": float(top_row.get("overall_pass_rate", 0.0) or 0.0),
                "top_estimated_total_cost_usd": float(
                    top_row.get("estimated_total_cost_usd", 0.0) or 0.0
                ),
                "top_avg_latency_ms": float(top_row.get("avg_latency_ms", 0.0) or 0.0),
                "x_post": str(run_payload.get("x_post", "")),
                "analysis_x_post": str(analysis_payload.get("x_post", "")),
            }
        )

    aggregate_leaderboard = build_study_aggregate_leaderboard(pack_exports)
    highlights = build_matrix_analysis_highlights(aggregate_leaderboard)
    pareto_frontier = build_cost_quality_frontier(aggregate_leaderboard)
    x_post = render_study_x_post(
        aggregate_leaderboard,
        pack_summaries=pack_summaries,
        label=label,
    )

    study_metadata = {
        "study_id": study_id,
        "generated_at_utc": utc_now_iso(),
        "git_commit": get_git_commit(),
        "label": label or "",
        "config_path": str(config_path),
        "protocol_path": str(protocol_path),
        "config_snapshot_path": config_snapshot_path.name,
        "protocol_snapshot_path": protocol_snapshot_path.name,
        "scenario_packs": list(packs),
        "include_ablations": include_ablations,
        "suites": list(suite_names),
        "behavior_scoring_mode": behavior_scoring_mode,
        "runtime_profile": runtime_profile,
        "entry_count": len(entries),
    }

    payload = {
        "study_metadata": study_metadata,
        "pack_runs": pack_summaries,
        "aggregate_leaderboard": aggregate_leaderboard,
        "highlights": highlights,
        "pareto_frontier": pareto_frontier,
        "x_post": x_post,
    }

    (study_dir / "study_summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    (study_dir / "study_summary.md").write_text(render_study_summary_markdown(payload))
    write_csv(study_dir / "pack_summary.csv", pack_summaries)
    write_csv(study_dir / "aggregate_leaderboard.csv", aggregate_leaderboard)
    write_csv(study_dir / "pareto_frontier.csv", pareto_frontier)
    (study_dir / "x_post.txt").write_text(x_post + "\n")

    return study_dir


def build_study_aggregate_leaderboard(
    pack_exports: Sequence[Mapping[str, object]],
) -> List[Dict[str, object]]:
    groups: Dict[tuple[str, str, str, str, str], Dict[str, object]] = {}

    for pack_export in pack_exports:
        scenario_pack = str(pack_export.get("scenario_pack", ""))
        leaderboard_rows = pack_export.get("leaderboard", [])
        if not isinstance(leaderboard_rows, list):
            continue
        for row in leaderboard_rows:
            if not isinstance(row, dict):
                continue
            key = (
                str(row.get("entry_name", "")),
                str(row.get("runtime_name", "")),
                str(row.get("eval_mode", "")),
                str(row.get("provider_name", "")),
                str(row.get("requested_model", "")),
            )
            group = groups.setdefault(
                key,
                {
                    "packs_covered": set(),
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
                    "metric_totals": {},
                },
            )
            overall_total = int(row.get("overall_total", 0) or 0)
            group["packs_covered"].add(scenario_pack)
            for field in (
                "overall_passed",
                "overall_total",
                "contradiction_passed",
                "contradiction_total",
                "natural_passed",
                "natural_total",
                "natural_extraction_passed",
                "natural_extraction_total",
                "natural_behavior_passed",
                "natural_behavior_total",
                "parse_failures",
                "empty_answers",
                "errors",
                "skipped",
            ):
                group[field] = int(group[field]) + int(row.get(field, 0) or 0)
            group["estimated_total_cost_usd"] = float(group["estimated_total_cost_usd"]) + float(
                row.get("estimated_total_cost_usd", 0.0) or 0.0
            )

            metric_totals = group["metric_totals"]
            assert isinstance(metric_totals, dict)
            for metric_key, metric_value in row.items():
                if not metric_key.startswith("avg_"):
                    continue
                metric_totals[metric_key] = float(metric_totals.get(metric_key, 0.0)) + (
                    float(metric_value or 0.0) * overall_total
                )

    aggregate_rows: List[Dict[str, object]] = []
    for key in sorted(groups):
        entry_name, runtime_name, eval_mode, provider_name, requested_model = key
        group = groups[key]
        overall_total = int(group["overall_total"])
        metric_totals = group["metric_totals"]
        assert isinstance(metric_totals, dict)
        row: Dict[str, object] = {
            "entry_name": entry_name,
            "runtime_name": runtime_name,
            "eval_mode": eval_mode,
            "provider_name": provider_name,
            "requested_model": requested_model,
            "packs_covered": ",".join(sorted(group["packs_covered"])),
            "overall_passed": int(group["overall_passed"]),
            "overall_total": overall_total,
            "overall_pass_rate": (int(group["overall_passed"]) / overall_total) if overall_total else 0.0,
            "contradiction_passed": int(group["contradiction_passed"]),
            "contradiction_total": int(group["contradiction_total"]),
            "natural_passed": int(group["natural_passed"]),
            "natural_total": int(group["natural_total"]),
            "natural_extraction_passed": int(group["natural_extraction_passed"]),
            "natural_extraction_total": int(group["natural_extraction_total"]),
            "natural_behavior_passed": int(group["natural_behavior_passed"]),
            "natural_behavior_total": int(group["natural_behavior_total"]),
            "parse_failures": int(group["parse_failures"]),
            "empty_answers": int(group["empty_answers"]),
            "errors": int(group["errors"]),
            "skipped": int(group["skipped"]),
            "estimated_total_cost_usd": float(group["estimated_total_cost_usd"]),
        }
        for metric_key, total_value in sorted(metric_totals.items()):
            row[metric_key] = (float(total_value) / overall_total) if overall_total else 0.0
        aggregate_rows.append(row)

    aggregate_rows.sort(
        key=lambda row: (
            -float(row.get("overall_pass_rate", 0.0) or 0.0),
            -float(row.get("avg_score", 0.0) or 0.0),
            float(row.get("estimated_total_cost_usd", 0.0) or 0.0),
            float(row.get("avg_latency_ms", 0.0) or 0.0),
            str(row.get("entry_name", "")),
        )
    )
    return aggregate_rows


def render_study_summary_markdown(payload: Mapping[str, object]) -> str:
    metadata = payload.get("study_metadata", {})
    pack_runs = payload.get("pack_runs", [])
    leaderboard = payload.get("aggregate_leaderboard", [])
    highlights = payload.get("highlights", {})
    frontier = payload.get("pareto_frontier", [])

    if not isinstance(metadata, dict):
        metadata = {}
    if not isinstance(pack_runs, list):
        pack_runs = []
    if not isinstance(leaderboard, list):
        leaderboard = []
    if not isinstance(highlights, dict):
        highlights = {}
    if not isinstance(frontier, list):
        frontier = []

    lines = [
        "# BrainLayer Study Summary",
        "",
        f"- Study: `{metadata.get('study_id', '')}`",
        f"- Generated: `{metadata.get('generated_at_utc', '')}`",
        f"- Commit: `{metadata.get('git_commit', '')}`",
        f"- Config snapshot: `{metadata.get('config_snapshot_path', '')}`",
        f"- Protocol snapshot: `{metadata.get('protocol_snapshot_path', '')}`",
        f"- Packs: `{', '.join(metadata.get('scenario_packs', []))}`",
        f"- Suites: `{', '.join(metadata.get('suites', []))}`",
        f"- Ablations: `{metadata.get('include_ablations', False)}`",
        f"- Behavior scoring: `{metadata.get('behavior_scoring_mode', '')}`",
        "",
    ]

    if highlights:
        lines.extend(["## Highlights", ""])
        for label, row in (
            ("Top accuracy", highlights.get("top_accuracy", {})),
            ("Best value", highlights.get("best_value", {})),
            ("Cheapest", highlights.get("cheapest", {})),
            ("Fastest", highlights.get("fastest", {})),
        ):
            if not isinstance(row, dict) or not row:
                continue
            lines.append(
                f"- {label}: `{row.get('entry_name', '')}` "
                f"({int(row.get('overall_passed', 0) or 0)}/{int(row.get('overall_total', 0) or 0)}, "
                f"cost=${float(row.get('estimated_total_cost_usd', 0.0) or 0.0):.4f}, "
                f"latency={float(row.get('avg_latency_ms', 0.0) or 0.0):.1f}ms)"
            )
        lines.append("")

    lines.extend(["## Aggregate Leaderboard", ""])
    for row in leaderboard:
        if not isinstance(row, dict):
            continue
        lines.append(
            f"- `{row.get('entry_name', '')}` / `{row.get('requested_model', '')}`: "
            f"{int(row.get('overall_passed', 0) or 0)}/{int(row.get('overall_total', 0) or 0)}, "
            f"packs=`{row.get('packs_covered', '')}`, "
            f"avg_score={float(row.get('avg_score', 0.0) or 0.0):.2f}, "
            f"cost=${float(row.get('estimated_total_cost_usd', 0.0) or 0.0):.4f}, "
            f"latency={float(row.get('avg_latency_ms', 0.0) or 0.0):.1f}ms"
        )
    if not leaderboard:
        lines.append("- No aggregate rows found.")
    lines.append("")

    lines.extend(["## Pack Runs", ""])
    for row in pack_runs:
        if not isinstance(row, dict):
            continue
        lines.append(
            f"- `{row.get('scenario_pack', '')}`: top=`{row.get('top_entry_name', '')}` "
            f"{int(row.get('top_overall_passed', 0) or 0)}/{int(row.get('top_overall_total', 0) or 0)}, "
            f"matrix=`{row.get('matrix_run_dir', '')}`, analysis=`{row.get('analysis_dir', '')}`"
        )
    if not pack_runs:
        lines.append("- No pack runs found.")
    lines.append("")

    lines.extend(["## Pareto Frontier", ""])
    for row in frontier:
        if not isinstance(row, dict):
            continue
        lines.append(
            f"- `{row.get('entry_name', '')}`: "
            f"pass_rate={float(row.get('overall_pass_rate', 0.0) or 0.0):.2%}, "
            f"cost=${float(row.get('estimated_total_cost_usd', 0.0) or 0.0):.4f}, "
            f"latency={float(row.get('avg_latency_ms', 0.0) or 0.0):.1f}ms"
        )
    if not frontier:
        lines.append("- No frontier rows found.")
    lines.append("")

    lines.extend(["## X Post", "", str(payload.get("x_post", "")), ""])
    return "\n".join(lines)


def render_study_x_post(
    leaderboard: Sequence[Mapping[str, object]],
    *,
    pack_summaries: Sequence[Mapping[str, object]],
    label: str | None,
) -> str:
    if not leaderboard:
        return "BrainLayer frozen study completed."

    prefix = "BrainLayer frozen study"
    if label:
        prefix = f"{prefix} ({label})"

    top_rows = list(leaderboard[:3])
    top_bits = [
        f"{row.get('entry_name', '')} {int(row.get('overall_passed', 0) or 0)}/{int(row.get('overall_total', 0) or 0)}"
        for row in top_rows
    ]
    highlights = build_matrix_analysis_highlights(leaderboard)
    best_value = highlights.get("best_value", {})
    held_out_row = next(
        (
            row
            for row in pack_summaries
            if str(row.get("scenario_pack", "")) == "held_out"
        ),
        {},
    )

    parts = [
        f"{prefix}: top aggregate rows {', '.join(top_bits)}.",
        (
            f"Held-out leader: {held_out_row.get('top_entry_name', '')} "
            f"{int(held_out_row.get('top_overall_passed', 0) or 0)}/"
            f"{int(held_out_row.get('top_overall_total', 0) or 0)}."
            if isinstance(held_out_row, dict) and held_out_row
            else ""
        ),
        (
            f"Best value: {best_value.get('entry_name', '')} "
            f"${float(best_value.get('estimated_total_cost_usd', 0.0) or 0.0):.4f}."
            if isinstance(best_value, dict) and best_value
            else ""
        ),
    ]
    return " ".join(part for part in parts if part).strip()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the frozen BrainLayer study protocol across standard, hard, and held-out packs."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_STUDY_CONFIG,
        help="Path to the model matrix config used for the study.",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=DEFAULT_STUDY_PROTOCOL,
        help="Path to the frozen study protocol Markdown file.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_STUDY_EXPORT_ROOT,
        help="Directory for study bundles.",
    )
    parser.add_argument(
        "--packs",
        default="standard,hard,held_out",
        help="Comma-separated scenario packs to run. Use 'all' for the default frozen set.",
    )
    parser.add_argument(
        "--with-ablations",
        action="store_true",
        help="Run ablations for every entry in the chosen config.",
    )
    parser.add_argument(
        "--suite",
        choices=("all", "contradiction", "natural"),
        default="all",
        help="Run both suites or restrict the study to one suite.",
    )
    parser.add_argument(
        "--score-exact",
        action="store_true",
        help="Disable judge-backed semantic behavior scoring and require exact normalized matches.",
    )
    parser.add_argument(
        "--label",
        help="Optional label to attach to the study bundle and underlying pack runs.",
    )
    parser.add_argument(
        "--runtime-profile",
        choices=(RUNTIME_PROFILE_DEFAULT, RUNTIME_PROFILE_STUDY_V2),
        default=RUNTIME_PROFILE_DEFAULT,
        help="Choose the default BrainLayer runtime set or the study-v2 stronger-baseline set.",
    )
    args = parser.parse_args(argv)

    suites = SUITE_NAMES if args.suite == "all" else (args.suite,)
    study_dir = run_study(
        config_path=args.config,
        protocol_path=args.protocol,
        export_root=args.output_root,
        scenario_packs=parse_study_scenario_packs(args.packs),
        include_ablations=args.with_ablations,
        suites=suites,
        behavior_scoring_mode="exact" if args.score_exact else "judge",
        label=args.label,
        runtime_profile=args.runtime_profile,
    )
    payload = json.loads((study_dir / "study_summary.json").read_text())
    print(render_study_summary_markdown(payload))
    print(f"Study bundle written to {study_dir}")
    return 0


def _study_id(label: str | None) -> str:
    timestamp = utc_now_compact()
    base = slugify_label(label or "study")
    return f"{timestamp}-{base}"


def _pack_label(label: str | None, pack: str) -> str:
    if label:
        return f"{label}-{pack}"
    return f"study-{pack}"


__all__ = [
    "DEFAULT_STUDY_CONFIG",
    "DEFAULT_STUDY_EXPORT_ROOT",
    "DEFAULT_STUDY_PROTOCOL",
    "DEFAULT_STUDY_SCENARIO_PACKS",
    "build_study_aggregate_leaderboard",
    "parse_study_scenario_packs",
    "render_study_summary_markdown",
    "render_study_x_post",
    "run_study",
]
