from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

from .benchmark_harness import slugify_label, utc_now_compact, utc_now_iso, write_csv


DEFAULT_MATRIX_HISTORY = Path("artifacts/matrix_runs/matrix_history.jsonl")
DEFAULT_ANALYSIS_ROOT = Path("artifacts/matrix_analysis")


@dataclass(frozen=True)
class MatrixHistoryRun:
    metadata: Dict[str, object]
    summary: List[Dict[str, object]]
    leaderboard: List[Dict[str, object]]
    results: List[Dict[str, object]]
    x_post: str = ""


def load_matrix_history(path: str | Path) -> List[MatrixHistoryRun]:
    history_path = Path(path)
    if not history_path.exists():
        raise FileNotFoundError(f"Matrix history file not found: {history_path}")

    runs: List[MatrixHistoryRun] = []
    with history_path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                continue
            metadata = payload.get("metadata", {})
            summary = payload.get("summary", [])
            leaderboard = payload.get("leaderboard", [])
            results = payload.get("results", [])
            x_post = str(payload.get("x_post", ""))
            if not isinstance(metadata, dict):
                metadata = {}
            if not isinstance(summary, list):
                summary = []
            if not isinstance(leaderboard, list):
                leaderboard = []
            if not isinstance(results, list):
                results = []
            runs.append(
                MatrixHistoryRun(
                    metadata={str(key): value for key, value in metadata.items()},
                    summary=[_stringify_keys(row) for row in summary if isinstance(row, dict)],
                    leaderboard=[_stringify_keys(row) for row in leaderboard if isinstance(row, dict)],
                    results=[_stringify_keys(row) for row in results if isinstance(row, dict)],
                    x_post=x_post,
                )
            )
    return runs


def select_matrix_history_run(
    runs: Sequence[MatrixHistoryRun],
    *,
    run_id: str | None = None,
) -> MatrixHistoryRun:
    if not runs:
        raise ValueError("Matrix history is empty.")

    if run_id:
        for run in runs:
            if str(run.metadata.get("run_id", "")) == run_id:
                return run
        raise ValueError(f"Run id not found in matrix history: {run_id}")

    return max(
        runs,
        key=lambda run: (
            str(run.metadata.get("generated_at_utc", "")),
            str(run.metadata.get("run_id", "")),
        ),
    )


def build_matrix_analysis(
    run: MatrixHistoryRun,
    *,
    include_history_overview: bool = True,
    all_runs: Sequence[MatrixHistoryRun] | None = None,
) -> Dict[str, object]:
    leaderboard_rows = [_normalize_leaderboard_row(row) for row in run.leaderboard]
    focus_rows = _focus_rows(leaderboard_rows)
    pareto_frontier = build_cost_quality_frontier(focus_rows)
    suite_summary = [_normalize_summary_row(row) for row in run.summary]
    history_overview = (
        build_history_overview(all_runs or [run]) if include_history_overview else []
    )
    highlights = build_matrix_analysis_highlights(focus_rows)
    x_post = render_matrix_analysis_x_post(run, focus_rows, pareto_frontier)

    return {
        "run_metadata": dict(run.metadata),
        "focus_mode": "live_only" if any(row.get("eval_mode") == "live" for row in focus_rows) else "all_model_rows",
        "leaderboard": focus_rows,
        "pareto_frontier": pareto_frontier,
        "suite_summary": suite_summary,
        "history_overview": history_overview,
        "highlights": highlights,
        "source_x_post": run.x_post,
        "x_post": x_post,
    }


def build_history_overview(runs: Sequence[MatrixHistoryRun]) -> List[Dict[str, object]]:
    overview: List[Dict[str, object]] = []
    for run in sorted(
        runs,
        key=lambda item: (
            str(item.metadata.get("generated_at_utc", "")),
            str(item.metadata.get("run_id", "")),
        ),
    ):
        leaderboard_rows = [_normalize_leaderboard_row(row) for row in run.leaderboard]
        focus_rows = _focus_rows(leaderboard_rows)
        top_row = focus_rows[0] if focus_rows else {}
        overview.append(
            {
                "run_id": str(run.metadata.get("run_id", "")),
                "generated_at_utc": str(run.metadata.get("generated_at_utc", "")),
                "label": str(run.metadata.get("label", "")),
                "scenario_pack": str(run.metadata.get("scenario_pack", "")),
                "entry_count": int(run.metadata.get("entry_count", 0) or 0),
                "case_count": int(run.metadata.get("case_count", 0) or 0),
                "top_entry_name": str(top_row.get("entry_name", "")),
                "top_requested_model": str(top_row.get("requested_model", "")),
                "top_overall_pass_rate": float(top_row.get("overall_pass_rate", 0.0) or 0.0),
                "top_estimated_total_cost_usd": float(
                    top_row.get("estimated_total_cost_usd", 0.0) or 0.0
                ),
                "top_avg_latency_ms": float(top_row.get("avg_latency_ms", 0.0) or 0.0),
            }
        )
    return overview


def build_cost_quality_frontier(rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    candidates = [
        dict(row)
        for row in rows
        if str(row.get("runtime_name", "")) == "model_loop"
    ]
    candidates.sort(
        key=lambda row: (
            float(row.get("estimated_total_cost_usd", 0.0) or 0.0),
            -float(row.get("overall_pass_rate", 0.0) or 0.0),
            float(row.get("avg_latency_ms", 0.0) or 0.0),
            str(row.get("entry_name", "")),
        )
    )

    frontier: List[Dict[str, object]] = []
    best_pass_rate = -1.0
    best_latency_for_pass = float("inf")
    for row in candidates:
        pass_rate = float(row.get("overall_pass_rate", 0.0) or 0.0)
        latency_ms = float(row.get("avg_latency_ms", 0.0) or 0.0)
        if pass_rate > best_pass_rate + 1e-9:
            frontier.append(dict(row))
            best_pass_rate = pass_rate
            best_latency_for_pass = latency_ms
            continue
        if abs(pass_rate - best_pass_rate) <= 1e-9 and latency_ms < best_latency_for_pass - 1e-9:
            frontier.append(dict(row))
            best_latency_for_pass = latency_ms
    return frontier


def build_matrix_analysis_highlights(rows: Sequence[Mapping[str, object]]) -> Dict[str, Dict[str, object]]:
    if not rows:
        return {}

    top_accuracy = max(
        rows,
        key=lambda row: (
            float(row.get("overall_pass_rate", 0.0) or 0.0),
            float(row.get("avg_score", 0.0) or 0.0),
            -float(row.get("avg_latency_ms", 0.0) or 0.0),
            str(row.get("entry_name", "")),
        ),
    )
    fastest = min(
        rows,
        key=lambda row: (
            float(row.get("avg_latency_ms", 0.0) or 0.0),
            str(row.get("entry_name", "")),
        ),
    )

    priced_rows = [row for row in rows if float(row.get("estimated_total_cost_usd", 0.0) or 0.0) > 0.0]
    cheapest = (
        min(
            priced_rows,
            key=lambda row: (
                float(row.get("estimated_total_cost_usd", 0.0) or 0.0),
                str(row.get("entry_name", "")),
            ),
        )
        if priced_rows
        else {}
    )

    top_pass_rate = float(top_accuracy.get("overall_pass_rate", 0.0) or 0.0)
    best_value_pool = [
        row for row in priced_rows if abs(float(row.get("overall_pass_rate", 0.0) or 0.0) - top_pass_rate) <= 1e-9
    ]
    best_value = (
        min(
            best_value_pool,
            key=lambda row: (
                float(row.get("estimated_total_cost_usd", 0.0) or 0.0),
                float(row.get("avg_latency_ms", 0.0) or 0.0),
                str(row.get("entry_name", "")),
            ),
        )
        if best_value_pool
        else dict(top_accuracy)
    )

    return {
        "top_accuracy": dict(top_accuracy),
        "fastest": dict(fastest),
        "cheapest": dict(cheapest),
        "best_value": dict(best_value),
    }


def render_matrix_analysis_markdown(analysis: Mapping[str, object]) -> str:
    metadata = analysis.get("run_metadata", {})
    leaderboard = analysis.get("leaderboard", [])
    frontier = analysis.get("pareto_frontier", [])
    highlights = analysis.get("highlights", {})
    suite_summary = analysis.get("suite_summary", [])
    history_overview = analysis.get("history_overview", [])

    if not isinstance(metadata, dict):
        metadata = {}
    if not isinstance(leaderboard, list):
        leaderboard = []
    if not isinstance(frontier, list):
        frontier = []
    if not isinstance(highlights, dict):
        highlights = {}
    if not isinstance(suite_summary, list):
        suite_summary = []
    if not isinstance(history_overview, list):
        history_overview = []

    lines = [
        "# BrainLayer Matrix Analysis",
        "",
        f"- Run: `{metadata.get('run_id', '')}`",
        f"- Generated: `{metadata.get('generated_at_utc', '')}`",
        f"- Commit: `{metadata.get('git_commit', '')}`",
        f"- Label: `{metadata.get('label', '')}`",
        f"- Scenario pack: `{metadata.get('scenario_pack', '')}`",
        f"- Focus mode: `{analysis.get('focus_mode', '')}`",
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

    lines.extend(["## Leaderboard", ""])
    if leaderboard:
        for row in leaderboard:
            if not isinstance(row, dict):
                continue
            lines.append(
                f"- `{row.get('entry_name', '')}` / `{row.get('requested_model', '')}`: "
                f"{int(row.get('overall_passed', 0) or 0)}/{int(row.get('overall_total', 0) or 0)}, "
                f"avg_score={float(row.get('avg_score', 0.0) or 0.0):.2f}, "
                f"cost=${float(row.get('estimated_total_cost_usd', 0.0) or 0.0):.4f}, "
                f"latency={float(row.get('avg_latency_ms', 0.0) or 0.0):.1f}ms, "
                f"errors={int(row.get('errors', 0) or 0)}, parse_failures={int(row.get('parse_failures', 0) or 0)}"
            )
    else:
        lines.append("- No rows found.")
    lines.append("")

    lines.extend(["## Pareto Frontier", ""])
    if frontier:
        for row in frontier:
            if not isinstance(row, dict):
                continue
            lines.append(
                f"- `{row.get('entry_name', '')}`: "
                f"pass_rate={float(row.get('overall_pass_rate', 0.0) or 0.0):.2%}, "
                f"cost=${float(row.get('estimated_total_cost_usd', 0.0) or 0.0):.4f}, "
                f"latency={float(row.get('avg_latency_ms', 0.0) or 0.0):.1f}ms"
            )
    else:
        lines.append("- No frontier rows found.")
    lines.append("")

    lines.extend(["## Suite Summary", ""])
    for row in suite_summary:
        if not isinstance(row, dict):
            continue
        lines.append(
            f"- `{row.get('entry_name', '')}` / `{row.get('suite_name', '')}`: "
            f"{int(row.get('passed', 0) or 0)}/{int(row.get('total', 0) or 0)}, "
            f"avg_score={float(row.get('avg_score', 0.0) or 0.0):.2f}, "
            f"cost=${float(row.get('estimated_total_cost_usd', 0.0) or 0.0):.4f}, "
            f"latency={float(row.get('avg_latency_ms', 0.0) or 0.0):.1f}ms"
        )
    lines.append("")

    if history_overview:
        lines.extend(["## Run History", ""])
        for row in history_overview[-5:]:
            if not isinstance(row, dict):
                continue
            lines.append(
                f"- `{row.get('run_id', '')}`: top=`{row.get('top_entry_name', '')}`, "
                f"pass_rate={float(row.get('top_overall_pass_rate', 0.0) or 0.0):.2%}, "
                f"cost=${float(row.get('top_estimated_total_cost_usd', 0.0) or 0.0):.4f}"
            )
        lines.append("")

    lines.extend(["## X Post", "", str(analysis.get("x_post", "")), ""])
    return "\n".join(lines)


def render_matrix_analysis_x_post(
    run: MatrixHistoryRun,
    rows: Sequence[Mapping[str, object]],
    frontier: Sequence[Mapping[str, object]],
) -> str:
    if not rows:
        return "BrainLayer matrix analysis completed."

    has_live_rows = any(str(row.get("eval_mode", "")) == "live" for row in rows)
    prefix = "BrainLayer live matrix" if has_live_rows else "BrainLayer matrix"
    label = str(run.metadata.get("label", "")).strip()
    if label:
        prefix = f"{prefix} ({label})"

    top_rows = list(rows[:3])
    top_bits = [
        (
            f"{row.get('entry_name', '')} "
            f"{int(row.get('overall_passed', 0) or 0)}/{int(row.get('overall_total', 0) or 0)} "
            f"${float(row.get('estimated_total_cost_usd', 0.0) or 0.0):.4f}"
        )
        for row in top_rows
    ]
    fastest = min(
        rows,
        key=lambda row: (
            float(row.get("avg_latency_ms", 0.0) or 0.0),
            str(row.get("entry_name", "")),
        ),
    )
    best_value = build_matrix_analysis_highlights(rows).get("best_value", {})
    frontier_names = ", ".join(str(row.get("entry_name", "")) for row in frontier[:3] if row)
    parts = [
        f"{prefix}: top rows {', '.join(top_bits)}.",
        (
            f"Best value: {best_value.get('entry_name', '')} "
            f"at ${float(best_value.get('estimated_total_cost_usd', 0.0) or 0.0):.4f}."
            if isinstance(best_value, dict) and best_value
            else ""
        ),
        (
            f"Fastest avg latency: {fastest.get('entry_name', '')} "
            f"{float(fastest.get('avg_latency_ms', 0.0) or 0.0):.1f}ms."
        ),
        f"Cost/quality frontier: {frontier_names}." if frontier_names else "",
    ]
    return " ".join(part for part in parts if part).strip()


def render_cost_quality_svg(rows: Sequence[Mapping[str, object]]) -> str:
    width = 720
    height = 440
    left = 80
    right = 40
    top = 50
    bottom = 70
    plot_width = width - left - right
    plot_height = height - top - bottom
    if not rows:
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
            '<text x="40" y="40" font-family="monospace" font-size="18">No rows available.</text>'
            "</svg>"
        )

    costs = [float(row.get("estimated_total_cost_usd", 0.0) or 0.0) for row in rows]
    pass_rates = [float(row.get("overall_pass_rate", 0.0) or 0.0) for row in rows]
    min_cost = min(costs)
    max_cost = max(costs)
    min_pass = min(min(pass_rates), 0.0)
    max_pass = max(max(pass_rates), 1.0)
    cost_span = max(max_cost - min_cost, 1e-9)
    pass_span = max(max_pass - min_pass, 1e-9)

    def x_for(cost: float) -> float:
        return left + ((cost - min_cost) / cost_span) * plot_width

    def y_for(pass_rate: float) -> float:
        return top + plot_height - ((pass_rate - min_pass) / pass_span) * plot_height

    elements = [
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="white" />',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#222" stroke-width="1.5" />',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#222" stroke-width="1.5" />',
        '<text x="40" y="28" font-family="monospace" font-size="20">BrainLayer Cost vs Quality</text>',
        f'<text x="{left + (plot_width / 2) - 80}" y="{height - 20}" font-family="monospace" font-size="14">Estimated total cost (USD)</text>',
        '<text transform="rotate(-90 22 210)" x="22" y="210" font-family="monospace" font-size="14">Overall pass rate</text>',
    ]

    for index, tick in enumerate(range(0, 6)):
        x_pos = left + (plot_width / 5.0) * index
        tick_cost = min_cost + (cost_span / 5.0) * index
        elements.append(
            f'<line x1="{x_pos:.1f}" y1="{top + plot_height}" x2="{x_pos:.1f}" y2="{top + plot_height + 6}" stroke="#555" stroke-width="1" />'
        )
        elements.append(
            f'<text x="{x_pos - 18:.1f}" y="{top + plot_height + 24}" font-family="monospace" font-size="12">${tick_cost:.4f}</text>'
        )

    for index, tick in enumerate(range(0, 6)):
        y_pos = top + plot_height - (plot_height / 5.0) * index
        tick_pass = min_pass + (pass_span / 5.0) * index
        elements.append(
            f'<line x1="{left - 6}" y1="{y_pos:.1f}" x2="{left}" y2="{y_pos:.1f}" stroke="#555" stroke-width="1" />'
        )
        elements.append(
            f'<text x="18" y="{y_pos + 4:.1f}" font-family="monospace" font-size="12">{tick_pass:.0%}</text>'
        )

    palette = ["#0b7285", "#c92a2a", "#2b8a3e", "#e67700", "#5f3dc4", "#495057"]
    for index, row in enumerate(rows):
        cost = float(row.get("estimated_total_cost_usd", 0.0) or 0.0)
        pass_rate = float(row.get("overall_pass_rate", 0.0) or 0.0)
        x_pos = x_for(cost)
        y_pos = y_for(pass_rate)
        color = palette[index % len(palette)]
        label = str(row.get("entry_name", ""))
        elements.append(
            f'<circle cx="{x_pos:.1f}" cy="{y_pos:.1f}" r="6" fill="{color}" stroke="#111" stroke-width="1" />'
        )
        elements.append(
            f'<text x="{x_pos + 10:.1f}" y="{y_pos - 8:.1f}" font-family="monospace" font-size="12">{_escape_xml(label)}</text>'
        )

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        + "".join(elements)
        + "</svg>\n"
    )


def export_matrix_analysis(
    history_path: str | Path,
    export_root: str | Path,
    *,
    run_id: str | None = None,
    label: str | None = None,
) -> Path:
    runs = load_matrix_history(history_path)
    run = select_matrix_history_run(runs, run_id=run_id)
    analysis = build_matrix_analysis(run, all_runs=runs)
    analysis_metadata = {
        "analysis_id": _analysis_id(run, label=label),
        "generated_at_utc": utc_now_iso(),
        "history_path": str(Path(history_path)),
        "selected_run_id": str(run.metadata.get("run_id", "")),
    }

    export_dir = Path(export_root) / str(analysis_metadata["analysis_id"])
    export_dir.mkdir(parents=True, exist_ok=True)

    leaderboard_rows = [
        dict(row) for row in analysis.get("leaderboard", []) if isinstance(row, dict)
    ]
    frontier_rows = [
        dict(row) for row in analysis.get("pareto_frontier", []) if isinstance(row, dict)
    ]
    suite_rows = [
        dict(row) for row in analysis.get("suite_summary", []) if isinstance(row, dict)
    ]
    history_rows = [
        dict(row) for row in analysis.get("history_overview", []) if isinstance(row, dict)
    ]

    payload = {
        "analysis_metadata": analysis_metadata,
        **analysis,
    }

    (export_dir / "report.json").write_text(json.dumps(payload, indent=2) + "\n")
    (export_dir / "report.md").write_text(render_matrix_analysis_markdown(analysis))
    write_csv(export_dir / "leaderboard.csv", leaderboard_rows)
    write_csv(export_dir / "pareto_frontier.csv", frontier_rows)
    write_csv(export_dir / "suite_summary.csv", suite_rows)
    write_csv(export_dir / "run_history.csv", history_rows)
    (export_dir / "x_post.txt").write_text(str(analysis.get("x_post", "")) + "\n")
    (export_dir / "cost_vs_quality.svg").write_text(render_cost_quality_svg(leaderboard_rows))

    return export_dir


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Analyze BrainLayer matrix history and export a cost/quality/latency report."
    )
    parser.add_argument(
        "--history",
        type=Path,
        default=DEFAULT_MATRIX_HISTORY,
        help="Path to matrix_history.jsonl.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_ANALYSIS_ROOT,
        help="Directory for analysis outputs.",
    )
    parser.add_argument(
        "--run-id",
        help="Optional run id to analyze. Defaults to the latest run in history.",
    )
    parser.add_argument(
        "--label",
        help="Optional label for the analysis export directory.",
    )
    args = parser.parse_args(argv)

    export_dir = export_matrix_analysis(
        args.history,
        args.output_root,
        run_id=args.run_id,
        label=args.label,
    )
    payload = json.loads((export_dir / "report.json").read_text())
    print(render_matrix_analysis_markdown(payload))
    print(f"Analysis exports written to {export_dir}")
    return 0


def _analysis_id(run: MatrixHistoryRun, *, label: str | None) -> str:
    timestamp = utc_now_compact()
    base_label = label or str(run.metadata.get("run_id", "")) or "analysis"
    return f"{timestamp}-{slugify_label(base_label)}"


def _focus_rows(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    model_rows = [dict(row) for row in rows if str(row.get("runtime_name", "")) == "model_loop"]
    live_rows = [row for row in model_rows if str(row.get("eval_mode", "")) == "live"]
    focus_rows = live_rows or model_rows or [dict(row) for row in rows]
    focus_rows.sort(
        key=lambda row: (
            -float(row.get("overall_pass_rate", 0.0) or 0.0),
            -float(row.get("avg_score", 0.0) or 0.0),
            float(row.get("estimated_total_cost_usd", 0.0) or 0.0),
            float(row.get("avg_latency_ms", 0.0) or 0.0),
            str(row.get("entry_name", "")),
        )
    )
    return focus_rows


def _normalize_leaderboard_row(row: Mapping[str, object]) -> Dict[str, object]:
    payload = dict(row)
    for key in (
        "overall_pass_rate",
        "estimated_total_cost_usd",
        "avg_score",
        "avg_latency_ms",
        "avg_estimated_cost_usd",
        "avg_usage_total_tokens",
    ):
        if key in payload:
            payload[key] = float(payload.get(key, 0.0) or 0.0)
    for key in (
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
        if key in payload:
            payload[key] = int(payload.get(key, 0) or 0)
    return payload


def _normalize_summary_row(row: Mapping[str, object]) -> Dict[str, object]:
    payload = dict(row)
    for key in ("pass_rate", "estimated_total_cost_usd", "avg_score", "avg_latency_ms"):
        if key in payload:
            payload[key] = float(payload.get(key, 0.0) or 0.0)
    for key in (
        "passed",
        "total",
        "extraction_passed",
        "extraction_total",
        "behavior_passed",
        "behavior_total",
        "parse_failures",
        "empty_answers",
        "errors",
        "skipped",
    ):
        if key in payload:
            payload[key] = int(payload.get(key, 0) or 0)
    return payload


def _stringify_keys(payload: Mapping[str, object]) -> Dict[str, object]:
    return {str(key): value for key, value in payload.items()}


def _escape_xml(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


__all__ = [
    "DEFAULT_ANALYSIS_ROOT",
    "DEFAULT_MATRIX_HISTORY",
    "MatrixHistoryRun",
    "build_cost_quality_frontier",
    "build_history_overview",
    "build_matrix_analysis",
    "export_matrix_analysis",
    "load_matrix_history",
    "render_cost_quality_svg",
    "render_matrix_analysis_markdown",
    "render_matrix_analysis_x_post",
    "select_matrix_history_run",
]
