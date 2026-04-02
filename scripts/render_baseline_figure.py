from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_STUDY_DIR = Path("artifacts/study_runs/20260402T175429Z-study-v1-gemini-core")
DEFAULT_OUTPUT = Path("docs/figures/study-v1-gemini-core-overview.svg")

PACK_ORDER = ("standard", "hard", "held_out")
MODEL_ORDER = ("gemini-2.5-flash", "gemini-2.5-flash-lite")
MODEL_LABELS = {
    "gemini-2.5-flash": "Gemini 2.5 Flash",
    "gemini-2.5-flash-lite": "Gemini 2.5 Flash-Lite",
}
MODEL_COLORS = {
    "gemini-2.5-flash": "#0b7285",
    "gemini-2.5-flash-lite": "#e67700",
}


def load_study_payload(study_dir: Path) -> dict:
    return json.loads((study_dir / "study_summary.json").read_text())


def load_pack_rows(study_dir: Path) -> dict[str, dict[str, dict]]:
    rows_by_pack: dict[str, dict[str, dict]] = {}
    matrix_root = study_dir / "matrix_runs"
    for results_path in sorted(matrix_root.glob("2026*/results.json")):
        payload = json.loads(results_path.read_text())
        pack = str(payload.get("metadata", {}).get("scenario_pack", ""))
        rows = {
            str(row.get("entry_name", "")): row
            for row in payload.get("leaderboard", [])
            if isinstance(row, dict)
        }
        rows_by_pack[pack] = rows
    return rows_by_pack


def render_svg(study_payload: dict, pack_rows: dict[str, dict[str, dict]]) -> str:
    width = 1200
    height = 820
    chart_x = 90
    chart_y = 180
    chart_w = 610
    chart_h = 360
    max_total = 19
    scale = chart_w / max_total

    aggregate_rows = {
        str(row.get("entry_name", "")): row
        for row in study_payload.get("aggregate_leaderboard", [])
        if isinstance(row, dict)
    }

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f7f3eb"/>',
        '<rect x="48" y="40" width="1104" height="740" rx="28" fill="#fffaf2" stroke="#d7c6a8" stroke-width="2"/>',
        '<text x="88" y="98" font-family="Georgia, Times New Roman, serif" font-size="34" fill="#1d2a2e">BrainLayer Frozen Baseline</text>',
        '<text x="88" y="132" font-family="Helvetica, Arial, sans-serif" font-size="18" fill="#5a5f62">study-v1-gemini-core | authored + hard are strong; held-out is the real frontier</text>',
        '<text x="90" y="168" font-family="Helvetica, Arial, sans-serif" font-size="13" fill="#7a746b">Passes per pack</text>',
    ]

    for idx in range(max_total + 1):
        x = chart_x + idx * scale
        label = str(idx)
        if idx == max_total:
            label = f"{idx}"
        elements.append(
            f'<line x1="{x:.1f}" y1="{chart_y}" x2="{x:.1f}" y2="{chart_y + chart_h}" stroke="#e7dcc7" stroke-width="1"/>'
        )
        elements.append(
            f'<text x="{x - 4:.1f}" y="{chart_y + chart_h + 22}" font-family="Helvetica, Arial, sans-serif" font-size="11" fill="#7a746b">{label}</text>'
        )

    row_gap = 118
    bar_h = 26
    for pack_index, pack in enumerate(PACK_ORDER):
        y = chart_y + pack_index * row_gap
        title = pack.replace("_", " ").title()
        total = 14 if pack != "standard" else 19
        elements.append(
            f'<text x="{chart_x - 4}" y="{y - 18}" text-anchor="start" font-family="Helvetica, Arial, sans-serif" font-size="19" fill="#263238">{title}</text>'
        )
        elements.append(
            f'<text x="{chart_x + chart_w + 12}" y="{y - 18}" font-family="Helvetica, Arial, sans-serif" font-size="12" fill="#7a746b">/{total}</text>'
        )
        rows = pack_rows.get(pack, {})
        for model_index, model in enumerate(MODEL_ORDER):
            row = rows.get(model, {})
            passed = int(row.get("overall_passed", 0) or 0)
            total = int(row.get("overall_total", 19 if pack == "standard" else 14) or 0)
            bar_y = y + model_index * 36
            fill_w = scale * passed
            track_w = scale * total
            color = MODEL_COLORS[model]
            label = MODEL_LABELS[model]
            elements.extend(
                [
                    f'<rect x="{chart_x}" y="{bar_y}" width="{track_w:.1f}" height="{bar_h}" rx="13" fill="#efe3ce"/>',
                    f'<rect x="{chart_x}" y="{bar_y}" width="{fill_w:.1f}" height="{bar_h}" rx="13" fill="{color}"/>',
                    f'<text x="{chart_x - 8}" y="{bar_y + 18}" text-anchor="end" font-family="Helvetica, Arial, sans-serif" font-size="13" fill="#364046">{label}</text>',
                    f'<text x="{chart_x + track_w + 12:.1f}" y="{bar_y + 18}" font-family="Helvetica, Arial, sans-serif" font-size="13" fill="#364046">{passed}/{total}</text>',
                ]
            )

    right_x = 760
    card_w = 330
    card_h = 148
    for idx, model in enumerate(MODEL_ORDER):
        row = aggregate_rows[model]
        x = right_x
        y = 180 + idx * 174
        color = MODEL_COLORS[model]
        label = MODEL_LABELS[model]
        overall = f"{int(row.get('overall_passed', 0) or 0)}/{int(row.get('overall_total', 0) or 0)}"
        cost = f"${float(row.get('estimated_total_cost_usd', 0.0) or 0.0):.4f}"
        latency = f"{float(row.get('avg_latency_ms', 0.0) or 0.0):.0f}ms"
        extraction = f"{int(row.get('natural_extraction_passed', 0) or 0)}/{int(row.get('natural_extraction_total', 0) or 0)}"
        behavior = f"{int(row.get('natural_behavior_passed', 0) or 0)}/{int(row.get('natural_behavior_total', 0) or 0)}"
        elements.extend(
            [
                f'<rect x="{x}" y="{y}" width="{card_w}" height="{card_h}" rx="22" fill="#fff" stroke="{color}" stroke-width="2"/>',
                f'<rect x="{x}" y="{y}" width="18" height="{card_h}" rx="9" fill="{color}"/>',
                f'<text x="{x + 34}" y="{y + 38}" font-family="Georgia, Times New Roman, serif" font-size="24" fill="#1f2c31">{label}</text>',
                f'<text x="{x + 34}" y="{y + 72}" font-family="Helvetica, Arial, sans-serif" font-size="15" fill="#415057">overall {overall}</text>',
                f'<text x="{x + 34}" y="{y + 98}" font-family="Helvetica, Arial, sans-serif" font-size="14" fill="#415057">natural extraction {extraction} | behavior {behavior}</text>',
                f'<text x="{x + 34}" y="{y + 124}" font-family="Helvetica, Arial, sans-serif" font-size="14" fill="#415057">cost {cost} | avg latency {latency}</text>',
            ]
        )

    note_y = 560
    elements.extend(
        [
            '<rect x="760" y="560" width="330" height="150" rx="22" fill="#f3eee3" stroke="#d8ccba" stroke-width="1.5"/>',
            '<text x="784" y="594" font-family="Georgia, Times New Roman, serif" font-size="23" fill="#1f2c31">Read Of The Baseline</text>',
            '<text x="784" y="626" font-family="Helvetica, Arial, sans-serif" font-size="15" fill="#425055">1. Contradiction handling is effectively solved in this baseline.</text>',
            '<text x="784" y="650" font-family="Helvetica, Arial, sans-serif" font-size="15" fill="#425055">2. Natural extraction is strong on authored and hard packs.</text>',
            '<text x="784" y="674" font-family="Helvetica, Arial, sans-serif" font-size="15" fill="#425055">3. Held-out wording remains the real generalization boundary.</text>',
            '<text x="90" y="612" font-family="Georgia, Times New Roman, serif" font-size="22" fill="#1f2c31">Why freeze here?</text>',
            '<text x="90" y="640" font-family="Helvetica, Arial, sans-serif" font-size="15" fill="#425055">The remaining misses are in held-out language variation, not broken infrastructure.</text>',
            '<text x="90" y="664" font-family="Helvetica, Arial, sans-serif" font-size="15" fill="#425055">So this baseline stays intentionally imperfect instead of tuning held-out phrasing away.</text>',
            '<text x="90" y="726" font-family="Helvetica, Arial, sans-serif" font-size="12" fill="#7a746b">Source bundle: artifacts/study_runs/20260402T175429Z-study-v1-gemini-core</text>',
            '</svg>',
        ]
    )
    return "\n".join(elements)


def main() -> int:
    parser = argparse.ArgumentParser(description="Render a single SVG figure from the frozen BrainLayer baseline bundle.")
    parser.add_argument("--study-dir", type=Path, default=DEFAULT_STUDY_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    study_payload = load_study_payload(args.study_dir)
    pack_rows = load_pack_rows(args.study_dir)
    svg = render_svg(study_payload, pack_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(svg)
    print(f"Wrote baseline figure to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
