import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from brainlayer.matrix_analysis import (
    build_cost_quality_frontier,
    export_matrix_analysis,
    load_matrix_history,
    select_matrix_history_run,
)
from brainlayer.model_matrix import export_model_matrix_results, load_model_matrix_entries, run_model_matrix


ROOT = Path(__file__).resolve().parent.parent


class MatrixAnalysisTests(unittest.TestCase):
    def test_load_live_openai_matrix_config_has_real_pricing(self) -> None:
        entries = load_model_matrix_entries(ROOT / "examples" / "model_matrix.openai.live.json")

        self.assertEqual(
            [entry.name for entry in entries],
            ["openai-gpt-5.1", "openai-gpt-5-mini", "openai-gpt-5-nano"],
        )
        self.assertTrue(all(entry.mode == "live" for entry in entries))
        self.assertTrue(all(entry.input_cost_per_1k_tokens > 0.0 for entry in entries))
        self.assertTrue(all(entry.output_cost_per_1k_tokens > 0.0 for entry in entries))
        self.assertTrue(all(entry.max_output_tokens_field == "max_completion_tokens" for entry in entries))

    def test_export_matrix_analysis_writes_report_files(self) -> None:
        entries = load_model_matrix_entries(ROOT / "examples" / "model_matrix.sample.json")
        results = run_model_matrix(entries, include_ablations=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            export_root = Path(tmpdir) / "matrix_runs"
            export_model_matrix_results(
                results,
                export_root,
                include_ablations=False,
                label="analysis-smoke",
            )
            analysis_root = Path(tmpdir) / "analysis"
            export_dir = export_matrix_analysis(
                export_root / "matrix_history.jsonl",
                analysis_root,
                label="smoke",
            )

            self.assertTrue((export_dir / "report.json").exists())
            self.assertTrue((export_dir / "report.md").exists())
            self.assertTrue((export_dir / "leaderboard.csv").exists())
            self.assertTrue((export_dir / "pareto_frontier.csv").exists())
            self.assertTrue((export_dir / "suite_summary.csv").exists())
            self.assertTrue((export_dir / "run_history.csv").exists())
            self.assertTrue((export_dir / "x_post.txt").exists())
            self.assertTrue((export_dir / "cost_vs_quality.svg").exists())

            payload = json.loads((export_dir / "report.json").read_text())
            self.assertEqual(
                payload["run_metadata"]["label"],
                "analysis-smoke",
            )
            self.assertIn("leaderboard", payload)
            self.assertIn("pareto_frontier", payload)
            self.assertIn("highlights", payload)

    def test_select_latest_matrix_history_run_uses_latest_generated_timestamp(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            history_path = Path(tmpdir) / "matrix_history.jsonl"
            older = {
                "metadata": {"run_id": "older", "generated_at_utc": "2026-03-30T12:00:00+00:00"},
                "summary": [],
                "leaderboard": [],
                "results": [],
            }
            newer = {
                "metadata": {"run_id": "newer", "generated_at_utc": "2026-04-01T12:00:00+00:00"},
                "summary": [],
                "leaderboard": [],
                "results": [],
            }
            history_path.write_text(json.dumps(older) + "\n" + json.dumps(newer) + "\n")

            runs = load_matrix_history(history_path)
            selected = select_matrix_history_run(runs)

        self.assertEqual(selected.metadata["run_id"], "newer")

    def test_cost_quality_frontier_prefers_better_pass_rate_or_latency(self) -> None:
        rows = [
            {
                "entry_name": "cheap-mid",
                "runtime_name": "model_loop",
                "overall_pass_rate": 0.8,
                "estimated_total_cost_usd": 0.001,
                "avg_latency_ms": 120.0,
            },
            {
                "entry_name": "cheap-mid-slower",
                "runtime_name": "model_loop",
                "overall_pass_rate": 0.8,
                "estimated_total_cost_usd": 0.002,
                "avg_latency_ms": 180.0,
            },
            {
                "entry_name": "better",
                "runtime_name": "model_loop",
                "overall_pass_rate": 0.9,
                "estimated_total_cost_usd": 0.003,
                "avg_latency_ms": 160.0,
            },
        ]

        frontier = build_cost_quality_frontier(rows)

        self.assertEqual(
            [row["entry_name"] for row in frontier],
            ["cheap-mid", "better"],
        )

    def test_analysis_cli_reports_output_directory(self) -> None:
        entries = load_model_matrix_entries(ROOT / "examples" / "model_matrix.sample.json")
        results = run_model_matrix(entries, include_ablations=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            export_root = Path(tmpdir) / "matrix_runs"
            export_model_matrix_results(
                results,
                export_root,
                include_ablations=False,
                label="cli-smoke",
            )
            analysis_root = Path(tmpdir) / "analysis"
            completed = subprocess.run(
                [
                    "python3",
                    str(ROOT / "scripts" / "analyze_matrix_history.py"),
                    "--history",
                    str(export_root / "matrix_history.jsonl"),
                    "--output-root",
                    str(analysis_root),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

        self.assertIn("BrainLayer Matrix Analysis", completed.stdout)
        self.assertIn("Analysis exports written to", completed.stdout)


if __name__ == "__main__":
    unittest.main()
