import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from brainlayer.study_runner import (
    build_study_aggregate_leaderboard,
    parse_study_scenario_packs,
    run_study,
)


ROOT = Path(__file__).resolve().parent.parent


class StudyRunnerTests(unittest.TestCase):
    def test_parse_study_scenario_packs_supports_all_and_dedupes(self) -> None:
        self.assertEqual(parse_study_scenario_packs("all"), ("standard", "hard", "held_out"))
        self.assertEqual(
            parse_study_scenario_packs("standard,held_out,standard"),
            ("standard", "held_out"),
        )
        self.assertEqual(
            parse_study_scenario_packs("standard,external_dev,external_held_out"),
            ("standard", "external_dev", "external_held_out"),
        )

    def test_run_study_writes_frozen_bundle_and_aggregate_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            study_dir = run_study(
                config_path=ROOT / "examples" / "model_matrix.sample.json",
                protocol_path=ROOT / "docs" / "study_protocol.md",
                export_root=Path(tmpdir) / "study_runs",
                scenario_packs=("standard", "held_out"),
                label="heuristic-study",
            )

            self.assertTrue((study_dir / "study_protocol.md").exists())
            self.assertTrue((study_dir / "study_config.json").exists())
            self.assertTrue((study_dir / "study_summary.json").exists())
            self.assertTrue((study_dir / "study_summary.md").exists())
            self.assertTrue((study_dir / "aggregate_leaderboard.csv").exists())
            self.assertTrue((study_dir / "pack_summary.csv").exists())
            self.assertTrue((study_dir / "pareto_frontier.csv").exists())
            self.assertTrue((study_dir / "x_post.txt").exists())
            self.assertTrue((study_dir / "matrix_runs" / "matrix_history.jsonl").exists())

            payload = json.loads((study_dir / "study_summary.json").read_text())
            self.assertEqual(
                payload["study_metadata"]["scenario_packs"],
                ["standard", "held_out"],
            )
            self.assertEqual(len(payload["pack_runs"]), 2)
            self.assertEqual(payload["aggregate_leaderboard"][0]["overall_total"], 33)
            self.assertIn("Held-out leader", payload["x_post"])

    def test_build_study_aggregate_leaderboard_combines_pack_rows(self) -> None:
        aggregate = build_study_aggregate_leaderboard(
            [
                {
                    "scenario_pack": "standard",
                    "leaderboard": [
                        {
                            "entry_name": "alpha",
                            "runtime_name": "model_loop",
                            "eval_mode": "heuristic",
                            "provider_name": "test",
                            "requested_model": "test-model",
                            "overall_passed": 19,
                            "overall_total": 19,
                            "contradiction_passed": 8,
                            "contradiction_total": 8,
                            "natural_passed": 11,
                            "natural_total": 11,
                            "natural_extraction_passed": 6,
                            "natural_extraction_total": 6,
                            "natural_behavior_passed": 5,
                            "natural_behavior_total": 5,
                            "parse_failures": 0,
                            "empty_answers": 0,
                            "errors": 0,
                            "skipped": 0,
                            "estimated_total_cost_usd": 1.0,
                            "avg_score": 1.0,
                            "avg_latency_ms": 10.0,
                        }
                    ],
                },
                {
                    "scenario_pack": "held_out",
                    "leaderboard": [
                        {
                            "entry_name": "alpha",
                            "runtime_name": "model_loop",
                            "eval_mode": "heuristic",
                            "provider_name": "test",
                            "requested_model": "test-model",
                            "overall_passed": 14,
                            "overall_total": 14,
                            "contradiction_passed": 4,
                            "contradiction_total": 4,
                            "natural_passed": 10,
                            "natural_total": 10,
                            "natural_extraction_passed": 5,
                            "natural_extraction_total": 5,
                            "natural_behavior_passed": 5,
                            "natural_behavior_total": 5,
                            "parse_failures": 0,
                            "empty_answers": 0,
                            "errors": 0,
                            "skipped": 0,
                            "estimated_total_cost_usd": 2.0,
                            "avg_score": 1.0,
                            "avg_latency_ms": 20.0,
                        }
                    ],
                },
            ]
        )

        self.assertEqual(len(aggregate), 1)
        self.assertEqual(aggregate[0]["overall_passed"], 33)
        self.assertEqual(aggregate[0]["overall_total"], 33)
        self.assertEqual(aggregate[0]["packs_covered"], "held_out,standard")
        self.assertAlmostEqual(aggregate[0]["estimated_total_cost_usd"], 3.0)
        self.assertAlmostEqual(aggregate[0]["avg_latency_ms"], (19 * 10.0 + 14 * 20.0) / 33)

    def test_study_cli_writes_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            completed = subprocess.run(
                [
                    "python3",
                    str(ROOT / "scripts" / "run_study.py"),
                    "--config",
                    str(ROOT / "examples" / "model_matrix.sample.json"),
                    "--protocol",
                    str(ROOT / "docs" / "study_protocol.md"),
                    "--output-root",
                    str(Path(tmpdir) / "study_runs"),
                    "--packs",
                    "standard,held_out",
                    "--label",
                    "cli-study",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertIn("BrainLayer Study Summary", completed.stdout)
            self.assertIn("Study bundle written to", completed.stdout)


if __name__ == "__main__":
    unittest.main()
