import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from brainlayer.llm import ModelMessage, ModelResponse, StaticLLMAdapter
from brainlayer.model_eval import HeuristicBrainLayerEvalAdapter
from brainlayer.model_matrix import (
    build_matrix_leaderboard,
    export_model_matrix_results,
    load_model_matrix_entries,
    run_model_matrix,
)
from brainlayer.natural_eval import HeuristicNaturalConversationAdapter


ROOT = Path(__file__).resolve().parent.parent


def make_live_like_override_map() -> dict[tuple[str, str], StaticLLMAdapter]:
    contradiction = HeuristicBrainLayerEvalAdapter()
    natural = HeuristicNaturalConversationAdapter()

    def contradiction_handler(messages: list[ModelMessage]) -> ModelResponse:
        response = contradiction.generate(messages, model="ignored")
        return ModelResponse(
            content=response.content,
            model="test-live-model",
            finish_reason="stop",
            usage={"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15},
        )

    def natural_handler(messages: list[ModelMessage]) -> ModelResponse:
        response = natural.generate(messages, model="ignored")
        return ModelResponse(
            content=response.content,
            model="test-live-model",
            finish_reason="stop",
            usage={"prompt_tokens": 17, "completion_tokens": 6, "total_tokens": 23},
        )

    return {
        ("live-like", "contradiction"): StaticLLMAdapter(handler=contradiction_handler),
        ("live-like", "natural"): StaticLLMAdapter(handler=natural_handler),
    }


class ModelMatrixTests(unittest.TestCase):
    def test_load_model_matrix_entries_from_sample_config(self) -> None:
        entries = load_model_matrix_entries(ROOT / "examples" / "model_matrix.sample.json")

        self.assertEqual([entry.name for entry in entries], ["heuristic-baseline", "heuristic-repeat"])
        self.assertTrue(all(entry.mode == "heuristic" for entry in entries))

    def test_matrix_runner_combines_both_suites_for_multiple_entries(self) -> None:
        entries = load_model_matrix_entries(ROOT / "examples" / "model_matrix.sample.json")
        results = run_model_matrix(entries, include_ablations=False)

        self.assertEqual(len(results), 38)
        leaderboard = build_matrix_leaderboard(results)
        self.assertEqual(len(leaderboard), 2)
        self.assertTrue(all(row.overall_passed == 19 for row in leaderboard))
        self.assertTrue(all(row.natural_extraction_passed == 6 for row in leaderboard))

    def test_matrix_runner_supports_hard_scenario_pack(self) -> None:
        entries = load_model_matrix_entries(ROOT / "examples" / "model_matrix.sample.json")
        results = run_model_matrix(entries, include_ablations=False, scenario_pack="hard")

        self.assertEqual(len(results), 28)
        leaderboard = build_matrix_leaderboard(results)
        self.assertEqual(len(leaderboard), 2)
        self.assertTrue(all(row.overall_passed == 14 for row in leaderboard))
        self.assertTrue(all(row.natural_extraction_passed == 5 for row in leaderboard))

    def test_matrix_export_writes_results_summary_leaderboard_history_and_x_post(self) -> None:
        entries = load_model_matrix_entries(ROOT / "examples" / "model_matrix.sample.json")
        results = run_model_matrix(entries, include_ablations=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            export_root = Path(tmpdir) / "exports"
            run_dir = export_model_matrix_results(
                results,
                export_root,
                include_ablations=False,
                label="smoke",
            )

            self.assertTrue((run_dir / "results.json").exists())
            self.assertTrue((run_dir / "results.csv").exists())
            self.assertTrue((run_dir / "summary.csv").exists())
            self.assertTrue((run_dir / "leaderboard.csv").exists())
            self.assertTrue((run_dir / "x_post.txt").exists())
            self.assertTrue((run_dir / "case_artifacts").exists())
            self.assertTrue((export_root / "matrix_history.csv").exists())
            self.assertTrue((export_root / "matrix_history.jsonl").exists())

            payload = json.loads((run_dir / "results.json").read_text())
            self.assertEqual(payload["metadata"]["label"], "smoke")
            self.assertFalse(payload["metadata"]["include_ablations"])
            self.assertIn("BrainLayer matrix", payload["x_post"])
            self.assertIn("score", payload["results"][0])
            self.assertIn("score_method", payload["results"][0])
            self.assertIn("artifact_path", payload["results"][0])

            artifact = json.loads((run_dir / payload["results"][0]["artifact_path"]).read_text())
            self.assertIn("prompt_messages", artifact)
            self.assertIn("judge", artifact)
            self.assertIn("exported_state", artifact)

    def test_matrix_cli_reports_leaderboard(self) -> None:
        completed = subprocess.run(
            [
                "python3",
                str(ROOT / "scripts" / "run_model_matrix.py"),
                "--config",
                str(ROOT / "examples" / "model_matrix.sample.json"),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn("BrainLayer Model Matrix Report", completed.stdout)
        self.assertIn("heuristic-baseline / model_loop", completed.stdout)

    def test_matrix_dump_states_writes_exported_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dump_dir = Path(tmpdir) / "states"
            completed = subprocess.run(
                [
                    "python3",
                    str(ROOT / "scripts" / "run_model_matrix.py"),
                    "--config",
                    str(ROOT / "examples" / "model_matrix.sample.json"),
                    "--suite",
                    "contradiction",
                    "--dump-states",
                    str(dump_dir),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertIn("Matrix state dumps written", completed.stdout)
            state_path = (
                dump_dir
                / "heuristic-baseline__contradiction__model_loop__model_goal_revision__revised_goal.json"
            )
            payload = json.loads(state_path.read_text())
            self.assertIn("working_state", payload)
            self.assertIn("episodes", payload)

    def test_matrix_runner_preserves_live_like_metadata_through_overrides(self) -> None:
        live_entry_payload = [
            {
                "name": "live-like",
                "mode": "live",
                "provider_name": "test_provider",
                "model": "test-live-model",
                "enabled": True,
            }
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "matrix.json"
            config_path.write_text(json.dumps({"entries": live_entry_payload}, indent=2) + "\n")
            entries = load_model_matrix_entries(config_path)

        results = run_model_matrix(
            entries,
            include_ablations=False,
            adapter_overrides=make_live_like_override_map(),
        )

        self.assertTrue(all(result.eval_mode == "live" for result in results))
        self.assertTrue(all(result.provider_name == "test_provider" for result in results))
        self.assertTrue(all(result.response_model == "test-live-model" for result in results))

    def test_matrix_pricing_estimates_cost_when_entry_rates_are_present(self) -> None:
        live_entry_payload = [
            {
                "name": "live-priced",
                "mode": "live",
                "provider_name": "test_provider",
                "model": "test-live-model",
                "input_cost_per_1k_tokens": 0.001,
                "output_cost_per_1k_tokens": 0.002,
                "enabled": True,
            }
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "matrix.json"
            config_path.write_text(json.dumps({"entries": live_entry_payload}, indent=2) + "\n")
            entries = load_model_matrix_entries(config_path)

            results = run_model_matrix(
                entries,
                include_ablations=False,
                adapter_overrides={
                    ("live-priced", "contradiction"): make_live_like_override_map()[
                        ("live-like", "contradiction")
                    ],
                    ("live-priced", "natural"): make_live_like_override_map()[
                        ("live-like", "natural")
                    ],
                },
            )
            leaderboard = build_matrix_leaderboard(results)

        self.assertTrue(all(result.estimated_cost_usd > 0.0 for result in results))
        self.assertEqual(len(leaderboard), 1)
        self.assertGreater(leaderboard[0].estimated_total_cost_usd, 0.0)
        self.assertGreater(leaderboard[0].avg_metrics.get("estimated_cost_usd", 0.0), 0.0)


if __name__ == "__main__":
    unittest.main()
