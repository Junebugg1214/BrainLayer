import json
import tempfile
import subprocess
import unittest
from pathlib import Path

from brainlayer.model_eval import export_model_eval_results, run_model_eval_suite


ROOT = Path(__file__).resolve().parent.parent


class ModelEvalTests(unittest.TestCase):
    def test_full_model_loop_passes_contradiction_suite(self) -> None:
        results = run_model_eval_suite(include_ablations=False)
        model_loop_results = [result for result in results if result.runtime_name == "model_loop"]
        self.assertEqual(len(model_loop_results), 8)
        self.assertTrue(all(result.passed for result in model_loop_results))

    def test_ablations_show_targeted_runtime_regressions(self) -> None:
        results = run_model_eval_suite()

        def lookup(runtime_name: str, scenario_slug: str, checkpoint: str) -> object:
            for result in results:
                if (
                    result.runtime_name == runtime_name
                    and result.scenario_slug == scenario_slug
                    and result.checkpoint == checkpoint
                ):
                    return result
            self.fail(f"Missing result for {runtime_name} on {scenario_slug}/{checkpoint}")

        self.assertFalse(
            lookup(
                "model_loop_no_consolidation",
                "model_hint_then_correction",
                "hint_consolidated",
            ).passed
        )
        self.assertFalse(
            lookup(
                "model_loop_no_working_state",
                "model_goal_revision",
                "initial_goal",
            ).passed
        )
        self.assertFalse(
            lookup(
                "model_loop_no_working_state",
                "model_goal_revision",
                "revised_goal",
            ).passed
        )
        self.assertFalse(
            lookup(
                "model_loop_no_autobio",
                "model_relationship_revision",
                "initial_frame",
            ).passed
        )
        self.assertFalse(
            lookup(
                "model_loop_no_autobio",
                "model_relationship_revision",
                "revised_frame",
            ).passed
        )

    def test_no_forgetting_retains_more_state_than_full_model_loop(self) -> None:
        results = run_model_eval_suite()

        full_result = None
        no_forgetting_result = None
        for result in results:
            if (
                result.runtime_name == "model_loop"
                and result.scenario_slug == "model_hint_then_correction"
                and result.checkpoint == "post_correction"
            ):
                full_result = result
            if (
                result.runtime_name == "model_loop_no_forgetting"
                and result.scenario_slug == "model_hint_then_correction"
                and result.checkpoint == "post_correction"
            ):
                no_forgetting_result = result

        self.assertIsNotNone(full_result)
        self.assertIsNotNone(no_forgetting_result)
        self.assertLess(
            full_result.state_metrics["episodes"],
            no_forgetting_result.state_metrics["episodes"],
        )

    def test_model_eval_script_reports_summary(self) -> None:
        completed = subprocess.run(
            ["python3", str(ROOT / "scripts" / "run_model_evals.py"), "--core-only"],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn("Model-Backed BrainLayer Eval Report", completed.stdout)
        self.assertIn("model_loop: 8/8", completed.stdout)

    def test_export_model_eval_results_writes_csv_json_history_and_x_post(self) -> None:
        results = run_model_eval_suite(include_ablations=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            export_root = Path(tmpdir) / "exports"
            run_dir = export_model_eval_results(
                results,
                export_root,
                include_ablations=False,
                label="smoke",
            )

            self.assertTrue((run_dir / "results.json").exists())
            self.assertTrue((run_dir / "results.csv").exists())
            self.assertTrue((run_dir / "summary.csv").exists())
            self.assertTrue((run_dir / "x_post.txt").exists())
            self.assertTrue((export_root / "model_eval_history.csv").exists())
            self.assertTrue((export_root / "model_eval_history.jsonl").exists())

            payload = json.loads((run_dir / "results.json").read_text())
            self.assertEqual(payload["metadata"]["label"], "smoke")
            self.assertFalse(payload["metadata"]["include_ablations"])
            self.assertIn("BrainLayer model-loop eval", payload["x_post"])


if __name__ == "__main__":
    unittest.main()
