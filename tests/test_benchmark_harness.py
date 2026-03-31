import json
import tempfile
import unittest
from pathlib import Path

from brainlayer.benchmark_harness import export_results, run_suite
from brainlayer.session import BrainLayerSession
from brainlayer.storage import load_state, save_state
from brainlayer.validation import BrainLayerValidationError, validate_state_dict


ROOT = Path(__file__).resolve().parent.parent


class BenchmarkHarnessTests(unittest.TestCase):
    def test_brainlayer_agent_solves_all_seed_experiments(self) -> None:
        results = run_suite()
        brainlayer_results = [result for result in results if result.agent_name == "brainlayer"]
        self.assertEqual(len(brainlayer_results), 12)
        self.assertTrue(all(result.passed for result in brainlayer_results))

    def test_ablation_variants_show_targeted_regressions(self) -> None:
        results = run_suite()

        def lookup(agent_name: str, scenario_slug: str, checkpoint: str = "final") -> object:
            for result in results:
                if (
                    result.agent_name == agent_name
                    and result.scenario_slug == scenario_slug
                    and result.checkpoint == checkpoint
                ):
                    return result
            self.fail(f"Missing result for {agent_name} on {scenario_slug}/{checkpoint}")

        self.assertFalse(lookup("brainlayer_no_consolidation", "hint_consolidation").passed)
        self.assertFalse(lookup("brainlayer_no_autobio", "autobio_continuity").passed)
        self.assertFalse(lookup("brainlayer_no_working_state", "goal_focus").passed)
        self.assertFalse(
            lookup(
                "brainlayer_no_consolidation",
                "long_horizon_project_reuse",
                "midpoint",
            ).passed
        )
        self.assertFalse(
            lookup(
                "brainlayer_no_autobio",
                "long_horizon_collaboration_continuity",
                "late_frame",
            ).passed
        )

    def test_no_forgetting_retains_more_state_than_full_brainlayer(self) -> None:
        results = run_suite()

        full_result = None
        no_forgetting_result = None
        for result in results:
            if (
                result.agent_name == "brainlayer"
                and result.scenario_slug == "long_horizon_project_reuse"
                and result.checkpoint == "late_recall"
            ):
                full_result = result
            if (
                result.agent_name == "brainlayer_no_forgetting"
                and result.scenario_slug == "long_horizon_project_reuse"
                and result.checkpoint == "late_recall"
            ):
                no_forgetting_result = result

        self.assertIsNotNone(full_result)
        self.assertIsNotNone(no_forgetting_result)
        self.assertLess(
            full_result.state_metrics["episodes"],
            no_forgetting_result.state_metrics["episodes"],
        )

    def test_long_horizon_checkpoints_land_correctly(self) -> None:
        results = run_suite(include_ablations=False)
        lookup = {
            (result.scenario_slug, result.checkpoint, result.agent_name): result
            for result in results
        }

        self.assertTrue(
            lookup[
                ("long_horizon_preference_revision", "midpoint", "brainlayer")
            ].passed
        )
        self.assertTrue(
            lookup[
                ("long_horizon_preference_revision", "final_revision", "brainlayer")
            ].passed
        )
        self.assertTrue(
            lookup[
                ("long_horizon_project_reuse", "late_recall", "brainlayer")
            ].passed
        )
        self.assertTrue(
            lookup[
                ("long_horizon_collaboration_continuity", "late_frame", "brainlayer")
            ].passed
        )

    def test_export_results_writes_csv_json_history_and_x_post(self) -> None:
        results = run_suite(include_ablations=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            export_root = Path(tmpdir) / "exports"
            run_dir = export_results(
                results,
                export_root,
                include_ablations=False,
                label="smoke",
            )

            self.assertTrue((run_dir / "results.json").exists())
            self.assertTrue((run_dir / "results.csv").exists())
            self.assertTrue((run_dir / "summary.csv").exists())
            self.assertTrue((run_dir / "x_post.txt").exists())
            self.assertTrue((export_root / "history.csv").exists())
            self.assertTrue((export_root / "history.jsonl").exists())

            payload = json.loads((run_dir / "results.json").read_text())
            self.assertEqual(payload["metadata"]["label"], "smoke")
            self.assertFalse(payload["metadata"]["include_ablations"])
            self.assertIn("BrainLayer eval", payload["x_post"])

    def test_brainlayer_state_schema_lists_all_layers(self) -> None:
        schema_path = ROOT / "schemas" / "brainlayer-state.schema.json"
        schema = json.loads(schema_path.read_text())
        required = set(schema["required"])
        self.assertEqual(
            required,
            {
                "working_state",
                "episodes",
                "beliefs",
                "autobiographical_state",
                "procedures",
            },
        )

    def test_state_round_trip_loads_and_saves_cleanly(self) -> None:
        session = BrainLayerSession()
        session.observe(
            text="Primary goal for this task: preserve source citations in every answer.",
            memory_type="goal",
            payload={
                "key": "primary_goal",
                "value": "preserve source citations",
                "summary": "The current primary goal is to preserve source citations in every answer.",
            },
            salience=0.9,
        )
        session.observe(
            text="Update that framing: we are research partners exploring BrainLayer together.",
            memory_type="relationship",
            payload={
                "key": "collaboration_mode",
                "value": "research partner",
                "summary": "The collaboration mode is research partner.",
                "themes": "relationship,research-mode",
            },
            salience=0.95,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "state.json"
            save_state(session.state, path)
            loaded = load_state(path)
            self.assertEqual(loaded.to_dict(), session.state.to_dict())

    def test_invalid_state_is_rejected(self) -> None:
        invalid_payload = {
            "working_state": [],
            "episodes": [],
            "beliefs": [],
            "autobiographical_state": [],
        }
        with self.assertRaises(BrainLayerValidationError):
            validate_state_dict(invalid_payload)


if __name__ == "__main__":
    unittest.main()
