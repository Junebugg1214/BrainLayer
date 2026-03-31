import json
import tempfile
import unittest
from pathlib import Path

from brainlayer.benchmark_harness import run_suite
from brainlayer.session import BrainLayerSession
from brainlayer.storage import load_state, save_state
from brainlayer.validation import BrainLayerValidationError, validate_state_dict


ROOT = Path(__file__).resolve().parent.parent


class BenchmarkHarnessTests(unittest.TestCase):
    def test_brainlayer_agent_solves_all_seed_experiments(self) -> None:
        results = run_suite()
        brainlayer_results = [result for result in results if result.agent_name == "brainlayer"]
        self.assertEqual(len(brainlayer_results), 6)
        self.assertTrue(all(result.passed for result in brainlayer_results))

    def test_ablation_variants_show_targeted_regressions(self) -> None:
        results = run_suite()

        def lookup(agent_name: str, scenario_slug: str) -> object:
            for result in results:
                if result.agent_name == agent_name and result.scenario_slug == scenario_slug:
                    return result
            self.fail(f"Missing result for {agent_name} on {scenario_slug}")

        self.assertFalse(lookup("brainlayer_no_consolidation", "hint_consolidation").passed)
        self.assertFalse(lookup("brainlayer_no_autobio", "autobio_continuity").passed)
        self.assertFalse(lookup("brainlayer_no_working_state", "goal_focus").passed)

    def test_no_forgetting_retains_more_state_than_full_brainlayer(self) -> None:
        results = run_suite()

        full_result = None
        no_forgetting_result = None
        for result in results:
            if result.agent_name == "brainlayer" and result.scenario_slug == "hint_consolidation":
                full_result = result
            if (
                result.agent_name == "brainlayer_no_forgetting"
                and result.scenario_slug == "hint_consolidation"
            ):
                no_forgetting_result = result

        self.assertIsNotNone(full_result)
        self.assertIsNotNone(no_forgetting_result)
        self.assertLess(
            full_result.state_metrics["episodes"],
            no_forgetting_result.state_metrics["episodes"],
        )

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
