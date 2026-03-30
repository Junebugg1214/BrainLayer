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
        self.assertEqual(len(brainlayer_results), 5)
        self.assertTrue(all(result.passed for result in brainlayer_results))

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
