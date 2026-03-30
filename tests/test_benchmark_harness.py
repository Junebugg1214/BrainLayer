import json
import unittest
from pathlib import Path

from brainlayer.benchmark_harness import run_suite


ROOT = Path(__file__).resolve().parent.parent


class BenchmarkHarnessTests(unittest.TestCase):
    def test_brainlayer_agent_solves_all_three_seed_experiments(self) -> None:
        results = run_suite()
        brainlayer_results = [result for result in results if result.agent_name == "brainlayer"]
        self.assertEqual(len(brainlayer_results), 3)
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


if __name__ == "__main__":
    unittest.main()
