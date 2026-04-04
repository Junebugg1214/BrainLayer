import json
import unittest
from pathlib import Path

from brainlayer.llm import StaticLLMAdapter
from brainlayer.model_matrix import load_model_matrix_entries
from brainlayer.preflight import render_model_preflight, resolve_matrix_entry, run_model_preflight


ROOT = Path(__file__).resolve().parent.parent


class PreflightTests(unittest.TestCase):
    def test_resolve_matrix_entry_selects_named_entry(self) -> None:
        entry = resolve_matrix_entry(
            ROOT / "examples" / "model_matrix.anthropic.core.live.json",
            entry_name="anthropic-claude-haiku-4.5",
        )

        self.assertEqual(entry.name, "anthropic-claude-haiku-4.5")
        self.assertEqual(entry.provider_name, "anthropic_messages")

    def test_run_model_preflight_uses_adapter_and_renders_summary(self) -> None:
        entry = load_model_matrix_entries(ROOT / "examples" / "model_matrix.sample.json")[0]
        adapter = StaticLLMAdapter(
            response=json.dumps(
                {
                    "assistant_response": "OK",
                    "memory_observations": [],
                }
            )
        )

        result = run_model_preflight(
            entry,
            adapter=adapter,
            prompt="Return exactly the word OK.",
        )

        self.assertEqual(result.entry_name, entry.name)
        self.assertEqual(result.requested_model, entry.requested_model)
        self.assertTrue(result.output_preview)
        rendered = render_model_preflight(result)
        self.assertIn("BrainLayer Model Preflight", rendered)
        self.assertIn(entry.name, rendered)


if __name__ == "__main__":
    unittest.main()
