import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from brainlayer.llm import StaticLLMAdapter
from brainlayer.runtime import BrainLayerRuntime
from brainlayer.scenarios import Observation
from brainlayer.session import BrainLayerSession


ROOT = Path(__file__).resolve().parent.parent


class BrainLayerRuntimeTests(unittest.TestCase):
    def test_runtime_retrieves_state_and_records_turn(self) -> None:
        session = BrainLayerSession()
        session.observe(
            text="The user prefers concise and direct replies.",
            memory_type="preference",
            payload={
                "key": "response_style",
                "value": "concise",
                "proposition": "The user prefers concise and direct replies.",
            },
            salience=0.95,
        )
        session.observe(
            text="Primary goal for this task: preserve citations.",
            memory_type="goal",
            payload={
                "key": "primary_goal",
                "value": "preserve citations",
                "summary": "The current primary goal is to preserve citations.",
            },
            salience=0.9,
        )

        adapter = StaticLLMAdapter(
            response=json.dumps(
                {
                    "assistant_response": "I will reply concisely and preserve citations.",
                    "episodic_summary": "The assistant committed to a concise, citation-preserving reply.",
                    "memory_observations": [],
                }
            )
        )
        runtime = BrainLayerRuntime(adapter, session=session)
        result = runtime.run_turn("Draft the reply for the user.")

        self.assertEqual(
            result.assistant_response,
            "I will reply concisely and preserve citations.",
        )
        retrieved_layers = {memory.layer for memory in result.retrieved_memories}
        self.assertIn("working_state", retrieved_layers)
        self.assertIn("beliefs", retrieved_layers)
        self.assertTrue(
            any(
                episode.summary == "The assistant committed to a concise, citation-preserving reply."
                for episode in session.state.episodes
            )
        )

    def test_runtime_applies_model_generated_memory_updates(self) -> None:
        session = BrainLayerSession()
        runtime = BrainLayerRuntime(
            StaticLLMAdapter(
                response=json.dumps(
                    {
                        "assistant_response": "I will keep the answer brief.",
                        "episodic_summary": "The assistant aligned with a concise style.",
                        "memory_observations": [
                            {
                                "text": "The user prefers brief default replies.",
                                "memory_type": "preference",
                                "salience": 0.93,
                                "payload": {
                                    "key": "response_style",
                                    "value": "brief",
                                    "proposition": "The user prefers brief default replies.",
                                },
                            }
                        ],
                    }
                )
            ),
            session=session,
        )

        result = runtime.run_turn("How should you respond going forward?")

        self.assertEqual(len(result.applied_observations), 1)
        self.assertTrue(
            any(
                belief.key == "response_style" and belief.value == "brief"
                for belief in session.state.beliefs
            )
        )

    def test_runtime_skips_invalid_model_generated_memory_updates(self) -> None:
        session = BrainLayerSession()
        runtime = BrainLayerRuntime(
            StaticLLMAdapter(
                response=json.dumps(
                    {
                        "assistant_response": "I will keep the answer brief.",
                        "episodic_summary": "The assistant tried to store a malformed update.",
                        "memory_observations": [
                            {
                                "text": "Malformed preference update.",
                                "memory_type": "preference",
                                "salience": 0.9,
                                "payload": {
                                    "value": "brief",
                                    "proposition": "The user prefers brief replies.",
                                },
                            }
                        ],
                    }
                )
            ),
            session=session,
        )

        result = runtime.run_turn("How should you respond going forward?")

        self.assertEqual(len(result.applied_observations), 0)
        self.assertFalse(session.state.beliefs)

    def test_runtime_rejects_malformed_preference_hint_payloads(self) -> None:
        session = BrainLayerSession()
        runtime = BrainLayerRuntime(
            StaticLLMAdapter(
                response=json.dumps(
                    {
                        "assistant_response": "Noted.",
                        "episodic_summary": "The assistant saw a malformed hint payload.",
                        "memory_observations": [
                            {
                                "memory_type": "preference_hint",
                                "salience": 0.4,
                                "payload": {
                                    "value": "concise",
                                    "proposition": "The user likely prefers concise replies.",
                                },
                            }
                        ],
                    }
                )
            ),
            session=session,
        )

        result = runtime.run_turn("Record a likely response style preference.")

        self.assertEqual(len(result.applied_observations), 0)
        self.assertFalse(session.state.beliefs)

    def test_runtime_falls_back_to_plain_text_model_output(self) -> None:
        runtime = BrainLayerRuntime(StaticLLMAdapter(response="Plain text answer."))

        result = runtime.run_turn("Say something useful.")

        self.assertEqual(result.assistant_response, "Plain text answer.")
        self.assertFalse(result.applied_observations)

    def test_cli_runs_in_dry_run_mode_and_persists_state(self) -> None:
        response = json.dumps(
            {
                "assistant_response": "Keep it concise.",
                "episodic_summary": "The assistant chose a concise answer.",
                "memory_observations": [
                    {
                        "memory_type": "preference",
                        "salience": 0.9,
                        "payload": {
                            "key": "response_style",
                            "value": "concise",
                            "proposition": "The user prefers concise replies.",
                        },
                    }
                ],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "live_state.json"
            completed = subprocess.run(
                [
                    "python3",
                    str(ROOT / "scripts" / "run_model_loop.py"),
                    "--prompt",
                    "What style should you use?",
                    "--state",
                    str(state_path),
                    "--observe-file",
                    str(ROOT / "examples" / "live_turn_observations.sample.json"),
                    "--dry-run-response",
                    response,
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertIn("Assistant response:", completed.stdout)
            self.assertTrue(state_path.exists())
            payload = json.loads(state_path.read_text())
            self.assertTrue(payload["beliefs"])


if __name__ == "__main__":
    unittest.main()
