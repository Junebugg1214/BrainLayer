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

    def test_runtime_normalizes_flattened_preference_observations(self) -> None:
        session = BrainLayerSession()
        runtime = BrainLayerRuntime(
            StaticLLMAdapter(
                response=json.dumps(
                    {
                        "assistant_response": "Got it.",
                        "episodic_summary": "The assistant inferred a brief style preference.",
                        "memory_observations": [
                            {
                                "memory_type": "preference",
                                "key": "response_style",
                                "value": "very brief",
                                "proposition": "User prefers very brief answers when multitasking between meetings.",
                            }
                        ],
                    }
                )
            ),
            session=session,
        )

        result = runtime.run_turn("I'm skimming between meetings, so please keep this really brief.")

        self.assertEqual(len(result.applied_observations), 1)
        self.assertEqual(result.applied_observations[0].payload["value"], "brief")
        self.assertTrue(
            any(
                belief.key == "response_style" and belief.value == "brief"
                for belief in session.state.beliefs
            )
        )

    def test_runtime_infers_missing_memory_type_from_flattened_preference(self) -> None:
        session = BrainLayerSession()
        runtime = BrainLayerRuntime(
            StaticLLMAdapter(
                response=json.dumps(
                    {
                        "assistant_response": "Understood! I'll keep it brief.",
                        "episodic_summary": "User prefers brief responses while skimming between meetings.",
                        "memory_observations": [
                            {
                                "key": "response_style",
                                "value": "brief",
                                "proposition": "User prefers brief responses.",
                            }
                        ],
                    }
                )
            ),
            session=session,
        )

        result = runtime.run_turn("I'm skimming between meetings, so please keep this really brief.")

        self.assertEqual(len(result.applied_observations), 1)
        self.assertEqual(result.applied_observations[0].memory_type, "preference")
        self.assertTrue(
            any(
                belief.key == "response_style" and belief.value == "brief"
                for belief in session.state.beliefs
            )
        )

    def test_runtime_infers_missing_memory_type_from_flattened_lesson(self) -> None:
        session = BrainLayerSession()
        runtime = BrainLayerRuntime(
            StaticLLMAdapter(
                response=json.dumps(
                    {
                        "assistant_response": "Next time, I'll check GitHub authentication before retrying the release.",
                        "episodic_summary": "User emphasized checking GitHub auth before retrying a release.",
                        "memory_observations": [
                            {
                                "trigger": "retry_release",
                                "action": "check authentication",
                                "summary": "User wants to ensure GitHub authentication is checked before retrying a release.",
                            }
                        ],
                    }
                )
            ),
            session=session,
        )

        result = runtime.run_turn(
            "Last time the release failed because we retried before checking GitHub auth. Next time, check auth first."
        )

        self.assertEqual(len(result.applied_observations), 1)
        self.assertEqual(result.applied_observations[0].memory_type, "lesson")
        self.assertTrue(
            any(
                procedure.trigger == "retry_release" and procedure.steps[0] == "check authentication"
                for procedure in session.state.procedures
            )
        )

    def test_runtime_normalizes_goal_aliases_into_primary_goal(self) -> None:
        session = BrainLayerSession()
        runtime = BrainLayerRuntime(
            StaticLLMAdapter(
                response=json.dumps(
                    {
                        "assistant_response": "Understood.",
                        "episodic_summary": "The assistant inferred a citation-preservation goal.",
                        "memory_observations": [
                            {
                                "memory_type": "goal",
                                "key": "citation_integrity",
                                "value": "preserve",
                                "summary": "User wants all answers to keep citations intact.",
                            }
                        ],
                    }
                )
            ),
            session=session,
        )

        result = runtime.run_turn(
            "Before anything else, let's make sure every answer keeps the citations intact."
        )

        self.assertEqual(len(result.applied_observations), 1)
        self.assertEqual(result.applied_observations[0].payload["key"], "primary_goal")
        self.assertEqual(result.applied_observations[0].payload["value"], "preserve citations")
        self.assertTrue(
            any(
                item.key == "primary_goal" and item.value == "preserve citations"
                for item in session.state.working_state
            )
        )

    def test_runtime_normalizes_coinvestigator_relationship_value(self) -> None:
        session = BrainLayerSession()
        runtime = BrainLayerRuntime(
            StaticLLMAdapter(
                response=json.dumps(
                    {
                        "assistant_response": "I'll work with you as a co-investigator.",
                        "episodic_summary": "The user wants a co-investigator instead of a contractor.",
                        "memory_observations": [
                            {
                                "memory_type": "relationship",
                                "payload": {
                                    "key": "collaboration_mode",
                                    "value": "co-investigator",
                                    "summary": "The user wants a co-investigator on the memory study.",
                                    "themes": ["collaboration", "investigation"],
                                },
                            }
                        ],
                    }
                )
            ),
            session=session,
        )

        result = runtime.run_turn(
            "Don't behave like a contractor taking tickets. I need a co-investigator on the memory study."
        )

        self.assertEqual(len(result.applied_observations), 1)
        self.assertEqual(result.applied_observations[0].payload["value"], "research partner")
        self.assertTrue(
            any(
                note.key == "collaboration_mode" and note.value == "research partner"
                for note in session.state.autobiographical_state
            )
        )

    def test_runtime_normalizes_terser_preference_value(self) -> None:
        session = BrainLayerSession()
        runtime = BrainLayerRuntime(
            StaticLLMAdapter(
                response=json.dumps(
                    {
                        "assistant_response": "Understood.",
                        "episodic_summary": "The user wants an even terser response style.",
                        "memory_observations": [
                            {
                                "memory_type": "preference",
                                "payload": {
                                    "key": "response_style",
                                    "value": "terser",
                                    "proposition": "The user wants responses to be even terser.",
                                },
                            }
                        ],
                    }
                )
            ),
            session=session,
        )

        result = runtime.run_turn("You can make it even terser than that.")

        self.assertEqual(len(result.applied_observations), 1)
        self.assertEqual(result.applied_observations[0].payload["value"], "concise")
        self.assertTrue(
            any(
                belief.key == "response_style" and belief.value == "concise"
                for belief in session.state.beliefs
            )
        )

    def test_runtime_normalizes_rollout_auth_lesson_trigger(self) -> None:
        session = BrainLayerSession()
        runtime = BrainLayerRuntime(
            StaticLLMAdapter(
                response=json.dumps(
                    {
                        "assistant_response": "I'll verify the login before rerunning rollouts.",
                        "episodic_summary": "The user explained that a rollout slipped because auth expired.",
                        "memory_observations": [
                            {
                                "memory_type": "lesson",
                                "payload": {
                                    "trigger": "rollout slipped",
                                    "action": "check authentication",
                                    "summary": (
                                        "The last rollout slipped because GitHub auth had expired. "
                                        "Next time, verify the login first."
                                    ),
                                },
                            }
                        ],
                    }
                )
            ),
            session=session,
        )

        result = runtime.run_turn(
            "The last rollout slipped because we reran before checking whether GitHub auth had expired. "
            "Next time verify the login first."
        )

        self.assertEqual(len(result.applied_observations), 1)
        self.assertEqual(result.applied_observations[0].payload["trigger"], "retry_release")
        self.assertTrue(
            any(
                procedure.trigger == "retry_release" and procedure.steps[0] == "check authentication"
                for procedure in session.state.procedures
            )
        )

    def test_runtime_recovers_missing_preference_observation_from_turn_content(self) -> None:
        session = BrainLayerSession()
        session.observe(
            text="The user prefers brief replies.",
            memory_type="preference",
            payload={
                "key": "response_style",
                "value": "brief",
                "proposition": "The user prefers brief replies.",
            },
            salience=0.9,
        )
        runtime = BrainLayerRuntime(
            StaticLLMAdapter(
                response=json.dumps(
                    {
                        "assistant_response": (
                            "For the methods memo, I will provide a detailed explanation of the reasoning "
                            "behind each method and the rationale for its selection."
                        ),
                        "episodic_summary": "User requested full reasoning for methods memo.",
                        "memory_observations": [],
                    }
                )
            ),
            session=session,
        )

        result = runtime.run_turn(
            "For the methods memo though, I need the full reasoning spelled out."
        )

        self.assertEqual(len(result.applied_observations), 1)
        self.assertEqual(result.applied_observations[0].payload["value"], "detailed")
        self.assertTrue(
            any(
                belief.key == "response_style" and belief.value == "detailed"
                for belief in session.state.beliefs
            )
        )

    def test_runtime_prefers_revised_goal_when_summary_mentions_old_and_new_goal(self) -> None:
        session = BrainLayerSession()
        runtime = BrainLayerRuntime(
            StaticLLMAdapter(
                response=json.dumps(
                    {
                        "assistant_response": "Understood. The primary goal is now to ship the evaluation summary today.",
                        "episodic_summary": "User updated primary goal to prioritize shipping the evaluation summary today.",
                        "memory_observations": [
                            {
                                "memory_type": "goal",
                                "key": "primary_goal",
                                "value": "ship evaluation summary today",
                                "summary": "User changed main goal from preserving citations to shipping the eval summary today.",
                            }
                        ],
                    }
                )
            ),
            session=session,
        )

        result = runtime.run_turn(
            "Actually the deadline moved up, so the main thing now is shipping the eval summary today."
        )

        self.assertEqual(len(result.applied_observations), 1)
        self.assertEqual(result.applied_observations[0].payload["value"], "ship eval summary")
        self.assertTrue(
            any(
                item.key == "primary_goal" and item.value == "ship eval summary"
                for item in session.state.working_state
            )
        )

    def test_runtime_normalizes_detailed_reasoning_goal_value(self) -> None:
        session = BrainLayerSession()
        runtime = BrainLayerRuntime(
            StaticLLMAdapter(
                response=json.dumps(
                    {
                        "assistant_response": "Understood.",
                        "episodic_summary": "The appendix defense requires the full reasoning chain.",
                        "memory_observations": [
                            {
                                "memory_type": "goal",
                                "payload": {
                                    "key": "primary_goal",
                                    "value": "provide the full chain of reasoning",
                                    "summary": "The appendix defense needs the full chain of reasoning.",
                                },
                            }
                        ],
                    }
                )
            ),
            session=session,
        )

        result = runtime.run_turn("For the appendix defense, I need the whole chain of reasoning.")

        self.assertEqual(len(result.applied_observations), 1)
        self.assertEqual(result.applied_observations[0].payload["value"], "provide detailed reasoning")
        self.assertTrue(
            any(
                item.key == "primary_goal" and item.value == "provide detailed reasoning"
                for item in session.state.working_state
            )
        )

    def test_runtime_adds_style_override_when_detailed_goal_is_active(self) -> None:
        session = BrainLayerSession()
        session.observe(
            text="The user prefers brief replies.",
            memory_type="preference",
            payload={
                "key": "response_style",
                "value": "brief",
                "proposition": "The user prefers brief replies.",
            },
            salience=0.9,
        )
        session.observe(
            text="The appendix defense requires detailed reasoning.",
            memory_type="goal",
            payload={
                "key": "primary_goal",
                "value": "provide detailed reasoning",
                "summary": "The current primary goal is to provide detailed reasoning.",
            },
            salience=0.92,
        )

        runtime = BrainLayerRuntime(
            StaticLLMAdapter(
                response=json.dumps(
                    {
                        "assistant_response": "Detailed.",
                        "episodic_summary": "The assistant used the active detailed-reasoning override.",
                        "memory_observations": [],
                    }
                )
            ),
            session=session,
        )

        result = runtime.run_turn("How should you answer by default right now?")

        self.assertIn("[derived_override] response_style = detailed.", result.prompt_messages[1].content)
        self.assertIn("overrides older brief defaults", result.prompt_messages[1].content)

    def test_runtime_normalizes_relationship_key_aliases(self) -> None:
        session = BrainLayerSession()
        runtime = BrainLayerRuntime(
            StaticLLMAdapter(
                response=json.dumps(
                    {
                        "assistant_response": "Understood.",
                        "episodic_summary": "The assistant inferred the collaboration framing.",
                        "memory_observations": [
                            {
                                "memory_type": "relationship",
                                "key": "collaboration_style",
                                "value": "research partner",
                                "summary": "The user wants a research partner, not a task runner.",
                                "themes": ["relationship", "research-mode"],
                            }
                        ],
                    }
                )
            ),
            session=session,
        )

        result = runtime.run_turn(
            "I don't just need a task runner here; think with me like a research partner on this."
        )

        self.assertEqual(len(result.applied_observations), 1)
        self.assertEqual(result.applied_observations[0].payload["key"], "collaboration_mode")
        self.assertEqual(result.applied_observations[0].payload["value"], "research partner")
        self.assertTrue(
            any(
                note.key == "collaboration_mode" and note.value == "research partner"
                for note in session.state.autobiographical_state
            )
        )

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
