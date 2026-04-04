import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from brainlayer.llm import LLMAdapter, LLMError, ModelMessage, ModelResponse, StaticLLMAdapter
from brainlayer.natural_eval import (
    HeuristicNaturalConversationAdapter,
    NaturalEvalScenario,
    NaturalEvalTurn,
    default_natural_eval_runtime_config,
    export_natural_eval_results,
    run_natural_eval_suite,
)
from brainlayer.model_eval import RUNTIME_PROFILE_STUDY_V2


ROOT = Path(__file__).resolve().parent.parent


def make_live_like_adapter() -> StaticLLMAdapter:
    heuristic = HeuristicNaturalConversationAdapter()

    def handler(messages: list[ModelMessage]) -> ModelResponse:
        response = heuristic.generate(messages, model="ignored")
        return ModelResponse(
            content=response.content,
            model="test-live-model",
            finish_reason="stop",
            usage={"prompt_tokens": 18, "completion_tokens": 6, "total_tokens": 24},
        )

    return StaticLLMAdapter(handler=handler)


def make_semantic_natural_adapter() -> StaticLLMAdapter:
    def handler(messages: list[ModelMessage]) -> ModelResponse:
        user_message = messages[-1].content if messages else ""
        if "I'm skimming between meetings" in user_message:
            return ModelResponse(
                content=json.dumps(
                    {
                        "assistant_response": "I'll keep it concise.",
                        "episodic_summary": "The user signaled a short-answer preference.",
                        "memory_observations": [
                            {
                                "text": "The user prefers concise replies.",
                                "memory_type": "preference",
                                "payload": {
                                    "key": "response_style",
                                    "value": "concise",
                                    "proposition": "The user prefers concise replies.",
                                },
                                "salience": 0.95,
                            }
                        ],
                    }
                ),
                model="test-semantic-natural-model",
                finish_reason="stop",
            )
        if "How should you answer by default right now?" in user_message:
            return ModelResponse(
                content=json.dumps(
                    {
                        "assistant_response": "concise",
                        "episodic_summary": "Answered with a semantic paraphrase of brief.",
                        "memory_observations": [],
                    }
                ),
                model="test-semantic-natural-model",
                finish_reason="stop",
            )
        return ModelResponse(
            content='{"assistant_response":"unknown","episodic_summary":"unknown","memory_observations":[]}',
            model="test-semantic-natural-model",
            finish_reason="stop",
        )

    return StaticLLMAdapter(handler=handler)


class FailingAdapter(LLMAdapter):
    def generate(
        self,
        messages: list[ModelMessage],
        *,
        model: str,
        temperature: float = 0.2,
        max_output_tokens: int = 900,
    ) -> ModelResponse:
        del messages, model, temperature, max_output_tokens
        raise LLMError("boom")


class NaturalEvalTests(unittest.TestCase):
    def test_natural_eval_prompt_mentions_canonical_brainlayer_keys(self) -> None:
        prompt = default_natural_eval_runtime_config().system_prompt

        self.assertIn("key=response_style", prompt)
        self.assertIn("key=primary_goal", prompt)
        self.assertIn("key=collaboration_mode", prompt)
        self.assertIn("trigger=retry_release", prompt)
        self.assertIn("action=check authentication", prompt)

    def test_full_natural_suite_passes_with_heuristic_adapter(self) -> None:
        results = run_natural_eval_suite(include_ablations=False)
        model_loop_results = [result for result in results if result.runtime_name == "model_loop"]
        self.assertEqual(len(model_loop_results), 11)
        self.assertTrue(all(result.passed for result in model_loop_results))

    def test_hard_natural_suite_passes_with_heuristic_adapter(self) -> None:
        results = run_natural_eval_suite(include_ablations=False, scenario_pack="hard")
        model_loop_results = [result for result in results if result.runtime_name == "model_loop"]
        self.assertEqual(len(model_loop_results), 10)
        self.assertTrue(all(result.passed for result in model_loop_results))

    def test_held_out_natural_suite_passes_with_heuristic_adapter(self) -> None:
        results = run_natural_eval_suite(include_ablations=False, scenario_pack="held_out")
        model_loop_results = [result for result in results if result.runtime_name == "model_loop"]
        self.assertEqual(len(model_loop_results), 10)
        self.assertTrue(all(result.passed for result in model_loop_results))

    def test_external_dev_natural_suite_passes_with_heuristic_adapter(self) -> None:
        results = run_natural_eval_suite(include_ablations=False, scenario_pack="external_dev")
        model_loop_results = [result for result in results if result.runtime_name == "model_loop"]
        self.assertEqual(len(model_loop_results), 16)
        self.assertTrue(all(result.passed for result in model_loop_results))

    def test_external_held_out_natural_suite_passes_with_heuristic_adapter(self) -> None:
        results = run_natural_eval_suite(
            include_ablations=False,
            scenario_pack="external_held_out",
        )
        model_loop_results = [result for result in results if result.runtime_name == "model_loop"]
        self.assertEqual(len(model_loop_results), 16)
        self.assertTrue(all(result.passed for result in model_loop_results))

    def test_consolidation_stress_natural_suite_passes_with_heuristic_adapter(self) -> None:
        results = run_natural_eval_suite(
            include_ablations=False,
            scenario_pack="consolidation_stress",
        )
        model_loop_results = [result for result in results if result.runtime_name == "model_loop"]
        self.assertEqual(len(model_loop_results), 8)
        self.assertTrue(all(result.passed for result in model_loop_results))

    def test_forgetting_stress_natural_suite_passes_with_heuristic_adapter(self) -> None:
        results = run_natural_eval_suite(
            include_ablations=False,
            scenario_pack="forgetting_stress",
        )
        model_loop_results = [result for result in results if result.runtime_name == "model_loop"]
        self.assertEqual(len(model_loop_results), 8)
        self.assertTrue(all(result.passed for result in model_loop_results))

    def test_study_v2_runtime_profile_exposes_stronger_baselines(self) -> None:
        results = run_natural_eval_suite(
            include_ablations=False,
            runtime_profile=RUNTIME_PROFILE_STUDY_V2,
        )

        runtime_names = {result.runtime_name for result in results}
        self.assertEqual(
            runtime_names,
            {
                "brainlayer_full",
                "context_only",
                "naive_retrieval",
                "structured_no_consolidation",
                "summary_state",
            },
        )

        def lookup(runtime_name: str, scenario_slug: str, checkpoint: str) -> object:
            for result in results:
                if (
                    result.runtime_name == runtime_name
                    and result.scenario_slug == scenario_slug
                    and result.checkpoint == checkpoint
                ):
                    return result
            self.fail(f"Missing result for {runtime_name} on {scenario_slug}/{checkpoint}")

        self.assertTrue(
            lookup("brainlayer_full", "natural_goal_shift", "extract_revised_goal").passed
        )
        self.assertTrue(
            lookup("structured_no_consolidation", "natural_goal_shift", "extract_revised_goal").passed
        )
        self.assertFalse(
            lookup("context_only", "natural_goal_shift", "extract_revised_goal").passed
        )

    def test_study_v2_runtime_profile_with_ablations_exposes_brainlayer_component_variants(self) -> None:
        results = run_natural_eval_suite(
            include_ablations=True,
            runtime_profile=RUNTIME_PROFILE_STUDY_V2,
        )

        runtime_names = {result.runtime_name for result in results}
        self.assertEqual(
            runtime_names,
            {
                "brainlayer_full",
                "brainlayer_no_autobio",
                "brainlayer_no_consolidation",
                "brainlayer_no_forgetting",
                "brainlayer_no_working_state",
                "context_only",
                "naive_retrieval",
                "structured_no_consolidation",
                "summary_state",
            },
        )

    def test_natural_ablations_show_targeted_regressions(self) -> None:
        results = run_natural_eval_suite()

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
                "natural_hint_accumulation",
                "extract_hint_consolidation",
            ).passed
        )
        self.assertFalse(
            lookup(
                "model_loop_no_autobio",
                "natural_relationship_reframe",
                "extract_relationship",
            ).passed
        )
        self.assertFalse(
            lookup(
                "model_loop_no_working_state",
                "natural_goal_shift",
                "extract_revised_goal",
            ).passed
        )

    def test_hard_natural_pack_shows_targeted_regressions(self) -> None:
        results = run_natural_eval_suite(scenario_pack="hard")

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
                "natural_delayed_hint_accumulation",
                "extract_hint_consolidation",
            ).passed
        )
        self.assertFalse(
            lookup(
                "model_loop_no_autobio",
                "natural_collaboration_reframe_after_noise",
                "extract_relationship",
            ).passed
        )
        self.assertFalse(
            lookup(
                "model_loop_no_working_state",
                "natural_long_horizon_goal_shift",
                "extract_revised_goal",
            ).passed
        )

    def test_consolidation_stress_natural_pack_shows_no_consolidation_regression(self) -> None:
        results = run_natural_eval_suite(scenario_pack="consolidation_stress")

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
                "consolidation_stress_natural_goal_hints",
                "extract_goal",
            ).passed
        )
        self.assertFalse(
            lookup(
                "model_loop_no_consolidation",
                "consolidation_stress_natural_lesson_hints",
                "extract_lesson",
            ).passed
        )

    def test_natural_eval_script_reports_summary(self) -> None:
        completed = subprocess.run(
            ["python3", str(ROOT / "scripts" / "run_natural_model_evals.py"), "--core-only"],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn("Natural Conversation BrainLayer Eval Report", completed.stdout)
        self.assertIn("model_loop: 11/11", completed.stdout)

    def test_natural_eval_script_reports_hard_pack_summary(self) -> None:
        completed = subprocess.run(
            [
                "python3",
                str(ROOT / "scripts" / "run_natural_model_evals.py"),
                "--core-only",
                "--scenario-pack",
                "hard",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn("Natural Conversation BrainLayer Eval Report", completed.stdout)
        self.assertIn("model_loop: 10/10", completed.stdout)

    def test_natural_eval_script_reports_held_out_pack_summary(self) -> None:
        completed = subprocess.run(
            [
                "python3",
                str(ROOT / "scripts" / "run_natural_model_evals.py"),
                "--core-only",
                "--scenario-pack",
                "held_out",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn("Natural Conversation BrainLayer Eval Report", completed.stdout)
        self.assertIn("model_loop: 10/10", completed.stdout)

    def test_export_natural_eval_results_writes_csv_json_history_and_x_post(self) -> None:
        results = run_natural_eval_suite(include_ablations=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            export_root = Path(tmpdir) / "exports"
            run_dir = export_natural_eval_results(
                results,
                export_root,
                include_ablations=False,
                label="smoke",
            )

            self.assertTrue((run_dir / "results.json").exists())
            self.assertTrue((run_dir / "results.csv").exists())
            self.assertTrue((run_dir / "summary.csv").exists())
            self.assertTrue((run_dir / "x_post.txt").exists())
            self.assertTrue((run_dir / "case_artifacts").exists())
            self.assertTrue((export_root / "natural_eval_history.csv").exists())
            self.assertTrue((export_root / "natural_eval_history.jsonl").exists())

            payload = json.loads((run_dir / "results.json").read_text())
            self.assertEqual(payload["metadata"]["label"], "smoke")
            self.assertFalse(payload["metadata"]["include_ablations"])
            self.assertIn("BrainLayer natural eval", payload["x_post"])
            self.assertIn("score", payload["results"][0])
            self.assertIn("score_method", payload["results"][0])
            self.assertIn("artifact_path", payload["results"][0])

            artifact = json.loads((run_dir / payload["results"][0]["artifact_path"]).read_text())
            self.assertIn("prompt_messages", artifact)
            self.assertIn("retrieved_memories", artifact)
            self.assertIn("raw_model_output", artifact)
            self.assertIn("judge", artifact)
            self.assertIn("exported_state", artifact)

    def test_judge_backed_scoring_handles_structural_and_behavior_paraphrases(self) -> None:
        scenario = NaturalEvalScenario(
            slug="natural_semantic_preference",
            title="Natural Semantic Preference",
            description="Extraction and behavior scoring should accept semantic paraphrases.",
            turns=[
                NaturalEvalTurn(
                    prompt="I'm skimming between meetings, so please keep this really brief.",
                    checkpoint="extract_preference",
                    evaluation_type="extraction",
                    target_layer="beliefs",
                    target_key="response_style",
                    expected_value="brief",
                ),
                NaturalEvalTurn(
                    prompt="How should you answer by default right now?",
                    checkpoint="behavior_preference",
                    evaluation_type="behavior",
                    expected_value="brief",
                ),
            ],
        )

        judged_results = run_natural_eval_suite(
            [scenario],
            include_ablations=False,
            adapter=make_semantic_natural_adapter(),
            behavior_scoring_mode="judge",
        )
        exact_results = run_natural_eval_suite(
            [scenario],
            include_ablations=False,
            adapter=make_semantic_natural_adapter(),
            behavior_scoring_mode="exact",
        )

        extraction_result = next(
            result for result in judged_results if result.evaluation_type == "extraction"
        )
        behavior_result = next(
            result for result in judged_results if result.evaluation_type == "behavior"
        )
        exact_behavior_result = next(
            result for result in exact_results if result.evaluation_type == "behavior"
        )

        self.assertTrue(extraction_result.passed)
        self.assertEqual(extraction_result.score_method, "structural_semantic_match")
        self.assertTrue(behavior_result.passed)
        self.assertEqual(behavior_result.score_method, "heuristic_semantic_judge")
        self.assertFalse(exact_behavior_result.passed)
        self.assertEqual(exact_behavior_result.score_method, "exact_match")

    def test_live_like_natural_mode_records_metadata_and_usage(self) -> None:
        results = run_natural_eval_suite(
            include_ablations=False,
            adapter=make_live_like_adapter(),
            eval_mode="live",
            provider_name="test_provider",
            requested_model="test-live-model",
        )

        self.assertTrue(all(result.eval_mode == "live" for result in results))
        self.assertTrue(all(result.provider_name == "test_provider" for result in results))
        self.assertTrue(all(result.response_model == "test-live-model" for result in results))
        self.assertTrue(all(result.usage_metrics.get("total_tokens") == 24.0 for result in results))

    def test_natural_live_mode_gracefully_records_runtime_errors(self) -> None:
        scenario = NaturalEvalScenario(
            slug="natural_failure",
            title="Natural Failure",
            description="Error handling smoke test for natural eval mode.",
            turns=[
                NaturalEvalTurn(
                    prompt="I'm skimming between meetings, so please keep this really brief.",
                ),
                NaturalEvalTurn(
                    prompt="How should you answer by default right now?",
                    checkpoint="after_failure",
                    evaluation_type="behavior",
                    expected_value="brief",
                ),
            ],
        )

        results = run_natural_eval_suite(
            [scenario],
            include_ablations=False,
            adapter=FailingAdapter(),
            eval_mode="live",
            provider_name="test_provider",
            requested_model="broken-model",
        )

        self.assertEqual(len(results), 1)
        self.assertFalse(results[0].passed)
        self.assertTrue(results[0].skipped)
        self.assertIn("boom", results[0].error)


if __name__ == "__main__":
    unittest.main()
