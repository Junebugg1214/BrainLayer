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
    export_natural_eval_results,
    run_natural_eval_suite,
)


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
    def test_full_natural_suite_passes_with_heuristic_adapter(self) -> None:
        results = run_natural_eval_suite(include_ablations=False)
        model_loop_results = [result for result in results if result.runtime_name == "model_loop"]
        self.assertEqual(len(model_loop_results), 11)
        self.assertTrue(all(result.passed for result in model_loop_results))

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

    def test_natural_eval_script_reports_summary(self) -> None:
        completed = subprocess.run(
            ["python3", str(ROOT / "scripts" / "run_natural_model_evals.py"), "--core-only"],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn("Natural Conversation BrainLayer Eval Report", completed.stdout)
        self.assertIn("model_loop: 11/11", completed.stdout)

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
            self.assertTrue((export_root / "natural_eval_history.csv").exists())
            self.assertTrue((export_root / "natural_eval_history.jsonl").exists())

            payload = json.loads((run_dir / "results.json").read_text())
            self.assertEqual(payload["metadata"]["label"], "smoke")
            self.assertFalse(payload["metadata"]["include_ablations"])
            self.assertIn("BrainLayer natural eval", payload["x_post"])

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
