import sys
import tempfile
import unittest
from pathlib import Path

from brainlayer.repeat_runner import (
    build_model_eval_command,
    build_natural_eval_command,
    render_repeat_summary_markdown,
    run_logged_command,
)


class RepeatRunnerTests(unittest.TestCase):
    def test_build_natural_eval_command_includes_filters(self) -> None:
        command = build_natural_eval_command(
            mode="live",
            model="claude-sonnet-4-5-20250929",
            provider_name="anthropic_messages",
            base_url="https://api.anthropic.com",
            api_key_env="ANTHROPIC_API_KEY",
            request_path="/v1/messages",
            timeout_seconds=12.0,
            max_output_tokens_field="none",
            temperature=0.0,
            max_output_tokens=700,
            scenario_pack="forgetting_stress",
            runtime_profile="study_v2",
            export_results=Path("artifacts/natural_eval_runs"),
            label="repeat-check",
            scenario_slugs=["a", "b"],
            runtime_names=["brainlayer_full", "brainlayer_no_forgetting"],
        )

        self.assertIn("--scenario-slug", command)
        self.assertIn("--runtime-name", command)
        self.assertIn("brainlayer_full", command)
        self.assertIn("brainlayer_no_forgetting", command)
        self.assertIn("a", command)
        self.assertIn("b", command)

    def test_run_logged_command_times_out(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "repeat.log"
            result = run_logged_command(
                [
                    sys.executable,
                    "-c",
                    "import time; print('start', flush=True); time.sleep(2)",
                ],
                log_path=log_path,
                wall_clock_seconds=0.2,
            )

            self.assertTrue(result.timed_out)
            self.assertEqual(result.status, "timeout")
            self.assertTrue(log_path.exists())
            self.assertIn("start", log_path.read_text(encoding="utf-8"))

    def test_build_model_eval_command_includes_filters(self) -> None:
        command = build_model_eval_command(
            mode="live",
            model="claude-sonnet-4-5-20250929",
            provider_name="anthropic_messages",
            base_url="https://api.anthropic.com",
            api_key_env="ANTHROPIC_API_KEY",
            request_path="/v1/messages",
            timeout_seconds=12.0,
            max_output_tokens_field="none",
            temperature=0.0,
            max_output_tokens=400,
            scenario_pack="consolidation_stress",
            runtime_profile="study_v2",
            export_results=Path("artifacts/model_eval_repeat_runs"),
            label="repeat-check",
            scenario_slugs=["x"],
            runtime_names=["brainlayer_full", "brainlayer_no_consolidation"],
        )

        self.assertIn("--scenario-slug", command)
        self.assertIn("--runtime-name", command)
        self.assertIn("brainlayer_no_consolidation", command)
        self.assertIn("x", command)

    def test_render_repeat_summary_markdown(self) -> None:
        markdown = render_repeat_summary_markdown(
            [
                type(
                    "Stub",
                    (),
                    {
                        "repeat_index": 1,
                        "label": "demo-repeat1",
                        "status": "completed",
                        "duration_seconds": 12.3,
                        "exit_code": 0,
                        "timed_out": False,
                        "log_path": "/tmp/demo.log",
                        "run_dir": "/tmp/run",
                    },
                )()
            ]
        )

        self.assertIn("demo-repeat1", markdown)
        self.assertIn("/tmp/demo.log", markdown)
        self.assertIn("/tmp/run", markdown)
