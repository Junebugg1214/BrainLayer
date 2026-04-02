import unittest

from brainlayer.judging import (
    BehaviorJudgeInput,
    HeuristicBehaviorJudge,
    score_structured_value,
)


class JudgingTests(unittest.TestCase):
    def test_behavior_judge_accepts_confirm_github_authentication(self) -> None:
        decision = HeuristicBehaviorJudge().score(
            BehaviorJudgeInput(
                scenario_slug="",
                scenario_title="",
                scenario_description="",
                checkpoint="",
                prompt="",
                expected="check authentication",
                actual="Confirm GitHub authentication first.",
            )
        )

        self.assertTrue(decision.passed)

    def test_behavior_judge_accepts_deliver_eval_report_as_ship_eval_report(self) -> None:
        decision = HeuristicBehaviorJudge().score(
            BehaviorJudgeInput(
                scenario_slug="",
                scenario_title="",
                scenario_description="",
                checkpoint="",
                prompt="",
                expected="ship eval report",
                actual="The main goal right now is to deliver the evaluation report tonight due to a deadline change.",
            )
        )

        self.assertTrue(decision.passed)

    def test_structured_value_accepts_shipping_eval_summary_today(self) -> None:
        decision = score_structured_value(
            "ship eval summary",
            "shipping the evaluation summary today",
            target_layer="working_state",
            target_key="primary_goal",
        )

        self.assertTrue(decision.passed)
        self.assertEqual(decision.method, "structural_semantic_match")


if __name__ == "__main__":
    unittest.main()
