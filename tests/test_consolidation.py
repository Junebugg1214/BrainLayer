import tempfile
import unittest
from pathlib import Path

from brainlayer import BrainLayerSession, ConsolidationConfig
from brainlayer.storage import load_state


class ConsolidationEngineTests(unittest.TestCase):
    def test_repeated_preference_hints_promote_to_belief(self) -> None:
        session = BrainLayerSession(auto_consolidate=False)
        session.observe(
            text="Across several turns, requests keep shrinking from full draft to short answer.",
            memory_type="preference_hint",
            payload={
                "key": "response_style",
                "value": "concise",
                "proposition": "The user likely prefers concise responses.",
            },
            salience=0.4,
        )
        session.observe(
            text="The latest edits consistently cut long drafts down to the short version.",
            memory_type="preference_hint",
            payload={
                "key": "response_style",
                "value": "concise",
                "proposition": "The user likely prefers concise responses.",
            },
            salience=0.42,
        )

        report = session.consolidate()

        active_beliefs = [belief for belief in session.state.beliefs if belief.status == "active"]
        self.assertEqual(len(active_beliefs), 1)
        self.assertEqual(active_beliefs[0].key, "response_style")
        self.assertEqual(active_beliefs[0].value, "concise")
        self.assertIn("response_style", report.promoted_belief_keys)

    def test_forgetting_drops_low_salience_noise_and_pauses_excess_working_items(self) -> None:
        session = BrainLayerSession(
            auto_consolidate=False,
            consolidation_config=ConsolidationConfig(max_active_working_items=2),
        )
        session.observe(
            text="Primary goal: preserve citations in every answer.",
            memory_type="goal",
            payload={
                "key": "primary_goal",
                "value": "preserve citations",
                "summary": "The current primary goal is to preserve citations.",
            },
            salience=0.95,
        )
        session.observe(
            text="Secondary goal: keep answers concise.",
            memory_type="goal",
            payload={
                "key": "secondary_goal",
                "value": "keep answers concise",
                "summary": "A secondary goal is to keep answers concise.",
            },
            salience=0.7,
        )
        session.observe(
            text="Tertiary goal: mention prior benchmarks.",
            memory_type="goal",
            payload={
                "key": "tertiary_goal",
                "value": "mention prior benchmarks",
                "summary": "A tertiary goal is to mention prior benchmarks.",
            },
            salience=0.36,
        )
        session.observe(
            text="The color note says the table border should be gray.",
            memory_type="noise",
            payload={"value": "gray border"},
            salience=0.12,
        )

        report = session.consolidate()

        active_items = [item for item in session.state.working_state if item.status == "active"]
        paused_items = [item for item in session.state.working_state if item.status == "paused"]
        self.assertEqual(len(active_items), 2)
        self.assertEqual(len(paused_items), 1)
        self.assertEqual(len(report.forgotten_episode_ids), 1)
        self.assertEqual(len(report.paused_working_item_ids), 1)
        self.assertFalse(
            any("noise" in episode.tags for episode in session.state.episodes if episode.salience < 0.3)
        )

    def test_session_round_trip_preserves_consolidated_state(self) -> None:
        session = BrainLayerSession(auto_consolidate=False)
        session.observe(
            text="Before retrying a release, confirm GitHub auth first.",
            memory_type="lesson_hint",
            payload={
                "trigger": "retry_release",
                "action": "check authentication",
                "summary": "Before retrying a release, confirm GitHub auth first.",
            },
            salience=0.41,
        )
        session.observe(
            text="The retry rule remains: check auth before trying the release again.",
            memory_type="lesson_hint",
            payload={
                "trigger": "retry_release",
                "action": "check authentication",
                "summary": "Before retrying a release, confirm GitHub auth first.",
            },
            salience=0.45,
        )
        session.consolidate()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "consolidated.json"
            session.save(path)
            loaded = load_state(path)
            self.assertEqual(len(loaded.procedures), 1)
            self.assertEqual(loaded.procedures[0].trigger, "retry_release")
            self.assertEqual(loaded.procedures[0].steps[0], "check authentication")


if __name__ == "__main__":
    unittest.main()
