from __future__ import annotations

from .shared import *


class SimulationMeanRewardFromPayloadTest(unittest.TestCase):
    """Tests for mean_reward_from_behavior_payload()."""

    def test_empty_payload_returns_zero(self) -> None:
        result = mean_reward_from_behavior_payload({})
        self.assertEqual(result, 0.0)

    def test_empty_legacy_scenarios_returns_zero(self) -> None:
        result = mean_reward_from_behavior_payload(
            {"legacy_scenarios": {}}
        )
        self.assertEqual(result, 0.0)

    def test_non_dict_legacy_scenarios_returns_zero(self) -> None:
        result = mean_reward_from_behavior_payload(
            {"legacy_scenarios": None}
        )
        self.assertEqual(result, 0.0)

    def test_single_scenario_mean_reward(self) -> None:
        payload = {
            "legacy_scenarios": {
                "night_rest": {"mean_reward": 2.5},
            }
        }
        result = mean_reward_from_behavior_payload(payload)
        self.assertAlmostEqual(result, 2.5)

    def test_multiple_scenarios_averaged(self) -> None:
        payload = {
            "legacy_scenarios": {
                "night_rest": {"mean_reward": 1.0},
                "food_deprivation": {"mean_reward": 3.0},
            }
        }
        result = mean_reward_from_behavior_payload(payload)
        self.assertAlmostEqual(result, 2.0)

    def test_non_dict_scenario_data_skipped(self) -> None:
        payload = {
            "legacy_scenarios": {
                "night_rest": {"mean_reward": 4.0},
                "bad_entry": "not_a_dict",
            }
        }
        result = mean_reward_from_behavior_payload(payload)
        self.assertAlmostEqual(result, 4.0)

    def test_missing_mean_reward_key_defaults_to_zero(self) -> None:
        payload = {
            "legacy_scenarios": {
                "night_rest": {"other_key": 99},
            }
        }
        result = mean_reward_from_behavior_payload(payload)
        self.assertAlmostEqual(result, 0.0)
