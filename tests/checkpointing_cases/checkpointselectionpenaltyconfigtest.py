from __future__ import annotations

from .shared import *


class CheckpointSelectionPenaltyConfigTest(unittest.TestCase):
    def test_direct_penalty_config_normalizes_mode_and_weights(self) -> None:
        """
        Verify that CheckpointSelectionConfig normalizes the penalty mode and preserves penalty weights.

        Asserts that providing penalty_mode="direct" results in the enum value CheckpointPenaltyMode.DIRECT, that to_summary() reports "direct" for penalty_mode, and that override and dominance penalty weights are retained.
        """
        config = CheckpointSelectionConfig(
            metric="scenario_success_rate",
            override_penalty_weight=0.4,
            dominance_penalty_weight=0.2,
            penalty_mode="direct",
        )

        self.assertEqual(config.penalty_mode, CheckpointPenaltyMode.DIRECT)
        self.assertEqual(config.to_summary()["penalty_mode"], "direct")
        self.assertAlmostEqual(config.override_penalty_weight, 0.4)
        self.assertAlmostEqual(config.dominance_penalty_weight, 0.2)

    def test_direct_mode_penalizes_high_override_with_same_success_rate(self) -> None:
        low_override = {
            "scenario_success_rate": 0.8,
            "episode_success_rate": 0.6,
            "mean_reward": 1.0,
            "evaluation_summary": {
                "mean_final_reflex_override_rate": 0.1,
                "mean_reflex_dominance": 0.0,
            },
            "episode": 1,
        }
        high_override = {
            "scenario_success_rate": 0.8,
            "episode_success_rate": 0.6,
            "mean_reward": 1.0,
            "evaluation_summary": {
                "mean_final_reflex_override_rate": 0.7,
                "mean_reflex_dominance": 0.0,
            },
            "episode": 2,
        }
        config = CheckpointSelectionConfig(
            metric="scenario_success_rate",
            override_penalty_weight=1.0,
            penalty_mode=CheckpointPenaltyMode.DIRECT,
        )

        self.assertGreater(
            checkpoint_candidate_sort_key(
                low_override,
                selection_config=config,
            ),
            checkpoint_candidate_sort_key(
                high_override,
                selection_config=config,
            ),
        )

    def test_tiebreaker_mode_preserves_legacy_six_tuple(self) -> None:
        candidate = {
            "scenario_success_rate": 0.5,
            "episode_success_rate": 0.4,
            "mean_reward": 1.0,
            "evaluation_summary": {
                "mean_final_reflex_override_rate": 0.2,
                "mean_reflex_dominance": 0.3,
            },
            "episode": 7,
        }

        key = checkpoint_candidate_sort_key(
            candidate,
            selection_config=CheckpointSelectionConfig(
                metric="scenario_success_rate",
                penalty_mode=CheckpointPenaltyMode.TIEBREAKER,
            ),
        )

        self.assertEqual(key, (0.5, 0.4, 1.0, -0.2, -0.3, 7))

    def test_direct_mode_composite_score_uses_penalty_formula(self) -> None:
        candidate = {
            "scenario_success_rate": 0.8,
            "episode_success_rate": 0.4,
            "mean_reward": 1.0,
            "evaluation_summary": {
                "mean_final_reflex_override_rate": 0.25,
                "mean_reflex_dominance": 0.5,
            },
            "episode": 7,
        }

        key = checkpoint_candidate_sort_key(
            candidate,
            selection_config=CheckpointSelectionConfig(
                metric="scenario_success_rate",
                override_penalty_weight=0.4,
                dominance_penalty_weight=0.2,
                penalty_mode=CheckpointPenaltyMode.DIRECT,
            ),
        )

        self.assertAlmostEqual(key[0], 0.8 - (0.4 * 0.25) - (0.2 * 0.5))
        self.assertEqual(key[1:4], (0.8, 0.4, 1.0))
