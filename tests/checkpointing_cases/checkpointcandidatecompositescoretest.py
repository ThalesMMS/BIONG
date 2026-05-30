from __future__ import annotations

from .shared import *


class CheckpointCandidateCompositeScoreTest(unittest.TestCase):
    """Tests for checkpoint_candidate_composite_score()."""

    def test_zero_penalty_weights_returns_primary_metric_value(self) -> None:
        candidate = {
            "scenario_success_rate": 0.75,
            "episode_success_rate": 0.5,
            "mean_reward": 1.0,
            "evaluation_summary": {
                "mean_final_reflex_override_rate": 0.5,
                "mean_reflex_dominance": 0.5,
            },
            "episode": 5,
        }
        config = CheckpointSelectionConfig(
            metric="scenario_success_rate",
            override_penalty_weight=0.0,
            dominance_penalty_weight=0.0,
            penalty_mode=CheckpointPenaltyMode.TIEBREAKER,
        )
        score = checkpoint_candidate_composite_score(candidate, config)
        self.assertAlmostEqual(score, 0.75)

    def test_composite_score_penalizes_override_rate(self) -> None:
        candidate = {
            "scenario_success_rate": 0.8,
            "episode_success_rate": 0.5,
            "mean_reward": 1.0,
            "evaluation_summary": {
                "mean_final_reflex_override_rate": 0.4,
                "mean_reflex_dominance": 0.0,
            },
            "episode": 5,
        }
        config = CheckpointSelectionConfig(
            metric="scenario_success_rate",
            override_penalty_weight=0.5,
            dominance_penalty_weight=0.0,
            penalty_mode=CheckpointPenaltyMode.TIEBREAKER,
        )
        # 0.8 - (0.5 * 0.4) = 0.8 - 0.2 = 0.6
        score = checkpoint_candidate_composite_score(candidate, config)
        self.assertAlmostEqual(score, 0.6)

    def test_composite_score_penalizes_dominance(self) -> None:
        candidate = {
            "scenario_success_rate": 0.9,
            "episode_success_rate": 0.5,
            "mean_reward": 1.0,
            "evaluation_summary": {
                "mean_final_reflex_override_rate": 0.0,
                "mean_reflex_dominance": 0.5,
            },
            "episode": 5,
        }
        config = CheckpointSelectionConfig(
            metric="scenario_success_rate",
            override_penalty_weight=0.0,
            dominance_penalty_weight=0.2,
            penalty_mode=CheckpointPenaltyMode.TIEBREAKER,
        )
        # 0.9 - (0.2 * 0.5) = 0.9 - 0.1 = 0.8
        score = checkpoint_candidate_composite_score(candidate, config)
        self.assertAlmostEqual(score, 0.8)

    def test_composite_score_uses_primary_metric_regardless_of_mode(self) -> None:
        candidate = {
            "scenario_success_rate": 0.6,
            "episode_success_rate": 0.9,
            "mean_reward": 1.5,
            "evaluation_summary": {},
            "episode": 1,
        }
        config_tiebreaker = CheckpointSelectionConfig(
            metric="episode_success_rate",
            penalty_mode=CheckpointPenaltyMode.TIEBREAKER,
        )
        config_direct = CheckpointSelectionConfig(
            metric="episode_success_rate",
            penalty_mode=CheckpointPenaltyMode.DIRECT,
        )
        # Both configs should produce composite score based on episode_success_rate
        score_tb = checkpoint_candidate_composite_score(candidate, config_tiebreaker)
        score_direct = checkpoint_candidate_composite_score(candidate, config_direct)
        self.assertAlmostEqual(score_tb, 0.9)
        self.assertAlmostEqual(score_direct, 0.9)

    def test_tiebreaker_mode_is_overridden_for_composite_calculation(self) -> None:
        """composite_score always uses direct penalty mode internally."""
        candidate = {
            "scenario_success_rate": 0.7,
            "episode_success_rate": 0.5,
            "mean_reward": 1.0,
            "evaluation_summary": {
                "mean_final_reflex_override_rate": 0.3,
                "mean_reflex_dominance": 0.1,
            },
            "episode": 1,
        }
        config = CheckpointSelectionConfig(
            metric="scenario_success_rate",
            override_penalty_weight=1.0,
            dominance_penalty_weight=1.0,
            penalty_mode=CheckpointPenaltyMode.TIEBREAKER,
        )
        # 0.7 - (1.0 * 0.3) - (1.0 * 0.1) = 0.7 - 0.3 - 0.1 = 0.3
        score = checkpoint_candidate_composite_score(candidate, config)
        self.assertAlmostEqual(score, 0.3)
