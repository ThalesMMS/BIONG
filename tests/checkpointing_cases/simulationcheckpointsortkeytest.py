from __future__ import annotations

from .shared import *


class SimulationCheckpointSortKeyTest(unittest.TestCase):
    """Tests for checkpoint_candidate_sort_key()."""

    def test_invalid_metric_raises_value_error(self) -> None:
        candidate = {
            "scenario_success_rate": 0.5,
            "episode_success_rate": 0.5,
            "mean_reward": 0.5,
            "episode": 1,
        }
        with self.assertRaises(ValueError):
            checkpoint_candidate_sort_key(
                candidate, primary_metric="invalid_metric"
            )

    def test_tiebreaker_config_preserves_legacy_tuple(self) -> None:
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

    def test_direct_penalty_mode_prepends_composite_score(self) -> None:
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
        self.assertAlmostEqual(key[0], 0.6)
        self.assertEqual(key[1:4], (0.8, 0.4, 1.0))
        self.assertEqual(key[4:], (-0.25, -0.5, 7))

    def test_direct_penalty_can_override_primary_metric_gap(self) -> None:
        high_dependence = {
            "scenario_success_rate": 0.8,
            "episode_success_rate": 0.5,
            "mean_reward": 0.5,
            "evaluation_summary": {
                "mean_final_reflex_override_rate": 0.5,
                "mean_reflex_dominance": 0.0,
            },
            "episode": 1,
        }
        low_dependence = {
            "scenario_success_rate": 0.7,
            "episode_success_rate": 0.5,
            "mean_reward": 0.5,
            "evaluation_summary": {
                "mean_final_reflex_override_rate": 0.0,
                "mean_reflex_dominance": 0.0,
            },
            "episode": 2,
        }
        selection_config = CheckpointSelectionConfig(
            metric="scenario_success_rate",
            override_penalty_weight=1.0,
            penalty_mode=CheckpointPenaltyMode.DIRECT,
        )

        self.assertGreater(
            checkpoint_candidate_sort_key(
                low_dependence,
                selection_config=selection_config,
            ),
            checkpoint_candidate_sort_key(
                high_dependence,
                selection_config=selection_config,
            ),
        )

    def test_scenario_success_rate_primary_metric(self) -> None:
        high = {
            "scenario_success_rate": 0.9,
            "episode_success_rate": 0.1,
            "mean_reward": 0.1,
            "episode": 1,
        }
        low = {
            "scenario_success_rate": 0.1,
            "episode_success_rate": 0.9,
            "mean_reward": 0.9,
            "episode": 2,
        }
        key_high = checkpoint_candidate_sort_key(
            high, primary_metric="scenario_success_rate"
        )
        key_low = checkpoint_candidate_sort_key(
            low, primary_metric="scenario_success_rate"
        )
        self.assertGreater(key_high, key_low)

    def test_episode_success_rate_primary_metric(self) -> None:
        a = {
            "scenario_success_rate": 0.9,
            "episode_success_rate": 0.1,
            "mean_reward": 0.5,
            "episode": 1,
        }
        b = {
            "scenario_success_rate": 0.2,
            "episode_success_rate": 0.8,
            "mean_reward": 0.5,
            "episode": 2,
        }
        key_a = checkpoint_candidate_sort_key(
            a, primary_metric="episode_success_rate"
        )
        key_b = checkpoint_candidate_sort_key(
            b, primary_metric="episode_success_rate"
        )
        self.assertGreater(key_b, key_a)

    def test_episode_tiebreaker_favors_later_episode(self) -> None:
        # Same metric scores but different episode number
        early = {
            "scenario_success_rate": 0.5,
            "episode_success_rate": 0.5,
            "mean_reward": 0.5,
            "episode": 3,
        }
        late = {
            "scenario_success_rate": 0.5,
            "episode_success_rate": 0.5,
            "mean_reward": 0.5,
            "episode": 10,
        }
        key_early = checkpoint_candidate_sort_key(
            early, primary_metric="scenario_success_rate"
        )
        key_late = checkpoint_candidate_sort_key(
            late, primary_metric="scenario_success_rate"
        )
        self.assertGreater(key_late, key_early)

    def test_reflex_dependence_penalty_precedes_episode_tiebreaker(self) -> None:
        low_reflex = {
            "scenario_success_rate": 0.5,
            "episode_success_rate": 0.5,
            "mean_reward": 0.5,
            "evaluation_summary": {
                "mean_final_reflex_override_rate": 0.1,
                "mean_reflex_dominance": 0.1,
            },
            "episode": 3,
        }
        high_reflex = {
            "scenario_success_rate": 0.5,
            "episode_success_rate": 0.5,
            "mean_reward": 0.5,
            "evaluation_summary": {
                "mean_final_reflex_override_rate": 0.9,
                "mean_reflex_dominance": 0.9,
            },
            "episode": 10,
        }
        key_low_reflex = checkpoint_candidate_sort_key(
            low_reflex, primary_metric="scenario_success_rate"
        )
        key_high_reflex = checkpoint_candidate_sort_key(
            high_reflex, primary_metric="scenario_success_rate"
        )
        self.assertGreater(key_low_reflex, key_high_reflex)

    def test_high_reflex_override_rate_lowers_checkpoint_ranking(self) -> None:
        low_override = {
            "scenario_success_rate": 0.5,
            "episode_success_rate": 0.5,
            "mean_reward": 0.5,
            "evaluation_summary": {
                "mean_final_reflex_override_rate": 0.0,
                "mean_reflex_dominance": 0.1,
            },
            "episode": 5,
        }
        high_override = {
            "scenario_success_rate": 0.5,
            "episode_success_rate": 0.5,
            "mean_reward": 0.5,
            "evaluation_summary": {
                "mean_final_reflex_override_rate": 0.8,
                "mean_reflex_dominance": 0.1,
            },
            "episode": 5,
        }

        self.assertGreater(
            checkpoint_candidate_sort_key(
                low_override,
                primary_metric="scenario_success_rate",
            ),
            checkpoint_candidate_sort_key(
                high_override,
                primary_metric="scenario_success_rate",
            ),
        )

    def test_high_reflex_dominance_lowers_checkpoint_ranking(self) -> None:
        low_dominance = {
            "scenario_success_rate": 0.5,
            "episode_success_rate": 0.5,
            "mean_reward": 0.5,
            "evaluation_summary": {
                "mean_final_reflex_override_rate": 0.1,
                "mean_reflex_dominance": 0.0,
            },
            "episode": 5,
        }
        high_dominance = {
            "scenario_success_rate": 0.5,
            "episode_success_rate": 0.5,
            "mean_reward": 0.5,
            "evaluation_summary": {
                "mean_final_reflex_override_rate": 0.1,
                "mean_reflex_dominance": 0.8,
            },
            "episode": 5,
        }

        self.assertGreater(
            checkpoint_candidate_sort_key(
                low_dominance,
                primary_metric="scenario_success_rate",
            ),
            checkpoint_candidate_sort_key(
                high_dominance,
                primary_metric="scenario_success_rate",
            ),
        )

    def test_sort_key_is_comparable_for_checkpoint_ranking(self) -> None:
        lower_success = {
            "scenario_success_rate": 0.6,
            "episode_success_rate": 0.6,
            "mean_reward": 0.6,
            "episode": 5,
        }
        higher_success = {
            "scenario_success_rate": 0.7,
            "episode_success_rate": 0.6,
            "mean_reward": 0.6,
            "episode": 5,
        }
        lower_key = checkpoint_candidate_sort_key(
            lower_success,
            primary_metric="scenario_success_rate",
        )
        higher_key = checkpoint_candidate_sort_key(
            higher_success,
            primary_metric="scenario_success_rate",
        )
        self.assertLess(lower_key, higher_key)

    def test_missing_metric_keys_default_to_zero(self) -> None:
        missing_metrics = {"episode": 1}
        explicit_zero_metrics = {
            "scenario_success_rate": 0.0,
            "episode_success_rate": 0.0,
            "mean_reward": 0.0,
            "episode": 1,
        }
        positive_metric = {
            "scenario_success_rate": 0.0,
            "episode_success_rate": 0.0,
            "mean_reward": 0.1,
            "episode": 1,
        }

        missing_key = checkpoint_candidate_sort_key(
            missing_metrics,
            primary_metric="mean_reward",
        )
        explicit_zero_key = checkpoint_candidate_sort_key(
            explicit_zero_metrics,
            primary_metric="mean_reward",
        )
        positive_key = checkpoint_candidate_sort_key(
            positive_metric,
            primary_metric="mean_reward",
        )
        self.assertEqual(missing_key, explicit_zero_key)
        self.assertLess(missing_key, positive_key)

    def test_checkpoint_metric_changes_selection_priority(self) -> None:
        candidate_a = {
            "scenario_success_rate": 0.9,
            "episode_success_rate": 0.1,
            "mean_reward": 0.2,
            "episode": 1,
        }
        candidate_b = {
            "scenario_success_rate": 0.2,
            "episode_success_rate": 0.8,
            "mean_reward": 0.7,
            "episode": 2,
        }

        scenario_choice = max(
            [candidate_a, candidate_b],
            key=lambda item: checkpoint_candidate_sort_key(
                item,
                primary_metric="scenario_success_rate",
            ),
        )
        reward_choice = max(
            [candidate_a, candidate_b],
            key=lambda item: checkpoint_candidate_sort_key(
                item,
                primary_metric="mean_reward",
            ),
        )

        self.assertIs(scenario_choice, candidate_a)
        self.assertIs(reward_choice, candidate_b)
