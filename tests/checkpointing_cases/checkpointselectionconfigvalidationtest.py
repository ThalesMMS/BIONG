from __future__ import annotations

from .shared import *


class CheckpointSelectionConfigValidationTest(unittest.TestCase):
    """Tests for CheckpointSelectionConfig validation in __post_init__."""

    def test_invalid_metric_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            CheckpointSelectionConfig(metric="invalid_metric")

    def test_invalid_penalty_mode_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            CheckpointSelectionConfig(
                metric="scenario_success_rate",
                penalty_mode="bogus_mode",
            )

    def test_infinite_override_weight_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            CheckpointSelectionConfig(
                metric="scenario_success_rate",
                override_penalty_weight=float("inf"),
            )

    def test_negative_override_weight_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            CheckpointSelectionConfig(
                metric="scenario_success_rate",
                override_penalty_weight=-0.1,
            )

    def test_negative_dominance_weight_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            CheckpointSelectionConfig(
                metric="scenario_success_rate",
                dominance_penalty_weight=-1.0,
            )

    def test_nan_dominance_weight_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            CheckpointSelectionConfig(
                metric="scenario_success_rate",
                dominance_penalty_weight=float("nan"),
            )

    def test_all_valid_metrics_accepted(self) -> None:
        for metric in ("scenario_success_rate", "episode_success_rate", "mean_reward"):
            with self.subTest(metric=metric):
                config = CheckpointSelectionConfig(metric=metric)
                self.assertEqual(config.metric, metric)

    def test_to_summary_returns_all_expected_keys(self) -> None:
        config = CheckpointSelectionConfig(
            metric="episode_success_rate",
            override_penalty_weight=0.3,
            dominance_penalty_weight=0.1,
            penalty_mode="tiebreaker",
        )
        summary = config.to_summary()
        self.assertEqual(summary["metric"], "episode_success_rate")
        self.assertAlmostEqual(float(summary["override_penalty_weight"]), 0.3)
        self.assertAlmostEqual(float(summary["dominance_penalty_weight"]), 0.1)
        self.assertEqual(summary["penalty_mode"], "tiebreaker")

    def test_string_penalty_mode_normalized_to_enum(self) -> None:
        config = CheckpointSelectionConfig(
            metric="scenario_success_rate",
            penalty_mode="tiebreaker",
        )
        self.assertEqual(config.penalty_mode, CheckpointPenaltyMode.TIEBREAKER)

    def test_build_summary_rejects_mixed_config_and_legacy_kwargs(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not allow mixing"):
            build_checkpointing_summary(
                selection="best",
                checkpoint_interval=5,
                selection_scenario_episodes=2,
                selection_config=CheckpointSelectionConfig(
                    metric="scenario_success_rate",
                ),
                override_penalty_weight=1.0,
            )
