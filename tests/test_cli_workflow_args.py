"""Tests for spider_cortex_sim.cli.workflow_args module.

Covers _build_compare_training_kwargs and _build_checkpoint_selection_config,
extracted from the main CLI module in the PR refactor.
"""

from __future__ import annotations

import argparse
import io
import sys
import unittest
from unittest.mock import MagicMock, patch

from spider_cortex_sim.cli.workflow_args import (
    _build_checkpoint_selection_config,
    _build_compare_training_kwargs,
)
from spider_cortex_sim.checkpointing import CheckpointSelectionConfig


def _make_args(**kwargs) -> argparse.Namespace:
    """Build a minimal argparse.Namespace with the expected fields."""
    defaults = {
        "width": 12,
        "height": 12,
        "food_count": 4,
        "day_length": 18,
        "night_length": 12,
        "gamma": 0.96,
        "module_lr": 0.01,
        "motor_lr": 0.012,
        "module_dropout": 0.05,
        "reward_profile": "classic",
        "map_template": "central_burrow",
        "operational_profile": "default_v1",
        "noise_profile": "none",
        "budget_profile": "smoke",
        "checkpoint_selection": "none",
        "checkpoint_interval": None,
        "checkpoint_dir": None,
        "checkpoint_metric": "scenario_success_rate",
        "checkpoint_override_penalty": 0.0,
        "checkpoint_dominance_penalty": 0.0,
        "checkpoint_penalty_mode": "tiebreaker",
        "curriculum_profile": "none",
        "_parser": None,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def _make_budget(**kwargs):
    """Build a minimal budget-like namespace."""
    defaults = {
        "max_steps": 200,
        "episodes": 10,
        "eval_episodes": 2,
        "behavior_seeds": (1, 2),
        "ablation_seeds": (1, 2),
        "scenario_episodes": 5,
        "checkpoint_interval": 10,
        "selection_scenario_episodes": 5,
        "profile": "smoke",
        "benchmark_strength": "smoke",
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


class BuildCompareTrainingKwargsTest(unittest.TestCase):
    """Tests for _build_compare_training_kwargs."""

    def test_returns_dict(self) -> None:
        args = _make_args()
        budget = _make_budget()
        config = CheckpointSelectionConfig(metric="scenario_success_rate")
        result = _build_compare_training_kwargs(args, budget, config)
        self.assertIsInstance(result, dict)

    def test_contains_required_keys(self) -> None:
        args = _make_args()
        budget = _make_budget()
        config = CheckpointSelectionConfig(metric="scenario_success_rate")
        result = _build_compare_training_kwargs(args, budget, config)

        expected_keys = [
            "width", "height", "food_count", "day_length", "night_length",
            "max_steps", "episodes", "evaluation_episodes", "gamma",
            "module_lr", "motor_lr", "module_dropout", "reward_profile",
            "map_template", "operational_profile", "noise_profile",
            "budget_profile", "seeds", "episodes_per_scenario",
            "checkpoint_selection", "checkpoint_selection_config",
            "checkpoint_interval", "checkpoint_dir", "curriculum_profile",
        ]
        for key in expected_keys:
            self.assertIn(key, result, f"Missing expected key: {key}")

    def test_seeds_is_tuple_of_behavior_seeds(self) -> None:
        args = _make_args()
        budget = _make_budget(behavior_seeds=(7, 13, 42))
        config = CheckpointSelectionConfig(metric="scenario_success_rate")
        result = _build_compare_training_kwargs(args, budget, config)
        self.assertEqual(result["seeds"], (7, 13, 42))

    def test_seeds_is_always_tuple(self) -> None:
        args = _make_args()
        budget = _make_budget(behavior_seeds=[1])
        config = CheckpointSelectionConfig(metric="scenario_success_rate")
        result = _build_compare_training_kwargs(args, budget, config)
        self.assertIsInstance(result["seeds"], tuple)

    def test_width_and_height_from_args(self) -> None:
        args = _make_args(width=20, height=15)
        budget = _make_budget()
        config = CheckpointSelectionConfig(metric="scenario_success_rate")
        result = _build_compare_training_kwargs(args, budget, config)
        self.assertEqual(result["width"], 20)
        self.assertEqual(result["height"], 15)

    def test_max_steps_from_budget(self) -> None:
        args = _make_args()
        budget = _make_budget(max_steps=500)
        config = CheckpointSelectionConfig(metric="scenario_success_rate")
        result = _build_compare_training_kwargs(args, budget, config)
        self.assertEqual(result["max_steps"], 500)

    def test_episodes_from_budget(self) -> None:
        args = _make_args()
        budget = _make_budget(episodes=100)
        config = CheckpointSelectionConfig(metric="scenario_success_rate")
        result = _build_compare_training_kwargs(args, budget, config)
        self.assertEqual(result["episodes"], 100)

    def test_evaluation_episodes_from_budget(self) -> None:
        args = _make_args()
        budget = _make_budget(eval_episodes=20)
        config = CheckpointSelectionConfig(metric="scenario_success_rate")
        result = _build_compare_training_kwargs(args, budget, config)
        self.assertEqual(result["evaluation_episodes"], 20)

    def test_episodes_per_scenario_from_budget(self) -> None:
        args = _make_args()
        budget = _make_budget(scenario_episodes=3)
        config = CheckpointSelectionConfig(metric="scenario_success_rate")
        result = _build_compare_training_kwargs(args, budget, config)
        self.assertEqual(result["episodes_per_scenario"], 3)

    def test_checkpoint_selection_config_passed_through(self) -> None:
        args = _make_args()
        budget = _make_budget()
        config = CheckpointSelectionConfig(metric="episode_success_rate")
        result = _build_compare_training_kwargs(args, budget, config)
        self.assertIs(result["checkpoint_selection_config"], config)

    def test_curriculum_profile_from_args(self) -> None:
        args = _make_args(curriculum_profile="ecological_v1")
        budget = _make_budget()
        config = CheckpointSelectionConfig(metric="scenario_success_rate")
        result = _build_compare_training_kwargs(args, budget, config)
        self.assertEqual(result["curriculum_profile"], "ecological_v1")

    def test_checkpoint_dir_from_args(self) -> None:
        args = _make_args(checkpoint_dir="/tmp/checkpoints")
        budget = _make_budget()
        config = CheckpointSelectionConfig(metric="scenario_success_rate")
        result = _build_compare_training_kwargs(args, budget, config)
        self.assertEqual(result["checkpoint_dir"], "/tmp/checkpoints")

    def test_reward_profile_from_args(self) -> None:
        args = _make_args(reward_profile="sparse")
        budget = _make_budget()
        config = CheckpointSelectionConfig(metric="scenario_success_rate")
        result = _build_compare_training_kwargs(args, budget, config)
        self.assertEqual(result["reward_profile"], "sparse")


class BuildCheckpointSelectionConfigTest(unittest.TestCase):
    """Tests for _build_checkpoint_selection_config."""

    def test_returns_checkpoint_selection_config(self) -> None:
        args = _make_args(
            checkpoint_metric="scenario_success_rate",
            checkpoint_override_penalty=0.0,
            checkpoint_dominance_penalty=0.0,
            checkpoint_penalty_mode="tiebreaker",
        )
        result = _build_checkpoint_selection_config(args)
        self.assertIsInstance(result, CheckpointSelectionConfig)

    def test_metric_is_set_from_args(self) -> None:
        args = _make_args(
            checkpoint_metric="episode_success_rate",
            checkpoint_override_penalty=0.0,
            checkpoint_dominance_penalty=0.0,
            checkpoint_penalty_mode="tiebreaker",
        )
        result = _build_checkpoint_selection_config(args)
        self.assertEqual(result.metric, "episode_success_rate")

    def test_override_penalty_weight_is_set(self) -> None:
        args = _make_args(
            checkpoint_metric="scenario_success_rate",
            checkpoint_override_penalty=0.5,
            checkpoint_dominance_penalty=0.0,
            checkpoint_penalty_mode="tiebreaker",
        )
        result = _build_checkpoint_selection_config(args)
        self.assertAlmostEqual(result.override_penalty_weight, 0.5)

    def test_dominance_penalty_weight_is_set(self) -> None:
        args = _make_args(
            checkpoint_metric="scenario_success_rate",
            checkpoint_override_penalty=0.0,
            checkpoint_dominance_penalty=1.0,
            checkpoint_penalty_mode="direct",
        )
        result = _build_checkpoint_selection_config(args)
        self.assertAlmostEqual(result.dominance_penalty_weight, 1.0)

    def test_invalid_metric_raises_system_exit(self) -> None:
        args = _make_args(
            checkpoint_metric="nonexistent_metric",
            checkpoint_override_penalty=0.0,
            checkpoint_dominance_penalty=0.0,
            checkpoint_penalty_mode="tiebreaker",
            _parser=None,
        )
        stderr = io.StringIO()
        with self.assertRaises(SystemExit) as ctx:
            with patch("sys.stderr", stderr):
                _build_checkpoint_selection_config(args)
        self.assertEqual(ctx.exception.code, 2)

    def test_scenario_success_rate_metric_is_valid(self) -> None:
        args = _make_args(
            checkpoint_metric="scenario_success_rate",
            checkpoint_override_penalty=0.0,
            checkpoint_dominance_penalty=0.0,
            checkpoint_penalty_mode="tiebreaker",
        )
        result = _build_checkpoint_selection_config(args)
        self.assertEqual(result.metric, "scenario_success_rate")

    def test_mean_reward_metric_is_valid(self) -> None:
        args = _make_args(
            checkpoint_metric="mean_reward",
            checkpoint_override_penalty=0.0,
            checkpoint_dominance_penalty=0.0,
            checkpoint_penalty_mode="tiebreaker",
        )
        result = _build_checkpoint_selection_config(args)
        self.assertEqual(result.metric, "mean_reward")

    def test_direct_penalty_mode_is_accepted(self) -> None:
        args = _make_args(
            checkpoint_metric="scenario_success_rate",
            checkpoint_override_penalty=1.0,
            checkpoint_dominance_penalty=1.0,
            checkpoint_penalty_mode="direct",
        )
        result = _build_checkpoint_selection_config(args)
        self.assertIsInstance(result, CheckpointSelectionConfig)


if __name__ == "__main__":
    unittest.main()