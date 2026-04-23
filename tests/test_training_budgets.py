import unittest
import json
import os
import subprocess
import sys
import tempfile
from unittest import mock
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from spider_cortex_sim.ablations import BrainAblationConfig, canonical_ablation_configs
from spider_cortex_sim.agent import BrainStep, SpiderBrain
from spider_cortex_sim.curriculum import (
    CurriculumPhaseDefinition,
    PromotionCheckCriteria,
)
from spider_cortex_sim.checkpointing import CheckpointSelectionConfig
from spider_cortex_sim.export import save_behavior_csv, save_summary
from spider_cortex_sim.interfaces import ACTION_TO_INDEX, LOCOMOTION_ACTIONS
from spider_cortex_sim.operational_profiles import DEFAULT_OPERATIONAL_PROFILE, OperationalProfile
from spider_cortex_sim.scenarios import SCENARIO_NAMES
from spider_cortex_sim.simulation import SpiderSimulation
from spider_cortex_sim.world import SpiderWorld

class SimulationBudgetAttributesTest(unittest.TestCase):
    """Tests for budget-related attributes and methods added to SpiderSimulation."""

    def test_default_budget_profile_name_is_custom(self) -> None:
        sim = SpiderSimulation(seed=7, max_steps=20)
        self.assertEqual(sim.budget_profile_name, "custom")

    def test_default_benchmark_strength_is_custom(self) -> None:
        sim = SpiderSimulation(seed=7, max_steps=20)
        self.assertEqual(sim.benchmark_strength, "custom")

    def test_budget_profile_name_set_in_constructor(self) -> None:
        sim = SpiderSimulation(
            seed=7,
            max_steps=20,
            budget_profile_name="dev",
            benchmark_strength="quick",
        )
        self.assertEqual(sim.budget_profile_name, "dev")
        self.assertEqual(sim.benchmark_strength, "quick")

    def test_budget_summary_in_constructor_is_deep_copied(self) -> None:
        original_summary = {
            "profile": "smoke",
            "benchmark_strength": "quick",
            "resolved": {"episodes": 6},
            "overrides": {},
        }
        sim = SpiderSimulation(seed=7, max_steps=20, budget_summary=original_summary)
        # Mutating original should not affect sim.budget_summary
        original_summary["profile"] = "mutated"
        self.assertEqual(sim.budget_summary["profile"], "smoke")

    def test_budget_summary_in_constructor_updates_budget_metadata(self) -> None:
        sim = SpiderSimulation(
            seed=7,
            max_steps=20,
            budget_summary={
                "profile": "smoke",
                "benchmark_strength": "quick",
                "resolved": {
                    "episodes": 6,
                    "eval_episodes": 1,
                    "max_steps": 20,
                    "scenario_episodes": 1,
                    "comparison_seeds": [7],
                    "checkpoint_interval": 2,
                    "selection_scenario_episodes": 1,
                    "behavior_seeds": [7],
                    "ablation_seeds": [7],
                },
                "overrides": {},
            },
        )

        self.assertEqual(sim.budget_profile_name, "smoke")
        self.assertEqual(sim.benchmark_strength, "quick")

        _, _, rows = sim.evaluate_behavior_suite(["night_rest"])
        self.assertEqual(rows[0]["budget_profile"], "smoke")
        self.assertEqual(rows[0]["benchmark_strength"], "quick")

    def test_budget_summary_default_when_none(self) -> None:
        sim = SpiderSimulation(seed=7, max_steps=20)
        self.assertIn("profile", sim.budget_summary)
        self.assertIn("benchmark_strength", sim.budget_summary)
        self.assertIn("resolved", sim.budget_summary)
        self.assertEqual(sim.budget_summary["profile"], "custom")

    def test_checkpoint_source_default_is_final(self) -> None:
        sim = SpiderSimulation(seed=7, max_steps=20)
        self.assertEqual(sim.checkpoint_source, "final")

    def test_checkpoint_source_remains_final_after_regular_train(self) -> None:
        sim = SpiderSimulation(seed=7, max_steps=10)
        sim.train(episodes=2, evaluation_episodes=0, capture_evaluation_trace=False)
        self.assertEqual(sim.checkpoint_source, "final")

    def test_train_with_invalid_checkpoint_selection_raises(self) -> None:
        sim = SpiderSimulation(seed=7, max_steps=10)
        with self.assertRaises(ValueError):
            sim.train(
                episodes=2,
                evaluation_episodes=0,
                capture_evaluation_trace=False,
                checkpoint_selection="invalid",
            )

    def test_train_with_invalid_checkpoint_metric_raises_before_training(self) -> None:
        sim = SpiderSimulation(seed=7, max_steps=10)
        run_episode_calls = 0
        original_run_episode = sim.run_episode

        def wrapped_run_episode(*args, **kwargs):
            nonlocal run_episode_calls
            run_episode_calls += 1
            return original_run_episode(*args, **kwargs)

        sim.run_episode = wrapped_run_episode  # type: ignore[method-assign]

        with self.assertRaises(ValueError):
            sim.train(
                episodes=2,
                evaluation_episodes=0,
                capture_evaluation_trace=False,
                checkpoint_selection="best",
                checkpoint_selection_config=CheckpointSelectionConfig(
                    metric="not_a_real_metric",
                ),
            )

        self.assertEqual(run_episode_calls, 0)

    def test_budget_summary_updated_by_train(self) -> None:
        sim = SpiderSimulation(seed=7, max_steps=30)
        sim.train(episodes=4, evaluation_episodes=1, capture_evaluation_trace=False)
        resolved = sim.budget_summary["resolved"]
        self.assertEqual(resolved["episodes"], 4)
        self.assertEqual(resolved["eval_episodes"], 1)
        self.assertEqual(resolved["max_steps"], 30)

    def test_budget_in_summary_config_after_train(self) -> None:
        sim = SpiderSimulation(
            seed=7,
            max_steps=20,
            budget_profile_name="smoke",
            benchmark_strength="quick",
            budget_summary={
                "profile": "smoke",
                "benchmark_strength": "quick",
                "resolved": {
                    "episodes": 6,
                    "eval_episodes": 1,
                    "max_steps": 20,
                    "scenario_episodes": 1,
                    "comparison_seeds": [7],
                    "checkpoint_interval": 2,
                    "selection_scenario_episodes": 1,
                    "behavior_seeds": [7],
                    "ablation_seeds": [7],
                },
                "overrides": {},
            },
        )
        summary, _ = sim.train(episodes=0, evaluation_episodes=0, capture_evaluation_trace=False)
        self.assertIn("budget", summary["config"])
        self.assertEqual(summary["config"]["budget"]["profile"], "smoke")
        self.assertEqual(summary["config"]["budget"]["benchmark_strength"], "quick")

    def test_set_runtime_budget_updates_episodes(self) -> None:
        sim = SpiderSimulation(seed=7, max_steps=20)
        sim._set_runtime_budget(episodes=10, evaluation_episodes=2)
        self.assertEqual(sim.budget_summary["resolved"]["episodes"], 10)
        self.assertEqual(sim.budget_summary["resolved"]["eval_episodes"], 2)

    def test_set_runtime_budget_updates_max_steps(self) -> None:
        sim = SpiderSimulation(seed=7, max_steps=55)
        sim._set_runtime_budget(episodes=0, evaluation_episodes=0)
        self.assertEqual(sim.budget_summary["resolved"]["max_steps"], 55)

    def test_set_runtime_budget_updates_scenario_episodes(self) -> None:
        sim = SpiderSimulation(seed=7, max_steps=20)
        sim._set_runtime_budget(episodes=0, evaluation_episodes=0, scenario_episodes=3)
        self.assertEqual(sim.budget_summary["resolved"]["scenario_episodes"], 3)

    def test_set_runtime_budget_default_scenario_episodes_when_absent(self) -> None:
        sim = SpiderSimulation(seed=7, max_steps=20)
        # Remove scenario_episodes to test default insertion
        sim.budget_summary.setdefault("resolved", {}).pop("scenario_episodes", None)
        sim._set_runtime_budget(episodes=0, evaluation_episodes=0)
        self.assertEqual(sim.budget_summary["resolved"]["scenario_episodes"], 1)

    def test_set_runtime_budget_updates_behavior_seeds(self) -> None:
        sim = SpiderSimulation(seed=7, max_steps=20)
        sim._set_runtime_budget(
            episodes=0, evaluation_episodes=0, behavior_seeds=(11, 13)
        )
        self.assertEqual(sim.budget_summary["resolved"]["behavior_seeds"], [11, 13])

    def test_set_runtime_budget_updates_ablation_seeds(self) -> None:
        sim = SpiderSimulation(seed=7, max_steps=20)
        sim._set_runtime_budget(
            episodes=0, evaluation_episodes=0, ablation_seeds=(99,)
        )
        self.assertEqual(sim.budget_summary["resolved"]["ablation_seeds"], [99])

    def test_set_runtime_budget_updates_checkpoint_interval(self) -> None:
        sim = SpiderSimulation(seed=7, max_steps=20)
        sim._set_runtime_budget(
            episodes=0, evaluation_episodes=0, checkpoint_interval=7
        )
        self.assertEqual(sim.budget_summary["resolved"]["checkpoint_interval"], 7)
