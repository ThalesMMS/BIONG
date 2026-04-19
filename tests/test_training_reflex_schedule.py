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
from spider_cortex_sim.export import save_behavior_csv, save_summary
from spider_cortex_sim.interfaces import ACTION_TO_INDEX, LOCOMOTION_ACTIONS
from spider_cortex_sim.operational_profiles import DEFAULT_OPERATIONAL_PROFILE, OperationalProfile
from spider_cortex_sim.scenarios import SCENARIO_NAMES
from spider_cortex_sim.simulation import SpiderSimulation
from spider_cortex_sim.world import SpiderWorld

from tests.fixtures.training import SpiderTrainingTestBase

class SpiderTrainingReflexScheduleTest(SpiderTrainingTestBase):
    def test_train_records_reflex_schedule_and_eval_without_reflex_support(self) -> None:
        """Verify reflex annealing schedule and paired eval summaries."""
        sim = SpiderSimulation(seed=19, max_steps=20)
        summary, _ = sim.train(
            episodes=3,
            evaluation_episodes=1,
            capture_evaluation_trace=False,
            reflex_anneal_final_scale=0.25,
        )

        self.assertIn("reflex_schedule", summary["config"])
        self.assertEqual(summary["config"]["reflex_schedule"]["mode"], "linear")
        self.assertAlmostEqual(summary["config"]["reflex_schedule"]["start_scale"], 1.0)
        self.assertAlmostEqual(summary["config"]["reflex_schedule"]["final_scale"], 0.25)
        self.assertIn("mean_reflex_usage_rate", summary["evaluation"])
        self.assertIn("mean_module_reflex_usage_rate", summary["evaluation"])
        self.assertIn("mean_module_contribution_share", summary["evaluation"])
        self.assertIn("mean_effective_module_count", summary["evaluation"])
        self.assertEqual(summary["evaluation"]["eval_reflex_scale"], 0.0)
        self.assertEqual(summary["evaluation"]["competence_type"], "self_sufficient")
        self.assertIn("self_sufficient", summary["evaluation"])
        self.assertIn("scaffolded", summary["evaluation"])
        self.assertIn("primary_benchmark", summary["evaluation"])
        self.assertIn("competence_gap", summary["evaluation"])
        self.assertEqual(
            summary["evaluation"]["self_sufficient"]["competence_type"],
            "self_sufficient",
        )
        self.assertEqual(
            summary["evaluation"]["scaffolded"]["competence_type"],
            "scaffolded",
        )
        self.assertEqual(
            summary["evaluation"]["primary_benchmark"],
            summary["evaluation"]["self_sufficient"],
        )
        self.assertIn(
            "scenario_success_rate_delta",
            summary["evaluation"]["competence_gap"],
        )
        self.assertIn("evaluation_without_reflex_support", summary)
        self.assertEqual(summary["evaluation_without_reflex_support"]["eval_reflex_scale"], 0.0)
        self.assertTrue(summary["evaluation_without_reflex_support"]["primary"])
        self.assertEqual(
            summary["evaluation_without_reflex_support"]["summary"],
            summary["evaluation"]["self_sufficient"],
        )
        self.assertIn("evaluation_with_reflex_support", summary)
        self.assertAlmostEqual(
            summary["evaluation_with_reflex_support"]["eval_reflex_scale"],
            0.25,
        )

    def test_set_training_episode_reflex_scale_linear_interpolation(self) -> None:
        sim = SpiderSimulation(seed=7, max_steps=5)
        # With 5 episodes total, episode 0 → scale=1.0, episode 4 → scale=0.0
        scale_at_start = sim._set_training_episode_reflex_scale(
            episode_index=0,
            total_episodes=5,
            final_scale=0.0,
        )
        scale_at_end = sim._set_training_episode_reflex_scale(
            episode_index=4,
            total_episodes=5,
            final_scale=0.0,
        )
        self.assertAlmostEqual(scale_at_start, 1.0)
        self.assertAlmostEqual(scale_at_end, 0.0)

    def test_set_training_episode_reflex_scale_midpoint(self) -> None:
        sim = SpiderSimulation(seed=7, max_steps=5)
        # Episode 2 out of 5 (0-indexed): progress = 2/4 = 0.5; scale = 1.0 + (0.2 - 1.0) * 0.5 = 0.6
        scale = sim._set_training_episode_reflex_scale(
            episode_index=2,
            total_episodes=5,
            final_scale=0.2,
        )
        self.assertAlmostEqual(scale, 0.6)

    def test_set_training_episode_reflex_scale_single_episode_keeps_start(self) -> None:
        sim = SpiderSimulation(seed=7, max_steps=5)
        # With total_episodes=1, scale should be start_scale regardless of final_scale
        scale = sim._set_training_episode_reflex_scale(
            episode_index=0,
            total_episodes=1,
            final_scale=0.0,
        )
        self.assertAlmostEqual(scale, 1.0)

    def test_set_training_episode_reflex_scale_clamps_negative_final_to_zero(self) -> None:
        sim = SpiderSimulation(seed=7, max_steps=5)
        # negative final_scale should be clamped to 0
        scale = sim._set_training_episode_reflex_scale(
            episode_index=4,
            total_episodes=5,
            final_scale=-1.0,
        )
        self.assertAlmostEqual(scale, 0.0)

    def test_set_training_episode_reflex_scale_updates_brain_scale(self) -> None:
        sim = SpiderSimulation(seed=7, max_steps=5)
        sim._set_training_episode_reflex_scale(
            episode_index=0,
            total_episodes=5,
            final_scale=0.5,
        )
        self.assertAlmostEqual(sim.brain.current_reflex_scale, 1.0)
        sim._set_training_episode_reflex_scale(
            episode_index=4,
            total_episodes=5,
            final_scale=0.5,
        )
        self.assertAlmostEqual(sim.brain.current_reflex_scale, 0.5)

    def test_reflex_schedule_summary_enabled_when_anneal_differs_from_start(self) -> None:
        sim = SpiderSimulation(seed=7, max_steps=10)
        summary, _ = sim.train(
            episodes=2,
            evaluation_episodes=0,
            capture_evaluation_trace=False,
            reflex_anneal_final_scale=0.5,
        )
        schedule = summary["config"]["reflex_schedule"]
        self.assertTrue(schedule["enabled"])
        self.assertAlmostEqual(schedule["start_scale"], 1.0)
        self.assertAlmostEqual(schedule["final_scale"], 0.5)
        self.assertEqual(schedule["mode"], "linear")
        self.assertEqual(schedule["episodes"], 2)

    def test_reflex_schedule_summary_disabled_when_anneal_matches_start(self) -> None:
        sim = SpiderSimulation(seed=7, max_steps=10)
        summary, _ = sim.train(
            episodes=2,
            evaluation_episodes=0,
            capture_evaluation_trace=False,
            reflex_anneal_final_scale=1.0,  # same as start
        )
        schedule = summary["config"]["reflex_schedule"]
        self.assertFalse(schedule["enabled"])

    def test_evaluation_without_reflex_support_absent_for_monolithic(self) -> None:
        from spider_cortex_sim.ablations import BrainAblationConfig
        config = BrainAblationConfig(
            name="monolithic_policy",
            architecture="monolithic",
            module_dropout=0.0,
            enable_reflexes=False,
            enable_auxiliary_targets=False,
            disabled_modules=(),
        )
        sim = SpiderSimulation(seed=7, max_steps=5, brain_config=config)
        summary, _ = sim.train(
            episodes=0,
            evaluation_episodes=2,
            capture_evaluation_trace=False,
        )
        # Monolithic brains don't get evaluation_without_reflex_support
        self.assertNotIn("evaluation_without_reflex_support", summary)

    def test_eval_reflex_scale_restores_brain_scale_after_evaluation(self) -> None:
        sim = SpiderSimulation(seed=7, max_steps=5)
        sim.brain.set_runtime_reflex_scale(0.8)
        # evaluate_behavior_suite with eval_reflex_scale should restore the scale after
        sim.evaluate_behavior_suite(
            ["night_rest"],
            episodes_per_scenario=1,
            eval_reflex_scale=0.0,
        )
        self.assertAlmostEqual(sim.brain.current_reflex_scale, 0.8)
