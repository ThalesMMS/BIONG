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

class SpiderTrainingTestBase(unittest.TestCase):
    @staticmethod
    def _claim_suite_base_argv(*, include_suite_flag: bool = True) -> list[str]:
        argv = [
            sys.executable,
            "-m",
            "spider_cortex_sim",
            "--episodes",
            "0",
            "--eval-episodes",
            "0",
        ]
        if include_suite_flag:
            argv.append("--claim-test-suite")
        argv.extend(
            [
                "--claim-test",
                "learning_without_privileged_signals",
                "--budget-profile",
                "smoke",
                "--learning-evidence-long-budget-profile",
                "smoke",
            ]
        )
        return argv

    @staticmethod
    def _fake_run_episode(
        episode_index: int,
        *,
        training: bool,
        sample: bool,
        render: bool = False,
        capture_trace: bool = False,
        scenario_name: str | None = None,
        debug_trace: bool = False,
        policy_mode: str = "normal",
    ) -> tuple[dict[str, Any], list[Any]]:
        """Return minimal run_episode output for curriculum scheduler tests."""
        return {"episode": episode_index, "scenario": scenario_name}, []

    def _assert_finite_summary(self, summary: dict[str, object]) -> None:
        """
        Assert that the evaluation section of `summary` contains finite numeric values for key metrics.
        
        Parameters:
            summary (dict): A summary dictionary that must include an "evaluation" mapping with numeric metrics such as
                "mean_reward", "mean_food", "mean_sleep", "mean_predator_contacts", "mean_predator_escapes",
                "mean_night_shelter_occupancy_rate", "mean_night_stillness_rate", "mean_predator_response_latency",
                "mean_sleep_debt", "mean_food_distance_delta", "mean_shelter_distance_delta",
                "mean_predator_mode_transitions", and "survival_rate".
        """
        evaluation = summary["evaluation"]
        numeric_keys = [
            "mean_reward",
            "mean_food",
            "mean_sleep",
            "mean_predator_contacts",
            "mean_predator_escapes",
            "mean_night_shelter_occupancy_rate",
            "mean_night_stillness_rate",
            "mean_predator_response_latency",
            "mean_sleep_debt",
            "mean_food_distance_delta",
            "mean_shelter_distance_delta",
            "mean_predator_mode_transitions",
            "survival_rate",
        ]
        for key in numeric_keys:
            self.assertTrue(np.isfinite(evaluation[key]), key)
