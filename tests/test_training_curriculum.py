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

class SpiderTrainingCurriculumTest(SpiderTrainingTestBase):
    def test_train_curriculum_records_training_regime_and_phase_summary(self) -> None:
        sim = SpiderSimulation(seed=19, max_steps=20)
        summary, _ = sim.train(
            episodes=6,
            evaluation_episodes=1,
            capture_evaluation_trace=False,
            curriculum_profile="ecological_v1",
        )

        self.assertEqual(summary["config"]["training_regime"]["mode"], "curriculum")
        self.assertEqual(
            summary["config"]["training_regime"]["curriculum_profile"],
            "ecological_v1",
        )
        self.assertIn("curriculum", summary)
        self.assertEqual(summary["curriculum"]["profile"], "ecological_v1")
        self.assertEqual(len(summary["curriculum"]["phases"]), 4)
        self.assertIn(
            summary["curriculum"]["phases"][0]["status"],
            {"promoted", "max_budget_exhausted", "skipped_no_budget"},
        )

    def test_train_curriculum_is_reproducible_for_same_seed(self) -> None:
        sim_a = SpiderSimulation(seed=21, max_steps=20)
        sim_b = SpiderSimulation(seed=21, max_steps=20)

        summary_a, _ = sim_a.train(
            episodes=6,
            evaluation_episodes=0,
            capture_evaluation_trace=False,
            curriculum_profile="ecological_v1",
        )
        summary_b, _ = sim_b.train(
            episodes=6,
            evaluation_episodes=0,
            capture_evaluation_trace=False,
            curriculum_profile="ecological_v1",
        )

        self.assertEqual(summary_a["curriculum"], summary_b["curriculum"])

    def test_set_training_regime_metadata_flat_mode(self) -> None:
        sim = SpiderSimulation(seed=7, max_steps=10)
        sim._set_training_regime_metadata(curriculum_profile="none", episodes=6)
        regime = sim._latest_training_regime_summary
        self.assertEqual(regime["mode"], "flat")
        self.assertEqual(regime["curriculum_profile"], "none")
        self.assertEqual(regime["resolved_budget"]["total_training_episodes"], 6)
        self.assertEqual(regime["resolved_budget"]["phase_episode_budgets"], [])

    def test_set_training_regime_metadata_curriculum_mode(self) -> None:
        sim = SpiderSimulation(seed=7, max_steps=10)
        sim._set_training_regime_metadata(curriculum_profile="ecological_v1", episodes=12)
        regime = sim._latest_training_regime_summary
        self.assertEqual(regime["mode"], "curriculum")
        self.assertEqual(regime["curriculum_profile"], "ecological_v1")
        self.assertEqual(regime["resolved_budget"]["total_training_episodes"], 12)
        self.assertEqual(len(regime["resolved_budget"]["phase_episode_budgets"]), 4)
        self.assertEqual(sum(regime["resolved_budget"]["phase_episode_budgets"]), 12)

    def test_set_training_regime_metadata_rejects_invalid_profile(self) -> None:
        sim = SpiderSimulation(seed=7, max_steps=10)
        with self.assertRaises(ValueError):
            sim._set_training_regime_metadata(
                curriculum_profile="bad_profile",
                episodes=6,
            )

    def test_set_training_regime_metadata_curriculum_summary_stored(self) -> None:
        sim = SpiderSimulation(seed=7, max_steps=10)
        curriculum_summary = {"profile": "ecological_v1", "phases": []}
        sim._set_training_regime_metadata(
            curriculum_profile="ecological_v1",
            episodes=6,
            curriculum_summary=curriculum_summary,
        )
        self.assertIsNotNone(sim._latest_curriculum_summary)
        self.assertEqual(sim._latest_curriculum_summary["profile"], "ecological_v1")

    def test_set_training_regime_metadata_none_curriculum_summary_clears_latest(self) -> None:
        sim = SpiderSimulation(seed=7, max_steps=10)
        # First set with a summary
        sim._set_training_regime_metadata(
            curriculum_profile="ecological_v1",
            episodes=6,
            curriculum_summary={"profile": "ecological_v1", "phases": []},
        )
        # Then clear it
        sim._set_training_regime_metadata(
            curriculum_profile="ecological_v1",
            episodes=6,
            curriculum_summary=None,
        )
        self.assertIsNone(sim._latest_curriculum_summary)

    def test_execute_training_schedule_invalid_profile_raises_value_error(self) -> None:
        sim = SpiderSimulation(seed=7, max_steps=10)
        with self.assertRaises(ValueError):
            sim._execute_training_schedule(episodes=4, curriculum_profile="bad_profile")

    def test_execute_training_schedule_flat_zero_episodes_returns_empty(self) -> None:
        sim = SpiderSimulation(seed=7, max_steps=10)
        history = sim._execute_training_schedule(episodes=0, curriculum_profile="none")
        self.assertEqual(history, [])

    def test_execute_training_schedule_flat_returns_correct_episode_count(self) -> None:
        sim = SpiderSimulation(seed=7, max_steps=10)
        history = sim._execute_training_schedule(episodes=3, curriculum_profile="none")
        self.assertEqual(len(history), 3)

    def test_execute_training_schedule_flat_sets_regime_mode_flat(self) -> None:
        sim = SpiderSimulation(seed=7, max_steps=10)
        sim._execute_training_schedule(episodes=2, curriculum_profile="none")
        self.assertEqual(sim._latest_training_regime_summary["mode"], "flat")

    def test_execute_training_schedule_curriculum_sets_regime_mode_curriculum(self) -> None:
        sim = SpiderSimulation(seed=7, max_steps=10)
        sim._execute_training_schedule(episodes=6, curriculum_profile="ecological_v1")
        self.assertEqual(sim._latest_training_regime_summary["mode"], "curriculum")

    def test_execute_training_schedule_flat_calls_checkpoint_callback(self) -> None:
        sim = SpiderSimulation(seed=7, max_steps=10)
        call_counts = []
        sim._execute_training_schedule(
            episodes=3,
            curriculum_profile="none",
            checkpoint_callback=lambda n: call_counts.append(n),
        )
        self.assertEqual(call_counts, [1, 2, 3])

    def test_train_flat_regime_default_mode_is_flat_in_summary(self) -> None:
        """Default training (no curriculum_profile) should record mode=flat."""
        sim = SpiderSimulation(seed=7, max_steps=10)
        summary, _ = sim.train(
            episodes=2,
            evaluation_episodes=0,
            capture_evaluation_trace=False,
        )
        self.assertEqual(summary["config"]["training_regime"]["mode"], "flat")
        self.assertEqual(
            summary["config"]["training_regime"]["curriculum_profile"], "none"
        )
        self.assertNotIn("curriculum", summary)

    def test_train_invalid_curriculum_profile_raises_value_error(self) -> None:
        sim = SpiderSimulation(seed=7, max_steps=10)
        with self.assertRaises(ValueError):
            sim.train(
                episodes=2,
                evaluation_episodes=0,
                capture_evaluation_trace=False,
                curriculum_profile="invalid_profile",
            )

    def test_train_curriculum_phase_budgets_sum_equals_episodes(self) -> None:
        sim = SpiderSimulation(seed=7, max_steps=10)
        episodes = 6
        summary, _ = sim.train(
            episodes=episodes,
            evaluation_episodes=0,
            capture_evaluation_trace=False,
            curriculum_profile="ecological_v1",
        )
        phase_budgets = summary["config"]["training_regime"]["resolved_budget"][
            "phase_episode_budgets"
        ]
        self.assertEqual(sum(phase_budgets), episodes)

    def test_train_curriculum_summary_has_four_phases(self) -> None:
        sim = SpiderSimulation(seed=7, max_steps=10)
        summary, _ = sim.train(
            episodes=6,
            evaluation_episodes=0,
            capture_evaluation_trace=False,
            curriculum_profile="ecological_v1",
        )
        self.assertEqual(len(summary["curriculum"]["phases"]), 4)

    def test_train_curriculum_phases_have_required_fields(self) -> None:
        """
        Verify curriculum phases in the training summary include all required metadata fields.
        
        When training with the "ecological_v1" curriculum profile, each phase entry in summary["curriculum"]["phases"] must contain the following fields: name, skill_name, training_scenarios, promotion_scenarios, promotion_check_specs, success_threshold, max_episodes, min_episodes, allocated_episodes, carryover_in, episodes_executed, status, promotion_reason, promotion_checks, final_metrics, and final_check_results.
        """
        sim = SpiderSimulation(seed=7, max_steps=10)
        summary, _ = sim.train(
            episodes=6,
            evaluation_episodes=0,
            capture_evaluation_trace=False,
            curriculum_profile="ecological_v1",
        )
        required_fields = {
            "name",
            "skill_name",
            "training_scenarios",
            "promotion_scenarios",
            "promotion_check_specs",
            "success_threshold",
            "max_episodes",
            "min_episodes",
            "allocated_episodes",
            "carryover_in",
            "episodes_executed",
            "status",
            "promotion_reason",
            "promotion_checks",
            "final_metrics",
            "final_check_results",
        }
        for phase in summary["curriculum"]["phases"]:
            for field in required_fields:
                self.assertIn(field, phase, msg=f"Missing field {field!r}")

    def test_train_curriculum_total_episodes_recorded_correctly(self) -> None:
        sim = SpiderSimulation(seed=7, max_steps=10)
        summary, _ = sim.train(
            episodes=6,
            evaluation_episodes=0,
            capture_evaluation_trace=False,
            curriculum_profile="ecological_v1",
        )
        self.assertEqual(
            summary["curriculum"]["total_training_episodes"], 6
        )

    def test_train_curriculum_promotion_carries_budget_forward(self) -> None:
        sim = SpiderSimulation(seed=7, max_steps=10)
        fake_run_episode = self._fake_run_episode

        eval_calls = {"count": 0}

        def fake_evaluate_behavior_suite(*args, **kwargs):
            eval_calls["count"] += 1
            if eval_calls["count"] == 1:
                payload = {
                    "summary": {
                        "scenario_success_rate": 1.0,
                        "episode_success_rate": 1.0,
                    },
                    "suite": {},
                }
            else:
                payload = {
                    "summary": {
                        "scenario_success_rate": 0.0,
                        "episode_success_rate": 0.0,
                    },
                    "suite": {},
                }
            return payload, [], []

        with mock.patch.object(sim, "run_episode", side_effect=fake_run_episode), \
             mock.patch.object(sim, "evaluate_behavior_suite", side_effect=fake_evaluate_behavior_suite):
            training_history = sim._execute_training_schedule(
                episodes=12,
                curriculum_profile="ecological_v1",
            )

        self.assertEqual(len(training_history), 12)
        curriculum = sim._latest_curriculum_summary
        self.assertIsNotNone(curriculum)
        phase_1 = curriculum["phases"][0]
        phase_2 = curriculum["phases"][1]
        phase_4 = curriculum["phases"][3]
        self.assertEqual(phase_1["status"], "promoted")
        self.assertEqual(phase_1["allocated_episodes"], 2)
        self.assertEqual(phase_1["episodes_executed"], 1)
        self.assertEqual(
            phase_1["promotion_checks"][0]["promotion_reason"],
            "threshold_fallback",
        )
        self.assertEqual(phase_1["promotion_checks"][0]["check_results"], {})
        self.assertEqual(phase_1["promotion_reason"], "threshold_fallback")
        self.assertEqual(phase_1["final_check_results"], {})
        self.assertEqual(phase_2["carryover_in"], 1)
        self.assertEqual(phase_2["allocated_episodes"], 3)
        self.assertEqual(phase_4["carryover_in"], 0)
        self.assertEqual(phase_4["allocated_episodes"], phase_4["max_episodes"])
        self.assertEqual(curriculum["executed_training_episodes"], 12)

    def test_train_curriculum_promotes_with_explicit_check_specs(self) -> None:
        """
        Verifies that a curriculum phase with explicit PromotionCheckCriteria is promoted when evaluation checks meet the required pass rates.
        
        Asserts the stored curriculum phase record includes the provided `skill_name` and `promotion_check_specs`, that the promotion attempt records passed checks and `promotion_reason == "all_checks_passed"`, and that `final_check_results` mirrors the promotion attempt's check results.
        """
        sim = SpiderSimulation(seed=7, max_steps=10)
        criteria = PromotionCheckCriteria(
            scenario="food_deprivation",
            check_name="hunger_reduced",
            required_pass_rate=1.0,
        )
        phase = CurriculumPhaseDefinition(
            name="phase_hunger",
            training_scenarios=("food_deprivation",),
            promotion_scenarios=("food_deprivation",),
            success_threshold=1.0,
            max_episodes=1,
            min_episodes=1,
            skill_name="hunger_commitment",
            promotion_check_specs=(criteria,),
        )
        fake_run_episode = self._fake_run_episode

        payload = {
            "summary": {
                "scenario_success_rate": 0.0,
                "episode_success_rate": 0.0,
            },
            "suite": {
                "food_deprivation": {
                    "checks": {
                        "hunger_reduced": {"pass_rate": 1.0},
                    },
                }
            },
        }
        with mock.patch.object(
            SpiderSimulation,
            "_resolve_curriculum_profile",
            return_value=[phase],
        ), mock.patch.object(
            sim, "run_episode", side_effect=fake_run_episode
        ), mock.patch.object(
            sim, "evaluate_behavior_suite", return_value=(payload, [], [])
        ):
            sim._execute_training_schedule(
                episodes=1,
                curriculum_profile="ecological_v1",
            )

        curriculum = sim._latest_curriculum_summary
        self.assertIsNotNone(curriculum)
        phase_record = curriculum["phases"][0]
        self.assertEqual(phase_record["status"], "promoted")
        self.assertEqual(phase_record["skill_name"], "hunger_commitment")
        self.assertEqual(
            phase_record["promotion_check_specs"],
            [
                {
                    "scenario": "food_deprivation",
                    "check_name": "hunger_reduced",
                    "required_pass_rate": 1.0,
                    "aggregation": "all",
                }
            ],
        )
        promotion_attempt = phase_record["promotion_checks"][0]
        self.assertTrue(promotion_attempt["promotion_criteria_passed"])
        self.assertEqual(
            promotion_attempt["promotion_reason"],
            "all_checks_passed",
        )
        self.assertEqual(
            promotion_attempt["check_results"],
            {
                "food_deprivation": {
                    "hunger_reduced": {
                        "scenario": "food_deprivation",
                        "pass_rate": 1.0,
                        "required": 1.0,
                        "passed": True,
                    }
                }
            },
        )
        self.assertEqual(phase_record["promotion_reason"], "all_checks_passed")
        self.assertEqual(
            phase_record["final_check_results"],
            promotion_attempt["check_results"],
        )

    def test_train_curriculum_explicit_check_failure_blocks_aggregate_success(self) -> None:
        sim = SpiderSimulation(seed=7, max_steps=10)
        criteria = PromotionCheckCriteria(
            scenario="food_deprivation",
            check_name="hunger_reduced",
            required_pass_rate=1.0,
        )
        phase = CurriculumPhaseDefinition(
            name="phase_hunger",
            training_scenarios=("food_deprivation",),
            promotion_scenarios=("food_deprivation",),
            success_threshold=0.0,
            max_episodes=1,
            min_episodes=1,
            skill_name="hunger_commitment",
            promotion_check_specs=(criteria,),
        )
        fake_run_episode = self._fake_run_episode

        payload = {
            "summary": {
                "scenario_success_rate": 1.0,
                "episode_success_rate": 1.0,
            },
            "suite": {
                "food_deprivation": {
                    "checks": {
                        "hunger_reduced": {"pass_rate": 0.0},
                    },
                }
            },
        }
        with mock.patch.object(
            SpiderSimulation,
            "_resolve_curriculum_profile",
            return_value=[phase],
        ), mock.patch.object(
            sim, "run_episode", side_effect=fake_run_episode
        ), mock.patch.object(
            sim, "evaluate_behavior_suite", return_value=(payload, [], [])
        ):
            sim._execute_training_schedule(
                episodes=1,
                curriculum_profile="ecological_v1",
            )

        curriculum = sim._latest_curriculum_summary
        self.assertIsNotNone(curriculum)
        phase_record = curriculum["phases"][0]
        self.assertEqual(phase_record["status"], "max_budget_exhausted")
        self.assertEqual(phase_record["promotion_reason"], "check_failed:hunger_reduced")
        promotion_attempt = phase_record["promotion_checks"][0]
        self.assertEqual(promotion_attempt["scenario_success_rate"], 1.0)
        self.assertFalse(promotion_attempt["promotion_criteria_passed"])
        self.assertEqual(
            phase_record["final_check_results"],
            {
                "food_deprivation": {
                    "hunger_reduced": {
                        "scenario": "food_deprivation",
                        "pass_rate": 0.0,
                        "required": 1.0,
                        "passed": False,
                    }
                }
            },
        )

    def test_train_curriculum_final_phase_absorbs_carried_budget(self) -> None:
        sim = SpiderSimulation(seed=9, max_steps=10)
        fake_run_episode = self._fake_run_episode

        eval_payloads = [
            {"summary": {"scenario_success_rate": 1.0, "episode_success_rate": 1.0}, "suite": {}},
            {"summary": {"scenario_success_rate": 1.0, "episode_success_rate": 1.0}, "suite": {}},
            {"summary": {"scenario_success_rate": 1.0, "episode_success_rate": 1.0}, "suite": {}},
            {"summary": {"scenario_success_rate": 1.0, "episode_success_rate": 1.0}, "suite": {}},
        ]

        with mock.patch.object(sim, "run_episode", side_effect=fake_run_episode), \
             mock.patch.object(sim, "evaluate_behavior_suite", side_effect=[
                 (payload, [], []) for payload in eval_payloads
             ]):
            training_history = sim._execute_training_schedule(
                episodes=12,
                curriculum_profile="ecological_v1",
            )

        self.assertEqual(len(training_history), 12)
        curriculum = sim._latest_curriculum_summary
        self.assertIsNotNone(curriculum)
        phase_4 = curriculum["phases"][3]
        self.assertEqual(phase_4["status"], "promoted")
        self.assertEqual(phase_4["carryover_in"], 4)
        self.assertEqual(phase_4["allocated_episodes"], 8)
        self.assertEqual(phase_4["episodes_executed"], 8)
        self.assertEqual(curriculum["executed_training_episodes"], 12)

    def test_train_curriculum_phase_status_is_valid_value(self) -> None:
        """All phase statuses must be one of the known terminal values."""
        valid_statuses = {"promoted", "max_budget_exhausted", "skipped_no_budget"}
        sim = SpiderSimulation(seed=7, max_steps=10)
        summary, _ = sim.train(
            episodes=6,
            evaluation_episodes=0,
            capture_evaluation_trace=False,
            curriculum_profile="ecological_v1",
        )
        for phase in summary["curriculum"]["phases"]:
            self.assertIn(phase["status"], valid_statuses)

    def test_train_curriculum_csv_rows_contain_regime_columns(self) -> None:
        sim = SpiderSimulation(seed=7, max_steps=10)
        _, rows = sim.train(
            episodes=6,
            evaluation_episodes=1,
            capture_evaluation_trace=False,
            curriculum_profile="ecological_v1",
        )
        for row in rows:
            self.assertIn("training_regime", row)
            self.assertIn("curriculum_profile", row)
            self.assertIn("curriculum_phase", row)
            self.assertIn("curriculum_skill", row)
            self.assertIn("curriculum_phase_status", row)
            self.assertIn("curriculum_promotion_reason", row)

    def test_train_flat_csv_rows_training_regime_is_flat(self) -> None:
        sim = SpiderSimulation(seed=7, max_steps=10)
        _, rows = sim.train(
            episodes=2,
            evaluation_episodes=1,
            capture_evaluation_trace=False,
        )
        for row in rows:
            self.assertEqual(row["training_regime"], "flat")
            self.assertEqual(row["curriculum_profile"], "none")
            self.assertEqual(row["curriculum_phase"], "")
            self.assertEqual(row["curriculum_skill"], "")
            self.assertEqual(row["curriculum_phase_status"], "")
            self.assertEqual(row["curriculum_promotion_reason"], "")
