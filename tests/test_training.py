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


class SpiderTrainingTest(unittest.TestCase):
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

    def test_online_learning_updates_parameters(self) -> None:
        sim = SpiderSimulation(seed=7, max_steps=20)
        obs = sim.world.reset(seed=7)
        decision = sim.brain.act(obs, sim.bus, sample=True)
        before = sim.brain.motor_cortex.W1.copy()
        next_obs, reward, done, _ = sim.world.step(decision.action_idx)
        sim.brain.learn(decision, reward, next_obs, done)
        self.assertFalse(np.allclose(before, sim.brain.motor_cortex.W1))

    def test_classic_training_keeps_metrics_and_minimal_foraging(self) -> None:
        """
        Verify that training with the "classic" reward profile produces finite metrics, demonstrates minimal foraging, and records correct configuration and budget resolution.
        
        This test runs a short training session with the classic reward profile on the central_burrow map and asserts:
        - evaluation metrics are finite,
        - mean food collected and feeding reward component are positive,
        - presence of predator- and sleep-related evaluation fields (sleep debt, predator state ticks, night role distribution, dominant predator state, mode transitions),
        - the summary's config reflects the requested world/reward/map, architecture metadata, operational profile, and a default noise_profile of "none",
        - the resolved budget in the summary records the requested episodes, evaluation episodes, and max_steps.
        """
        brain_config = BrainAblationConfig(
            name="classic_legacy_foraging_smoke",
            use_learned_arbitration=True,
            enable_deterministic_guards=True,
            enable_food_direction_bias=True,
            warm_start_scale=1.0,
            gate_adjustment_bounds=(0.5, 1.5),
            credit_strategy="broadcast",
        )
        sim = SpiderSimulation(
            seed=7,
            max_steps=90,
            reward_profile="classic",
            map_template="central_burrow",
            brain_config=brain_config,
        )
        for resolved_config in (sim.brain_config, sim.brain.config):
            self.assertTrue(resolved_config.use_learned_arbitration)
            self.assertTrue(resolved_config.enable_deterministic_guards)
            self.assertTrue(resolved_config.enable_food_direction_bias)
            self.assertAlmostEqual(resolved_config.warm_start_scale, 1.0)
            self.assertEqual(resolved_config.gate_adjustment_bounds, (0.5, 1.5))
            self.assertEqual(resolved_config.credit_strategy, "broadcast")
        summary, _ = sim.train(episodes=20, evaluation_episodes=3, capture_evaluation_trace=False)
        evaluation = summary["evaluation"]

        self._assert_finite_summary(summary)
        self.assertGreater(evaluation["mean_food"], 0.0)
        self.assertIn("mean_sleep_debt", evaluation)
        self.assertIn("mean_reward_components", evaluation)
        self.assertIn("mean_predator_state_ticks", evaluation)
        self.assertIn("mean_night_role_distribution", evaluation)
        self.assertIn("dominant_predator_state", evaluation)
        self.assertIn("mean_predator_mode_transitions", evaluation)
        self.assertIn("mean_motor_slip_rate", evaluation)
        self.assertIn("mean_orientation_alignment", evaluation)
        self.assertIn("mean_terrain_difficulty", evaluation)
        self.assertIn("motor_stage", summary)
        self.assertIn("evaluation", summary["motor_stage"])
        self.assertIn("mean_motor_slip_rate", summary["motor_stage"]["evaluation"])
        self.assertGreater(evaluation["mean_reward_components"]["feeding"], 0.0)
        self.assertEqual(summary["config"]["world"]["reward_profile"], "classic")
        self.assertEqual(summary["config"]["world"]["map_template"], "central_burrow")
        self.assertEqual(summary["config"]["architecture_version"], SpiderBrain.ARCHITECTURE_VERSION)
        self.assertTrue(summary["config"]["architecture_fingerprint"])
        self.assertEqual(summary["config"]["operational_profile"]["name"], "default_v1")
        self.assertEqual(summary["config"]["operational_profile"]["version"], 1)
        self.assertEqual(summary["config"]["noise_profile"]["name"], "none")
        self.assertEqual(summary["config"]["budget"]["profile"], "custom")
        self.assertEqual(summary["config"]["budget"]["benchmark_strength"], "custom")
        self.assertEqual(summary["config"]["budget"]["resolved"]["episodes"], 20)
        self.assertEqual(summary["config"]["budget"]["resolved"]["eval_episodes"], 3)
        self.assertEqual(summary["config"]["budget"]["resolved"]["max_steps"], 90)

    def test_ecological_profile_reduces_progress_shaping_in_scripted_transition(self) -> None:
        """
        Asserts that an ecological reward profile yields less food-progress shaping than the classic profile for a controlled MOVE_RIGHT transition.
        
        Creates two SpiderWorld instances with the same deterministic state (spider position, food position, lizard position, and high hunger), performs a single MOVE_RIGHT step in each, and verifies that the classic world produces a larger `reward_components["food_progress"]` than the ecological world. Also verifies the ecological world's `reward_profile` remains `"ecological"`.
        """
        classic = SpiderWorld(seed=11, reward_profile="classic", lizard_move_interval=999999)
        ecological = SpiderWorld(seed=11, reward_profile="ecological", lizard_move_interval=999999)
        classic.reset(seed=11)
        ecological.reset(seed=11)

        classic.state.x, classic.state.y = 1, 1
        ecological.state.x, ecological.state.y = 1, 1
        classic.food_positions = [(3, 1)]
        ecological.food_positions = [(3, 1)]
        classic.lizard.x, classic.lizard.y = 0, classic.height - 1
        ecological.lizard.x, ecological.lizard.y = 0, ecological.height - 1
        classic.state.hunger = 0.9
        ecological.state.hunger = 0.9

        _, _, _, classic_info = classic.step(ACTION_TO_INDEX["MOVE_RIGHT"])
        _, _, _, ecological_info = ecological.step(ACTION_TO_INDEX["MOVE_RIGHT"])

        self.assertGreater(classic_info["reward_components"]["food_progress"], ecological_info["reward_components"]["food_progress"])
        self.assertEqual(ecological.reward_profile, "ecological")

    def test_non_default_map_stays_runnable(self) -> None:
        sim = SpiderSimulation(seed=13, max_steps=80, map_template="two_shelters")
        summary, _ = sim.train(episodes=10, evaluation_episodes=2, capture_evaluation_trace=False)

        self._assert_finite_summary(summary)
        self.assertEqual(summary["config"]["world"]["map_template"], "two_shelters")
        self.assertIn("mean_reward", summary["evaluation"])
        self.assertIn("mean_predator_state_occupancy", summary["evaluation"])

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

    # ---------------------------------------------------------------------------
    # _set_training_regime_metadata (instance method)
    # ---------------------------------------------------------------------------

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

    # ---------------------------------------------------------------------------
    # _execute_training_schedule
    # ---------------------------------------------------------------------------

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

    # ---------------------------------------------------------------------------
    # Train with flat regime — regression tests
    # ---------------------------------------------------------------------------

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

    def test_scenario_runner_reports_named_results(self) -> None:
        """
        Verify that run_scenarios returns a mapping keyed by the provided scenario names and that each scenario's results include expected evaluation metrics; also confirm a non-empty trace is produced whose final entry contains required debug fields.
        
        Runs the scenarios "night_rest", "recover_after_failed_chase", and "two_shelter_tradeoff" and asserts:
        - The result mapping keys exactly match the provided scenario names.
        - "night_rest" includes "mean_reward".
        - "recover_after_failed_chase" includes "mean_reward_components" and "mean_predator_mode_transitions".
        - "two_shelter_tradeoff" includes "dominant_predator_state".
        - The returned trace is non-empty and the final trace entry contains "debug", "distance_deltas", "predator_escape", and "event_log".
        """
        sim = SpiderSimulation(seed=17, max_steps=40)
        results, trace = sim.run_scenarios(
            ["night_rest", "recover_after_failed_chase", "two_shelter_tradeoff"],
            capture_trace=True,
            debug_trace=True,
        )

        self.assertEqual(set(results.keys()), {"night_rest", "recover_after_failed_chase", "two_shelter_tradeoff"})
        self.assertIn("mean_reward", results["night_rest"])
        self.assertIn("mean_reward_components", results["recover_after_failed_chase"])
        self.assertIn("dominant_predator_state", results["two_shelter_tradeoff"])
        self.assertIn("mean_predator_mode_transitions", results["recover_after_failed_chase"])
        self.assertTrue(trace)
        self.assertIn("debug", trace[-1])
        self.assertIn("distance_deltas", trace[-1])
        self.assertIn("predator_escape", trace[-1])
        self.assertIn("event_log", trace[-1])
        self.assertIsInstance(trace[-1]["event_log"], list)
        self.assertTrue(trace[-1]["event_log"])
        self.assertIsInstance(trace[-1]["event_log"][0], dict)
        self.assertIn("stage", trace[-1]["event_log"][0])
        self.assertIn("name", trace[-1]["event_log"][0])
        self.assertIn("payload", trace[-1]["event_log"][0])

    def test_behavior_suite_reports_scorecards_and_legacy_metrics(self) -> None:
        """
        Verify evaluate_behavior_suite returns complete scorecards, legacy scenario metadata, annotated rows, and a populated trace.
        
        Asserts the returned payload contains "suite", "summary", and "legacy_scenarios"; that the suite keys match the requested scenarios; that the "night_rest" scorecard includes success rate, checks (including "deep_night_shelter"), behavior metrics, failures, and legacy metrics; that returned rows are non-empty and the first row contains expected metadata (reward_profile "classic", scenario_map and evaluation_map "central_burrow", correct architecture version and non-empty architecture_fingerprint, operational_profile "default_v1" version 1, noise_profile "none", and a noise_profile_config); and that the trace is non-empty with the final trace entry containing a "debug" field.
        """
        sim = SpiderSimulation(seed=23, max_steps=20)
        payload, trace, rows = sim.evaluate_behavior_suite(
            ["night_rest", "food_deprivation"],
            capture_trace=True,
            debug_trace=True,
        )

        self.assertIn("suite", payload)
        self.assertIn("summary", payload)
        self.assertIn("legacy_scenarios", payload)
        self.assertEqual(set(payload["suite"].keys()), {"night_rest", "food_deprivation"})
        night_rest = payload["suite"]["night_rest"]
        self.assertIn("success_rate", night_rest)
        self.assertIn("checks", night_rest)
        self.assertIn("behavior_metrics", night_rest)
        self.assertIn("failures", night_rest)
        self.assertIn("legacy_metrics", night_rest)
        self.assertIn("deep_night_shelter", night_rest["checks"])
        self.assertTrue(rows)
        self.assertEqual(rows[0]["reward_profile"], "classic")
        self.assertEqual(rows[0]["scenario_map"], "central_burrow")
        self.assertEqual(rows[0]["evaluation_map"], "central_burrow")
        self.assertEqual(rows[0]["architecture_version"], SpiderBrain.ARCHITECTURE_VERSION)
        self.assertTrue(rows[0]["architecture_fingerprint"])
        self.assertEqual(rows[0]["operational_profile"], "default_v1")
        self.assertEqual(rows[0]["operational_profile_version"], 1)
        self.assertEqual(rows[0]["noise_profile"], "none")
        self.assertIn("noise_profile_config", rows[0])
        self.assertTrue(trace)
        self.assertIn("debug", trace[-1])

    def test_behavior_suite_summary_only_skips_rows(self) -> None:
        sim = SpiderSimulation(seed=23, max_steps=20)
        payload, trace, rows = sim.evaluate_behavior_suite(
            ["night_rest"],
            summary_only=True,
        )

        self.assertIn("summary", payload)
        self.assertIn("night_rest", payload["suite"])
        self.assertEqual(trace, [])
        self.assertEqual(rows, [])

    def test_behavior_suite_is_reproducible_for_same_seed(self) -> None:
        sim_a = SpiderSimulation(seed=29, max_steps=16)
        sim_b = SpiderSimulation(seed=29, max_steps=16)

        payload_a, _, rows_a = sim_a.evaluate_behavior_suite(["night_rest"])
        payload_b, _, rows_b = sim_b.evaluate_behavior_suite(["night_rest"])

        self.assertEqual(
            payload_a["suite"]["night_rest"]["success_rate"],
            payload_b["suite"]["night_rest"]["success_rate"],
        )
        self.assertEqual(
            payload_a["suite"]["night_rest"]["checks"],
            payload_b["suite"]["night_rest"]["checks"],
        )
        self.assertEqual(rows_a, rows_b)

    def test_behavior_suite_is_reproducible_for_same_seed_with_noise_profile(self) -> None:
        sim_a = SpiderSimulation(seed=29, max_steps=16, noise_profile="low")
        sim_b = SpiderSimulation(seed=29, max_steps=16, noise_profile="low")

        payload_a, _, rows_a = sim_a.evaluate_behavior_suite(["night_rest"])
        payload_b, _, rows_b = sim_b.evaluate_behavior_suite(["night_rest"])

        self.assertEqual(payload_a["suite"]["night_rest"]["checks"], payload_b["suite"]["night_rest"]["checks"])
        self.assertEqual(rows_a, rows_b)

    def test_monolithic_policy_training_keeps_metrics_finite(self) -> None:
        config = canonical_ablation_configs(module_dropout=0.0)["monolithic_policy"]
        sim = SpiderSimulation(
            seed=31,
            max_steps=60,
            brain_config=config,
        )
        summary, _ = sim.train(episodes=8, evaluation_episodes=2, capture_evaluation_trace=False)

        self._assert_finite_summary(summary)
        self.assertEqual(summary["config"]["brain"]["architecture"], "monolithic")
        self.assertEqual(summary["config"]["brain"]["name"], "monolithic_policy")
        self.assertIn("monolithic_policy", summary["parameter_norms"])
        self.assertIn("action_center", summary["parameter_norms"])
        self.assertIn("motor_cortex", summary["parameter_norms"])

    def test_run_episode_rejects_training_with_non_normal_policy_mode_early(self) -> None:
        sim = SpiderSimulation(seed=7, max_steps=10)
        sim.world.reset(seed=7)
        before = sim.world.state_dict()

        with self.assertRaisesRegex(ValueError, "training=True.*policy_mode='normal'"):
            sim.run_episode(0, training=True, sample=False, policy_mode="reflex_only")

        self.assertEqual(sim.world.state_dict(), before)
    def test_best_checkpoint_selection_restores_selected_checkpoint(self) -> None:
        """
        Verifies that selecting the "best" checkpoint during training persists, restores, and annotates the chosen checkpoint.
        
        Creates scripted checkpoint candidates with differing evaluation metrics, runs training with checkpoint selection set to "best", and asserts:
        - the training summary records the checkpointing selection, metric, and evaluation reflex scale,
        - the selected checkpoint is the one with the highest metric and its metadata (episode and eval_reflex_scale) is recorded,
        - restoring from the saved "best" and "last" checkpoints reproduces the expected model parameters (best matches the post-selection model, last differs),
        - behavior evaluations are annotated with checkpoint_source == "best".
        """
        sim = SpiderSimulation(seed=61, max_steps=12)

        def scripted_capture_checkpoint(
            *,
            root_dir: Path,
            episode: int,
            scenario_names: Sequence[str],
            metric: str,
            selection_scenario_episodes: int,
            eval_reflex_scale: Optional[float] = None,
        ) -> dict[str, object]:
            del scenario_names, metric, selection_scenario_episodes, eval_reflex_scale
            checkpoint_name = f"episode_{int(episode):05d}"
            checkpoint_path = root_dir / checkpoint_name
            sim.brain.save(checkpoint_path)
            score = 0.9 if int(episode) == 1 else 0.1
            return {
                "name": checkpoint_name,
                "episode": int(episode),
                "path": checkpoint_path,
                "metric": "scenario_success_rate",
                "scenario_success_rate": score,
                "episode_success_rate": score,
                "mean_reward": score,
            }

        sim._capture_checkpoint_candidate = scripted_capture_checkpoint  # type: ignore[method-assign]

        with tempfile.TemporaryDirectory() as tmpdir:
            summary, _ = sim.train(
                2,
                evaluation_episodes=0,
                capture_evaluation_trace=False,
                checkpoint_selection="best",
                checkpoint_metric="scenario_success_rate",
                checkpoint_interval=1,
                checkpoint_dir=Path(tmpdir),
                checkpoint_scenario_names=["night_rest"],
                selection_scenario_episodes=1,
            )

            self.assertIn("checkpointing", summary)
            self.assertEqual(summary["checkpointing"]["selection"], "best")
            self.assertEqual(summary["checkpointing"]["metric"], "scenario_success_rate")
            self.assertEqual(summary["checkpointing"]["eval_reflex_scale"], 0.0)
            self.assertEqual(summary["checkpointing"]["selected_checkpoint"]["episode"], 1)
            self.assertEqual(
                summary["checkpointing"]["selected_checkpoint"]["eval_reflex_scale"],
                0.0,
            )

            best_sim = SpiderSimulation(seed=61, max_steps=12)
            best_sim.brain.load(Path(tmpdir) / "best")
            last_sim = SpiderSimulation(seed=61, max_steps=12)
            last_sim.brain.load(Path(tmpdir) / "last")

            np.testing.assert_allclose(sim.brain.motor_cortex.W1, best_sim.brain.motor_cortex.W1)
            self.assertFalse(
                np.allclose(sim.brain.motor_cortex.W1, last_sim.brain.motor_cortex.W1)
            )

            _, _, rows = sim.evaluate_behavior_suite(["night_rest"])
            self.assertEqual(rows[0]["checkpoint_source"], "best")

    def test_direct_checkpoint_penalty_records_composite_scores(self) -> None:
        sim = SpiderSimulation(seed=62, max_steps=5)

        def scripted_capture_checkpoint(
            *,
            root_dir: Path,
            episode: int,
            scenario_names: Sequence[str],
            metric: str,
            selection_scenario_episodes: int,
            eval_reflex_scale: Optional[float] = None,
        ) -> dict[str, object]:
            """
            Persist a scripted checkpoint to disk and return a synthetic checkpoint-candidate metadata dictionary used by tests.
            
            Parameters:
                root_dir (Path): Directory where the checkpoint subdirectory will be created.
                episode (int): Episode index used to name the checkpoint (formatted as "episode_XXXXX").
                scenario_names: Ignored in this scripted implementation (kept for API compatibility).
                metric (str): Ignored; the returned candidate uses the fixed `"scenario_success_rate"` metric.
                selection_scenario_episodes (int): Ignored in this scripted implementation.
                eval_reflex_scale (float | None): Ignored in this scripted implementation.
            
            Returns:
                dict: A checkpoint-candidate mapping with these keys:
                    - name (str): Checkpoint directory name (e.g., "episode_00001").
                    - episode (int): Episode index.
                    - path (Path): Filesystem path to the saved checkpoint directory.
                    - metric (str): Primary metric name (`"scenario_success_rate"`).
                    - scenario_success_rate (float): Simulated scenario success rate (0.8 for episode 1, 0.7 otherwise).
                    - episode_success_rate (float): Simulated episode success rate (0.5).
                    - mean_reward (float): Simulated mean reward (0.25).
                    - evaluation_summary (dict): Nested evaluation info containing:
                        - mean_final_reflex_override_rate (float)
                        - mean_reflex_dominance (float)
            """
            del scenario_names, metric, selection_scenario_episodes, eval_reflex_scale
            checkpoint_name = f"episode_{int(episode):05d}"
            checkpoint_path = root_dir / checkpoint_name
            sim.brain.save(checkpoint_path)
            if int(episode) == 1:
                scenario_success_rate = 0.8
                override_rate = 0.9
            else:
                scenario_success_rate = 0.7
                override_rate = 0.0
            return {
                "name": checkpoint_name,
                "episode": int(episode),
                "path": checkpoint_path,
                "metric": "scenario_success_rate",
                "scenario_success_rate": scenario_success_rate,
                "episode_success_rate": 0.5,
                "mean_reward": 0.25,
                "evaluation_summary": {
                    "mean_final_reflex_override_rate": override_rate,
                    "mean_reflex_dominance": 0.0,
                },
            }

        sim._capture_checkpoint_candidate = scripted_capture_checkpoint  # type: ignore[method-assign]

        summary, _ = sim.train(
            2,
            evaluation_episodes=0,
            capture_evaluation_trace=False,
            checkpoint_selection="best",
            checkpoint_metric="scenario_success_rate",
            checkpoint_interval=1,
            checkpoint_scenario_names=["night_rest"],
            selection_scenario_episodes=1,
            checkpoint_override_penalty=1.0,
            checkpoint_penalty_mode="direct",
        )

        checkpointing = summary["checkpointing"]
        self.assertEqual(checkpointing["penalty_mode"], "direct")
        self.assertEqual(
            checkpointing["penalty_config"],
            {
                "metric": "scenario_success_rate",
                "override_penalty_weight": 1.0,
                "dominance_penalty_weight": 0.0,
                "penalty_mode": "direct",
            },
        )
        self.assertEqual(checkpointing["selected_checkpoint"]["episode"], 2)
        self.assertAlmostEqual(
            checkpointing["selected_checkpoint"]["composite_score"],
            0.7,
        )
        generated = checkpointing["generated_checkpoints"]
        self.assertTrue(all("composite_score" in candidate for candidate in generated))
        first_candidate = next(
            candidate for candidate in generated if candidate["episode"] == 1
        )
        self.assertAlmostEqual(first_candidate["composite_score"], -0.1)

    def test_best_checkpoint_selection_keeps_post_training_eval_offset(self) -> None:
        sim = SpiderSimulation(seed=71, max_steps=6)
        original_run_episode = sim.run_episode
        recorded_eval_indices: list[int] = []

        def wrapped_run_episode(
            episode_index: int,
            *,
            training: bool,
            sample: bool,
            render: bool = False,
            capture_trace: bool = False,
            scenario_name: str | None = None,
            debug_trace: bool = False,
            policy_mode: str = "normal",
        ):
            if not training and scenario_name is None:
                recorded_eval_indices.append(int(episode_index))
            return original_run_episode(
                episode_index,
                training=training,
                sample=sample,
                render=render,
                capture_trace=capture_trace,
                scenario_name=scenario_name,
                debug_trace=debug_trace,
                policy_mode=policy_mode,
            )

        def scripted_capture_checkpoint(
            *,
            root_dir: Path,
            episode: int,
            scenario_names,
            metric: str,
            selection_scenario_episodes: int,
            eval_reflex_scale: float | None = None,
        ):
            del scenario_names, metric, selection_scenario_episodes, eval_reflex_scale
            checkpoint_name = f"episode_{int(episode):05d}"
            checkpoint_path = root_dir / checkpoint_name
            sim.brain.save(checkpoint_path)
            return {
                "name": checkpoint_name,
                "episode": int(episode),
                "path": checkpoint_path,
                "metric": "scenario_success_rate",
                "scenario_success_rate": float(episode),
                "episode_success_rate": float(episode),
                "mean_reward": float(episode),
            }

        sim.run_episode = wrapped_run_episode  # type: ignore[method-assign]
        sim._capture_checkpoint_candidate = scripted_capture_checkpoint  # type: ignore[method-assign]

        sim.train(
            2,
            evaluation_episodes=2,
            capture_evaluation_trace=False,
            checkpoint_selection="best",
            checkpoint_interval=1,
            checkpoint_scenario_names=["night_rest"],
            selection_scenario_episodes=1,
        )

        self.assertEqual(recorded_eval_indices, [2, 3, 2, 3])

    def test_checkpoint_candidate_scoring_uses_no_reflex_scale(self) -> None:
        sim = SpiderSimulation(seed=7, max_steps=10)
        recorded_eval_scales: list[float | None] = []
        original_capture = sim._capture_checkpoint_candidate

        def wrapped_capture_checkpoint(
            *,
            root_dir: Path,
            episode: int,
            scenario_names,
            metric: str,
            selection_scenario_episodes: int,
            eval_reflex_scale: float | None = None,
        ):
            recorded_eval_scales.append(eval_reflex_scale)
            return original_capture(
                root_dir=root_dir,
                episode=episode,
                scenario_names=scenario_names,
                metric=metric,
                selection_scenario_episodes=selection_scenario_episodes,
                eval_reflex_scale=eval_reflex_scale,
            )

        sim._capture_checkpoint_candidate = wrapped_capture_checkpoint  # type: ignore[method-assign]

        sim.train(
            3,
            evaluation_episodes=0,
            capture_evaluation_trace=False,
            checkpoint_selection="best",
            checkpoint_interval=1,
            checkpoint_scenario_names=["night_rest"],
            selection_scenario_episodes=1,
            reflex_anneal_final_scale=0.25,
        )

        self.assertTrue(recorded_eval_scales)
        for value in recorded_eval_scales:
            self.assertAlmostEqual(float(value), 0.0)

    def test_cli_invalid_reflex_scale_validation_is_reported_as_usage_error(self) -> None:
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "spider_cortex_sim",
                "--reflex-scale",
                "nan",
            ],
            cwd=Path(__file__).resolve().parents[1],
            env={**os.environ, "PYTHONPATH": "."},
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("--reflex-scale", proc.stderr)
        self.assertRegex(proc.stderr.lower(), r"finit")

    def test_cli_brain_config_validation_is_reported_as_usage_error(self) -> None:
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "spider_cortex_sim",
                "--module-reflex-scale",
                "motor_cortex_context=0.5",
            ],
            cwd=Path(__file__).resolve().parents[1],
            env={**os.environ, "PYTHONPATH": "."},
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("module_reflex_scales", proc.stderr)
        self.assertIn("invalid", proc.stderr.lower())

    def test_cli_rejects_custom_reflex_flags_for_ablation_workflows(self) -> None:
        """
        Verify the CLI rejects custom reflex flags when running an ablation workflow.
        
        Runs the package CLI with an ablation variant and a custom `--reflex-scale`, then asserts the process exits with a non-zero status and that stderr indicates the ablation context does not support the custom reflex flag.
        """
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "spider_cortex_sim",
                "--episodes",
                "0",
                "--eval-episodes",
                "0",
                "--ablation-variant",
                "monolithic_policy",
                "--reflex-scale",
                "0.5",
            ],
            cwd=Path(__file__).resolve().parents[1],
            env={**os.environ, "PYTHONPATH": "."},
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertNotEqual(proc.returncode, 0)
        self.assertRegex(proc.stderr.lower(), r"ablation|abla")
        self.assertRegex(proc.stderr.lower(), r"suport|support")

    def test_cli_rejects_custom_reflex_flags_for_claim_test_suite(self) -> None:
        for claim_args in (
            ["--claim-test-suite"],
            ["--claim-test", "learning_without_privileged_signals"],
        ):
            with self.subTest(claim_args=claim_args):
                proc = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "spider_cortex_sim",
                        "--episodes",
                        "0",
                        "--eval-episodes",
                        "0",
                        *claim_args,
                        "--reflex-scale",
                        "0.5",
                    ],
                    cwd=Path(__file__).resolve().parents[1],
                    env={**os.environ, "PYTHONPATH": "."},
                    text=True,
                    capture_output=True,
                    check=False,
                )

                self.assertNotEqual(proc.returncode, 0)
                self.assertIn("--claim-test-suite", proc.stderr)
                self.assertRegex(proc.stderr.lower(), r"suport|support")

    def test_cli_behavior_suite_emits_json_and_csv(self) -> None:
        """
        Run the package CLI and assert it emits a behavior-evaluation JSON payload and writes a CSV with expected headers.
        
        Executes the CLI for a minimal run (no training episodes) requesting the `night_rest` behavior scenario and a full summary, then:
        - Asserts the stdout JSON contains a top-level `behavior_evaluation` with a `suite` entry that includes `night_rest`.
        - Asserts a CSV file was created and its header row contains key columns including `scenario`, `scenario_map`, `evaluation_map`, `budget_profile`, `benchmark_strength`, `architecture_version`, `architecture_fingerprint`, `operational_profile`, `operational_profile_version`, `noise_profile`, `checkpoint_source`, and `check_deep_night_shelter_passed`.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "behavior.csv"
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "spider_cortex_sim",
                    "--episodes",
                    "0",
                    "--eval-episodes",
                    "0",
                    "--behavior-scenario",
                    "night_rest",
                    "--budget-profile",
                    "smoke",
                    "--full-summary",
                    "--behavior-csv",
                    str(csv_path),
                ],
                cwd=Path(__file__).resolve().parents[1],
                env={**os.environ, "PYTHONPATH": "."},
                text=True,
                capture_output=True,
                check=True,
            )

            payload = json.loads(proc.stdout)
            self.assertIn("behavior_evaluation", payload)
            self.assertIn("night_rest", payload["behavior_evaluation"]["suite"])
            self.assertTrue(csv_path.exists())
            header = csv_path.read_text(encoding="utf-8").splitlines()[0]
            self.assertIn("scenario", header)
            self.assertIn("scenario_map", header)
            self.assertIn("evaluation_map", header)
            self.assertIn("budget_profile", header)
            self.assertIn("benchmark_strength", header)
            self.assertIn("architecture_version", header)
            self.assertIn("architecture_fingerprint", header)
            self.assertIn("operational_profile", header)
            self.assertIn("operational_profile_version", header)
            self.assertIn("noise_profile", header)
            self.assertIn("checkpoint_source", header)
            self.assertIn("check_deep_night_shelter_passed", header)

    def test_cli_ablation_variant_emits_summary_and_csv_columns(self) -> None:
        """
        Verify the CLI emits an ablation summary JSON and a CSV containing expected ablation and metadata columns.
        
        Asserts the JSON payload includes ablation metadata: `reference_variant` equal to `"modular_full"`, `budget_profile` and `benchmark_strength` resolved for the smoke profile, `seeds` equal to `[7]`, `scenario_names` matching SCENARIO_NAMES, presence of the requested `monolithic_policy` variant, and an empty `suite`. Asserts the generated CSV exists and its header contains ablation and metadata columns including: `ablation_variant`, `ablation_architecture`, `budget_profile`, `benchmark_strength`, `architecture_version`, `architecture_fingerprint`, `operational_profile`, `operational_profile_version`, `noise_profile`, and `checkpoint_source`.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "ablation.csv"
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "spider_cortex_sim",
                    "--episodes",
                    "0",
                    "--eval-episodes",
                    "0",
                    "--ablation-variant",
                    "monolithic_policy",
                    "--budget-profile",
                    "smoke",
                    "--full-summary",
                    "--behavior-csv",
                    str(csv_path),
                ],
                cwd=Path(__file__).resolve().parents[1],
                env={**os.environ, "PYTHONPATH": "."},
                text=True,
                capture_output=True,
                check=True,
            )

            payload = json.loads(proc.stdout)
            ablations = payload["behavior_evaluation"]["ablations"]
            self.assertEqual(ablations["reference_variant"], "modular_full")
            self.assertEqual(ablations["budget_profile"], "smoke")
            self.assertEqual(ablations["benchmark_strength"], "quick")
            self.assertEqual(ablations["seeds"], [7])
            self.assertEqual(ablations["scenario_names"], list(SCENARIO_NAMES))
            self.assertIn("monolithic_policy", ablations["variants"])
            self.assertEqual(payload["behavior_evaluation"]["suite"], {})
            self.assertTrue(csv_path.exists())
            header = csv_path.read_text(encoding="utf-8").splitlines()[0]
            self.assertIn("ablation_variant", header)
            self.assertIn("ablation_architecture", header)
            self.assertIn("budget_profile", header)
            self.assertIn("benchmark_strength", header)
            self.assertIn("architecture_version", header)
            self.assertIn("architecture_fingerprint", header)
            self.assertIn("operational_profile", header)
            self.assertIn("operational_profile_version", header)
            self.assertIn("noise_profile", header)
            self.assertIn("checkpoint_source", header)

    def test_cli_learning_evidence_emits_summary_and_csv_columns(self) -> None:
        """
        Verify the CLI produces a learning-evidence JSON summary and a CSV file containing expected learning-evidence columns.
        
        Runs the command-line tool with learning-evidence and related flags, then asserts the produced JSON contains a learning_evidence section with:
        - reference_condition equal to "trained_without_reflex_support"
        - budget_profile and long_budget_profile set to "smoke"
        - a "trained_final" condition present
        
        Also asserts a CSV file is written and its header contains the columns:
        "learning_evidence_condition", "learning_evidence_policy_mode", "learning_evidence_train_episodes", "learning_evidence_frozen_after_episode", and "learning_evidence_checkpoint_source".
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "learning_evidence.csv"
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "spider_cortex_sim",
                    "--episodes",
                    "0",
                    "--eval-episodes",
                    "0",
                    "--learning-evidence",
                    "--behavior-scenario",
                    "night_rest",
                    "--budget-profile",
                    "smoke",
                    "--learning-evidence-long-budget-profile",
                    "smoke",
                    "--full-summary",
                    "--behavior-csv",
                    str(csv_path),
                ],
                cwd=Path(__file__).resolve().parents[1],
                env={**os.environ, "PYTHONPATH": "."},
                text=True,
                capture_output=True,
                check=True,
            )

            payload = json.loads(proc.stdout)
            learning_evidence = payload["behavior_evaluation"]["learning_evidence"]
            self.assertEqual(
                learning_evidence["reference_condition"],
                "trained_without_reflex_support",
            )
            self.assertEqual(learning_evidence["budget_profile"], "smoke")
            self.assertEqual(learning_evidence["long_budget_profile"], "smoke")
            self.assertIn("trained_final", learning_evidence["conditions"])
            self.assertTrue(csv_path.exists())
            header = csv_path.read_text(encoding="utf-8").splitlines()[0]
            self.assertIn("learning_evidence_condition", header)
            self.assertIn("learning_evidence_policy_mode", header)
            self.assertIn("learning_evidence_train_episodes", header)
            self.assertIn("learning_evidence_frozen_after_episode", header)
            self.assertIn("learning_evidence_checkpoint_source", header)

    def test_cli_claim_test_suite_emits_summary_and_csv_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "claim_tests.csv"
            proc = subprocess.run(
                self._claim_suite_base_argv()
                + ["--full-summary", "--behavior-csv", str(csv_path)],
                cwd=Path(__file__).resolve().parents[1],
                env={**os.environ, "PYTHONPATH": "."},
                text=True,
                capture_output=True,
                check=True,
            )

            payload = json.loads(proc.stdout)
            claim_tests = payload["behavior_evaluation"]["claim_tests"]
            self.assertIn("claims", claim_tests)
            self.assertIn("summary", claim_tests)
            self.assertIn("metadata", claim_tests)
            self.assertIn(
                "learning_without_privileged_signals",
                claim_tests["claims"],
            )
            self.assertIn("claims_passed", claim_tests["summary"])
            self.assertTrue(csv_path.exists())
            header = csv_path.read_text(encoding="utf-8").splitlines()[0]
            self.assertIn("claim_test", header)
            self.assertIn("claim_test_status", header)
            self.assertIn("claim_test_passed", header)
            self.assertIn("claim_test_primary_metric", header)

    def test_cli_claim_test_suite_short_summary_is_condensed(self) -> None:
        """
        Verify that running the CLI claim-test-suite without requesting a full summary returns a condensed summary structure.
        
        The test invokes the CLI for the specified claim test and asserts the JSON output includes top-level claim-test summary fields:
        `claims`, `claims_passed`, `claims_failed`, and `all_primary_claims_passed`. It also asserts the per-claim entry for
        `learning_without_privileged_signals` is condensed to only include the `passed` key.
        """
        proc = subprocess.run(
            self._claim_suite_base_argv(),
            cwd=Path(__file__).resolve().parents[1],
            env={**os.environ, "PYTHONPATH": "."},
            text=True,
            capture_output=True,
            check=True,
        )

        payload = json.loads(proc.stdout)
        claim_tests = payload["behavior_evaluation"]["claim_tests"]
        self.assertIn("claims", claim_tests)
        self.assertIn("claims_passed", claim_tests)
        self.assertIn("claims_failed", claim_tests)
        self.assertIn("claims_skipped", claim_tests)
        self.assertIn("all_primary_claims_passed", claim_tests)
        self.assertEqual(
            set(claim_tests["claims"]["learning_without_privileged_signals"].keys()),
            {"passed"},
        )

    def test_cli_claim_test_without_suite_flag_runs_claim_suite(self) -> None:
        proc = subprocess.run(
            self._claim_suite_base_argv(include_suite_flag=False),
            cwd=Path(__file__).resolve().parents[1],
            env={**os.environ, "PYTHONPATH": "."},
            text=True,
            capture_output=True,
            check=True,
        )

        payload = json.loads(proc.stdout)
        self.assertIn("claim_tests", payload["behavior_evaluation"])
        self.assertIn(
            "learning_without_privileged_signals",
            payload["behavior_evaluation"]["claim_tests"]["claims"],
        )

    def test_cli_curriculum_emits_summary_comparison_and_csv_columns(self) -> None:
        """
        Verifies the CLI emits a curriculum training summary, includes a curriculum comparison in the behavior evaluation, and writes a CSV with curriculum-related columns.
        
        Asserts that the produced JSON payload sets the training regime mode to "curriculum", contains a "curriculum" section, and that the behavior evaluation includes a "curriculum_comparison" with `curriculum_profile` equal to "ecological_v1". Also asserts a CSV file is created and its header contains the columns: "training_regime", "curriculum_profile", "curriculum_phase", "curriculum_skill", "curriculum_phase_status", and "curriculum_promotion_reason".
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "curriculum.csv"
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "spider_cortex_sim",
                    "--episodes",
                    "2",
                    "--eval-episodes",
                    "1",
                    "--behavior-scenario",
                    "night_rest",
                    "--budget-profile",
                    "smoke",
                    "--curriculum-profile",
                    "ecological_v1",
                    "--full-summary",
                    "--behavior-csv",
                    str(csv_path),
                ],
                cwd=Path(__file__).resolve().parents[1],
                env={**os.environ, "PYTHONPATH": "."},
                text=True,
                capture_output=True,
                check=True,
            )

            payload = json.loads(proc.stdout)
            self.assertEqual(payload["config"]["training_regime"]["mode"], "curriculum")
            self.assertIn("curriculum", payload)
            self.assertIn("curriculum_comparison", payload["behavior_evaluation"])
            self.assertEqual(
                payload["behavior_evaluation"]["curriculum_comparison"]["curriculum_profile"],
                "ecological_v1",
            )
            self.assertTrue(csv_path.exists())
            header = csv_path.read_text(encoding="utf-8").splitlines()[0]
            self.assertIn("training_regime", header)
            self.assertIn("curriculum_profile", header)
            self.assertIn("curriculum_phase", header)
            self.assertIn("curriculum_skill", header)
            self.assertIn("curriculum_phase_status", header)
            self.assertIn("curriculum_promotion_reason", header)

    def test_cli_noise_robustness_emits_summary_and_csv_columns(self) -> None:
        """
        Verify the CLI --noise-robustness option emits a JSON robustness matrix summary and a CSV with the expected noise-profile columns.
        
        Asserts that the JSON payload contains a `behavior_evaluation.robustness_matrix` with a `matrix_spec.cell_count` of 16 and includes `matrix`, `train_marginals`, `eval_marginals`, and `robustness_score`. Also asserts the emitted CSV file exists and its header contains `train_noise_profile`, `train_noise_profile_config`, `eval_noise_profile`, and `eval_noise_profile_config`.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            for flag in ("--noise-robustness", "--behavior-noise-robustness"):
                with self.subTest(flag=flag):
                    csv_path = Path(tmpdir) / f"{flag.lstrip('-').replace('-', '_')}.csv"
                    proc = subprocess.run(  # noqa: S603
                        [
                            sys.executable,
                            "-m",
                            "spider_cortex_sim",
                            "--episodes",
                            "0",
                            "--eval-episodes",
                            "0",
                            flag,
                            "--behavior-scenario",
                            "night_rest",
                            "--budget-profile",
                            "smoke",
                            "--full-summary",
                            "--behavior-csv",
                            str(csv_path),
                        ],
                        cwd=Path(__file__).resolve().parents[1],
                        env={**os.environ, "PYTHONPATH": "."},
                        text=True,
                        capture_output=True,
                        check=True,
                    )

                    payload = json.loads(proc.stdout)
                    self.assertEqual(
                        payload["config"]["training_regime"],
                        {"name": "noise_robustness"},
                    )
                    self.assertIn("summary", payload["behavior_evaluation"])
                    self.assertIn("suite", payload["behavior_evaluation"])
                    self.assertIn("legacy_scenarios", payload["behavior_evaluation"])
                    robustness = payload["behavior_evaluation"]["robustness_matrix"]
                    self.assertEqual(robustness["matrix_spec"]["cell_count"], 16)
                    self.assertIn("matrix", robustness)
                    self.assertIn("train_marginals", robustness)
                    self.assertIn("eval_marginals", robustness)
                    self.assertIn("robustness_score", robustness)
                    self.assertTrue(csv_path.exists())
                    header = csv_path.read_text(encoding="utf-8").splitlines()[0]
                    self.assertIn("train_noise_profile", header)
                    self.assertIn("train_noise_profile_config", header)
                    self.assertIn("eval_noise_profile", header)
                    self.assertIn("eval_noise_profile_config", header)

    def test_cli_noise_robustness_short_summary_reports_score_and_dimensions(self) -> None:
        for flag in ("--noise-robustness", "--behavior-noise-robustness"):
            with self.subTest(flag=flag):
                proc = subprocess.run(  # noqa: S603
                    [
                        sys.executable,
                        "-m",
                        "spider_cortex_sim",
                        "--episodes",
                        "0",
                        "--eval-episodes",
                        "0",
                        flag,
                        "--behavior-scenario",
                        "night_rest",
                        "--budget-profile",
                        "smoke",
                    ],
                    cwd=Path(__file__).resolve().parents[1],
                    env={**os.environ, "PYTHONPATH": "."},
                    text=True,
                    capture_output=True,
                    check=True,
                )

                payload = json.loads(proc.stdout)
                robustness = payload["behavior_evaluation"]["robustness_matrix"]
                self.assertIn("robustness_score", robustness)
                self.assertEqual(robustness["matrix_dimensions"], "4x4")
                self.assertEqual(payload["training_regime"], {"name": "noise_robustness"})

    def test_cli_noise_robustness_rejects_trace_output(self) -> None:
        for flag in ("--noise-robustness", "--behavior-noise-robustness"):
            with self.subTest(flag=flag):
                with tempfile.TemporaryDirectory() as tmpdir:
                    trace_path = Path(tmpdir) / "trace.json"
                    proc = subprocess.run(  # noqa: S603
                        [
                            sys.executable,
                            "-m",
                            "spider_cortex_sim",
                            "--episodes",
                            "0",
                            "--eval-episodes",
                            "0",
                            flag,
                            "--behavior-scenario",
                            "night_rest",
                            "--budget-profile",
                            "smoke",
                            "--trace",
                            str(trace_path),
                        ],
                        cwd=Path(__file__).resolve().parents[1],
                        env={**os.environ, "PYTHONPATH": "."},
                        text=True,
                        capture_output=True,
                    )

                    self.assertNotEqual(proc.returncode, 0)
                    self.assertIn(
                        "--trace is not supported with --noise-robustness.",
                        proc.stderr,
                    )
                    self.assertFalse(trace_path.exists())

    def test_cli_noise_robustness_rejects_render_and_debug_flags(self) -> None:
        conflict_cases = (
            (["--render-eval"], "--render-eval"),
            (["--debug-trace"], "--debug-trace"),
        )
        for robustness_flag in ("--noise-robustness", "--behavior-noise-robustness"):
            for extra_args, rejected_flag in conflict_cases:
                with self.subTest(
                    robustness_flag=robustness_flag,
                    rejected_flag=rejected_flag,
                ):
                    proc = subprocess.run(  # noqa: S603
                        [
                            sys.executable,
                            "-m",
                            "spider_cortex_sim",
                            "--episodes",
                            "0",
                            "--eval-episodes",
                            "0",
                            robustness_flag,
                            "--behavior-scenario",
                            "night_rest",
                            "--budget-profile",
                            "smoke",
                            *extra_args,
                        ],
                        cwd=Path(__file__).resolve().parents[1],
                        env={**os.environ, "PYTHONPATH": "."},
                        text=True,
                        capture_output=True,
                    )

                    self.assertNotEqual(proc.returncode, 0)
                    self.assertIn(
                        f"{rejected_flag} is not supported with --noise-robustness.",
                        proc.stderr,
                    )

    def test_cli_noise_robustness_rejects_curriculum_profile(self) -> None:
        for flag in ("--noise-robustness", "--behavior-noise-robustness"):
            with self.subTest(flag=flag):
                proc = subprocess.run(  # noqa: S603
                    [
                        sys.executable,
                        "-m",
                        "spider_cortex_sim",
                        "--episodes",
                        "0",
                        "--eval-episodes",
                        "0",
                        flag,
                        "--behavior-scenario",
                        "night_rest",
                        "--budget-profile",
                        "smoke",
                        "--curriculum-profile",
                        "ecological_v1",
                    ],
                    cwd=Path(__file__).resolve().parents[1],
                    env={**os.environ, "PYTHONPATH": "."},
                    text=True,
                    capture_output=True,
                )

                self.assertNotEqual(proc.returncode, 0)
                self.assertIn(
                    "--curriculum-profile is not supported with --noise-robustness.",
                    proc.stderr,
                )

    def test_cli_noise_robustness_rejects_noise_profile(self) -> None:
        for flag in ("--noise-robustness", "--behavior-noise-robustness"):
            with self.subTest(flag=flag):
                proc = subprocess.run(  # noqa: S603
                    [
                        sys.executable,
                        "-m",
                        "spider_cortex_sim",
                        "--episodes",
                        "0",
                        "--eval-episodes",
                        "0",
                        flag,
                        "--behavior-scenario",
                        "night_rest",
                        "--budget-profile",
                        "smoke",
                        "--noise-profile",
                        "low",
                    ],
                    cwd=Path(__file__).resolve().parents[1],
                    env={**os.environ, "PYTHONPATH": "."},
                    text=True,
                    capture_output=True,
                )

                self.assertNotEqual(proc.returncode, 0)
                self.assertIn(
                    "--noise-profile is not supported with --noise-robustness.",
                    proc.stderr,
                )

    def test_cli_noise_robustness_rejects_incompatible_workflow_flags(self) -> None:
        """
        Verifies the CLI rejects workflow flags that are incompatible with noise-robustness modes.
        
        This test runs the CLI with each of the noise-robustness entry points (`--noise-robustness`
        and `--behavior-noise-robustness`) combined with a set of mutually incompatible workflow
        flags (e.g., comparison, ablation, learning-evidence, claim-test-suite) and asserts the
        process exits with a non-zero status and emits an error message indicating the specific
        flag is not supported with noise-robustness.
        """
        conflict_cases = (
            (["--compare-profiles"], "--compare-profiles"),
            (["--compare-maps"], "--compare-maps"),
            (["--behavior-compare-profiles"], "--behavior-compare-profiles"),
            (["--behavior-compare-maps"], "--behavior-compare-maps"),
            (["--ablation-suite"], "--ablation-suite"),
            (["--ablation-variant", "monolithic_policy"], "--ablation-variant"),
            (["--learning-evidence"], "--learning-evidence"),
            (["--claim-test-suite"], "--claim-test-suite"),
            (
                ["--claim-test", "learning_without_privileged_signals"],
                "--claim-test-suite",
            ),
        )
        for robustness_flag in ("--noise-robustness", "--behavior-noise-robustness"):
            for extra_args, rejected_flag in conflict_cases:
                with self.subTest(
                    robustness_flag=robustness_flag,
                    rejected_flag=rejected_flag,
                ):
                    proc = subprocess.run(  # noqa: S603
                        [
                            sys.executable,
                            "-m",
                            "spider_cortex_sim",
                            "--episodes",
                            "0",
                            "--eval-episodes",
                            "0",
                            robustness_flag,
                            "--behavior-scenario",
                            "night_rest",
                            "--budget-profile",
                            "smoke",
                            *extra_args,
                        ],
                        cwd=Path(__file__).resolve().parents[1],
                        env={**os.environ, "PYTHONPATH": "."},
                        text=True,
                        capture_output=True,
                    )

                    self.assertNotEqual(proc.returncode, 0)
                    self.assertIn(
                        f"{rejected_flag} is not supported with --noise-robustness.",
                        proc.stderr,
                    )

    def test_offline_analysis_module_emits_report_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            behavior_csv = Path(tmpdir) / "behavior.csv"
            report_dir = Path(tmpdir) / "offline_report"
            summary_path = Path(tmpdir) / "summary.json"

            sim = SpiderSimulation(seed=11, max_steps=8)
            payload, _, rows = sim.evaluate_behavior_suite(["night_rest"])
            summary, _ = sim.train(
                episodes=0,
                evaluation_episodes=0,
                capture_evaluation_trace=False,
            )
            summary["behavior_evaluation"] = payload
            save_behavior_csv(rows, behavior_csv)
            save_summary(summary, summary_path)

            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "spider_cortex_sim.offline_analysis",
                    "--summary",
                    str(summary_path),
                    "--behavior-csv",
                    str(behavior_csv),
                    "--output-dir",
                    str(report_dir),
                ],
                cwd=Path(__file__).resolve().parents[1],
                env={**os.environ, "PYTHONPATH": "."},
                text=True,
                capture_output=True,
                check=True,
            )

            payload = json.loads(proc.stdout)
            self.assertIn("generated_files", payload)
            self.assertTrue((report_dir / "report.md").exists())
            self.assertTrue((report_dir / "report.json").exists())
            self.assertTrue((report_dir / "training_eval.svg").exists())
            self.assertTrue((report_dir / "scenario_success.svg").exists())
            self.assertTrue((report_dir / "robustness_matrix.svg").exists())
            self.assertTrue((report_dir / "scenario_checks.csv").exists())
            self.assertTrue((report_dir / "reward_components.csv").exists())
            report_md = (report_dir / "report.md").read_text(encoding="utf-8")
            self.assertIn("night_rest", report_md)

    def test_cli_does_not_synthesize_checkpointing_without_selection_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = Path(tmpdir) / "summary.json"
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "spider_cortex_sim",
                    "--budget-profile",
                    "smoke",
                    "--checkpoint-selection",
                    "best",
                    "--summary",
                    str(summary_path),
                    "--full-summary",
                ],
                cwd=Path(__file__).resolve().parents[1],
                env={**os.environ, "PYTHONPATH": "."},
                text=True,
                capture_output=True,
                check=True,
            )

            payload = json.loads(proc.stdout)
            self.assertNotIn("checkpointing", payload)
            file_payload = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertNotIn("checkpointing", file_payload)

    def test_predator_response_latency_metric_with_scripted_episode(self) -> None:
        """
        Validate predator response latency metric, trace contents, and reward consistency using a scripted deterministic episode.
        
        Runs a single scripted episode that produces exactly one predator response event and asserts the following observable outcomes:
        - `stats.predator_response_events` is 1.
        - `stats.mean_predator_response_latency` is 2.0.
        - the captured `trace` has length 3.
        - the final trace entry contains `reward_components`, `distance_deltas`, and `debug`.
        - the final trace entry's `reward` equals the sum of its `reward_components` values.
        """
        profile_summary = DEFAULT_OPERATIONAL_PROFILE.to_summary()
        profile_summary["perception"]["perceptual_delay_ticks"] = 0.0
        profile = OperationalProfile.from_summary(profile_summary)
        sim = SpiderSimulation(seed=41, max_steps=3, operational_profile=profile)
        sim.world.lizard_move_interval = 999999

        original_reset = sim.world.reset

        def scripted_reset(seed: int | None = None):
            """
            Reset the simulation world to a deterministic scripted state suitable for reproducible tests.
            
            Parameters:
            	seed (int | None): Optional random seed forwarded to the original reset to ensure reproducibility.
            
            Returns:
            	observation (dict): The world's initial observation after applying the scripted state.
            """
            original_reset(seed=seed)
            sim.world.tick = 1
            sim.world.state.x, sim.world.state.y = 0, 0
            sim.world.state.hunger = 0.1
            sim.world.state.fatigue = 0.1
            sim.world.state.sleep_debt = 0.1
            sim.world.state.health = 1.0
            sim.world.food_positions = [(sim.world.width - 1, sim.world.height - 1)]
            sim.world.lizard.x, sim.world.lizard.y = 0, 0
            sim.world.lizard.profile = None
            sim.world.refresh_memory(initial=True)
            return sim.world.observe()

        scripted_actions = iter(["STAY", "MOVE_RIGHT", "MOVE_RIGHT"])

        def scripted_act(
            _observation,
            _bus,
            sample: bool = True,
            policy_mode: str = "normal",
        ):
            """
            Produce a deterministic BrainStep for scripted testing using the next scripted action.
            
            Parameters:
                _observation: Ignored; present to match the actor API.
                _bus: Ignored; present to match the actor API.
                sample (bool): Ignored; included for API compatibility.
                policy_mode (str): Ignored; included for API compatibility.
            
            Returns:
                BrainStep: Deterministic step whose action indexes (action_idx, action_intent_idx, motor_action_idx)
                are taken from the next value of `scripted_actions`; whose `policy` and `action_center_policy` are a
                uniform distribution over `LOCOMOTION_ACTIONS`; whose logits arrays (`action_center_logits`,
                `motor_correction_logits`, `total_logits`) are all zeros; `value` is 0.0; `motor_override` is False;
                and whose `action_center_input` and `motor_input` are minimal zero-length float arrays.
            """
            del sample, policy_mode
            action_idx = ACTION_TO_INDEX[next(scripted_actions)]
            zeros = np.zeros(len(LOCOMOTION_ACTIONS), dtype=float)
            policy = np.full(
                len(LOCOMOTION_ACTIONS),
                1.0 / len(LOCOMOTION_ACTIONS),
                dtype=float,
            )
            return BrainStep(
                module_results=[],
                action_center_logits=zeros.copy(),
                action_center_policy=policy.copy(),
                motor_correction_logits=zeros.copy(),
                total_logits=zeros.copy(),
                policy=policy,
                value=0.0,
                action_intent_idx=action_idx,
                motor_action_idx=action_idx,
                action_idx=action_idx,
                motor_override=False,
                action_center_input=np.zeros(1, dtype=float),
                motor_input=np.zeros(1, dtype=float),
            )

        sim.world.reset = scripted_reset
        sim.brain.act = scripted_act

        stats, trace = sim.run_episode(
            episode_index=0,
            training=False,
            sample=False,
            capture_trace=True,
            debug_trace=True,
        )

        self.assertEqual(stats.predator_response_events, 1)
        self.assertEqual(stats.mean_predator_response_latency, 2.0)
        self.assertEqual(len(trace), 3)
        self.assertIn("reward_components", trace[-1])
        self.assertAlmostEqual(trace[-1]["reward"], sum(trace[-1]["reward_components"].values()))
        self.assertIn("distance_deltas", trace[-1])
        self.assertIn("event_log", trace[-1])
        self.assertIsInstance(trace[-1]["event_log"], list)
        self.assertTrue(trace[-1]["event_log"])
        self.assertIsInstance(trace[-1]["event_log"][0], dict)
        self.assertIn("stage", trace[-1]["event_log"][0])
        self.assertIn("name", trace[-1]["event_log"][0])
        self.assertIn("payload", trace[-1]["event_log"][0])
        self.assertIn("debug", trace[-1])

    def test_brain_step_and_debug_trace_expose_reflex_metadata(self) -> None:
        """
        Ensure brain action steps and debug traces include reflex metadata and related debug fields.
        
        Verifies that a reflex produced by brain.act is present on the `action.proposal` bus message from `alert_center` with matching `action` and `reason`, and that the proposal payload includes gating/valence/intent fields (`valence_role`, `gate_weight`, `gated_logits`, `intent_before_gating`, `intent_after_gating`) which are JSON-serializable. Runs a scripted episode with `capture_trace=True` and `debug_trace=True` and verifies the trace records the reflex at `trace[0]["debug"]["reflexes"]["alert_center"]["reflex"]` with the expected `action` and `reason`, that top-level debug includes `action_center` and `motor_cortex` (including `selected_intent` and `correction_logits`), that arbitration information and `suppressed_modules` are present, and that debug/reflex payloads are JSON-serializable.
        """
        sim = SpiderSimulation(seed=53, max_steps=1)
        sim.world.reset(seed=53)
        sim.world.state.x, sim.world.state.y = 0, 0
        sim.world.state.heading_dx = 1
        sim.world.state.heading_dy = 0
        sim.world.lizard.x, sim.world.lizard.y = 1, 0
        observation = sim.world.observe()

        decision = sim.brain.act(observation, sim.bus, sample=False)
        alert_result = next(result for result in decision.module_results if result.name == "alert_center")
        proposal_payload = next(
            message.payload
            for message in sim.bus.topic_messages("action.proposal")
            if message.sender == "alert_center"
        )

        self.assertIsNotNone(alert_result.reflex)
        self.assertEqual(proposal_payload["reflex"]["action"], alert_result.reflex.action)
        self.assertEqual(proposal_payload["reflex"]["reason"], alert_result.reflex.reason)
        self.assertIn("valence_role", proposal_payload)
        self.assertIn("gate_weight", proposal_payload)
        self.assertIn("contribution_share", proposal_payload)
        self.assertIn("gated_logits", proposal_payload)
        json.dumps(proposal_payload["reflex"])
        json.dumps(proposal_payload["gated_logits"])
        json.dumps(proposal_payload["valence_role"])
        json.dumps(proposal_payload["gate_weight"])
        json.dumps(proposal_payload["contribution_share"])
        json.dumps(proposal_payload["intent_before_gating"])
        json.dumps(proposal_payload["intent_after_gating"])

        trace_sim = SpiderSimulation(seed=59, max_steps=1)
        original_reset = trace_sim.world.reset

        def scripted_reset(seed: int | None = None):
            original_reset(seed=seed)
            trace_sim.world.state.x, trace_sim.world.state.y = 0, 0
            trace_sim.world.state.heading_dx = 1
            trace_sim.world.state.heading_dy = 0
            trace_sim.world.lizard.x, trace_sim.world.lizard.y = 1, 0
            return trace_sim.world.observe()

        trace_sim.world.reset = scripted_reset
        _, trace = trace_sim.run_episode(
            episode_index=0,
            training=False,
            sample=False,
            capture_trace=True,
            debug_trace=True,
        )

        self.assertEqual(len(trace), 1)
        debug_alert = trace[0]["debug"]["reflexes"]["alert_center"]["reflex"]
        self.assertIsNotNone(debug_alert)
        self.assertEqual(debug_alert["action"], "MOVE_LEFT")
        self.assertEqual(debug_alert["reason"], "retreat_from_visible_predator")
        self.assertIn("action_center", trace[0]["debug"])
        self.assertIn("motor_cortex", trace[0]["debug"])
        self.assertIn("selected_intent", trace[0]["debug"]["action_center"])
        self.assertIn("winning_valence", trace[0]["debug"]["action_center"])
        self.assertIn("module_gates", trace[0]["debug"]["action_center"])
        self.assertIn("module_contribution_share", trace[0]["debug"]["action_center"])
        self.assertIn("dominant_module", trace[0]["debug"]["action_center"])
        self.assertIn("effective_module_count", trace[0]["debug"]["action_center"])
        self.assertIn("arbitration", trace[0]["debug"])
        self.assertIn("suppressed_modules", trace[0]["debug"]["arbitration"])
        self.assertIn("module_agreement_rate", trace[0]["debug"]["arbitration"])
        self.assertIn("contribution_share", trace[0]["debug"]["reflexes"]["alert_center"])
        self.assertIn("correction_logits", trace[0]["debug"]["motor_cortex"])
        motor_debug = trace[0]["debug"]["motor_cortex"]
        self.assertIn("arbitrated_intent", motor_debug)
        self.assertIn("motor_selected_action", motor_debug)
        self.assertIn("intended_action", motor_debug)
        self.assertIn("executed_action", motor_debug)
        self.assertIn("orientation_alignment", motor_debug)
        self.assertIn("terrain_difficulty", motor_debug)
        self.assertIn("execution_difficulty", motor_debug)
        self.assertIn("slip_reason", motor_debug)
        self.assertIsInstance(motor_debug["execution_slip_occurred"], bool)
        self.assertIn("slip_reason", trace[0])
        json.dumps(debug_alert)
        json.dumps(trace[0]["debug"]["action_center"])
        json.dumps(trace[0]["debug"]["arbitration"])
        json.dumps(motor_debug)

    def test_behavior_suite_rows_include_independence_metrics(self) -> None:
        sim = SpiderSimulation(seed=61, max_steps=8)
        payload, _, rows = sim.evaluate_behavior_suite(
            names=["night_rest"],
            episodes_per_scenario=1,
            capture_trace=False,
            debug_trace=False,
        )

        self.assertIn("suite", payload)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertIn("metric_module_contribution_alert_center", row)
        self.assertIn("metric_dominant_module", row)
        self.assertIn("metric_dominant_module_share", row)
        self.assertIn("metric_effective_module_count", row)
        self.assertIn("metric_module_agreement_rate", row)
        self.assertIn("metric_module_disagreement_rate", row)
        self.assertIn("metric_motor_slip_rate", row)
        self.assertIn("metric_mean_orientation_alignment", row)
        self.assertIn("metric_mean_terrain_difficulty", row)

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
                checkpoint_metric="not_a_real_metric",
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





class CheckpointSourceAnnotationTest(unittest.TestCase):
    """Tests that checkpoint_source is properly annotated in behavior rows."""

    def test_evaluate_behavior_suite_rows_have_checkpoint_source(self) -> None:
        """
        Verify evaluate_behavior_suite produces non-empty rows annotated with checkpoint_source and noise_profile defaults.
        
        Asserts that the returned rows list is non-empty, that the first row contains a "checkpoint_source" field equal to "final", and that the "noise_profile" field equals "none".
        """
        sim = SpiderSimulation(seed=7, max_steps=10)
        _, _, rows = sim.evaluate_behavior_suite(["night_rest"])
        self.assertTrue(rows)
        self.assertIn("checkpoint_source", rows[0])
        self.assertEqual(rows[0]["checkpoint_source"], "final")
        self.assertEqual(rows[0]["noise_profile"], "none")

    def test_checkpoint_source_final_in_rows_after_regular_train(self) -> None:
        sim = SpiderSimulation(seed=7, max_steps=10)
        sim.train(episodes=2, evaluation_episodes=0, capture_evaluation_trace=False)
        _, _, rows = sim.evaluate_behavior_suite(["night_rest"])
        self.assertTrue(rows)
        self.assertEqual(rows[0]["checkpoint_source"], "final")

    def test_budget_profile_annotation_in_behavior_rows(self) -> None:
        sim = SpiderSimulation(
            seed=7,
            max_steps=10,
            budget_profile_name="smoke",
            benchmark_strength="quick",
        )
        _, _, rows = sim.evaluate_behavior_suite(["night_rest"])
        self.assertTrue(rows)
        self.assertEqual(rows[0]["budget_profile"], "smoke")
        self.assertEqual(rows[0]["benchmark_strength"], "quick")

    def test_annotate_behavior_rows_includes_budget_fields(self) -> None:
        sim = SpiderSimulation(seed=7, max_steps=10, budget_profile_name="dev", benchmark_strength="quick")
        raw_rows = [{"some_key": "some_val"}]
        annotated = sim._annotate_behavior_rows(raw_rows)
        self.assertTrue(annotated)
        self.assertIn("budget_profile", annotated[0])
        self.assertIn("benchmark_strength", annotated[0])
        self.assertIn("checkpoint_source", annotated[0])
        self.assertIn("ablation_variant", annotated[0])
        self.assertIn("ablation_architecture", annotated[0])
        self.assertIn("operational_profile", annotated[0])
        self.assertIn("noise_profile", annotated[0])
        self.assertIn("eval_noise_profile", annotated[0])

    def test_annotate_behavior_rows_merges_extra_metadata(self) -> None:
        sim = SpiderSimulation(seed=7, max_steps=10)
        raw_rows = [{"existing_key": "val"}]
        extra = {
            "learning_evidence_condition": "trained_final",
            "learning_evidence_policy_mode": "normal",
            "learning_evidence_train_episodes": 6,
        }
        annotated = sim._annotate_behavior_rows(raw_rows, extra_metadata=extra)
        self.assertEqual(len(annotated), 1)
        self.assertEqual(annotated[0]["learning_evidence_condition"], "trained_final")
        self.assertEqual(annotated[0]["learning_evidence_policy_mode"], "normal")
        self.assertEqual(annotated[0]["learning_evidence_train_episodes"], 6)
        # Original keys preserved
        self.assertEqual(annotated[0]["existing_key"], "val")

    def test_annotate_behavior_rows_extra_metadata_none_does_not_add_keys(self) -> None:
        sim = SpiderSimulation(seed=7, max_steps=10)
        raw_rows = [{"only_key": "v"}]
        annotated = sim._annotate_behavior_rows(raw_rows, extra_metadata=None)
        self.assertNotIn("learning_evidence_condition", annotated[0])

    def test_annotate_behavior_rows_extra_metadata_does_not_mutate_original(self) -> None:
        sim = SpiderSimulation(seed=7, max_steps=10)
        original = {"k": "v"}
        annotated = sim._annotate_behavior_rows([original], extra_metadata={"extra": 1})
        self.assertNotIn("extra", original)
        self.assertIn("extra", annotated[0])

    def test_swap_eval_noise_profile_restores_original_world_noise(self) -> None:
        sim = SpiderSimulation(seed=7, max_steps=10, noise_profile="low")
        self.assertEqual(sim.world.noise_profile.name, "low")

        with sim._swap_eval_noise_profile("high"):
            self.assertEqual(sim.world.noise_profile.name, "high")

        self.assertEqual(sim.world.noise_profile.name, "low")

        with self.assertRaises(RuntimeError):
            with sim._swap_eval_noise_profile("high"):
                self.assertEqual(sim.world.noise_profile.name, "high")
                raise RuntimeError("exercise context-manager cleanup")

        self.assertEqual(sim.world.noise_profile.name, "low")

    def test_annotate_behavior_rows_includes_train_and_eval_noise_fields(self) -> None:
        sim = SpiderSimulation(seed=7, max_steps=10, noise_profile="low")

        with sim._swap_eval_noise_profile("high"):
            annotated = sim._annotate_behavior_rows(
                [{"scenario": "night_rest"}],
                train_noise_profile="low",
            )

        self.assertEqual(annotated[0]["train_noise_profile"], "low")
        self.assertEqual(annotated[0]["eval_noise_profile"], "high")
        self.assertEqual(annotated[0]["noise_profile"], "high")
        self.assertIn("train_noise_profile_config", annotated[0])
        self.assertIn("eval_noise_profile_config", annotated[0])

    def test_consume_episodes_without_learning_returns_stats_list(self) -> None:
        sim = SpiderSimulation(seed=5, max_steps=8)
        history = sim._consume_episodes_without_learning(episodes=2, episode_start=0)
        self.assertEqual(len(history), 2)

    def test_consume_episodes_without_learning_zero_episodes_returns_empty(self) -> None:
        sim = SpiderSimulation(seed=5, max_steps=8)
        history = sim._consume_episodes_without_learning(episodes=0)
        self.assertEqual(history, [])

    def test_consume_episodes_without_learning_negative_episodes_returns_empty(self) -> None:
        sim = SpiderSimulation(seed=5, max_steps=8)
        history = sim._consume_episodes_without_learning(episodes=-5)
        self.assertEqual(history, [])

    def test_consume_episodes_without_learning_does_not_update_weights(self) -> None:
        sim = SpiderSimulation(seed=5, max_steps=8)
        weights_before = sim.brain.motor_cortex.W1.copy()
        sim._consume_episodes_without_learning(episodes=2, episode_start=0)
        np.testing.assert_array_equal(weights_before, sim.brain.motor_cortex.W1)

    def test_consume_episodes_without_learning_accepts_policy_mode_reflex_only(self) -> None:
        sim = SpiderSimulation(seed=5, max_steps=8)
        # reflex_only is only valid for modular architecture; default brain is modular
        history = sim._consume_episodes_without_learning(
            episodes=1, episode_start=0, policy_mode="reflex_only"
        )
        self.assertEqual(len(history), 1)


if __name__ == "__main__":
    unittest.main()
