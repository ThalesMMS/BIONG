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

class SpiderTrainingCoreTest(SpiderTrainingTestBase):
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
