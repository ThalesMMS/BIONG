"""Comprehensive tests for the action_center addition introduced in this PR.

Covers:
- architecture_signature() new structure (action_center, motor_cortex keys)
- SpiderBrain.ARCHITECTURE_VERSION = 12
- SpiderBrain.action_center / motor_cortex network types
- BrainStep new fields (action_center_logits, action_center_policy,
  action_intent_idx, motor_override, action_center_input)
- SpiderBrain.estimate_value uses action_center, not motor_cortex alone
- _build_motor_input NaN / inf sanitization
- ActionContextObservation and MotorContextObservation field composition
- ACTION_CONTEXT_INTERFACE / MOTOR_CONTEXT_INTERFACE properties
- OBSERVATION_DIMS updated entries
- build_action_context_observation / build_motor_context_observation
"""

import json
import tempfile
import unittest
from pathlib import Path
from typing import ClassVar
from unittest.mock import patch

import numpy as np

from spider_cortex_sim.ablations import (
    BrainAblationConfig,
    MONOLITHIC_POLICY_NAME,
    TRUE_MONOLITHIC_POLICY_NAME,
    canonical_ablation_configs,
    default_brain_config,
    resolve_ablation_configs,
)
from spider_cortex_sim.agent import BrainStep, SpiderBrain
from spider_cortex_sim.arbitration import ArbitrationDecision
from spider_cortex_sim.bus import MessageBus
from spider_cortex_sim.interfaces import (
    ACTION_CONTEXT_INTERFACE,
    ACTION_DELTAS,
    ACTION_TO_INDEX,
    LOCOMOTION_ACTIONS,
    MODULE_INTERFACES,
    MOTOR_CONTEXT_INTERFACE,
    MODULE_INTERFACE_BY_NAME,
    OBSERVATION_DIMS,
    ActionContextObservation,
    MotorContextObservation,
    architecture_signature,
)
from spider_cortex_sim.direct_policy_options import OPTION_TO_INDEX
from spider_cortex_sim.maps import CLUTTER, NARROW, OPEN
from spider_cortex_sim.modules import CorticalModuleBank, ModuleResult
from spider_cortex_sim.nn import (
    ArbitrationNetwork,
    MotorNetwork,
    OwnedOptionControllerTrueMonolithicNetwork,
    ProposalNetwork,
)
from spider_cortex_sim.noise import compute_execution_difficulty
from spider_cortex_sim.operational_profiles import DEFAULT_OPERATIONAL_PROFILE, OperationalProfile
from spider_cortex_sim.phase import PHASE_TO_INDEX
from spider_cortex_sim.perception import (
    PerceivedTarget,
    build_action_context_observation,
    build_motor_context_observation,
)
from spider_cortex_sim.world import SpiderWorld


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

from tests.fixtures.action_center import (
    _null_percept,
    _blank_obs,
    _module_interface,
    _vector_for,
    _valence_vector,
    _profile_with_updates,
)
from tests.fixtures.action_center_monolithic import MonolithicArchitectureFixtures


class SpiderBrainMonolithicBasicsTest(MonolithicArchitectureFixtures, unittest.TestCase):
    def _owned_option_config(self) -> BrainAblationConfig:
        return resolve_ablation_configs(
            ["true_monolithic_owned_option_controller_policy"],
            module_dropout=0.0,
        )[0]

    def _owned_option_network(
        self,
        *,
        seed: int = 1,
    ) -> OwnedOptionControllerTrueMonolithicNetwork:
        return OwnedOptionControllerTrueMonolithicNetwork(
            input_dim=sum(spec.input_dim for spec in MODULE_INTERFACES),
            hidden_dim=16,
            output_dim=len(LOCOMOTION_ACTIONS),
            rng=np.random.default_rng(seed),
            event_buffer_size=4,
            option_ttl=4,
        )

    def _monolithic_vector(self, values: dict[tuple[str, str], float]) -> np.ndarray:
        x = np.zeros(sum(spec.input_dim for spec in MODULE_INTERFACES), dtype=float)
        offset = 0
        for spec in MODULE_INTERFACES:
            for signal_name, value in (
                (signal_name, value)
                for (module_name, signal_name), value in values.items()
                if module_name == spec.name
            ):
                x[offset + spec.signal_names.index(signal_name)] = float(value)
            offset += spec.input_dim
        return x

    def _force_owned_option(
        self,
        network: OwnedOptionControllerTrueMonolithicNetwork,
        option_name: str,
    ) -> None:
        network.W2_option.fill(0.0)
        network.b2_option.fill(-8.0)
        network.b2_option[OPTION_TO_INDEX[option_name]] = 8.0
        network.W2_option_leaf.fill(0.0)
        network.b2_option_leaf.fill(0.0)

    def test_monolithic_brain_has_no_module_bank(self) -> None:
        brain = SpiderBrain(seed=1, config=self._monolithic_config())
        self.assertIsNone(brain.module_bank)

    def test_monolithic_brain_has_monolithic_policy(self) -> None:
        brain = SpiderBrain(seed=1, config=self._monolithic_config())
        self.assertIsNotNone(brain.monolithic_policy)

    def test_modular_brain_has_module_bank(self) -> None:
        brain = SpiderBrain(seed=1)
        self.assertIsNotNone(brain.module_bank)

    def test_modular_brain_has_no_monolithic_policy(self) -> None:
        brain = SpiderBrain(seed=1)
        self.assertIsNone(brain.monolithic_policy)

    def test_monolithic_act_returns_single_module_result(self) -> None:
        brain = SpiderBrain(seed=5, config=self._monolithic_config())
        obs = self._build_observation()
        step = brain.act(obs, sample=False)
        self.assertEqual(len(step.module_results), 1)
        self.assertEqual(step.module_results[0].name, SpiderBrain.MONOLITHIC_POLICY_NAME)

    def test_true_monolithic_brain_initializes_direct_policy_only(self) -> None:
        brain = SpiderBrain(seed=5, config=self._true_monolithic_config())
        self.assertIsNone(brain.module_bank)
        self.assertIsNone(brain.monolithic_policy)
        self.assertIsNotNone(brain.true_monolithic_policy)

    def test_true_monolithic_act_returns_single_direct_module_result(self) -> None:
        brain = SpiderBrain(seed=5, config=self._true_monolithic_config())
        obs = self._build_observation()
        step = brain.act(obs, sample=False)
        self.assertEqual(len(step.module_results), 1)
        self.assertEqual(step.module_results[0].name, TRUE_MONOLITHIC_POLICY_NAME)
        self.assertGreaterEqual(step.action_idx, 0)
        self.assertLess(step.action_idx, len(LOCOMOTION_ACTIONS))
        self.assertTrue(np.isfinite(step.value))
        self.assertTrue(np.all(np.isfinite(step.policy)))
        self.assertEqual(step.motor_correction_logits.shape, (len(LOCOMOTION_ACTIONS),))
        self.assertTrue(np.allclose(step.motor_correction_logits, 0.0))
        self.assertIsNotNone(step.arbitration_decision)
        self.assertEqual(
            step.arbitration_decision.module_gates[TRUE_MONOLITHIC_POLICY_NAME],
            1.0,
        )

    def test_public_action_space_remains_primitive_for_owned_option_variant(self) -> None:
        self.assertEqual(
            tuple(LOCOMOTION_ACTIONS),
            (
                "MOVE_UP",
                "MOVE_DOWN",
                "MOVE_LEFT",
                "MOVE_RIGHT",
                "STAY",
                "ORIENT_UP",
                "ORIENT_DOWN",
                "ORIENT_LEFT",
                "ORIENT_RIGHT",
            ),
        )
        self.assertNotIn("MOVE_TO_FOOD", LOCOMOTION_ACTIONS)
        self.assertNotIn("MOVE_TO_SHELTER", LOCOMOTION_ACTIONS)

    def test_owned_option_variant_instantiates_dedicated_controller(self) -> None:
        config = self._owned_option_config()
        self.assertTrue(config.direct_policy_owned_option_controller)
        self.assertFalse(config.enable_food_direction_bias)
        brain = SpiderBrain(seed=5, config=config)
        self.assertIsInstance(
            brain.true_monolithic_policy,
            OwnedOptionControllerTrueMonolithicNetwork,
        )

    def test_owned_option_active_leaf_controls_final_logits(self) -> None:
        network = self._owned_option_network()
        self._force_owned_option(network, "RETURN_TO_SHELTER")
        network.b2_option_leaf[
            OPTION_TO_INDEX["FORAGE"],
            ACTION_TO_INDEX["MOVE_LEFT"],
        ] = 10.0
        network.b2_option_leaf[
            OPTION_TO_INDEX["RETURN_TO_SHELTER"],
            ACTION_TO_INDEX["MOVE_RIGHT"],
        ] = 4.0
        logits, _, _ = network.forward(
            np.zeros(network.input_dim, dtype=float),
            store_cache=True,
        )
        self.assertEqual(network.last_option_summary["selected_option"], "RETURN_TO_SHELTER")
        self.assertEqual(int(np.argmax(logits)), ACTION_TO_INDEX["MOVE_RIGHT"])
        self.assertEqual(
            int(np.argmax(network.last_option_summary["option_leaf_logits"])),
            ACTION_TO_INDEX["MOVE_RIGHT"],
        )

    def test_owned_option_inactive_leaf_does_not_contribute_to_final_logits(self) -> None:
        network = self._owned_option_network()
        self._force_owned_option(network, "REST")
        network.b2_option_leaf[
            OPTION_TO_INDEX["FORAGE"],
            ACTION_TO_INDEX["MOVE_RIGHT"],
        ] = 12.0
        network.b2_option_leaf[
            OPTION_TO_INDEX["REST"],
            ACTION_TO_INDEX["STAY"],
        ] = 3.0
        logits, _, _ = network.forward(
            np.zeros(network.input_dim, dtype=float),
            store_cache=True,
        )
        self.assertEqual(network.last_option_summary["selected_option"], "REST")
        self.assertEqual(int(np.argmax(logits)), ACTION_TO_INDEX["STAY"])

    def test_owned_option_safety_mask_only_masks_blocked_leaf_action(self) -> None:
        network = self._owned_option_network()
        self._force_owned_option(network, "FORAGE")
        network.b2_option_leaf[
            OPTION_TO_INDEX["FORAGE"],
            ACTION_TO_INDEX["MOVE_RIGHT"],
        ] = 4.0
        network.b2_option_leaf[
            OPTION_TO_INDEX["FORAGE"],
            ACTION_TO_INDEX["MOVE_LEFT"],
        ] = 3.0
        network.set_runtime_observation_meta(
            {
                "local_affordances": {
                    "MOVE_RIGHT": {"blocked": True},
                    "MOVE_LEFT": {"blocked": False},
                }
            }
        )
        logits, _, _ = network.forward(
            np.zeros(network.input_dim, dtype=float),
            store_cache=True,
        )
        summary = network.last_option_summary
        self.assertTrue(summary["safety_mask_applied"])
        self.assertEqual(summary["safety_masked_actions"], ["MOVE_RIGHT"])
        self.assertEqual(
            int(np.argmax(summary["option_leaf_logits"])),
            ACTION_TO_INDEX["MOVE_RIGHT"],
        )
        self.assertEqual(int(np.argmax(logits)), ACTION_TO_INDEX["MOVE_LEFT"])

    def test_owned_forage_leaf_emits_primitive_move_toward_visible_food(self) -> None:
        network = self._owned_option_network()
        self._force_owned_option(network, "FORAGE")
        logits, _, _ = network.forward(
            self._monolithic_vector(
                {
                    ("hunger_center", "food_visible"): 1.0,
                    ("hunger_center", "food_certainty"): 1.0,
                    ("hunger_center", "food_dx"): 1.0,
                    ("hunger_center", "food_dy"): 0.0,
                }
            ),
            store_cache=True,
        )
        self.assertEqual(int(np.argmax(logits)), ACTION_TO_INDEX["MOVE_RIGHT"])

    def test_owned_return_leaf_emits_primitive_move_toward_shelter_memory(self) -> None:
        network = self._owned_option_network()
        self._force_owned_option(network, "RETURN_TO_SHELTER")
        logits, _, _ = network.forward(
            self._monolithic_vector(
                {
                    ("sleep_center", "shelter_memory_dx"): -1.0,
                    ("sleep_center", "shelter_memory_dy"): 0.0,
                    ("sleep_center", "shelter_memory_age"): 0.0,
                }
            ),
            store_cache=True,
        )
        self.assertEqual(int(np.argmax(logits)), ACTION_TO_INDEX["MOVE_LEFT"])

    def test_owned_rest_leaf_favors_stay_in_shelter(self) -> None:
        network = self._owned_option_network()
        self._force_owned_option(network, "REST")
        logits, _, _ = network.forward(
            self._monolithic_vector(
                {
                    ("sleep_center", "on_shelter"): 1.0,
                    ("sleep_center", "fatigue"): 0.8,
                    ("sleep_center", "sleep_debt"): 0.6,
                }
            ),
            store_cache=True,
        )
        self.assertEqual(int(np.argmax(logits)), ACTION_TO_INDEX["STAY"])

    def test_owned_post_rest_leaf_favors_exit_without_external_bias(self) -> None:
        network = self._owned_option_network()
        self._force_owned_option(network, "POST_REST_REACTIVATE")
        network.set_runtime_observation_meta(
            {
                "local_geodesic_consequences": {
                    "MOVE_UP": {
                        "exit_geodesic_delta": 1.0,
                        "next_on_exit_target": True,
                    },
                    "MOVE_DOWN": {"exit_geodesic_delta": -1.0},
                    "MOVE_LEFT": {"exit_geodesic_delta": 0.0},
                    "MOVE_RIGHT": {"exit_geodesic_delta": 0.0},
                    "STAY": {"exit_geodesic_delta": 0.0},
                }
            }
        )
        logits, _, _ = network.forward(
            self._monolithic_vector({("sleep_center", "on_shelter"): 1.0}),
            store_cache=True,
        )
        self.assertEqual(int(np.argmax(logits)), ACTION_TO_INDEX["MOVE_UP"])

    def test_owned_option_checkpoint_round_trips_flag_and_leaf_weights(self) -> None:
        config = self._owned_option_config()
        brain = SpiderBrain(seed=5, config=config)
        assert isinstance(
            brain.true_monolithic_policy,
            OwnedOptionControllerTrueMonolithicNetwork,
        )
        brain.true_monolithic_policy.b2_option_leaf[
            OPTION_TO_INDEX["REST"],
            ACTION_TO_INDEX["STAY"],
        ] = 7.25
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "brain"
            brain.save(checkpoint_dir)
            metadata = json.loads((checkpoint_dir / "metadata.json").read_text())
            self.assertTrue(
                metadata["ablation_config"]["direct_policy_owned_option_controller"]
            )
            self.assertTrue(
                metadata["modules"][TRUE_MONOLITHIC_POLICY_NAME][
                    "owned_option_controller"
                ]
            )
            restored = SpiderBrain(seed=9, config=config)
            restored.load(checkpoint_dir)
            assert isinstance(
                restored.true_monolithic_policy,
                OwnedOptionControllerTrueMonolithicNetwork,
            )
            self.assertAlmostEqual(
                restored.true_monolithic_policy.b2_option_leaf[
                    OPTION_TO_INDEX["REST"],
                    ACTION_TO_INDEX["STAY"],
                ],
                7.25,
            )

    def test_true_monolithic_estimate_value_returns_finite_float(self) -> None:
        brain = SpiderBrain(seed=7, config=self._true_monolithic_config())
        obs = self._build_observation()
        value = brain.estimate_value(obs)
        self.assertIsInstance(value, float)
        self.assertTrue(np.isfinite(value))

    def test_true_monolithic_learn_updates_weights_and_returns_valid_stats(self) -> None:
        brain = SpiderBrain(seed=17, config=self._true_monolithic_config())
        obs = self._build_observation()
        step = brain.act(obs, sample=True)
        next_obs = self._build_observation()
        before = {
            key: value.copy()
            for key, value in brain.true_monolithic_policy.state_dict().items()
            if isinstance(value, np.ndarray)
        }
        stats = brain.learn(step, reward=1.5, next_observation=next_obs, done=True)
        self.assertIn("reward", stats)
        self.assertEqual(stats["credit_strategy"], "broadcast")
        self.assertEqual(
            stats["module_credit_weights"],
            {TRUE_MONOLITHIC_POLICY_NAME: 1.0},
        )
        self.assertIn(TRUE_MONOLITHIC_POLICY_NAME, stats["module_gradient_norms"])
        self.assertTrue(np.isfinite(stats["td_error"]))
        after = {
            key: value.copy()
            for key, value in brain.true_monolithic_policy.state_dict().items()
            if isinstance(value, np.ndarray)
        }
        self.assertTrue(
            any(not np.allclose(before[key], after[key]) for key in before)
        )

    def test_true_monolithic_handoff_teacher_losses_use_full_weight_when_active(self) -> None:
        config = resolve_ablation_configs(
            ["true_monolithic_option_affordance_position_teacher_option_policy"],
            module_dropout=0.0,
        )[0]
        brain = SpiderBrain(seed=23, config=config)
        obs = self._build_observation()
        step = brain.act_train(obs, sample=False)
        step.affordance_blocked_targets = np.zeros_like(
            step.affordance_blocked_logits,
        )
        step.affordance_role_targets = np.zeros(len(LOCOMOTION_ACTIONS), dtype=int)
        step.geometry_targets = np.zeros_like(step.geometry_logits)
        step.shelter_position_targets = np.zeros(
            len(LOCOMOTION_ACTIONS),
            dtype=int,
        )
        step.teacher_action_target_idx = int(ACTION_TO_INDEX["MOVE_RIGHT"])
        step.teacher_option_target_idx = 0
        stats = brain.learn(
            step,
            reward=0.25,
            next_observation=self._build_observation(),
            done=False,
        )
        self.assertEqual(stats["handoff_teacher_weight"], 1.0)
        self.assertEqual(stats["handoff_option_teacher_weight"], 1.0)
        self.assertTrue(stats["handoff_teacher_active"])
        self.assertTrue(stats["handoff_option_teacher_active"])
        self.assertGreater(stats["handoff_teacher_loss"], 0.0)
        self.assertGreater(stats["handoff_option_teacher_loss"], 0.0)

    def test_true_monolithic_continuation_scenarios_boost_auxiliary_weights(self) -> None:
        config = resolve_ablation_configs(
            ["true_monolithic_option_affordance_position_teacher_option_policy"],
            module_dropout=0.0,
        )[0]
        brain = SpiderBrain(seed=29, config=config)
        obs = self._build_observation()
        step = brain.act_train(obs, sample=False)
        step.scenario_name = "continuous_survival_post_rest_inside_v1"
        step.phase_target = "POST_REST_REACTIVATE"
        step.phase_target_idx = int(PHASE_TO_INDEX["POST_REST_REACTIVATE"])
        step.affordance_blocked_targets = np.zeros_like(
            step.affordance_blocked_logits,
        )
        step.affordance_role_targets = np.zeros(len(LOCOMOTION_ACTIONS), dtype=int)
        step.geometry_targets = np.zeros_like(step.geometry_logits)
        step.shelter_position_targets = np.zeros(
            len(LOCOMOTION_ACTIONS),
            dtype=int,
        )
        step.teacher_action_target_idx = int(ACTION_TO_INDEX["MOVE_RIGHT"])
        step.teacher_action_target_stage = "handoff_release"
        step.teacher_option_target_idx = 0
        step.teacher_option_target_stage = "option_reactivate"
        stats = brain.learn(
            step,
            reward=0.25,
            next_observation=self._build_observation(),
            done=False,
        )
        self.assertEqual(stats["handoff_teacher_weight"], 3.0)
        self.assertEqual(stats["handoff_option_teacher_weight"], 3.0)
        self.assertEqual(stats["phase_weight"], 0.6)
        self.assertEqual(stats["affordance_blocked_weight"], 0.2)
        self.assertEqual(stats["affordance_role_weight"], 0.2)
        self.assertEqual(stats["shelter_position_weight"], 0.2)
        self.assertEqual(
            stats["continuation_weight_profile"]["teacher_action"],
            3.0,
        )

    def test_true_monolithic_post_rest_action_teacher_stages_boost_action_weight(self) -> None:
        config = resolve_ablation_configs(
            [
                "true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_backbone_post_rest_action_teacher_option_replay_policy"
            ],
            module_dropout=0.0,
        )[0]
        brain = SpiderBrain(seed=129, config=config)
        obs = self._build_observation()
        step = brain.act_train(obs, sample=False)
        step.scenario_name = "continuous_survival_post_rest_entrance_v1"
        step.teacher_action_target_idx = int(ACTION_TO_INDEX["MOVE_RIGHT"])
        step.teacher_action_target_stage = "handoff_post_rest_forage"
        step.teacher_option_target_idx = 0
        step.teacher_option_target_stage = "option_post_rest_forage"
        stats = brain.learn(
            step,
            reward=0.25,
            next_observation=self._build_observation(),
            done=False,
        )
        self.assertEqual(stats["handoff_teacher_weight"], 3.0)
        self.assertEqual(
            stats["continuation_weight_profile"]["teacher_action"],
            3.0,
        )

    def test_true_monolithic_post_rest_release_sequence_replay_boost_raises_weights_and_passes(self) -> None:
        config = resolve_ablation_configs(
            [
                "true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_backbone_post_rest_release_sequence_replay_boost_option_replay_policy"
            ],
            module_dropout=0.0,
        )[0]
        brain = SpiderBrain(seed=131, config=config)
        obs = self._build_observation()
        step = brain.act_train(obs, sample=False)
        step.observation = dict(obs)
        step.scenario_name = "continuous_survival_post_rest_inside_v1"
        step.phase_target = "POST_REST_REACTIVATE"
        step.phase_target_idx = int(PHASE_TO_INDEX["POST_REST_REACTIVATE"])
        step.affordance_blocked_targets = np.zeros_like(
            step.affordance_blocked_logits,
        )
        step.affordance_role_targets = np.zeros(len(LOCOMOTION_ACTIONS), dtype=int)
        step.geometry_targets = np.zeros_like(step.geometry_logits)
        step.shelter_position_targets = np.zeros(
            len(LOCOMOTION_ACTIONS),
            dtype=int,
        )
        step.teacher_action_target_idx = int(ACTION_TO_INDEX["MOVE_RIGHT"])
        step.teacher_action_target_stage = "handoff_release"
        step.teacher_option_target_idx = 0
        step.teacher_option_target_stage = "option_reactivate"
        stats = brain.learn(
            step,
            reward=0.25,
            next_observation=self._build_observation(),
            done=False,
        )
        self.assertTrue(stats["post_rest_sequence_replay_boost_active"])
        self.assertEqual(stats["handoff_teacher_weight"], 6.0)
        self.assertEqual(stats["handoff_option_teacher_weight"], 4.0)
        self.assertEqual(stats["phase_weight"], 0.8)
        self.assertEqual(stats["continuation_replay_passes_configured"], 6)
        self.assertEqual(stats["continuation_replay_passes_applied"], 6)
        self.assertEqual(stats["continuation_replay_lr_scale"], 1.0)

    def test_true_monolithic_post_rest_release_sequence_distill_adds_sequence_loss(self) -> None:
        config = resolve_ablation_configs(
            [
                "true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_backbone_post_rest_release_sequence_distill_option_replay_policy"
            ],
            module_dropout=0.0,
        )[0]
        brain = SpiderBrain(seed=132, config=config)
        obs = self._build_observation()
        step = brain.act_train(obs, sample=False)
        step.observation = dict(obs)
        step.scenario_name = "continuous_survival_post_rest_inside_v1"
        step.phase_target = "POST_REST_REACTIVATE"
        step.phase_target_idx = int(PHASE_TO_INDEX["POST_REST_REACTIVATE"])
        step.affordance_blocked_targets = np.zeros_like(
            step.affordance_blocked_logits,
        )
        step.affordance_role_targets = np.zeros(len(LOCOMOTION_ACTIONS), dtype=int)
        step.geometry_targets = np.zeros_like(step.geometry_logits)
        step.shelter_position_targets = np.zeros(
            len(LOCOMOTION_ACTIONS),
            dtype=int,
        )
        step.teacher_action_target_idx = int(ACTION_TO_INDEX["MOVE_RIGHT"])
        step.teacher_action_target_stage = "handoff_release"
        step.teacher_option_target_idx = 0
        step.teacher_option_target_stage = "option_reactivate"
        stats = brain.learn(
            step,
            reward=0.25,
            next_observation=self._build_observation(),
            done=False,
        )
        self.assertTrue(stats["post_rest_sequence_replay_boost_active"])
        self.assertTrue(stats["post_rest_sequence_distill_active"])
        self.assertGreater(stats["post_rest_sequence_distill_loss"], 0.0)
        self.assertGreater(stats["post_rest_sequence_distill_action_loss"], 0.0)
        self.assertGreater(stats["post_rest_sequence_distill_option_loss"], 0.0)
        self.assertGreater(stats["post_rest_sequence_distill_phase_loss"], 0.0)
        self.assertEqual(stats["continuation_replay_passes_configured"], 6)
        self.assertEqual(stats["continuation_replay_passes_applied"], 6)

    def test_true_monolithic_continuation_replay_passes_apply_when_enabled(self) -> None:
        config = resolve_ablation_configs(
            ["true_monolithic_option_affordance_position_phase_teacher_option_replay_policy"],
            module_dropout=0.0,
        )[0]
        brain = SpiderBrain(seed=31, config=config)
        obs = self._build_observation()
        step = brain.act_train(obs, sample=False)
        step.observation = dict(obs)
        step.scenario_name = "continuous_survival_post_rest_inside_v1"
        step.phase_target = "POST_REST_REACTIVATE"
        step.phase_target_idx = int(PHASE_TO_INDEX["POST_REST_REACTIVATE"])
        step.affordance_blocked_targets = np.zeros_like(
            step.affordance_blocked_logits,
        )
        step.affordance_role_targets = np.zeros(len(LOCOMOTION_ACTIONS), dtype=int)
        step.geometry_targets = np.zeros_like(step.geometry_logits)
        step.shelter_position_targets = np.zeros(
            len(LOCOMOTION_ACTIONS),
            dtype=int,
        )
        step.teacher_action_target_idx = int(ACTION_TO_INDEX["MOVE_RIGHT"])
        step.teacher_action_target_stage = "handoff_release"
        step.teacher_option_target_idx = 0
        step.teacher_option_target_stage = "option_reactivate"
        stats = brain.learn(
            step,
            reward=0.25,
            next_observation=self._build_observation(),
            done=False,
        )
        self.assertEqual(stats["continuation_replay_passes_configured"], 2)
        self.assertEqual(stats["continuation_replay_passes_applied"], 2)
        self.assertEqual(stats["continuation_replay_lr_scale"], 0.5)
        self.assertGreater(stats["continuation_replay_total_loss"], 0.0)
        self.assertGreater(stats["continuation_replay_total_grad_norm"], 0.0)
        self.assertGreater(stats["phase_loss"], 0.0)

    def test_true_monolithic_continuation_margin_loss_applies_when_enabled(self) -> None:
        config = resolve_ablation_configs(
            [
                "true_monolithic_option_affordance_position_phase_teacher_option_margin_replay_policy"
            ],
            module_dropout=0.0,
        )[0]
        brain = SpiderBrain(seed=32, config=config)
        obs = self._build_observation()
        step = brain.act_train(obs, sample=False)
        step.observation = dict(obs)
        step.scenario_name = "continuous_survival_post_rest_inside_v1"
        step.phase_target = "POST_REST_REACTIVATE"
        step.phase_target_idx = int(PHASE_TO_INDEX["POST_REST_REACTIVATE"])
        step.affordance_blocked_targets = np.zeros_like(step.affordance_blocked_logits)
        step.affordance_role_targets = np.zeros(len(LOCOMOTION_ACTIONS), dtype=int)
        step.geometry_targets = np.zeros_like(step.geometry_logits)
        step.shelter_position_targets = np.zeros(len(LOCOMOTION_ACTIONS), dtype=int)
        step.teacher_action_target_idx = int(ACTION_TO_INDEX["MOVE_RIGHT"])
        step.teacher_action_target_stage = "handoff_release"
        step.teacher_option_target_idx = 0
        step.teacher_option_target_name = "FORAGE"
        step.teacher_option_target_stage = "option_post_rest_forage"
        stats = brain.learn(
            step,
            reward=0.25,
            next_observation=self._build_observation(),
            done=False,
        )
        self.assertEqual(stats["continuation_margin_weight"], 1.0)
        self.assertGreater(stats["continuation_margin_loss"], 0.0)
        self.assertGreater(stats["continuation_margin_grad_norm"], 0.0)

    def test_true_monolithic_geodesic_frontier_continuation_margin_loss_applies_when_enabled(
        self,
    ) -> None:
        config = resolve_ablation_configs(
            [
                "true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_backbone_geodesic_inputs_post_rest_probe_replayable_teacher_distill_option_margin_replay_policy"
            ],
            module_dropout=0.0,
        )[0]
        brain = SpiderBrain(seed=33, config=config)
        obs = self._build_observation()
        step = brain.act_train(obs, sample=False)
        step.observation = dict(obs)
        step.scenario_name = "continuous_survival_post_rest_inside_v1"
        step.phase_target = "POST_REST_REACTIVATE"
        step.phase_target_idx = int(PHASE_TO_INDEX["POST_REST_REACTIVATE"])
        step.affordance_blocked_targets = np.zeros_like(step.affordance_blocked_logits)
        step.affordance_role_targets = np.zeros(len(LOCOMOTION_ACTIONS), dtype=int)
        step.geometry_targets = np.zeros_like(step.geometry_logits)
        step.shelter_position_targets = np.zeros(len(LOCOMOTION_ACTIONS), dtype=int)
        step.teacher_action_target_idx = int(ACTION_TO_INDEX["MOVE_RIGHT"])
        step.teacher_action_target_stage = "handoff_release"
        step.teacher_option_target_idx = 0
        step.teacher_option_target_name = "FORAGE"
        step.teacher_option_target_stage = "option_post_rest_forage"
        stats = brain.learn(
            step,
            reward=0.25,
            next_observation=self._build_observation(),
            done=False,
        )
        self.assertEqual(stats["continuation_margin_weight"], 1.0)
        self.assertGreater(stats["continuation_margin_loss"], 0.0)
        self.assertGreater(stats["continuation_margin_grad_norm"], 0.0)

    def test_true_monolithic_frozen_proposer_skips_weight_update_and_reports_zero_norm(self) -> None:
        brain = SpiderBrain(seed=17, config=self._true_monolithic_config())
        obs = self._build_observation()
        step = brain.act(obs, sample=True)
        next_obs = self._build_observation()
        before = self._numeric_state(brain.true_monolithic_policy)

        brain.freeze_proposers([TRUE_MONOLITHIC_POLICY_NAME])
        stats = brain.learn(step, reward=1.5, next_observation=next_obs, done=True)

        self.assertEqual(brain.frozen_module_names(), [TRUE_MONOLITHIC_POLICY_NAME])
        self.assertFalse(
            self._state_changed(before, self._numeric_state(brain.true_monolithic_policy))
        )
        self.assertEqual(
            stats["module_gradient_norms"][TRUE_MONOLITHIC_POLICY_NAME],
            0.0,
        )

    def test_true_monolithic_counterfactual_credit_is_coerced_before_credit_logic(self) -> None:
        brain = SpiderBrain(
            seed=17,
            config=self._true_monolithic_config(credit_strategy="counterfactual"),
        )
        obs = self._build_observation()
        step = brain.act(obs, sample=True)

        def fail_counterfactual(*args, **kwargs):
            raise AssertionError(
                "_compute_counterfactual_credit should not run for true_monolithic"
            )

        brain._compute_counterfactual_credit = fail_counterfactual
        stats = brain.learn(step, reward=1.0, next_observation=obs, done=False)

        self.assertEqual(stats["credit_strategy"], "broadcast")
        self.assertEqual(
            stats["module_credit_weights"],
            {TRUE_MONOLITHIC_POLICY_NAME: 1.0},
        )
        self.assertEqual(stats["counterfactual_credit_weights"], {})

    def test_true_monolithic_save_and_load_round_trip(self) -> None:
        config = self._true_monolithic_config()
        brain = SpiderBrain(seed=11, config=config)
        obs = self._build_observation()
        step_before = brain.act(obs, sample=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "true_mono_brain"
            brain.save(save_path)
            self.assertTrue((save_path / "true_monolithic_policy.npz").exists())

            brain2 = SpiderBrain(seed=999, config=config)
            brain2.load(save_path)
            step_after = brain2.act(obs, sample=False)

        np.testing.assert_allclose(step_before.policy, step_after.policy, atol=1e-5)
        np.testing.assert_allclose(step_before.value, step_after.value, atol=1e-5)

    def test_monolithic_act_result_has_none_interface(self) -> None:
        brain = SpiderBrain(seed=5, config=self._monolithic_config())
        obs = self._build_observation()
        step = brain.act(obs, sample=False)
        self.assertIsNone(step.module_results[0].interface)

    def test_monolithic_act_result_is_active(self) -> None:
        brain = SpiderBrain(seed=5, config=self._monolithic_config())
        obs = self._build_observation()
        step = brain.act(obs, sample=False)
        self.assertTrue(step.module_results[0].active)

    def test_monolithic_module_names_includes_monolithic_policy(self) -> None:
        brain = SpiderBrain(seed=5, config=self._monolithic_config())
        names = brain._module_names()
        self.assertIn(SpiderBrain.MONOLITHIC_POLICY_NAME, names)
        self.assertIn("arbitration_network", names)
        self.assertIn("action_center", names)
        self.assertIn("motor_cortex", names)

    def test_modular_module_names_includes_each_module(self) -> None:
        brain = SpiderBrain(seed=5)
        names = brain._module_names()
        self.assertIn("visual_cortex", names)
        self.assertIn("alert_center", names)
        self.assertIn("action_center", names)
        self.assertIn("motor_cortex", names)
        self.assertNotIn(SpiderBrain.MONOLITHIC_POLICY_NAME, names)

    def test_monolithic_parameter_norms_has_monolithic_policy(self) -> None:
        brain = SpiderBrain(seed=3, config=self._monolithic_config())
        norms = brain.parameter_norms()
        self.assertIn(SpiderBrain.MONOLITHIC_POLICY_NAME, norms)
        self.assertIn("arbitration_network", norms)
        self.assertIn("action_center", norms)
        self.assertIn("motor_cortex", norms)

    def test_monolithic_parameter_counts_has_monolithic_policy(self) -> None:
        brain = SpiderBrain(seed=3, config=self._monolithic_config())
        counts = brain.count_parameters()
        self.assertIn(SpiderBrain.MONOLITHIC_POLICY_NAME, counts)
        self.assertIn("arbitration_network", counts)
        self.assertIn("action_center", counts)
        self.assertIn("motor_cortex", counts)
        self.assertGreater(counts[SpiderBrain.MONOLITHIC_POLICY_NAME], 0)

    def test_modular_parameter_norms_has_module_names(self) -> None:
        brain = SpiderBrain(seed=3)
        norms = brain.parameter_norms()
        self.assertIn("visual_cortex", norms)
        self.assertIn("alert_center", norms)
        self.assertIn("action_center", norms)
        self.assertIn("motor_cortex", norms)
        self.assertNotIn(SpiderBrain.MONOLITHIC_POLICY_NAME, norms)

    def test_modular_parameter_counts_has_module_names(self) -> None:
        brain = SpiderBrain(seed=3)
        counts = brain.count_parameters()
        self.assertIn("visual_cortex", counts)
        self.assertIn("alert_center", counts)
        self.assertIn("action_center", counts)
        self.assertIn("motor_cortex", counts)
        self.assertNotIn(SpiderBrain.MONOLITHIC_POLICY_NAME, counts)

    def test_modular_and_monolithic_parameter_counts_distinguish_architectures(self) -> None:
        modular_brain = SpiderBrain(seed=3)
        monolithic_brain = SpiderBrain(seed=3, config=self._monolithic_config())

        modular_counts = modular_brain.count_parameters()
        monolithic_counts = monolithic_brain.count_parameters()

        self.assertIn("visual_cortex", modular_counts)
        self.assertIn(SpiderBrain.MONOLITHIC_POLICY_NAME, monolithic_counts)
        self.assertGreater(sum(modular_counts.values()), 0)
        self.assertGreater(sum(monolithic_counts.values()), 0)
        shared_keys = set(modular_counts) & set(monolithic_counts)
        self.assertTrue(
            set(modular_counts) != set(monolithic_counts)
            or any(modular_counts[key] != monolithic_counts[key] for key in shared_keys)
        )

    def test_monolithic_estimate_value_returns_finite_float(self) -> None:
        brain = SpiderBrain(seed=7, config=self._monolithic_config())
        obs = self._build_observation()
        value = brain.estimate_value(obs)
        self.assertIsInstance(value, float)
        self.assertTrue(np.isfinite(value))

    def test_monolithic_action_idx_is_valid(self) -> None:
        brain = SpiderBrain(seed=7, config=self._monolithic_config())
        obs = self._build_observation()
        step = brain.act(obs, sample=False)
        self.assertGreaterEqual(step.action_idx, 0)
        self.assertLess(step.action_idx, len(LOCOMOTION_ACTIONS))

    def test_monolithic_step_exposes_arbitration_metadata(self) -> None:
        brain = SpiderBrain(seed=7, config=self._monolithic_config())
        obs = self._build_observation()
        step = brain.act(obs, sample=False)
        self.assertIsNotNone(step.arbitration_decision)
        self.assertEqual(step.arbitration_decision.module_gates["monolithic_policy"], 1.0)
        self.assertEqual(
            step.arbitration_decision.module_contribution_share["monolithic_policy"],
            1.0,
        )

    def test_monolithic_save_and_load_round_trip(self) -> None:
        """
        Verifies that saving then loading a monolithic SpiderBrain reproduces its policy and value outputs.

        Uses a fixed observation and monolithic configuration: saves the brain to disk, loads it into a new SpiderBrain instance with the same configuration, and asserts that the resulting policy and value vectors match within a small numerical tolerance.
        """
        config = self._monolithic_config()
        brain = SpiderBrain(seed=11, config=config)
        obs = self._build_observation()
        step_before = brain.act(obs, sample=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "mono_brain"
            brain.save(save_path)

            brain2 = SpiderBrain(seed=999, config=config)
            brain2.load(save_path)
            step_after = brain2.act(obs, sample=False)

        np.testing.assert_allclose(step_before.policy, step_after.policy, atol=1e-5)
        np.testing.assert_allclose(step_before.value, step_after.value, atol=1e-5)

    def test_loading_mismatched_architecture_version_raises_value_error(self) -> None:
        config = self._monolithic_config()
        brain = SpiderBrain(seed=11, config=config)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "mono_brain"
            brain.save(save_path)

            metadata_path = save_path / SpiderBrain._METADATA_FILE
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["architecture_version"] = SpiderBrain.ARCHITECTURE_VERSION - 1
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

            brain2 = SpiderBrain(seed=999, config=config)
            with self.assertRaises(ValueError):
                brain2.load(save_path)

    def test_save_metadata_contains_registry_and_fingerprint(self) -> None:
        config = self._monolithic_config()
        brain = SpiderBrain(seed=11, config=config)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "mono_brain"
            brain.save(save_path)

            metadata_path = save_path / SpiderBrain._METADATA_FILE
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertIn("interface_registry", metadata)
            self.assertIn("architecture_fingerprint", metadata)
            self.assertEqual(
                metadata["architecture_fingerprint"],
                metadata["architecture"]["fingerprint"],
            )

    def test_save_writes_arbitration_network_weights(self) -> None:
        config = self._monolithic_config()
        brain = SpiderBrain(seed=11, config=config)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "mono_brain"
            brain.save(save_path)

            self.assertTrue((save_path / "arbitration_network.npz").exists())
            with np.load(save_path / "arbitration_network.npz") as arbitration_npz:
                self.assertIn("input_dim", arbitration_npz.files)
                self.assertIn("W2_gate", arbitration_npz.files)
            metadata = json.loads(
                (save_path / SpiderBrain._METADATA_FILE).read_text(encoding="utf-8")
            )
            self.assertIn("arbitration_network", metadata["modules"])
            self.assertEqual(
                metadata["modules"]["arbitration_network"]["type"],
                "arbitration",
            )

    def test_loading_missing_arbitration_network_weights_raises_file_not_found(self) -> None:
        config = self._monolithic_config()
        brain = SpiderBrain(seed=11, config=config)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "mono_brain"
            brain.save(save_path)
            (save_path / "arbitration_network.npz").unlink()

            brain2 = SpiderBrain(seed=999, config=config)
            with self.assertRaisesRegex(FileNotFoundError, "arbitration weights"):
                brain2.load(save_path)

    def test_loading_mismatched_interface_registry_raises_value_error(self) -> None:
        config = self._monolithic_config()
        brain = SpiderBrain(seed=11, config=config)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "mono_brain"
            brain.save(save_path)

            metadata_path = save_path / SpiderBrain._METADATA_FILE
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["interface_registry"]["interfaces"]["motor_cortex_context"]["version"] = 999
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

            brain2 = SpiderBrain(seed=999, config=config)
            with self.assertRaisesRegex(ValueError, "interface registry incompatible"):
                brain2.load(save_path)

    def test_loading_monolithic_into_modular_raises_value_error(self) -> None:
        mono_config = self._monolithic_config()
        brain_mono = SpiderBrain(seed=13, config=mono_config)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "mono_brain"
            brain_mono.save(save_path)

            brain_modular = SpiderBrain(seed=13)
            with self.assertRaises(ValueError):
                brain_modular.load(save_path)

    def test_loading_different_modular_ablation_config_raises_value_error(self) -> None:
        saved_brain = SpiderBrain(seed=19, config=default_brain_config(module_dropout=0.0))
        different_config = BrainAblationConfig(
            name="no_module_reflexes",
            architecture="modular",
            module_dropout=0.0,
            enable_reflexes=False,
            enable_auxiliary_targets=False,
            disabled_modules=(),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "modular_brain"
            saved_brain.save(save_path)

            mismatched_brain = SpiderBrain(seed=19, config=different_config)
            with self.assertRaises(ValueError):
                mismatched_brain.load(save_path)

    def test_loading_different_operational_profile_raises_value_error(self) -> None:
        saved_brain = SpiderBrain(seed=23)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "profiled_brain"
            saved_brain.save(save_path)

            mismatched_brain = SpiderBrain(
                seed=23,
                operational_profile=_profile_with_updates(predator_threat_smell_threshold=0.01),
            )
            with self.assertRaises(ValueError):
                mismatched_brain.load(save_path)

    def test_monolithic_config_stored_on_brain(self) -> None:
        config = self._monolithic_config()
        brain = SpiderBrain(seed=7, config=config)
        self.assertEqual(brain.config.architecture, "monolithic")
        self.assertEqual(brain.config.name, "monolithic_policy")

    def test_monolithic_learn_does_not_raise(self) -> None:
        brain = SpiderBrain(seed=17, config=self._monolithic_config())
        obs = self._build_observation()
        step = brain.act(obs, sample=True)
        next_obs = self._build_observation()
        stats = brain.learn(step, reward=0.5, next_observation=next_obs, done=False)
        self.assertIn("reward", stats)
        self.assertTrue(np.isfinite(stats["td_error"]))
