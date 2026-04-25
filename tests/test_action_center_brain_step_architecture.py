"""Comprehensive tests for the action_center addition introduced in this PR.

Covers:
- architecture_signature() new structure (action_center, motor_cortex keys)
- SpiderBrain.ARCHITECTURE_VERSION = 13
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
    canonical_ablation_configs,
    default_brain_config,
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
from spider_cortex_sim.maps import CLUTTER, NARROW, OPEN
from spider_cortex_sim.modules import CorticalModuleBank, ModuleResult
from spider_cortex_sim.nn import ArbitrationNetwork, MotorNetwork, ProposalNetwork
from spider_cortex_sim.noise import compute_execution_difficulty
from spider_cortex_sim.operational_profiles import DEFAULT_OPERATIONAL_PROFILE, OperationalProfile
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

class SpiderBrainArchitectureTest(unittest.TestCase):
    def setUp(self) -> None:
        """
        Initialize the test fixture by creating a deterministic SpiderBrain instance.

        Creates a SpiderBrain with seed=42 and module_dropout=0.0 so tests run with reproducible weights and no module dropout.
        """
        self.brain = SpiderBrain(seed=42, module_dropout=0.0)

    def test_architecture_version(self) -> None:
        self.assertEqual(SpiderBrain.ARCHITECTURE_VERSION, 14)

    def test_legacy_act_fallback_preserves_requested_sample_mode(self) -> None:
        calls: list[dict[str, object]] = []

        def legacy_act(observation, bus=None, *, sample=False, policy_mode="normal"):
            calls.append({"sample": sample, "policy_mode": policy_mode})
            return "legacy-step"

        self.brain.act = legacy_act
        obs = _blank_obs()

        self.assertEqual(self.brain.act_exploration(obs), "legacy-step")
        self.assertTrue(calls[-1]["sample"])
        self.assertEqual(calls[-1]["policy_mode"], "normal")

        self.assertEqual(self.brain.act_inference(obs, sample=True), "legacy-step")
        self.assertTrue(calls[-1]["sample"])

        self.assertEqual(self.brain.act_train(obs, sample=False), "legacy-step")
        self.assertFalse(calls[-1]["sample"])

    def test_action_center_is_motor_network(self) -> None:
        """action_center must be a MotorNetwork (has value head)."""
        self.assertIsInstance(self.brain.action_center, MotorNetwork)

    def test_arbitration_network_is_arbitration_network(self) -> None:
        self.assertIsInstance(self.brain.arbitration_network, ArbitrationNetwork)

    def test_arbitration_regularization_defaults(self) -> None:
        self.assertEqual(self.brain.arbitration_regularization_weight, 0.1)
        self.assertEqual(self.brain.arbitration_valence_regularization_weight, 0.1)

    def test_no_regularization_arbitration_variant_disables_regularizers(self) -> None:
        brain = SpiderBrain(
            seed=42,
            module_dropout=0.0,
            config=BrainAblationConfig(name="learned_arbitration_no_regularization"),
        )
        self.assertEqual(brain.arbitration_regularization_weight, 0.0)
        self.assertEqual(brain.arbitration_valence_regularization_weight, 0.0)

    def test_motor_cortex_is_proposal_network(self) -> None:
        """motor_cortex must be a ProposalNetwork (no value head)."""
        self.assertIsInstance(self.brain.motor_cortex, ProposalNetwork)

    def test_action_center_has_value_head_attributes(self) -> None:
        """MotorNetwork has W2_policy and W2_value; ProposalNetwork does not."""
        self.assertTrue(hasattr(self.brain.action_center, "W2_policy"))
        self.assertTrue(hasattr(self.brain.action_center, "W2_value"))

    def test_motor_cortex_lacks_value_head(self) -> None:
        self.assertFalse(hasattr(self.brain.motor_cortex, "W2_policy"))
        self.assertFalse(hasattr(self.brain.motor_cortex, "W2_value"))

    def test_motor_cortex_has_proposal_attributes(self) -> None:
        self.assertTrue(hasattr(self.brain.motor_cortex, "W2"))
        self.assertTrue(hasattr(self.brain.motor_cortex, "b2"))

    def test_module_names_includes_action_center_before_motor_cortex(self) -> None:
        names = self.brain._module_names()
        self.assertIn("arbitration_network", names)
        self.assertIn("action_center", names)
        self.assertIn("motor_cortex", names)
        arb_idx = names.index("arbitration_network")
        ac_idx = names.index("action_center")
        mc_idx = names.index("motor_cortex")
        self.assertLess(arb_idx, ac_idx)
        self.assertLess(ac_idx, mc_idx)

    def test_parameter_norms_includes_action_center(self) -> None:
        norms = self.brain.parameter_norms()
        self.assertIn("arbitration_network", norms)
        self.assertIn("action_center", norms)

    def test_parameter_norms_action_center_norm_is_positive(self) -> None:
        norms = self.brain.parameter_norms()
        self.assertGreater(norms["arbitration_network"], 0.0)
        self.assertGreater(norms["action_center"], 0.0)

    def test_parameter_norms_motor_cortex_still_present(self) -> None:
        norms = self.brain.parameter_norms()
        self.assertIn("motor_cortex", norms)

    def test_freeze_proposers_defaults_to_active_proposer_modules(self) -> None:
        self.brain.freeze_proposers()
        self.assertEqual(
            self.brain.frozen_module_names(),
            sorted(spec.name for spec in self.brain.module_bank.enabled_specs),
        )
        self.assertFalse(self.brain.is_module_frozen("action_center"))
        self.assertFalse(self.brain.is_module_frozen("motor_cortex"))

    def test_unfreeze_proposers_can_remove_subset(self) -> None:
        self.brain.freeze_proposers(("alert_center", "sleep_center"))
        self.brain.unfreeze_proposers(("sleep_center",))
        self.assertEqual(self.brain.frozen_module_names(), ["alert_center"])

    def test_freeze_proposers_rejects_non_proposer_names(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown proposer modules"):
            self.brain.freeze_proposers(("action_center",))

class BrainStepNewFieldsTest(unittest.TestCase):
    def setUp(self) -> None:
        """
        Prepare the test fixture by creating a SpiderBrain and a blank observation.

        Sets:
            self.brain: a SpiderBrain instance initialized with seed 99 and module_dropout 0.0.
            self.obs: a blank observation mapping produced by _blank_obs().
        """
        self.brain = SpiderBrain(seed=99, module_dropout=0.0)
        self.obs = _blank_obs()

    def _obs_for_valence(self, valence: str) -> dict[str, np.ndarray]:
        """
        Construct a test observation dictionary whose module and context vectors are tuned to a specified behavioral valence.

        This helper produces a mapping from observation keys (e.g., "action_context", "visual", "sensory", "hunger", "sleep", "alert") to numpy arrays representing interface vectors. Each returned vector encodes signal values chosen to exercise the corresponding valence so tests can validate arbitration, proposal, and downstream behaviour.

        Parameters:
            valence (str): Target valence to simulate; expected values include "threat", "hunger", "sleep", and "exploration". The chosen valence affects which signals are elevated in the returned observation vectors.

        Returns:
            dict[str, np.ndarray]: A mapping from observation names to their serialized numpy vector representations.
        """
        obs = _blank_obs()
        obs["action_context"] = _vector_for(
            ACTION_CONTEXT_INTERFACE,
            day=1.0 if valence == "exploration" else 0.0,
            night=1.0 if valence == "sleep" else 0.0,
            hunger=0.95 if valence == "hunger" else 0.05,
            fatigue=0.95 if valence == "sleep" else 0.08,
            recent_pain=0.7 if valence == "threat" else 0.0,
            recent_contact=0.8 if valence == "threat" else 0.0,
            on_shelter=1.0 if valence == "sleep" else 0.0,
            predator_visible=0.95 if valence == "threat" else 0.0,
            predator_certainty=0.92 if valence == "threat" else 0.0,
            sleep_debt=0.95 if valence == "sleep" else 0.05,
            shelter_role_level=0.9 if valence == "sleep" else 0.0,
        )
        obs["visual"] = _vector_for(
            _module_interface("visual_cortex"),
            food_visible=0.9 if valence == "hunger" else 0.0,
            food_certainty=0.85 if valence == "hunger" else 0.0,
            predator_visible=0.95 if valence == "threat" else 0.0,
            predator_certainty=0.9 if valence == "threat" else 0.0,
            day=1.0 if valence == "exploration" else 0.0,
            night=1.0 if valence == "sleep" else 0.0,
        )
        obs["sensory"] = _vector_for(
            _module_interface("sensory_cortex"),
            recent_pain=0.7 if valence == "threat" else 0.0,
            recent_contact=0.8 if valence == "threat" else 0.0,
            hunger=0.95 if valence == "hunger" else 0.05,
            fatigue=0.95 if valence == "sleep" else 0.08,
            food_smell_strength=0.7 if valence == "hunger" else 0.1,
            food_smell_dx=0.4 if valence == "exploration" else 0.0,
            predator_smell_strength=0.85 if valence == "threat" else 0.0,
            light=1.0 if valence == "exploration" else 0.0,
        )
        obs["hunger"] = _vector_for(
            _module_interface("hunger_center"),
            hunger=0.95 if valence == "hunger" else 0.05,
            food_visible=0.95 if valence == "hunger" else 0.0,
            food_certainty=0.9 if valence == "hunger" else 0.0,
            food_smell_strength=0.82 if valence == "hunger" else 0.0,
            food_memory_age=0.1 if valence == "hunger" else 1.0,
        )
        obs["sleep"] = _vector_for(
            _module_interface("sleep_center"),
            fatigue=0.95 if valence == "sleep" else 0.08,
            hunger=0.05,
            on_shelter=1.0 if valence == "sleep" else 0.0,
            night=1.0 if valence == "sleep" else 0.0,
            sleep_phase_level=0.6 if valence == "sleep" else 0.0,
            sleep_debt=0.95 if valence == "sleep" else 0.05,
            shelter_role_level=0.9 if valence == "sleep" else 0.0,
            shelter_trace_strength=0.8 if valence == "sleep" else 0.0,
            shelter_memory_age=0.1 if valence == "sleep" else 1.0,
        )
        obs["alert"] = _vector_for(
            _module_interface("alert_center"),
            predator_visible=0.95 if valence == "threat" else 0.0,
            predator_certainty=0.92 if valence == "threat" else 0.0,
            predator_smell_strength=0.9 if valence == "threat" else 0.0,
            predator_motion_salience=0.9 if valence == "threat" else 0.0,
            recent_pain=0.7 if valence == "threat" else 0.0,
            recent_contact=0.8 if valence == "threat" else 0.0,
            on_shelter=1.0 if valence == "sleep" else 0.0,
            night=1.0 if valence == "sleep" else 0.0,
        )
        return obs

    def test_act_returns_brain_step(self) -> None:
        step = self.brain.act(self.obs, bus=None, sample=False)
        self.assertIsInstance(step, BrainStep)

    def test_action_center_logits_shape(self) -> None:
        step = self.brain.act(self.obs, bus=None, sample=False)
        self.assertEqual(step.action_center_logits.shape, (len(LOCOMOTION_ACTIONS),))

    def test_action_center_policy_shape(self) -> None:
        step = self.brain.act(self.obs, bus=None, sample=False)
        self.assertEqual(step.action_center_policy.shape, (len(LOCOMOTION_ACTIONS),))

    def test_action_center_policy_sums_to_one(self) -> None:
        step = self.brain.act(self.obs, bus=None, sample=False)
        self.assertAlmostEqual(float(np.sum(step.action_center_policy)), 1.0, places=5)

    def test_action_center_policy_non_negative(self) -> None:
        step = self.brain.act(self.obs, bus=None, sample=False)
        self.assertTrue(np.all(step.action_center_policy >= 0.0))

    def test_action_intent_idx_is_valid(self) -> None:
        step = self.brain.act(self.obs, bus=None, sample=False)
        self.assertGreaterEqual(step.action_intent_idx, 0)
        self.assertLess(step.action_intent_idx, len(LOCOMOTION_ACTIONS))

    def test_action_idx_is_valid(self) -> None:
        step = self.brain.act(self.obs, bus=None, sample=False)
        self.assertGreaterEqual(step.action_idx, 0)
        self.assertLess(step.action_idx, len(LOCOMOTION_ACTIONS))

    def test_motor_override_is_bool(self) -> None:
        step = self.brain.act(self.obs, bus=None, sample=False)
        self.assertIsInstance(step.motor_override, bool)

    def test_action_center_input_is_not_empty(self) -> None:
        step = self.brain.act(self.obs, bus=None, sample=False)
        self.assertGreater(step.action_center_input.size, 0)

    def test_motor_input_is_not_empty(self) -> None:
        step = self.brain.act(self.obs, bus=None, sample=False)
        self.assertGreater(step.motor_input.size, 0)

    def test_motor_correction_logits_shape(self) -> None:
        step = self.brain.act(self.obs, bus=None, sample=False)
        self.assertEqual(step.motor_correction_logits.shape, (len(LOCOMOTION_ACTIONS),))

    def test_final_reflex_override_is_bool(self) -> None:
        step = self.brain.act(self.obs, bus=None, sample=False)
        self.assertIsInstance(step.final_reflex_override, bool)

    def test_value_is_finite_float(self) -> None:
        step = self.brain.act(self.obs, bus=None, sample=False)
        self.assertIsInstance(step.value, float)
        self.assertTrue(np.isfinite(step.value))

    def test_motor_execution_diagnostic_fields_are_present(self) -> None:
        step = self.brain.act(self.obs, bus=None, sample=False)
        self.assertIsInstance(step.orientation_alignment, float)
        self.assertIsInstance(step.terrain_difficulty, float)
        self.assertIsInstance(step.momentum, float)
        self.assertIsInstance(step.execution_difficulty, float)
        self.assertIsInstance(step.execution_slip_occurred, bool)
        self.assertIsInstance(step.motor_noise_applied, bool)
        self.assertIsInstance(step.motor_slip_occurred, bool)
        self.assertEqual(step.slip_reason, "none")

    def test_brainstep_includes_momentum_field(self) -> None:
        step = self.brain.act(self.obs, bus=None, sample=False)

        self.assertTrue(hasattr(step, "momentum"))
        self.assertIsInstance(step.momentum, float)

    def test_brain_step_momentum_comes_from_motor_context(self) -> None:
        obs = _blank_obs()
        obs["motor_context"] = _vector_for(
            MOTOR_CONTEXT_INTERFACE,
            momentum=0.73,
        )

        step = self.brain.act(obs, bus=None, sample=False)

        self.assertAlmostEqual(step.momentum, 0.73)

    def test_motor_execution_diagnostics_follow_selected_action(self) -> None:
        obs = _blank_obs()
        obs["motor_context"] = _vector_for(
            MOTOR_CONTEXT_INTERFACE,
            heading_dx=1.0,
            heading_dy=0.0,
            terrain_difficulty=0.7,
            fatigue=0.5,
        )
        step = self.brain.act(obs, bus=None, sample=False)
        expected_difficulty, expected_components = compute_execution_difficulty(
            heading=(1.0, 0.0),
            intended_direction=ACTION_DELTAS.get(LOCOMOTION_ACTIONS[step.action_idx], (0, 0)),
            terrain=NARROW,
            fatigue=0.5,
        )
        self.assertAlmostEqual(
            step.orientation_alignment,
            expected_components["orientation_alignment"],
        )
        self.assertAlmostEqual(step.terrain_difficulty, 0.7)
        self.assertAlmostEqual(step.execution_difficulty, expected_difficulty)

    def test_motor_execution_diagnostics_use_momentum(self) -> None:
        obs = _blank_obs()
        obs["motor_context"] = _vector_for(
            MOTOR_CONTEXT_INTERFACE,
            heading_dx=1.0,
            heading_dy=1.0,
            terrain_difficulty=0.7,
            fatigue=0.0,
            momentum=0.8,
        )

        diagnostics = self.brain._motor_execution_diagnostics(
            obs,
            ACTION_TO_INDEX["MOVE_RIGHT"],
        )

        self.assertAlmostEqual(diagnostics["momentum"], 0.8)
        self.assertLess(diagnostics["execution_difficulty"], 0.7)

    def test_action_center_input_has_expected_dim(self) -> None:
        """action_center_input must equal num_modules*action_dim + action_context_dim."""
        step = self.brain.act(self.obs, bus=None, sample=False)
        num_modules = len(self.brain.module_bank.enabled_specs)
        action_dim = self.brain.action_dim
        expected_dim = num_modules * action_dim + ACTION_CONTEXT_INTERFACE.input_dim
        self.assertEqual(step.action_center_input.shape, (expected_dim,))

    def test_action_context_nan_inf_are_sanitized_before_binding(self) -> None:
        obs = _blank_obs()
        obs["action_context"] = np.array(
            [np.nan, np.inf, -np.inf] + [0.0] * (ACTION_CONTEXT_INTERFACE.input_dim - 3),
            dtype=float,
        )
        step = self.brain.act(obs, bus=None, sample=False)
        self.assertTrue(np.isfinite(step.action_center_input).all())
        self.assertTrue(all(np.isfinite(v) for v in step.arbitration_decision.valence_scores.values()))

    def test_motor_input_has_expected_dim(self) -> None:
        """motor_input must equal action_dim + motor_context_dim."""
        step = self.brain.act(self.obs, bus=None, sample=False)
        expected_dim = self.brain.action_dim + MOTOR_CONTEXT_INTERFACE.input_dim
        self.assertEqual(step.motor_input.shape, (expected_dim,))

    def test_arbitration_decision_is_present(self) -> None:
        step = self.brain.act(self.obs, bus=None, sample=False)
        self.assertIsNotNone(step.arbitration_decision)

    def test_threat_case_wins_threat_valence(self) -> None:
        step = self.brain.act(self._obs_for_valence("threat"), bus=None, sample=False)
        self.assertEqual(step.arbitration_decision.winning_valence, "threat")
        self.assertLess(step.arbitration_decision.module_gates["hunger_center"], 0.5)

    def test_hunger_case_wins_hunger_valence(self) -> None:
        step = self.brain.act(self._obs_for_valence("hunger"), bus=None, sample=False)
        self.assertEqual(step.arbitration_decision.winning_valence, "hunger")
        hunger_result = next(result for result in step.module_results if result.name == "hunger_center")
        self.assertEqual(hunger_result.valence_role, "hunger")

    def test_sleep_case_wins_sleep_valence(self) -> None:
        step = self.brain.act(self._obs_for_valence("sleep"), bus=None, sample=False)
        self.assertEqual(step.arbitration_decision.winning_valence, "sleep")
        self.assertLess(step.arbitration_decision.module_gates["visual_cortex"], 0.6)

    def test_exploration_case_wins_exploration_valence(self) -> None:
        step = self.brain.act(self._obs_for_valence("exploration"), bus=None, sample=False)
        self.assertEqual(step.arbitration_decision.winning_valence, "exploration")
        self.assertGreaterEqual(step.arbitration_decision.module_gates["visual_cortex"], 0.95)

class ExecutionDifficultyModelTest(unittest.TestCase):
    def test_open_aligned_movement_has_no_execution_difficulty(self) -> None:
        difficulty, components = compute_execution_difficulty(
            heading=(1.0, 0.0),
            intended_direction=(1.0, 0.0),
            terrain=OPEN,
            fatigue=0.0,
        )
        self.assertAlmostEqual(difficulty, 0.0)
        self.assertAlmostEqual(components["orientation_alignment"], 1.0)
        self.assertAlmostEqual(components["terrain_difficulty"], 0.0)
        self.assertAlmostEqual(components["fatigue_factor"], 0.0)

    def test_opposed_movement_uses_terrain_and_fatigue_multiplier(self) -> None:
        difficulty, components = compute_execution_difficulty(
            heading=(1.0, 0.0),
            intended_direction=(-1.0, 0.0),
            terrain=CLUTTER,
            fatigue=0.5,
        )
        self.assertAlmostEqual(components["orientation_alignment"], 0.0)
        self.assertAlmostEqual(components["terrain_difficulty"], 0.4)
        self.assertAlmostEqual(components["fatigue_factor"], 0.5)
        self.assertAlmostEqual(difficulty, 0.6)

    def test_high_difficulty_is_clipped_to_one(self) -> None:
        difficulty, components = compute_execution_difficulty(
            heading=(1.0, 0.0),
            intended_direction=(-1.0, 0.0),
            terrain=NARROW,
            fatigue=1.0,
        )
        self.assertAlmostEqual(components["raw_difficulty"], 1.4)
        self.assertAlmostEqual(difficulty, 1.0)

    def test_stay_action_has_no_orientation_mismatch(self) -> None:
        difficulty, components = compute_execution_difficulty(
            heading=(0.0, -1.0),
            intended_direction=(0.0, 0.0),
            terrain=NARROW,
            fatigue=1.0,
        )
        self.assertAlmostEqual(components["orientation_alignment"], 1.0)
        self.assertAlmostEqual(components["orientation_mismatch"], 0.0)
        self.assertAlmostEqual(difficulty, 0.0)

class BuildMotorInputSanitizationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.brain = SpiderBrain(seed=5, module_dropout=0.0)
        self.motor_obs = {
            MOTOR_CONTEXT_INTERFACE.observation_key: np.zeros(
                MOTOR_CONTEXT_INTERFACE.input_dim, dtype=float
            )
        }

    def test_nan_in_intent_becomes_zero(self) -> None:
        intent = np.full(self.brain.action_dim, np.nan)
        result = self.brain._build_motor_input(intent, self.motor_obs)
        # First action_dim values must be 0.0
        np.testing.assert_array_equal(
            result[: self.brain.action_dim], np.zeros(self.brain.action_dim)
        )

    def test_posinf_in_intent_becomes_one(self) -> None:
        intent = np.full(self.brain.action_dim, np.inf)
        result = self.brain._build_motor_input(intent, self.motor_obs)
        np.testing.assert_array_equal(
            result[: self.brain.action_dim], np.ones(self.brain.action_dim)
        )

    def test_neginf_in_intent_becomes_minus_one(self) -> None:
        intent = np.full(self.brain.action_dim, -np.inf)
        result = self.brain._build_motor_input(intent, self.motor_obs)
        np.testing.assert_array_equal(
            result[: self.brain.action_dim], np.full(self.brain.action_dim, -1.0)
        )

    def test_mixed_nan_inf_sanitized(self) -> None:
        intent = np.zeros(self.brain.action_dim, dtype=float)
        intent[:4] = [np.nan, np.inf, -np.inf, 0.5]
        result = self.brain._build_motor_input(intent, self.motor_obs)
        expected = np.zeros(self.brain.action_dim, dtype=float)
        expected[:4] = [0.0, 1.0, -1.0, 0.5]
        np.testing.assert_array_equal(result[: self.brain.action_dim], expected)

    def test_motor_context_nan_inf_sanitized(self) -> None:
        intent = np.zeros(self.brain.action_dim, dtype=float)
        self.motor_obs[MOTOR_CONTEXT_INTERFACE.observation_key][:3] = [
            np.nan,
            np.inf,
            -np.inf,
        ]

        result = self.brain._build_motor_input(intent, self.motor_obs)

        motor_context = result[self.brain.action_dim :]
        self.assertEqual(motor_context.shape, (MOTOR_CONTEXT_INTERFACE.input_dim,))
        self.assertTrue(np.all(np.isfinite(motor_context)))
        np.testing.assert_array_equal(motor_context[:3], [0.0, 1.0, -1.0])

    def test_valid_intent_passes_through_unchanged(self) -> None:
        intent = np.linspace(0.1, 0.9, self.brain.action_dim, dtype=float)
        result = self.brain._build_motor_input(intent, self.motor_obs)
        np.testing.assert_allclose(result[: self.brain.action_dim], intent)

    def test_output_length_equals_action_dim_plus_motor_context_dim(self) -> None:
        intent = np.zeros(self.brain.action_dim)
        result = self.brain._build_motor_input(intent, self.motor_obs)
        expected_len = self.brain.action_dim + MOTOR_CONTEXT_INTERFACE.input_dim
        self.assertEqual(result.shape, (expected_len,))
