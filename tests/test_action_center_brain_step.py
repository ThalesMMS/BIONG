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
        self.assertEqual(SpiderBrain.ARCHITECTURE_VERSION, 12)

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
        num_modules = len(self.brain.module_bank.specs)
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

class EstimateValueTest(unittest.TestCase):
    """Tests that estimate_value uses action_center (which has a value head)."""

    def setUp(self) -> None:
        self.brain = SpiderBrain(seed=17, module_dropout=0.0)
        self.obs = _blank_obs()

    def test_estimate_value_returns_finite_float(self) -> None:
        value = self.brain.estimate_value(self.obs)
        self.assertIsInstance(value, float)
        self.assertTrue(np.isfinite(value))

    def test_zeroing_motor_cortex_does_not_eliminate_value(self) -> None:
        """Zeroing motor_cortex should not make estimate_value return 0.0
        because estimate_value uses action_center, not motor_cortex."""
        # Zero all motor_cortex parameters
        self.brain.motor_cortex.W1.fill(0.0)
        self.brain.motor_cortex.b1.fill(0.0)
        self.brain.motor_cortex.W2.fill(0.0)
        self.brain.motor_cortex.b2.fill(0.0)
        # estimate_value should still work through action_center
        value = self.brain.estimate_value(self.obs)
        self.assertIsInstance(value, float)
        self.assertTrue(np.isfinite(value))

    def test_zeroing_action_center_weights_affects_value(self) -> None:
        """Zeroing action_center weights and biases should make value 0.0
        since action_center.W2_value @ 0 + 0 = 0."""
        self.brain.action_center.W1.fill(0.0)
        self.brain.action_center.b1.fill(0.0)
        self.brain.action_center.W2_policy.fill(0.0)
        self.brain.action_center.b2_policy.fill(0.0)
        self.brain.action_center.W2_value.fill(0.0)
        self.brain.action_center.b2_value.fill(0.0)
        value = self.brain.estimate_value(self.obs)
        self.assertAlmostEqual(value, 0.0, places=10)

    def test_estimate_value_matches_act_value(self) -> None:
        """estimate_value and act().value can differ (different cache states),
        but both should be finite."""
        step = self.brain.act(self.obs, bus=None, sample=False)
        value = self.brain.estimate_value(self.obs)
        self.assertTrue(np.isfinite(step.value))
        self.assertTrue(np.isfinite(value))

class ObserveWorldActionContextTest(unittest.TestCase):
    def setUp(self) -> None:
        self.world = SpiderWorld(seed=23)
        self.obs = self.world.reset(seed=23)

    def test_action_context_key_present(self) -> None:
        self.assertIn("action_context", self.obs)

    def test_action_context_is_numpy_array(self) -> None:
        """
        Assert that the 'action_context' entry in the test observation is a NumPy ndarray.

        Verifies the observation mapping includes the "action_context" key whose value is an instance of numpy.ndarray.
        """
        self.assertIsInstance(self.obs["action_context"], np.ndarray)

    def test_action_context_shape(self) -> None:
        self.assertEqual(self.obs["action_context"].shape, (15,))

    def test_motor_context_shape_is_14(self) -> None:
        """The motor_context vector includes momentum as field #14."""
        self.assertEqual(self.obs["motor_context"].shape, (14,))

    def test_action_context_reflects_hunger(self) -> None:
        self.world.state.hunger = 0.66
        obs = self.world.observe()
        mapping = ACTION_CONTEXT_INTERFACE.bind_values(obs["action_context"])
        self.assertAlmostEqual(mapping["hunger"], 0.66, places=5)

    def test_action_context_has_sleep_debt(self) -> None:
        self.world.state.sleep_debt = 0.44
        obs = self.world.observe()
        mapping = ACTION_CONTEXT_INTERFACE.bind_values(obs["action_context"])
        self.assertAlmostEqual(mapping["sleep_debt"], 0.44, places=5)

    def test_motor_context_lacks_hunger(self) -> None:
        """motor_context vector must not contain a 'hunger' entry."""
        mapping = MOTOR_CONTEXT_INTERFACE.bind_values(self.obs["motor_context"])
        self.assertNotIn("hunger", mapping)

    def test_motor_context_lacks_sleep_debt(self) -> None:
        mapping = MOTOR_CONTEXT_INTERFACE.bind_values(self.obs["motor_context"])
        self.assertNotIn("sleep_debt", mapping)

    def test_action_context_and_motor_context_share_on_food_value(self) -> None:
        """Both contexts must agree on on_food flag."""
        obs = self.world.observe()
        ac_mapping = ACTION_CONTEXT_INTERFACE.bind_values(obs["action_context"])
        mc_mapping = MOTOR_CONTEXT_INTERFACE.bind_values(obs["motor_context"])
        self.assertAlmostEqual(ac_mapping["on_food"], mc_mapping["on_food"], places=5)

    def test_action_context_and_motor_context_share_on_shelter_value(self) -> None:
        obs = self.world.observe()
        ac_mapping = ACTION_CONTEXT_INTERFACE.bind_values(obs["action_context"])
        mc_mapping = MOTOR_CONTEXT_INTERFACE.bind_values(obs["motor_context"])
        self.assertAlmostEqual(ac_mapping["on_shelter"], mc_mapping["on_shelter"], places=5)

class ActionCenterRegressionTest(unittest.TestCase):
    """Additional boundary and regression cases."""

    def test_brain_step_greedy_and_sample_give_valid_actions(self) -> None:
        brain = SpiderBrain(seed=31, module_dropout=0.0)
        obs = _blank_obs()
        for sample in (True, False):
            with self.subTest(sample=sample):
                step = brain.act(obs, bus=None, sample=sample)
                self.assertIn(step.action_idx, range(len(LOCOMOTION_ACTIONS)))
                self.assertIn(step.action_intent_idx, range(len(LOCOMOTION_ACTIONS)))

    def test_action_center_logits_are_finite(self) -> None:
        brain = SpiderBrain(seed=7, module_dropout=0.0)
        obs = _blank_obs()
        step = brain.act(obs, bus=None, sample=False)
        self.assertTrue(np.all(np.isfinite(step.action_center_logits)))

    def test_motor_correction_logits_are_finite(self) -> None:
        brain = SpiderBrain(seed=7, module_dropout=0.0)
        obs = _blank_obs()
        step = brain.act(obs, bus=None, sample=False)
        self.assertTrue(np.all(np.isfinite(step.motor_correction_logits)))

    def test_action_center_input_is_finite(self) -> None:
        brain = SpiderBrain(seed=7, module_dropout=0.0)
        obs = _blank_obs()
        step = brain.act(obs, bus=None, sample=False)
        self.assertTrue(np.all(np.isfinite(step.action_center_input)))

    def test_action_center_policy_does_not_contain_nan(self) -> None:
        brain = SpiderBrain(seed=77, module_dropout=0.0)
        obs = _blank_obs()
        step = brain.act(obs, bus=None, sample=False)
        self.assertFalse(np.any(np.isnan(step.action_center_policy)))

    def test_motor_override_false_when_no_motor_correction(self) -> None:
        """If motor_cortex weights are zero, correction logits are 0; motor_override
        should be False when action_center intent matches total argmax."""
        brain = SpiderBrain(seed=55, module_dropout=0.0)
        # Zero motor_cortex so it adds no correction
        brain.motor_cortex.W1.fill(0.0)
        brain.motor_cortex.b1.fill(0.0)
        brain.motor_cortex.W2.fill(0.0)
        brain.motor_cortex.b2.fill(0.0)
        obs = _blank_obs()
        step = brain.act(obs, bus=None, sample=False)
        # With zero motor correction, action_center_logits == total_logits
        np.testing.assert_allclose(step.action_center_logits, step.total_logits, atol=1e-10)
        self.assertFalse(step.motor_override)

    def test_action_context_observation_key_is_action_context(self) -> None:
        self.assertEqual(ActionContextObservation.observation_key, "action_context")

    def test_motor_context_observation_key_is_motor_context(self) -> None:
        self.assertEqual(MotorContextObservation.observation_key, "motor_context")

    def test_action_dim_matches_locomotion_actions(self) -> None:
        """action_dim on brain must match the locomotion action contract."""
        brain = SpiderBrain(seed=1, module_dropout=0.0)
        self.assertEqual(brain.action_dim, len(LOCOMOTION_ACTIONS))

class BoundObservationTest(unittest.TestCase):
    """Tests for SpiderBrain._bound_observation."""

    def setUp(self) -> None:
        self.brain = SpiderBrain(seed=1, module_dropout=0.0)

    def test_valid_interface_returns_dict(self) -> None:
        obs = _blank_obs()
        result = self.brain._bound_observation("visual_cortex", obs)
        self.assertIsInstance(result, dict)

    def test_valid_interface_keys_are_strings(self) -> None:
        obs = _blank_obs()
        result = self.brain._bound_observation("hunger_center", obs)
        for key in result:
            self.assertIsInstance(key, str)

    def test_valid_interface_values_are_floats(self) -> None:
        obs = _blank_obs()
        result = self.brain._bound_observation("alert_center", obs)
        for value in result.values():
            self.assertIsInstance(value, float)

    def test_unknown_interface_raises_key_error(self) -> None:
        obs = _blank_obs()
        with self.assertRaises(KeyError):
            self.brain._bound_observation("nonexistent_interface", obs)

    def test_all_module_interfaces_bindable(self) -> None:
        obs = _blank_obs()
        for iface in MODULE_INTERFACES:
            result = self.brain._bound_observation(iface.name, obs)
            self.assertIsInstance(result, dict)
            self.assertGreater(len(result), 0)

    def test_nonzero_values_propagated(self) -> None:
        iface = _module_interface("hunger_center")
        obs = _blank_obs()
        obs[iface.observation_key] = _vector_for(iface, hunger=0.88)
        result = self.brain._bound_observation("hunger_center", obs)
        self.assertIn("hunger", result)
        self.assertAlmostEqual(result["hunger"], 0.88, places=5)

class ModuleValenceRolesTest(unittest.TestCase):
    """Tests for SpiderBrain.MODULE_VALENCE_ROLES mapping."""

    def test_alert_center_maps_to_threat(self) -> None:
        self.assertEqual(SpiderBrain.MODULE_VALENCE_ROLES["alert_center"], "threat")

    def test_hunger_center_maps_to_hunger(self) -> None:
        self.assertEqual(SpiderBrain.MODULE_VALENCE_ROLES["hunger_center"], "hunger")

    def test_sleep_center_maps_to_sleep(self) -> None:
        self.assertEqual(SpiderBrain.MODULE_VALENCE_ROLES["sleep_center"], "sleep")

    def test_visual_cortex_maps_to_support(self) -> None:
        self.assertEqual(SpiderBrain.MODULE_VALENCE_ROLES["visual_cortex"], "support")

    def test_sensory_cortex_maps_to_support(self) -> None:
        self.assertEqual(SpiderBrain.MODULE_VALENCE_ROLES["sensory_cortex"], "support")

    def test_monolithic_policy_maps_to_integrated_policy(self) -> None:
        self.assertEqual(
            SpiderBrain.MODULE_VALENCE_ROLES[SpiderBrain.MONOLITHIC_POLICY_NAME],
            "integrated_policy",
        )

    def test_missing_module_falls_back_to_support_via_get(self) -> None:
        role = SpiderBrain.MODULE_VALENCE_ROLES.get("unknown_module", "support")
        self.assertEqual(role, "support")

class ModuleResultNewFieldsDefaultsTest(unittest.TestCase):
    """Tests for the new ModuleResult fields (valence_role, gate_weight, gated_logits)."""

    def _make_result(self):
        """
        Create a default ModuleResult for testing with neutral action scores and a placeholder observation.

        The returned ModuleResult has zero `logits`, uniform `probs` across `LOCOMOTION_ACTIONS`, `name` set to "test_module", `observation_key` set to "test", a zero-valued observation of length 3, and `active` set to True.

        Returns:
            ModuleResult: A test ModuleResult instance populated as described above.
        """
        from spider_cortex_sim.modules import ModuleResult
        logits = np.zeros(len(LOCOMOTION_ACTIONS), dtype=float)
        probs = np.ones(len(LOCOMOTION_ACTIONS), dtype=float) / len(LOCOMOTION_ACTIONS)
        return ModuleResult(
            interface=None,
            name="test_module",
            observation_key="test",
            observation=np.zeros(3),
            logits=logits,
            probs=probs,
            active=True,
        )

    def test_valence_role_default_is_support(self) -> None:
        result = self._make_result()
        self.assertEqual(result.valence_role, "support")

    def test_gate_weight_default_is_one(self) -> None:
        result = self._make_result()
        self.assertEqual(result.gate_weight, 1.0)

    def test_gated_logits_default_is_none(self) -> None:
        result = self._make_result()
        self.assertIsNone(result.gated_logits)

    def test_all_module_results_after_act_have_non_none_gated_logits(self) -> None:
        brain = SpiderBrain(seed=7, module_dropout=0.0)
        step = brain.act(_blank_obs(), bus=None, sample=False)
        for result in step.module_results:
            self.assertIsNotNone(result.gated_logits)

    def test_all_module_results_have_valence_role_string(self) -> None:
        brain = SpiderBrain(seed=7, module_dropout=0.0)
        step = brain.act(_blank_obs(), bus=None, sample=False)
        for result in step.module_results:
            self.assertIsInstance(result.valence_role, str)

    def test_all_module_results_have_gate_weight_float(self) -> None:
        brain = SpiderBrain(seed=7, module_dropout=0.0)
        step = brain.act(_blank_obs(), bus=None, sample=False)
        for result in step.module_results:
            self.assertIsInstance(result.gate_weight, float)

class ValenceNormalizationEdgeCaseTest(unittest.TestCase):
    """Tests for learned valence probabilities when all evidence logits are equal."""

    def test_all_zero_evidence_logits_softmax_to_uniform_scores(self) -> None:
        import unittest.mock

        brain = SpiderBrain(seed=99, module_dropout=0.0)
        obs = _blank_obs()
        module_results = brain._proposal_results(obs, store_cache=False, training=False)

        with unittest.mock.patch("spider_cortex_sim.agent.clamp_unit", return_value=0.0):
            arb = brain._compute_arbitration(module_results, obs)

        for valence in SpiderBrain.VALENCE_ORDER:
            self.assertAlmostEqual(arb.valence_scores[valence], 0.25, places=5)
        self.assertEqual(arb.winning_valence, "threat")

        total = sum(arb.valence_scores.values())
        self.assertAlmostEqual(total, 1.0, places=5)

    def test_valence_scores_are_non_negative_for_blank_obs(self) -> None:
        brain = SpiderBrain(seed=99, module_dropout=0.0)
        step = brain.act(_blank_obs(), bus=None, sample=False)
        for valence, score in step.arbitration_decision.valence_scores.items():
            self.assertGreaterEqual(score, 0.0, f"Score for {valence} is negative")

class ReflexOnlyModeArbitrationTest(unittest.TestCase):
    """Tests that reflex_only mode does not expose arbitration from prior act calls."""

    def setUp(self) -> None:
        self.brain = SpiderBrain(seed=22, module_dropout=0.0)
        self.obs = _blank_obs()

    def test_reflex_only_mode_also_has_arbitration_decision(self) -> None:
        # reflex_only still runs arbitration since it's computed before the mode branch
        step = self.brain.act(self.obs, bus=None, sample=False, policy_mode="reflex_only")
        # Test that the act call succeeds and returns a BrainStep
        self.assertIsInstance(step, BrainStep)
        # Test that arbitration_decision is computed for reflex_only path
        self.assertIsNotNone(step.arbitration_decision)
        self.assertIsInstance(step.arbitration_decision, ArbitrationDecision)

    def test_invalid_policy_mode_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.brain.act(self.obs, bus=None, sample=False, policy_mode="invalid_mode")

class ArbitrationDecisionPredatorProximityCompatTest(unittest.TestCase):
    """Tests that the legacy predator_proximity key is injected and sorted correctly."""

    def _make_with_threat(self, extra_threat_keys: dict | None = None) -> "ArbitrationDecision":
        threat_evidence = {"predator_visible": 1.0, "predator_certainty": 0.9}
        if extra_threat_keys:
            threat_evidence.update(extra_threat_keys)
        return ArbitrationDecision(
            strategy="priority_gating",
            winning_valence="threat",
            valence_scores={"threat": 0.8, "hunger": 0.1, "sleep": 0.05, "exploration": 0.05},
            module_gates={"alert_center": 1.0},
            suppressed_modules=[],
            evidence={"threat": threat_evidence},
            intent_before_gating_idx=0,
            intent_after_gating_idx=0,
        )

    def test_predator_proximity_injected_as_zero(self) -> None:
        d = self._make_with_threat()
        payload = d.to_payload()
        self.assertIn("predator_proximity", payload["evidence"]["threat"])
        self.assertEqual(payload["evidence"]["threat"]["predator_proximity"], 0.0)

    def test_injected_predator_proximity_keys_are_sorted(self) -> None:
        d = self._make_with_threat()
        payload = d.to_payload()
        keys = list(payload["evidence"]["threat"].keys())
        self.assertEqual(keys, sorted(keys))

    def test_predator_proximity_not_injected_when_already_present(self) -> None:
        """If the evidence dict already has predator_proximity, it must not be overwritten."""
        d = ArbitrationDecision(
            strategy="priority_gating",
            winning_valence="threat",
            valence_scores={},
            module_gates={},
            suppressed_modules=[],
            evidence={"threat": {"predator_proximity": 0.77}},
            intent_before_gating_idx=0,
            intent_after_gating_idx=0,
        )
        payload = d.to_payload()
        # The already-present value should be preserved
        self.assertAlmostEqual(payload["evidence"]["threat"]["predator_proximity"], 0.77)

    def test_no_threat_evidence_no_injection(self) -> None:
        """If threat is absent from evidence, no predator_proximity is injected."""
        d = ArbitrationDecision(
            strategy="priority_gating",
            winning_valence="hunger",
            valence_scores={},
            module_gates={},
            suppressed_modules=[],
            evidence={"hunger": {"hunger": 0.9}},
            intent_before_gating_idx=0,
            intent_after_gating_idx=0,
        )
        payload = d.to_payload()
        self.assertNotIn("predator_proximity", payload["evidence"].get("hunger", {}))

class CorticalModuleBankResultFieldsTest(unittest.TestCase):
    """Tests that CorticalModuleBank.forward initializes new reflex fields on ModuleResult."""

    def _make_bank(self) -> CorticalModuleBank:
        """
        Create a CorticalModuleBank configured for deterministic test runs.

        Returns:
            CorticalModuleBank: instance with action_dim equal to len(LOCOMOTION_ACTIONS), an RNG seeded with 99, and module_dropout set to 0.0.
        """
        rng = np.random.default_rng(99)
        return CorticalModuleBank(
            action_dim=len(LOCOMOTION_ACTIONS),
            rng=rng,
            module_dropout=0.0,
        )

    def _blank_observation(self, bank: CorticalModuleBank) -> dict:
        """
        Build an observation dictionary containing zeroed input vectors for every spec in the cortical module bank.

        Parameters:
            bank (CorticalModuleBank): Module bank whose specs determine observation keys and input dimensions.

        Returns:
            dict: Mapping from each spec.observation_key to a NumPy float array of zeros with length equal to spec.input_dim.
        """
        obs = {}
        for spec in bank.specs:
            obs[spec.observation_key] = np.zeros(spec.input_dim, dtype=float)
        return obs

    def test_forward_sets_neural_logits_to_copy_of_logits(self) -> None:
        bank = self._make_bank()
        obs = self._blank_observation(bank)
        results = bank.forward(obs, store_cache=False, training=False)
        for result in results:
            if result.active:
                self.assertIsNotNone(result.neural_logits)
                np.testing.assert_allclose(result.neural_logits, result.logits)

    def test_forward_sets_reflex_delta_logits_to_zeros(self) -> None:
        bank = self._make_bank()
        obs = self._blank_observation(bank)
        results = bank.forward(obs, store_cache=False, training=False)
        for result in results:
            if result.active:
                self.assertIsNotNone(result.reflex_delta_logits)
                np.testing.assert_allclose(
                    result.reflex_delta_logits,
                    np.zeros_like(result.logits),
                )

    def test_forward_sets_post_reflex_logits_to_copy_of_logits(self) -> None:
        bank = self._make_bank()
        obs = self._blank_observation(bank)
        results = bank.forward(obs, store_cache=False, training=False)
        for result in results:
            if result.active:
                self.assertIsNotNone(result.post_reflex_logits)
                np.testing.assert_allclose(result.post_reflex_logits, result.logits)
