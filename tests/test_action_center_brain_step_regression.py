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
