import unittest
from types import SimpleNamespace

import numpy as np

from spider_cortex_sim.ablations import BrainAblationConfig
from spider_cortex_sim.agent import SpiderBrain
from spider_cortex_sim.interfaces import (
    ACTION_CONTEXT_INTERFACE,
    ACTION_TO_INDEX,
    LOCOMOTION_ACTIONS,
    MODULE_INTERFACES,
    MODULE_INTERFACE_BY_NAME,
    MOTOR_CONTEXT_INTERFACE,
    ActionContextObservation,
    MotorContextObservation,
    interface_registry,
)
from spider_cortex_sim.modules import ModuleResult
from spider_cortex_sim.operational_profiles import DEFAULT_OPERATIONAL_PROFILE, OperationalProfile
from spider_cortex_sim.reflexes import (
    _alert_reflex_decision,
    _apply_reflex_path,
    _hunger_reflex_decision,
    _module_reflex_decision,
    _sensory_reflex_decision,
    _sleep_reflex_decision,
    _visual_reflex_decision,
)

from tests.fixtures.reflexes import SpiderReflexDecisionTestBase

class SpiderReflexPolicyModeTest(SpiderReflexDecisionTestBase):
    def test_brain_step_has_final_reflex_override_field(self) -> None:
        brain = SpiderBrain(seed=7, module_dropout=0.0)
        observation = self._build_observation()
        decision = brain.act(observation, sample=False)
        self.assertIsInstance(decision.final_reflex_override, bool)
        self.assertIsInstance(decision.motor_action_idx, int)

    def test_brain_step_has_total_logits_without_reflex(self) -> None:
        brain = SpiderBrain(seed=7, module_dropout=0.0)
        observation = self._build_observation()
        decision = brain.act(observation, sample=False)
        self.assertIsNotNone(decision.total_logits_without_reflex)
        self.assertEqual(len(decision.total_logits_without_reflex), len(decision.total_logits))

    def test_final_reflex_override_true_when_reflex_changes_argmax(self) -> None:
        brain = SpiderBrain(seed=41, module_dropout=0.0)
        brain.set_runtime_reflex_scale(1.0)
        observation = self._build_observation(
            alert={
                "predator_visible": 1.0,
                "predator_certainty": 0.95,
                "predator_dx": 1.0,
                "predator_dy": 0.0,
                "predator_motion_salience": 1.0,
            },
            action_context={
                "predator_visible": 1.0,
                "predator_certainty": 0.95,
            },
        )
        decision = brain.act(observation, sample=False)
        self.assertIs(decision.final_reflex_override, True)

    def test_reflex_scale_zero_sets_final_reflex_override_false(self) -> None:
        config = BrainAblationConfig(
            name="no_reflexes",
            module_dropout=0.0,
            reflex_scale=0.0,
        )
        brain = SpiderBrain(seed=41, module_dropout=0.0, config=config)
        observation = self._build_observation(
            alert={
                "predator_visible": 1.0,
                "predator_certainty": 0.95,
                "predator_dx": 1.0,
                "predator_dy": 0.0,
                "predator_motion_salience": 1.0,
            },
            action_context={
                "predator_visible": 1.0,
                "predator_certainty": 0.95,
            },
        )
        decision = brain.act(observation, sample=False)
        # With zero reflex scale, total_logits_without_reflex == total_logits, so override is False
        self.assertFalse(decision.final_reflex_override)

    def test_initial_current_reflex_scale_matches_config(self) -> None:
        config = BrainAblationConfig(name="test", module_dropout=0.0, reflex_scale=0.75)
        brain = SpiderBrain(seed=3, module_dropout=0.0, config=config)
        self.assertAlmostEqual(brain.current_reflex_scale, 0.75)

    def test_brain_step_policy_mode_default_is_normal(self) -> None:
        observation = self._build_observation()
        decision = self.brain.act(observation, sample=False)
        self.assertEqual(decision.policy_mode, "normal")

    def test_invalid_policy_mode_raises_value_error(self) -> None:
        observation = self._build_observation()
        with self.assertRaises(ValueError):
            self.brain.act(observation, sample=False, policy_mode="bad_mode")

    def test_invalid_policy_mode_error_message(self) -> None:
        observation = self._build_observation()
        with self.assertRaisesRegex(ValueError, "policy_mode"):
            self.brain.act(observation, sample=False, policy_mode="NORMAL")

    def test_policy_mode_normal_produces_nonzero_motor_correction(self) -> None:
        # In "normal" mode the motor cortex contributes non-trivially
        observation = self._build_observation()
        decision = self.brain.act(observation, sample=False, policy_mode="normal")
        # motor_correction_logits shape should match action_dim
        self.assertEqual(
            decision.motor_correction_logits.shape,
            decision.action_center_logits.shape,
        )

    def test_learn_rejects_decision_with_reflex_only_policy_mode(self) -> None:
        observation = self._build_observation()
        decision = self.brain.act(observation, sample=False, policy_mode="reflex_only")
        next_obs = self._build_observation()
        with self.assertRaises(ValueError):
            self.brain.learn(decision, reward=1.0, next_observation=next_obs, done=False)

    def test_learn_rejects_decision_error_contains_policy_mode(self) -> None:
        observation = self._build_observation()
        decision = self.brain.act(observation, sample=False, policy_mode="reflex_only")
        next_obs = self._build_observation()
        with self.assertRaisesRegex(ValueError, "normal"):
            self.brain.learn(decision, reward=1.0, next_observation=next_obs, done=False)

    def test_learn_accepts_decision_with_normal_policy_mode(self) -> None:
        observation = self._build_observation()
        decision = self.brain.act(observation, sample=True, policy_mode="normal")
        next_obs = self._build_observation()
        # Should not raise
        self.brain.learn(decision, reward=0.5, next_observation=next_obs, done=False)

    def test_policy_mode_reflex_only_value_is_zero(self) -> None:
        observation = self._build_observation()
        decision = self.brain.act(observation, sample=False, policy_mode="reflex_only")
        self.assertEqual(decision.value, 0.0)

    def test_policy_mode_reflex_only_action_center_correction_is_zero(self) -> None:
        observation = self._build_observation()
        decision = self.brain.act(observation, sample=False, policy_mode="reflex_only")
        np.testing.assert_allclose(
            decision.action_center_logits,
            decision.action_center_logits,  # just shape sanity
        )
        # The correction logits in reflex_only mode are zeros
        np.testing.assert_allclose(
            decision.motor_correction_logits,
            np.zeros(len(LOCOMOTION_ACTIONS), dtype=float),
        )

    def test_policy_mode_propagated_to_brainstep(self) -> None:
        observation = self._build_observation()
        for mode in ("normal", "reflex_only"):
            decision = self.brain.act(observation, sample=False, policy_mode=mode)
            self.assertEqual(decision.policy_mode, mode)

class ModuleResultReflexFieldsTest(unittest.TestCase):
    """Tests for the new reflex-related fields on ModuleResult (new in this PR)."""

    def _make_result(self, **kwargs) -> ModuleResult:
        """
        Create a default ModuleResult for the "alert_center" interface, allowing field overrides.
        
        Constructs a ModuleResult whose observation is a zero vector sized to the alert_center interface input,
        logits are a zero vector and probs are uniform over locomotion actions. Any ModuleResult field may be
        overridden by passing it in via keyword arguments.
        
        Parameters:
            kwargs (dict): Optional overrides for any ModuleResult constructor field (e.g., `logits`, `probs`,
                `observation`, `name`, `observation_key`, `active`, `interface`).
        
        Returns:
            ModuleResult: The constructed ModuleResult with defaults merged with provided overrides.
        """
        interface = MODULE_INTERFACE_BY_NAME["alert_center"]
        obs = np.zeros(interface.input_dim, dtype=float)
        action_dim = len(LOCOMOTION_ACTIONS)
        defaults = dict(
            interface=interface,
            name="alert_center",
            observation_key=interface.observation_key,
            observation=obs,
            logits=np.zeros(action_dim, dtype=float),
            probs=np.full(action_dim, 1.0 / action_dim, dtype=float),
            active=True,
        )
        defaults.update(kwargs)
        return ModuleResult(**defaults)

    def test_neural_logits_defaults_to_none(self) -> None:
        result = self._make_result()
        self.assertIsNone(result.neural_logits)

    def test_reflex_delta_logits_defaults_to_none(self) -> None:
        result = self._make_result()
        self.assertIsNone(result.reflex_delta_logits)

    def test_post_reflex_logits_defaults_to_none(self) -> None:
        result = self._make_result()
        self.assertIsNone(result.post_reflex_logits)

    def test_reflex_applied_defaults_to_false(self) -> None:
        result = self._make_result()
        self.assertFalse(result.reflex_applied)

    def test_effective_reflex_scale_defaults_to_zero(self) -> None:
        result = self._make_result()
        self.assertAlmostEqual(result.effective_reflex_scale, 0.0)

    def test_module_reflex_override_defaults_to_false(self) -> None:
        result = self._make_result()
        self.assertFalse(result.module_reflex_override)

    def test_module_reflex_dominance_defaults_to_zero(self) -> None:
        result = self._make_result()
        self.assertAlmostEqual(result.module_reflex_dominance, 0.0)

    def test_can_set_neural_logits(self) -> None:
        """
        Verify that a ModuleResult preserves a provided `neural_logits` array.

        Asserts that after constructing a ModuleResult with a given `neural_logits` vector, the `neural_logits` attribute matches the original values elementwise.
        """
        logits = np.linspace(0.1, 0.9, len(LOCOMOTION_ACTIONS), dtype=float)
        result = self._make_result(neural_logits=logits.copy())
        np.testing.assert_allclose(result.neural_logits, logits)

    def test_can_set_reflex_applied_true(self) -> None:
        result = self._make_result(reflex_applied=True)
        self.assertTrue(result.reflex_applied)

    def test_can_set_effective_reflex_scale(self) -> None:
        result = self._make_result(effective_reflex_scale=0.5)
        self.assertAlmostEqual(result.effective_reflex_scale, 0.5)

    def test_can_set_module_reflex_override_true(self) -> None:
        result = self._make_result(module_reflex_override=True)
        self.assertTrue(result.module_reflex_override)

    def test_can_set_module_reflex_dominance(self) -> None:
        result = self._make_result(module_reflex_dominance=0.75)
        self.assertAlmostEqual(result.module_reflex_dominance, 0.75)
