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

class SpiderReflexBrainIntegrationTest(SpiderReflexDecisionTestBase):
    def test_reflex_scale_zero_leaves_module_logits_unchanged(self) -> None:
        """
        Verifies that a global reflex scale of 0.0 prevents any reflex from altering module logits.

        Creates a SpiderBrain configured with reflex_scale=0.0 and a predator-like observation, runs a step,
        and asserts that the alert_center's neural logits are unchanged after reflex processing:
        - `neural_logits` equals `post_reflex_logits`
        - `reflex_delta_logits` is all zeros
        - `effective_reflex_scale` equals 0.0
        - `reflex_applied` is False
        """
        brain = SpiderBrain(
            seed=13,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="zero_reflex_scale",
                module_dropout=0.0,
                reflex_scale=0.0,
            ),
        )
        observation = self._build_observation(
            alert={
                "predator_visible": 1.0,
                "predator_certainty": 0.9,
                "predator_dx": 1.0,
                "predator_dy": 0.0,
                "predator_motion_salience": 1.0,
            },
            action_context={
                "predator_visible": 1.0,
                "predator_certainty": 0.9,
            },
        )

        decision = brain.act(observation, sample=False)
        alert_result = next(result for result in decision.module_results if result.name == "alert_center")

        np.testing.assert_allclose(alert_result.neural_logits, alert_result.post_reflex_logits)
        np.testing.assert_allclose(
            alert_result.reflex_delta_logits,
            np.zeros_like(alert_result.reflex_delta_logits),
        )
        self.assertAlmostEqual(alert_result.effective_reflex_scale, 0.0)
        self.assertFalse(alert_result.reflex_applied)

    def test_policy_mode_reflex_only_uses_post_reflex_proposals(self) -> None:
        observation = self._build_observation(
            alert={
                "predator_visible": 1.0,
                "predator_certainty": 1.0,
                "predator_dx": 1.0,
                "predator_dy": 0.0,
            },
            action_context={
                "predator_visible": 1.0,
                "predator_certainty": 1.0,
            },
        )

        decision = self.brain.act(observation, sample=False, policy_mode="reflex_only")
        alert_result = next(
            result for result in decision.module_results if result.name == "alert_center"
        )

        self.assertEqual(decision.policy_mode, "reflex_only")
        self.assertIsNotNone(alert_result.reflex)
        self.assertEqual(alert_result.reflex.action, "MOVE_LEFT")
        self.assertEqual(decision.action_idx, ACTION_TO_INDEX["MOVE_LEFT"])
        np.testing.assert_allclose(
            decision.motor_correction_logits,
            np.zeros(len(LOCOMOTION_ACTIONS), dtype=float),
        )

    def test_policy_mode_reflex_only_rejects_monolithic_architecture(self) -> None:
        monolithic_brain = SpiderBrain(
            seed=3,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="monolithic_policy",
                architecture="monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                disabled_modules=(),
                reflex_scale=0.0,
            ),
        )
        observation = self._build_observation()

        with self.assertRaisesRegex(ValueError, "reflex_only"):
            monolithic_brain.act(observation, sample=False, policy_mode="reflex_only")

    def test_global_and_module_reflex_scales_multiply(self) -> None:
        """
        Verifies that global reflex scale and per-module reflex scale multiply to produce the effective reflex scale and that reflex strengths/auxiliary weights are scaled accordingly.

        Asserts that:
        - The computed effective reflex scale for "alert_center" equals the product of the global reflex_scale and the module override (0.5 * 0.5 = 0.25).
        - The reflex logit strength is base_logit_strength multiplied by the effective reflex scale.
        - The reflex auxiliary weight is base_auxiliary_weight multiplied by the effective reflex scale.
        - The auxiliary gradients dictionary contains an entry for "alert_center".
        """
        brain = SpiderBrain(
            seed=17,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="scaled_reflexes",
                module_dropout=0.0,
                reflex_scale=0.5,
                module_reflex_scales={"alert_center": 0.5},
            ),
        )
        observation = self._build_observation(
            alert={
                "predator_visible": 1.0,
                "predator_certainty": 0.9,
                "predator_dx": 1.0,
                "predator_dy": 0.0,
                "predator_motion_salience": 1.0,
            },
            action_context={
                "predator_visible": 1.0,
                "predator_certainty": 0.9,
            },
        )

        decision = brain.act(observation, sample=False)
        alert_result = next(result for result in decision.module_results if result.name == "alert_center")
        aux_grads = brain._auxiliary_module_gradients(decision.module_results)
        base_logit_strength = float(brain.reflex_logit_strengths["alert_center"])
        base_auxiliary_weight = float(brain.reflex_aux_weights["alert_center"])

        self.assertAlmostEqual(alert_result.effective_reflex_scale, 0.25)
        self.assertAlmostEqual(
            alert_result.reflex.logit_strength,
            base_logit_strength * alert_result.effective_reflex_scale,
            places=6,
        )
        self.assertAlmostEqual(
            alert_result.reflex.auxiliary_weight,
            base_auxiliary_weight * alert_result.effective_reflex_scale,
            places=6,
        )
        self.assertIn("alert_center", aux_grads)

    def test_action_input_uses_contract_validated_action_context(self) -> None:
        """
        Verifies that the action input is formed by concatenating module logits with a validated action-context vector.

        Builds module results with distinct constant logits, creates an ActionContextObservation and converts it to the interface vector, calls _build_action_input with those inputs, and asserts the returned vector equals the concatenation of all module logits followed by the action-context vector.
        """
        module_results = []
        for idx, interface in enumerate(MODULE_INTERFACES, start=1):
            result = self._module_result(interface.name)
            result.logits = np.full(len(LOCOMOTION_ACTIONS), float(idx), dtype=float)
            module_results.append(result)

        action_context_view = ActionContextObservation(
            hunger=0.1,
            fatigue=0.2,
            health=0.3,
            recent_pain=0.4,
            recent_contact=0.5,
            on_food=0.6,
            on_shelter=0.7,
            predator_visible=0.8,
            predator_certainty=0.9,
            day=1.0,
            night=0.0,
            last_move_dx=-1.0,
            last_move_dy=1.0,
            sleep_debt=0.25,
            shelter_role_level=0.5,
        )
        observation = {
            ACTION_CONTEXT_INTERFACE.observation_key: ACTION_CONTEXT_INTERFACE.vector_from_mapping(
                action_context_view.as_mapping()
            )
        }

        action_input = self.brain._build_action_input(module_results, observation)

        expected = np.concatenate(
            [
                np.concatenate([result.logits for result in module_results], axis=0),
                ACTION_CONTEXT_INTERFACE.vector_from_mapping(action_context_view.as_mapping()),
            ],
            axis=0,
        )
        np.testing.assert_allclose(action_input, expected)

    def test_action_input_rejects_invalid_action_context_shape(self) -> None:
        module_results = [self._module_result(interface.name) for interface in MODULE_INTERFACES]

        with self.assertRaisesRegex(ValueError, "action_center_context"):
            self.brain._build_action_input(
                module_results,
                {
                    ACTION_CONTEXT_INTERFACE.observation_key: np.zeros(
                        ACTION_CONTEXT_INTERFACE.input_dim - 1,
                        dtype=float,
                    )
                },
            )

    def test_motor_input_uses_contract_validated_motor_context(self) -> None:
        """
        Verifies that _build_motor_input produces a concatenation of a validated action intent and the motor-context vector.

        Constructs a MotorContextObservation, converts it through MOTOR_CONTEXT_INTERFACE, calls _build_motor_input with a one-hot action intent and the observation, and asserts the returned motor input equals the action intent followed by the validated motor-context vector.
        """
        motor_context_view = MotorContextObservation(
            on_food=0.6,
            on_shelter=0.7,
            predator_visible=0.8,
            predator_certainty=0.9,
            day=1.0,
            night=0.0,
            last_move_dx=-1.0,
            last_move_dy=1.0,
            shelter_role_level=0.5,
            heading_dx=1.0,
            heading_dy=0.0,
            terrain_difficulty=0.4,
            fatigue=0.2,
        )
        action_intent = np.zeros(len(LOCOMOTION_ACTIONS), dtype=float)
        action_intent[ACTION_TO_INDEX["MOVE_DOWN"]] = 1.0
        observation = {
            MOTOR_CONTEXT_INTERFACE.observation_key: MOTOR_CONTEXT_INTERFACE.vector_from_mapping(
                motor_context_view.as_mapping()
            )
        }

        motor_input = self.brain._build_motor_input(action_intent, observation)

        expected = np.concatenate(
            [
                action_intent,
                MOTOR_CONTEXT_INTERFACE.vector_from_mapping(motor_context_view.as_mapping()),
            ],
            axis=0,
        )
        np.testing.assert_allclose(motor_input, expected)

    def test_motor_input_rejects_invalid_motor_context_shape(self) -> None:
        action_intent = np.zeros(len(LOCOMOTION_ACTIONS), dtype=float)

        with self.assertRaisesRegex(ValueError, "motor_cortex_context"):
            self.brain._build_motor_input(
                action_intent,
                {
                    MOTOR_CONTEXT_INTERFACE.observation_key: np.zeros(
                        MOTOR_CONTEXT_INTERFACE.input_dim - 1,
                        dtype=float,
                    )
                },
            )

    def test_motor_input_rejects_invalid_action_intent_shape(self) -> None:
        with self.assertRaisesRegex(ValueError, "action_intent"):
            self.brain._build_motor_input(
                np.zeros(len(LOCOMOTION_ACTIONS) - 1, dtype=float),
                {
                    MOTOR_CONTEXT_INTERFACE.observation_key: np.zeros(
                        MOTOR_CONTEXT_INTERFACE.input_dim,
                        dtype=float,
                    )
                },
            )

    def test_reflex_aux_weight_from_operational_profile(self) -> None:
        # Verify that a custom reflex auxiliary weight from the operational profile
        # changes the magnitude of the auxiliary gradient compared to default.
        default_brain = SpiderBrain(seed=3, module_dropout=0.0)
        custom_brain = SpiderBrain(
            seed=3,
            module_dropout=0.0,
            operational_profile=self._profile_with_reflex_config_updates(alert_aux_weight=0.99),
        )
        result = self._module_result(
            "alert_center",
            predator_visible=1.0,
            predator_certainty=0.9,
            predator_dx=0.0,
            predator_dy=1.0,
        )
        result.reflex = self._reflex_decision(result, brain=default_brain)
        custom_result = self._module_result(
            "alert_center",
            predator_visible=1.0,
            predator_certainty=0.9,
            predator_dx=0.0,
            predator_dy=1.0,
        )
        custom_result.reflex = self._reflex_decision(custom_result, brain=custom_brain)

        default_grads = default_brain._auxiliary_module_gradients([result])
        custom_grads = custom_brain._auxiliary_module_gradients([custom_result])

        # Default aux weight is 0.32, custom is 0.99; gradients should differ in magnitude
        default_mag = float(np.linalg.norm(default_grads["alert_center"]))
        custom_mag = float(np.linalg.norm(custom_grads["alert_center"]))
        self.assertGreater(custom_mag, default_mag)

    def test_operational_profile_stored_on_brain(self) -> None:
        profile = self._profile_with_reflex_config_updates(alert_aux_weight=0.5)
        brain = SpiderBrain(seed=7, module_dropout=0.0, operational_profile=profile)
        self.assertIs(brain.operational_profile, profile)

    def test_brain_reflex_thresholds_come_from_operational_profile(self) -> None:
        profile = self._profile_with_reflex_config_updates(visual_predator_certainty=0.99)
        brain = SpiderBrain(seed=7, module_dropout=0.0, operational_profile=profile)
        self.assertAlmostEqual(brain.reflex_thresholds["visual_cortex"]["predator_certainty"], 0.99)

    def test_brain_reflex_aux_weights_come_from_operational_profile(self) -> None:
        profile = self._profile_with_reflex_config_updates(alert_aux_weight=0.77)
        brain = SpiderBrain(seed=7, module_dropout=0.0, operational_profile=profile)
        self.assertAlmostEqual(brain.reflex_aux_weights["alert_center"], 0.77)

    def test_brain_defaults_to_default_operational_profile(self) -> None:
        brain = SpiderBrain(seed=7, module_dropout=0.0)
        self.assertEqual(brain.operational_profile.to_summary()["name"], "default_v1")

    def test_hunger_reflex_threshold_uses_operational_profile(self) -> None:
        # Raise stay_hunger threshold so the reflex won't fire for moderate hunger
        summary = DEFAULT_OPERATIONAL_PROFILE.to_summary()
        summary["name"] = "hunger_threshold_test"
        summary["version"] = 50
        summary["brain"]["reflex_thresholds"]["hunger_center"]["stay_hunger"] = 0.90
        profile = OperationalProfile.from_summary(summary)
        custom_brain = SpiderBrain(seed=3, module_dropout=0.0, operational_profile=profile)
        result = self._module_result(
            "hunger_center",
            on_food=1.0,
            hunger=0.5,
        )
        self.assertIsNone(self._reflex_decision(result, brain=custom_brain))

    def test_set_runtime_reflex_scale_updates_current_scale(self) -> None:
        brain = SpiderBrain(seed=7, module_dropout=0.0)
        brain.set_runtime_reflex_scale(0.5)
        self.assertAlmostEqual(brain.current_reflex_scale, 0.5)

    def test_set_runtime_reflex_scale_clamps_negative_to_zero(self) -> None:
        brain = SpiderBrain(seed=7, module_dropout=0.0)
        brain.set_runtime_reflex_scale(-1.0)
        self.assertAlmostEqual(brain.current_reflex_scale, 0.0)

    def test_set_runtime_reflex_scale_zero_is_accepted(self) -> None:
        brain = SpiderBrain(seed=7, module_dropout=0.0)
        brain.set_runtime_reflex_scale(0.0)
        self.assertAlmostEqual(brain.current_reflex_scale, 0.0)

    def test_set_runtime_reflex_scale_large_value_accepted(self) -> None:
        brain = SpiderBrain(seed=7, module_dropout=0.0)
        brain.set_runtime_reflex_scale(3.0)
        self.assertAlmostEqual(brain.current_reflex_scale, 3.0)

    def test_set_runtime_reflex_scale_non_finite_raises_value_error(self) -> None:
        brain = SpiderBrain(seed=7, module_dropout=0.0)
        with self.assertRaises(ValueError):
            brain.set_runtime_reflex_scale(float("nan"))

    def test_reset_runtime_reflex_scale_restores_config_value(self) -> None:
        config = BrainAblationConfig(name="test_brain", module_dropout=0.0, reflex_scale=0.6)
        brain = SpiderBrain(seed=7, module_dropout=0.0, config=config)
        brain.set_runtime_reflex_scale(0.1)
        brain.reset_runtime_reflex_scale()
        self.assertAlmostEqual(brain.current_reflex_scale, 0.6)

    def test_reset_runtime_reflex_scale_default_config_restores_to_one(self) -> None:
        brain = SpiderBrain(seed=7, module_dropout=0.0)
        brain.set_runtime_reflex_scale(0.0)
        brain.reset_runtime_reflex_scale()
        self.assertAlmostEqual(brain.current_reflex_scale, 1.0)

    def test_reset_runtime_reflex_scale_rejects_non_finite_config_value(self) -> None:
        brain = SpiderBrain(seed=7, module_dropout=0.0)
        brain.config = SimpleNamespace(reflex_scale=float("inf"))  # type: ignore[assignment]
        with self.assertRaises(ValueError):
            brain.reset_runtime_reflex_scale()

    def test_effective_reflex_scale_returns_zero_for_monolithic(self) -> None:
        config = BrainAblationConfig(name="mono", architecture="monolithic")
        brain = SpiderBrain(seed=7, module_dropout=0.0, config=config)
        result = brain._effective_reflex_scale("alert_center")
        self.assertAlmostEqual(result, 0.0)

    def test_effective_reflex_scale_returns_zero_when_reflexes_disabled(self) -> None:
        config = BrainAblationConfig(
            name="no_reflex",
            architecture="modular",
            module_dropout=0.0,
            enable_reflexes=False,
        )
        brain = SpiderBrain(seed=7, module_dropout=0.0, config=config)
        result = brain._effective_reflex_scale("alert_center")
        self.assertAlmostEqual(result, 0.0)

    def test_effective_reflex_scale_uses_current_reflex_scale(self) -> None:
        brain = SpiderBrain(seed=7, module_dropout=0.0)
        brain.set_runtime_reflex_scale(0.4)
        result = brain._effective_reflex_scale("alert_center")
        self.assertAlmostEqual(result, 0.4)

    def test_effective_reflex_scale_multiplies_module_override(self) -> None:
        config = BrainAblationConfig(
            name="scaled",
            module_dropout=0.0,
            reflex_scale=0.8,
            module_reflex_scales={"alert_center": 0.5},
        )
        brain = SpiderBrain(seed=7, module_dropout=0.0, config=config)
        result = brain._effective_reflex_scale("alert_center")
        self.assertAlmostEqual(result, 0.4)

    def test_effective_reflex_scale_module_not_in_overrides_uses_one(self) -> None:
        config = BrainAblationConfig(
            name="partial_scale",
            module_dropout=0.0,
            reflex_scale=0.6,
            module_reflex_scales={"alert_center": 0.5},
        )
        brain = SpiderBrain(seed=7, module_dropout=0.0, config=config)
        result = brain._effective_reflex_scale("hunger_center")
        self.assertAlmostEqual(result, 0.6)

    def test_effective_reflex_scale_clamps_negative_product_to_zero(self) -> None:
        # Negative runtime inputs are clamped through the public setter before use.
        brain = SpiderBrain(seed=7, module_dropout=0.0)
        brain.set_runtime_reflex_scale(-1.0)
        result = brain._effective_reflex_scale("alert_center")
        self.assertGreaterEqual(result, 0.0)
        self.assertAlmostEqual(result, 0.0)
