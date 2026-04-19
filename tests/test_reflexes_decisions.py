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

class SpiderReflexDecisionTest(SpiderReflexDecisionTestBase):
    def test_visual_reflex_function_accepts_explicit_context(self) -> None:
        observations = self._module_result(
            "visual_cortex",
            predator_visible=1.0,
            predator_certainty=0.9,
            predator_dx=1.0,
            predator_dy=0.0,
        ).named_observation()

        reflex = _visual_reflex_decision(
            observations,
            operational_profile=DEFAULT_OPERATIONAL_PROFILE,
            interface_registry=interface_registry(),
        )

        self.assertIsNotNone(reflex)
        self.assertEqual(reflex.action, "MOVE_LEFT")
        self.assertEqual(reflex.reason, "retreat_from_visible_predator")

        self._assert_reordered_reflex_decision(
            _visual_reflex_decision,
            observations,
            expected_action="MOVE_LEFT",
            expected_reason="retreat_from_visible_predator",
        )

    def test_sensory_reflex_function_accepts_explicit_context(self) -> None:
        observations = self._module_result(
            "sensory_cortex",
            recent_contact=1.0,
            predator_smell_dx=0.0,
            predator_smell_dy=1.0,
        ).named_observation()

        reflex = _sensory_reflex_decision(
            observations,
            operational_profile=DEFAULT_OPERATIONAL_PROFILE,
            interface_registry=interface_registry(),
        )

        self.assertIsNotNone(reflex)
        self.assertEqual(reflex.action, "MOVE_UP")
        self.assertEqual(reflex.reason, "retreat_from_immediate_threat")

        self._assert_reordered_reflex_decision(
            _sensory_reflex_decision,
            observations,
            expected_action="MOVE_UP",
            expected_reason="retreat_from_immediate_threat",
        )

    def test_hunger_reflex_function_accepts_explicit_context(self) -> None:
        observations = self._module_result(
            "hunger_center",
            hunger=0.8,
            on_food=1.0,
        ).named_observation()

        reflex = _hunger_reflex_decision(
            observations,
            operational_profile=DEFAULT_OPERATIONAL_PROFILE,
            interface_registry=interface_registry(),
        )

        self.assertIsNotNone(reflex)
        self.assertEqual(reflex.action, "STAY")
        self.assertEqual(reflex.reason, "stay_on_food")

        self._assert_reordered_reflex_decision(
            _hunger_reflex_decision,
            observations,
            expected_action="STAY",
            expected_reason="stay_on_food",
        )

    def test_sleep_reflex_function_accepts_explicit_context(self) -> None:
        observations = self._module_result(
            "sleep_center",
            on_shelter=1.0,
            sleep_phase_level=0.8,
            hunger=0.1,
        ).named_observation()

        reflex = _sleep_reflex_decision(
            observations,
            operational_profile=DEFAULT_OPERATIONAL_PROFILE,
            interface_registry=interface_registry(),
        )

        self.assertIsNotNone(reflex)
        self.assertEqual(reflex.action, "STAY")
        self.assertEqual(reflex.reason, "stay_while_sleeping")

        self._assert_reordered_reflex_decision(
            _sleep_reflex_decision,
            observations,
            expected_action="STAY",
            expected_reason="stay_while_sleeping",
        )

    def test_alert_reflex_function_accepts_explicit_context(self) -> None:
        observations = self._module_result(
            "alert_center",
            predator_visible=1.0,
            predator_certainty=0.9,
            predator_dx=1.0,
            predator_dy=0.0,
        ).named_observation()

        reflex = _alert_reflex_decision(
            observations,
            operational_profile=DEFAULT_OPERATIONAL_PROFILE,
            interface_registry=interface_registry(),
        )

        self.assertIsNotNone(reflex)
        self.assertEqual(reflex.action, "MOVE_LEFT")
        self.assertEqual(reflex.reason, "retreat_from_visible_predator")

        self._assert_reordered_reflex_decision(
            _alert_reflex_decision,
            observations,
            expected_action="MOVE_LEFT",
            expected_reason="retreat_from_visible_predator",
        )

    def test_apply_reflex_path_accepts_explicit_context(self) -> None:
        """
        Verify that _apply_reflex_path applies a hunger reflex to a ModuleResult when provided explicit context.
        
        Creates a hunger-center ModuleResult with hunger and on_food signals, calls _apply_reflex_path with explicit
        ablation_config, operational_profile, interface_registry, current_reflex_scale, and module_valence_roles,
        and asserts that a reflex was produced with action "STAY", that the reflex was applied, and that the module's
        valence_role is set to "hunger".
        """
        result = self._module_result(
            "hunger_center",
            hunger=0.8,
            on_food=1.0,
        )

        _apply_reflex_path(
            [result],
            ablation_config=self.brain.config,
            operational_profile=self.brain.operational_profile,
            interface_registry=self.brain._interface_registry(),
            current_reflex_scale=self.brain.current_reflex_scale,
            module_valence_roles=self.brain.MODULE_VALENCE_ROLES,
        )

        self.assertIsNotNone(result.reflex)
        self.assertEqual(result.reflex.action, "STAY")
        self.assertTrue(result.reflex_applied)
        self.assertEqual(result.valence_role, "hunger")

        reordered_result = self._module_result(
            "hunger_center",
            hunger=0.8,
            on_food=1.0,
        )
        registry = self._reordered_action_registry()

        _apply_reflex_path(
            [reordered_result],
            ablation_config=self.brain.config,
            operational_profile=self.brain.operational_profile,
            interface_registry=registry,
            current_reflex_scale=self.brain.current_reflex_scale,
            module_valence_roles=self.brain.MODULE_VALENCE_ROLES,
        )

        self.assertIsNotNone(reordered_result.reflex)
        self.assertEqual(reordered_result.reflex.action, "STAY")
        self.assertEqual(reordered_result.reflex.reason, "stay_on_food")
        self.assertTrue(reordered_result.reflex_applied)
        self.assertEqual(reordered_result.valence_role, "hunger")
        self._assert_registry_local_target(reordered_result.reflex, registry)

    def test_visual_reflex_uses_named_predator_signals(self) -> None:
        result = self._module_result(
            "visual_cortex",
            predator_visible=1.0,
            predator_certainty=0.8,
            predator_dx=1.0,
            predator_dy=0.0,
        )

        reflex = self._reflex_decision(result)

        self.assertIsNotNone(reflex)
        self.assertEqual(reflex.action, "MOVE_LEFT")
        self.assertEqual(reflex.reason, "retreat_from_visible_predator")
        self.assertEqual(
            set(reflex.triggers.keys()),
            {"predator_visible", "predator_certainty", "predator_dx", "predator_dy"},
        )

    def test_visual_reflex_skips_low_certainty_food(self) -> None:
        result = self._module_result(
            "visual_cortex",
            food_visible=1.0,
            food_certainty=0.2,
            food_dx=1.0,
            food_dy=0.0,
        )

        self.assertIsNone(self._reflex_decision(result))

    def test_hunger_reflex_follows_visible_food_when_hungry(self) -> None:
        result = self._module_result(
            "hunger_center",
            hunger=0.7,
            food_visible=1.0,
            food_certainty=0.9,
            food_dx=1.0,
            food_dy=0.0,
        )

        reflex = self._reflex_decision(result)

        self.assertIsNotNone(reflex)
        self.assertEqual(reflex.action, "MOVE_RIGHT")
        self.assertEqual(reflex.reason, "approach_visible_food")

    def test_hunger_reflex_skips_when_not_hungry(self) -> None:
        result = self._module_result(
            "hunger_center",
            hunger=0.1,
            food_visible=1.0,
            food_certainty=0.9,
            food_dx=1.0,
            food_dy=0.0,
        )

        self.assertIsNone(self._reflex_decision(result))

    def test_sleep_reflex_stays_when_already_sleeping_in_shelter(self) -> None:
        result = self._module_result(
            "sleep_center",
            on_shelter=1.0,
            sleep_phase_level=0.7,
            hunger=0.2,
        )

        reflex = self._reflex_decision(result)

        self.assertIsNotNone(reflex)
        self.assertEqual(reflex.action, "STAY")
        self.assertEqual(reflex.reason, "stay_while_sleeping")

    def test_sleep_reflex_skips_when_hunger_is_too_high(self) -> None:
        result = self._module_result(
            "sleep_center",
            on_shelter=1.0,
            sleep_phase_level=0.7,
            hunger=0.9,
        )

        self.assertIsNone(self._reflex_decision(result))

    def test_alert_reflex_retreats_from_visible_predator(self) -> None:
        """
        Verifies that the alert reflex causes a retreat when a predator is clearly visible.
        
        Constructs an `alert_center` ModuleResult with predator visibility, high certainty, and position signals,
        calls the reflex decision helper, and asserts the reflex exists with action `"MOVE_LEFT"`
        and reason `"retreat_from_visible_predator"`.
        """
        result = self._module_result(
            "alert_center",
            predator_visible=1.0,
            predator_certainty=0.9,
            predator_dx=1.0,
            predator_dy=0.0,
        )

        reflex = self._reflex_decision(result)

        self.assertIsNotNone(reflex)
        self.assertEqual(reflex.action, "MOVE_LEFT")
        self.assertEqual(reflex.reason, "retreat_from_visible_predator")

    def test_alert_reflex_skips_when_threat_signals_are_absent(self) -> None:
        result = self._module_result("alert_center")

        self.assertIsNone(self._reflex_decision(result))

    def test_alert_reflex_retreats_from_predator_trace_when_occluded(self) -> None:
        """retreat_from_predator_trace fires when predator is occluded and a trace exists."""
        result = self._module_result(
            "alert_center",
            predator_occluded=1.0,
            predator_trace_dx=1.0,
            predator_trace_dy=0.0,
            predator_trace_strength=0.8,
        )

        reflex = self._reflex_decision(result)

        self.assertIsNotNone(reflex)
        self.assertEqual(reflex.reason, "retreat_from_predator_trace")
        # retreat direction is away from trace (dx=1.0 => move LEFT)
        self.assertEqual(reflex.action, "MOVE_LEFT")

    def test_alert_reflex_retreat_from_predator_trace_skipped_when_trace_strength_zero(self) -> None:
        """retreat_from_predator_trace must NOT fire when predator_trace_strength == 0.0."""
        result = self._module_result(
            "alert_center",
            predator_occluded=1.0,
            predator_smell_strength=0.9,
            predator_trace_dx=1.0,
            predator_trace_dy=0.0,
            predator_trace_strength=0.0,  # zero strength blocks the reflex
        )

        # With zero trace strength, the new guard prevents this branch from firing.
        # The reflex may be None or may take an earlier branch (memory/escape); we only
        # verify it does NOT produce the predator-trace reason.
        reflex = self._reflex_decision(result)
        if reflex is not None:
            self.assertNotEqual(reflex.reason, "retreat_from_predator_trace")

    def test_alert_reflex_predator_trace_triggers_include_trace_fields(self) -> None:
        """retreat_from_predator_trace reflex triggers must include trace fields."""
        result = self._module_result(
            "alert_center",
            predator_smell_strength=0.9,
            predator_trace_dx=0.0,
            predator_trace_dy=1.0,
            predator_trace_strength=0.6,
        )

        reflex = self._reflex_decision(result)

        self.assertIsNotNone(reflex)
        self.assertEqual(reflex.reason, "retreat_from_predator_trace")
        self.assertIn("predator_trace_dx", reflex.triggers)
        self.assertIn("predator_trace_dy", reflex.triggers)
        self.assertIn("predator_trace_strength", reflex.triggers)

    def test_alert_reflex_predator_trace_does_not_include_home_vector(self) -> None:
        """The predator-trace reflex must not reference home_dx or home_dy (removed fields)."""
        result = self._module_result(
            "alert_center",
            predator_occluded=1.0,
            predator_trace_dx=1.0,
            predator_trace_dy=0.0,
            predator_trace_strength=0.7,
        )

        reflex = self._reflex_decision(result)

        self.assertIsNotNone(reflex)
        self.assertEqual(reflex.reason, "retreat_from_predator_trace")
        self.assertNotIn("home_dx", reflex.triggers)
        self.assertNotIn("home_dy", reflex.triggers)

    def test_alert_reflex_predator_trace_fires_when_recent_pain(self) -> None:
        """retreat_from_predator_trace fires on recent_pain when a trace exists."""
        result = self._module_result(
            "alert_center",
            recent_pain=0.9,
            predator_trace_dx=-1.0,
            predator_trace_dy=0.0,
            predator_trace_strength=0.5,
        )

        reflex = self._reflex_decision(result)

        self.assertIsNotNone(reflex)
        self.assertEqual(reflex.reason, "retreat_from_predator_trace")

    def test_alert_reflex_predator_trace_fires_when_recent_contact(self) -> None:
        """retreat_from_predator_trace fires on recent_contact when a trace exists."""
        result = self._module_result(
            "alert_center",
            recent_contact=0.9,
            predator_trace_dx=0.0,
            predator_trace_dy=-1.0,
            predator_trace_strength=0.5,
        )

        reflex = self._reflex_decision(result)

        self.assertIsNotNone(reflex)
        self.assertEqual(reflex.reason, "retreat_from_predator_trace")

    def test_sleep_reflex_follows_shelter_trace_when_fatigued(self) -> None:
        """follow_shelter_trace_to_rest fires when fatigued and a shelter trace exists."""
        result = self._module_result(
            "sleep_center",
            fatigue=0.9,
            hunger=0.1,
            shelter_trace_dx=1.0,
            shelter_trace_dy=0.0,
            shelter_trace_strength=0.7,
        )

        reflex = self._reflex_decision(result)

        self.assertIsNotNone(reflex)
        self.assertEqual(reflex.reason, "follow_shelter_trace_to_rest")
        # direction toward trace (dx=1.0 => move RIGHT)
        self.assertEqual(reflex.action, "MOVE_RIGHT")

    def test_sleep_reflex_shelter_trace_skipped_when_trace_strength_zero(self) -> None:
        """follow_shelter_trace_to_rest must NOT fire when shelter_trace_strength == 0.0."""
        result = self._module_result(
            "sleep_center",
            fatigue=0.9,
            sleep_debt=0.9,
            hunger=0.1,
            shelter_trace_dx=1.0,
            shelter_trace_dy=0.0,
            shelter_trace_strength=0.0,  # zero strength blocks the reflex
        )

        reflex = self._reflex_decision(result)
        if reflex is not None:
            self.assertNotEqual(reflex.reason, "follow_shelter_trace_to_rest")

    def test_sleep_reflex_shelter_trace_triggers_include_trace_fields(self) -> None:
        """follow_shelter_trace_to_rest triggers must contain shelter trace fields."""
        result = self._module_result(
            "sleep_center",
            fatigue=0.9,
            hunger=0.1,
            shelter_trace_dx=0.0,
            shelter_trace_dy=1.0,
            shelter_trace_strength=0.5,
        )

        reflex = self._reflex_decision(result)

        self.assertIsNotNone(reflex)
        self.assertEqual(reflex.reason, "follow_shelter_trace_to_rest")
        self.assertIn("shelter_trace_dx", reflex.triggers)
        self.assertIn("shelter_trace_dy", reflex.triggers)
        self.assertIn("shelter_trace_strength", reflex.triggers)

    def test_sleep_reflex_shelter_trace_does_not_include_home_vector(self) -> None:
        """The shelter-trace reflex must not reference home_dx or home_dy (removed fields)."""
        result = self._module_result(
            "sleep_center",
            fatigue=0.9,
            hunger=0.1,
            shelter_trace_dx=1.0,
            shelter_trace_dy=0.0,
            shelter_trace_strength=0.6,
        )

        reflex = self._reflex_decision(result)

        self.assertIsNotNone(reflex)
        self.assertEqual(reflex.reason, "follow_shelter_trace_to_rest")
        self.assertNotIn("home_dx", reflex.triggers)
        self.assertNotIn("home_dy", reflex.triggers)

    def test_sleep_reflex_shelter_trace_fires_during_night(self) -> None:
        """follow_shelter_trace_to_rest fires at night when a shelter trace exists."""
        result = self._module_result(
            "sleep_center",
            night=1.0,
            hunger=0.1,
            shelter_trace_dx=-1.0,
            shelter_trace_dy=0.0,
            shelter_trace_strength=0.4,
        )

        reflex = self._reflex_decision(result)

        self.assertIsNotNone(reflex)
        self.assertEqual(reflex.reason, "follow_shelter_trace_to_rest")

    def test_sleep_reflex_shelter_trace_blocked_by_high_hunger(self) -> None:
        """follow_shelter_trace_to_rest must not fire when hunger exceeds threshold."""
        result = self._module_result(
            "sleep_center",
            fatigue=0.9,
            night=1.0,
            hunger=0.99,  # very hungry; blocks sleep-related movement
            shelter_trace_dx=1.0,
            shelter_trace_dy=0.0,
            shelter_trace_strength=0.8,
        )

        reflex = self._reflex_decision(result)
        if reflex is not None:
            self.assertNotEqual(reflex.reason, "follow_shelter_trace_to_rest")

    def test_auxiliary_gradients_use_recorded_reflex_target(self) -> None:
        custom_brain = SpiderBrain(
            seed=3,
            module_dropout=0.0,
            operational_profile=self._profile_with_reflex_config_updates(alert_aux_weight=0.77),
        )
        result = self._module_result(
            "alert_center",
            predator_visible=1.0,
            predator_certainty=0.9,
            predator_dx=1.0,
            predator_dy=0.0,
        )
        result.reflex = self._reflex_decision(result, brain=custom_brain)

        aux_grads = custom_brain._auxiliary_module_gradients([result])

        self.assertIn("alert_center", aux_grads)
        expected = custom_brain.reflex_aux_weights["alert_center"] * (result.probs - result.reflex.target_probs)
        np.testing.assert_allclose(aux_grads["alert_center"], expected)
        self.assertEqual(int(np.argmax(result.reflex.target_probs)), ACTION_TO_INDEX[result.reflex.action])

    def test_visual_reflex_threshold_uses_operational_profile(self) -> None:
        custom_brain = SpiderBrain(
            seed=5,
            module_dropout=0.0,
            operational_profile=self._profile_with_reflex_config_updates(visual_predator_certainty=0.95),
        )
        result = self._module_result(
            "visual_cortex",
            predator_visible=1.0,
            predator_certainty=0.8,
            predator_dx=1.0,
            predator_dy=0.0,
        )

        self.assertIsNone(self._reflex_decision(result, brain=custom_brain))
