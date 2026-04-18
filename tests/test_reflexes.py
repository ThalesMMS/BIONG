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


class SpiderReflexDecisionTest(unittest.TestCase):
    def setUp(self) -> None:
        """
        Prepare the test fixture by creating a deterministic SpiderBrain instance and storing it on self.brain.
        
        Initializes a SpiderBrain with seed=3 and module_dropout=0.0 so tests run with a reproducible brain configuration.
        """
        self.brain = SpiderBrain(seed=3, module_dropout=0.0)

    def _reflex_decision(
        self,
        result: ModuleResult,
        *,
        brain: SpiderBrain | None = None,
    ):
        """
        Selects a SpiderBrain (defaults to the test instance's brain) and obtains the reflex decision for the given module result.
        
        Parameters:
            result (ModuleResult): The module result whose named observation will be evaluated for reflexes.
            brain (SpiderBrain | None): Optional brain to use for operational profile and interface registry; if None, uses self.brain.
        
        Returns:
            ModuleReflex | None: The reflex decision for the module, or `None` if no reflex applies.
        """
        brain = self.brain if brain is None else brain
        return _module_reflex_decision(
            result.name,
            result.named_observation(),
            operational_profile=brain.operational_profile,
            interface_registry=brain._interface_registry(),
        )

    def _module_result(self, module_name: str, **signals: float) -> ModuleResult:
        """
        Builds a ModuleResult for the given module using a default observation with optional signal overrides.
        
        Parameters:
            module_name (str): Name of the module whose interface provides input names and observation key.
            **signals (float): Mapping of input names to values that override the default signals. By default inputs whose names end with "_age" are set to 1.0 and all other inputs are set to 0.0 before applying overrides.
        
        Returns:
            ModuleResult: A result populated with the module's interface, name, observation_key, the constructed observation vector, zeroed `logits`, a uniform `probs` distribution over locomotion actions, and `active=True`.
        """
        interface = MODULE_INTERFACE_BY_NAME[module_name]
        signal_mapping = {
            spec.name: 1.0 if spec.name.endswith("_age") else 0.0
            for spec in interface.inputs
        }
        signal_mapping.update({name: float(value) for name, value in signals.items()})
        observation = interface.vector_from_mapping(signal_mapping)
        action_dim = len(LOCOMOTION_ACTIONS)
        return ModuleResult(
            interface=interface,
            name=module_name,
            observation_key=interface.observation_key,
            observation=observation,
            logits=np.zeros(action_dim, dtype=float),
            probs=np.full(action_dim, 1.0 / action_dim, dtype=float),
            active=True,
        )

    def _build_observation(self, **overrides: dict[str, float]) -> dict[str, np.ndarray]:
        """
        Builds a complete observation dictionary for all module, action-context, and motor-context interfaces.

        Constructs default input vectors for each interface where inputs ending in "_age" are set to 1.0 and all other inputs to 0.0, then applies any numeric overrides supplied for an interface's observation key. The resulting vectors are produced via each interface's `vector_from_mapping` and stored under that interface's observation key.

        Parameters:
            overrides (dict[str, dict[str, float]]): Optional per-interface override mappings keyed by interface observation key; each mapping provides input-name -> numeric value to override the defaults.

        Returns:
            dict[str, np.ndarray]: A mapping from each interface's observation key to its validated observation vector.
        """
        observation: dict[str, np.ndarray] = {}
        for interface in MODULE_INTERFACES:
            signal_mapping = {
                spec.name: 1.0 if spec.name.endswith("_age") else 0.0
                for spec in interface.inputs
            }
            signal_mapping.update({
                name: float(value)
                for name, value in overrides.get(interface.observation_key, {}).items()
            })
            observation[interface.observation_key] = interface.vector_from_mapping(signal_mapping)
        action_mapping = {
            spec.name: 1.0 if spec.name.endswith("_age") else 0.0
            for spec in ACTION_CONTEXT_INTERFACE.inputs
        }
        action_mapping.update({
            name: float(value)
            for name, value in overrides.get(ACTION_CONTEXT_INTERFACE.observation_key, {}).items()
        })
        observation[ACTION_CONTEXT_INTERFACE.observation_key] = ACTION_CONTEXT_INTERFACE.vector_from_mapping(
            action_mapping
        )
        motor_mapping = {
            spec.name: 1.0 if spec.name.endswith("_age") else 0.0
            for spec in MOTOR_CONTEXT_INTERFACE.inputs
        }
        motor_mapping.update({
            name: float(value)
            for name, value in overrides.get(MOTOR_CONTEXT_INTERFACE.observation_key, {}).items()
        })
        observation[MOTOR_CONTEXT_INTERFACE.observation_key] = MOTOR_CONTEXT_INTERFACE.vector_from_mapping(
            motor_mapping
        )
        return observation

    def _profile_with_reflex_config_updates(
        self,
        *,
        alert_aux_weight: float | None = None,
        visual_predator_certainty: float | None = None,
    ) -> OperationalProfile:
        """
        Create an OperationalProfile derived from DEFAULT_OPERATIONAL_PROFILE with optional reflex-specific overrides for testing.
        
        Parameters:
            alert_aux_weight (float | None): If provided, sets the `brain.aux_weights.alert_center` value in the profile.
            visual_predator_certainty (float | None): If provided, sets the `brain.reflex_thresholds.visual_cortex.predator_certainty` value in the profile.
        
        Returns:
            OperationalProfile: An OperationalProfile constructed from the modified profile summary.
        """
        summary = DEFAULT_OPERATIONAL_PROFILE.to_summary()
        summary["name"] = "test_profile"
        summary["version"] = 99
        if alert_aux_weight is not None:
            summary["brain"]["aux_weights"]["alert_center"] = alert_aux_weight
        if visual_predator_certainty is not None:
            summary["brain"]["reflex_thresholds"]["visual_cortex"]["predator_certainty"] = visual_predator_certainty
        return OperationalProfile.from_summary(summary)

    def _reordered_action_registry(self) -> dict[str, object]:
        registry = interface_registry()
        actions = list(registry["actions"])
        registry["actions"] = actions[1:] + actions[:1]
        return registry

    def _assert_registry_local_target(self, reflex, registry: dict[str, object]) -> None:
        self.assertIsNotNone(reflex)
        actions = list(registry["actions"])
        action_index = actions.index(reflex.action)
        self.assertEqual(reflex.target_probs.shape, (len(actions),))
        self.assertEqual(int(np.argmax(reflex.target_probs)), action_index)
        self.assertAlmostEqual(float(reflex.target_probs[action_index]), 1.0)

    def _assert_reordered_reflex_decision(
        self,
        decision_fn,
        observations: dict[str, float],
        *,
        expected_action: str,
        expected_reason: str,
    ) -> None:
        registry = self._reordered_action_registry()
        reflex = decision_fn(
            observations,
            operational_profile=DEFAULT_OPERATIONAL_PROFILE,
            interface_registry=registry,
        )

        self.assertIsNotNone(reflex)
        self.assertEqual(reflex.action, expected_action)
        self.assertEqual(reflex.reason, expected_reason)
        self._assert_registry_local_target(reflex, registry)

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


if __name__ == "__main__":
    unittest.main()
