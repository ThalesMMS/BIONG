from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from spider_cortex_sim.ablations import (
    BrainAblationConfig,
    MODULE_NAMES,
    MULTI_PREDATOR_SCENARIOS,
    MULTI_PREDATOR_SCENARIO_GROUPS,
    VISUAL_PREDATOR_SCENARIOS,
    OLFACTORY_PREDATOR_SCENARIOS,
    MONOLITHIC_POLICY_NAME,
    canonical_ablation_configs,
    canonical_ablation_scenario_groups,
    compare_predator_type_ablation_performance,
    canonical_ablation_variant_names,
    default_brain_config,
    resolve_ablation_scenario_group,
    resolve_ablation_configs,
    _safe_float,
    _mean,
    _scenario_success_rate,
)
from spider_cortex_sim.agent import SpiderBrain
from spider_cortex_sim.interfaces import (
    ACTION_CONTEXT_INTERFACE,
    LOCOMOTION_ACTIONS,
    MODULE_INTERFACE_BY_NAME,
    MODULE_INTERFACES,
    MOTOR_CONTEXT_INTERFACE,
)
from spider_cortex_sim.modules import CorticalModuleBank, ModuleResult
from spider_cortex_sim.nn import RecurrentProposalNetwork
from spider_cortex_sim.operational_profiles import DEFAULT_OPERATIONAL_PROFILE, OperationalProfile
from spider_cortex_sim.simulation import SpiderSimulation


def _blank_mapping(interface) -> dict[str, float]:
    """
    Map each input signal name to a default numeric value where signals ending with "_age" map to 1.0 and all others map to 0.0.
    
    Parameters:
        interface: An object with an iterable `inputs` attribute whose items have a `name` string.
    
    Returns:
        dict[str, float]: A mapping from input signal name to 1.0 for names ending with "_age", otherwise 0.0.
    """
    return {
        signal.name: (1.0 if signal.name.endswith("_age") else 0.0)
        for signal in interface.inputs
    }


def _build_observation(**overrides: dict[str, float]) -> dict[str, np.ndarray]:
    """
    Builds a complete observation dict for all module, action-context, and motor-context interfaces, applying any per-interface signal overrides.
    
    Parameters:
        overrides (dict[str, dict[str, float]]): Optional per-interface overrides keyed by each interface's
            `observation_key`. Each value is a mapping from signal name to numeric value that will replace
            the default for that signal.
    
    Returns:
        dict[str, np.ndarray]: A mapping from each interface's `observation_key` to its vectorized observation
        (NumPy array).
    """
    observation: dict[str, np.ndarray] = {}
    for interface in MODULE_INTERFACES:
        mapping = _blank_mapping(interface)
        mapping.update({name: float(value) for name, value in overrides.get(interface.observation_key, {}).items()})
        observation[interface.observation_key] = interface.vector_from_mapping(mapping)
    action_mapping = _blank_mapping(ACTION_CONTEXT_INTERFACE)
    action_mapping.update({name: float(value) for name, value in overrides.get(ACTION_CONTEXT_INTERFACE.observation_key, {}).items()})
    observation[ACTION_CONTEXT_INTERFACE.observation_key] = ACTION_CONTEXT_INTERFACE.vector_from_mapping(action_mapping)
    motor_mapping = _blank_mapping(MOTOR_CONTEXT_INTERFACE)
    motor_mapping.update({name: float(value) for name, value in overrides.get(MOTOR_CONTEXT_INTERFACE.observation_key, {}).items()})
    observation[MOTOR_CONTEXT_INTERFACE.observation_key] = MOTOR_CONTEXT_INTERFACE.vector_from_mapping(motor_mapping)
    return observation


def _profile_with_updates(**reward_updates: float) -> OperationalProfile:
    """
    Create an OperationalProfile based on the default profile with specified reward component overrides.
    
    Parameters:
        reward_updates (float): Keyword mapping of reward component names to numeric values; each provided value is coerced to float and merged into the profile's "reward" section.
    
    Returns:
        OperationalProfile: A profile derived from the default operational profile with the supplied reward updates applied. The returned profile is labeled "ablation_test_profile" and has version 21.
    """
    summary = DEFAULT_OPERATIONAL_PROFILE.to_summary()
    summary["name"] = "ablation_test_profile"
    summary["version"] = 21
    summary["reward"].update({name: float(value) for name, value in reward_updates.items()})
    return OperationalProfile.from_summary(summary)



class AblationScenarioGroupTest(unittest.TestCase):
    def test_multi_predator_group_resolves_expected_scenarios(self) -> None:
        self.assertEqual(
            resolve_ablation_scenario_group("multi_predator_ecology"),
            MULTI_PREDATOR_SCENARIOS,
        )

    def test_compare_predator_type_ablation_performance_summarizes_visual_vs_olfactory_groups(self) -> None:
        """
        Verifies that compare_predator_type_ablation_performance summarizes mean success rates for visual and olfactory predator scenario groups and produces the expected sign change in visual_minus_olfactory_success_rate between visual and sensory cortex drops.
        
        Constructs a payload with per-variant scenario success rates and deltas, calls compare_predator_type_ablation_performance, and asserts that the result is available, that the grouped mean success rates for visual and olfactory predator scenarios match expected values, and that the visual-minus-olfactory difference is negative for the visual-cortex drop and positive for the sensory-cortex drop.
        """
        payload = {
            "variants": {
                "drop_visual_cortex": {
                    "suite": {
                        "visual_olfactory_pincer": {"success_rate": 0.2},
                        "visual_hunter_open_field": {"success_rate": 0.1},
                        "olfactory_ambush": {"success_rate": 0.8},
                    }
                },
                "drop_sensory_cortex": {
                    "suite": {
                        "visual_olfactory_pincer": {"success_rate": 0.3},
                        "visual_hunter_open_field": {"success_rate": 0.9},
                        "olfactory_ambush": {"success_rate": 0.2},
                    }
                },
            },
            "deltas_vs_reference": {
                "drop_visual_cortex": {
                    "scenarios": {
                        "visual_olfactory_pincer": {"success_rate_delta": -0.3},
                        "visual_hunter_open_field": {"success_rate_delta": -0.6},
                        "olfactory_ambush": {"success_rate_delta": -0.1},
                    }
                },
                "drop_sensory_cortex": {
                    "scenarios": {
                        "visual_olfactory_pincer": {"success_rate_delta": -0.2},
                        "visual_hunter_open_field": {"success_rate_delta": -0.1},
                        "olfactory_ambush": {"success_rate_delta": -0.7},
                    }
                },
            },
        }

        result = compare_predator_type_ablation_performance(payload)

        self.assertTrue(result["available"])
        visual_drop = result["comparisons"]["drop_visual_cortex"]
        sensory_drop = result["comparisons"]["drop_sensory_cortex"]
        self.assertAlmostEqual(
            visual_drop["visual_predator_scenarios"]["mean_success_rate"],
            0.15,
        )
        self.assertAlmostEqual(
            visual_drop["olfactory_predator_scenarios"]["mean_success_rate"],
            0.5,
        )
        self.assertLess(
            visual_drop["visual_minus_olfactory_success_rate"],
            0.0,
        )
        self.assertGreater(
            sensory_drop["visual_minus_olfactory_success_rate"],
            0.0,
        )


class BrainAblationBehaviorTest(unittest.TestCase):
    def _module_result(self, module_name: str, **signals: float) -> ModuleResult:
        """
        Create a synthetic ModuleResult for testing with specified observation signal values.
        
        Parameters:
            module_name (str): Name of the module whose interface and observation vector are used.
            **signals (float): Per-signal overrides for the interface's observation mapping; keys are signal names and values coerced to float.
        
        Returns:
            ModuleResult: Active ModuleResult for the named module with zero logits and uniform action probabilities, using the interface's observation_key and the constructed observation vector.
        """
        interface = MODULE_INTERFACE_BY_NAME[module_name]
        mapping = _blank_mapping(interface)
        mapping.update({name: float(value) for name, value in signals.items()})
        observation = interface.vector_from_mapping(mapping)
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

    def test_disabled_module_is_inactive_and_zeroed(self) -> None:
        config = BrainAblationConfig(
            name="drop_alert_center",
            architecture="modular",
            module_dropout=0.0,
            enable_reflexes=True,
            enable_auxiliary_targets=True,
            disabled_modules=("alert_center",),
        )
        sim = SpiderSimulation(seed=7, max_steps=8, brain_config=config)
        observation = sim.world.reset(seed=7)

        decision = sim.brain.act(observation, sample=False)
        alert_result = next(result for result in decision.module_results if result.name == "alert_center")

        self.assertFalse(alert_result.active)
        self.assertIsNone(alert_result.reflex)
        np.testing.assert_allclose(alert_result.logits, np.zeros(len(LOCOMOTION_ACTIONS), dtype=float))
        np.testing.assert_allclose(
            alert_result.probs,
            np.full(len(LOCOMOTION_ACTIONS), 1.0 / len(LOCOMOTION_ACTIONS), dtype=float),
        )

    def test_reflex_disabled_leaves_reflex_none(self) -> None:
        """
        Ensures that turning off reflexes leaves the alert_center module's reflex unset.
        
        Constructs an observation containing predator-related signals, runs two SpiderBrain instances
        with identical seeds and dropout but with reflexes enabled vs. disabled, and asserts that
        the alert_center ModuleResult contains a reflex when enabled and `None` when reflexes are disabled.
        """
        observation = _build_observation(
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
        control = SpiderBrain(
            seed=11,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="with_reflexes",
                architecture="modular",
                module_dropout=0.0,
                enable_reflexes=True,
                enable_auxiliary_targets=True,
                disabled_modules=(),
            ),
        )
        no_reflexes = SpiderBrain(
            seed=11,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="no_reflexes",
                architecture="modular",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=True,
                disabled_modules=(),
            ),
        )

        control_result = next(result for result in control.act(observation, sample=False).module_results if result.name == "alert_center")
        ablated_result = next(result for result in no_reflexes.act(observation, sample=False).module_results if result.name == "alert_center")

        self.assertIsNotNone(control_result.reflex)
        self.assertIsNone(ablated_result.reflex)

    def test_auxiliary_targets_can_be_disabled_independently(self) -> None:
        """
        Verify that disabling auxiliary targets prevents any auxiliary-module gradients from being produced.
        
        Creates a modular SpiderBrain with reflexes enabled but auxiliary targets disabled, constructs an alert_center ModuleResult with a forced reflex decision, and asserts that _auxiliary_module_gradients returns an empty dict.
        """
        brain = SpiderBrain(
            seed=3,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="no_aux",
                architecture="modular",
                module_dropout=0.0,
                enable_reflexes=True,
                enable_auxiliary_targets=False,
                disabled_modules=(),
            ),
        )
        result = self._module_result(
            "alert_center",
            predator_visible=1.0,
            predator_certainty=0.8,
            predator_dx=1.0,
            predator_dy=0.0,
        )
        result.reflex = brain._module_reflex_decision(result)

        self.assertEqual(brain._auxiliary_module_gradients([result]), {})

    def test_fixed_arbitration_baseline_matches_fixed_formula_behavior(self) -> None:
        configs = canonical_ablation_configs(module_dropout=0.0)
        canonical_fixed = configs["fixed_arbitration_baseline"]
        manual_fixed = BrainAblationConfig(
            name="manual_fixed_arbitration",
            module_dropout=0.0,
            use_learned_arbitration=False,
        )
        observation = _build_observation(
            action_context={
                "predator_visible": 1.0,
                "predator_certainty": 0.9,
                "hunger": 0.95,
                "day": 1.0,
            },
            alert={
                "predator_visible": 1.0,
                "predator_certainty": 0.9,
                "predator_motion_salience": 0.8,
                "predator_smell_strength": 0.7,
            },
            hunger={
                "hunger": 0.95,
                "food_visible": 1.0,
                "food_certainty": 0.9,
                "food_smell_strength": 0.8,
                "food_memory_age": 0.0,
            },
        )
        canonical_brain = SpiderBrain(seed=19, module_dropout=0.0, config=canonical_fixed)
        manual_brain = SpiderBrain(seed=19, module_dropout=0.0, config=manual_fixed)

        canonical_step = canonical_brain.act(observation, sample=False)
        manual_step = manual_brain.act(observation, sample=False)

        self.assertFalse(canonical_step.arbitration_decision.learned_adjustment)
        self.assertFalse(manual_step.arbitration_decision.learned_adjustment)
        self.assertEqual(
            canonical_step.arbitration_decision.winning_valence,
            manual_step.arbitration_decision.winning_valence,
        )
        self.assertEqual(
            canonical_step.arbitration_decision.module_gates,
            manual_step.arbitration_decision.module_gates,
        )
        np.testing.assert_allclose(canonical_step.action_center_policy, manual_step.action_center_policy)


class BrainAblationConfigValidationTest(unittest.TestCase):
    """Tests for BrainAblationConfig construction and validation."""

    def test_default_architecture_is_modular(self) -> None:
        config = BrainAblationConfig(name="test")
        self.assertEqual(config.architecture, "modular")

    def test_default_module_dropout_is_float(self) -> None:
        config = BrainAblationConfig(name="test", module_dropout=0.1)
        self.assertIsInstance(config.module_dropout, float)
        self.assertAlmostEqual(config.module_dropout, 0.1)

    def test_default_enable_reflexes_true(self) -> None:
        config = BrainAblationConfig(name="test")
        self.assertTrue(config.enable_reflexes)

    def test_default_enable_auxiliary_targets_true(self) -> None:
        config = BrainAblationConfig(name="test")
        self.assertTrue(config.enable_auxiliary_targets)

    def test_default_use_learned_arbitration_true(self) -> None:
        config = BrainAblationConfig(name="test")
        self.assertTrue(config.use_learned_arbitration)

    def test_manual_arbitration_scaffolding_defaults_disabled(self) -> None:
        config = BrainAblationConfig(name="test")
        self.assertFalse(config.enable_deterministic_guards)
        self.assertFalse(config.enable_food_direction_bias)

    def test_arbitration_prior_defaults(self) -> None:
        config = BrainAblationConfig(name="test")
        self.assertAlmostEqual(config.warm_start_scale, 1.0)
        self.assertEqual(config.gate_adjustment_bounds, (0.5, 1.5))

    def test_default_credit_strategy_is_broadcast(self) -> None:
        config = BrainAblationConfig()
        self.assertEqual(config.credit_strategy, "broadcast")
        self.assertFalse(config.uses_local_credit_only)
        self.assertFalse(config.uses_counterfactual_credit)

    def test_credit_strategy_accepts_local_only(self) -> None:
        config = BrainAblationConfig(name="test", credit_strategy="local_only")
        self.assertTrue(config.uses_local_credit_only)
        self.assertFalse(config.uses_counterfactual_credit)

    def test_credit_strategy_accepts_counterfactual(self) -> None:
        config = BrainAblationConfig(name="test", credit_strategy="counterfactual")
        self.assertFalse(config.uses_local_credit_only)
        self.assertTrue(config.uses_counterfactual_credit)

    def test_invalid_credit_strategy_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            BrainAblationConfig(name="test", credit_strategy="unknown")

    def test_uses_local_credit_only_is_true_only_for_local_only_strategy(self) -> None:
        cases = {
            "broadcast": False,
            "local_only": True,
            "counterfactual": False,
        }
        for strategy, expected in cases.items():
            with self.subTest(strategy=strategy):
                config = BrainAblationConfig(name="test", credit_strategy=strategy)
                self.assertIs(config.uses_local_credit_only, expected)

    def test_uses_counterfactual_credit_is_true_only_for_counterfactual_strategy(self) -> None:
        cases = {
            "broadcast": False,
            "local_only": False,
            "counterfactual": True,
        }
        for strategy, expected in cases.items():
            with self.subTest(strategy=strategy):
                config = BrainAblationConfig(name="test", credit_strategy=strategy)
                self.assertIs(config.uses_counterfactual_credit, expected)

    def test_use_learned_arbitration_is_coerced_to_bool(self) -> None:
        fixed = BrainAblationConfig(name="fixed", use_learned_arbitration=0)
        learned = BrainAblationConfig(name="learned", use_learned_arbitration=1)
        self.assertIs(fixed.use_learned_arbitration, False)
        self.assertIs(learned.use_learned_arbitration, True)

    def test_manual_scaffolding_flags_are_coerced_to_bool(self) -> None:
        config = BrainAblationConfig(
            name="test",
            enable_deterministic_guards=1,
            enable_food_direction_bias=0,
        )
        self.assertIs(config.enable_deterministic_guards, True)
        self.assertIs(config.enable_food_direction_bias, False)

    def test_warm_start_scale_accepts_bounds(self) -> None:
        no_start = BrainAblationConfig(name="no_start", warm_start_scale=0.0)
        full_start = BrainAblationConfig(name="full_start", warm_start_scale=1.0)
        self.assertAlmostEqual(no_start.warm_start_scale, 0.0)
        self.assertAlmostEqual(full_start.warm_start_scale, 1.0)

    def test_warm_start_scale_rejects_out_of_range_values(self) -> None:
        for value in (-0.01, 1.01):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "warm_start_scale"):
                    BrainAblationConfig(name="test", warm_start_scale=value)

    def test_warm_start_scale_rejects_non_finite_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "warm_start_scale"):
            BrainAblationConfig(name="test", warm_start_scale=float("nan"))

    def test_gate_adjustment_bounds_are_coerced_to_float_tuple(self) -> None:
        config = BrainAblationConfig(
            name="test",
            gate_adjustment_bounds=(0, 2),
        )
        self.assertEqual(config.gate_adjustment_bounds, (0.0, 2.0))
        self.assertIsInstance(config.gate_adjustment_bounds[0], float)

    def test_gate_adjustment_bounds_reject_unordered_values(self) -> None:
        for bounds in ((1.0, 1.0), (1.5, 0.5)):
            with self.subTest(bounds=bounds):
                with self.assertRaisesRegex(ValueError, "gate_adjustment_bounds"):
                    BrainAblationConfig(name="test", gate_adjustment_bounds=bounds)

    def test_gate_adjustment_bounds_reject_wrong_length(self) -> None:
        with self.assertRaisesRegex(ValueError, "gate_adjustment_bounds"):
            BrainAblationConfig(name="test", gate_adjustment_bounds=(0.5,))

    def test_gate_adjustment_bounds_reject_non_finite_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "gate_adjustment_bounds"):
            BrainAblationConfig(
                name="test",
                gate_adjustment_bounds=(0.5, float("inf")),
            )

    def test_default_disabled_modules_empty(self) -> None:
        config = BrainAblationConfig(name="test")
        self.assertEqual(config.disabled_modules, ())

    def test_recurrent_modules_default_empty(self) -> None:
        config = BrainAblationConfig(name="test")
        self.assertEqual(config.recurrent_modules, ())
        self.assertFalse(config.is_recurrent)

    def test_invalid_architecture_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            BrainAblationConfig(name="test", architecture="invalid_arch")

    def test_invalid_module_name_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            BrainAblationConfig(
                name="test",
                architecture="modular",
                disabled_modules=("nonexistent_module",),
            )

    def test_monolithic_with_disabled_modules_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            BrainAblationConfig(
                name="test",
                architecture="monolithic",
                disabled_modules=("alert_center",),
            )

    def test_disabled_modules_are_sorted_and_deduplicated(self) -> None:
        config = BrainAblationConfig(
            name="test",
            architecture="modular",
            disabled_modules=("sleep_center", "alert_center", "alert_center"),
        )
        self.assertEqual(config.disabled_modules, ("alert_center", "sleep_center"))

    def test_recurrent_modules_are_deduplicated_preserving_order(self) -> None:
        config = BrainAblationConfig(
            name="test",
            architecture="modular",
            recurrent_modules=("sleep_center", "alert_center", "sleep_center"),
        )
        self.assertEqual(config.recurrent_modules, ("sleep_center", "alert_center"))

    def test_recurrent_modules_validation(self) -> None:
        with self.assertRaises(ValueError):
            BrainAblationConfig(
                name="test",
                architecture="modular",
                recurrent_modules=("nonexistent_module",),
            )

    def test_recurrent_modules_requires_modular_architecture(self) -> None:
        with self.assertRaises(ValueError):
            BrainAblationConfig(
                name="test",
                architecture="monolithic",
                recurrent_modules=("alert_center",),
            )

    def test_recurrent_modules_items_are_coerced_to_str(self) -> None:
        class StrLike(str):
            pass

        config = BrainAblationConfig(
            name="test",
            architecture="modular",
            recurrent_modules=(StrLike("alert_center"),),
        )
        for item in config.recurrent_modules:
            self.assertIs(type(item), str)

    def test_is_recurrent_false_for_explicitly_empty_recurrent_modules(self) -> None:
        config = BrainAblationConfig(
            name="test",
            architecture="modular",
            recurrent_modules=(),
        )
        self.assertFalse(config.is_recurrent)

    def test_all_module_names_valid_as_recurrent_modules(self) -> None:
        # All canonical module names must be accepted without error
        config = BrainAblationConfig(
            name="all_recurrent",
            architecture="modular",
            recurrent_modules=MODULE_NAMES,
        )
        self.assertEqual(set(config.recurrent_modules), set(MODULE_NAMES))
        self.assertTrue(config.is_recurrent)

    def test_is_modular_property_true_for_modular(self) -> None:
        config = BrainAblationConfig(name="test", architecture="modular")
        self.assertTrue(config.is_modular)
        self.assertFalse(config.is_monolithic)

    def test_is_monolithic_property_true_for_monolithic(self) -> None:
        config = BrainAblationConfig(name="test", architecture="monolithic")
        self.assertTrue(config.is_monolithic)
        self.assertFalse(config.is_modular)

    def test_is_recurrent_property_true_when_recurrent_modules_present(self) -> None:
        config = BrainAblationConfig(
            name="test",
            recurrent_modules=("alert_center",),
        )
        self.assertTrue(config.is_recurrent)

    def test_to_summary_returns_correct_keys(self) -> None:
        config = BrainAblationConfig(
            name="drop_visual_cortex",
            architecture="modular",
            module_dropout=0.03,
            enable_reflexes=True,
            enable_auxiliary_targets=False,
            disabled_modules=("visual_cortex",),
        )
        summary = config.to_summary()
        self.assertEqual(summary["name"], "drop_visual_cortex")
        self.assertEqual(summary["architecture"], "modular")
        self.assertAlmostEqual(summary["module_dropout"], 0.03)
        self.assertTrue(summary["enable_reflexes"])
        self.assertFalse(summary["enable_auxiliary_targets"])
        self.assertTrue(summary["use_learned_arbitration"])
        self.assertEqual(summary["credit_strategy"], "broadcast")
        self.assertEqual(summary["disabled_modules"], ["visual_cortex"])
        self.assertEqual(summary["recurrent_modules"], [])
        self.assertFalse(summary["is_recurrent"])

    def test_to_summary_disabled_modules_is_list(self) -> None:
        config = BrainAblationConfig(name="test", disabled_modules=("hunger_center",))
        summary = config.to_summary()
        self.assertIsInstance(summary["disabled_modules"], list)

    def test_to_summary_recurrent_modules_is_list(self) -> None:
        config = BrainAblationConfig(
            name="test",
            recurrent_modules=("alert_center", "sleep_center"),
        )
        summary = config.to_summary()
        self.assertIsInstance(summary["recurrent_modules"], list)
        self.assertEqual(summary["recurrent_modules"], ["alert_center", "sleep_center"])
        self.assertTrue(summary["is_recurrent"])

    def test_to_summary_canonicalizes_recurrent_module_order(self) -> None:
        config = BrainAblationConfig(
            name="test",
            recurrent_modules=("sleep_center", "alert_center"),
        )
        summary = config.to_summary()
        self.assertEqual(summary["recurrent_modules"], ["alert_center", "sleep_center"])

    def test_frozen_dataclass_is_immutable(self) -> None:
        config = BrainAblationConfig(name="test")
        with self.assertRaises((AttributeError, TypeError)):
            config.name = "other"  # type: ignore[misc]

    def test_module_dropout_coerced_to_float(self) -> None:
        config = BrainAblationConfig(name="test", module_dropout=0)
        self.assertIsInstance(config.module_dropout, float)
        self.assertEqual(config.module_dropout, 0.0)

    def test_reflex_scale_default_is_one(self) -> None:
        config = BrainAblationConfig(name="test")
        self.assertAlmostEqual(config.reflex_scale, 1.0)

    def test_reflex_scale_coerced_to_float(self) -> None:
        config = BrainAblationConfig(name="test", reflex_scale=0)
        self.assertIsInstance(config.reflex_scale, float)
        self.assertEqual(config.reflex_scale, 0.0)

    def test_reflex_scale_negative_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            BrainAblationConfig(name="test", reflex_scale=-0.1)

    def test_reflex_scale_non_finite_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            BrainAblationConfig(name="test", reflex_scale=float("nan"))

    def test_module_dropout_non_finite_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            BrainAblationConfig(name="test", module_dropout=float("inf"))

    def test_reflex_scale_zero_accepted(self) -> None:
        config = BrainAblationConfig(name="test", reflex_scale=0.0)
        self.assertAlmostEqual(config.reflex_scale, 0.0)

    def test_reflex_scale_large_value_accepted(self) -> None:
        config = BrainAblationConfig(name="test", reflex_scale=2.5)
        self.assertAlmostEqual(config.reflex_scale, 2.5)

    def test_module_reflex_scales_default_is_empty(self) -> None:
        config = BrainAblationConfig(name="test")
        self.assertEqual(config.module_reflex_scales, {})

    def test_module_reflex_scales_valid_module_accepted(self) -> None:
        config = BrainAblationConfig(
            name="test",
            module_reflex_scales={"alert_center": 0.5},
        )
        self.assertAlmostEqual(config.module_reflex_scales["alert_center"], 0.5)

    def test_module_reflex_scales_invalid_module_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            BrainAblationConfig(
                name="test",
                module_reflex_scales={"nonexistent_module": 0.5},
            )

    def test_module_reflex_scales_reject_non_reflex_module(self) -> None:
        with self.assertRaises(ValueError):
            BrainAblationConfig(
                name="test",
                module_reflex_scales={"motor_cortex_context": 0.5},
            )

    def test_module_reflex_scales_negative_value_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            BrainAblationConfig(
                name="test",
                module_reflex_scales={"alert_center": -0.1},
            )

    def test_module_reflex_scales_non_finite_value_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            BrainAblationConfig(
                name="test",
                module_reflex_scales={"alert_center": float("nan")},
            )

    def test_module_reflex_scales_zero_accepted(self) -> None:
        config = BrainAblationConfig(
            name="test",
            module_reflex_scales={"alert_center": 0.0},
        )
        self.assertAlmostEqual(config.module_reflex_scales["alert_center"], 0.0)

    def test_monolithic_with_module_reflex_scales_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            BrainAblationConfig(
                name="test",
                architecture="monolithic",
                module_reflex_scales={"alert_center": 0.5},
            )

    def test_module_reflex_scales_are_sorted_by_key(self) -> None:
        config = BrainAblationConfig(
            name="test",
            module_reflex_scales={
                "visual_cortex": 0.8,
                "alert_center": 0.5,
                "hunger_center": 0.3,
            },
        )
        keys = list(config.module_reflex_scales.keys())
        self.assertEqual(keys, sorted(keys))

    def test_to_summary_includes_reflex_scale(self) -> None:
        config = BrainAblationConfig(name="test", reflex_scale=0.75)
        summary = config.to_summary()
        self.assertIn("reflex_scale", summary)
        self.assertAlmostEqual(summary["reflex_scale"], 0.75)

    def test_to_summary_includes_module_reflex_scales(self) -> None:
        config = BrainAblationConfig(
            name="test",
            module_reflex_scales={"alert_center": 0.5},
        )
        summary = config.to_summary()
        self.assertIn("module_reflex_scales", summary)
        self.assertIsInstance(summary["module_reflex_scales"], dict)
        self.assertAlmostEqual(summary["module_reflex_scales"]["alert_center"], 0.5)

    def test_to_summary_module_reflex_scales_is_plain_dict(self) -> None:
        config = BrainAblationConfig(name="test")
        summary = config.to_summary()
        self.assertIsInstance(summary["module_reflex_scales"], dict)

    def test_to_summary_includes_use_learned_arbitration(self) -> None:
        config = BrainAblationConfig(
            name="fixed_arbitration_baseline",
            use_learned_arbitration=False,
        )
        summary = config.to_summary()
        self.assertIn("use_learned_arbitration", summary)
        self.assertFalse(summary["use_learned_arbitration"])

    def test_to_summary_includes_arbitration_scaffolding_controls(self) -> None:
        config = BrainAblationConfig(
            name="test",
            enable_deterministic_guards=True,
            enable_food_direction_bias=True,
            warm_start_scale=0.25,
            gate_adjustment_bounds=(0.25, 1.75),
        )
        summary = config.to_summary()
        self.assertTrue(summary["enable_deterministic_guards"])
        self.assertTrue(summary["enable_food_direction_bias"])
        self.assertAlmostEqual(summary["warm_start_scale"], 0.25)
        self.assertEqual(summary["gate_adjustment_bounds"], [0.25, 1.75])

    def test_module_reflex_scales_values_coerced_to_float(self) -> None:
        config = BrainAblationConfig(
            name="test",
            module_reflex_scales={"alert_center": 1},
        )
        self.assertIsInstance(config.module_reflex_scales["alert_center"], float)


class DefaultBrainConfigTest(unittest.TestCase):
    """Tests for default_brain_config factory."""

    def test_default_brain_config_name(self) -> None:
        config = default_brain_config()
        self.assertEqual(config.name, "modular_full")

    def test_default_brain_config_architecture(self) -> None:
        config = default_brain_config()
        self.assertEqual(config.architecture, "modular")

    def test_default_brain_config_custom_dropout(self) -> None:
        config = default_brain_config(module_dropout=0.10)
        self.assertAlmostEqual(config.module_dropout, 0.10)

    def test_default_brain_config_reflexes_and_aux_enabled(self) -> None:
        config = default_brain_config()
        self.assertTrue(config.enable_reflexes)
        self.assertTrue(config.enable_auxiliary_targets)

    def test_default_brain_config_manual_arbitration_scaffolding_disabled(self) -> None:
        config = default_brain_config()
        self.assertFalse(config.enable_deterministic_guards)
        self.assertFalse(config.enable_food_direction_bias)

    def test_default_brain_config_no_disabled_modules(self) -> None:
        config = default_brain_config()
        self.assertEqual(config.disabled_modules, ())

    def test_default_brain_config_reflex_scale_is_one(self) -> None:
        config = default_brain_config()
        self.assertAlmostEqual(config.reflex_scale, 1.0)

    def test_default_brain_config_module_reflex_scales_is_empty(self) -> None:
        config = default_brain_config()
        self.assertEqual(config.module_reflex_scales, {})


class CanonicalAblationConfigsTest(unittest.TestCase):
    """Tests for canonical_ablation_configs and canonical_ablation_variant_names."""

    def test_contains_modular_full(self) -> None:
        configs = canonical_ablation_configs()
        self.assertIn("modular_full", configs)

    def test_contains_no_module_dropout(self) -> None:
        configs = canonical_ablation_configs()
        no_drop = configs["no_module_dropout"]
        self.assertEqual(no_drop.module_dropout, 0.0)
        self.assertTrue(no_drop.enable_reflexes)

    def test_contains_no_module_reflexes(self) -> None:
        configs = canonical_ablation_configs()
        no_reflex = configs["no_module_reflexes"]
        self.assertFalse(no_reflex.enable_reflexes)
        self.assertFalse(no_reflex.enable_auxiliary_targets)

    def test_contains_modular_recurrent(self) -> None:
        configs = canonical_ablation_configs()
        recurrent = configs["modular_recurrent"]
        self.assertTrue(recurrent.is_modular)
        self.assertTrue(recurrent.is_recurrent)
        self.assertEqual(
            recurrent.recurrent_modules,
            ("alert_center", "sleep_center", "hunger_center"),
        )

    def test_contains_modular_recurrent_all(self) -> None:
        configs = canonical_ablation_configs()
        recurrent_all = configs["modular_recurrent_all"]
        self.assertTrue(recurrent_all.is_modular)
        self.assertTrue(recurrent_all.is_recurrent)
        self.assertEqual(recurrent_all.recurrent_modules, MODULE_NAMES)

    def test_contains_local_credit_only(self) -> None:
        configs = canonical_ablation_configs()
        local_credit = configs["local_credit_only"]
        self.assertTrue(local_credit.is_modular)
        self.assertTrue(local_credit.enable_reflexes)
        self.assertTrue(local_credit.enable_auxiliary_targets)
        self.assertTrue(local_credit.uses_local_credit_only)

    def test_contains_counterfactual_credit(self) -> None:
        """
        Verifies the canonical ablation registry contains a `counterfactual_credit` variant with the expected modular settings and credit-strategy flags.
        
        Asserts the variant is modular, keeps reflexes and auxiliary targets enabled, uses learned arbitration, matches `modular_full` for dropout, disabled modules, reflex scale and per-module reflex scales, advertises counterfactual credit usage, and does not use the local-only credit strategy.
        """
        configs = canonical_ablation_configs()
        counterfactual = configs["counterfactual_credit"]
        modular_full = configs["modular_full"]
        self.assertTrue(counterfactual.is_modular)
        self.assertTrue(counterfactual.enable_reflexes)
        self.assertTrue(counterfactual.enable_auxiliary_targets)
        self.assertTrue(counterfactual.use_learned_arbitration)
        self.assertEqual(counterfactual.module_dropout, modular_full.module_dropout)
        self.assertEqual(counterfactual.disabled_modules, modular_full.disabled_modules)
        self.assertEqual(counterfactual.recurrent_modules, modular_full.recurrent_modules)
        self.assertEqual(counterfactual.reflex_scale, modular_full.reflex_scale)
        self.assertEqual(
            counterfactual.module_reflex_scales,
            modular_full.module_reflex_scales,
        )
        self.assertTrue(counterfactual.uses_counterfactual_credit)
        self.assertFalse(counterfactual.uses_local_credit_only)

    def test_contains_constrained_arbitration(self) -> None:
        configs = canonical_ablation_configs()
        constrained = configs["constrained_arbitration"]
        self.assertTrue(constrained.is_modular)
        self.assertTrue(constrained.use_learned_arbitration)
        self.assertTrue(constrained.enable_deterministic_guards)
        self.assertTrue(constrained.enable_food_direction_bias)
        self.assertAlmostEqual(constrained.warm_start_scale, 1.0)
        self.assertEqual(constrained.gate_adjustment_bounds, (0.5, 1.5))

    def test_contains_weaker_prior_arbitration(self) -> None:
        configs = canonical_ablation_configs()
        weaker = configs["weaker_prior_arbitration"]
        self.assertTrue(weaker.is_modular)
        self.assertTrue(weaker.use_learned_arbitration)
        self.assertFalse(weaker.enable_deterministic_guards)
        self.assertFalse(weaker.enable_food_direction_bias)
        self.assertAlmostEqual(weaker.warm_start_scale, 0.5)
        self.assertEqual(weaker.gate_adjustment_bounds, (0.5, 1.5))

    def test_contains_minimal_arbitration(self) -> None:
        configs = canonical_ablation_configs()
        minimal = configs["minimal_arbitration"]
        self.assertTrue(minimal.is_modular)
        self.assertTrue(minimal.use_learned_arbitration)
        self.assertFalse(minimal.enable_deterministic_guards)
        self.assertFalse(minimal.enable_food_direction_bias)
        self.assertAlmostEqual(minimal.warm_start_scale, 0.0)
        self.assertEqual(minimal.gate_adjustment_bounds, (0.1, 2.0))

    def test_arbitration_prior_variants_share_modular_baseline_settings(self) -> None:
        """
        Verify that arbitration-prior variants inherit the modular baseline's settings for module_dropout, disabled_modules, recurrent_modules, reflex_scale, and module_reflex_scales.
        """
        configs = canonical_ablation_configs()
        modular_full = configs["modular_full"]
        for name in (
            "constrained_arbitration",
            "weaker_prior_arbitration",
            "minimal_arbitration",
        ):
            with self.subTest(name=name):
                variant = configs[name]
                self.assertEqual(variant.module_dropout, modular_full.module_dropout)
                self.assertEqual(variant.disabled_modules, modular_full.disabled_modules)
                self.assertEqual(variant.recurrent_modules, modular_full.recurrent_modules)
                self.assertEqual(variant.reflex_scale, modular_full.reflex_scale)
                self.assertEqual(
                    variant.module_reflex_scales,
                    modular_full.module_reflex_scales,
                )

    def test_arbitration_prior_variant_order_is_ladder(self) -> None:
        names = canonical_ablation_variant_names()
        self.assertLess(
            names.index("counterfactual_credit"),
            names.index("constrained_arbitration"),
        )
        self.assertLess(
            names.index("constrained_arbitration"),
            names.index("weaker_prior_arbitration"),
        )
        self.assertLess(
            names.index("weaker_prior_arbitration"),
            names.index("minimal_arbitration"),
        )
        self.assertLess(
            names.index("minimal_arbitration"),
            names.index("fixed_arbitration_baseline"),
        )

    def test_contains_fixed_arbitration_baseline(self) -> None:
        configs = canonical_ablation_configs()
        fixed = configs["fixed_arbitration_baseline"]
        self.assertTrue(fixed.is_modular)
        self.assertFalse(fixed.use_learned_arbitration)

    def test_contains_learned_arbitration_no_regularization(self) -> None:
        configs = canonical_ablation_configs()
        learned_no_reg = configs["learned_arbitration_no_regularization"]
        self.assertTrue(learned_no_reg.is_modular)
        self.assertTrue(learned_no_reg.use_learned_arbitration)

    def test_learned_arbitration_no_regularization_disables_brain_regularizers(self) -> None:
        config = canonical_ablation_configs()["learned_arbitration_no_regularization"]
        brain = SpiderBrain(seed=5, module_dropout=0.0, config=config)
        self.assertEqual(brain.arbitration_regularization_weight, 0.0)
        self.assertEqual(brain.arbitration_valence_regularization_weight, 0.0)

    def test_contains_monolithic_policy(self) -> None:
        configs = canonical_ablation_configs()
        mono = configs["monolithic_policy"]
        self.assertEqual(mono.architecture, "monolithic")
        self.assertFalse(mono.enable_reflexes)

    def test_contains_drop_variants_for_all_modules(self) -> None:
        configs = canonical_ablation_configs()
        for module_name in MODULE_NAMES:
            key = f"drop_{module_name}"
            self.assertIn(key, configs, f"Missing drop variant: {key}")
            self.assertIn(module_name, configs[key].disabled_modules)

    def test_custom_dropout_propagates_to_modular_full(self) -> None:
        configs = canonical_ablation_configs(module_dropout=0.08)
        self.assertAlmostEqual(configs["modular_full"].module_dropout, 0.08)

    def test_variant_names_match_config_keys(self) -> None:
        configs = canonical_ablation_configs()
        names = canonical_ablation_variant_names()
        self.assertEqual(names, tuple(configs.keys()))

    def test_variant_names_starts_with_modular_full(self) -> None:
        names = canonical_ablation_variant_names()
        self.assertEqual(names[0], "modular_full")

    def test_variant_names_include_monolithic_policy(self) -> None:
        names = canonical_ablation_variant_names()
        self.assertIn("monolithic_policy", names)

    def test_variant_names_count(self) -> None:
        """
        Verify the canonical ablation variant name registry contains the expected number of variants.
        
        The test asserts that canonical_ablation_variant_names() returns a list whose length equals:
        15 (core variants including recurrent, credit, reflex-scale, and arbitration variants)
        + one drop variant per module (len(MODULE_NAMES)) + 1 (monolithic_policy).
        """
        names = canonical_ablation_variant_names()
        # modular_full + no_module_dropout + no_module_reflexes
        # + modular_recurrent + modular_recurrent_all
        # + local_credit_only + counterfactual_credit
        # + constrained_arbitration + weaker_prior_arbitration + minimal_arbitration
        # + fixed_arbitration_baseline + learned_arbitration_no_regularization
        # + reflex_scale_0_25/_0_50/_0_75 + drop_ variants + monolithic_policy
        expected_count = 15 + len(MODULE_NAMES) + 1
        self.assertEqual(len(names), expected_count)

    def test_contains_reflex_scale_0_25(self) -> None:
        configs = canonical_ablation_configs()
        self.assertIn("reflex_scale_0_25", configs)
        variant = configs["reflex_scale_0_25"]
        self.assertAlmostEqual(variant.reflex_scale, 0.25)
        self.assertEqual(variant.architecture, "modular")
        self.assertTrue(variant.enable_reflexes)
        self.assertTrue(variant.enable_auxiliary_targets)

    def test_contains_reflex_scale_0_50(self) -> None:
        configs = canonical_ablation_configs()
        self.assertIn("reflex_scale_0_50", configs)
        variant = configs["reflex_scale_0_50"]
        self.assertAlmostEqual(variant.reflex_scale, 0.50)
        self.assertEqual(variant.architecture, "modular")
        self.assertTrue(variant.enable_reflexes)

    def test_contains_reflex_scale_0_75(self) -> None:
        configs = canonical_ablation_configs()
        self.assertIn("reflex_scale_0_75", configs)
        variant = configs["reflex_scale_0_75"]
        self.assertAlmostEqual(variant.reflex_scale, 0.75)
        self.assertEqual(variant.architecture, "modular")
        self.assertTrue(variant.enable_reflexes)

    def test_reflex_scale_variants_have_no_disabled_modules(self) -> None:
        configs = canonical_ablation_configs()
        for name in ("reflex_scale_0_25", "reflex_scale_0_50", "reflex_scale_0_75"):
            self.assertEqual(configs[name].disabled_modules, ())

    def test_reflex_scale_variants_have_empty_module_reflex_scales(self) -> None:
        configs = canonical_ablation_configs()
        for name in ("reflex_scale_0_25", "reflex_scale_0_50", "reflex_scale_0_75"):
            self.assertEqual(configs[name].module_reflex_scales, {})

    def test_no_module_reflexes_has_reflex_scale_zero(self) -> None:
        configs = canonical_ablation_configs()
        variant = configs["no_module_reflexes"]
        self.assertAlmostEqual(variant.reflex_scale, 0.0)

    def test_monolithic_policy_has_reflex_scale_zero(self) -> None:
        configs = canonical_ablation_configs()
        variant = configs["monolithic_policy"]
        self.assertAlmostEqual(variant.reflex_scale, 0.0)

    def test_modular_full_has_reflex_scale_one(self) -> None:
        configs = canonical_ablation_configs()
        variant = configs["modular_full"]
        self.assertAlmostEqual(variant.reflex_scale, 1.0)

    def test_drop_variants_have_reflex_scale_one(self) -> None:
        configs = canonical_ablation_configs()
        for module_name in MODULE_NAMES:
            variant = configs[f"drop_{module_name}"]
            self.assertAlmostEqual(variant.reflex_scale, 1.0)

    def test_reflex_scale_variants_custom_dropout_propagates(self) -> None:
        """
        Ensures a custom module_dropout value is propagated to the canonical reflex-scale ablation variants.
        
        Verifies that each of the "reflex_scale_0_25", "reflex_scale_0_50", and "reflex_scale_0_75" configs returned by canonical_ablation_configs(module_dropout=0.09) has module_dropout equal to 0.09.
        """
        configs = canonical_ablation_configs(module_dropout=0.09)
        for name in ("reflex_scale_0_25", "reflex_scale_0_50", "reflex_scale_0_75"):
            self.assertAlmostEqual(configs[name].module_dropout, 0.09)

    def test_arbitration_prior_variants_custom_dropout_propagates(self) -> None:
        configs = canonical_ablation_configs(module_dropout=0.09)
        for name in (
            "constrained_arbitration",
            "weaker_prior_arbitration",
            "minimal_arbitration",
        ):
            self.assertAlmostEqual(configs[name].module_dropout, 0.09)


class ResolveAblationConfigsTest(unittest.TestCase):
    """Tests for resolve_ablation_configs."""

    def test_none_returns_all_canonical_variants(self) -> None:
        configs = resolve_ablation_configs(None)
        names = [c.name for c in configs]
        self.assertEqual(names, list(canonical_ablation_variant_names()))

    def test_specific_names_returns_correct_order(self) -> None:
        configs = resolve_ablation_configs(["monolithic_policy", "modular_full"])
        self.assertEqual(configs[0].name, "monolithic_policy")
        self.assertEqual(configs[1].name, "modular_full")

    def test_single_name_returns_single_config(self) -> None:
        configs = resolve_ablation_configs(["no_module_dropout"])
        self.assertEqual(len(configs), 1)
        self.assertEqual(configs[0].name, "no_module_dropout")

    def test_new_arbitration_variants_resolve(self) -> None:
        configs = resolve_ablation_configs([
            "constrained_arbitration",
            "weaker_prior_arbitration",
            "minimal_arbitration",
            "fixed_arbitration_baseline",
            "learned_arbitration_no_regularization",
        ])
        self.assertEqual([c.name for c in configs], [
            "constrained_arbitration",
            "weaker_prior_arbitration",
            "minimal_arbitration",
            "fixed_arbitration_baseline",
            "learned_arbitration_no_regularization",
        ])
        self.assertTrue(configs[0].enable_deterministic_guards)
        self.assertAlmostEqual(configs[1].warm_start_scale, 0.5)
        self.assertEqual(configs[2].gate_adjustment_bounds, (0.1, 2.0))
        self.assertFalse(configs[3].use_learned_arbitration)
        self.assertTrue(configs[4].use_learned_arbitration)

    def test_new_recurrent_variants_resolve(self) -> None:
        configs = resolve_ablation_configs([
            "modular_recurrent",
            "modular_recurrent_all",
        ])
        self.assertEqual(
            configs[0].recurrent_modules,
            ("alert_center", "sleep_center", "hunger_center"),
        )
        self.assertEqual(configs[1].recurrent_modules, MODULE_NAMES)

    def test_counterfactual_credit_resolves(self) -> None:
        configs = resolve_ablation_configs(["counterfactual_credit"])
        self.assertEqual(configs[0].name, "counterfactual_credit")
        self.assertTrue(configs[0].uses_counterfactual_credit)

    def test_unknown_name_raises_key_error(self) -> None:
        with self.assertRaises(KeyError):
            resolve_ablation_configs(["nonexistent_variant"])

    def test_module_dropout_propagates(self) -> None:
        configs = resolve_ablation_configs(["modular_full"], module_dropout=0.07)
        self.assertAlmostEqual(configs[0].module_dropout, 0.07)

    def test_empty_list_preserves_empty_selection(self) -> None:
        configs = resolve_ablation_configs([])
        self.assertEqual(configs, [])


class CorticalModuleBankDisabledModulesTest(unittest.TestCase):
    """Tests for CorticalModuleBank.disabled_modules feature (new in this PR)."""

    def _make_bank(self, disabled: tuple[str, ...] = ()) -> CorticalModuleBank:
        """
        Constructs a CorticalModuleBank with a fixed RNG, zero module dropout, and the specified disabled modules.
        
        Parameters:
            disabled (tuple[str, ...]): Names of modules to disable in the returned bank.
        
        Returns:
            CorticalModuleBank: Bank with action_dim equal to the number of locomotion actions, RNG seeded with 42, module_dropout set to 0.0, and disabled_modules set to `disabled`.
        """
        rng = np.random.default_rng(42)
        return CorticalModuleBank(
            action_dim=len(LOCOMOTION_ACTIONS),
            rng=rng,
            module_dropout=0.0,
            disabled_modules=disabled,
        )

    def _blank_observation(self) -> dict:
        """
        Constructs a blank observation mapping for the brain's module interfaces.
        
        Returns:
            dict: Mapping from each interface's `observation_key` to a NumPy float array of zeros with length equal to that interface's `input_dim`.
        """
        obs = {}
        for spec in self._make_bank().specs:
            obs[spec.observation_key] = np.zeros(spec.input_dim, dtype=float)
        return obs

    def test_disabled_modules_stored_as_set(self) -> None:
        bank = self._make_bank(("alert_center",))
        self.assertIsInstance(bank.disabled_modules, set)

    def test_unknown_disabled_module_raises_value_error(self) -> None:
        rng = np.random.default_rng(42)
        with self.assertRaisesRegex(ValueError, "nonexistent_module"):
            CorticalModuleBank(
                action_dim=len(LOCOMOTION_ACTIONS),
                rng=rng,
                module_dropout=0.0,
                disabled_modules=("nonexistent_module",),
            )

    def test_disabled_module_produces_zero_logits(self) -> None:
        bank = self._make_bank(("hunger_center",))
        obs = self._blank_observation()
        results = bank.forward(obs, store_cache=False, training=False)
        hunger_result = next(r for r in results if r.name == "hunger_center")
        np.testing.assert_allclose(hunger_result.logits, np.zeros(len(LOCOMOTION_ACTIONS), dtype=float))

    def test_disabled_module_skips_forward_pass(self) -> None:
        """
        Verifies that a disabled module's forward pass is skipped and its outputs are zeroed.
        
        Asserts the module's forward method is not invoked, the corresponding ModuleResult is marked inactive, and its logits are all zeros with length equal to the locomotion action space.
        """
        bank = self._make_bank(("hunger_center",))
        obs = self._blank_observation()

        def _disabled_forward(*args: object, **kwargs: object) -> np.ndarray:
            raise AssertionError("disabled module forward should not run")

        bank.modules["hunger_center"].forward = _disabled_forward  # type: ignore[assignment]
        results = bank.forward(obs, store_cache=False, training=False)

        hunger_result = next(r for r in results if r.name == "hunger_center")
        self.assertFalse(hunger_result.active)
        np.testing.assert_allclose(hunger_result.logits, np.zeros(len(LOCOMOTION_ACTIONS), dtype=float))

    def test_disabled_module_marked_inactive(self) -> None:
        bank = self._make_bank(("sleep_center",))
        obs = self._blank_observation()
        results = bank.forward(obs, store_cache=False, training=False)
        sleep_result = next(r for r in results if r.name == "sleep_center")
        self.assertFalse(sleep_result.active)

    def test_enabled_modules_remain_active(self) -> None:
        bank = self._make_bank(("alert_center",))
        obs = self._blank_observation()
        results = bank.forward(obs, store_cache=False, training=False)
        active_names = [r.name for r in results if r.active]
        self.assertNotIn("alert_center", active_names)
        self.assertIn("visual_cortex", active_names)
        self.assertIn("hunger_center", active_names)

    def test_no_disabled_modules_all_active(self) -> None:
        bank = self._make_bank(())
        obs = self._blank_observation()
        results = bank.forward(obs, store_cache=False, training=False)
        self.assertTrue(all(r.active for r in results))

    def test_dropout_keeps_one_non_disabled_module_active(self) -> None:
        rng = np.random.default_rng(42)
        bank = CorticalModuleBank(
            action_dim=len(LOCOMOTION_ACTIONS),
            rng=rng,
            module_dropout=1.0,
            disabled_modules=(
                "sensory_cortex",
                "hunger_center",
                "sleep_center",
                "alert_center",
            ),
        )
        obs = self._blank_observation()
        results = bank.forward(obs, store_cache=False, training=True)

        active_names = [result.name for result in results if result.active]
        self.assertEqual(active_names, ["visual_cortex"])


class CorticalModuleBankRecurrentModulesTest(unittest.TestCase):
    """Tests for selecting and resetting recurrent module proposers."""

    def _make_bank(self, recurrent: tuple[str, ...] = ()) -> CorticalModuleBank:
        rng = np.random.default_rng(44)
        return CorticalModuleBank(
            action_dim=len(LOCOMOTION_ACTIONS),
            rng=rng,
            module_dropout=0.0,
            recurrent_modules=recurrent,
        )

    def test_recurrent_module_uses_recurrent_network(self) -> None:
        bank = self._make_bank(("alert_center",))
        self.assertIsInstance(bank.modules["alert_center"], RecurrentProposalNetwork)
        self.assertNotIsInstance(bank.modules["hunger_center"], RecurrentProposalNetwork)

    def test_has_recurrent_modules_reflects_configuration(self) -> None:
        self.assertFalse(self._make_bank(()).has_recurrent_modules)
        self.assertTrue(self._make_bank(("alert_center",)).has_recurrent_modules)

    def test_unknown_recurrent_module_raises_value_error(self) -> None:
        rng = np.random.default_rng(44)
        with self.assertRaisesRegex(ValueError, "nonexistent_module"):
            CorticalModuleBank(
                action_dim=len(LOCOMOTION_ACTIONS),
                rng=rng,
                module_dropout=0.0,
                recurrent_modules=("nonexistent_module",),
            )

    def test_reset_hidden_states_clears_recurrent_modules(self) -> None:
        bank = self._make_bank(("alert_center",))
        recurrent = bank.modules["alert_center"]
        self.assertIsInstance(recurrent, RecurrentProposalNetwork)
        recurrent.hidden_state[:] = 1.0
        bank.reset_hidden_states()
        np.testing.assert_allclose(
            recurrent.hidden_state,
            np.zeros(recurrent.hidden_dim, dtype=float),
        )

    def test_snapshot_and_restore_hidden_states_round_trip(self) -> None:
        bank = self._make_bank(("alert_center",))
        recurrent = bank.modules["alert_center"]
        self.assertIsInstance(recurrent, RecurrentProposalNetwork)
        recurrent.hidden_state[:] = 0.25
        snapshot = bank.snapshot_hidden_states()
        recurrent.hidden_state[:] = 0.75
        bank.restore_hidden_states(snapshot)
        np.testing.assert_allclose(
            recurrent.hidden_state,
            np.full(recurrent.hidden_dim, 0.25, dtype=float),
        )

    def test_spider_brain_reset_hidden_states_delegates_to_module_bank(self) -> None:
        brain = SpiderBrain(seed=5, module_dropout=0.0)
        brain.module_bank = self._make_bank(("alert_center",))
        recurrent = brain.module_bank.modules["alert_center"]
        self.assertIsInstance(recurrent, RecurrentProposalNetwork)
        recurrent.hidden_state[:] = 1.0
        brain.reset_hidden_states()
        np.testing.assert_allclose(
            recurrent.hidden_state,
            np.zeros(recurrent.hidden_dim, dtype=float),
        )

    def test_spider_brain_estimate_value_does_not_commit_hidden_state(self) -> None:
        brain = SpiderBrain(seed=5, module_dropout=0.0)
        brain.module_bank = self._make_bank(("alert_center",))
        recurrent = brain.module_bank.modules["alert_center"]
        self.assertIsInstance(recurrent, RecurrentProposalNetwork)
        recurrent.hidden_state[:] = 0.5
        observation = _build_observation()
        brain.estimate_value(observation)
        np.testing.assert_allclose(
            recurrent.hidden_state,
            np.full(recurrent.hidden_dim, 0.5, dtype=float),
        )

    def test_recurrent_hidden_state_accumulates_within_episode_and_resets_between_episodes(self) -> None:
        config = BrainAblationConfig(
            name="recurrent_integration",
            architecture="modular",
            module_dropout=0.0,
            recurrent_modules=("alert_center",),
        )
        sim = SpiderSimulation(seed=17, max_steps=6, brain_config=config)
        observation = sim.world.reset(seed=17)
        sim.brain.reset_hidden_states()
        recurrent = sim.brain.module_bank.modules["alert_center"]
        self.assertIsInstance(recurrent, RecurrentProposalNetwork)
        np.testing.assert_allclose(
            recurrent.hidden_state,
            np.zeros(recurrent.hidden_dim, dtype=float),
        )

        first_step = sim.brain.act(observation, sample=False)
        hidden_after_first = recurrent.hidden_state.copy()
        self.assertGreater(float(np.linalg.norm(hidden_after_first)), 0.0)

        next_observation, _, _, _ = sim.world.step(first_step.action_idx)
        sim.brain.act(next_observation, sample=False)
        hidden_after_second = recurrent.hidden_state.copy()
        self.assertGreater(float(np.linalg.norm(hidden_after_second)), 0.0)
        self.assertFalse(np.allclose(hidden_after_second, hidden_after_first))

        sim.world.reset(seed=18)
        sim.brain.reset_hidden_states()
        np.testing.assert_allclose(
            recurrent.hidden_state,
            np.zeros(recurrent.hidden_dim, dtype=float),
        )

    def test_snapshot_returns_empty_dict_when_no_recurrent_modules(self) -> None:
        bank = self._make_bank(())
        snapshot = bank.snapshot_hidden_states()
        self.assertEqual(snapshot, {})

    def test_restore_raises_key_error_for_unknown_module(self) -> None:
        bank = self._make_bank(("alert_center",))
        recurrent = bank.modules["alert_center"]
        self.assertIsInstance(recurrent, RecurrentProposalNetwork)
        snapshot = {"nonexistent_module": np.zeros(recurrent.hidden_dim)}
        with self.assertRaises(KeyError):
            bank.restore_hidden_states(snapshot)

    def test_restore_raises_value_error_for_non_recurrent_module(self) -> None:
        bank = self._make_bank(("alert_center",))
        # hunger_center is feed-forward; it has no hidden_state attribute
        non_recurrent = bank.modules["hunger_center"]
        self.assertNotIsInstance(non_recurrent, RecurrentProposalNetwork)
        snapshot = {"hunger_center": np.zeros(10)}
        with self.assertRaises(ValueError):
            bank.restore_hidden_states(snapshot)

    def test_multiple_recurrent_modules_all_get_recurrent_networks(self) -> None:
        bank = self._make_bank(("alert_center", "sleep_center", "hunger_center"))
        for name in ("alert_center", "sleep_center", "hunger_center"):
            self.assertIsInstance(bank.modules[name], RecurrentProposalNetwork,
                                  f"{name} should be RecurrentProposalNetwork")
        # Feed-forward modules should remain feed-forward
        for name in ("visual_cortex", "sensory_cortex"):
            self.assertNotIsInstance(bank.modules[name], RecurrentProposalNetwork,
                                     f"{name} should not be RecurrentProposalNetwork")

    def test_snapshot_returns_copies_not_live_references(self) -> None:
        bank = self._make_bank(("alert_center",))
        recurrent = bank.modules["alert_center"]
        self.assertIsInstance(recurrent, RecurrentProposalNetwork)
        recurrent.hidden_state[:] = 0.3
        snapshot = bank.snapshot_hidden_states()
        # Mutate live hidden state after snapshot
        recurrent.hidden_state[:] = 0.9
        # Snapshot should retain the value at snapshot time
        np.testing.assert_allclose(
            snapshot["alert_center"],
            np.full(recurrent.hidden_dim, 0.3, dtype=float),
        )

    def test_reset_hidden_states_does_not_affect_feedforward_module_identity(self) -> None:
        # Resetting hidden states should not raise and must leave feed-forward modules intact
        bank = self._make_bank(("alert_center",))
        ff_before = bank.modules["hunger_center"]
        bank.reset_hidden_states()
        ff_after = bank.modules["hunger_center"]
        self.assertIs(ff_before, ff_after)

    def test_spider_brain_reset_hidden_states_is_noop_with_no_module_bank(self) -> None:
        brain = SpiderBrain(seed=7, module_dropout=0.0)
        brain.module_bank = None
        # Should not raise
        brain.reset_hidden_states()

    def test_run_episode_resets_recurrent_hidden_state_at_start(self) -> None:
        config = BrainAblationConfig(
            name="recurrent_episode_reset",
            architecture="modular",
            module_dropout=0.0,
            recurrent_modules=("alert_center",),
        )
        sim = SpiderSimulation(seed=19, max_steps=3, brain_config=config)
        recurrent = sim.brain.module_bank.modules["alert_center"]
        self.assertIsInstance(recurrent, RecurrentProposalNetwork)
        dirty_state = np.full(recurrent.hidden_dim, 99.0, dtype=float)
        recurrent.hidden_state[:] = dirty_state
        reset_snapshots: list[np.ndarray] = []
        original_reset_hidden_states = sim.brain.reset_hidden_states

        def wrapped_reset_hidden_states() -> None:
            original_reset_hidden_states()
            reset_snapshots.append(recurrent.get_hidden_state())

        with mock.patch.object(
            sim.brain,
            "reset_hidden_states",
            side_effect=wrapped_reset_hidden_states,
        ) as reset_mock:
            sim.run_episode(0, training=False, sample=False)

        self.assertEqual(reset_mock.call_count, 1)
        self.assertEqual(len(reset_snapshots), 1)
        np.testing.assert_allclose(
            reset_snapshots[0],
            np.zeros(recurrent.hidden_dim, dtype=float),
        )
        self.assertFalse(np.allclose(reset_snapshots[0], dirty_state))
        self.assertTrue(np.all(np.isfinite(recurrent.hidden_state)))


class ModuleResultNamedObservationTest(unittest.TestCase):
    """Tests for ModuleResult.named_observation (new None-guard added in this PR)."""

    def test_named_observation_returns_empty_dict_for_none_interface(self) -> None:
        result = ModuleResult(
            interface=None,
            name="monolithic_policy",
            observation_key="monolithic_policy",
            observation=np.zeros(len(LOCOMOTION_ACTIONS), dtype=float),
            logits=np.zeros(len(LOCOMOTION_ACTIONS), dtype=float),
            probs=np.full(
                len(LOCOMOTION_ACTIONS),
                1.0 / len(LOCOMOTION_ACTIONS),
                dtype=float,
            ),
            active=True,
        )
        self.assertEqual(result.named_observation(), {})

    def test_named_observation_works_with_valid_interface(self) -> None:
        interface = MODULE_INTERFACE_BY_NAME["alert_center"]
        obs = np.zeros(interface.input_dim, dtype=float)
        result = ModuleResult(
            interface=interface,
            name="alert_center",
            observation_key=interface.observation_key,
            observation=obs,
            logits=np.zeros(len(LOCOMOTION_ACTIONS), dtype=float),
            probs=np.full(
                len(LOCOMOTION_ACTIONS),
                1.0 / len(LOCOMOTION_ACTIONS),
                dtype=float,
            ),
            active=True,
        )
        named = result.named_observation()
        self.assertIsInstance(named, dict)
        self.assertIn("predator_visible", named)

class SpiderSimulationBrainConfigTest(unittest.TestCase):
    """Tests for the brain_config parameter added to SpiderSimulation."""

    def test_default_brain_config_is_modular_full(self) -> None:
        sim = SpiderSimulation(seed=7, max_steps=5)
        self.assertEqual(sim.brain_config.name, "modular_full")
        self.assertEqual(sim.brain_config.architecture, "modular")

    def test_custom_brain_config_stored(self) -> None:
        config = BrainAblationConfig(name="no_module_dropout", architecture="modular", module_dropout=0.0)
        sim = SpiderSimulation(seed=7, max_steps=5, brain_config=config)
        self.assertEqual(sim.brain_config.name, "no_module_dropout")

    def test_module_dropout_comes_from_brain_config(self) -> None:
        config = BrainAblationConfig(name="no_module_dropout", architecture="modular", module_dropout=0.0)
        sim = SpiderSimulation(seed=7, max_steps=5, brain_config=config)
        self.assertEqual(sim.module_dropout, 0.0)

    def test_training_td_update_publishes_credit_diagnostics(self) -> None:
        sim = SpiderSimulation(seed=7, max_steps=2)

        sim.run_episode(
            episode_index=0,
            training=True,
            sample=False,
        )

        td_updates = [
            message.payload
            for message in sim.bus.history()
            if message.topic == "td_update"
        ]
        self.assertGreater(len(td_updates), 0)
        payload = td_updates[0]
        self.assertIn("module_credit_weights", payload)
        self.assertIn("module_gradient_norms", payload)
        self.assertEqual(payload["credit_strategy"], "broadcast")

    def test_brain_config_propagated_to_brain(self) -> None:
        config = BrainAblationConfig(
            name="drop_alert_center",
            architecture="modular",
            module_dropout=0.0,
            disabled_modules=("alert_center",),
        )
        sim = SpiderSimulation(seed=7, max_steps=5, brain_config=config)
        self.assertEqual(sim.brain.config.name, "drop_alert_center")
        self.assertIn("alert_center", sim.brain.config.disabled_modules)

    def test_annotate_behavior_rows_uses_effective_eval_reflex_scale(self) -> None:
        config = BrainAblationConfig(
            name="no_module_reflexes",
            architecture="modular",
            module_dropout=0.0,
            enable_reflexes=False,
            enable_auxiliary_targets=False,
            reflex_scale=0.0,
        )
        sim = SpiderSimulation(seed=7, max_steps=5, brain_config=config)

        rows = sim._annotate_behavior_rows([{"scenario": "night_rest"}], eval_reflex_scale=1.0)

        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(float(rows[0]["eval_reflex_scale"]), 0.0)


class AnnotateBehaviorRowsTest(unittest.TestCase):
    """Tests for SpiderSimulation._annotate_behavior_rows (new in this PR)."""

    def test_annotate_adds_ablation_variant(self) -> None:
        sim = SpiderSimulation(seed=7, max_steps=5)
        rows = [{"scenario": "night_rest", "success": 1}]
        annotated = sim._annotate_behavior_rows(rows)
        self.assertIn("ablation_variant", annotated[0])

    def test_annotate_adds_ablation_architecture(self) -> None:
        sim = SpiderSimulation(seed=7, max_steps=5)
        rows = [{"scenario": "night_rest", "success": 1}]
        annotated = sim._annotate_behavior_rows(rows)
        self.assertIn("ablation_architecture", annotated[0])

    def test_annotate_adds_operational_profile_metadata(self) -> None:
        sim = SpiderSimulation(seed=7, max_steps=5)
        rows = [{"scenario": "night_rest", "success": 1}]
        annotated = sim._annotate_behavior_rows(rows)
        self.assertEqual(annotated[0]["operational_profile"], "default_v1")
        self.assertEqual(annotated[0]["operational_profile_version"], 1)
        self.assertEqual(annotated[0]["noise_profile"], "none")
        self.assertIn("noise_profile_config", annotated[0])

    def test_annotate_variant_matches_config_name(self) -> None:
        config = BrainAblationConfig(name="drop_sleep_center", architecture="modular", module_dropout=0.0, disabled_modules=("sleep_center",))
        sim = SpiderSimulation(seed=7, max_steps=5, brain_config=config)
        rows = [{"scenario": "night_rest", "success": 0}]
        annotated = sim._annotate_behavior_rows(rows)
        self.assertEqual(annotated[0]["ablation_variant"], "drop_sleep_center")

    def test_annotate_architecture_matches_config(self) -> None:
        config = BrainAblationConfig(name="monolithic_policy", architecture="monolithic", module_dropout=0.0)
        sim = SpiderSimulation(seed=7, max_steps=5, brain_config=config)
        rows = [{"scenario": "night_rest", "success": 0}]
        annotated = sim._annotate_behavior_rows(rows)
        self.assertEqual(annotated[0]["ablation_architecture"], "monolithic")

    def test_annotate_preserves_original_row_content(self) -> None:
        sim = SpiderSimulation(seed=7, max_steps=5)
        rows = [{"scenario": "night_rest", "success": 1, "failures": []}]
        annotated = sim._annotate_behavior_rows(rows)
        self.assertEqual(annotated[0]["scenario"], "night_rest")
        self.assertEqual(annotated[0]["success"], 1)

    def test_annotate_does_not_mutate_original_rows(self) -> None:
        sim = SpiderSimulation(seed=7, max_steps=5)
        rows = [{"scenario": "night_rest"}]
        sim._annotate_behavior_rows(rows)
        self.assertNotIn("ablation_variant", rows[0])

    def test_annotate_empty_rows_returns_empty(self) -> None:
        sim = SpiderSimulation(seed=7, max_steps=5)
        annotated = sim._annotate_behavior_rows([])
        self.assertEqual(annotated, [])

    def test_annotate_multiple_rows(self) -> None:
        sim = SpiderSimulation(seed=7, max_steps=5)
        rows = [{"scenario": "night_rest"}, {"scenario": "predator_edge"}]
        annotated = sim._annotate_behavior_rows(rows)
        self.assertEqual(len(annotated), 2)
        for row in annotated:
            self.assertIn("ablation_variant", row)




class BuildSummaryBrainConfigTest(unittest.TestCase):
    """Tests that _build_summary now includes brain config info."""

    def test_summary_config_has_brain_key(self) -> None:
        sim = SpiderSimulation(seed=7, max_steps=5)
        summary = sim._build_summary([], [])
        self.assertIn("brain", summary["config"])

    def test_summary_brain_config_architecture_matches(self) -> None:
        sim = SpiderSimulation(seed=7, max_steps=5)
        summary = sim._build_summary([], [])
        self.assertEqual(summary["config"]["brain"]["architecture"], "modular")

    def test_summary_config_has_operational_profile(self) -> None:
        sim = SpiderSimulation(seed=7, max_steps=5)
        summary = sim._build_summary([], [])
        self.assertIn("operational_profile", summary["config"])
        self.assertEqual(summary["config"]["operational_profile"]["name"], "default_v1")
        self.assertIn("version", summary["config"]["operational_profile"])
        self.assertEqual(summary["config"]["operational_profile"]["version"], 1)
        self.assertIn("noise_profile", summary["config"])
        self.assertEqual(summary["config"]["noise_profile"]["name"], "none")

    def test_summary_brain_config_name_matches(self) -> None:
        """
        Verify that SpiderSimulation._build_summary includes the default brain config name "modular_full".
        """
        sim = SpiderSimulation(seed=7, max_steps=5)
        summary = sim._build_summary([], [])
        self.assertEqual(summary["config"]["brain"]["name"], "modular_full")

    def test_summary_has_architecture_traceability(self) -> None:
        sim = SpiderSimulation(seed=7, max_steps=5)
        summary = sim._build_summary([], [])
        self.assertEqual(summary["config"]["architecture_version"], SpiderBrain.ARCHITECTURE_VERSION)
        self.assertTrue(summary["config"]["architecture_fingerprint"])

    def test_summary_includes_reward_audit(self) -> None:
        """
        Verifies that SpiderSimulation._build_summary includes a reward_audit section with expected keys and values.
        
        Asserts that the summary contains a top-level "reward_audit" entry, that its "current_profile" is "classic", that "predator_dist" appears in "observation_signals", and that "austere" is listed among "reward_profiles".
        """
        sim = SpiderSimulation(seed=7, max_steps=5)
        summary = sim._build_summary([], [])
        self.assertIn("reward_audit", summary)
        self.assertEqual(summary["reward_audit"]["current_profile"], "classic")
        self.assertIn("predator_dist", summary["reward_audit"]["observation_signals"])
        self.assertIn("austere", summary["reward_audit"]["reward_profiles"])
        self.assertIn("comparison", summary["reward_audit"])
        self.assertIn("classic", summary["reward_audit"]["comparison"]["profiles"])
        self.assertIn("shaping_reduction_status", summary)
        self.assertEqual(
            summary["shaping_reduction_status"]["roadmap_target_count"],
            len(
                summary["reward_audit"]["reward_profiles"]["classic"][
                    "reduction_roadmap_status"
                ]
            ),
        )

    def test_summary_monolithic_brain_config_in_summary(self) -> None:
        config = BrainAblationConfig(name="monolithic_policy", architecture="monolithic", module_dropout=0.0)
        sim = SpiderSimulation(seed=7, max_steps=5, brain_config=config)
        summary = sim._build_summary([], [])
        self.assertEqual(summary["config"]["brain"]["architecture"], "monolithic")



class SpiderSimulationOperationalProfileTest(unittest.TestCase):
    """Tests for operational_profile propagation through SpiderSimulation."""

    def test_simulation_stores_resolved_operational_profile(self) -> None:
        sim = SpiderSimulation(seed=7, max_steps=5)
        self.assertEqual(sim.operational_profile.name, "default_v1")

    def test_simulation_accepts_profile_by_name(self) -> None:
        sim = SpiderSimulation(seed=7, max_steps=5, operational_profile="default_v1")
        self.assertEqual(sim.operational_profile.name, "default_v1")

    def test_simulation_accepts_profile_instance(self) -> None:
        profile = _profile_with_updates(predator_threat_smell_threshold=0.01)
        sim = SpiderSimulation(seed=7, max_steps=5, operational_profile=profile)
        self.assertIs(sim.operational_profile, profile)

    def test_simulation_propagates_profile_to_world(self) -> None:
        profile = _profile_with_updates(predator_threat_smell_threshold=0.01)
        sim = SpiderSimulation(seed=7, max_steps=5, operational_profile=profile)
        self.assertIs(sim.world.operational_profile, profile)

    def test_simulation_propagates_profile_to_brain(self) -> None:
        profile = _profile_with_updates(predator_threat_smell_threshold=0.01)
        sim = SpiderSimulation(seed=7, max_steps=5, operational_profile=profile)
        self.assertIs(sim.brain.operational_profile, profile)

    def test_summary_operational_profile_version_is_int(self) -> None:
        sim = SpiderSimulation(seed=7, max_steps=5)
        summary = sim._build_summary([], [])
        self.assertIsInstance(summary["config"]["operational_profile"]["version"], int)

    def test_annotate_behavior_rows_with_custom_profile(self) -> None:
        profile = _profile_with_updates(predator_threat_smell_threshold=0.01)
        sim = SpiderSimulation(seed=7, max_steps=5, operational_profile=profile)
        rows = [{"scenario": "night_rest"}]
        annotated = sim._annotate_behavior_rows(rows)
        self.assertEqual(annotated[0]["operational_profile"], "ablation_test_profile")
        self.assertEqual(annotated[0]["operational_profile_version"], 21)
        self.assertEqual(annotated[0]["noise_profile"], "none")





class LocalCreditOnlyVariantTest(unittest.TestCase):
    """Tests for the local_credit_only ablation variant (new in this PR)."""

    def test_uses_local_credit_only_is_false_for_modular_full(self) -> None:
        config = BrainAblationConfig(name="modular_full")
        self.assertFalse(config.uses_local_credit_only)

    def test_uses_local_credit_only_is_false_for_no_module_dropout(self) -> None:
        config = BrainAblationConfig(name="no_module_dropout", module_dropout=0.0)
        self.assertFalse(config.uses_local_credit_only)

    def test_uses_local_credit_only_is_false_for_no_module_reflexes(self) -> None:
        config = BrainAblationConfig(name="no_module_reflexes", enable_reflexes=False)
        self.assertFalse(config.uses_local_credit_only)

    def test_uses_local_credit_only_is_false_for_monolithic_policy(self) -> None:
        config = BrainAblationConfig(name="monolithic_policy", architecture="monolithic")
        self.assertFalse(config.uses_local_credit_only)

    def test_uses_local_credit_only_is_true_only_for_local_only_strategy(self) -> None:
        config = BrainAblationConfig(
            name="custom_local_credit",
            credit_strategy="local_only",
        )
        self.assertTrue(config.uses_local_credit_only)

    def test_uses_local_credit_only_is_false_for_drop_variants(self) -> None:
        for module_name in MODULE_NAMES:
            config = BrainAblationConfig(
                name=f"drop_{module_name}",
                disabled_modules=(module_name,),
            )
            self.assertFalse(
                config.uses_local_credit_only,
                f"Expected drop_{module_name} to have uses_local_credit_only=False",
            )

    def test_local_credit_only_is_modular_not_monolithic(self) -> None:
        configs = canonical_ablation_configs()
        lc = configs["local_credit_only"]
        self.assertTrue(lc.is_modular)
        self.assertFalse(lc.is_monolithic)

    def test_local_credit_only_has_reflexes_and_aux_targets_enabled(self) -> None:
        configs = canonical_ablation_configs()
        lc = configs["local_credit_only"]
        self.assertTrue(lc.enable_reflexes)
        self.assertTrue(lc.enable_auxiliary_targets)

    def test_local_credit_only_has_no_disabled_modules(self) -> None:
        configs = canonical_ablation_configs()
        lc = configs["local_credit_only"]
        self.assertEqual(lc.disabled_modules, ())

    def test_local_credit_only_has_reflex_scale_one(self) -> None:
        configs = canonical_ablation_configs()
        lc = configs["local_credit_only"]
        self.assertAlmostEqual(lc.reflex_scale, 1.0)

    def test_local_credit_only_uses_local_only_strategy(self) -> None:
        configs = canonical_ablation_configs()
        lc = configs["local_credit_only"]
        self.assertEqual(lc.credit_strategy, "local_only")
        self.assertTrue(lc.uses_local_credit_only)
        self.assertFalse(lc.uses_counterfactual_credit)

    def test_local_credit_only_appears_after_no_module_reflexes_in_canonical_order(self) -> None:
        names = canonical_ablation_variant_names()
        self.assertIn("local_credit_only", names)
        self.assertIn("no_module_reflexes", names)
        idx_lc = names.index("local_credit_only")
        idx_nr = names.index("no_module_reflexes")
        self.assertGreater(
            idx_lc,
            idx_nr,
            "local_credit_only should appear after no_module_reflexes in canonical order",
        )

    def test_local_credit_only_appears_before_monolithic_policy_in_canonical_order(self) -> None:
        """
        Assert that the canonical ablation variant ordering lists "local_credit_only" before "monolithic_policy".
        
        Verifies the relative positions of these two canonical variant names returned by canonical_ablation_variant_names().
        """
        names = canonical_ablation_variant_names()
        idx_lc = names.index("local_credit_only")
        idx_mono = names.index("monolithic_policy")
        self.assertLess(
            idx_lc,
            idx_mono,
            "local_credit_only should appear before monolithic_policy in canonical order",
        )

    def test_uses_local_credit_only_is_not_tied_to_variant_name(self) -> None:
        config = BrainAblationConfig(
            name="local_credit_only",
            architecture="modular",
            enable_reflexes=True,
            enable_auxiliary_targets=True,
            disabled_modules=(),
            reflex_scale=1.0,
            module_reflex_scales={},
        )
        self.assertFalse(config.uses_local_credit_only)

    def test_resolve_ablation_configs_returns_local_credit_only_by_name(self) -> None:
        configs = resolve_ablation_configs(["local_credit_only"])
        self.assertEqual(len(configs), 1)
        self.assertEqual(configs[0].name, "local_credit_only")
        self.assertTrue(configs[0].uses_local_credit_only)

    def test_local_credit_only_to_summary_includes_name(self) -> None:
        configs = canonical_ablation_configs()
        summary = configs["local_credit_only"].to_summary()
        self.assertEqual(summary["name"], "local_credit_only")
        self.assertEqual(summary["architecture"], "modular")
        self.assertEqual(summary["credit_strategy"], "local_only")

    def test_local_credit_only_custom_dropout_propagates(self) -> None:
        configs = canonical_ablation_configs(module_dropout=0.12)
        lc = configs["local_credit_only"]
        self.assertAlmostEqual(lc.module_dropout, 0.12)


class CanonicalModularFullLearnedArbitrationTest(unittest.TestCase):
    """Tests that the modular_full canonical config explicitly enables learned arbitration."""

    def test_modular_full_has_use_learned_arbitration_true(self) -> None:
        configs = canonical_ablation_configs()
        mf = configs["modular_full"]
        self.assertTrue(mf.use_learned_arbitration)

    def test_no_module_dropout_has_use_learned_arbitration_true(self) -> None:
        configs = canonical_ablation_configs()
        self.assertTrue(configs["no_module_dropout"].use_learned_arbitration)

    def test_no_module_reflexes_has_use_learned_arbitration_true(self) -> None:
        configs = canonical_ablation_configs()
        self.assertTrue(configs["no_module_reflexes"].use_learned_arbitration)

    def test_fixed_arbitration_baseline_has_use_learned_arbitration_false(self) -> None:
        configs = canonical_ablation_configs()
        self.assertFalse(configs["fixed_arbitration_baseline"].use_learned_arbitration)

    def test_learned_arbitration_no_regularization_has_use_learned_arbitration_true(self) -> None:
        configs = canonical_ablation_configs()
        self.assertTrue(configs["learned_arbitration_no_regularization"].use_learned_arbitration)

    def test_drop_variants_have_use_learned_arbitration_true(self) -> None:
        configs = canonical_ablation_configs()
        for module_name in MODULE_NAMES:
            config = configs[f"drop_{module_name}"]
            self.assertTrue(
                config.use_learned_arbitration,
                f"drop_{module_name} should have use_learned_arbitration=True",
            )

    def test_to_summary_for_modular_full_includes_use_learned_arbitration(self) -> None:
        configs = canonical_ablation_configs()
        summary = configs["modular_full"].to_summary()
        self.assertIn("use_learned_arbitration", summary)
        self.assertTrue(summary["use_learned_arbitration"])

    def test_to_summary_for_fixed_baseline_has_false(self) -> None:
        configs = canonical_ablation_configs()
        summary = configs["fixed_arbitration_baseline"].to_summary()
        self.assertFalse(summary["use_learned_arbitration"])

class BrainAblationConfigCreditStrategyExtendedTest(unittest.TestCase):
    """Additional tests for the credit_strategy field introduced in this PR."""

    def test_credit_strategy_is_coerced_to_str(self) -> None:
        """credit_strategy should be stored as a str regardless of input type."""
        # Pass a string-like integer as credit_strategy by subclassing str
        class StrLike(str):
            pass

        config = BrainAblationConfig(name="test", credit_strategy=StrLike("broadcast"))
        self.assertIsInstance(config.credit_strategy, str)

    def test_credit_strategy_coercion_matches_expected_value(self) -> None:
        """credit_strategy is stored exactly as the coerced str."""
        for strategy in ("broadcast", "local_only", "counterfactual"):
            with self.subTest(strategy=strategy):
                config = BrainAblationConfig(name="test", credit_strategy=strategy)
                self.assertEqual(config.credit_strategy, strategy)

    def test_default_name_is_custom(self) -> None:
        """When BrainAblationConfig is created without a name, it defaults to 'custom'."""
        config = BrainAblationConfig()
        self.assertEqual(config.name, "custom")

    def test_credit_strategy_multiple_invalid_values_raise(self) -> None:
        """Each unsupported credit strategy string raises ValueError."""
        invalid_strategies = ["global", "monte_carlo", "td_error", "", "BROADCAST"]
        for strategy in invalid_strategies:
            with self.subTest(strategy=strategy):
                with self.assertRaises(ValueError):
                    BrainAblationConfig(name="test", credit_strategy=strategy)

    def test_to_summary_local_only_credit_strategy(self) -> None:
        """to_summary includes credit_strategy='local_only' for local_only configs."""
        config = BrainAblationConfig(name="test_lc", credit_strategy="local_only")
        summary = config.to_summary()
        self.assertEqual(summary["credit_strategy"], "local_only")

    def test_to_summary_counterfactual_credit_strategy(self) -> None:
        """to_summary includes credit_strategy='counterfactual' for counterfactual configs."""
        config = BrainAblationConfig(name="test_cf", credit_strategy="counterfactual")
        summary = config.to_summary()
        self.assertEqual(summary["credit_strategy"], "counterfactual")

    def test_credit_strategy_is_frozen(self) -> None:
        """BrainAblationConfig is frozen; attempting to set credit_strategy raises."""
        config = BrainAblationConfig(name="test", credit_strategy="broadcast")
        with self.assertRaises((AttributeError, TypeError)):
            config.credit_strategy = "counterfactual"  # type: ignore[misc]

    def test_counterfactual_credit_uses_counterfactual_is_independent_of_name(self) -> None:
        """uses_counterfactual_credit is determined by credit_strategy, not by variant name."""
        config_with_name = BrainAblationConfig(
            name="counterfactual_credit",
            credit_strategy="broadcast",
        )
        self.assertFalse(config_with_name.uses_counterfactual_credit)

        config_without_name = BrainAblationConfig(
            name="something_else",
            credit_strategy="counterfactual",
        )
        self.assertTrue(config_without_name.uses_counterfactual_credit)

    def test_local_credit_only_is_independent_of_name(self) -> None:
        """uses_local_credit_only is determined by credit_strategy, not by variant name."""
        config_with_name = BrainAblationConfig(
            name="local_credit_only",
            credit_strategy="broadcast",
        )
        self.assertFalse(config_with_name.uses_local_credit_only)

        config_without_name = BrainAblationConfig(
            name="something_entirely_different",
            credit_strategy="local_only",
        )
        self.assertTrue(config_without_name.uses_local_credit_only)

    def test_counterfactual_credit_canonical_config_to_summary_round_trip(self) -> None:
        """counterfactual_credit canonical config survives a to_summary round-trip check."""
        configs = canonical_ablation_configs()
        cf = configs["counterfactual_credit"]
        summary = cf.to_summary()
        self.assertEqual(summary["name"], "counterfactual_credit")
        self.assertEqual(summary["credit_strategy"], "counterfactual")
        self.assertEqual(summary["architecture"], "modular")
        self.assertTrue(summary["enable_reflexes"])
        self.assertTrue(summary["enable_auxiliary_targets"])

    def test_local_credit_only_canonical_config_to_summary_round_trip(self) -> None:
        """local_credit_only canonical config to_summary includes credit_strategy='local_only'."""
        configs = canonical_ablation_configs()
        lc = configs["local_credit_only"]
        summary = lc.to_summary()
        self.assertEqual(summary["name"], "local_credit_only")
        self.assertEqual(summary["credit_strategy"], "local_only")


class CreditStrategyLearnDiagnosticsTest(unittest.TestCase):
    """Tests verifying learn() diagnostics for the new credit-strategy fields."""

    @staticmethod
    def _build_obs() -> dict[str, np.ndarray]:
        return _build_observation()

    def test_learn_stats_sorted_credit_weights_keys(self) -> None:
        """module_credit_weights returned by learn() are sorted by module name."""
        brain = SpiderBrain(
            seed=1,
            config=BrainAblationConfig(
                name="test",
                architecture="modular",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                credit_strategy="broadcast",
            ),
        )
        obs = self._build_obs()
        # sample=True so training mode is active and networks cache state for backward
        step = brain.act(obs, sample=True)
        stats = brain.learn(step, reward=0.0, next_observation=obs, done=True)

        keys = list(stats["module_credit_weights"].keys())
        self.assertEqual(keys, sorted(keys))

    def test_learn_stats_sorted_gradient_norm_keys(self) -> None:
        """module_gradient_norms returned by learn() are sorted by module name."""
        brain = SpiderBrain(
            seed=2,
            config=BrainAblationConfig(
                name="test",
                architecture="modular",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                credit_strategy="broadcast",
            ),
        )
        obs = self._build_obs()
        # sample=True so training mode is active and networks cache state for backward
        step = brain.act(obs, sample=True)
        stats = brain.learn(step, reward=0.0, next_observation=obs, done=True)

        keys = list(stats["module_gradient_norms"].keys())
        self.assertEqual(keys, sorted(keys))

    def test_learn_raises_value_error_for_reflex_only_policy_mode(self) -> None:
        """learn() raises ValueError when the decision has policy_mode != 'normal'."""
        brain = SpiderBrain(
            seed=3,
            config=BrainAblationConfig(
                name="test",
                architecture="modular",
                module_dropout=0.0,
                enable_reflexes=True,
                enable_auxiliary_targets=False,
                credit_strategy="broadcast",
            ),
        )
        obs = self._build_obs()
        step = brain.act(obs, sample=False, policy_mode="normal")
        step.policy_mode = "reflex_only"

        with self.assertRaises(ValueError):
            brain.learn(step, reward=0.0, next_observation=obs, done=True)


class SafeFloatTest(unittest.TestCase):
    """Tests for ablations._safe_float() - new in this PR."""

    def test_valid_int_converts_to_float(self) -> None:
        self.assertEqual(_safe_float(3), 3.0)

    def test_valid_float_returns_float(self) -> None:
        self.assertAlmostEqual(_safe_float(0.75), 0.75)

    def test_string_numeric_converts(self) -> None:
        self.assertAlmostEqual(_safe_float("0.5"), 0.5)

    def test_none_returns_zero(self) -> None:
        self.assertEqual(_safe_float(None), 0.0)

    def test_nan_returns_zero(self) -> None:
        import math
        self.assertEqual(_safe_float(float("nan")), 0.0)

    def test_positive_infinity_returns_zero(self) -> None:
        self.assertEqual(_safe_float(float("inf")), 0.0)

    def test_negative_infinity_returns_zero(self) -> None:
        self.assertEqual(_safe_float(float("-inf")), 0.0)

    def test_non_numeric_string_returns_zero(self) -> None:
        self.assertEqual(_safe_float("not_a_number"), 0.0)

    def test_empty_string_returns_zero(self) -> None:
        self.assertEqual(_safe_float(""), 0.0)

    def test_zero_returns_zero_float(self) -> None:
        self.assertEqual(_safe_float(0), 0.0)

    def test_negative_finite_value_returns_float(self) -> None:
        self.assertAlmostEqual(_safe_float(-1.5), -1.5)

    def test_bool_true_converts_to_one(self) -> None:
        self.assertAlmostEqual(_safe_float(True), 1.0)

    def test_bool_false_converts_to_zero(self) -> None:
        self.assertEqual(_safe_float(False), 0.0)


class AblationMeanTest(unittest.TestCase):
    """Tests for ablations._mean() - new in this PR."""

    def test_empty_sequence_returns_zero(self) -> None:
        self.assertEqual(_mean([]), 0.0)

    def test_single_element(self) -> None:
        self.assertAlmostEqual(_mean([0.7]), 0.7)

    def test_multiple_elements(self) -> None:
        self.assertAlmostEqual(_mean([0.0, 1.0, 0.5]), 0.5)

    def test_all_zeros(self) -> None:
        self.assertAlmostEqual(_mean([0.0, 0.0, 0.0]), 0.0)

    def test_all_ones(self) -> None:
        self.assertAlmostEqual(_mean([1.0, 1.0, 1.0]), 1.0)

    def test_tuple_input(self) -> None:
        self.assertAlmostEqual(_mean((0.2, 0.4, 0.6)), 0.4)

    def test_result_is_float(self) -> None:
        result = _mean([1, 2, 3])
        self.assertIsInstance(result, float)


class ScenarioSuccessRateTest(unittest.TestCase):
    """Tests for ablations._scenario_success_rate() - new in this PR."""

    def test_reads_from_suite_key(self) -> None:
        payload = {
            "suite": {
                "visual_olfactory_pincer": {"success_rate": 0.6},
            }
        }
        result = _scenario_success_rate(payload, "visual_olfactory_pincer")
        self.assertAlmostEqual(result, 0.6)

    def test_reads_from_legacy_scenarios_key_when_suite_is_not_mapping(self) -> None:
        # The legacy_scenarios path is only reached when "suite" is not a mapping
        payload = {
            "suite": "invalid",
            "legacy_scenarios": {
                "olfactory_ambush": {"success_rate": 0.4},
            }
        }
        result = _scenario_success_rate(payload, "olfactory_ambush")
        self.assertAlmostEqual(result, 0.4)

    def test_suite_takes_precedence_over_legacy(self) -> None:
        payload = {
            "suite": {"s1": {"success_rate": 0.8}},
            "legacy_scenarios": {"s1": {"success_rate": 0.2}},
        }
        result = _scenario_success_rate(payload, "s1")
        self.assertAlmostEqual(result, 0.8)

    def test_missing_scenario_in_suite_returns_none(self) -> None:
        payload = {"suite": {}}
        result = _scenario_success_rate(payload, "nonexistent_scenario")
        self.assertIsNone(result)

    def test_empty_payload_returns_none(self) -> None:
        result = _scenario_success_rate({}, "visual_olfactory_pincer")
        self.assertIsNone(result)

    def test_suite_not_a_mapping_falls_through_to_legacy(self) -> None:
        payload = {
            "suite": "invalid",
            "legacy_scenarios": {"s1": {"success_rate": 0.3}},
        }
        result = _scenario_success_rate(payload, "s1")
        self.assertAlmostEqual(result, 0.3)

    def test_missing_success_rate_key_returns_none(self) -> None:
        payload = {"suite": {"s1": {}}}
        result = _scenario_success_rate(payload, "s1")
        self.assertIsNone(result)

    def test_both_suite_and_legacy_missing_returns_none(self) -> None:
        result = _scenario_success_rate({"other_key": {}}, "any_scenario")
        self.assertIsNone(result)

    def test_explicit_none_success_rate_returns_none(self) -> None:
        payload = {"suite": {"s1": {"success_rate": None}}}
        result = _scenario_success_rate(payload, "s1")
        self.assertIsNone(result)

    def test_invalid_success_rate_returns_none(self) -> None:
        payload = {"suite": {"s1": {"success_rate": "not-a-number"}}}
        result = _scenario_success_rate(payload, "s1")
        self.assertIsNone(result)


class CanonicalAblationScenarioGroupsTest(unittest.TestCase):
    """Tests for canonical_ablation_scenario_groups() - new in this PR."""

    def test_returns_all_three_groups(self) -> None:
        groups = canonical_ablation_scenario_groups()
        self.assertIn("multi_predator_ecology", groups)
        self.assertIn("visual_predator_scenarios", groups)
        self.assertIn("olfactory_predator_scenarios", groups)

    def test_multi_predator_ecology_matches_constant(self) -> None:
        groups = canonical_ablation_scenario_groups()
        self.assertEqual(groups["multi_predator_ecology"], MULTI_PREDATOR_SCENARIOS)

    def test_visual_predator_scenarios_matches_constant(self) -> None:
        groups = canonical_ablation_scenario_groups()
        self.assertEqual(groups["visual_predator_scenarios"], VISUAL_PREDATOR_SCENARIOS)

    def test_olfactory_predator_scenarios_matches_constant(self) -> None:
        groups = canonical_ablation_scenario_groups()
        self.assertEqual(groups["olfactory_predator_scenarios"], OLFACTORY_PREDATOR_SCENARIOS)

    def test_values_are_tuples(self) -> None:
        groups = canonical_ablation_scenario_groups()
        for name, scenarios in groups.items():
            with self.subTest(group=name):
                self.assertIsInstance(scenarios, tuple)

    def test_visual_pincer_in_all_groups(self) -> None:
        groups = canonical_ablation_scenario_groups()
        for name, scenarios in groups.items():
            with self.subTest(group=name):
                self.assertIn("visual_olfactory_pincer", scenarios)


class ResolveAblationScenarioGroupEdgeCasesTest(unittest.TestCase):
    """Additional edge-case tests for resolve_ablation_scenario_group() - new in this PR."""

    def test_unknown_group_raises_key_error(self) -> None:
        with self.assertRaises(KeyError):
            resolve_ablation_scenario_group("nonexistent_group")

    def test_key_error_message_lists_available_groups(self) -> None:
        try:
            resolve_ablation_scenario_group("bad_group")
        except KeyError as e:
            msg = str(e)
            self.assertIn("multi_predator_ecology", msg)
            self.assertIn("visual_predator_scenarios", msg)

    def test_returns_tuple_type(self) -> None:
        result = resolve_ablation_scenario_group("multi_predator_ecology")
        self.assertIsInstance(result, tuple)

    def test_visual_group_contains_correct_scenarios(self) -> None:
        result = resolve_ablation_scenario_group("visual_predator_scenarios")
        self.assertIn("visual_olfactory_pincer", result)
        self.assertIn("visual_hunter_open_field", result)
        self.assertNotIn("olfactory_ambush", result)

    def test_olfactory_group_contains_correct_scenarios(self) -> None:
        result = resolve_ablation_scenario_group("olfactory_predator_scenarios")
        self.assertIn("visual_olfactory_pincer", result)
        self.assertIn("olfactory_ambush", result)
        self.assertNotIn("visual_hunter_open_field", result)


class ComparePredatorTypeAblationEdgeCasesTest(unittest.TestCase):
    """Edge-case and boundary tests for compare_predator_type_ablation_performance() - new in this PR."""

    def test_non_mapping_variants_value_returns_unavailable(self) -> None:
        # The only way to get available=False is variants not being a Mapping
        result = compare_predator_type_ablation_performance({"variants": [1, 2, 3]})
        self.assertFalse(result["available"])
        self.assertEqual(result["comparisons"], {})

    def test_non_mapping_variants_returns_unavailable(self) -> None:
        result = compare_predator_type_ablation_performance({"variants": "not_a_mapping"})
        self.assertFalse(result["available"])

    def test_variant_not_in_payload_is_skipped(self) -> None:
        # A missing variant has no scenario runs, so predator-type comparisons should
        # only include variants with real multi-predator scenario data.
        payload = {
            "variants": {
                "drop_sensory_cortex": {
                    "suite": {
                        "visual_olfactory_pincer": {"success_rate": 0.5},
                        "olfactory_ambush": {"success_rate": 0.3},
                        "visual_hunter_open_field": {"success_rate": 0.7},
                    }
                }
            }
        }
        result = compare_predator_type_ablation_performance(payload)
        self.assertTrue(result["available"])
        self.assertIn("drop_sensory_cortex", result["comparisons"])
        self.assertNotIn("drop_visual_cortex", result["comparisons"])

    def test_custom_variant_names(self) -> None:
        payload = {
            "variants": {
                "my_custom_variant": {
                    "suite": {
                        "visual_olfactory_pincer": {"success_rate": 0.6},
                        "olfactory_ambush": {"success_rate": 0.4},
                        "visual_hunter_open_field": {"success_rate": 0.8},
                    }
                }
            }
        }
        result = compare_predator_type_ablation_performance(
            payload,
            variant_names=["my_custom_variant"],
        )
        self.assertTrue(result["available"])
        self.assertIn("my_custom_variant", result["comparisons"])

    def test_scenario_groups_always_present_in_result(self) -> None:
        result = compare_predator_type_ablation_performance({})
        self.assertIn("scenario_groups", result)
        self.assertIn("multi_predator_ecology", result["scenario_groups"])

    def test_no_scenarios_present_is_unavailable(self) -> None:
        payload = {
            "variants": {
                "drop_visual_cortex": {
                    "suite": {}
                }
            }
        }
        result = compare_predator_type_ablation_performance(payload)
        self.assertFalse(result["available"])
        self.assertEqual(result["comparisons"], {})

    def test_visual_minus_olfactory_is_rounded_to_six_decimals(self) -> None:
        payload = {
            "variants": {
                "drop_visual_cortex": {
                    "suite": {
                        "visual_olfactory_pincer": {"success_rate": 1 / 3},
                        "visual_hunter_open_field": {"success_rate": 1 / 9},
                        "olfactory_ambush": {"success_rate": 0.0},
                    }
                }
            }
        }
        result = compare_predator_type_ablation_performance(payload)
        diff = result["comparisons"]["drop_visual_cortex"]["visual_minus_olfactory_success_rate"]
        expected = round((((1 / 3) + (1 / 9)) / 2.0) - (((1 / 3) + 0.0) / 2.0), 6)
        self.assertEqual(diff, expected)

    def test_visual_minus_olfactory_is_none_when_a_group_lacks_success_rate_data(self) -> None:
        payload = {
            "variants": {
                "drop_visual_cortex": {
                    "suite": {
                        "visual_hunter_open_field": {"success_rate": 0.25},
                    }
                }
            }
        }

        result = compare_predator_type_ablation_performance(payload)

        comparison = result["comparisons"]["drop_visual_cortex"]
        self.assertEqual(comparison["visual_predator_scenarios"]["scenario_count"], 1)
        self.assertEqual(comparison["olfactory_predator_scenarios"]["scenario_count"], 0)
        self.assertIsNone(comparison["visual_minus_olfactory_success_rate"])

    def test_visual_minus_olfactory_delta_is_none_when_a_group_lacks_delta_data(self) -> None:
        payload = {
            "variants": {
                "drop_visual_cortex": {
                    "suite": {
                        "visual_hunter_open_field": {"success_rate": 0.25},
                        "olfactory_ambush": {"success_rate": 0.5},
                    }
                }
            },
            "deltas_vs_reference": {
                "drop_visual_cortex": {
                    "scenarios": {
                        "visual_hunter_open_field": {"success_rate_delta": -0.3},
                    }
                }
            },
        }

        result = compare_predator_type_ablation_performance(payload)

        comparison = result["comparisons"]["drop_visual_cortex"]
        self.assertEqual(comparison["visual_predator_scenarios"]["scenario_delta_count"], 1)
        self.assertEqual(comparison["olfactory_predator_scenarios"]["scenario_delta_count"], 0)
        self.assertIsNone(comparison["visual_minus_olfactory_success_rate_delta"])

    def test_invalid_success_rate_is_not_counted_in_group_metrics(self) -> None:
        payload = {
            "variants": {
                "drop_visual_cortex": {
                    "suite": {
                        "visual_olfactory_pincer": {"success_rate": "bad"},
                        "visual_hunter_open_field": {"success_rate": 0.25},
                        "olfactory_ambush": {"success_rate": 0.5},
                    }
                }
            }
        }
        result = compare_predator_type_ablation_performance(payload)
        visual_group = result["comparisons"]["drop_visual_cortex"]["visual_predator_scenarios"]
        self.assertEqual(visual_group["scenario_count"], 1)
        self.assertEqual(visual_group["mean_success_rate"], 0.25)

    def test_delta_computed_from_deltas_vs_reference(self) -> None:
        payload = {
            "variants": {
                "drop_visual_cortex": {
                    "suite": {
                        "visual_olfactory_pincer": {"success_rate": 0.2},
                        "visual_hunter_open_field": {"success_rate": 0.1},
                        "olfactory_ambush": {"success_rate": 0.8},
                    }
                }
            },
            "deltas_vs_reference": {
                "drop_visual_cortex": {
                    "scenarios": {
                        "visual_olfactory_pincer": {"success_rate_delta": -0.4},
                        "visual_hunter_open_field": {"success_rate_delta": -0.6},
                        "olfactory_ambush": {"success_rate_delta": 0.1},
                    }
                }
            },
        }
        result = compare_predator_type_ablation_performance(payload)
        comparison = result["comparisons"]["drop_visual_cortex"]
        # Visual group: pincer -0.4, open_field -0.6 → mean = -0.5
        # Olfactory group: pincer -0.4, ambush 0.1 → mean = -0.15
        visual_delta = comparison["visual_predator_scenarios"]["mean_success_rate_delta"]
        olfactory_delta = comparison["olfactory_predator_scenarios"]["mean_success_rate_delta"]
        self.assertAlmostEqual(visual_delta, -0.5, places=5)
        self.assertAlmostEqual(olfactory_delta, -0.15, places=5)

    def test_invalid_success_rate_delta_is_not_counted_in_group_metrics(self) -> None:
        payload = {
            "variants": {
                "drop_visual_cortex": {
                    "suite": {
                        "visual_olfactory_pincer": {"success_rate": 0.2},
                        "visual_hunter_open_field": {"success_rate": 0.1},
                        "olfactory_ambush": {"success_rate": 0.8},
                    }
                }
            },
            "deltas_vs_reference": {
                "drop_visual_cortex": {
                    "scenarios": {
                        "visual_olfactory_pincer": {"success_rate_delta": "bad"},
                        "visual_hunter_open_field": {"success_rate_delta": -0.6},
                        "olfactory_ambush": {"success_rate_delta": 0.1},
                    }
                }
            },
        }
        result = compare_predator_type_ablation_performance(payload)
        comparison = result["comparisons"]["drop_visual_cortex"]
        self.assertAlmostEqual(
            comparison["visual_predator_scenarios"]["mean_success_rate_delta"],
            -0.6,
            places=5,
        )
        self.assertAlmostEqual(
            comparison["olfactory_predator_scenarios"]["mean_success_rate_delta"],
            0.1,
            places=5,
        )

    def test_legacy_scenarios_key_works_when_suite_not_mapping(self) -> None:
        # legacy_scenarios only reached when "suite" is not a Mapping
        payload = {
            "variants": {
                "drop_visual_cortex": {
                    "suite": "not_a_mapping",
                    "legacy_scenarios": {
                        "visual_olfactory_pincer": {"success_rate": 0.3},
                        "visual_hunter_open_field": {"success_rate": 0.5},
                        "olfactory_ambush": {"success_rate": 0.6},
                    }
                }
            }
        }
        result = compare_predator_type_ablation_performance(payload)
        self.assertTrue(result["available"])
        comparison = result["comparisons"]["drop_visual_cortex"]
        # visual scenarios: pincer=0.3, open_field=0.5 → mean=0.4
        self.assertAlmostEqual(
            comparison["visual_predator_scenarios"]["mean_success_rate"], 0.4
        )


class MultiPredatorScenarioConstantsTest(unittest.TestCase):
    """Tests for new constants in ablations.py - MULTI_PREDATOR_SCENARIOS etc."""

    def test_multi_predator_scenarios_contains_three_entries(self) -> None:
        self.assertEqual(len(MULTI_PREDATOR_SCENARIOS), 3)

    def test_visual_predator_scenarios_contains_two_entries(self) -> None:
        self.assertEqual(len(VISUAL_PREDATOR_SCENARIOS), 2)

    def test_olfactory_predator_scenarios_contains_two_entries(self) -> None:
        self.assertEqual(len(OLFACTORY_PREDATOR_SCENARIOS), 2)

    def test_visual_olfactory_pincer_in_all_groups(self) -> None:
        self.assertIn("visual_olfactory_pincer", MULTI_PREDATOR_SCENARIOS)
        self.assertIn("visual_olfactory_pincer", VISUAL_PREDATOR_SCENARIOS)
        self.assertIn("visual_olfactory_pincer", OLFACTORY_PREDATOR_SCENARIOS)

    def test_visual_hunter_open_field_in_visual_not_olfactory(self) -> None:
        self.assertIn("visual_hunter_open_field", VISUAL_PREDATOR_SCENARIOS)
        self.assertNotIn("visual_hunter_open_field", OLFACTORY_PREDATOR_SCENARIOS)

    def test_olfactory_ambush_in_olfactory_not_visual(self) -> None:
        self.assertIn("olfactory_ambush", OLFACTORY_PREDATOR_SCENARIOS)
        self.assertNotIn("olfactory_ambush", VISUAL_PREDATOR_SCENARIOS)

    def test_multi_predator_scenario_groups_has_three_keys(self) -> None:
        self.assertEqual(len(MULTI_PREDATOR_SCENARIO_GROUPS), 3)

    def test_multi_predator_scenario_groups_keys(self) -> None:
        keys = set(MULTI_PREDATOR_SCENARIO_GROUPS.keys())
        self.assertIn("multi_predator_ecology", keys)
        self.assertIn("visual_predator_scenarios", keys)
        self.assertIn("olfactory_predator_scenarios", keys)


if __name__ == "__main__":
    unittest.main()
