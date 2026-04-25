from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from spider_cortex_sim.ablations import (
    A4_FINE_MODULES,
    BrainAblationConfig,
    COARSE_ROLLUP_MODULES,
    MODULE_NAMES,
    MULTI_PREDATOR_SCENARIOS,
    MULTI_PREDATOR_SCENARIO_GROUPS,
    VISUAL_PREDATOR_SCENARIOS,
    OLFACTORY_PREDATOR_SCENARIOS,
    MONOLITHIC_POLICY_NAME,
    TRUE_MONOLITHIC_POLICY_NAME,
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

from tests.fixtures.ablations import _blank_mapping, _build_observation, _profile_with_updates

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

    def test_is_true_monolithic_property_true_only_for_true_monolithic(self) -> None:
        config = BrainAblationConfig(name="test", architecture="true_monolithic")
        self.assertTrue(config.is_true_monolithic)
        self.assertFalse(config.is_modular)
        self.assertFalse(config.is_monolithic)

    def test_is_true_monolithic_property_false_for_other_architectures(self) -> None:
        self.assertFalse(
            BrainAblationConfig(name="test", architecture="modular").is_true_monolithic
        )
        self.assertFalse(
            BrainAblationConfig(name="test", architecture="monolithic").is_true_monolithic
        )

    def test_true_monolithic_with_disabled_modules_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            BrainAblationConfig(
                name="test",
                architecture="true_monolithic",
                disabled_modules=("alert_center",),
            )

    def test_true_monolithic_with_recurrent_modules_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            BrainAblationConfig(
                name="test",
                architecture="true_monolithic",
                recurrent_modules=("alert_center",),
            )

    def test_true_monolithic_with_module_reflex_scales_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            BrainAblationConfig(
                name="test",
                architecture="true_monolithic",
                module_reflex_scales={"alert_center": 0.5},
            )

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
        self.assertEqual(
            summary["architecture_description"],
            "full modular with arbitration",
        )

    def test_to_summary_includes_true_monolithic_architecture_description(self) -> None:
        config = BrainAblationConfig(
            name=TRUE_MONOLITHIC_POLICY_NAME,
            architecture="true_monolithic",
            module_dropout=0.0,
            enable_reflexes=False,
            enable_auxiliary_targets=False,
            use_learned_arbitration=False,
            reflex_scale=0.0,
        )
        summary = config.to_summary()
        self.assertIn("architecture_description", summary)
        self.assertEqual(
            summary["architecture_description"],
            "true monolithic direct control",
        )

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

    def test_to_summary_preserves_recurrent_module_order(self) -> None:
        config = BrainAblationConfig(
            name="test",
            recurrent_modules=("sleep_center", "alert_center"),
        )
        summary = config.to_summary()
        self.assertEqual(summary["recurrent_modules"], ["sleep_center", "alert_center"])

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

    def test_module_dropout_outside_probability_range_raises_value_error(self) -> None:
        for value in (-0.1, 1.1):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    BrainAblationConfig(name="test", module_dropout=value)

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
