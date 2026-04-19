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
