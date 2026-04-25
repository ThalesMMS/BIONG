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

    def test_default_brain_config_is_a4_fine_topology(self) -> None:
        config = default_brain_config()
        enabled = set(MODULE_NAMES) - set(config.disabled_modules)
        self.assertEqual(enabled, set(A4_FINE_MODULES))
        self.assertEqual(set(config.disabled_modules), set(COARSE_ROLLUP_MODULES))

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
        self.assertEqual(recurrent_all.recurrent_modules, A4_FINE_MODULES)

    def test_contains_local_credit_only(self) -> None:
        configs = canonical_ablation_configs()
        local_credit = configs["local_credit_only"]
        self.assertTrue(local_credit.is_modular)
        self.assertTrue(local_credit.enable_reflexes)
        self.assertTrue(local_credit.enable_auxiliary_targets)
        self.assertTrue(local_credit.uses_local_credit_only)

    def test_a4_variants_use_fine_topology(self) -> None:
        configs = canonical_ablation_configs()
        for name in (
            "modular_full",
            "no_module_dropout",
            "no_module_reflexes",
            "modular_recurrent",
            "modular_recurrent_all",
            "local_credit_only",
            "counterfactual_credit",
            "constrained_arbitration",
            "weaker_prior_arbitration",
            "minimal_arbitration",
            "fixed_arbitration_baseline",
            "learned_arbitration_no_regularization",
            "reflex_scale_0_25",
            "reflex_scale_0_50",
            "reflex_scale_0_75",
        ):
            with self.subTest(name=name):
                config = configs[name]
                enabled = set(MODULE_NAMES) - set(config.disabled_modules)
                self.assertEqual(enabled, set(A4_FINE_MODULES))
                self.assertEqual(
                    set(config.disabled_modules),
                    set(COARSE_ROLLUP_MODULES),
                )

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

    def test_contains_three_center_modular_local_credit(self) -> None:
        configs = canonical_ablation_configs()
        variant = configs["three_center_modular_local_credit"]
        three_center = configs["three_center_modular"]
        self.assertTrue(variant.is_modular)
        self.assertTrue(variant.uses_local_credit_only)
        self.assertEqual(variant.disabled_modules, three_center.disabled_modules)
        self.assertEqual(variant.recurrent_modules, three_center.recurrent_modules)

    def test_contains_three_center_modular_counterfactual(self) -> None:
        configs = canonical_ablation_configs()
        variant = configs["three_center_modular_counterfactual"]
        three_center = configs["three_center_modular"]
        self.assertTrue(variant.is_modular)
        self.assertTrue(variant.uses_counterfactual_credit)
        self.assertEqual(variant.disabled_modules, three_center.disabled_modules)
        self.assertEqual(variant.recurrent_modules, three_center.recurrent_modules)

    def test_contains_four_center_modular(self) -> None:
        configs = canonical_ablation_configs()
        variant = configs["four_center_modular"]
        self.assertTrue(variant.is_modular)
        self.assertEqual(
            variant.disabled_modules,
            ("alert_center", "hunger_center", "perception_center", "sleep_center"),
        )
        self.assertEqual(variant.recurrent_modules, ())

    def test_contains_four_center_modular_local_credit(self) -> None:
        configs = canonical_ablation_configs()
        variant = configs["four_center_modular_local_credit"]
        four_center = configs["four_center_modular"]
        self.assertTrue(variant.is_modular)
        self.assertTrue(variant.uses_local_credit_only)
        self.assertEqual(variant.disabled_modules, four_center.disabled_modules)
        self.assertEqual(variant.recurrent_modules, four_center.recurrent_modules)

    def test_contains_four_center_modular_counterfactual(self) -> None:
        configs = canonical_ablation_configs()
        variant = configs["four_center_modular_counterfactual"]
        four_center = configs["four_center_modular"]
        self.assertTrue(variant.is_modular)
        self.assertTrue(variant.uses_counterfactual_credit)
        self.assertEqual(variant.disabled_modules, four_center.disabled_modules)
        self.assertEqual(variant.recurrent_modules, four_center.recurrent_modules)

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

    def test_contains_true_monolithic_policy(self) -> None:
        configs = canonical_ablation_configs()
        mono = configs["true_monolithic_policy"]
        self.assertEqual(mono.architecture, "true_monolithic")
        self.assertFalse(mono.enable_reflexes)
        self.assertFalse(mono.enable_auxiliary_targets)
        self.assertFalse(mono.use_learned_arbitration)
        self.assertAlmostEqual(mono.reflex_scale, 0.0)
        self.assertAlmostEqual(mono.module_dropout, 0.0)

    def test_contains_drop_variants_for_a4_fine_modules(self) -> None:
        configs = canonical_ablation_configs()
        for module_name in A4_FINE_MODULES:
            key = f"drop_{module_name}"
            self.assertIn(key, configs, f"Missing drop variant: {key}")
            self.assertEqual(
                set(configs[key].disabled_modules),
                {*COARSE_ROLLUP_MODULES, module_name},
            )
        for module_name in COARSE_ROLLUP_MODULES:
            self.assertNotIn(f"drop_{module_name}", configs)

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

    def test_variant_names_include_three_center_modular(self) -> None:
        names = canonical_ablation_variant_names()
        self.assertIn("three_center_modular", names)

    def test_three_center_modular_config_matches_three_center_topology(self) -> None:
        config = canonical_ablation_configs()["three_center_modular"]
        disabled = set(config.disabled_modules)
        enabled = set(MODULE_NAMES) - disabled
        self.assertEqual(
            disabled,
            {
                "visual_cortex",
                "sensory_cortex",
                "hunger_center",
                "sleep_center",
                "alert_center",
            },
        )
        self.assertEqual(
            enabled,
            {
                "perception_center",
                "homeostasis_center",
                "threat_center",
            },
        )

    def test_variant_names_include_true_monolithic_policy(self) -> None:
        names = canonical_ablation_variant_names()
        self.assertIn("true_monolithic_policy", names)
        self.assertGreater(
            names.index("true_monolithic_policy"),
            names.index("monolithic_policy"),
        )

    def test_variant_names_count(self) -> None:
        """
        Assert the canonical ablation variant name registry contains the expected number of variants.
        
        The test computes the expected total as 24 plus the number of A4 fine modules plus 2 monolithic baselines, and verifies that this equals the length of canonical_ablation_variant_names().
        """
        names = canonical_ablation_variant_names()
        # modular_full + no_module_dropout + no_module_reflexes + three_center_modular
        # + three_center_modular_local_credit + three_center_modular_counterfactual
        # + four_center_modular + four_center_modular_local_credit
        # + four_center_modular_counterfactual
        # + modular_recurrent + modular_recurrent_all
        # + local_credit_only + counterfactual_credit
        # + constrained_arbitration + weaker_prior_arbitration + minimal_arbitration
        # + fixed_arbitration_baseline + learned_arbitration_no_regularization
        # + reflex_scale_0_25/_0_50/_0_75 + drop_ variants + monolithic_policy
        # + true_monolithic_policy
        expected_count = 24 + len(A4_FINE_MODULES) + 2
        self.assertEqual(len(names), expected_count)
        self.assertIn("modular_full_broadcast", names)
        self.assertIn("three_center_modular_broadcast", names)
        self.assertIn("four_center_modular_broadcast", names)

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

    def test_reflex_scale_variants_use_a4_fine_topology(self) -> None:
        configs = canonical_ablation_configs()
        for name in ("reflex_scale_0_25", "reflex_scale_0_50", "reflex_scale_0_75"):
            self.assertEqual(
                set(configs[name].disabled_modules),
                set(COARSE_ROLLUP_MODULES),
            )

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

    def test_true_monolithic_policy_has_reflex_scale_zero(self) -> None:
        configs = canonical_ablation_configs()
        variant = configs["true_monolithic_policy"]
        self.assertAlmostEqual(variant.reflex_scale, 0.0)

    def test_modular_full_has_reflex_scale_one(self) -> None:
        configs = canonical_ablation_configs()
        variant = configs["modular_full"]
        self.assertAlmostEqual(variant.reflex_scale, 1.0)

    def test_drop_variants_have_reflex_scale_one(self) -> None:
        configs = canonical_ablation_configs()
        for module_name in A4_FINE_MODULES:
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
