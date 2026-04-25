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

class ResolveAblationConfigsTest(unittest.TestCase):
    """Tests for resolve_ablation_configs."""

    def test_none_returns_all_canonical_variants(self) -> None:
        configs = resolve_ablation_configs(None)
        names = [c.name for c in configs]
        self.assertEqual(names, list(canonical_ablation_variant_names()))

    def test_specific_names_returns_correct_order(self) -> None:
        configs = resolve_ablation_configs(
            ["monolithic_policy", "true_monolithic_policy", "modular_full"]
        )
        self.assertEqual(configs[0].name, "monolithic_policy")
        self.assertEqual(configs[1].name, "true_monolithic_policy")
        self.assertEqual(configs[2].name, "modular_full")

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
        self.assertEqual(configs[1].recurrent_modules, A4_FINE_MODULES)

    def test_counterfactual_credit_resolves(self) -> None:
        configs = resolve_ablation_configs(["counterfactual_credit"])
        self.assertEqual(configs[0].name, "counterfactual_credit")
        self.assertTrue(configs[0].uses_counterfactual_credit)

    def test_three_center_credit_variants_resolve(self) -> None:
        configs = resolve_ablation_configs(
            [
                "three_center_modular_local_credit",
                "three_center_modular_counterfactual",
            ]
        )
        self.assertTrue(configs[0].uses_local_credit_only)
        self.assertTrue(configs[1].uses_counterfactual_credit)

    def test_four_center_credit_variants_resolve(self) -> None:
        configs = resolve_ablation_configs(
            [
                "four_center_modular_local_credit",
                "four_center_modular_counterfactual",
            ]
        )
        self.assertTrue(configs[0].uses_local_credit_only)
        self.assertTrue(configs[1].uses_counterfactual_credit)

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
