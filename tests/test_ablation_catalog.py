"""Tests for spider_cortex_sim/ablation/catalog.py

Covers:
- canonical_ablation_configs(): returns complete registry, variant properties
- canonical_ablation_variant_names(): returns ordered tuple of names
- resolve_ablation_configs(): resolves names, raises on unknown, handles None
- canonical_ablation_scenario_groups(): returns multi-predator group mapping
- resolve_ablation_scenario_group(): resolves single group, raises on unknown
"""

from __future__ import annotations

import unittest

from spider_cortex_sim.ablation.catalog import (
    canonical_ablation_configs,
    canonical_ablation_scenario_groups,
    canonical_ablation_variant_names,
    resolve_ablation_configs,
    resolve_ablation_scenario_group,
)
from spider_cortex_sim.ablation.config import (
    A4_FINE_MODULES,
    COARSE_ROLLUP_MODULES,
    MONOLITHIC_POLICY_NAME,
    MULTI_PREDATOR_SCENARIO_GROUPS,
    TRUE_MONOLITHIC_POLICY_NAME,
    BrainAblationConfig,
)


class CanonicalAblationConfigsTest(unittest.TestCase):
    """Tests for canonical_ablation_configs()."""

    def setUp(self) -> None:
        self.registry = canonical_ablation_configs()

    def test_returns_dict(self) -> None:
        self.assertIsInstance(self.registry, dict)

    def test_registry_is_non_empty(self) -> None:
        self.assertGreater(len(self.registry), 0)

    def test_all_values_are_brain_ablation_configs(self) -> None:
        for name, config in self.registry.items():
            self.assertIsInstance(config, BrainAblationConfig, msg=f"'{name}' is not a BrainAblationConfig")

    def test_modular_full_is_present(self) -> None:
        self.assertIn("modular_full", self.registry)

    def test_monolithic_policy_is_present(self) -> None:
        self.assertIn("monolithic_policy", self.registry)

    def test_true_monolithic_policy_is_present(self) -> None:
        self.assertIn("true_monolithic_policy", self.registry)

    def test_drop_variants_present_for_each_a4_module(self) -> None:
        for module_name in A4_FINE_MODULES:
            key = f"drop_{module_name}"
            self.assertIn(key, self.registry, msg=f"Missing drop variant: {key}")

    def test_no_module_dropout_variant(self) -> None:
        self.assertIn("no_module_dropout", self.registry)
        config = self.registry["no_module_dropout"]
        self.assertAlmostEqual(config.module_dropout, 0.0)

    def test_no_module_reflexes_variant(self) -> None:
        self.assertIn("no_module_reflexes", self.registry)
        config = self.registry["no_module_reflexes"]
        self.assertFalse(config.enable_reflexes)
        self.assertAlmostEqual(config.reflex_scale, 0.0)

    def test_monolithic_policy_config_architecture(self) -> None:
        config = self.registry["monolithic_policy"]
        self.assertEqual(config.architecture, "monolithic")
        self.assertEqual(config.name, MONOLITHIC_POLICY_NAME)

    def test_true_monolithic_policy_config_architecture(self) -> None:
        config = self.registry["true_monolithic_policy"]
        self.assertEqual(config.architecture, "true_monolithic")
        self.assertEqual(config.name, TRUE_MONOLITHIC_POLICY_NAME)

    def test_local_credit_only_variant(self) -> None:
        self.assertIn("local_credit_only", self.registry)
        config = self.registry["local_credit_only"]
        self.assertEqual(config.credit_strategy, "local_only")
        self.assertTrue(config.uses_local_credit_only)

    def test_counterfactual_credit_variant(self) -> None:
        self.assertIn("counterfactual_credit", self.registry)
        config = self.registry["counterfactual_credit"]
        self.assertEqual(config.credit_strategy, "counterfactual")
        self.assertTrue(config.uses_counterfactual_credit)

    def test_three_center_modular_variants(self) -> None:
        for variant in (
            "three_center_modular",
            "three_center_modular_local_credit",
            "three_center_modular_counterfactual",
        ):
            self.assertIn(variant, self.registry)

    def test_four_center_modular_variants(self) -> None:
        for variant in (
            "four_center_modular",
            "four_center_modular_local_credit",
            "four_center_modular_counterfactual",
        ):
            self.assertIn(variant, self.registry)

    def test_recurrent_variants(self) -> None:
        self.assertIn("modular_recurrent", self.registry)
        config = self.registry["modular_recurrent"]
        self.assertTrue(config.is_recurrent)

        self.assertIn("modular_recurrent_all", self.registry)
        config_all = self.registry["modular_recurrent_all"]
        self.assertTrue(config_all.is_recurrent)

    def test_reflex_scale_variants(self) -> None:
        for name, expected_scale in [
            ("reflex_scale_0_25", 0.25),
            ("reflex_scale_0_50", 0.50),
            ("reflex_scale_0_75", 0.75),
        ]:
            self.assertIn(name, self.registry)
            config = self.registry[name]
            self.assertAlmostEqual(config.reflex_scale, expected_scale)

    def test_arbitration_variants_present(self) -> None:
        for variant in (
            "constrained_arbitration",
            "weaker_prior_arbitration",
            "minimal_arbitration",
            "fixed_arbitration_baseline",
            "learned_arbitration_no_regularization",
        ):
            self.assertIn(variant, self.registry)

    def test_constrained_arbitration_has_guards_and_food_bias(self) -> None:
        config = self.registry["constrained_arbitration"]
        self.assertTrue(config.enable_deterministic_guards)
        self.assertTrue(config.enable_food_direction_bias)

    def test_fixed_arbitration_baseline_not_learned(self) -> None:
        config = self.registry["fixed_arbitration_baseline"]
        self.assertFalse(config.use_learned_arbitration)

    def test_weaker_prior_has_warm_start_0_5(self) -> None:
        config = self.registry["weaker_prior_arbitration"]
        self.assertAlmostEqual(config.warm_start_scale, 0.5)

    def test_minimal_arbitration_has_zero_warm_start(self) -> None:
        config = self.registry["minimal_arbitration"]
        self.assertAlmostEqual(config.warm_start_scale, 0.0)

    def test_keys_match_config_names(self) -> None:
        for key, config in self.registry.items():
            self.assertEqual(config.name, key, msg=f"Config name mismatch for key '{key}'")

    def test_custom_module_dropout_applied_to_modular_variants(self) -> None:
        registry = canonical_ablation_configs(module_dropout=0.1)
        for name, config in registry.items():
            if config.is_modular and name != "no_module_dropout":
                self.assertAlmostEqual(
                    config.module_dropout,
                    0.1,
                    msg=f"Expected dropout 0.1 for '{name}', got {config.module_dropout}",
                )

    def test_drop_module_variant_has_module_disabled(self) -> None:
        for module_name in A4_FINE_MODULES:
            key = f"drop_{module_name}"
            config = self.registry[key]
            self.assertIn(
                module_name,
                config.disabled_modules,
                msg=f"Expected {module_name} in disabled_modules for '{key}'",
            )
            # Should also have COARSE_ROLLUP_MODULES disabled
            for coarse in COARSE_ROLLUP_MODULES:
                self.assertIn(coarse, config.disabled_modules)


class CanonicalAblationVariantNamesTest(unittest.TestCase):
    """Tests for canonical_ablation_variant_names()."""

    def setUp(self) -> None:
        self.names = canonical_ablation_variant_names()

    def test_returns_tuple(self) -> None:
        self.assertIsInstance(self.names, tuple)

    def test_starts_with_modular_full(self) -> None:
        self.assertEqual(self.names[0], "modular_full")

    def test_ends_with_monolithic_baselines(self) -> None:
        # Ends with monolithic variants
        self.assertIn("monolithic_policy", self.names)
        self.assertIn("true_monolithic_policy", self.names)

    def test_contains_all_registry_keys(self) -> None:
        registry = canonical_ablation_configs()
        for key in registry.keys():
            self.assertIn(key, self.names)

    def test_length_matches_registry(self) -> None:
        registry = canonical_ablation_configs()
        self.assertEqual(len(self.names), len(registry))

    def test_all_a4_drop_variants_present(self) -> None:
        for module_name in A4_FINE_MODULES:
            self.assertIn(f"drop_{module_name}", self.names)

    def test_order_preserved_as_dict_order(self) -> None:
        registry = canonical_ablation_configs()
        self.assertEqual(self.names, tuple(registry.keys()))


class ResolveAblationConfigsTest(unittest.TestCase):
    """Tests for resolve_ablation_configs()."""

    def test_none_returns_all_canonical_configs_in_order(self) -> None:
        configs = resolve_ablation_configs(None)
        self.assertIsInstance(configs, list)
        all_names = canonical_ablation_variant_names()
        self.assertEqual(len(configs), len(all_names))

    def test_single_name_resolves_correctly(self) -> None:
        configs = resolve_ablation_configs(["modular_full"])
        self.assertEqual(len(configs), 1)
        self.assertEqual(configs[0].name, "modular_full")

    def test_multiple_names_resolve_in_requested_order(self) -> None:
        names = ["monolithic_policy", "modular_full"]
        configs = resolve_ablation_configs(names)
        self.assertEqual(len(configs), 2)
        self.assertEqual(configs[0].name, "monolithic_policy")
        self.assertEqual(configs[1].name, "modular_full")

    def test_unknown_name_raises_key_error(self) -> None:
        with self.assertRaises(KeyError):
            resolve_ablation_configs(["nonexistent_variant"])

    def test_error_message_lists_unknown_names(self) -> None:
        try:
            resolve_ablation_configs(["bad_variant_a", "bad_variant_b"])
        except KeyError as exc:
            self.assertIn("bad_variant_a", str(exc))
            self.assertIn("bad_variant_b", str(exc))
        else:
            self.fail("Expected KeyError was not raised")

    def test_empty_names_returns_empty_list(self) -> None:
        configs = resolve_ablation_configs([])
        self.assertEqual(configs, [])

    def test_custom_module_dropout_applied(self) -> None:
        configs = resolve_ablation_configs(["modular_full"], module_dropout=0.2)
        self.assertAlmostEqual(configs[0].module_dropout, 0.2)

    def test_duplicate_names_resolves_each_occurrence(self) -> None:
        # Duplicate names should resolve to the same config twice
        configs = resolve_ablation_configs(["modular_full", "modular_full"])
        self.assertEqual(len(configs), 2)
        self.assertEqual(configs[0].name, "modular_full")
        self.assertEqual(configs[1].name, "modular_full")


class CanonicalAblationScenarioGroupsTest(unittest.TestCase):
    """Tests for canonical_ablation_scenario_groups()."""

    def setUp(self) -> None:
        self.groups = canonical_ablation_scenario_groups()

    def test_returns_dict(self) -> None:
        self.assertIsInstance(self.groups, dict)

    def test_contains_expected_group_names(self) -> None:
        self.assertIn("multi_predator_ecology", self.groups)
        self.assertIn("visual_predator_scenarios", self.groups)
        self.assertIn("olfactory_predator_scenarios", self.groups)

    def test_values_are_tuples_of_strings(self) -> None:
        for name, scenarios in self.groups.items():
            self.assertIsInstance(scenarios, tuple, msg=f"Group '{name}' value is not a tuple")
            for scenario in scenarios:
                self.assertIsInstance(scenario, str)

    def test_matches_canonical_multi_predator_scenario_groups(self) -> None:
        self.assertEqual(self.groups, dict(MULTI_PREDATOR_SCENARIO_GROUPS))

    def test_returns_copy_not_reference(self) -> None:
        groups1 = canonical_ablation_scenario_groups()
        groups2 = canonical_ablation_scenario_groups()
        self.assertIsNot(groups1, MULTI_PREDATOR_SCENARIO_GROUPS)
        self.assertEqual(groups1, groups2)


class ResolveAblationScenarioGroupTest(unittest.TestCase):
    """Tests for resolve_ablation_scenario_group()."""

    def test_resolves_multi_predator_ecology(self) -> None:
        result = resolve_ablation_scenario_group("multi_predator_ecology")
        self.assertIsInstance(result, tuple)
        self.assertIn("visual_olfactory_pincer", result)
        self.assertIn("olfactory_ambush", result)
        self.assertIn("visual_hunter_open_field", result)

    def test_resolves_visual_predator_scenarios(self) -> None:
        result = resolve_ablation_scenario_group("visual_predator_scenarios")
        self.assertIn("visual_olfactory_pincer", result)
        self.assertIn("visual_hunter_open_field", result)
        self.assertNotIn("olfactory_ambush", result)

    def test_resolves_olfactory_predator_scenarios(self) -> None:
        result = resolve_ablation_scenario_group("olfactory_predator_scenarios")
        self.assertIn("visual_olfactory_pincer", result)
        self.assertIn("olfactory_ambush", result)
        self.assertNotIn("visual_hunter_open_field", result)

    def test_unknown_group_raises_key_error(self) -> None:
        with self.assertRaises(KeyError):
            resolve_ablation_scenario_group("nonexistent_group")

    def test_error_message_contains_unknown_name(self) -> None:
        try:
            resolve_ablation_scenario_group("bad_group")
        except KeyError as exc:
            self.assertIn("bad_group", str(exc))
        else:
            self.fail("Expected KeyError was not raised")

    def test_error_message_lists_available_groups(self) -> None:
        try:
            resolve_ablation_scenario_group("bad_group")
        except KeyError as exc:
            error_text = str(exc)
            self.assertIn("multi_predator_ecology", error_text)
        else:
            self.fail("Expected KeyError was not raised")


if __name__ == "__main__":
    unittest.main()