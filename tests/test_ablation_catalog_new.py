"""Tests for spider_cortex_sim.ablation.catalog module.

Covers canonical_ablation_configs, canonical_ablation_variant_names,
resolve_ablation_configs, canonical_ablation_scenario_groups,
and resolve_ablation_scenario_group as introduced in the PR refactor.
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
    BrainAblationConfig,
    MULTI_PREDATOR_SCENARIO_GROUPS,
)


class CanonicalAblationConfigsCatalogTest(unittest.TestCase):
    """Tests for canonical_ablation_configs() in the catalog module."""

    def test_returns_dict(self) -> None:
        result = canonical_ablation_configs()
        self.assertIsInstance(result, dict)

    def test_all_values_are_brain_ablation_config(self) -> None:
        for name, config in canonical_ablation_configs().items():
            self.assertIsInstance(config, BrainAblationConfig, f"Config {name!r} is not BrainAblationConfig")

    def test_modular_full_is_present(self) -> None:
        self.assertIn("modular_full", canonical_ablation_configs())

    def test_monolithic_policy_is_present(self) -> None:
        self.assertIn("monolithic_policy", canonical_ablation_configs())

    def test_true_monolithic_policy_is_present(self) -> None:
        self.assertIn("true_monolithic_policy", canonical_ablation_configs())

    def test_drop_variants_are_present_for_all_a4_fine_modules(self) -> None:
        configs = canonical_ablation_configs()
        for module_name in A4_FINE_MODULES:
            key = f"drop_{module_name}"
            self.assertIn(key, configs, f"Missing drop variant for {module_name!r}")

    def test_no_module_dropout_is_present(self) -> None:
        self.assertIn("no_module_dropout", canonical_ablation_configs())

    def test_no_module_reflexes_is_present(self) -> None:
        self.assertIn("no_module_reflexes", canonical_ablation_configs())

    def test_modular_recurrent_is_present(self) -> None:
        self.assertIn("modular_recurrent", canonical_ablation_configs())

    def test_local_credit_only_is_present(self) -> None:
        self.assertIn("local_credit_only", canonical_ablation_configs())

    def test_counterfactual_credit_is_present(self) -> None:
        self.assertIn("counterfactual_credit", canonical_ablation_configs())

    def test_custom_module_dropout_applied_to_modular_full(self) -> None:
        configs = canonical_ablation_configs(module_dropout=0.1)
        self.assertAlmostEqual(configs["modular_full"].module_dropout, 0.1)

    def test_custom_module_dropout_applied_to_drop_variants(self) -> None:
        configs = canonical_ablation_configs(module_dropout=0.15)
        for module_name in A4_FINE_MODULES:
            key = f"drop_{module_name}"
            self.assertAlmostEqual(
                configs[key].module_dropout, 0.15, msg=f"Dropout mismatch for {key}"
            )

    def test_monolithic_policy_has_monolithic_architecture(self) -> None:
        config = canonical_ablation_configs()["monolithic_policy"]
        self.assertEqual(config.architecture, "monolithic")

    def test_true_monolithic_has_true_monolithic_architecture(self) -> None:
        config = canonical_ablation_configs()["true_monolithic_policy"]
        self.assertEqual(config.architecture, "true_monolithic")

    def test_drop_variant_disabled_modules_include_dropped_module(self) -> None:
        configs = canonical_ablation_configs()
        for module_name in A4_FINE_MODULES:
            key = f"drop_{module_name}"
            self.assertIn(
                module_name,
                configs[key].disabled_modules,
                f"{key} does not disable {module_name!r}",
            )

    def test_four_center_modular_is_present(self) -> None:
        configs = canonical_ablation_configs()
        self.assertIn("four_center_modular", configs)

    def test_three_center_modular_is_present(self) -> None:
        configs = canonical_ablation_configs()
        self.assertIn("three_center_modular", configs)

    def test_constrained_arbitration_enables_deterministic_guards(self) -> None:
        config = canonical_ablation_configs()["constrained_arbitration"]
        self.assertTrue(config.enable_deterministic_guards)
        self.assertTrue(config.enable_food_direction_bias)

    def test_minimal_arbitration_has_zero_warm_start_scale(self) -> None:
        config = canonical_ablation_configs()["minimal_arbitration"]
        self.assertAlmostEqual(config.warm_start_scale, 0.0)

    def test_weaker_prior_arbitration_has_0_5_warm_start(self) -> None:
        config = canonical_ablation_configs()["weaker_prior_arbitration"]
        self.assertAlmostEqual(config.warm_start_scale, 0.5)

    def test_reflex_scale_variants_have_correct_scales(self) -> None:
        configs = canonical_ablation_configs()
        self.assertAlmostEqual(configs["reflex_scale_0_25"].reflex_scale, 0.25)
        self.assertAlmostEqual(configs["reflex_scale_0_50"].reflex_scale, 0.50)
        self.assertAlmostEqual(configs["reflex_scale_0_75"].reflex_scale, 0.75)

    def test_fixed_arbitration_baseline_has_learned_arbitration_false(self) -> None:
        config = canonical_ablation_configs()["fixed_arbitration_baseline"]
        self.assertFalse(config.use_learned_arbitration)


class CanonicalAblationVariantNamesTest(unittest.TestCase):
    """Tests for canonical_ablation_variant_names()."""

    def test_returns_tuple(self) -> None:
        result = canonical_ablation_variant_names()
        self.assertIsInstance(result, tuple)

    def test_first_element_is_modular_full(self) -> None:
        names = canonical_ablation_variant_names()
        self.assertEqual(names[0], "modular_full")

    def test_monolithic_variants_are_last(self) -> None:
        names = canonical_ablation_variant_names()
        # Last two entries should be monolithic variants
        last_two = set(names[-2:])
        self.assertIn("monolithic_policy", last_two | set(names))

    def test_all_a4_drop_variants_present(self) -> None:
        names = canonical_ablation_variant_names()
        for module_name in A4_FINE_MODULES:
            self.assertIn(f"drop_{module_name}", names)

    def test_names_match_canonical_configs_keys(self) -> None:
        names = canonical_ablation_variant_names()
        config_keys = list(canonical_ablation_configs().keys())
        self.assertEqual(list(names), config_keys)

    def test_no_duplicate_names(self) -> None:
        names = canonical_ablation_variant_names()
        self.assertEqual(len(names), len(set(names)))


class ResolveAblationConfigsCatalogTest(unittest.TestCase):
    """Tests for resolve_ablation_configs() in the catalog module."""

    def test_none_returns_all_canonical_configs(self) -> None:
        result = resolve_ablation_configs(None)
        self.assertIsInstance(result, list)
        expected_count = len(canonical_ablation_configs())
        self.assertEqual(len(result), expected_count)

    def test_specific_names_returns_those_configs(self) -> None:
        result = resolve_ablation_configs(["modular_full", "monolithic_policy"])
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].name, "modular_full")
        self.assertEqual(result[1].name, "monolithic_policy")

    def test_unknown_name_raises_key_error(self) -> None:
        with self.assertRaises(KeyError) as ctx:
            resolve_ablation_configs(["nonexistent_variant"])
        self.assertIn("nonexistent_variant", str(ctx.exception))

    def test_error_message_includes_available_variants(self) -> None:
        with self.assertRaises(KeyError) as ctx:
            resolve_ablation_configs(["bad_variant_name"])
        self.assertIn("Available:", str(ctx.exception))

    def test_preserves_order_of_requested_names(self) -> None:
        names = ["counterfactual_credit", "modular_full", "local_credit_only"]
        result = resolve_ablation_configs(names)
        self.assertEqual([c.name for c in result], names)

    def test_single_name_returns_list_of_one(self) -> None:
        result = resolve_ablation_configs(["modular_full"])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].name, "modular_full")

    def test_all_returned_configs_are_brain_ablation_config(self) -> None:
        result = resolve_ablation_configs(["modular_full", "monolithic_policy"])
        for config in result:
            self.assertIsInstance(config, BrainAblationConfig)

    def test_empty_names_raises_no_error_and_returns_empty(self) -> None:
        result = resolve_ablation_configs([])
        self.assertEqual(result, [])


class CanonicalAblationScenarioGroupsTest(unittest.TestCase):
    """Tests for canonical_ablation_scenario_groups() in catalog module."""

    def test_returns_dict(self) -> None:
        result = canonical_ablation_scenario_groups()
        self.assertIsInstance(result, dict)

    def test_matches_multi_predator_scenario_groups(self) -> None:
        result = canonical_ablation_scenario_groups()
        self.assertEqual(set(result.keys()), set(MULTI_PREDATOR_SCENARIO_GROUPS.keys()))

    def test_returns_copy_not_reference(self) -> None:
        result1 = canonical_ablation_scenario_groups()
        result2 = canonical_ablation_scenario_groups()
        result1["extra_key"] = ()
        self.assertNotIn("extra_key", result2)

    def test_all_values_are_tuples_of_strings(self) -> None:
        for group_name, scenarios in canonical_ablation_scenario_groups().items():
            self.assertIsInstance(scenarios, tuple, f"Group {group_name!r} is not a tuple")
            for scenario in scenarios:
                self.assertIsInstance(scenario, str)


class ResolveAblationScenarioGroupTest(unittest.TestCase):
    """Tests for resolve_ablation_scenario_group() in catalog module."""

    def test_returns_tuple(self) -> None:
        result = resolve_ablation_scenario_group("multi_predator_ecology")
        self.assertIsInstance(result, tuple)

    def test_returns_correct_scenarios_for_multi_predator(self) -> None:
        result = resolve_ablation_scenario_group("multi_predator_ecology")
        from spider_cortex_sim.ablation.config import MULTI_PREDATOR_SCENARIOS
        self.assertEqual(result, MULTI_PREDATOR_SCENARIOS)

    def test_returns_visual_scenarios(self) -> None:
        result = resolve_ablation_scenario_group("visual_predator_scenarios")
        from spider_cortex_sim.ablation.config import VISUAL_PREDATOR_SCENARIOS
        self.assertEqual(result, VISUAL_PREDATOR_SCENARIOS)

    def test_returns_olfactory_scenarios(self) -> None:
        result = resolve_ablation_scenario_group("olfactory_predator_scenarios")
        from spider_cortex_sim.ablation.config import OLFACTORY_PREDATOR_SCENARIOS
        self.assertEqual(result, OLFACTORY_PREDATOR_SCENARIOS)

    def test_unknown_group_raises_key_error(self) -> None:
        with self.assertRaises(KeyError) as ctx:
            resolve_ablation_scenario_group("completely_made_up_group")
        self.assertIn("completely_made_up_group", str(ctx.exception))

    def test_error_message_includes_available_groups(self) -> None:
        with self.assertRaises(KeyError) as ctx:
            resolve_ablation_scenario_group("bad_group")
        self.assertIn("Available:", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()