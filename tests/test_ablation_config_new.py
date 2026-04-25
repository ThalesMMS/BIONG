"""Tests for new code in spider_cortex_sim.ablation.config module.

Covers functions and structures introduced or changed in the PR:
- architecture_description()
- _arbitration_fields()
- BrainAblationConfig.to_summary() (new fields added)
- BrainAblationConfig.from_summary() (new in ablation subpackage, was inline before)
- default_brain_config() (moved to config.py)
- constants: MULTI_PREDATOR_SCENARIO_GROUPS, etc.
"""

from __future__ import annotations

import unittest

from spider_cortex_sim.ablation.config import (
    A4_FINE_MODULES,
    BrainAblationConfig,
    COARSE_ROLLUP_MODULES,
    MODULE_NAMES,
    MONOLITHIC_POLICY_NAME,
    MULTI_PREDATOR_SCENARIO_GROUPS,
    MULTI_PREDATOR_SCENARIOS,
    OLFACTORY_PREDATOR_SCENARIOS,
    PROPOSAL_SOURCE_NAMES,
    REFLEX_MODULE_NAMES,
    TRUE_MONOLITHIC_POLICY_NAME,
    VISUAL_PREDATOR_SCENARIOS,
    _arbitration_fields,
    architecture_description,
    default_brain_config,
)


class ArchitectureDescriptionTest(unittest.TestCase):
    """Tests for architecture_description()."""

    def test_modular_returns_full_modular(self) -> None:
        result = architecture_description("modular")
        self.assertIn("modular", result)
        self.assertIn("arbitration", result)

    def test_monolithic_returns_monolithic_description(self) -> None:
        result = architecture_description("monolithic")
        self.assertIn("monolithic", result)
        self.assertIn("proposer", result)

    def test_true_monolithic_returns_direct_control(self) -> None:
        result = architecture_description("true_monolithic")
        self.assertIn("monolithic", result)
        self.assertIn("direct", result)

    def test_unknown_falls_back_to_modular_description(self) -> None:
        result = architecture_description("something_else")
        self.assertIn("modular", result)

    def test_returns_string(self) -> None:
        self.assertIsInstance(architecture_description("modular"), str)
        self.assertIsInstance(architecture_description("monolithic"), str)
        self.assertIsInstance(architecture_description("true_monolithic"), str)

    def test_input_coerced_to_str(self) -> None:
        # Should coerce the input and not raise
        result = architecture_description(object())
        self.assertIsInstance(result, str)


class ArbitrationFieldsTest(unittest.TestCase):
    """Tests for _arbitration_fields()."""

    def test_returns_dict(self) -> None:
        result = _arbitration_fields()
        self.assertIsInstance(result, dict)

    def test_contains_all_expected_keys(self) -> None:
        result = _arbitration_fields()
        expected_keys = [
            "use_learned_arbitration",
            "enable_deterministic_guards",
            "enable_food_direction_bias",
            "warm_start_scale",
            "gate_adjustment_bounds",
        ]
        for key in expected_keys:
            self.assertIn(key, result)

    def test_defaults_are_sensible(self) -> None:
        result = _arbitration_fields()
        self.assertTrue(result["use_learned_arbitration"])
        self.assertFalse(result["enable_deterministic_guards"])
        self.assertFalse(result["enable_food_direction_bias"])
        self.assertAlmostEqual(result["warm_start_scale"], 1.0)
        self.assertEqual(result["gate_adjustment_bounds"], (0.5, 1.5))

    def test_overrides_are_applied(self) -> None:
        result = _arbitration_fields(
            use_learned_arbitration=False,
            enable_deterministic_guards=True,
            warm_start_scale=0.5,
        )
        self.assertFalse(result["use_learned_arbitration"])
        self.assertTrue(result["enable_deterministic_guards"])
        self.assertAlmostEqual(result["warm_start_scale"], 0.5)

    def test_gate_adjustment_bounds_can_be_customized(self) -> None:
        result = _arbitration_fields(gate_adjustment_bounds=(0.1, 2.0))
        self.assertEqual(result["gate_adjustment_bounds"], (0.1, 2.0))


class DefaultBrainConfigTest(unittest.TestCase):
    """Tests for default_brain_config() in the ablation.config submodule."""

    def test_returns_brain_ablation_config(self) -> None:
        config = default_brain_config()
        self.assertIsInstance(config, BrainAblationConfig)

    def test_name_is_modular_full(self) -> None:
        config = default_brain_config()
        self.assertEqual(config.name, "modular_full")

    def test_architecture_is_modular(self) -> None:
        config = default_brain_config()
        self.assertEqual(config.architecture, "modular")

    def test_reflexes_enabled(self) -> None:
        config = default_brain_config()
        self.assertTrue(config.enable_reflexes)

    def test_auxiliary_targets_enabled(self) -> None:
        config = default_brain_config()
        self.assertTrue(config.enable_auxiliary_targets)

    def test_coarse_rollup_modules_disabled(self) -> None:
        config = default_brain_config()
        for module in COARSE_ROLLUP_MODULES:
            self.assertIn(module, config.disabled_modules)

    def test_no_recurrent_modules(self) -> None:
        config = default_brain_config()
        self.assertEqual(config.recurrent_modules, ())

    def test_reflex_scale_is_1(self) -> None:
        config = default_brain_config()
        self.assertAlmostEqual(config.reflex_scale, 1.0)

    def test_credit_strategy_is_route_mask(self) -> None:
        config = default_brain_config()
        self.assertEqual(config.credit_strategy, "route_mask")

    def test_module_dropout_custom_value(self) -> None:
        config = default_brain_config(module_dropout=0.1)
        self.assertAlmostEqual(config.module_dropout, 0.1)

    def test_capacity_profile_can_be_passed(self) -> None:
        config = default_brain_config(capacity_profile="current")
        self.assertIsNotNone(config.capacity_profile)


class BrainAblationConfigToSummaryNewFieldsTest(unittest.TestCase):
    """Tests for to_summary() covering fields added in the PR's ablation.config module."""

    def _config(self) -> BrainAblationConfig:
        return default_brain_config()

    def test_to_summary_contains_capacity_profile_name(self) -> None:
        summary = self._config().to_summary()
        self.assertIn("capacity_profile_name", summary)

    def test_to_summary_contains_capacity_profile_version(self) -> None:
        summary = self._config().to_summary()
        self.assertIn("capacity_profile_version", summary)

    def test_to_summary_contains_capacity_scale_factor(self) -> None:
        summary = self._config().to_summary()
        self.assertIn("capacity_scale_factor", summary)

    def test_to_summary_contains_module_hidden_dims(self) -> None:
        summary = self._config().to_summary()
        self.assertIn("module_hidden_dims", summary)
        self.assertIsInstance(summary["module_hidden_dims"], dict)

    def test_to_summary_contains_integration_hidden_dim(self) -> None:
        summary = self._config().to_summary()
        self.assertIn("integration_hidden_dim", summary)
        self.assertIsInstance(summary["integration_hidden_dim"], int)

    def test_to_summary_contains_monolithic_hidden_dim(self) -> None:
        summary = self._config().to_summary()
        self.assertIn("monolithic_hidden_dim", summary)
        self.assertIsInstance(summary["monolithic_hidden_dim"], int)

    def test_to_summary_monolithic_hidden_dim_equals_sum_of_module_dims(self) -> None:
        config = self._config()
        summary = config.to_summary()
        expected = sum(summary["module_hidden_dims"].values())
        self.assertEqual(summary["monolithic_hidden_dim"], expected)

    def test_to_summary_gate_adjustment_bounds_is_list(self) -> None:
        summary = self._config().to_summary()
        self.assertIsInstance(summary["gate_adjustment_bounds"], list)
        self.assertEqual(len(summary["gate_adjustment_bounds"]), 2)

    def test_to_summary_disabled_modules_is_list(self) -> None:
        summary = self._config().to_summary()
        self.assertIsInstance(summary["disabled_modules"], list)

    def test_to_summary_recurrent_modules_is_list(self) -> None:
        summary = self._config().to_summary()
        self.assertIsInstance(summary["recurrent_modules"], list)

    def test_to_summary_architecture_description_is_non_empty_string(self) -> None:
        summary = self._config().to_summary()
        self.assertIn("architecture_description", summary)
        self.assertIsInstance(summary["architecture_description"], str)
        self.assertTrue(len(summary["architecture_description"]) > 0)


class BrainAblationConfigFromSummaryTest(unittest.TestCase):
    """Tests for BrainAblationConfig.from_summary() (extracted to ablation.config)."""

    def test_from_to_summary_roundtrip(self) -> None:
        original = default_brain_config()
        summary = original.to_summary()
        reconstructed = BrainAblationConfig.from_summary(summary)
        self.assertEqual(reconstructed.name, original.name)
        self.assertEqual(reconstructed.architecture, original.architecture)
        self.assertAlmostEqual(reconstructed.module_dropout, original.module_dropout)
        self.assertEqual(reconstructed.credit_strategy, original.credit_strategy)
        self.assertEqual(reconstructed.disabled_modules, original.disabled_modules)

    def test_from_summary_raises_for_non_mapping(self) -> None:
        with self.assertRaises(ValueError):
            BrainAblationConfig.from_summary("not a mapping")  # type: ignore[arg-type]

    def test_from_summary_with_empty_dict_uses_defaults(self) -> None:
        config = BrainAblationConfig.from_summary({})
        self.assertEqual(config.name, "custom")
        self.assertEqual(config.architecture, "modular")

    def test_from_summary_gate_adjustment_bounds_defaults_for_string(self) -> None:
        config = BrainAblationConfig.from_summary(
            {"gate_adjustment_bounds": "invalid"}
        )
        self.assertEqual(config.gate_adjustment_bounds, (0.5, 1.5))

    def test_from_summary_capacity_profile_name_fallback(self) -> None:
        config = BrainAblationConfig.from_summary({"capacity_profile": "current"})
        self.assertEqual(config.capacity_profile_name, "current")

    def test_from_summary_capacity_profile_name_takes_priority(self) -> None:
        config = BrainAblationConfig.from_summary(
            {"capacity_profile_name": "current", "capacity_profile": "current"}
        )
        self.assertEqual(config.capacity_profile_name, "current")


class MultiPredatorConstantsTest(unittest.TestCase):
    """Tests for the multi-predator scenario constants in ablation.config."""

    def test_multi_predator_scenarios_is_tuple(self) -> None:
        self.assertIsInstance(MULTI_PREDATOR_SCENARIOS, tuple)

    def test_multi_predator_scenarios_non_empty(self) -> None:
        self.assertGreater(len(MULTI_PREDATOR_SCENARIOS), 0)

    def test_visual_predator_scenarios_is_subset_of_multi_predator(self) -> None:
        for scenario in VISUAL_PREDATOR_SCENARIOS:
            self.assertIn(scenario, MULTI_PREDATOR_SCENARIOS)

    def test_olfactory_predator_scenarios_is_subset_of_multi_predator(self) -> None:
        for scenario in OLFACTORY_PREDATOR_SCENARIOS:
            self.assertIn(scenario, MULTI_PREDATOR_SCENARIOS)

    def test_multi_predator_scenario_groups_has_three_keys(self) -> None:
        self.assertEqual(len(MULTI_PREDATOR_SCENARIO_GROUPS), 3)

    def test_multi_predator_scenario_groups_contains_expected_keys(self) -> None:
        self.assertIn("multi_predator_ecology", MULTI_PREDATOR_SCENARIO_GROUPS)
        self.assertIn("visual_predator_scenarios", MULTI_PREDATOR_SCENARIO_GROUPS)
        self.assertIn("olfactory_predator_scenarios", MULTI_PREDATOR_SCENARIO_GROUPS)

    def test_reflex_module_names_is_subset_of_module_names(self) -> None:
        for name in REFLEX_MODULE_NAMES:
            self.assertIn(name, MODULE_NAMES)

    def test_proposal_source_names_includes_all_module_names(self) -> None:
        for name in MODULE_NAMES:
            self.assertIn(name, PROPOSAL_SOURCE_NAMES)


if __name__ == "__main__":
    unittest.main()
