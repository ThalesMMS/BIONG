"""Tests for spider_cortex_sim/ablation/config.py

Covers:
- MODULE_NAMES, COARSE_ROLLUP_MODULES, A4_FINE_MODULES constants
- MULTI_PREDATOR_SCENARIO_GROUPS and related scenario constants
- architecture_description() function
- _normalize_module_names() (tested indirectly via BrainAblationConfig)
- _arbitration_fields() helper
- BrainAblationConfig: construction, validation, properties, to_summary/from_summary
- default_brain_config() factory
"""

from __future__ import annotations

import unittest

from spider_cortex_sim.ablation.config import (
    A4_FINE_MODULES,
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
    BrainAblationConfig,
    _arbitration_fields,
    architecture_description,
    default_brain_config,
)


class ModuleConstantsTest(unittest.TestCase):
    """Tests for module-level constants."""

    def test_module_names_is_non_empty_tuple(self) -> None:
        self.assertIsInstance(MODULE_NAMES, tuple)
        self.assertGreater(len(MODULE_NAMES), 0)

    def test_coarse_rollup_modules_are_subset_of_module_names(self) -> None:
        for mod in COARSE_ROLLUP_MODULES:
            self.assertIn(mod, MODULE_NAMES)

    def test_a4_fine_modules_are_subset_of_module_names(self) -> None:
        for mod in A4_FINE_MODULES:
            self.assertIn(mod, MODULE_NAMES)

    def test_a4_fine_modules_are_known(self) -> None:
        expected = {"visual_cortex", "sensory_cortex", "hunger_center", "sleep_center", "alert_center"}
        self.assertEqual(set(A4_FINE_MODULES), expected)

    def test_coarse_rollup_modules_are_known(self) -> None:
        expected = {"perception_center", "homeostasis_center", "threat_center"}
        self.assertEqual(set(COARSE_ROLLUP_MODULES), expected)

    def test_monolithic_policy_name(self) -> None:
        self.assertEqual(MONOLITHIC_POLICY_NAME, "monolithic_policy")

    def test_true_monolithic_policy_name(self) -> None:
        self.assertEqual(TRUE_MONOLITHIC_POLICY_NAME, "true_monolithic_policy")

    def test_reflex_module_names_subset_of_module_names(self) -> None:
        for mod in REFLEX_MODULE_NAMES:
            self.assertIn(mod, MODULE_NAMES)

    def test_proposal_source_names_contains_module_names(self) -> None:
        for name in MODULE_NAMES:
            self.assertIn(name, PROPOSAL_SOURCE_NAMES)
        self.assertIn(MONOLITHIC_POLICY_NAME, PROPOSAL_SOURCE_NAMES)
        self.assertIn(TRUE_MONOLITHIC_POLICY_NAME, PROPOSAL_SOURCE_NAMES)

    def test_multi_predator_scenarios_tuple(self) -> None:
        self.assertIsInstance(MULTI_PREDATOR_SCENARIOS, tuple)
        self.assertIn("visual_olfactory_pincer", MULTI_PREDATOR_SCENARIOS)
        self.assertIn("olfactory_ambush", MULTI_PREDATOR_SCENARIOS)
        self.assertIn("visual_hunter_open_field", MULTI_PREDATOR_SCENARIOS)

    def test_visual_predator_scenarios(self) -> None:
        self.assertIn("visual_olfactory_pincer", VISUAL_PREDATOR_SCENARIOS)
        self.assertIn("visual_hunter_open_field", VISUAL_PREDATOR_SCENARIOS)
        self.assertNotIn("olfactory_ambush", VISUAL_PREDATOR_SCENARIOS)

    def test_olfactory_predator_scenarios(self) -> None:
        self.assertIn("visual_olfactory_pincer", OLFACTORY_PREDATOR_SCENARIOS)
        self.assertIn("olfactory_ambush", OLFACTORY_PREDATOR_SCENARIOS)
        self.assertNotIn("visual_hunter_open_field", OLFACTORY_PREDATOR_SCENARIOS)

    def test_multi_predator_scenario_groups_has_three_groups(self) -> None:
        self.assertIn("multi_predator_ecology", MULTI_PREDATOR_SCENARIO_GROUPS)
        self.assertIn("visual_predator_scenarios", MULTI_PREDATOR_SCENARIO_GROUPS)
        self.assertIn("olfactory_predator_scenarios", MULTI_PREDATOR_SCENARIO_GROUPS)

    def test_multi_predator_scenario_groups_values(self) -> None:
        self.assertEqual(
            MULTI_PREDATOR_SCENARIO_GROUPS["multi_predator_ecology"],
            MULTI_PREDATOR_SCENARIOS,
        )
        self.assertEqual(
            MULTI_PREDATOR_SCENARIO_GROUPS["visual_predator_scenarios"],
            VISUAL_PREDATOR_SCENARIOS,
        )
        self.assertEqual(
            MULTI_PREDATOR_SCENARIO_GROUPS["olfactory_predator_scenarios"],
            OLFACTORY_PREDATOR_SCENARIOS,
        )


class ArchitectureDescriptionTest(unittest.TestCase):
    """Tests for architecture_description()."""

    def test_modular(self) -> None:
        result = architecture_description("modular")
        self.assertEqual(result, "full modular with arbitration")

    def test_monolithic(self) -> None:
        result = architecture_description("monolithic")
        self.assertEqual(result, "monolithic proposer + action/motor pipeline")

    def test_true_monolithic(self) -> None:
        result = architecture_description("true_monolithic")
        self.assertEqual(result, "true monolithic direct control")

    def test_unknown_falls_back_to_modular_description(self) -> None:
        result = architecture_description("anything_else")
        self.assertEqual(result, "full modular with arbitration")

    def test_coerces_to_str(self) -> None:
        # Passing something that converts to a string
        result = architecture_description(42)  # type: ignore[arg-type]
        self.assertEqual(result, "full modular with arbitration")


class ArbitrationFieldsTest(unittest.TestCase):
    """Tests for _arbitration_fields() helper."""

    def test_defaults(self) -> None:
        result = _arbitration_fields()
        self.assertEqual(result["use_learned_arbitration"], True)
        self.assertEqual(result["enable_deterministic_guards"], False)
        self.assertEqual(result["enable_food_direction_bias"], False)
        self.assertEqual(result["warm_start_scale"], 1.0)
        self.assertEqual(result["gate_adjustment_bounds"], (0.5, 1.5))

    def test_overrides(self) -> None:
        result = _arbitration_fields(
            use_learned_arbitration=False,
            enable_deterministic_guards=True,
            enable_food_direction_bias=True,
            warm_start_scale=0.5,
            gate_adjustment_bounds=(0.1, 2.0),
        )
        self.assertEqual(result["use_learned_arbitration"], False)
        self.assertEqual(result["enable_deterministic_guards"], True)
        self.assertEqual(result["enable_food_direction_bias"], True)
        self.assertEqual(result["warm_start_scale"], 0.5)
        self.assertEqual(result["gate_adjustment_bounds"], (0.1, 2.0))

    def test_returns_all_five_keys(self) -> None:
        result = _arbitration_fields()
        expected_keys = {
            "use_learned_arbitration",
            "enable_deterministic_guards",
            "enable_food_direction_bias",
            "warm_start_scale",
            "gate_adjustment_bounds",
        }
        self.assertEqual(set(result.keys()), expected_keys)


class BrainAblationConfigDefaultsTest(unittest.TestCase):
    """Tests for BrainAblationConfig with default values."""

    def setUp(self) -> None:
        self.config = BrainAblationConfig()

    def test_default_architecture_is_modular(self) -> None:
        self.assertEqual(self.config.architecture, "modular")

    def test_default_name(self) -> None:
        self.assertEqual(self.config.name, "custom")

    def test_default_module_dropout(self) -> None:
        self.assertAlmostEqual(self.config.module_dropout, 0.05)

    def test_default_credit_strategy(self) -> None:
        self.assertEqual(self.config.credit_strategy, "broadcast")

    def test_default_reflex_scale(self) -> None:
        self.assertAlmostEqual(self.config.reflex_scale, 1.0)

    def test_default_warm_start_scale(self) -> None:
        self.assertAlmostEqual(self.config.warm_start_scale, 1.0)

    def test_default_gate_adjustment_bounds(self) -> None:
        self.assertEqual(self.config.gate_adjustment_bounds, (0.5, 1.5))

    def test_is_frozen(self) -> None:
        with self.assertRaises((AttributeError, TypeError)):
            self.config.name = "something_else"  # type: ignore[misc]


class BrainAblationConfigArchitectureValidationTest(unittest.TestCase):
    """Tests for architecture validation."""

    def test_modular_architecture_accepted(self) -> None:
        config = BrainAblationConfig(architecture="modular")
        self.assertEqual(config.architecture, "modular")

    def test_monolithic_architecture_accepted(self) -> None:
        config = BrainAblationConfig(architecture="monolithic")
        self.assertEqual(config.architecture, "monolithic")

    def test_true_monolithic_architecture_accepted(self) -> None:
        config = BrainAblationConfig(architecture="true_monolithic")
        self.assertEqual(config.architecture, "true_monolithic")

    def test_invalid_architecture_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            BrainAblationConfig(architecture="invalid_arch")

    def test_architecture_coerced_to_string(self) -> None:
        # modular is valid; ensure string coercion happens
        config = BrainAblationConfig(architecture="modular")
        self.assertIsInstance(config.architecture, str)


class BrainAblationConfigModuleDropoutValidationTest(unittest.TestCase):
    """Tests for module_dropout validation."""

    def test_zero_module_dropout_accepted(self) -> None:
        config = BrainAblationConfig(module_dropout=0.0)
        self.assertAlmostEqual(config.module_dropout, 0.0)

    def test_positive_module_dropout_accepted(self) -> None:
        config = BrainAblationConfig(module_dropout=0.1)
        self.assertAlmostEqual(config.module_dropout, 0.1)

    def test_inf_module_dropout_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            BrainAblationConfig(module_dropout=float("inf"))

    def test_nan_module_dropout_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            BrainAblationConfig(module_dropout=float("nan"))


class BrainAblationConfigWarmStartScaleValidationTest(unittest.TestCase):
    """Tests for warm_start_scale validation."""

    def test_zero_warm_start_scale_accepted(self) -> None:
        config = BrainAblationConfig(warm_start_scale=0.0)
        self.assertAlmostEqual(config.warm_start_scale, 0.0)

    def test_one_warm_start_scale_accepted(self) -> None:
        config = BrainAblationConfig(warm_start_scale=1.0)
        self.assertAlmostEqual(config.warm_start_scale, 1.0)

    def test_mid_warm_start_scale_accepted(self) -> None:
        config = BrainAblationConfig(warm_start_scale=0.5)
        self.assertAlmostEqual(config.warm_start_scale, 0.5)

    def test_above_one_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            BrainAblationConfig(warm_start_scale=1.1)

    def test_below_zero_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            BrainAblationConfig(warm_start_scale=-0.1)

    def test_inf_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            BrainAblationConfig(warm_start_scale=float("inf"))


class BrainAblationConfigGateAdjustmentBoundsValidationTest(unittest.TestCase):
    """Tests for gate_adjustment_bounds validation."""

    def test_valid_bounds_accepted(self) -> None:
        config = BrainAblationConfig(gate_adjustment_bounds=(0.1, 2.0))
        self.assertEqual(config.gate_adjustment_bounds, (0.1, 2.0))

    def test_equal_bounds_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            BrainAblationConfig(gate_adjustment_bounds=(1.0, 1.0))

    def test_reversed_bounds_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            BrainAblationConfig(gate_adjustment_bounds=(2.0, 0.5))

    def test_inf_in_bounds_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            BrainAblationConfig(gate_adjustment_bounds=(0.5, float("inf")))

    def test_nan_in_bounds_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            BrainAblationConfig(gate_adjustment_bounds=(float("nan"), 1.5))

    def test_too_many_values_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            BrainAblationConfig(gate_adjustment_bounds=(0.5, 1.0, 1.5))  # type: ignore[arg-type]


class BrainAblationConfigCreditStrategyValidationTest(unittest.TestCase):
    """Tests for credit_strategy validation."""

    def test_broadcast_accepted(self) -> None:
        config = BrainAblationConfig(credit_strategy="broadcast")
        self.assertEqual(config.credit_strategy, "broadcast")

    def test_local_only_accepted(self) -> None:
        config = BrainAblationConfig(credit_strategy="local_only")
        self.assertEqual(config.credit_strategy, "local_only")

    def test_counterfactual_accepted(self) -> None:
        config = BrainAblationConfig(credit_strategy="counterfactual")
        self.assertEqual(config.credit_strategy, "counterfactual")

    def test_invalid_strategy_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            BrainAblationConfig(credit_strategy="unknown_strategy")


class BrainAblationConfigReflexScaleValidationTest(unittest.TestCase):
    """Tests for reflex_scale validation."""

    def test_zero_reflex_scale_accepted(self) -> None:
        config = BrainAblationConfig(reflex_scale=0.0)
        self.assertAlmostEqual(config.reflex_scale, 0.0)

    def test_one_reflex_scale_accepted(self) -> None:
        config = BrainAblationConfig(reflex_scale=1.0)
        self.assertAlmostEqual(config.reflex_scale, 1.0)

    def test_negative_reflex_scale_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            BrainAblationConfig(reflex_scale=-0.1)

    def test_inf_reflex_scale_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            BrainAblationConfig(reflex_scale=float("inf"))


class BrainAblationConfigDisabledModulesValidationTest(unittest.TestCase):
    """Tests for disabled_modules validation."""

    def test_empty_disabled_modules_accepted(self) -> None:
        config = BrainAblationConfig(disabled_modules=())
        self.assertEqual(config.disabled_modules, ())

    def test_valid_disabled_modules_accepted(self) -> None:
        config = BrainAblationConfig(disabled_modules=COARSE_ROLLUP_MODULES)
        for mod in COARSE_ROLLUP_MODULES:
            self.assertIn(mod, config.disabled_modules)

    def test_disabled_modules_deduplicated_and_sorted(self) -> None:
        # Provide duplicates and unsorted
        config = BrainAblationConfig(
            disabled_modules=("visual_cortex", "sensory_cortex", "visual_cortex")
        )
        # Should be deduplicated and sorted
        self.assertEqual(config.disabled_modules, ("sensory_cortex", "visual_cortex"))

    def test_invalid_disabled_module_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            BrainAblationConfig(disabled_modules=("nonexistent_module",))

    def test_monolithic_with_disabled_modules_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            BrainAblationConfig(
                architecture="monolithic",
                disabled_modules=("visual_cortex",),
            )

    def test_true_monolithic_with_disabled_modules_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            BrainAblationConfig(
                architecture="true_monolithic",
                disabled_modules=("visual_cortex",),
            )


class BrainAblationConfigRecurrentModulesValidationTest(unittest.TestCase):
    """Tests for recurrent_modules validation."""

    def test_empty_recurrent_modules_accepted(self) -> None:
        config = BrainAblationConfig(recurrent_modules=())
        self.assertEqual(config.recurrent_modules, ())

    def test_valid_recurrent_modules_preserves_order(self) -> None:
        # recurrent_modules preserves order (unlike disabled_modules which sorts)
        config = BrainAblationConfig(
            recurrent_modules=("alert_center", "sleep_center", "hunger_center")
        )
        self.assertEqual(
            config.recurrent_modules,
            ("alert_center", "sleep_center", "hunger_center"),
        )

    def test_recurrent_modules_deduplicated_preserving_first_occurrence(self) -> None:
        config = BrainAblationConfig(
            recurrent_modules=("alert_center", "sleep_center", "alert_center")
        )
        self.assertEqual(config.recurrent_modules, ("alert_center", "sleep_center"))

    def test_invalid_recurrent_module_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            BrainAblationConfig(recurrent_modules=("not_a_module",))

    def test_monolithic_with_recurrent_modules_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            BrainAblationConfig(
                architecture="monolithic",
                recurrent_modules=("alert_center",),
            )


class BrainAblationConfigModuleReflexScalesValidationTest(unittest.TestCase):
    """Tests for module_reflex_scales validation."""

    def test_empty_module_reflex_scales_accepted(self) -> None:
        config = BrainAblationConfig(module_reflex_scales={})
        self.assertEqual(config.module_reflex_scales, {})

    def test_valid_module_reflex_scales_accepted(self) -> None:
        # REFLEX_MODULE_NAMES contains proposal modules
        first_reflex_mod = REFLEX_MODULE_NAMES[0]
        config = BrainAblationConfig(module_reflex_scales={first_reflex_mod: 0.5})
        self.assertAlmostEqual(config.module_reflex_scales[first_reflex_mod], 0.5)

    def test_invalid_module_in_reflex_scales_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            BrainAblationConfig(module_reflex_scales={"nonexistent": 1.0})

    def test_negative_reflex_scale_in_module_reflex_scales_raises_value_error(self) -> None:
        first_reflex_mod = REFLEX_MODULE_NAMES[0]
        with self.assertRaises(ValueError):
            BrainAblationConfig(module_reflex_scales={first_reflex_mod: -0.1})

    def test_inf_reflex_scale_in_module_reflex_scales_raises_value_error(self) -> None:
        first_reflex_mod = REFLEX_MODULE_NAMES[0]
        with self.assertRaises(ValueError):
            BrainAblationConfig(module_reflex_scales={first_reflex_mod: float("inf")})

    def test_monolithic_with_module_reflex_scales_raises_value_error(self) -> None:
        first_reflex_mod = REFLEX_MODULE_NAMES[0]
        with self.assertRaises(ValueError):
            BrainAblationConfig(
                architecture="monolithic",
                module_reflex_scales={first_reflex_mod: 1.0},
            )

    def test_module_reflex_scales_sorted_by_key(self) -> None:
        # If multiple keys, they should be sorted
        if len(REFLEX_MODULE_NAMES) >= 2:
            mod1, mod2 = sorted(REFLEX_MODULE_NAMES[:2])
            config = BrainAblationConfig(
                module_reflex_scales={mod2: 0.5, mod1: 0.3}
            )
            keys = list(config.module_reflex_scales.keys())
            self.assertEqual(keys, sorted(keys))


class BrainAblationConfigPropertiesTest(unittest.TestCase):
    """Tests for BrainAblationConfig properties."""

    def test_is_modular_true_for_modular(self) -> None:
        config = BrainAblationConfig(architecture="modular")
        self.assertTrue(config.is_modular)
        self.assertFalse(config.is_monolithic)
        self.assertFalse(config.is_true_monolithic)

    def test_is_monolithic_true_for_monolithic(self) -> None:
        config = BrainAblationConfig(architecture="monolithic")
        self.assertFalse(config.is_modular)
        self.assertTrue(config.is_monolithic)
        self.assertFalse(config.is_true_monolithic)

    def test_is_true_monolithic_true_for_true_monolithic(self) -> None:
        config = BrainAblationConfig(architecture="true_monolithic")
        self.assertFalse(config.is_modular)
        self.assertFalse(config.is_monolithic)
        self.assertTrue(config.is_true_monolithic)

    def test_uses_local_credit_only_true_when_local_only(self) -> None:
        config = BrainAblationConfig(credit_strategy="local_only")
        self.assertTrue(config.uses_local_credit_only)
        self.assertFalse(config.uses_counterfactual_credit)

    def test_uses_counterfactual_credit_true_when_counterfactual(self) -> None:
        config = BrainAblationConfig(credit_strategy="counterfactual")
        self.assertTrue(config.uses_counterfactual_credit)
        self.assertFalse(config.uses_local_credit_only)

    def test_broadcast_uses_neither_local_nor_counterfactual(self) -> None:
        config = BrainAblationConfig(credit_strategy="broadcast")
        self.assertFalse(config.uses_local_credit_only)
        self.assertFalse(config.uses_counterfactual_credit)

    def test_is_recurrent_false_when_no_recurrent_modules(self) -> None:
        config = BrainAblationConfig(recurrent_modules=())
        self.assertFalse(config.is_recurrent)

    def test_is_recurrent_true_when_recurrent_modules_non_empty(self) -> None:
        config = BrainAblationConfig(recurrent_modules=("alert_center",))
        self.assertTrue(config.is_recurrent)

    def test_monolithic_hidden_dim_is_sum_of_module_dims(self) -> None:
        config = BrainAblationConfig()
        expected = sum(config.module_hidden_dims.values())
        self.assertEqual(config.monolithic_hidden_dim, expected)


class BrainAblationConfigToSummaryTest(unittest.TestCase):
    """Tests for BrainAblationConfig.to_summary()."""

    def setUp(self) -> None:
        self.config = BrainAblationConfig(name="test_config")
        self.summary = self.config.to_summary()

    def test_summary_has_expected_keys(self) -> None:
        expected_keys = {
            "name",
            "architecture",
            "architecture_description",
            "module_dropout",
            "enable_reflexes",
            "enable_auxiliary_targets",
            "use_learned_arbitration",
            "enable_deterministic_guards",
            "enable_food_direction_bias",
            "warm_start_scale",
            "gate_adjustment_bounds",
            "credit_strategy",
            "route_mask_threshold",
            "disabled_modules",
            "recurrent_modules",
            "is_recurrent",
            "reflex_scale",
            "module_reflex_scales",
            "capacity_profile_name",
            "capacity_profile",
            "capacity_profile_version",
            "capacity_scale_factor",
            "module_hidden_dims",
            "action_center_hidden_dim",
            "arbitration_hidden_dim",
            "motor_hidden_dim",
            "integration_hidden_dim",
            "monolithic_hidden_dim",
        }
        self.assertEqual(set(self.summary.keys()), expected_keys)

    def test_summary_name_matches(self) -> None:
        self.assertEqual(self.summary["name"], "test_config")

    def test_summary_architecture_matches(self) -> None:
        self.assertEqual(self.summary["architecture"], "modular")

    def test_summary_gate_adjustment_bounds_is_list(self) -> None:
        self.assertIsInstance(self.summary["gate_adjustment_bounds"], list)
        self.assertEqual(len(self.summary["gate_adjustment_bounds"]), 2)

    def test_summary_disabled_modules_is_list(self) -> None:
        self.assertIsInstance(self.summary["disabled_modules"], list)

    def test_summary_recurrent_modules_is_list(self) -> None:
        self.assertIsInstance(self.summary["recurrent_modules"], list)

    def test_summary_module_reflex_scales_is_dict(self) -> None:
        self.assertIsInstance(self.summary["module_reflex_scales"], dict)

    def test_summary_is_recurrent_bool(self) -> None:
        self.assertIsInstance(self.summary["is_recurrent"], bool)

    def test_summary_monolithic_hidden_dim_is_int(self) -> None:
        self.assertIsInstance(self.summary["monolithic_hidden_dim"], int)

    def test_summary_architecture_description_matches_function(self) -> None:
        self.assertEqual(
            self.summary["architecture_description"],
            architecture_description("modular"),
        )


class BrainAblationConfigFromSummaryTest(unittest.TestCase):
    """Tests for BrainAblationConfig.from_summary()."""

    def test_round_trip_via_summary(self) -> None:
        original = BrainAblationConfig(
            name="test_roundtrip",
            architecture="modular",
            module_dropout=0.1,
            credit_strategy="local_only",
            disabled_modules=COARSE_ROLLUP_MODULES,
            reflex_scale=0.5,
        )
        summary = original.to_summary()
        restored = BrainAblationConfig.from_summary(summary)
        self.assertEqual(restored.name, original.name)
        self.assertEqual(restored.architecture, original.architecture)
        self.assertAlmostEqual(restored.module_dropout, original.module_dropout)
        self.assertEqual(restored.credit_strategy, original.credit_strategy)
        self.assertAlmostEqual(restored.reflex_scale, original.reflex_scale)

    def test_from_summary_with_empty_mapping_uses_defaults(self) -> None:
        config = BrainAblationConfig.from_summary({})
        self.assertEqual(config.name, "custom")
        self.assertEqual(config.architecture, "modular")
        self.assertAlmostEqual(config.module_dropout, 0.05)

    def test_from_summary_raises_for_non_mapping(self) -> None:
        with self.assertRaises(ValueError):
            BrainAblationConfig.from_summary([1, 2, 3])  # type: ignore[arg-type]

    def test_from_summary_gate_bounds_string_uses_default(self) -> None:
        config = BrainAblationConfig.from_summary({"gate_adjustment_bounds": "invalid"})
        self.assertEqual(config.gate_adjustment_bounds, (0.5, 1.5))

    def test_from_summary_capacity_profile_name_fallback(self) -> None:
        # Should fall back to "capacity_profile" key
        config = BrainAblationConfig.from_summary({"capacity_profile": "current"})
        self.assertEqual(config.capacity_profile_name, "current")

    def test_from_summary_module_dropout_none_uses_default(self) -> None:
        config = BrainAblationConfig.from_summary({"module_dropout": None})
        self.assertAlmostEqual(config.module_dropout, 0.05)

    def test_from_summary_reflex_scale_none_uses_default(self) -> None:
        config = BrainAblationConfig.from_summary({"reflex_scale": None})
        self.assertAlmostEqual(config.reflex_scale, 1.0)

    def test_from_summary_warm_start_scale_none_uses_default(self) -> None:
        config = BrainAblationConfig.from_summary({"warm_start_scale": None})
        self.assertAlmostEqual(config.warm_start_scale, 1.0)

    def test_from_summary_preserves_disabled_modules(self) -> None:
        summary = {
            "disabled_modules": list(COARSE_ROLLUP_MODULES),
        }
        config = BrainAblationConfig.from_summary(summary)
        for mod in COARSE_ROLLUP_MODULES:
            self.assertIn(mod, config.disabled_modules)

    def test_from_summary_none_name_uses_default(self) -> None:
        config = BrainAblationConfig.from_summary({"name": None})
        self.assertEqual(config.name, "custom")

    def test_from_summary_none_architecture_uses_default(self) -> None:
        config = BrainAblationConfig.from_summary({"architecture": None})
        self.assertEqual(config.architecture, "modular")


class DefaultBrainConfigTest(unittest.TestCase):
    """Tests for default_brain_config() factory function."""

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

    def test_credit_strategy_is_route_mask(self) -> None:
        config = default_brain_config()
        self.assertEqual(config.credit_strategy, "route_mask")

    def test_coarse_rollup_modules_disabled(self) -> None:
        config = default_brain_config()
        for mod in COARSE_ROLLUP_MODULES:
            self.assertIn(mod, config.disabled_modules)

    def test_no_recurrent_modules(self) -> None:
        config = default_brain_config()
        self.assertEqual(config.recurrent_modules, ())

    def test_custom_module_dropout(self) -> None:
        config = default_brain_config(module_dropout=0.1)
        self.assertAlmostEqual(config.module_dropout, 0.1)

    def test_reflex_scale_is_1(self) -> None:
        config = default_brain_config()
        self.assertAlmostEqual(config.reflex_scale, 1.0)

    def test_deterministic_guards_disabled(self) -> None:
        config = default_brain_config()
        self.assertFalse(config.enable_deterministic_guards)

    def test_food_direction_bias_disabled(self) -> None:
        config = default_brain_config()
        self.assertFalse(config.enable_food_direction_bias)

    def test_learned_arbitration_enabled(self) -> None:
        config = default_brain_config()
        self.assertTrue(config.use_learned_arbitration)


class BrainAblationConfigMonolithicTest(unittest.TestCase):
    """Tests specific to monolithic architecture configurations."""

    def test_monolithic_config_is_not_modular(self) -> None:
        config = BrainAblationConfig(architecture="monolithic")
        self.assertFalse(config.is_modular)
        self.assertTrue(config.is_monolithic)

    def test_true_monolithic_config_is_not_modular(self) -> None:
        config = BrainAblationConfig(architecture="true_monolithic")
        self.assertFalse(config.is_modular)
        self.assertFalse(config.is_monolithic)
        self.assertTrue(config.is_true_monolithic)

    def test_monolithic_accepts_empty_disabled_modules(self) -> None:
        config = BrainAblationConfig(architecture="monolithic", disabled_modules=())
        self.assertEqual(config.disabled_modules, ())

    def test_monolithic_empty_module_reflex_scales_accepted(self) -> None:
        config = BrainAblationConfig(architecture="monolithic", module_reflex_scales={})
        self.assertEqual(config.module_reflex_scales, {})


class BrainAblationConfigModuleHiddenDimsTest(unittest.TestCase):
    """Tests for module_hidden_dims validation."""

    def test_invalid_module_in_hidden_dims_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            BrainAblationConfig(module_hidden_dims={"nonexistent_module": 64})

    def test_zero_hidden_dim_raises_value_error(self) -> None:
        first_module = MODULE_NAMES[0]
        with self.assertRaises(ValueError):
            BrainAblationConfig(module_hidden_dims={first_module: 0})

    def test_positive_hidden_dim_accepted(self) -> None:
        first_module = MODULE_NAMES[0]
        config = BrainAblationConfig(module_hidden_dims={first_module: 128})
        self.assertEqual(config.module_hidden_dims[first_module], 128)


if __name__ == "__main__":
    unittest.main()
