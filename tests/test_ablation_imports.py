"""Tests for import compatibility and re-export structure.

Covers:
- spider_cortex_sim/ablations.py: backward-compatible re-export via ablation package
- spider_cortex_sim/ablation/__init__.py: exports from config, catalog, predator_metrics
- spider_cortex_sim/ablation/scenario_groups.py: re-export module
- spider_cortex_sim/interfaces.py: passthrough re-export from interface_contracts
"""

from __future__ import annotations

import unittest


class AblationsBackwardCompatibilityTest(unittest.TestCase):
    """Tests that the old spider_cortex_sim.ablations public API still works via re-export."""

    def test_import_brain_ablation_config_from_ablations(self) -> None:
        from spider_cortex_sim.ablations import BrainAblationConfig
        self.assertIsNotNone(BrainAblationConfig)

    def test_import_canonical_ablation_configs_from_ablations(self) -> None:
        from spider_cortex_sim.ablations import canonical_ablation_configs
        self.assertIsNotNone(canonical_ablation_configs)

    def test_import_canonical_ablation_variant_names_from_ablations(self) -> None:
        from spider_cortex_sim.ablations import canonical_ablation_variant_names
        self.assertIsNotNone(canonical_ablation_variant_names)

    def test_import_resolve_ablation_configs_from_ablations(self) -> None:
        from spider_cortex_sim.ablations import resolve_ablation_configs
        self.assertIsNotNone(resolve_ablation_configs)

    def test_import_default_brain_config_from_ablations(self) -> None:
        from spider_cortex_sim.ablations import default_brain_config
        self.assertIsNotNone(default_brain_config)

    def test_import_module_names_from_ablations(self) -> None:
        from spider_cortex_sim.ablations import MODULE_NAMES
        self.assertIsInstance(MODULE_NAMES, tuple)

    def test_import_coarse_rollup_modules_from_ablations(self) -> None:
        from spider_cortex_sim.ablations import COARSE_ROLLUP_MODULES
        self.assertIsInstance(COARSE_ROLLUP_MODULES, tuple)

    def test_import_a4_fine_modules_from_ablations(self) -> None:
        from spider_cortex_sim.ablations import A4_FINE_MODULES
        self.assertIsInstance(A4_FINE_MODULES, tuple)

    def test_import_monolithic_policy_name_from_ablations(self) -> None:
        from spider_cortex_sim.ablations import MONOLITHIC_POLICY_NAME
        self.assertEqual(MONOLITHIC_POLICY_NAME, "monolithic_policy")

    def test_import_true_monolithic_policy_name_from_ablations(self) -> None:
        from spider_cortex_sim.ablations import TRUE_MONOLITHIC_POLICY_NAME
        self.assertEqual(TRUE_MONOLITHIC_POLICY_NAME, "true_monolithic_policy")

    def test_import_multi_predator_scenario_groups_from_ablations(self) -> None:
        from spider_cortex_sim.ablations import MULTI_PREDATOR_SCENARIO_GROUPS
        self.assertIsInstance(MULTI_PREDATOR_SCENARIO_GROUPS, dict)

    def test_import_compare_predator_type_ablation_performance_from_ablations(self) -> None:
        from spider_cortex_sim.ablations import compare_predator_type_ablation_performance
        self.assertIsNotNone(compare_predator_type_ablation_performance)

    def test_import_canonical_ablation_scenario_groups_from_ablations(self) -> None:
        from spider_cortex_sim.ablations import canonical_ablation_scenario_groups
        self.assertIsNotNone(canonical_ablation_scenario_groups)

    def test_import_resolve_ablation_scenario_group_from_ablations(self) -> None:
        from spider_cortex_sim.ablations import resolve_ablation_scenario_group
        self.assertIsNotNone(resolve_ablation_scenario_group)

    def test_import_architecture_description_from_ablations(self) -> None:
        from spider_cortex_sim.ablations import architecture_description
        self.assertIsNotNone(architecture_description)

    def test_ablations_module_functions_are_functional(self) -> None:
        """Smoke test: ensure imported functions actually run."""
        from spider_cortex_sim.ablations import (
            BrainAblationConfig,
            canonical_ablation_configs,
            default_brain_config,
        )
        config = default_brain_config()
        self.assertIsInstance(config, BrainAblationConfig)
        registry = canonical_ablation_configs()
        self.assertIn("modular_full", registry)


class AblationPackageInitTest(unittest.TestCase):
    """Tests that spider_cortex_sim.ablation package __init__ exports work correctly."""

    def test_brain_ablation_config_accessible_from_package(self) -> None:
        from spider_cortex_sim.ablation import BrainAblationConfig
        self.assertIsNotNone(BrainAblationConfig)

    def test_canonical_ablation_configs_accessible_from_package(self) -> None:
        from spider_cortex_sim.ablation import canonical_ablation_configs
        self.assertIsNotNone(canonical_ablation_configs)

    def test_compare_predator_type_accessible_from_package(self) -> None:
        from spider_cortex_sim.ablation import compare_predator_type_ablation_performance
        self.assertIsNotNone(compare_predator_type_ablation_performance)

    def test_module_names_accessible_from_package(self) -> None:
        from spider_cortex_sim.ablation import MODULE_NAMES
        self.assertIsInstance(MODULE_NAMES, tuple)

    def test_default_brain_config_accessible_from_package(self) -> None:
        from spider_cortex_sim.ablation import default_brain_config
        self.assertIsNotNone(default_brain_config)


class AblationScenarioGroupsModuleTest(unittest.TestCase):
    """Tests that spider_cortex_sim.ablation.scenario_groups exports correctly."""

    def test_multi_predator_scenario_groups_accessible(self) -> None:
        from spider_cortex_sim.ablation.scenario_groups import MULTI_PREDATOR_SCENARIO_GROUPS
        self.assertIsInstance(MULTI_PREDATOR_SCENARIO_GROUPS, dict)

    def test_multi_predator_scenarios_accessible(self) -> None:
        from spider_cortex_sim.ablation.scenario_groups import MULTI_PREDATOR_SCENARIOS
        self.assertIsInstance(MULTI_PREDATOR_SCENARIOS, tuple)

    def test_visual_predator_scenarios_accessible(self) -> None:
        from spider_cortex_sim.ablation.scenario_groups import VISUAL_PREDATOR_SCENARIOS
        self.assertIsInstance(VISUAL_PREDATOR_SCENARIOS, tuple)

    def test_olfactory_predator_scenarios_accessible(self) -> None:
        from spider_cortex_sim.ablation.scenario_groups import OLFACTORY_PREDATOR_SCENARIOS
        self.assertIsInstance(OLFACTORY_PREDATOR_SCENARIOS, tuple)

    def test_canonical_ablation_scenario_groups_accessible(self) -> None:
        from spider_cortex_sim.ablation.scenario_groups import canonical_ablation_scenario_groups
        self.assertIsNotNone(canonical_ablation_scenario_groups)

    def test_resolve_ablation_scenario_group_accessible(self) -> None:
        from spider_cortex_sim.ablation.scenario_groups import resolve_ablation_scenario_group
        self.assertIsNotNone(resolve_ablation_scenario_group)

    def test_scenario_groups_module_all_contains_expected_names(self) -> None:
        import spider_cortex_sim.ablation.scenario_groups as sg
        expected_exports = {
            "canonical_ablation_scenario_groups",
            "MULTI_PREDATOR_SCENARIO_GROUPS",
            "MULTI_PREDATOR_SCENARIOS",
            "OLFACTORY_PREDATOR_SCENARIOS",
            "resolve_ablation_scenario_group",
            "VISUAL_PREDATOR_SCENARIOS",
        }
        for name in expected_exports:
            self.assertIn(name, sg.__all__, msg=f"'{name}' not in __all__")


class InterfacesPassthroughTest(unittest.TestCase):
    """Tests that spider_cortex_sim.interfaces re-exports from interface_contracts."""

    def test_module_interfaces_accessible_from_interfaces(self) -> None:
        from spider_cortex_sim.interfaces import MODULE_INTERFACES
        self.assertIsNotNone(MODULE_INTERFACES)
        self.assertGreater(len(MODULE_INTERFACES), 0)

    def test_locomotion_actions_accessible_from_interfaces(self) -> None:
        from spider_cortex_sim.interfaces import LOCOMOTION_ACTIONS
        self.assertIsInstance(LOCOMOTION_ACTIONS, tuple)
        self.assertGreater(len(LOCOMOTION_ACTIONS), 0)

    def test_action_to_index_accessible_from_interfaces(self) -> None:
        from spider_cortex_sim.interfaces import ACTION_TO_INDEX
        self.assertIsInstance(ACTION_TO_INDEX, dict)

    def test_module_interface_by_name_accessible_from_interfaces(self) -> None:
        from spider_cortex_sim.interfaces import MODULE_INTERFACE_BY_NAME
        self.assertIsInstance(MODULE_INTERFACE_BY_NAME, dict)

    def test_interfaces_and_interface_contracts_are_consistent(self) -> None:
        """Ensure re-exports match between interfaces.py and interface_contracts package."""
        import spider_cortex_sim.interfaces as iface
        import spider_cortex_sim.interface_contracts as contracts

        # MODULE_INTERFACES should be the same object
        self.assertIs(iface.MODULE_INTERFACES, contracts.MODULE_INTERFACES)


class InterfaceContractsPackageTest(unittest.TestCase):
    """Tests that spider_cortex_sim.interface_contracts package exports work correctly."""

    def test_locomotion_actions_tuple(self) -> None:
        from spider_cortex_sim.interface_contracts import LOCOMOTION_ACTIONS
        self.assertIsInstance(LOCOMOTION_ACTIONS, tuple)
        expected_actions = {
            "MOVE_UP", "MOVE_DOWN", "MOVE_LEFT", "MOVE_RIGHT", "STAY",
        }
        for action in expected_actions:
            self.assertIn(action, LOCOMOTION_ACTIONS)

    def test_action_to_index_complete(self) -> None:
        from spider_cortex_sim.interface_contracts import ACTION_TO_INDEX, LOCOMOTION_ACTIONS
        self.assertEqual(set(ACTION_TO_INDEX.keys()), set(LOCOMOTION_ACTIONS))

    def test_module_interfaces_non_empty(self) -> None:
        from spider_cortex_sim.interface_contracts import MODULE_INTERFACES
        self.assertGreater(len(MODULE_INTERFACES), 0)

    def test_all_module_interfaces_have_names(self) -> None:
        from spider_cortex_sim.interface_contracts import MODULE_INTERFACES
        for spec in MODULE_INTERFACES:
            self.assertTrue(hasattr(spec, "name"))
            self.assertIsInstance(spec.name, str)
            self.assertTrue(spec.name)

    def test_observation_dims_has_positive_values(self) -> None:
        from spider_cortex_sim.interface_contracts import OBSERVATION_DIMS
        self.assertIsInstance(OBSERVATION_DIMS, dict)
        for key, dim in OBSERVATION_DIMS.items():
            self.assertGreater(dim, 0, msg=f"OBSERVATION_DIMS[{key!r}] = {dim} is not positive")

    def test_interface_contracts_variants_accessible(self) -> None:
        from spider_cortex_sim.interface_contracts.variants import (
            INTERFACE_VARIANTS,
            get_interface_variant,
            get_variant_levels,
        )
        self.assertIsInstance(INTERFACE_VARIANTS, dict)
        self.assertIsNotNone(get_interface_variant)
        self.assertIsNotNone(get_variant_levels)


if __name__ == "__main__":
    unittest.main()