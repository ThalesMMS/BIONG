"""Tests for the ablation subpackage re-exports and new module structure.

Verifies that:
1. spider_cortex_sim.ablations (facade module) still re-exports everything from
   the new ablation subpackage.
2. spider_cortex_sim.interfaces (facade module) still re-exports everything from
   the new interface_contracts subpackage.
3. spider_cortex_sim.ablation (the subpackage itself) exposes its public names.
4. The ablation/scenario_groups.py module provides the expected symbols.
"""

from __future__ import annotations

import unittest

from spider_cortex_sim.ablations import (
    A4_FINE_MODULES,
    BrainAblationConfig,
    COARSE_ROLLUP_MODULES,
    MODULE_NAMES,
    MONOLITHIC_POLICY_NAME,
    TRUE_MONOLITHIC_POLICY_NAME,
    MULTI_PREDATOR_SCENARIOS,
    MULTI_PREDATOR_SCENARIO_GROUPS,
    VISUAL_PREDATOR_SCENARIOS,
    OLFACTORY_PREDATOR_SCENARIOS,
    PROPOSAL_SOURCE_NAMES,
    REFLEX_MODULE_NAMES,
    canonical_ablation_configs,
    canonical_ablation_scenario_groups,
    canonical_ablation_variant_names,
    compare_predator_type_ablation_performance,
    default_brain_config,
    resolve_ablation_configs,
    resolve_ablation_scenario_group,
    _safe_float,
    _mean,
    _scenario_success_rate,
)
from spider_cortex_sim.interfaces import (
    ALL_INTERFACES,
    ACTION_CONTEXT_INTERFACE,
    ACTION_DELTAS,
    ACTION_TO_INDEX,
    ALERT_CENTER_V1_INTERFACE,
    LOCOMOTION_ACTIONS,
    MODULE_INTERFACES,
    MODULE_INTERFACE_BY_NAME,
    MOTOR_CONTEXT_INTERFACE,
    OBSERVATION_INTERFACE_BY_KEY,
    OBSERVATION_VIEW_BY_KEY,
    ORIENT_HEADINGS,
    ObservationView,
    SignalSpec,
    interface_registry,
    interface_registry_fingerprint,
    architecture_signature,
)


class AblationsFacadeReExportTest(unittest.TestCase):
    """Tests that spider_cortex_sim.ablations re-exports from the ablation subpackage."""

    def test_brain_ablation_config_is_from_ablation_subpackage(self) -> None:
        from spider_cortex_sim.ablation.config import (
            BrainAblationConfig as SubpkgConfig,
        )
        from spider_cortex_sim.ablations import BrainAblationConfig as FacadeConfig
        self.assertIs(FacadeConfig, SubpkgConfig)

    def test_canonical_ablation_configs_is_from_ablation_subpackage(self) -> None:
        from spider_cortex_sim.ablation.catalog import (
            canonical_ablation_configs as SubpkgFn,
        )
        from spider_cortex_sim.ablations import (
            canonical_ablation_configs as FacadeFn,
        )
        self.assertIs(FacadeFn, SubpkgFn)

    def test_compare_predator_type_is_from_ablation_subpackage(self) -> None:
        from spider_cortex_sim.ablation.predator_metrics import (
            compare_predator_type_ablation_performance as SubpkgFn,
        )
        from spider_cortex_sim.ablations import (
            compare_predator_type_ablation_performance as FacadeFn,
        )
        self.assertIs(FacadeFn, SubpkgFn)

    def test_a4_fine_modules_is_tuple(self) -> None:
        self.assertIsInstance(A4_FINE_MODULES, tuple)
        self.assertGreater(len(A4_FINE_MODULES), 0)

    def test_coarse_rollup_modules_is_tuple(self) -> None:
        self.assertIsInstance(COARSE_ROLLUP_MODULES, tuple)
        self.assertGreater(len(COARSE_ROLLUP_MODULES), 0)

    def test_module_names_is_superset_of_fine_modules(self) -> None:
        for name in A4_FINE_MODULES:
            self.assertIn(name, MODULE_NAMES)

    def test_module_names_is_superset_of_coarse_rollup(self) -> None:
        for name in COARSE_ROLLUP_MODULES:
            self.assertIn(name, MODULE_NAMES)

    def test_proposal_source_names_includes_monolithic(self) -> None:
        self.assertIn(MONOLITHIC_POLICY_NAME, PROPOSAL_SOURCE_NAMES)

    def test_proposal_source_names_includes_true_monolithic(self) -> None:
        self.assertIn(TRUE_MONOLITHIC_POLICY_NAME, PROPOSAL_SOURCE_NAMES)

    def test_multi_predator_scenario_groups_has_three_groups(self) -> None:
        self.assertEqual(len(MULTI_PREDATOR_SCENARIO_GROUPS), 3)

    def test_multi_predator_scenarios_are_tuples_of_strings(self) -> None:
        for name in MULTI_PREDATOR_SCENARIOS:
            self.assertIsInstance(name, str)

    def test_resolve_ablation_scenario_group_returns_tuple(self) -> None:
        result = resolve_ablation_scenario_group("multi_predator_ecology")
        self.assertIsInstance(result, tuple)

    def test_resolve_ablation_scenario_group_raises_for_unknown(self) -> None:
        with self.assertRaises(KeyError):
            resolve_ablation_scenario_group("completely_made_up_group")

    def test_canonical_ablation_scenario_groups_matches_multi_predator_groups(
        self,
    ) -> None:
        result = canonical_ablation_scenario_groups()
        self.assertEqual(set(result.keys()), set(MULTI_PREDATOR_SCENARIO_GROUPS.keys()))


class InterfacesFacadeReExportTest(unittest.TestCase):
    """Tests that spider_cortex_sim.interfaces re-exports from interface_contracts."""

    def test_module_interfaces_is_from_interface_contracts(self) -> None:
        from spider_cortex_sim.interface_contracts.module_specs import (
            MODULE_INTERFACES as SubpkgModuleInterfaces,
        )
        self.assertIs(MODULE_INTERFACES, SubpkgModuleInterfaces)

    def test_interface_registry_is_from_interface_contracts(self) -> None:
        from spider_cortex_sim.interface_contracts.registry import (
            interface_registry as SubpkgFn,
        )
        self.assertIs(interface_registry, SubpkgFn)

    def test_observation_view_is_from_interface_contracts(self) -> None:
        from spider_cortex_sim.interface_contracts.observations import (
            ObservationView as SubpkgView,
        )
        self.assertIs(ObservationView, SubpkgView)

    def test_locomotion_actions_is_tuple_of_strings(self) -> None:
        self.assertIsInstance(LOCOMOTION_ACTIONS, tuple)
        for action in LOCOMOTION_ACTIONS:
            self.assertIsInstance(action, str)

    def test_all_interfaces_includes_module_interfaces(self) -> None:
        module_names = {spec.name for spec in MODULE_INTERFACES}
        all_names = {spec.name for spec in ALL_INTERFACES}
        self.assertTrue(module_names.issubset(all_names))

    def test_module_interface_by_name_has_expected_keys(self) -> None:
        for spec in MODULE_INTERFACES:
            self.assertIn(spec.name, MODULE_INTERFACE_BY_NAME)

    def test_observation_view_by_key_is_populated(self) -> None:
        self.assertGreater(len(OBSERVATION_VIEW_BY_KEY), 0)

    def test_observation_interface_by_key_is_populated(self) -> None:
        self.assertGreater(len(OBSERVATION_INTERFACE_BY_KEY), 0)

    def test_action_to_index_maps_all_locomotion_actions(self) -> None:
        for action in LOCOMOTION_ACTIONS:
            self.assertIn(action, ACTION_TO_INDEX)

    def test_action_deltas_contains_movement_and_stay_only(self) -> None:
        self.assertEqual(
            set(ACTION_DELTAS.keys()),
            {"MOVE_UP", "MOVE_DOWN", "MOVE_LEFT", "MOVE_RIGHT", "STAY"},
        )

    def test_alert_center_v1_interface_is_subset_of_full_alert_interface(
        self,
    ) -> None:
        alert_full = MODULE_INTERFACE_BY_NAME["alert_center"]
        for signal_name in ALERT_CENTER_V1_INTERFACE.signal_names:
            self.assertIn(signal_name, alert_full.signal_names)

    def test_interface_registry_is_callable_and_returns_dict(self) -> None:
        result = interface_registry()
        self.assertIsInstance(result, dict)
        self.assertIn("schema_version", result)

    def test_interface_registry_fingerprint_is_hex_string(self) -> None:
        fingerprint = interface_registry_fingerprint()
        self.assertIsInstance(fingerprint, str)
        # SHA-256 hex is 64 chars
        self.assertEqual(len(fingerprint), 64)
        self.assertTrue(all(c in "0123456789abcdef" for c in fingerprint))

    def test_architecture_signature_modular_returns_dict(self) -> None:
        result = architecture_signature(proposal_backend="modular")
        self.assertIsInstance(result, dict)
        self.assertIn("fingerprint", result)
        self.assertIn("proposal_backend", result)
        self.assertEqual(result["proposal_backend"], "modular")

    def test_architecture_signature_true_monolithic_returns_dict(self) -> None:
        result = architecture_signature(proposal_backend="true_monolithic")
        self.assertIn("direct_policy", result)
        self.assertNotIn("arbitration_network", result)

    def test_architecture_signature_monolithic_has_arbitration_network(self) -> None:
        result = architecture_signature(proposal_backend="monolithic")
        self.assertIn("arbitration_network", result)
        self.assertNotIn("direct_policy", result)

    def test_orient_headings_is_non_empty_tuple(self) -> None:
        self.assertIsInstance(ORIENT_HEADINGS, tuple)
        self.assertGreater(len(ORIENT_HEADINGS), 0)


class AblationSubpackageScenarioGroupsTest(unittest.TestCase):
    """Tests for spider_cortex_sim.ablation.scenario_groups module."""

    def test_scenario_groups_module_is_importable(self) -> None:
        import spider_cortex_sim.ablation.scenario_groups  # noqa: F401

    def test_multi_predator_scenario_groups_exported(self) -> None:
        from spider_cortex_sim.ablation.scenario_groups import (
            MULTI_PREDATOR_SCENARIO_GROUPS,
        )
        self.assertIsInstance(MULTI_PREDATOR_SCENARIO_GROUPS, dict)

    def test_visual_predator_scenarios_exported(self) -> None:
        from spider_cortex_sim.ablation.scenario_groups import (
            VISUAL_PREDATOR_SCENARIOS,
        )
        self.assertIsInstance(VISUAL_PREDATOR_SCENARIOS, tuple)

    def test_olfactory_predator_scenarios_exported(self) -> None:
        from spider_cortex_sim.ablation.scenario_groups import (
            OLFACTORY_PREDATOR_SCENARIOS,
        )
        self.assertIsInstance(OLFACTORY_PREDATOR_SCENARIOS, tuple)

    def test_canonical_ablation_scenario_groups_exported(self) -> None:
        from spider_cortex_sim.ablation.scenario_groups import (
            canonical_ablation_scenario_groups,
        )
        self.assertTrue(callable(canonical_ablation_scenario_groups))

    def test_resolve_ablation_scenario_group_exported(self) -> None:
        from spider_cortex_sim.ablation.scenario_groups import (
            resolve_ablation_scenario_group,
        )
        self.assertTrue(callable(resolve_ablation_scenario_group))

    def test_visual_predator_scenarios_is_subset_of_multi_predator(self) -> None:
        from spider_cortex_sim.ablation.scenario_groups import (
            MULTI_PREDATOR_SCENARIOS,
            VISUAL_PREDATOR_SCENARIOS,
        )
        for scenario in VISUAL_PREDATOR_SCENARIOS:
            self.assertIn(scenario, MULTI_PREDATOR_SCENARIOS)

    def test_olfactory_predator_scenarios_is_subset_of_multi_predator(self) -> None:
        from spider_cortex_sim.ablation.scenario_groups import (
            MULTI_PREDATOR_SCENARIOS,
            OLFACTORY_PREDATOR_SCENARIOS,
        )
        for scenario in OLFACTORY_PREDATOR_SCENARIOS:
            self.assertIn(scenario, MULTI_PREDATOR_SCENARIOS)


class AblationSubpackageInitTest(unittest.TestCase):
    """Tests for spider_cortex_sim.ablation __init__.py re-exports."""

    def test_ablation_pkg_imports_brain_ablation_config(self) -> None:
        from spider_cortex_sim.ablation import BrainAblationConfig
        self.assertIsNotNone(BrainAblationConfig)

    def test_ablation_pkg_imports_canonical_ablation_configs(self) -> None:
        from spider_cortex_sim.ablation import canonical_ablation_configs
        self.assertTrue(callable(canonical_ablation_configs))

    def test_ablation_pkg_imports_compare_predator_type_ablation_performance(
        self,
    ) -> None:
        from spider_cortex_sim.ablation import (
            compare_predator_type_ablation_performance,
        )
        self.assertTrue(callable(compare_predator_type_ablation_performance))

    def test_ablation_pkg_has_all_attribute(self) -> None:
        import spider_cortex_sim.ablation as ablation_pkg
        self.assertTrue(hasattr(ablation_pkg, "__all__"))

    def test_ablation_pkg_all_is_non_empty(self) -> None:
        import spider_cortex_sim.ablation as ablation_pkg
        self.assertTrue(len(ablation_pkg.__all__) > 0)


if __name__ == "__main__":
    unittest.main()
