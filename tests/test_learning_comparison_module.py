"""Tests for spider_cortex_sim.learning_comparison package structure.

Covers the _export() helper introduced in the PR to unify re-exports
across the learning_comparison and interface_contracts subpackages,
and verifies the learning_comparison package re-exports all expected symbols.
"""

from __future__ import annotations

import types
import unittest


class LearningComparisonExportHelperTest(unittest.TestCase):
    """Tests for the _export() helper function in learning_comparison/__init__.py."""

    def test_export_injects_names_into_globals(self) -> None:
        """_export() should inject the module's public names into the caller's globals."""
        import spider_cortex_sim.learning_comparison as lc_pkg

        # The _export function injects submodule names into the package globals.
        # Verify that top-level names from each submodule are accessible.
        from spider_cortex_sim.learning_comparison.distillation import (
            build_distillation_comparison_report,
        )
        self.assertIs(
            getattr(lc_pkg, "build_distillation_comparison_report", None),
            build_distillation_comparison_report,
        )

    def test_export_injects_evidence_names(self) -> None:
        import spider_cortex_sim.learning_comparison as lc_pkg
        from spider_cortex_sim.learning_comparison.evidence import (
            build_learning_evidence_deltas,
            build_learning_evidence_summary,
        )
        self.assertIs(
            getattr(lc_pkg, "build_learning_evidence_deltas", None),
            build_learning_evidence_deltas,
        )
        self.assertIs(
            getattr(lc_pkg, "build_learning_evidence_summary", None),
            build_learning_evidence_summary,
        )

    def test_export_injects_runner_names(self) -> None:
        import spider_cortex_sim.learning_comparison as lc_pkg
        from spider_cortex_sim.learning_comparison.runner import (
            compare_learning_evidence,
        )
        self.assertIs(
            getattr(lc_pkg, "compare_learning_evidence", None),
            compare_learning_evidence,
        )

    def test_package_all_includes_distillation_report(self) -> None:
        import spider_cortex_sim.learning_comparison as lc_pkg
        self.assertIn("build_distillation_comparison_report", lc_pkg.__all__)

    def test_package_all_includes_evidence_functions(self) -> None:
        import spider_cortex_sim.learning_comparison as lc_pkg
        self.assertIn("build_learning_evidence_deltas", lc_pkg.__all__)
        self.assertIn("build_learning_evidence_summary", lc_pkg.__all__)

    def test_package_all_includes_runner_function(self) -> None:
        import spider_cortex_sim.learning_comparison as lc_pkg
        self.assertIn("compare_learning_evidence", lc_pkg.__all__)

    def test_export_returns_list_of_names(self) -> None:
        """_export() returns a list of exported names."""
        import spider_cortex_sim.learning_comparison as lc_pkg

        # The __all__ is constructed from the concatenation of export results.
        self.assertIsInstance(lc_pkg.__all__, list)

    def test_package_all_is_non_empty(self) -> None:
        import spider_cortex_sim.learning_comparison as lc_pkg
        self.assertTrue(len(lc_pkg.__all__) > 0)

    def test_no_private_names_in_all(self) -> None:
        import spider_cortex_sim.learning_comparison as lc_pkg
        private = [name for name in lc_pkg.__all__ if name.startswith("_")]
        self.assertEqual(private, [])


class InterfaceContractsExportHelperTest(unittest.TestCase):
    """Tests for the _export() helper pattern in interface_contracts/__init__.py."""

    def test_interface_contracts_exports_module_interfaces(self) -> None:
        from spider_cortex_sim.interface_contracts import MODULE_INTERFACES
        self.assertIsNotNone(MODULE_INTERFACES)
        self.assertTrue(len(MODULE_INTERFACES) > 0)

    def test_interface_contracts_exports_all_interfaces(self) -> None:
        from spider_cortex_sim.interface_contracts import ALL_INTERFACES
        self.assertIsNotNone(ALL_INTERFACES)
        self.assertTrue(len(ALL_INTERFACES) > 0)

    def test_interface_contracts_exports_observation_views(self) -> None:
        from spider_cortex_sim.interface_contracts import (
            ObservationView,
            VisualObservation,
            SensoryObservation,
            AlertObservation,
            HungerObservation,
            SleepObservation,
        )
        for cls in [ObservationView, VisualObservation, SensoryObservation,
                    AlertObservation, HungerObservation, SleepObservation]:
            self.assertTrue(issubclass(cls, ObservationView))

    def test_interface_contracts_exports_registry_functions(self) -> None:
        from spider_cortex_sim.interface_contracts import (
            interface_registry,
            interface_registry_fingerprint,
            architecture_signature,
        )
        self.assertTrue(callable(interface_registry))
        self.assertTrue(callable(interface_registry_fingerprint))
        self.assertTrue(callable(architecture_signature))

    def test_interface_contracts_exports_locomotion_actions(self) -> None:
        from spider_cortex_sim.interface_contracts import LOCOMOTION_ACTIONS
        self.assertIsNotNone(LOCOMOTION_ACTIONS)
        self.assertTrue(len(LOCOMOTION_ACTIONS) > 0)

    def test_interface_contracts_exports_interface_variants(self) -> None:
        from spider_cortex_sim.interface_contracts import (
            get_interface_variant,
            get_variant_levels,
            is_variant_interface,
            validate_variant_interfaces,
        )
        for fn in [get_interface_variant, get_variant_levels,
                   is_variant_interface, validate_variant_interfaces]:
            self.assertTrue(callable(fn))

    def test_interface_contracts_module_has_all(self) -> None:
        import spider_cortex_sim.interface_contracts as ic_pkg
        self.assertTrue(hasattr(ic_pkg, "__all__"))
        self.assertIsInstance(ic_pkg.__all__, list)


if __name__ == "__main__":
    unittest.main()
