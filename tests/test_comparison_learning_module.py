"""Tests for spider_cortex_sim.comparison_learning module.

The comparison_learning module is a thin facade introduced in the PR that
patches the _runner module inside learning_comparison with locally-resolved
functions and delegates to compare_learning_evidence.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch


class ComparisonLearningModuleImportTest(unittest.TestCase):
    """Smoke tests verifying the comparison_learning module is importable
    and exposes the expected public names."""

    def test_module_is_importable(self) -> None:
        import spider_cortex_sim.comparison_learning  # noqa: F401

    def test_module_has_all_attribute(self) -> None:
        import spider_cortex_sim.comparison_learning as cl
        self.assertTrue(hasattr(cl, "__all__"))

    def test_module_all_is_list_or_tuple(self) -> None:
        import spider_cortex_sim.comparison_learning as cl
        self.assertIsInstance(cl.__all__, (list, tuple))

    def test_no_private_names_in_all(self) -> None:
        import spider_cortex_sim.comparison_learning as cl
        private = [n for n in cl.__all__ if n.startswith("_")]
        self.assertEqual(private, [])


class ComparisonLearningReExportsTest(unittest.TestCase):
    """Verify that comparison_learning re-exports from learning_comparison submodules."""

    def test_compare_learning_evidence_accessible_via_comparison_learning(
        self,
    ) -> None:
        from spider_cortex_sim.comparison_learning import compare_learning_evidence
        self.assertTrue(callable(compare_learning_evidence))

    def test_build_learning_evidence_deltas_accessible_via_comparison_learning(
        self,
    ) -> None:
        from spider_cortex_sim.comparison_learning import build_learning_evidence_deltas
        self.assertTrue(callable(build_learning_evidence_deltas))

    def test_build_learning_evidence_summary_accessible_via_comparison_learning(
        self,
    ) -> None:
        from spider_cortex_sim.comparison_learning import build_learning_evidence_summary
        self.assertTrue(callable(build_learning_evidence_summary))

    def test_build_distillation_comparison_report_accessible_via_comparison_learning(
        self,
    ) -> None:
        from spider_cortex_sim.comparison_learning import (
            build_distillation_comparison_report,
        )
        self.assertTrue(callable(build_distillation_comparison_report))

    def test_compare_learning_evidence_is_same_as_learning_comparison_runner(
        self,
    ) -> None:
        from spider_cortex_sim.comparison_learning import compare_learning_evidence as cl_fn
        from spider_cortex_sim.learning_comparison.runner import (
            compare_learning_evidence as runner_fn,
        )
        self.assertIs(cl_fn, runner_fn)


class OfflineAnalysisReportModuleTest(unittest.TestCase):
    """Tests for offline_analysis/report.py (re-export module added in PR)."""

    def test_module_is_importable(self) -> None:
        import spider_cortex_sim.offline_analysis.report  # noqa: F401

    def test_build_report_data_is_accessible(self) -> None:
        from spider_cortex_sim.offline_analysis.report import build_report_data
        self.assertTrue(callable(build_report_data))

    def test_write_report_is_accessible(self) -> None:
        from spider_cortex_sim.offline_analysis.report import write_report
        self.assertTrue(callable(write_report))

    def test_module_all_contains_expected_names(self) -> None:
        import spider_cortex_sim.offline_analysis.report as report_mod
        self.assertIn("build_report_data", report_mod.__all__)
        self.assertIn("write_report", report_mod.__all__)

    def test_report_functions_match_underlying_implementation(self) -> None:
        from spider_cortex_sim.offline_analysis.report import (
            build_report_data,
            write_report,
        )
        from spider_cortex_sim.offline_analysis.reporting import (
            build_report_data as orig_build,
            write_report as orig_write,
        )
        self.assertIs(build_report_data, orig_build)
        self.assertIs(write_report, orig_write)


if __name__ == "__main__":
    unittest.main()