import csv
import json
import tempfile
import unittest
from unittest import mock
from pathlib import Path

from spider_cortex_sim.ablations import resolve_ablation_configs
from spider_cortex_sim.benchmark_package import (
    BENCHMARK_PACKAGE_CONTENTS,
    BenchmarkManifest,
    _benchmark_package_unified_ladder_payload,
    _get_git_info,
    assemble_benchmark_package,
    summarize_benchmark_manifest,
    _seed_level_rows_from_behavior_rows,
)
from spider_cortex_sim.offline_analysis.constants import LADDER_ADJACENT_COMPARISONS
from spider_cortex_sim.offline_analysis.extractors import (
    compute_modularity_conclusion,
    detect_missing_experiments,
    extract_unified_ladder_report,
)
from spider_cortex_sim.offline_analysis.tables import build_unified_ladder_tables
from spider_cortex_sim.simulation import SpiderSimulation

from tests.fixtures.benchmark_package import BenchmarkPackageFixtures


class BenchmarkPackageManifestTest(BenchmarkPackageFixtures, unittest.TestCase):
    def test_contents_contract_includes_capacity_sweep_outputs(self) -> None:
        self.assertIn("capacity_sweep_tables.json", BENCHMARK_PACKAGE_CONTENTS)
        self.assertIn("plots/capacity_comparison.svg", BENCHMARK_PACKAGE_CONTENTS)

    def test_canonical_contents_names_core_package_files(self) -> None:
        self.assertIn("benchmark_manifest.json", BENCHMARK_PACKAGE_CONTENTS)
        self.assertIn("aggregate_benchmark_tables.json", BENCHMARK_PACKAGE_CONTENTS)
        self.assertIn("ladder_comparison.json", BENCHMARK_PACKAGE_CONTENTS)
        self.assertIn("unified_ladder_report.json", BENCHMARK_PACKAGE_CONTENTS)
        self.assertIn("plots/training_eval.svg", BENCHMARK_PACKAGE_CONTENTS)
        self.assertIn("supporting_csvs/behavior_rows.csv", BENCHMARK_PACKAGE_CONTENTS)

    def test_extract_unified_ladder_report_with_complete_ladder_data(self) -> None:
        result = extract_unified_ladder_report(self._summary())

        self.assertTrue(result["available"])
        self.assertEqual(
            [row["rung"] for row in result["ladder_table"]["rows"]],
            ["A0", "A1", "A2", "A3", "A4"],
        )
        self.assertEqual(len(result["adjacent_comparisons"]), 4)

    def test_extract_unified_ladder_report_with_partial_ladder_data(self) -> None:
        summary = self._summary()
        variants = summary["behavior_evaluation"]["ablations"]["variants"]
        for variant_name in ("three_center_modular", "four_center_modular"):
            variants.pop(variant_name)

        result = extract_unified_ladder_report(summary)

        self.assertTrue(result["available"])
        self.assertTrue(
            any(
                item["experiment_name"] == "complete_architectural_ladder"
                for item in result["missing_experiments"]
            )
        )

    def test_compute_modularity_conclusion_supported(self) -> None:
        result = compute_modularity_conclusion(
            self._conclusion_input(
                delta=0.35,
                ci_lower=0.10,
                ci_upper=0.55,
                cohens_d=0.40,
            )
        )

        self.assertEqual(result["conclusion"], "modularity supported")

    def test_compute_modularity_conclusion_harmful(self) -> None:
        result = compute_modularity_conclusion(
            self._conclusion_input(
                delta=-0.30,
                ci_lower=-0.50,
                ci_upper=-0.10,
                cohens_d=-0.45,
            )
        )

        self.assertEqual(result["conclusion"], "modularity currently harmful")

    def test_compute_modularity_conclusion_confounded(self) -> None:
        result = compute_modularity_conclusion(
            self._conclusion_input(
                ratio=1.35,
                delta=0.35,
                ci_lower=0.10,
                ci_upper=0.55,
                cohens_d=0.40,
            )
        )

        self.assertEqual(result["conclusion"], "capacity/interface confounded")

    def test_compute_modularity_conclusion_inconclusive(self) -> None:
        result = compute_modularity_conclusion(
            self._conclusion_input(
                delta=0.03,
                ci_lower=-0.10,
                ci_upper=0.15,
                cohens_d=0.05,
            )
        )

        self.assertEqual(result["conclusion"], "modularity inconclusive")

    def test_detect_missing_experiments_reports_expected_gaps(self) -> None:
        report = {
            "ladder_table": {
                "rows": [
                    {"rung": "A0", "present": True, "n_seeds": 5},
                    {"rung": "A1", "present": False, "n_seeds": 0},
                    {"rung": "A2", "present": True, "n_seeds": 3},
                    {"rung": "A3", "present": True, "n_seeds": 4},
                    {"rung": "A4", "present": True, "n_seeds": 5},
                ]
            },
            "reward_shaping_sensitivity": {
                "profiles_available": ["classic", "austere"],
            },
            "interface_sufficiency_results": {"available": False},
            "no_reflex_competence": {"claim_result": {}, "rungs": []},
            "credit_assignment_comparison": {
                "strategies_by_rung": {"A2": ["broadcast"], "A3": [], "A4": []}
            },
            "capacity_matched_comparison": {
                "available": True,
                "capacity_matched": False,
            },
        }

        missing = detect_missing_experiments(report)
        experiment_names = {item["experiment_name"] for item in missing}

        self.assertIn("complete_architectural_ladder", experiment_names)
        self.assertIn("cross_profile_ladder_runs", experiment_names)
        self.assertIn("five_seed_ladder_replication", experiment_names)
        self.assertIn("module_interface_sufficiency_suite", experiment_names)
        self.assertIn("no_reflex_competence_validation", experiment_names)
        self.assertIn("credit_assignment_variants", experiment_names)
        self.assertIn("capacity_matched_ladder", experiment_names)

    def test_build_unified_ladder_tables_returns_expected_structure(self) -> None:
        extracted = extract_unified_ladder_report(self._summary())

        result = build_unified_ladder_tables(extracted)

        self.assertTrue(result["available"])
        self.assertIn("rows", result)
        self.assertIn("tables", result)
        self.assertIn("primary_rung_table", result["tables"])
        self.assertIn("adjacent_comparison_table", result["tables"])
        self.assertIn("capacity_summary_table", result["tables"])
        self.assertIn("conclusion_table", result["tables"])
        self.assertIn("missing_experiments_table", result["tables"])

    def test_benchmark_package_unified_ladder_payload_uses_required_top_level_keys(
        self,
    ) -> None:
        extracted = extract_unified_ladder_report(self._summary())
        report = {"unified_ladder_report": build_unified_ladder_tables(extracted)}

        payload = _benchmark_package_unified_ladder_payload(report)

        self.assertEqual(
            set(payload.keys()),
            {
                "rungs",
                "comparisons",
                "capacity_summary",
                "credit_summary",
                "shaping_sensitivity",
                "no_reflex_competence",
                "capability_probes",
                "conclusion",
                "missing_experiments",
                "limitations",
            },
        )
        self.assertTrue(payload["rungs"]["available"])
        self.assertIsInstance(payload["rungs"]["value"], list)
        self.assertIn("conclusion", payload["conclusion"]["value"])
        self.assertIsInstance(payload["limitations"]["value"], list)

    def test_benchmark_package_unified_ladder_payload_does_not_mark_schema_only_table_available(
        self,
    ) -> None:
        report = {
            "unified_ladder_report": {
                "tables": {
                    "primary_rung_table": {"columns": ["rung"], "rows": []},
                    "conclusion_table": {"columns": ["conclusion"], "rows": []},
                },
                "limitations": [],
            }
        }

        payload = _benchmark_package_unified_ladder_payload(report)

        self.assertFalse(payload["rungs"]["available"])
        self.assertFalse(payload["conclusion"]["available"])

    def test_build_unified_ladder_tables_preserves_primary_rung_presence_metadata(
        self,
    ) -> None:
        extracted = extract_unified_ladder_report(self._summary())
        tables = build_unified_ladder_tables(extracted)

        a0_row = tables["rows"]["primary_rung_table"][0]

        self.assertIn("present", a0_row)
        self.assertIn("limitations", a0_row)
        self.assertTrue(a0_row["present"])
        self.assertIsInstance(a0_row["limitations"], list)

    def test_build_unified_ladder_tables_keeps_conclusion_without_overall_available(
        self,
    ) -> None:
        result = build_unified_ladder_tables(
            {
                "available": True,
                "ladder_table": {"rows": []},
                "adjacent_comparisons": [],
                "capacity_matched_comparison": {"available": False},
                "overall_comparison": {"available": False},
                "conclusion": "capacity/interface confounded",
                "conclusion_rationale": "Capacity ratio exceeded threshold.",
                "confounds": ["capacity mismatch"],
                "missing_experiments": [],
                "credit_assignment_comparison": {"available": False},
                "reward_shaping_sensitivity": {"available": False},
                "no_reflex_competence": {"available": False},
                "capability_probe_boundaries": {"available": False},
                "limitations": [],
            }
        )

        self.assertEqual(
            result["tables"]["conclusion_table"]["rows"][0]["conclusion"],
            "capacity/interface confounded",
        )

    def test_build_unified_ladder_tables_preserves_missing_experiments_when_unavailable(
        self,
    ) -> None:
        result = build_unified_ladder_tables(
            {
                "available": False,
                "missing_experiments": [
                    {
                        "experiment_name": "capacity_matched_ladder",
                        "description": "Re-run A0-A4 with matched capacity.",
                        "impact_on_conclusion": "Capacity differences can dominate the ladder outcome.",
                    }
                ],
                "limitations": ["No unified architectural ladder report was available."],
            }
        )

        self.assertEqual(len(result["missing_experiments"]), 1)
        self.assertEqual(
            result["tables"]["missing_experiments_table"]["rows"][0]["experiment"],
            "capacity_matched_ladder",
        )
