from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from spider_cortex_sim.comparison_capacity import compare_capacity_sweep
from spider_cortex_sim.offline_analysis.extractors import extract_capacity_sweeps
from spider_cortex_sim.offline_analysis.report_data import build_report_data
from spider_cortex_sim.offline_analysis.report import write_report
from spider_cortex_sim.offline_analysis.tables import build_capacity_sweep_tables


class OfflineCapacitySweepsTest(unittest.TestCase):
    def _summary(self) -> dict[str, object]:
        payload, _ = compare_capacity_sweep(
            episodes=0,
            evaluation_episodes=0,
            names=("night_rest",),
            seeds=(7,),
        )
        return {
            "config": {
                "brain": {
                    "name": "modular_full",
                    "architecture": "modular",
                }
            },
            "behavior_evaluation": {
                "capacity_sweeps": payload,
            },
        }

    def test_extract_capacity_sweeps_builds_rows_and_interpretations(self) -> None:
        result = extract_capacity_sweeps(self._summary())

        self.assertTrue(result["available"])
        self.assertTrue(result["rows"])
        self.assertTrue(result["interpretations"])
        self.assertEqual(result["profiles"], ["current", "large", "medium", "small"])

    def test_build_capacity_sweep_tables_includes_guidance_and_matrix(self) -> None:
        tables = build_capacity_sweep_tables(self._summary())

        self.assertTrue(tables["available"])
        self.assertTrue(tables["curves"]["rows"])
        self.assertIn("scenario_success_rate", tables["matrices"])
        self.assertIn("interpretation_guidance", tables["metadata"])

    def test_write_report_emits_capacity_comparison_svg_when_data_exists(self) -> None:
        report = build_report_data(
            summary=self._summary(),
            trace=[],
            behavior_rows=[],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            outputs = write_report(tmpdir, report)
            capacity_svg = Path(outputs["capacity_comparison_svg"])
            report_md = (Path(tmpdir) / "report.md").read_text(encoding="utf-8")
            self.assertTrue(capacity_svg.exists())
            self.assertIn("## Capacity Sweep", report_md)
            self.assertIn("capacity_comparison.svg", report_md)

    def test_build_report_data_uses_capacity_sweep_parameter_counts_as_fallback(self) -> None:
        report = build_report_data(
            summary=self._summary(),
            trace=[],
            behavior_rows=[],
        )

        self.assertTrue(report["model_capacity"]["available"])
        self.assertEqual(report["model_capacity"]["artifact_state"], "available")
        self.assertGreater(report["model_capacity"]["total_trainable"], 0)
        self.assertTrue(report["parameter_counts"])
        self.assertTrue(report["model_capacity"]["networks"])


if __name__ == "__main__":
    unittest.main()
