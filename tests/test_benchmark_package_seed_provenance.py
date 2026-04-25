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


class BenchmarkPackageSeedProvenanceTest(BenchmarkPackageFixtures, unittest.TestCase):
    def test_summarize_benchmark_manifest_counts_manifest_file(self) -> None:
        manifest = BenchmarkManifest(**self._manifest_kwargs())
        summary = summarize_benchmark_manifest(manifest, output_dir=Path("package"))

        self.assertEqual(summary["file_count"], len(manifest.contents) + 1)
        self.assertEqual(summary["git_commit"], "abc123")
        self.assertIsNone(summary["git_tag"])

    def test_summarize_benchmark_manifest_does_not_double_count_manifest_file(
        self,
    ) -> None:
        kwargs = self._manifest_kwargs()
        kwargs["contents"] = [
            *kwargs["contents"],
            {"path": "benchmark_manifest.json", "bytes": 2, "sha256": "def"},
        ]
        manifest = BenchmarkManifest(**kwargs)
        summary = summarize_benchmark_manifest(manifest, output_dir=Path("package"))

        self.assertEqual(summary["file_count"], len(manifest.contents))

    def test_seed_level_rows_from_behavior_rows_aggregates_by_seed_context(self) -> None:
        rows = _seed_level_rows_from_behavior_rows(
            [
                {
                    "simulation_seed": 7,
                    "scenario": "night_rest",
                    "success": "true",
                    "ablation_variant": "modular_full",
                },
                {
                    "simulation_seed": 7,
                    "scenario": "night_rest",
                    "success": "false",
                    "ablation_variant": "modular_full",
                },
                {
                    "simulation_seed": 7,
                    "scenario": "predator_edge",
                    "success": True,
                    "ablation_variant": "modular_full",
                },
            ]
        )

        self.assertEqual(len(rows), 2)
        night_row = next(row for row in rows if row["scenario"] == "night_rest")
        self.assertEqual(night_row["metric_name"], "episode_success_rate")
        self.assertEqual(night_row["seed"], 7)
        self.assertAlmostEqual(night_row["value"], 0.5)

    def test_explicit_behavior_rows_take_precedence_over_behavior_csv_copy(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            behavior_csv = root / "behavior.csv"
            package_dir = root / "package"
            self._write_behavior_csv(behavior_csv)

            assemble_benchmark_package(
                package_dir,
                self._summary(),
                behavior_csv,
                behavior_rows=[
                    {
                        "simulation_seed": 9,
                        "scenario": "night_rest",
                        "success": "false",
                        "condition": "explicit_rows",
                    }
                ],
            )

            with (package_dir / "supporting_csvs" / "behavior_rows.csv").open(
                encoding="utf-8",
                newline="",
            ) as fh:
                copied_rows = list(csv.DictReader(fh))

        self.assertEqual(copied_rows[0]["simulation_seed"], "9")
        self.assertEqual(copied_rows[0]["condition"], "explicit_rows")

    def test_capture_benchmark_provenance_handles_missing_git(self) -> None:
        with mock.patch(
            "spider_cortex_sim.benchmark_package.subprocess.run",
            side_effect=OSError("git unavailable"),
        ):
            git_info = _get_git_info()

        self.assertIsNone(git_info["commit"])
        self.assertIsNone(git_info["tag"])
        self.assertIsNone(git_info["dirty"])
