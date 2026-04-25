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


class BenchmarkPackageAssemblyTest(BenchmarkPackageFixtures, unittest.TestCase):
    def test_benchmark_manifest_validates_invariants(self) -> None:
        self.assertEqual(BenchmarkManifest(**self._manifest_kwargs()).seed_count, 2)

        invalid_cases = (
            ("package_version", "", "package_version"),
            ("created_at", "", "created_at"),
            ("command_metadata", [], "command_metadata"),
            ("environment", [], "environment"),
            ("seed_count", -1, "seed_count"),
            ("seed_count", 1.5, "seed_count"),
            ("confidence_level", 1.5, "confidence_level"),
            ("contents", "report.json", "contents"),
            ("contents", ["report.json"], "contents entries"),
            ("limitations", "none", "limitations"),
            ("limitations", [1], "limitations entries"),
        )
        for field_name, value, message in invalid_cases:
            kwargs = self._manifest_kwargs()
            kwargs[field_name] = value
            with self.subTest(field_name=field_name):
                with self.assertRaisesRegex(ValueError, message):
                    BenchmarkManifest(**kwargs)

    def test_assemble_benchmark_package_writes_manifest_and_tables(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            behavior_csv = root / "behavior.csv"
            package_dir = root / "package"
            self._write_behavior_csv(behavior_csv)
            environment = {
                "git_commit": "abc123",
                "git_tag": "v0.1.0",
                "git_dirty": False,
                "python_version": "3.11.0",
                "platform": "linux",
            }

            with mock.patch(
                "spider_cortex_sim.benchmark_package._get_git_info",
                return_value={
                    "commit": "abc123",
                    "tag": "v0.1.0",
                    "dirty": False,
                },
            ), mock.patch(
                "spider_cortex_sim.benchmark_package._get_python_info",
                return_value={
                    "python_version": "3.11.0",
                    "platform": "linux",
                },
            ), mock.patch(
                "spider_cortex_sim.benchmark_package._get_pip_freeze",
                return_value="numpy==2.0.0\npygame-ce==2.5.0\n",
            ):
                manifest = assemble_benchmark_package(
                    package_dir,
                    self._summary(),
                    behavior_csv,
                    command_metadata={"argv": ["--budget-profile", "paper"]},
                )

            self.assertIsInstance(manifest, BenchmarkManifest)
            self.assertTrue((package_dir / "benchmark_manifest.json").exists())
            self.assertTrue((package_dir / "resolved_config.json").exists())
            self.assertTrue((package_dir / "pip_freeze.txt").exists())
            self.assertTrue((package_dir / "seed_level_rows.csv").exists())
            self.assertTrue((package_dir / "aggregate_benchmark_tables.json").exists())
            self.assertTrue((package_dir / "credit_table.json").exists())
            self.assertTrue((package_dir / "capacity_sweep_tables.json").exists())
            self.assertTrue((package_dir / "claim_test_tables.json").exists())
            self.assertTrue((package_dir / "effect_size_tables.json").exists())
            self.assertTrue((package_dir / "ladder_comparison.json").exists())
            self.assertTrue((package_dir / "unified_ladder_report.json").exists())
            self.assertTrue((package_dir / "report.md").exists())
            self.assertTrue((package_dir / "report.json").exists())
            self.assertTrue((package_dir / "plots" / "training_eval.svg").exists())
            self.assertTrue((package_dir / "plots" / "capacity_comparison.svg").exists())
            self.assertTrue((package_dir / "supporting_csvs" / "behavior_rows.csv").exists())
            self.assertTrue((package_dir / "limitations.txt").exists())

            manifest_payload = json.loads(
                (package_dir / "benchmark_manifest.json").read_text(encoding="utf-8")
            )
            aggregate_tables = json.loads(
                (package_dir / "aggregate_benchmark_tables.json").read_text(
                    encoding="utf-8"
                )
            )
            credit_table = json.loads(
                (package_dir / "credit_table.json").read_text(encoding="utf-8")
            )
            capacity_sweep_tables = json.loads(
                (package_dir / "capacity_sweep_tables.json").read_text(
                    encoding="utf-8"
                )
            )
            resolved_config = json.loads(
                (package_dir / "resolved_config.json").read_text(encoding="utf-8")
            )
            report_payload = json.loads(
                (package_dir / "report.json").read_text(encoding="utf-8")
            )
            pip_freeze = (package_dir / "pip_freeze.txt").read_text(encoding="utf-8")
            report_md = (package_dir / "report.md").read_text(encoding="utf-8")
            claim_tables = json.loads(
                (package_dir / "claim_test_tables.json").read_text(encoding="utf-8")
            )
            effect_tables = json.loads(
                (package_dir / "effect_size_tables.json").read_text(encoding="utf-8")
            )
            ladder_comparison = json.loads(
                (package_dir / "ladder_comparison.json").read_text(encoding="utf-8")
            )
            unified_ladder_report = json.loads(
                (package_dir / "unified_ladder_report.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(manifest_payload["package_version"], "1.0")
        self.assertEqual(manifest_payload["seed_count"], 2)
        self.assertEqual(manifest_payload["confidence_level"], 0.95)
        self.assertEqual(manifest_payload["environment"], environment)
        self.assertEqual(pip_freeze, "numpy==2.0.0\npygame-ce==2.5.0\n")
        self.assertTrue(manifest_payload["contents"])
        self.assertIn("metadata", capacity_sweep_tables)
        self.assertTrue(
            any(
                item["path"] == "pip_freeze.txt"
                for item in manifest_payload["contents"]
            )
        )
        self.assertTrue(
            any(
                item["path"] == "plots/capacity_comparison.svg"
                for item in manifest_payload["contents"]
            )
        )
        self.assertTrue(
            any(
                item["path"] == "aggregate_benchmark_tables.json"
                for item in manifest_payload["contents"]
            )
        )
        self.assertTrue(
            any(
                item["path"] == "credit_table.json"
                for item in manifest_payload["contents"]
            )
        )
        self.assertTrue(
            any(
                item["path"] == "ladder_comparison.json"
                for item in manifest_payload["contents"]
            )
        )
        self.assertTrue(
            any(
                item["path"] == "unified_ladder_report.json"
                for item in manifest_payload["contents"]
            )
        )
        self.assertIn("table", credit_table)
        self.assertIn("summary_statistics", credit_table)
        aggregate_ci_rows = self._rows_with_numeric_ci(aggregate_tables)
        claim_ci_rows = self._rows_with_numeric_ci(claim_tables)
        effect_ci_rows = self._rows_with_numeric_ci(effect_tables)
        self.assertTrue(aggregate_ci_rows)
        self.assertTrue(claim_ci_rows)
        self.assertTrue(effect_ci_rows)
        self.assertIn("architecture_capacity", aggregate_tables)
        self.assertTrue(aggregate_tables["architecture_capacity"]["rows"])
        self.assertIn("parameter_counts", resolved_config)
        self.assertEqual(
            resolved_config["parameter_counts"]["total_trainable"],
            840,
        )
        self.assertIn("capacity_analysis", report_payload)
        self.assertIn("unified_ladder_report", report_payload)
        self.assertIn("tables", report_payload["unified_ladder_report"])
        self.assertTrue(ladder_comparison["available"])
        self.assertEqual(
            set(unified_ladder_report.keys()),
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
        self.assertTrue(unified_ladder_report["rungs"]["available"])
        self.assertEqual(
            unified_ladder_report["conclusion"]["value"]["conclusion"],
            "capacity/interface confounded",
        )
        self.assertEqual(len(unified_ladder_report["rungs"]["value"]), 5)
        self.assertEqual(
            [
                (
                    item["baseline_rung"],
                    item["comparison_rung"],
                    item["metrics"]["baseline_variant"],
                    item["metrics"]["comparison_variant"],
                )
                for item in ladder_comparison["comparisons"]
            ],
            [
                ("A0", "A1", "true_monolithic_policy", "monolithic_policy"),
                ("A1", "A2", "monolithic_policy", "three_center_modular"),
                ("A2", "A3", "three_center_modular", "four_center_modular"),
                ("A3", "A4", "four_center_modular", "modular_full"),
            ],
        )
        self.assertEqual(
            report_payload["capacity_analysis"]["largest_variant"],
            "monolithic_policy",
        )
        self.assertAlmostEqual(
            float(report_payload["capacity_analysis"]["ratio"]),
            2.7,
            places=1,
        )
        self.assertIn("## Architecture Capacity", report_md)
        self.assertTrue(any(row.get("n_seeds") == 2 for row in aggregate_ci_rows))
        self.assertTrue(any(row.get("n_seeds") == 2 for row in claim_ci_rows))
        self.assertTrue(any(row.get("n_seeds") == 2 for row in effect_ci_rows))
        self.assertTrue(
            any(isinstance(row.get("cohens_d"), (int, float)) for row in effect_ci_rows)
        )

    def test_ladder_adjacent_comparisons_include_a3(self) -> None:
        self.assertEqual(
            LADDER_ADJACENT_COMPARISONS,
            (
                ("A0", "A1"),
                ("A1", "A2"),
                ("A2", "A3"),
                ("A3", "A4"),
            ),
        )

    def test_a3_ablation_variants_resolve(self) -> None:
        configs = resolve_ablation_configs(
            [
                "four_center_modular",
                "four_center_modular_local_credit",
                "four_center_modular_counterfactual",
            ]
        )

        self.assertEqual([config.name for config in configs], [
            "four_center_modular",
            "four_center_modular_local_credit",
            "four_center_modular_counterfactual",
        ])
        self.assertEqual(configs[0].architecture, "modular")
        self.assertEqual(configs[1].credit_strategy, "local_only")
        self.assertEqual(configs[2].credit_strategy, "counterfactual")

    def test_assemble_benchmark_package_merges_credit_table_limitations(self) -> None:
        summary = {
            "config": {
                "brain": {
                    "name": "modular_full",
                    "architecture": "modular",
                }
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            package_dir = Path(tmpdir) / "package"
            manifest = assemble_benchmark_package(
                package_dir,
                summary,
                behavior_rows=[],
            )
            limitations_text = (package_dir / "limitations.txt").read_text(
                encoding="utf-8"
            )

        self.assertIn(
            "No credit metrics were available in the summary payload.",
            manifest.limitations,
        )
        self.assertIn(
            "No credit metrics were available in the summary payload.",
            limitations_text,
        )

    def test_assemble_benchmark_package_preserves_parameter_counts_from_simulation_summary(
        self,
    ) -> None:
        sim = SpiderSimulation(seed=19, max_steps=20)
        summary, _ = sim.train(
            episodes=2,
            evaluation_episodes=1,
            capture_evaluation_trace=False,
        )

        self.assertIn("parameter_counts", summary)
        self.assertIn("per_network", summary["parameter_counts"])
        self.assertGreater(summary["parameter_counts"]["total"], 0)
        summary.setdefault("behavior_evaluation", {})["capacity_sweeps"] = self._summary()[
            "behavior_evaluation"
        ]["capacity_sweeps"]

        with tempfile.TemporaryDirectory() as tmpdir:
            package_dir = Path(tmpdir) / "package"
            assemble_benchmark_package(
                package_dir,
                summary,
                behavior_rows=[],
            )

            resolved_config = json.loads(
                (package_dir / "resolved_config.json").read_text(encoding="utf-8")
            )
            report_md = (package_dir / "report.md").read_text(encoding="utf-8")
            self.assertTrue((package_dir / "plots" / "capacity_comparison.svg").exists())

        self.assertEqual(
            resolved_config["parameter_counts"],
            summary["parameter_counts"],
        )
        self.assertIn("## Architecture Capacity", report_md)

    def test_assemble_benchmark_package_treats_non_mapping_config_as_empty(self) -> None:
        summary = self._summary()
        summary["config"] = None

        with tempfile.TemporaryDirectory() as tmpdir:
            package_dir = Path(tmpdir) / "package"
            assemble_benchmark_package(
                package_dir,
                summary,
                behavior_rows=[],
            )
            resolved_config = json.loads(
                (package_dir / "resolved_config.json").read_text(encoding="utf-8")
            )

        self.assertEqual(
            resolved_config,
            {"parameter_counts": summary["parameter_counts"]},
        )

    def test_assemble_benchmark_package_preserves_ladder_outputs_from_behavior_rows(
        self,
    ) -> None:
        summary = {
            "config": {
                "brain": {"name": "modular_full", "architecture": "modular"},
                "budget": {
                    "profile": "paper",
                    "benchmark_strength": "paper",
                    "resolved": {"episodes": 10},
                },
            },
            "checkpointing": {
                "selection": "best",
                "metric": "scenario_success_rate",
            },
        }
        behavior_rows = [
            {
                "simulation_seed": 1,
                "scenario": "night_rest",
                "success": False,
                "ablation_variant": "true_monolithic_policy",
                "ablation_architecture": "true_monolithic",
                "eval_reflex_scale": 0.0,
            },
            {
                "simulation_seed": 2,
                "scenario": "night_rest",
                "success": True,
                "ablation_variant": "true_monolithic_policy",
                "ablation_architecture": "true_monolithic",
                "eval_reflex_scale": 0.0,
            },
            {
                "simulation_seed": 1,
                "scenario": "night_rest",
                "success": True,
                "ablation_variant": "monolithic_policy",
                "ablation_architecture": "monolithic",
                "eval_reflex_scale": 0.0,
            },
            {
                "simulation_seed": 2,
                "scenario": "night_rest",
                "success": True,
                "ablation_variant": "monolithic_policy",
                "ablation_architecture": "monolithic",
                "eval_reflex_scale": 0.0,
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            package_dir = Path(tmpdir) / "package"
            assemble_benchmark_package(
                package_dir,
                summary,
                behavior_rows=behavior_rows,
                command_metadata={"argv": ["--budget-profile", "paper"]},
            )

            effect_tables = json.loads(
                (package_dir / "effect_size_tables.json").read_text(encoding="utf-8")
            )
            ladder_comparison = json.loads(
                (package_dir / "ladder_comparison.json").read_text(encoding="utf-8")
            )

        ladder_rows = [
            row for row in effect_tables["effect_sizes"]["rows"] if row["domain"] == "ladder"
        ]
        self.assertTrue(ladder_rows)
        self.assertTrue(ladder_comparison["available"])
        self.assertTrue(
            any(
                item["baseline_rung"] == "A0"
                and item["comparison_rung"] == "A1"
                for item in ladder_comparison["comparisons"]
            )
        )

    def test_assemble_benchmark_package_preserves_ladder_outputs_from_behavior_csv(
        self,
    ) -> None:
        summary = {
            "config": {
                "brain": {"name": "modular_full", "architecture": "modular"},
                "budget": {
                    "profile": "paper",
                    "benchmark_strength": "paper",
                    "resolved": {"episodes": 10},
                },
            },
            "checkpointing": {
                "selection": "best",
                "metric": "scenario_success_rate",
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            csv_path = root / "behavior.csv"
            package_dir = root / "package"
            with csv_path.open("w", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(
                    fh,
                    fieldnames=(
                        "simulation_seed",
                        "scenario",
                        "success",
                        "ablation_variant",
                        "ablation_architecture",
                        "eval_reflex_scale",
                    ),
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "simulation_seed": "1",
                        "scenario": "night_rest",
                        "success": "false",
                        "ablation_variant": "true_monolithic_policy",
                        "ablation_architecture": "true_monolithic",
                        "eval_reflex_scale": "0.0",
                    }
                )
                writer.writerow(
                    {
                        "simulation_seed": "2",
                        "scenario": "night_rest",
                        "success": "true",
                        "ablation_variant": "true_monolithic_policy",
                        "ablation_architecture": "true_monolithic",
                        "eval_reflex_scale": "0.0",
                    }
                )
                writer.writerow(
                    {
                        "simulation_seed": "1",
                        "scenario": "night_rest",
                        "success": "true",
                        "ablation_variant": "monolithic_policy",
                        "ablation_architecture": "monolithic",
                        "eval_reflex_scale": "0.0",
                    }
                )
                writer.writerow(
                    {
                        "simulation_seed": "2",
                        "scenario": "night_rest",
                        "success": "true",
                        "ablation_variant": "monolithic_policy",
                        "ablation_architecture": "monolithic",
                        "eval_reflex_scale": "0.0",
                    }
                )

            assemble_benchmark_package(
                package_dir,
                summary,
                behavior_csv=str(csv_path),
                command_metadata={"argv": ["--budget-profile", "paper"]},
            )

            effect_tables = json.loads(
                (package_dir / "effect_size_tables.json").read_text(encoding="utf-8")
            )
            ladder_comparison = json.loads(
                (package_dir / "ladder_comparison.json").read_text(encoding="utf-8")
            )

        ladder_rows = [
            row for row in effect_tables["effect_sizes"]["rows"] if row["domain"] == "ladder"
        ]
        self.assertTrue(ladder_rows)
        self.assertTrue(
            any(
                row["baseline"] == "true_monolithic_policy"
                and row["comparison"] == "monolithic_policy"
                for row in ladder_rows
            )
        )
        self.assertTrue(ladder_comparison["available"])
        self.assertTrue(
            any(
                item["baseline_rung"] == "A0"
                and item["comparison_rung"] == "A1"
                for item in ladder_comparison["comparisons"]
            )
        )
