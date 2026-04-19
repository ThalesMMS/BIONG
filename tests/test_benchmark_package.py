import csv
import json
import tempfile
import unittest
from pathlib import Path

from spider_cortex_sim.benchmark_package import (
    BENCHMARK_PACKAGE_CONTENTS,
    BenchmarkManifest,
    assemble_benchmark_package,
    summarize_benchmark_manifest,
    _seed_level_rows_from_behavior_rows,
)


class BenchmarkPackageTest(unittest.TestCase):
    def _uncertainty(self) -> dict[str, object]:
        return {
            "mean": 0.75,
            "ci_lower": 0.5,
            "ci_upper": 1.0,
            "std_error": 0.1,
            "n_seeds": 2,
            "confidence_level": 0.95,
            "seed_values": [0.5, 1.0],
        }

    def _summary(self) -> dict[str, object]:
        seed_level = [
            {
                "metric_name": "scenario_success_rate",
                "seed": 1,
                "value": 0.5,
                "condition": "modular_full",
            },
            {
                "metric_name": "scenario_success_rate",
                "seed": 2,
                "value": 1.0,
                "condition": "modular_full",
            },
        ]
        scenario_seed_level = [
            {
                **row,
                "scenario": "night_rest",
            }
            for row in seed_level
        ]
        monolithic_seed_level = [
            {
                "metric_name": "scenario_success_rate",
                "seed": 1,
                "value": 0.25,
                "condition": "monolithic_policy",
            },
            {
                "metric_name": "scenario_success_rate",
                "seed": 2,
                "value": 0.5,
                "condition": "monolithic_policy",
            },
        ]
        monolithic_scenario_seed_level = [
            {
                **row,
                "scenario": "night_rest",
            }
            for row in monolithic_seed_level
        ]
        return {
            "config": {
                "budget": {
                    "profile": "paper",
                    "benchmark_strength": "paper",
                    "resolved": {"episodes": 10},
                }
            },
            "checkpointing": {
                "selection": "best",
                "metric": "scenario_success_rate",
            },
            "behavior_evaluation": {
                "summary": {
                    "scenario_success_rate": 0.75,
                    "episode_success_rate": 0.75,
                },
                "suite": {
                    "night_rest": {
                        "success_rate": 0.75,
                        "uncertainty": {"success_rate": self._uncertainty()},
                        "seed_level": scenario_seed_level,
                    }
                },
                "ablations": {
                    "reference_variant": "modular_full",
                    "variants": {
                        "modular_full": {
                            "summary": {
                                "scenario_success_rate": 0.75,
                                "episode_success_rate": 0.75,
                            },
                            "suite": {
                                "night_rest": {
                                    "success_rate": 0.75,
                                    "uncertainty": {
                                        "success_rate": self._uncertainty()
                                    },
                                    "seed_level": scenario_seed_level,
                                }
                            },
                            "seed_level": seed_level,
                            "uncertainty": {
                                "scenario_success_rate": self._uncertainty()
                            },
                            "config": {"architecture": "modular"},
                        },
                        "monolithic_policy": {
                            "summary": {
                                "scenario_success_rate": 0.375,
                                "episode_success_rate": 0.375,
                            },
                            "suite": {
                                "night_rest": {
                                    "success_rate": 0.375,
                                    "uncertainty": {
                                        "success_rate": self._uncertainty()
                                    },
                                    "seed_level": monolithic_scenario_seed_level,
                                }
                            },
                            "seed_level": monolithic_seed_level,
                            "uncertainty": {
                                "scenario_success_rate": self._uncertainty()
                            },
                            "config": {"architecture": "monolithic"},
                        }
                    },
                },
                "claim_tests": {
                    "claims": {
                        "package_claim": {
                            "status": "passed",
                            "passed": True,
                            "reference_value": 0.5,
                            "comparison_values": {"modular_full": 0.75},
                            "delta": {"modular_full": 0.25},
                            "effect_size": {"modular_full": 0.25},
                            "reference_uncertainty": self._uncertainty(),
                            "comparison_uncertainty": {
                                "modular_full": self._uncertainty()
                            },
                            "delta_uncertainty": {
                                "modular_full": self._uncertainty()
                            },
                            "effect_size_uncertainty": {
                                "modular_full": self._uncertainty()
                            },
                            "cohens_d": {"modular_full": 1.0},
                            "effect_magnitude": {"modular_full": "large"},
                            "primary_metric": "scenario_success_rate",
                        }
                    }
                },
            },
        }

    def _write_behavior_csv(self, path: Path) -> None:
        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=(
                    "simulation_seed",
                    "scenario",
                    "success",
                    "ablation_variant",
                    "eval_reflex_scale",
                ),
            )
            writer.writeheader()
            writer.writerow(
                {
                    "simulation_seed": 1,
                    "scenario": "night_rest",
                    "success": "true",
                    "ablation_variant": "modular_full",
                    "eval_reflex_scale": 0.0,
                }
            )

    def _manifest_kwargs(self) -> dict[str, object]:
        return {
            "package_version": "1.0",
            "created_at": "2026-04-15T00:00:00+00:00",
            "command_metadata": {"argv": []},
            "budget_profile": {"profile": "paper"},
            "checkpoint_selection": {"selection": "best"},
            "contents": [{"path": "report.json", "bytes": 2, "sha256": "abc"}],
            "seed_count": 2,
            "confidence_level": 0.95,
            "limitations": ["Synthetic fixture."],
        }

    def _rows_with_numeric_ci(self, payload: object) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []

        def visit(value: object) -> None:
            if isinstance(value, dict):
                ci_lower = value.get("ci_lower")
                ci_upper = value.get("ci_upper")
                if isinstance(ci_lower, (int, float)) and isinstance(
                    ci_upper,
                    (int, float),
                ):
                    rows.append(value)
                for nested_value in value.values():
                    visit(nested_value)
            elif isinstance(value, list):
                for item in value:
                    visit(item)

        visit(payload)
        return rows

    def test_canonical_contents_names_core_package_files(self) -> None:
        self.assertIn("benchmark_manifest.json", BENCHMARK_PACKAGE_CONTENTS)
        self.assertIn("aggregate_benchmark_tables.json", BENCHMARK_PACKAGE_CONTENTS)
        self.assertIn("plots/training_eval.svg", BENCHMARK_PACKAGE_CONTENTS)
        self.assertIn("supporting_csvs/behavior_rows.csv", BENCHMARK_PACKAGE_CONTENTS)

    def test_benchmark_manifest_validates_invariants(self) -> None:
        self.assertEqual(BenchmarkManifest(**self._manifest_kwargs()).seed_count, 2)

        invalid_cases = (
            ("package_version", "", "package_version"),
            ("created_at", "", "created_at"),
            ("command_metadata", [], "command_metadata"),
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

            manifest = assemble_benchmark_package(
                package_dir,
                self._summary(),
                behavior_csv,
                command_metadata={"argv": ["--budget-profile", "paper"]},
            )

            self.assertIsInstance(manifest, BenchmarkManifest)
            self.assertTrue((package_dir / "benchmark_manifest.json").exists())
            self.assertTrue((package_dir / "resolved_config.json").exists())
            self.assertTrue((package_dir / "seed_level_rows.csv").exists())
            self.assertTrue((package_dir / "aggregate_benchmark_tables.json").exists())
            self.assertTrue((package_dir / "claim_test_tables.json").exists())
            self.assertTrue((package_dir / "effect_size_tables.json").exists())
            self.assertTrue((package_dir / "report.md").exists())
            self.assertTrue((package_dir / "report.json").exists())
            self.assertTrue((package_dir / "plots" / "training_eval.svg").exists())
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
            claim_tables = json.loads(
                (package_dir / "claim_test_tables.json").read_text(encoding="utf-8")
            )
            effect_tables = json.loads(
                (package_dir / "effect_size_tables.json").read_text(encoding="utf-8")
            )

        self.assertEqual(manifest_payload["package_version"], "1.0")
        self.assertEqual(manifest_payload["seed_count"], 2)
        self.assertEqual(manifest_payload["confidence_level"], 0.95)
        self.assertTrue(manifest_payload["contents"])
        self.assertTrue(
            any(
                item["path"] == "aggregate_benchmark_tables.json"
                for item in manifest_payload["contents"]
            )
        )
        aggregate_ci_rows = self._rows_with_numeric_ci(aggregate_tables)
        claim_ci_rows = self._rows_with_numeric_ci(claim_tables)
        effect_ci_rows = self._rows_with_numeric_ci(effect_tables)
        self.assertTrue(aggregate_ci_rows)
        self.assertTrue(claim_ci_rows)
        self.assertTrue(effect_ci_rows)
        self.assertTrue(any(row.get("n_seeds") == 2 for row in aggregate_ci_rows))
        self.assertTrue(any(row.get("n_seeds") == 2 for row in claim_ci_rows))
        self.assertTrue(any(row.get("n_seeds") == 2 for row in effect_ci_rows))
        self.assertTrue(
            any(isinstance(row.get("cohens_d"), (int, float)) for row in effect_ci_rows)
        )

    def test_summarize_benchmark_manifest_counts_manifest_file(self) -> None:
        manifest = BenchmarkManifest(**self._manifest_kwargs())
        summary = summarize_benchmark_manifest(manifest, output_dir=Path("package"))

        self.assertEqual(summary["file_count"], len(manifest.contents) + 1)

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


if __name__ == "__main__":
    unittest.main()
