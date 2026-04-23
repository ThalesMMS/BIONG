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


class BenchmarkPackageTest(unittest.TestCase):
    def test_contents_contract_includes_capacity_sweep_outputs(self) -> None:
        self.assertIn("capacity_sweep_tables.json", BENCHMARK_PACKAGE_CONTENTS)
        self.assertIn("plots/capacity_comparison.svg", BENCHMARK_PACKAGE_CONTENTS)

    def _uncertainty(
        self,
        *,
        mean: float = 0.75,
        seeds: list[float] | None = None,
    ) -> dict[str, object]:
        if seeds is None:
            seeds = [0.5, 1.0]
        return {
            "mean": mean,
            "ci_lower": min(seeds),
            "ci_upper": max(seeds),
            "std_error": 0.1,
            "n_seeds": len(seeds),
            "confidence_level": 0.95,
            "seed_values": list(seeds),
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
        true_monolithic_seed_level = [
            {
                "metric_name": "scenario_success_rate",
                "seed": 1,
                "value": 0.2,
                "condition": "true_monolithic_policy",
            },
            {
                "metric_name": "scenario_success_rate",
                "seed": 2,
                "value": 0.3,
                "condition": "true_monolithic_policy",
            },
        ]
        true_monolithic_scenario_seed_level = [
            {
                **row,
                "scenario": "night_rest",
            }
            for row in true_monolithic_seed_level
        ]
        three_center_seed_level = [
            {
                "metric_name": "scenario_success_rate",
                "seed": 1,
                "value": 0.55,
                "condition": "three_center_modular",
            },
            {
                "metric_name": "scenario_success_rate",
                "seed": 2,
                "value": 0.65,
                "condition": "three_center_modular",
            },
        ]
        three_center_scenario_seed_level = [
            {
                **row,
                "scenario": "night_rest",
            }
            for row in three_center_seed_level
        ]
        four_center_seed_level = [
            {
                "metric_name": "scenario_success_rate",
                "seed": 1,
                "value": 0.65,
                "condition": "four_center_modular",
            },
            {
                "metric_name": "scenario_success_rate",
                "seed": 2,
                "value": 0.75,
                "condition": "four_center_modular",
            },
        ]
        four_center_scenario_seed_level = [
            {
                **row,
                "scenario": "night_rest",
            }
            for row in four_center_seed_level
        ]
        return {
            "config": {
                "brain": {
                    "name": "modular_full",
                    "architecture": "modular",
                },
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
            "parameter_counts": {
                "architecture": "modular",
                "by_network": {
                    "visual_cortex": 120,
                    "sensory_cortex": 120,
                    "alert_center": 120,
                    "hunger_center": 120,
                    "sleep_center": 120,
                    "arbitration_network": 64,
                    "action_center": 96,
                    "motor_cortex": 80,
                },
                "total_trainable": 840,
                "proportions": {
                    "visual_cortex": 120 / 840,
                    "sensory_cortex": 120 / 840,
                    "alert_center": 120 / 840,
                    "hunger_center": 120 / 840,
                    "sleep_center": 120 / 840,
                    "arbitration_network": 64 / 840,
                    "action_center": 96 / 840,
                    "motor_cortex": 80 / 840,
                },
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
                            "parameter_counts": {
                                "architecture": "modular",
                                "by_network": {
                                    "visual_cortex": 120,
                                    "sensory_cortex": 120,
                                    "alert_center": 120,
                                    "hunger_center": 120,
                                    "sleep_center": 120,
                                    "arbitration_network": 64,
                                    "action_center": 96,
                                    "motor_cortex": 80,
                                },
                                "total_trainable": 840,
                                "proportions": {
                                    "visual_cortex": 120 / 840,
                                    "sensory_cortex": 120 / 840,
                                    "alert_center": 120 / 840,
                                    "hunger_center": 120 / 840,
                                    "sleep_center": 120 / 840,
                                    "arbitration_network": 64 / 840,
                                    "action_center": 96 / 840,
                                    "motor_cortex": 80 / 840,
                                },
                            },
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
                                        "success_rate": self._uncertainty(
                                            mean=0.375,
                                            seeds=[0.25, 0.5],
                                        )
                                    },
                                    "seed_level": monolithic_scenario_seed_level,
                                }
                            },
                            "seed_level": monolithic_seed_level,
                            "uncertainty": {
                                "scenario_success_rate": self._uncertainty(
                                    mean=0.375,
                                    seeds=[0.25, 0.5],
                                )
                            },
                            "config": {"architecture": "monolithic"},
                            "parameter_counts": {
                                "architecture": "monolithic",
                                "by_network": {
                                    "monolithic_policy": 1692,
                                    "arbitration_network": 64,
                                    "action_center": 96,
                                    "motor_cortex": 80,
                                },
                                "total_trainable": 1932,
                                "proportions": {
                                    "monolithic_policy": 1692 / 1932,
                                    "arbitration_network": 64 / 1932,
                                    "action_center": 96 / 1932,
                                    "motor_cortex": 80 / 1932,
                                },
                            },
                        },
                        "true_monolithic_policy": {
                            "summary": {
                                "scenario_success_rate": 0.25,
                                "episode_success_rate": 0.25,
                            },
                            "suite": {
                                "night_rest": {
                                    "success_rate": 0.25,
                                    "uncertainty": {
                                        "success_rate": self._uncertainty(
                                            mean=0.25,
                                            seeds=[0.2, 0.3],
                                        )
                                    },
                                    "seed_level": true_monolithic_scenario_seed_level,
                                }
                            },
                            "seed_level": true_monolithic_seed_level,
                            "uncertainty": {
                                "scenario_success_rate": self._uncertainty(
                                    mean=0.25,
                                    seeds=[0.2, 0.3],
                                )
                            },
                            "config": {"architecture": "true_monolithic"},
                            "parameter_counts": {
                                "architecture": "true_monolithic",
                                "by_network": {
                                    "true_monolithic_policy": 1500,
                                },
                                "total_trainable": 1500,
                                "proportions": {
                                    "true_monolithic_policy": 1.0,
                                },
                            },
                        },
                        "three_center_modular": {
                            "summary": {
                                "scenario_success_rate": 0.6,
                                "episode_success_rate": 0.6,
                            },
                            "suite": {
                                "night_rest": {
                                    "success_rate": 0.6,
                                    "uncertainty": {
                                        "success_rate": self._uncertainty(
                                            mean=0.6,
                                            seeds=[0.55, 0.65],
                                        )
                                    },
                                    "seed_level": three_center_scenario_seed_level,
                                }
                            },
                            "seed_level": three_center_seed_level,
                            "uncertainty": {
                                "scenario_success_rate": self._uncertainty(
                                    mean=0.6,
                                    seeds=[0.55, 0.65],
                                )
                            },
                            "config": {"architecture": "modular"},
                            "parameter_counts": {
                                "architecture": "modular",
                                "by_network": {
                                    "perception_center": 180,
                                    "homeostasis_center": 150,
                                    "threat_center": 150,
                                    "arbitration_network": 64,
                                    "action_center": 96,
                                    "motor_cortex": 80,
                                },
                                "total_trainable": 720,
                                "proportions": {
                                    "perception_center": 180 / 720,
                                    "homeostasis_center": 150 / 720,
                                    "threat_center": 150 / 720,
                                    "arbitration_network": 64 / 720,
                                    "action_center": 96 / 720,
                                    "motor_cortex": 80 / 720,
                                },
                            },
                        },
                        "four_center_modular": {
                            "summary": {
                                "scenario_success_rate": 0.7,
                                "episode_success_rate": 0.7,
                            },
                            "suite": {
                                "night_rest": {
                                    "success_rate": 0.7,
                                    "uncertainty": {
                                        "success_rate": self._uncertainty(
                                            mean=0.7,
                                            seeds=[0.65, 0.75],
                                        )
                                    },
                                    "seed_level": four_center_scenario_seed_level,
                                }
                            },
                            "seed_level": four_center_seed_level,
                            "uncertainty": {
                                "scenario_success_rate": self._uncertainty(
                                    mean=0.7,
                                    seeds=[0.65, 0.75],
                                )
                            },
                            "config": {"architecture": "modular"},
                            "parameter_counts": {
                                "architecture": "modular",
                                "by_network": {
                                    "visual_cortex": 120,
                                    "sensory_cortex": 120,
                                    "homeostasis_center": 150,
                                    "threat_center": 150,
                                    "arbitration_network": 64,
                                    "action_center": 96,
                                    "motor_cortex": 80,
                                },
                                "total_trainable": 780,
                                "proportions": {
                                    "visual_cortex": 120 / 780,
                                    "sensory_cortex": 120 / 780,
                                    "homeostasis_center": 150 / 780,
                                    "threat_center": 150 / 780,
                                    "arbitration_network": 64 / 780,
                                    "action_center": 96 / 780,
                                    "motor_cortex": 80 / 780,
                                },
                            },
                        },
                    },
                },
                "capacity_sweeps": {
                    "profiles": {
                        "current": {
                            "capacity_profile": {
                                "profile": "current",
                                "version": "v1",
                                "module_hidden_dims": {
                                    "visual_cortex": 32,
                                    "sensory_cortex": 28,
                                    "hunger_center": 26,
                                    "sleep_center": 24,
                                    "alert_center": 28,
                                    "perception_center": 36,
                                    "homeostasis_center": 28,
                                    "threat_center": 28,
                                },
                                "integration_hidden_dim": 32,
                                "scale_factor": 1.0,
                            },
                            "variants": {
                                "modular_full": {
                                    "summary": {
                                        "scenario_success_rate": 0.75,
                                        "episode_success_rate": 0.75,
                                    },
                                    "suite": {},
                                    "config": {"architecture": "modular"},
                                    "parameter_counts": {
                                        "total": 840,
                                        "per_network": {
                                            "visual_cortex": 120,
                                            "sensory_cortex": 120,
                                        },
                                    },
                                    "approximate_compute_cost": {
                                        "total": 2000,
                                        "unit": "approx_forward_macs",
                                        "per_network": {
                                            "visual_cortex": 400,
                                            "sensory_cortex": 400,
                                        },
                                    },
                                }
                            },
                        },
                        "large": {
                            "capacity_profile": {
                                "profile": "large",
                                "version": "v1",
                                "module_hidden_dims": {
                                    "visual_cortex": 64,
                                    "sensory_cortex": 56,
                                    "hunger_center": 52,
                                    "sleep_center": 48,
                                    "alert_center": 56,
                                    "perception_center": 72,
                                    "homeostasis_center": 56,
                                    "threat_center": 56,
                                },
                                "integration_hidden_dim": 64,
                                "scale_factor": 2.0,
                            },
                            "variants": {
                                "modular_full": {
                                    "summary": {
                                        "scenario_success_rate": 0.8,
                                        "episode_success_rate": 0.8,
                                    },
                                    "suite": {},
                                    "config": {"architecture": "modular"},
                                    "parameter_counts": {
                                        "total": 1680,
                                        "per_network": {
                                            "visual_cortex": 240,
                                            "sensory_cortex": 240,
                                        },
                                    },
                                    "approximate_compute_cost": {
                                        "total": 4000,
                                        "unit": "approx_forward_macs",
                                        "per_network": {
                                            "visual_cortex": 800,
                                            "sensory_cortex": 800,
                                        },
                                    },
                                }
                            },
                        },
                    }
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
            "environment": {
                "git_commit": "abc123",
                "git_tag": None,
                "git_dirty": False,
                "python_version": "3.11.0",
                "platform": "linux",
            },
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

    def _conclusion_input(
        self,
        *,
        ratio: float = 1.0,
        any_interface_insufficient: bool = False,
        delta: float | None = 0.3,
        ci_lower: float | None = 0.1,
        ci_upper: float | None = 0.5,
        cohens_d: float | None = 0.6,
    ) -> dict[str, object]:
        return {
            "capacity_matched_comparison": {
                "available": True,
                "capacity_matched": ratio <= 1.2,
                "ratio": ratio,
            },
            "interface_sufficiency_results": {
                "available": True,
                "any_interface_insufficient": any_interface_insufficient,
            },
            "overall_comparison": {
                "available": delta is not None,
                "delta": delta,
                "delta_ci_lower": ci_lower,
                "delta_ci_upper": ci_upper,
                "cohens_d": cohens_d,
            },
        }

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


if __name__ == "__main__":
    unittest.main()
