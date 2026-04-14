import json
import tempfile
import unittest
from pathlib import Path

from spider_cortex_sim.offline_analysis import (
    SHAPING_DEPENDENCE_WARNING_THRESHOLD,
    DEFAULT_MINIMAL_SHAPING_SURVIVAL_THRESHOLD,
    _mapping_or_empty,
    _noise_robustness_cell_summary,
    _noise_robustness_metrics,
    _normalize_noise_marginals,
    _ordered_noise_conditions,
    _profile_disposition_weights,
    _component_disposition_rows,
    _component_disposition_summary,
    _normalize_behavior_survival,
    _normalize_module_response_by_predator_type,
    _aggregate_specialization_from_scenarios,
    build_diagnostics,
    build_report_data,
    extract_ablations,
    extract_noise_robustness,
    extract_predator_type_specialization,
    extract_reflex_frequency,
    extract_shaping_audit,
    load_behavior_csv,
    load_summary,
    load_trace,
    normalize_behavior_rows,
    render_matrix_heatmap,
    run_offline_analysis,
    write_report,
)
from spider_cortex_sim.simulation import SpiderSimulation


REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKIN_SUMMARY = REPO_ROOT / "spider_summary.json"
CHECKIN_TRACE = REPO_ROOT / "spider_trace.jsonl"
SHAPING_GAP_EPSILON = 0.01
LARGE_SHAPING_GAP = SHAPING_DEPENDENCE_WARNING_THRESHOLD + SHAPING_GAP_EPSILON
SMALL_SHAPING_GAP = max(
    0.0,
    SHAPING_DEPENDENCE_WARNING_THRESHOLD - SHAPING_GAP_EPSILON,
)
EPISODE_SHAPING_GAP = SHAPING_DEPENDENCE_WARNING_THRESHOLD + (2 * SHAPING_GAP_EPSILON)
MEAN_REWARD_SHAPING_GAP = SHAPING_DEPENDENCE_WARNING_THRESHOLD + (3 * SHAPING_GAP_EPSILON)


class OfflineAnalysisLoadersTest(unittest.TestCase):
    def test_load_summary_reads_checkin_summary(self) -> None:
        summary = load_summary(CHECKIN_SUMMARY)
        self.assertIn("training", summary)
        self.assertIn("evaluation", summary)

    def test_load_trace_reads_checkin_trace(self) -> None:
        trace = load_trace(CHECKIN_TRACE)
        self.assertTrue(trace)
        self.assertIn("messages", trace[0])

    def test_load_behavior_csv_reads_csv_written_by_simulation(self) -> None:
        rows = [
            {
                "reward_profile": "classic",
                "scenario_map": "central_burrow",
                "evaluation_map": "central_burrow",
                "ablation_variant": "modular_full",
                "ablation_architecture": "modular",
                "reflex_scale": 1.0,
                "reflex_anneal_final_scale": 1.0,
                "eval_reflex_scale": 1.0,
                "budget_profile": "dev",
                "benchmark_strength": "quick",
                "operational_profile": "default_v1",
                "operational_profile_version": 1,
                "checkpoint_source": "final",
                "simulation_seed": 7,
                "episode_seed": 7001,
                "scenario": "night_rest",
                "scenario_description": "night rest",
                "scenario_objective": "sleep",
                "scenario_focus": "sleep",
                "episode": 0,
                "success": True,
                "failure_count": 0,
                "failures": "",
                "check_deep_night_shelter_passed": True,
                "check_deep_night_shelter_value": 1.0,
                "check_deep_night_shelter_expected": ">= 0.95",
            }
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "behavior.csv"
            SpiderSimulation.save_behavior_csv(rows, csv_path)
            loaded = load_behavior_csv(csv_path)

        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0]["scenario"], "night_rest")


class ShapingReportTest(unittest.TestCase):
    def _summary(
        self,
        scenario_success_rate_delta: float = LARGE_SHAPING_GAP,
    ) -> dict[str, object]:
        """
        Create a synthetic `summary` dictionary representing a reward audit used in tests.
        
        The `scenario_success_rate_delta` value is applied to the `"classic"` profile in
        `reward_audit.comparison.deltas_vs_minimal.classic.scenario_success_rate_delta`.
        
        Parameters:
            scenario_success_rate_delta (float): Delta for the classic profile's scenario-level success rate.
        
        Returns:
            dict: A summary-like dictionary containing `reward_audit` with:
                - `minimal_profile` (str)
                - `reward_components` (mapping of component metadata)
                - `reward_profiles` (profiles with disposition summaries)
                - `comparison` (including `deltas_vs_minimal` and `behavior_survival`)
        """
        return {
            "reward_audit": {
                "minimal_profile": "austere",
                "reward_components": {
                    "food_progress": {
                        "category": "progress",
                        "shaping_risk": "high",
                        "shaping_disposition": "removed",
                        "disposition_rationale": "Zeroed in the austere profile.",
                    }
                },
                "reward_profiles": {
                    "classic": {
                        "disposition_summary": {
                            "removed": {"total_weight_proxy": 1.2}
                        }
                    },
                    "austere": {
                        "disposition_summary": {
                            "removed": {"total_weight_proxy": 0.0}
                        }
                    },
                },
                "comparison": {
                    "minimal_profile": "austere",
                    "deltas_vs_minimal": {
                        "classic": {
                            "scenario_success_rate_delta": scenario_success_rate_delta,
                            "episode_success_rate_delta": EPISODE_SHAPING_GAP,
                            "mean_reward_delta": MEAN_REWARD_SHAPING_GAP,
                        },
                        "austere": {
                            "scenario_success_rate_delta": 0.0,
                            "episode_success_rate_delta": 0.0,
                            "mean_reward_delta": 0.0,
                        },
                    },
                    "behavior_survival": {
                        "minimal_profile": "austere",
                        "survival_threshold": 0.5,
                        "scenario_count": 1,
                        "surviving_scenario_count": 1,
                        "survival_rate": 1.0,
                        "scenarios": {
                            "night_rest": {
                                "austere_success_rate": 1.0,
                                "survives": True,
                                "episodes": 1,
                            }
                        },
                    },
                },
            }
        }

    def test_extract_shaping_audit_extracts_delta_metrics(self) -> None:
        result = extract_shaping_audit(self._summary())
        self.assertAlmostEqual(
            result["gap_metrics"]["scenario_success_rate_delta"],
            LARGE_SHAPING_GAP,
        )
        self.assertAlmostEqual(
            result["gap_metrics"]["episode_success_rate_delta"],
            EPISODE_SHAPING_GAP,
        )
        self.assertAlmostEqual(
            result["gap_metrics"]["mean_reward_delta"],
            MEAN_REWARD_SHAPING_GAP,
        )

    def test_build_report_data_includes_shaping_program_when_comparison_available(self) -> None:
        report = build_report_data(
            summary=self._summary(),
            trace=[],
            behavior_rows=[],
        )
        self.assertIn("shaping_program", report)
        self.assertTrue(report["shaping_program"]["available"])

    def test_write_report_contains_shaping_minimization_section(self) -> None:
        report = build_report_data(
            summary=self._summary(),
            trace=[],
            behavior_rows=[],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            write_report(tmpdir, report)
            report_md = (Path(tmpdir) / "report.md").read_text(encoding="utf-8")
        self.assertIn("## Shaping Minimization Program", report_md)
        self.assertIn("### Dense vs Minimal Gap", report_md)
        self.assertIn("### Profile-Level Summary", report_md)
        self.assertIn("### Component Dispositions", report_md)
        self.assertIn("### Behavior Survival", report_md)

    def test_write_report_renders_profile_level_shaping_summary(self) -> None:
        report = build_report_data(
            summary=self._summary(),
            trace=[],
            behavior_rows=[],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            write_report(tmpdir, report)
            report_md = (Path(tmpdir) / "report.md").read_text(encoding="utf-8")
        self.assertIn("Removed disposition weight gap (classic - austere): 1.20.", report_md)
        self.assertIn("| profile | disposition | total_weight_proxy |", report_md)
        self.assertIn("| classic | removed | 1.20 |", report_md)
        self.assertIn("| austere | removed | 0.00 |", report_md)
        self.assertIn("| disposition | component_count | components |", report_md)
        self.assertIn("| removed | 1 | food_progress |", report_md)

    def test_write_report_renders_warning_for_large_shaping_gap(self) -> None:
        report = build_report_data(
            summary=self._summary(scenario_success_rate_delta=LARGE_SHAPING_GAP),
            trace=[],
            behavior_rows=[],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            write_report(tmpdir, report)
            report_md = (Path(tmpdir) / "report.md").read_text(encoding="utf-8")
        self.assertIn("WARNING: High shaping dependence detected", report_md)

    def test_write_report_omits_warning_for_small_shaping_gap(self) -> None:
        report = build_report_data(
            summary=self._summary(scenario_success_rate_delta=SMALL_SHAPING_GAP),
            trace=[],
            behavior_rows=[],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            write_report(tmpdir, report)
            report_md = (Path(tmpdir) / "report.md").read_text(encoding="utf-8")
        self.assertNotIn("WARNING: High shaping dependence detected", report_md)

    def test_write_report_omits_placeholder_gap_without_reward_audit(self) -> None:
        report = build_report_data(summary={}, trace=[], behavior_rows=[])
        with tempfile.TemporaryDirectory() as tmpdir:
            write_report(tmpdir, report)
            report_md = (Path(tmpdir) / "report.md").read_text(encoding="utf-8")
        self.assertNotIn("| classic - austere | 0.00 | 0.00 | 0.00 |", report_md)
        self.assertNotIn("Removed disposition weight gap", report_md)


class OfflineAnalysisToleranceTest(unittest.TestCase):
    def _summary_with_shaping_program(self) -> dict[str, object]:
        """
        Return a synthetic summary dictionary that encodes a shaping-minimization program suitable for tests.
        
        The dictionary contains a `reward_audit` entry with:
        - `minimal_profile`: the identifier of the minimal shaping profile (`"austere"`).
        - `reward_components`: two example components (`"feeding"`, `"food_progress"`) with shaping risk, disposition, and optional rationale.
        - `reward_profiles`: `classic` and `austere` profiles, each providing `disposition_summary` values (`total_weight_proxy`) for dispositions such as `removed` and `outcome_signal`.
        - `comparison`: contains `minimal_profile`, `deltas_vs_minimal` (per-profile deltas for scenario/episode success rates and mean reward), and `behavior_survival` (availability flag, survival threshold, counts, and per-scenario survival details).
        
        Returns:
            dict: A synthetic summary shaped for shaping-audit and report-generation tests.
        """
        return {
            "reward_audit": {
                "minimal_profile": "austere",
                "reward_components": {
                    "feeding": {
                        "category": "event",
                        "shaping_risk": "low",
                        "shaping_disposition": "outcome_signal",
                    },
                    "food_progress": {
                        "category": "progress",
                        "shaping_risk": "high",
                        "shaping_disposition": "removed",
                        "disposition_rationale": "Zeroed in the austere profile.",
                    },
                },
                "reward_profiles": {
                    "classic": {
                        "disposition_summary": {
                            "removed": {"total_weight_proxy": 1.2},
                            "outcome_signal": {"total_weight_proxy": 3.48},
                        }
                    },
                    "austere": {
                        "disposition_summary": {
                            "removed": {"total_weight_proxy": 0.0},
                            "outcome_signal": {"total_weight_proxy": 3.48},
                        }
                    },
                },
                "comparison": {
                    "minimal_profile": "austere",
                    "deltas_vs_minimal": {
                        "classic": {
                            "scenario_success_rate_delta": LARGE_SHAPING_GAP,
                            "episode_success_rate_delta": EPISODE_SHAPING_GAP,
                            "mean_reward_delta": MEAN_REWARD_SHAPING_GAP,
                        },
                        "austere": {
                            "scenario_success_rate_delta": 0.0,
                            "episode_success_rate_delta": 0.0,
                            "mean_reward_delta": 0.0,
                        },
                    },
                    "behavior_survival": {
                        "available": True,
                        "minimal_profile": "austere",
                        "survival_threshold": 0.5,
                        "scenario_count": 2,
                        "surviving_scenario_count": 1,
                        "survival_rate": 0.5,
                        "scenarios": {
                            "night_rest": {
                                "austere_success_rate": 1.0,
                                "survives": True,
                                "episodes": 2,
                            },
                            "open_field_foraging": {
                                "austere_success_rate": 0.0,
                                "survives": False,
                                "episodes": 2,
                            },
                        },
                    },
                },
            }
        }

    def test_build_report_tolerates_summary_without_behavior_evaluation(self) -> None:
        summary = load_summary(CHECKIN_SUMMARY)
        report = build_report_data(summary=summary, trace=[], behavior_rows=[])

        self.assertTrue(report["training_eval"]["available"])
        self.assertFalse(report["scenario_success"]["available"])
        self.assertIn("No scenario-level success data was available.", report["limitations"])

    def test_build_report_tolerates_missing_comparisons_and_ablations(self) -> None:
        sim = SpiderSimulation(seed=3, max_steps=5)
        summary, _ = sim.train(
            episodes=2,
            evaluation_episodes=1,
            capture_evaluation_trace=False,
        )
        report = build_report_data(summary=summary, trace=[], behavior_rows=[])

        self.assertFalse(report["comparisons"]["available"])
        self.assertFalse(report["ablations"]["available"])
        self.assertTrue(report["training_eval"]["available"])

    def test_extract_shaping_audit_surfaces_gap_and_survival_flags(self) -> None:
        result = extract_shaping_audit(self._summary_with_shaping_program())

        self.assertTrue(result["available"])
        self.assertEqual(result["dense_profile"], "classic")
        self.assertEqual(result["minimal_profile"], "austere")
        self.assertTrue(result["interpretive_flags"]["shaping_dependent"])
        self.assertAlmostEqual(
            result["gap_metrics"]["scenario_success_rate_delta"],
            LARGE_SHAPING_GAP,
        )
        self.assertAlmostEqual(result["removed_weight_gap"], 1.2)
        self.assertIn("food_progress", result["disposition_summary"]["removed"]["components"])
        self.assertEqual(
            result["behavior_survival"]["surviving_scenario_count"],
            1,
        )

    def test_build_report_data_includes_shaping_program(self) -> None:
        report = build_report_data(
            summary=self._summary_with_shaping_program(),
            trace=[],
            behavior_rows=[],
        )

        self.assertIn("shaping_program", report)
        self.assertTrue(report["shaping_program"]["available"])
        self.assertTrue(
            report["shaping_program"]["interpretive_flags"]["shaping_dependent"]
        )

    def test_write_report_renders_shaping_program_section(self) -> None:
        """
        Verifies that writing a report produces a complete "Shaping Minimization Program" section when a shaping program is present in the summary.
        
        Asserts that the rendered report Markdown contains:
        - The main section header "## Shaping Minimization Program" and the subsection "### Dense vs Minimal Gap".
        - A warning line indicating high shaping dependence when applicable.
        - A table row for the `food_progress` component with disposition and removal status.
        - A summary line reporting surviving scenarios under minimal shaping ("1/2 scenarios survive minimal shaping").
        - A per-scenario row for "night_rest" showing its survival, success rate, and episode count.
        """
        report = build_report_data(
            summary=self._summary_with_shaping_program(),
            trace=[],
            behavior_rows=[],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            write_report(tmpdir, report)
            report_md = (Path(tmpdir) / "report.md").read_text(encoding="utf-8")

        self.assertIn("## Shaping Minimization Program", report_md)
        self.assertIn("### Dense vs Minimal Gap", report_md)
        self.assertIn("WARNING: High shaping dependence detected", report_md)
        self.assertIn("| food_progress | progress | high | removed |", report_md)
        self.assertIn("1/2 scenarios survive minimal shaping", report_md)
        self.assertIn("| night_rest | 1.00 | yes | 2 |", report_md)

    def test_extract_ablations_falls_back_to_behavior_csv_rows(self) -> None:
        rows = normalize_behavior_rows(
            [
                {
                    "scenario": "night_rest",
                    "success": True,
                    "ablation_variant": "modular_full",
                    "ablation_architecture": "modular",
                    "eval_reflex_scale": 1.0,
                },
                {
                    "scenario": "night_rest",
                    "success": False,
                    "ablation_variant": "monolithic_policy",
                    "ablation_architecture": "monolithic",
                    "eval_reflex_scale": 1.0,
                },
            ]
        )
        ablations = extract_ablations({}, rows)

        self.assertTrue(ablations["available"])
        self.assertIn("modular_full", ablations["variants"])
        self.assertIn("monolithic_policy", ablations["variants"])

    def test_extract_ablations_fallback_uses_minimal_reflex_rows(self) -> None:
        rows = normalize_behavior_rows(
            [
                {
                    "scenario": "night_rest",
                    "success": False,
                    "ablation_variant": "modular_full",
                    "ablation_architecture": "modular",
                    "eval_reflex_scale": 1.0,
                },
                {
                    "scenario": "night_rest",
                    "success": True,
                    "ablation_variant": "modular_full",
                    "ablation_architecture": "modular",
                    "eval_reflex_scale": 0.0,
                },
                {
                    "scenario": "night_rest",
                    "success": True,
                    "ablation_variant": "monolithic_policy",
                    "ablation_architecture": "monolithic",
                    "eval_reflex_scale": 1.0,
                },
                {
                    "scenario": "night_rest",
                    "success": False,
                    "ablation_variant": "monolithic_policy",
                    "ablation_architecture": "monolithic",
                    "eval_reflex_scale": 0.0,
                },
            ]
        )

        ablations = extract_ablations({}, rows)

        self.assertEqual(ablations["reference_variant"], "modular_full")
        self.assertEqual(
            ablations["variants"]["modular_full"]["summary"]["eval_reflex_scale"],
            0.0,
        )
        self.assertAlmostEqual(
            ablations["variants"]["modular_full"]["summary"]["scenario_success_rate"],
            1.0,
        )
        self.assertAlmostEqual(
            ablations["variants"]["monolithic_policy"]["summary"]["scenario_success_rate"],
            0.0,
        )
        self.assertAlmostEqual(
            ablations["deltas_vs_reference"]["monolithic_policy"]["summary"][
                "scenario_success_rate_delta"
            ],
            -1.0,
        )

    def test_extract_ablations_summary_promotes_without_reflex_support(self) -> None:
        """
        Verifies that `extract_ablations` promotes `without_reflex_support` metrics into primary variant summaries and that diagnostics reflect the promoted values.
        
        Asserts that when ablation variants include a `without_reflex_support` branch:
        - The variant's `summary` fields (e.g., `scenario_success_rate`) are replaced by the `without_reflex_support.summary` values.
        - The promoted variant receives `eval_reflex_scale == 0.0`.
        - Deltas versus the reference variant are computed from the promoted summaries.
        - `build_diagnostics` selects the best ablation variant based on the promoted scenario success rate and formats its label/value accordingly.
        """
        summary = {
            "behavior_evaluation": {
                "ablations": {
                    "reference_variant": "modular_full",
                    "variants": {
                        "modular_full": {
                            "config": {"architecture": "modular"},
                            "summary": {"scenario_success_rate": 1.0},
                            "suite": {"night_rest": {"success_rate": 1.0}},
                            "without_reflex_support": {
                                "summary": {
                                    "scenario_success_rate": 0.2,
                                    "episode_success_rate": 0.2,
                                    "mean_reward": 2.0,
                                },
                                "suite": {"night_rest": {"success_rate": 0.2}},
                            },
                        },
                        "no_module_reflexes": {
                            "config": {"architecture": "modular"},
                            "summary": {"scenario_success_rate": 0.1},
                            "suite": {"night_rest": {"success_rate": 0.1}},
                            "without_reflex_support": {
                                "summary": {
                                    "scenario_success_rate": 0.8,
                                    "episode_success_rate": 0.8,
                                    "mean_reward": 8.0,
                                },
                                "suite": {"night_rest": {"success_rate": 0.8}},
                            },
                        },
                    },
                }
            }
        }

        ablations = extract_ablations(summary, [])
        diagnostics = build_diagnostics(
            {},
            {"scenarios": []},
            ablations,
            {"modules": []},
        )

        self.assertAlmostEqual(
            ablations["variants"]["modular_full"]["summary"]["scenario_success_rate"],
            0.2,
        )
        self.assertEqual(
            ablations["variants"]["modular_full"]["summary"]["eval_reflex_scale"],
            0.0,
        )
        self.assertAlmostEqual(
            ablations["variants"]["no_module_reflexes"]["summary"]["scenario_success_rate"],
            0.8,
        )
        self.assertAlmostEqual(
            ablations["deltas_vs_reference"]["no_module_reflexes"]["summary"][
                "scenario_success_rate_delta"
            ],
            0.6,
        )
        best = next(
            item for item in diagnostics if item["label"] == "Best ablation variant"
        )
        self.assertEqual(best["value"], "no_module_reflexes (0.80)")

    def test_build_report_data_surfaces_no_reflex_primary_benchmark(self) -> None:
        summary = {
            "behavior_evaluation": {
                "ablations": {
                    "reference_variant": "modular_full",
                    "variants": {
                        "modular_full": {
                            "config": {"architecture": "modular"},
                            "summary": {"scenario_success_rate": 0.9},
                            "suite": {"night_rest": {"success_rate": 0.9}},
                            "without_reflex_support": {
                                "summary": {
                                    "scenario_success_rate": 0.4,
                                    "episode_success_rate": 0.4,
                                    "mean_reward": 4.0,
                                },
                                "suite": {"night_rest": {"success_rate": 0.4}},
                            },
                        }
                    },
                }
            },
            "evaluation_with_reflex_support": {
                "summary": {
                    "mean_final_reflex_override_rate": 0.2,
                    "mean_reflex_dominance": 0.5,
                }
            },
        }

        report = build_report_data(summary=summary, trace=[], behavior_rows=[])

        self.assertTrue(report["primary_benchmark"]["available"])
        self.assertEqual(report["primary_benchmark"]["metric"], "scenario_success_rate")
        self.assertAlmostEqual(
            report["primary_benchmark"]["scenario_success_rate"],
            0.4,
        )
        self.assertEqual(report["primary_benchmark"]["eval_reflex_scale"], 0.0)
        self.assertEqual(report["primary_benchmark"]["reference_variant"], "modular_full")
        diagnostics = {item["label"]: item for item in report["diagnostics"]}
        self.assertEqual(
            diagnostics["Reflex Dependence: override rate"]["status"],
            "warning",
        )
        self.assertTrue(
            diagnostics["Reflex Dependence: override rate"]["failure_indicator"]
        )
        self.assertEqual(
            diagnostics["Reflex Dependence: dominance"]["status"],
            "warning",
        )

    def test_build_report_data_does_not_treat_legacy_evaluation_as_no_reflex(self) -> None:
        summary = {"evaluation": {"scenario_success_rate": 0.9}}

        report = build_report_data(summary=summary, trace=[], behavior_rows=[])

        self.assertFalse(report["primary_benchmark"]["available"])
        self.assertIsNone(report["primary_benchmark"]["eval_reflex_scale"])
        self.assertEqual(report["primary_benchmark"]["source"], "none")

    def test_build_report_data_suppresses_reflex_dependence_without_debug_reflexes(self) -> None:
        trace = [
            {
                "messages": [
                    {
                        "sender": "visual_cortex",
                        "topic": "action.proposal",
                        "payload": {"reflex": True},
                    }
                ]
            }
        ]

        report = build_report_data(summary={}, trace=trace, behavior_rows=[])

        self.assertFalse(report["reflex_dependence"]["available"])
        diagnostics = {item["label"]: item for item in report["diagnostics"]}
        self.assertNotIn("Reflex Dependence: override rate", diagnostics)
        self.assertNotIn("Reflex Dependence: dominance", diagnostics)

    def test_write_report_ablation_table_includes_no_reflex_scale(self) -> None:
        report = build_report_data(
            summary={},
            trace=[],
            behavior_rows=[
                {
                    "scenario": "night_rest",
                    "success": False,
                    "ablation_variant": "modular_full",
                    "ablation_architecture": "modular",
                    "eval_reflex_scale": 1.0,
                },
                {
                    "scenario": "night_rest",
                    "success": True,
                    "ablation_variant": "modular_full",
                    "ablation_architecture": "modular",
                    "eval_reflex_scale": 0.0,
                },
            ],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            write_report(tmpdir, report)
            report_md = (Path(tmpdir) / "report.md").read_text(encoding="utf-8")

        self.assertIn("## Primary Benchmark", report_md)
        self.assertIn(
            "| metric | scenario_success_rate | eval_reflex_scale | reference_variant | source |",
            report_md,
        )
        self.assertIn(
            "| variant | architecture | eval_reflex_scale | scenario_success_rate |",
            report_md,
        )
        self.assertIn("| modular_full | modular | 0.00 | 1.00 |", report_md)

    def test_extract_ablations_includes_predator_type_comparisons(self) -> None:
        rows = normalize_behavior_rows(
            [
                {
                    "scenario": "visual_hunter_open_field",
                    "success": False,
                    "ablation_variant": "drop_visual_cortex",
                    "ablation_architecture": "modular",
                    "eval_reflex_scale": 0.0,
                },
                {
                    "scenario": "olfactory_ambush",
                    "success": True,
                    "ablation_variant": "drop_visual_cortex",
                    "ablation_architecture": "modular",
                    "eval_reflex_scale": 0.0,
                },
                {
                    "scenario": "visual_hunter_open_field",
                    "success": True,
                    "ablation_variant": "drop_sensory_cortex",
                    "ablation_architecture": "modular",
                    "eval_reflex_scale": 0.0,
                },
                {
                    "scenario": "olfactory_ambush",
                    "success": False,
                    "ablation_variant": "drop_sensory_cortex",
                    "ablation_architecture": "modular",
                    "eval_reflex_scale": 0.0,
                },
            ]
        )

        ablations = extract_ablations({}, rows)

        self.assertTrue(ablations["predator_type_comparisons"]["available"])
        comparisons = ablations["predator_type_comparisons"]["comparisons"]
        self.assertIn("drop_visual_cortex", comparisons)
        self.assertIn("drop_sensory_cortex", comparisons)

        unrelated_rows = normalize_behavior_rows(
            [
                {
                    "scenario": "open_field_foraging",
                    "success": True,
                    "ablation_variant": "drop_visual_cortex",
                    "ablation_architecture": "modular",
                    "eval_reflex_scale": 0.0,
                },
                {
                    "scenario": "corridor_gauntlet",
                    "success": False,
                    "ablation_variant": "drop_sensory_cortex",
                    "ablation_architecture": "modular",
                    "eval_reflex_scale": 0.0,
                },
            ]
        )
        unrelated_ablations = extract_ablations({}, unrelated_rows)
        self.assertFalse(unrelated_ablations["predator_type_comparisons"]["available"])


class PredatorTypeSpecializationReportTest(unittest.TestCase):
    def _summary(self) -> dict[str, object]:
        """
        Return a synthetic evaluation payload containing mean module responses by predator type.
        
        The returned dictionary mimics the structure produced by evaluation aggregates and includes
        per-predator-type mean responses for modules such as `visual_cortex`, `sensory_cortex`, and `alert_center`.
        
        Returns:
            dict[str, object]: A mapping with key `"evaluation"` containing
            `mean_module_response_by_predator_type`, where each predator type maps module names
            to their mean response values (floats between 0.0 and 1.0).
        """
        return {
            "evaluation": {
                "mean_module_response_by_predator_type": {
                    "visual": {
                        "visual_cortex": 0.75,
                        "sensory_cortex": 0.20,
                        "alert_center": 0.05,
                    },
                    "olfactory": {
                        "visual_cortex": 0.10,
                        "sensory_cortex": 0.80,
                        "alert_center": 0.10,
                    },
                }
            }
        }

    def test_extract_predator_type_specialization_reads_evaluation_aggregate(self) -> None:
        result = extract_predator_type_specialization(self._summary(), [], [])

        self.assertTrue(result["available"])
        self.assertEqual(result["source"], "summary.evaluation")
        self.assertEqual(
            result["predator_types"]["visual"]["dominant_module"],
            "visual_cortex",
        )
        self.assertEqual(
            result["predator_types"]["olfactory"]["dominant_module"],
            "sensory_cortex",
        )
        self.assertGreater(result["specialization_score"], 0.5)

    def test_build_report_data_includes_predator_type_specialization(self) -> None:
        report = build_report_data(summary=self._summary(), trace=[], behavior_rows=[])

        self.assertIn("predator_type_specialization", report)
        self.assertTrue(report["predator_type_specialization"]["available"])

    def test_write_report_contains_predator_type_specialization_section(self) -> None:
        report = build_report_data(summary=self._summary(), trace=[], behavior_rows=[])

        with tempfile.TemporaryDirectory() as tmpdir:
            write_report(tmpdir, report)
            report_md = (Path(tmpdir) / "report.md").read_text(encoding="utf-8")

        self.assertIn("## Predator Type Specialization", report_md)
        self.assertIn("Specialization score:", report_md)
        self.assertIn("| visual | 0.75 | 0.20 | visual_cortex |", report_md)


class ExtractNoiseRobustnessTest(unittest.TestCase):
    def test_extract_noise_robustness_reads_summary_payload(self) -> None:
        """
        Verifies that extract_noise_robustness reads a complete robustness_matrix from a summary payload and returns the expected matrix metadata and scores.
        
        Asserts that the result is marked available, that the source points to the summary robustness_matrix path, that train conditions in matrix_spec match the summary, that a specific matrix cell's scenario_success_rate matches the summary value, and that the reported robustness_score equals the provided value.
        """
        summary = {
            "behavior_evaluation": {
                "robustness_matrix": {
                    "matrix_spec": {
                        "train_conditions": ["none", "low"],
                        "eval_conditions": ["none", "low"],
                        "cell_count": 4,
                    },
                    "matrix": {
                        "none": {
                            "none": {"summary": {"scenario_success_rate": 1.0, "episode_success_rate": 1.0}},
                            "low": {"summary": {"scenario_success_rate": 0.5, "episode_success_rate": 0.5}},
                        },
                        "low": {
                            "none": {"summary": {"scenario_success_rate": 0.75, "episode_success_rate": 0.75}},
                            "low": {"summary": {"scenario_success_rate": 0.25, "episode_success_rate": 0.25}},
                        },
                    },
                    "train_marginals": {"none": 0.75, "low": 0.5},
                    "eval_marginals": {"none": 0.875, "low": 0.375},
                    "robustness_score": 0.625,
                    "diagonal_score": 0.625,
                    "off_diagonal_score": 0.625,
                }
            }
        }

        result = extract_noise_robustness(summary, [])

        self.assertTrue(result["available"])
        self.assertEqual(
            result["source"],
            "summary.behavior_evaluation.robustness_matrix",
        )
        self.assertEqual(result["matrix_spec"]["train_conditions"], ["none", "low"])
        self.assertEqual(result["matrix"]["none"]["low"]["scenario_success_rate"], 0.5)
        self.assertAlmostEqual(result["robustness_score"], 0.625)

    def test_extract_noise_robustness_falls_back_to_behavior_rows(self) -> None:
        rows = normalize_behavior_rows(
            [
                {
                    "scenario": "night_rest",
                    "success": True,
                    "train_noise_profile": "none",
                    "eval_noise_profile": "none",
                },
                {
                    "scenario": "night_rest",
                    "success": False,
                    "train_noise_profile": "none",
                    "eval_noise_profile": "high",
                },
                {
                    "scenario": "night_rest",
                    "success": True,
                    "train_noise_profile": "low",
                    "eval_noise_profile": "none",
                },
                {
                    "scenario": "night_rest",
                    "success": False,
                    "train_noise_profile": "low",
                    "eval_noise_profile": "high",
                },
            ]
        )

        result = extract_noise_robustness({}, rows)

        self.assertTrue(result["available"])
        self.assertEqual(result["source"], "behavior_csv")
        self.assertEqual(result["matrix_spec"]["cell_count"], 4)
        self.assertEqual(result["matrix"]["none"]["none"]["scenario_success_rate"], 1.0)
        self.assertEqual(result["matrix"]["low"]["high"]["scenario_success_rate"], 0.0)
        self.assertAlmostEqual(result["train_marginals"]["none"], 0.5)
        self.assertAlmostEqual(result["eval_marginals"]["high"], 0.0)

    def test_extract_noise_robustness_missing_data_returns_unavailable_structure(self) -> None:
        result = extract_noise_robustness({}, [])

        self.assertFalse(result["available"])
        self.assertEqual(result["source"], "none")
        self.assertEqual(result["matrix_spec"]["cell_count"], 0)
        self.assertFalse(result["metadata"]["complete"])
        self.assertTrue(
            any(
                "No noise robustness matrix payload" in limitation
                for limitation in result["limitations"]
            )
        )

    def test_extract_noise_robustness_partial_summary_marks_incomplete(self) -> None:
        summary = {
            "behavior_evaluation": {
                "robustness_matrix": {
                    "matrix_spec": {
                        "train_conditions": ["none", "low"],
                        "eval_conditions": ["none", "high"],
                        "cell_count": 4,
                    },
                    "matrix": {
                        "none": {
                            "none": {"summary": {"scenario_success_rate": 1.0, "episode_success_rate": 1.0}},
                        },
                    },
                }
            }
        }

        result = extract_noise_robustness(summary, [])

        self.assertFalse(result["available"])
        self.assertEqual(
            result["source"],
            "summary.behavior_evaluation.robustness_matrix",
        )
        self.assertFalse(result["metadata"]["complete"])
        self.assertEqual(result["metadata"]["available_cell_count"], 1)
        self.assertTrue(
            any(
                "some train/eval cells were missing" in limitation
                for limitation in result["limitations"]
            )
        )
        self.assertEqual(result["matrix"]["none"]["none"]["scenario_success_rate"], 1.0)
        self.assertIsNone(result["robustness_score"])

    def test_extract_noise_robustness_partial_summary_with_behavior_rows_reconstructs(self) -> None:
        summary = {
            "behavior_evaluation": {
                "robustness_matrix": {
                    "matrix_spec": {
                        "train_conditions": ["none", "low"],
                        "eval_conditions": ["none", "high"],
                        "cell_count": 4,
                    },
                    "matrix": {
                        "none": {
                            "none": {"summary": {"scenario_success_rate": 1.0, "episode_success_rate": 1.0}},
                        },
                    },
                }
            }
        }
        rows = normalize_behavior_rows(
            [
                {
                    "scenario": "night_rest",
                    "success": False,
                    "train_noise_profile": "none",
                    "eval_noise_profile": "high",
                },
                {
                    "scenario": "night_rest",
                    "success": True,
                    "train_noise_profile": "low",
                    "eval_noise_profile": "none",
                },
                {
                    "scenario": "night_rest",
                    "success": False,
                    "train_noise_profile": "low",
                    "eval_noise_profile": "high",
                },
            ]
        )

        result = extract_noise_robustness(summary, rows)

        self.assertTrue(result["available"])
        self.assertTrue(result["metadata"]["complete"])
        self.assertEqual(result["metadata"]["available_cell_count"], 4)
        self.assertEqual(result["matrix_spec"]["cell_count"], 4)
        self.assertEqual(result["matrix"]["none"]["none"]["scenario_success_rate"], 1.0)
        self.assertEqual(result["matrix"]["low"]["high"]["scenario_success_rate"], 0.0)
        self.assertFalse(
            any(
                "some train/eval cells were missing" in limitation
                for limitation in result["limitations"]
            )
        )

    def test_build_report_data_includes_noise_robustness(self) -> None:
        summary = {
            "behavior_evaluation": {
                "robustness_matrix": {
                    "matrix_spec": {
                        "train_conditions": ["none"],
                        "eval_conditions": ["none"],
                        "cell_count": 1,
                    },
                    "matrix": {
                        "none": {
                            "none": {"summary": {"scenario_success_rate": 1.0, "episode_success_rate": 1.0}}
                        }
                    },
                    "train_marginals": {"none": 1.0},
                    "eval_marginals": {"none": 1.0},
                    "robustness_score": 1.0,
                    "diagonal_score": 1.0,
                    "off_diagonal_score": 0.0,
                }
            }
        }

        report = build_report_data(summary=summary, trace=[], behavior_rows=[])

        self.assertIn("noise_robustness", report)
        self.assertTrue(report["noise_robustness"]["available"])

    def test_write_report_contains_noise_robustness_section_and_svg(self) -> None:
        report = build_report_data(
            summary={},
            trace=[],
            behavior_rows=[
                {
                    "scenario": "night_rest",
                    "success": True,
                    "train_noise_profile": "none",
                    "eval_noise_profile": "none",
                },
                {
                    "scenario": "night_rest",
                    "success": False,
                    "train_noise_profile": "none",
                    "eval_noise_profile": "high",
                },
            ],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            generated = write_report(tmpdir, report)
            report_md = (Path(tmpdir) / "report.md").read_text(encoding="utf-8")

        self.assertIn("## Noise Robustness Matrix", report_md)
        self.assertIn("| train \\ eval |", report_md)
        self.assertIn("Overall robustness score:", report_md)
        self.assertTrue(generated["robustness_matrix_svg"])

    def test_write_report_marks_missing_robustness_cells_and_scores(self) -> None:
        report = build_report_data(
            summary={
                "behavior_evaluation": {
                    "robustness_matrix": {
                        "matrix_spec": {
                            "train_conditions": ["none", "low"],
                            "eval_conditions": ["none", "high"],
                            "cell_count": 4,
                        },
                        "matrix": {
                            "none": {
                                "none": {
                                    "summary": {
                                        "scenario_success_rate": 1.0,
                                        "episode_success_rate": 1.0,
                                    }
                                }
                            }
                        },
                    }
                }
            },
            trace=[],
            behavior_rows=[],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            generated = write_report(tmpdir, report)
            report_md = (Path(tmpdir) / "report.md").read_text(encoding="utf-8")
            robustness_svg = Path(generated["robustness_matrix_svg"]).read_text(
                encoding="utf-8"
            )

        self.assertIn("| none | 1.00 | — | 1.00 |", report_md)
        self.assertIn("| low | — | — | — |", report_md)
        self.assertIn("| mean | 1.00 | — | — |", report_md)
        self.assertIn("Overall robustness score: —.", report_md)
        self.assertIn(">—</text>", robustness_svg)


class ShapingHelperFunctionsTest(unittest.TestCase):
    """Tests for private helper functions added in this PR."""

    # --- _mapping_or_empty ---

    def test_mapping_or_empty_returns_dict_unchanged(self) -> None:
        d = {"a": 1, "b": 2}
        result = _mapping_or_empty(d)
        self.assertIs(result, d)

    def test_mapping_or_empty_returns_empty_for_none(self) -> None:
        result = _mapping_or_empty(None)
        self.assertEqual(dict(result), {})

    def test_mapping_or_empty_returns_empty_for_list(self) -> None:
        result = _mapping_or_empty([1, 2, 3])
        self.assertEqual(dict(result), {})

    def test_mapping_or_empty_returns_empty_for_string(self) -> None:
        result = _mapping_or_empty("not a mapping")
        self.assertEqual(dict(result), {})

    def test_mapping_or_empty_returns_empty_for_int(self) -> None:
        result = _mapping_or_empty(42)
        self.assertEqual(dict(result), {})

    def test_mapping_or_empty_accepts_empty_dict(self) -> None:
        d: dict = {}
        result = _mapping_or_empty(d)
        self.assertIs(result, d)

    # --- _profile_disposition_weights ---

    def test_profile_disposition_weights_empty_input(self) -> None:
        result = _profile_disposition_weights({})
        self.assertEqual(result, {})

    def test_profile_disposition_weights_skips_non_mapping_profiles(self) -> None:
        profiles = {
            "good": {
                "disposition_summary": {
                    "removed": {"total_weight_proxy": 1.5},
                }
            },
            "bad": "not a mapping",
        }
        result = _profile_disposition_weights(profiles)
        self.assertIn("good", result)
        self.assertNotIn("bad", result)

    def test_profile_disposition_weights_skips_profile_with_non_mapping_disposition_summary(self) -> None:
        # When disposition_summary is not a Mapping, the entire profile is skipped
        profiles = {
            "profile_a": {
                "disposition_summary": "not_a_dict",
            }
        }
        result = _profile_disposition_weights(profiles)
        self.assertNotIn("profile_a", result)

    def test_profile_disposition_weights_extracts_weight_proxies(self) -> None:
        profiles = {
            "classic": {
                "disposition_summary": {
                    "removed": {"total_weight_proxy": 1.234567890},
                    "defended": {"total_weight_proxy": 2.5},
                }
            }
        }
        result = _profile_disposition_weights(profiles)
        self.assertIn("classic", result)
        self.assertAlmostEqual(result["classic"]["removed"], 1.234568, places=5)
        self.assertAlmostEqual(result["classic"]["defended"], 2.5)

    def test_profile_disposition_weights_rounds_to_six_decimals(self) -> None:
        profiles = {
            "p": {
                "disposition_summary": {
                    "removed": {"total_weight_proxy": 1.1234567},
                }
            }
        }
        result = _profile_disposition_weights(profiles)
        self.assertEqual(result["p"]["removed"], round(1.1234567, 6))

    def test_profile_disposition_weights_non_mapping_disposition_entry_yields_zero(self) -> None:
        profiles = {
            "p": {
                "disposition_summary": {
                    "removed": "not_a_dict",
                }
            }
        }
        result = _profile_disposition_weights(profiles)
        self.assertAlmostEqual(result["p"]["removed"], 0.0)

    def test_profile_disposition_weights_missing_total_weight_proxy_yields_zero(self) -> None:
        profiles = {
            "p": {
                "disposition_summary": {
                    "removed": {"some_other_key": 99.0},
                }
            }
        }
        result = _profile_disposition_weights(profiles)
        self.assertAlmostEqual(result["p"]["removed"], 0.0)

    def test_profile_disposition_weights_sorted_profile_keys(self) -> None:
        profiles = {
            "zzz": {"disposition_summary": {"removed": {"total_weight_proxy": 1.0}}},
            "aaa": {"disposition_summary": {"removed": {"total_weight_proxy": 2.0}}},
        }
        result = _profile_disposition_weights(profiles)
        self.assertEqual(list(result.keys()), ["aaa", "zzz"])

    # --- _component_disposition_rows ---

    def test_component_disposition_rows_empty_input(self) -> None:
        result = _component_disposition_rows({})
        self.assertEqual(result, [])

    def test_component_disposition_rows_skips_non_mapping_metadata(self) -> None:
        components = {
            "food_progress": {
                "category": "progress",
                "shaping_risk": "high",
                "shaping_disposition": "removed",
                "disposition_rationale": "Test",
            },
            "bad_entry": "not a mapping",
        }
        result = _component_disposition_rows(components)
        names = [row["component"] for row in result]
        self.assertIn("food_progress", names)
        self.assertNotIn("bad_entry", names)

    def test_component_disposition_rows_row_has_required_keys(self) -> None:
        components = {
            "food_progress": {
                "category": "progress",
                "shaping_risk": "high",
                "shaping_disposition": "removed",
                "disposition_rationale": "Some rationale",
            }
        }
        result = _component_disposition_rows(components)
        self.assertEqual(len(result), 1)
        row = result[0]
        for key in ("component", "category", "risk", "disposition", "rationale"):
            self.assertIn(key, row)

    def test_component_disposition_rows_coerces_fields_to_string(self) -> None:
        components = {
            "test_comp": {
                "category": None,
                "shaping_risk": None,
                "shaping_disposition": None,
                "disposition_rationale": None,
            }
        }
        result = _component_disposition_rows(components)
        self.assertEqual(len(result), 1)
        row = result[0]
        # None values coerced to empty string via `str(None or "")`
        self.assertIsInstance(row["category"], str)
        self.assertIsInstance(row["risk"], str)
        self.assertIsInstance(row["disposition"], str)
        self.assertIsInstance(row["rationale"], str)

    def test_component_disposition_rows_is_sorted_by_component_name(self) -> None:
        components = {
            "zzz_component": {"category": "event", "shaping_risk": "low", "shaping_disposition": "outcome_signal", "disposition_rationale": ""},
            "aaa_component": {"category": "progress", "shaping_risk": "high", "shaping_disposition": "removed", "disposition_rationale": "test"},
        }
        result = _component_disposition_rows(components)
        names = [row["component"] for row in result]
        self.assertEqual(names, sorted(names))

    # --- _component_disposition_summary ---

    def test_component_disposition_summary_empty_input(self) -> None:
        result = _component_disposition_summary([])
        self.assertEqual(result, {})

    def test_component_disposition_summary_groups_by_disposition(self) -> None:
        rows = [
            {"component": "food_progress", "disposition": "removed"},
            {"component": "shelter_progress", "disposition": "removed"},
            {"component": "feeding", "disposition": "outcome_signal"},
        ]
        result = _component_disposition_summary(rows)
        self.assertIn("removed", result)
        self.assertIn("outcome_signal", result)
        self.assertIn("food_progress", result["removed"]["components"])
        self.assertIn("shelter_progress", result["removed"]["components"])
        self.assertIn("feeding", result["outcome_signal"]["components"])

    def test_component_disposition_summary_skips_rows_with_empty_disposition(self) -> None:
        rows = [
            {"component": "food_progress", "disposition": ""},
            {"component": "feeding", "disposition": "outcome_signal"},
        ]
        result = _component_disposition_summary(rows)
        self.assertNotIn("", result)
        self.assertIn("outcome_signal", result)

    def test_component_disposition_summary_skips_rows_with_empty_component(self) -> None:
        rows = [
            {"component": "", "disposition": "removed"},
            {"component": "feeding", "disposition": "outcome_signal"},
        ]
        result = _component_disposition_summary(rows)
        self.assertEqual(result["outcome_signal"]["component_count"], 1)

    def test_component_disposition_summary_components_are_sorted(self) -> None:
        rows = [
            {"component": "zzz", "disposition": "removed"},
            {"component": "aaa", "disposition": "removed"},
        ]
        result = _component_disposition_summary(rows)
        self.assertEqual(result["removed"]["components"], ["aaa", "zzz"])

    def test_component_disposition_summary_component_count_matches_list(self) -> None:
        rows = [
            {"component": "food_progress", "disposition": "removed"},
            {"component": "shelter_progress", "disposition": "removed"},
            {"component": "feeding", "disposition": "outcome_signal"},
        ]
        result = _component_disposition_summary(rows)
        self.assertEqual(result["removed"]["component_count"], 2)
        self.assertEqual(result["outcome_signal"]["component_count"], 1)

    # --- _normalize_behavior_survival ---

    def test_normalize_behavior_survival_empty_input_returns_unavailable(self) -> None:
        result = _normalize_behavior_survival({})
        self.assertFalse(result["available"])
        self.assertEqual(result["scenario_count"], 0)
        self.assertEqual(result["surviving_scenario_count"], 0)
        self.assertAlmostEqual(result["survival_rate"], 0.0)
        self.assertEqual(result["scenarios"], [])

    def test_normalize_behavior_survival_uses_default_threshold_when_missing(self) -> None:
        result = _normalize_behavior_survival({})
        self.assertAlmostEqual(
            result["survival_threshold"],
            DEFAULT_MINIMAL_SHAPING_SURVIVAL_THRESHOLD,
        )

    def test_normalize_behavior_survival_respects_provided_threshold(self) -> None:
        result = _normalize_behavior_survival({"survival_threshold": 0.75})
        self.assertAlmostEqual(result["survival_threshold"], 0.75)

    def test_normalize_behavior_survival_scenarios_as_mapping(self) -> None:
        payload = {
            "scenarios": {
                "night_rest": {
                    "austere_success_rate": 1.0,
                    "survives": True,
                    "episodes": 3,
                }
            }
        }
        result = _normalize_behavior_survival(payload)
        self.assertTrue(result["available"])
        self.assertEqual(len(result["scenarios"]), 1)
        self.assertEqual(result["scenarios"][0]["scenario"], "night_rest")
        self.assertAlmostEqual(result["scenarios"][0]["austere_success_rate"], 1.0)
        self.assertTrue(result["scenarios"][0]["survives"])
        self.assertEqual(result["scenarios"][0]["episodes"], 3)

    def test_normalize_behavior_survival_accepts_success_rate_alias(self) -> None:
        payload = {
            "scenarios": {
                "night_rest": {
                    "success_rate": 0.75,
                    "survives": True,
                    "episodes": 3,
                }
            }
        }
        result = _normalize_behavior_survival(payload)
        self.assertAlmostEqual(result["scenarios"][0]["austere_success_rate"], 0.75)

    def test_normalize_behavior_survival_derives_survives_when_missing(self) -> None:
        payload = {
            "scenarios": {
                "night_rest": {"success_rate": 0.75, "episodes": 3},
                "open_field": {"austere_success_rate": 0.0, "episodes": 3},
            }
        }
        result = _normalize_behavior_survival(payload)
        scenarios = {row["scenario"]: row for row in result["scenarios"]}
        self.assertTrue(scenarios["night_rest"]["survives"])
        self.assertFalse(scenarios["open_field"]["survives"])

    def test_normalize_behavior_survival_derives_survives_for_list_payloads(self) -> None:
        payload = {
            "scenarios": [
                {
                    "scenario": "night_rest",
                    "austere_success_rate": 0.5,
                    "episodes": 3,
                }
            ]
        }
        result = _normalize_behavior_survival(payload)
        self.assertTrue(result["scenarios"][0]["survives"])

    def test_normalize_behavior_survival_derives_survives_using_threshold(self) -> None:
        payload = {
            "survival_threshold": 0.5,
            "scenarios": {
                "open_field": {
                    "austere_success_rate": 0.1,
                    "episodes": 3,
                }
            },
        }
        result = _normalize_behavior_survival(payload)
        self.assertFalse(result["scenarios"][0]["survives"])

    def test_normalize_behavior_survival_scenarios_as_list(self) -> None:
        payload = {
            "scenarios": [
                {
                    "scenario": "night_rest",
                    "austere_success_rate": 0.6,
                    "survives": True,
                    "episodes": 2,
                }
            ]
        }
        result = _normalize_behavior_survival(payload)
        self.assertTrue(result["available"])
        self.assertEqual(len(result["scenarios"]), 1)
        self.assertEqual(result["scenarios"][0]["scenario"], "night_rest")

    def test_normalize_behavior_survival_scenarios_as_neither_yields_empty(self) -> None:
        payload = {"scenarios": "not_a_collection"}
        result = _normalize_behavior_survival(payload)
        self.assertEqual(result["scenarios"], [])
        self.assertFalse(result["available"])

    def test_normalize_behavior_survival_explicit_available_false_overrides(self) -> None:
        payload = {
            "available": False,
            "scenarios": {
                "night_rest": {"austere_success_rate": 1.0, "survives": True, "episodes": 1}
            },
        }
        result = _normalize_behavior_survival(payload)
        self.assertFalse(result["available"])

    def test_normalize_behavior_survival_uses_provided_scenario_count(self) -> None:
        payload = {
            "scenario_count": 5,
            "surviving_scenario_count": 3,
            "survival_rate": 0.6,
            "scenarios": {
                "night_rest": {"austere_success_rate": 1.0, "survives": True, "episodes": 1}
            },
        }
        result = _normalize_behavior_survival(payload)
        self.assertEqual(result["scenario_count"], 5)
        self.assertEqual(result["surviving_scenario_count"], 3)
        self.assertAlmostEqual(result["survival_rate"], 0.6)

    def test_normalize_behavior_survival_computes_rate_when_not_provided(self) -> None:
        payload = {
            "scenarios": {
                "night_rest": {"austere_success_rate": 1.0, "survives": True, "episodes": 1},
                "open_field": {"austere_success_rate": 0.0, "survives": False, "episodes": 1},
            }
        }
        result = _normalize_behavior_survival(payload)
        self.assertAlmostEqual(result["survival_rate"], 0.5)
        self.assertEqual(result["surviving_scenario_count"], 1)
        self.assertEqual(result["scenario_count"], 2)

    def test_normalize_behavior_survival_scenarios_sorted_alphabetically(self) -> None:
        payload = {
            "scenarios": {
                "zzz_scenario": {"austere_success_rate": 1.0, "survives": True, "episodes": 1},
                "aaa_scenario": {"austere_success_rate": 0.0, "survives": False, "episodes": 1},
            }
        }
        result = _normalize_behavior_survival(payload)
        names = [row["scenario"] for row in result["scenarios"]]
        self.assertEqual(names, sorted(names))

    def test_normalize_behavior_survival_skips_non_mapping_scenario_entries(self) -> None:
        payload = {
            "scenarios": {
                "night_rest": {"austere_success_rate": 1.0, "survives": True, "episodes": 1},
                "bad_entry": "not a mapping",
            }
        }
        result = _normalize_behavior_survival(payload)
        names = [row["scenario"] for row in result["scenarios"]]
        self.assertIn("night_rest", names)
        self.assertNotIn("bad_entry", names)

    def test_normalize_behavior_survival_minimal_profile_is_captured(self) -> None:
        payload = {
            "minimal_profile": "austere",
            "scenarios": {
                "s": {"austere_success_rate": 1.0, "survives": True, "episodes": 1}
            },
        }
        result = _normalize_behavior_survival(payload)
        self.assertEqual(result["minimal_profile"], "austere")

    def test_normalize_behavior_survival_empty_scenario_name_skipped(self) -> None:
        payload = {
            "scenarios": [
                {"scenario": "", "austere_success_rate": 1.0, "survives": True, "episodes": 1},
                {"scenario": "valid", "austere_success_rate": 0.5, "survives": False, "episodes": 2},
            ]
        }
        result = _normalize_behavior_survival(payload)
        names = [row["scenario"] for row in result["scenarios"]]
        self.assertNotIn("", names)
        self.assertIn("valid", names)


class ExtractShapingAuditEdgeCasesTest(unittest.TestCase):
    """Edge-case and boundary tests for extract_shaping_audit."""

    def test_extract_shaping_audit_empty_summary_returns_unavailable(self) -> None:
        result = extract_shaping_audit({})
        self.assertFalse(result["available"])
        self.assertEqual(result["source"], "none")
        self.assertEqual(result["dense_profile"], "classic")
        self.assertEqual(result["minimal_profile"], "austere")

    def test_extract_shaping_audit_no_reward_audit_sets_limitations(self) -> None:
        result = extract_shaping_audit({})
        self.assertIn("No reward_audit payload was available.", result["limitations"])

    def test_extract_shaping_audit_no_reward_audit_gap_metrics_are_zero(self) -> None:
        result = extract_shaping_audit({})
        self.assertAlmostEqual(result["gap_metrics"]["scenario_success_rate_delta"], 0.0)
        self.assertAlmostEqual(result["gap_metrics"]["episode_success_rate_delta"], 0.0)
        self.assertAlmostEqual(result["gap_metrics"]["mean_reward_delta"], 0.0)

    def test_extract_shaping_audit_no_reward_audit_flags_are_false(self) -> None:
        result = extract_shaping_audit({})
        self.assertFalse(result["interpretive_flags"]["gap_available"])
        self.assertFalse(result["interpretive_flags"]["shaping_dependent"])

    def test_extract_shaping_audit_available_with_survival_only_payload(self) -> None:
        summary = {
            "reward_audit": {
                "comparison": {
                    "minimal_profile": "austere",
                    "behavior_survival": {
                        "available": True,
                        "minimal_profile": "austere",
                        "scenario_count": 1,
                        "surviving_scenario_count": 1,
                        "survival_rate": 1.0,
                        "scenarios": {
                            "night_rest": {
                                "austere_success_rate": 1.0,
                                "survives": True,
                                "episodes": 1,
                            }
                        },
                    },
                }
            }
        }
        result = extract_shaping_audit(summary)
        self.assertTrue(result["available"])
        self.assertFalse(result["interpretive_flags"]["gap_available"])
        self.assertTrue(result["behavior_survival"]["available"])

    def test_extract_shaping_audit_thresholds_exposed_in_result(self) -> None:
        result = extract_shaping_audit({})
        self.assertAlmostEqual(
            result["thresholds"]["shaping_dependence"],
            SHAPING_DEPENDENCE_WARNING_THRESHOLD,
        )
        self.assertAlmostEqual(
            result["thresholds"]["behavior_survival"],
            DEFAULT_MINIMAL_SHAPING_SURVIVAL_THRESHOLD,
        )

    def test_extract_shaping_audit_gap_exactly_at_threshold_is_not_dependent(self) -> None:
        summary = {
            "reward_audit": {
                "comparison": {
                    "minimal_profile": "austere",
                    "deltas_vs_minimal": {
                        "classic": {
                            "scenario_success_rate_delta": SHAPING_DEPENDENCE_WARNING_THRESHOLD,
                            "episode_success_rate_delta": 0.0,
                            "mean_reward_delta": 0.0,
                        }
                    },
                }
            }
        }
        result = extract_shaping_audit(summary)
        self.assertFalse(result["interpretive_flags"]["shaping_dependent"])

    def test_extract_shaping_audit_gap_above_threshold_is_dependent(self) -> None:
        summary = {
            "reward_audit": {
                "comparison": {
                    "minimal_profile": "austere",
                    "deltas_vs_minimal": {
                        "classic": {
                            "scenario_success_rate_delta": (
                                SHAPING_DEPENDENCE_WARNING_THRESHOLD
                                + SHAPING_GAP_EPSILON
                            ),
                            "episode_success_rate_delta": 0.0,
                            "mean_reward_delta": 0.0,
                        }
                    },
                }
            }
        }
        result = extract_shaping_audit(summary)
        self.assertTrue(result["interpretive_flags"]["shaping_dependent"])

    def test_extract_shaping_audit_gap_zero_interpretation_says_matches_or_exceeds(self) -> None:
        summary = {
            "reward_audit": {
                "comparison": {
                    "minimal_profile": "austere",
                    "deltas_vs_minimal": {
                        "classic": {
                            "scenario_success_rate_delta": 0.0,
                            "episode_success_rate_delta": 0.0,
                            "mean_reward_delta": 0.0,
                        }
                    },
                }
            }
        }
        result = extract_shaping_audit(summary)
        self.assertIn("matches or exceeds", result["interpretation"])

    def test_extract_shaping_audit_gap_below_threshold_but_positive_interpretation(self) -> None:
        summary = {
            "reward_audit": {
                "comparison": {
                    "minimal_profile": "austere",
                    "deltas_vs_minimal": {
                        "classic": {
                            "scenario_success_rate_delta": SMALL_SHAPING_GAP,
                            "episode_success_rate_delta": 0.0,
                            "mean_reward_delta": 0.0,
                        }
                    },
                }
            }
        }
        result = extract_shaping_audit(summary)
        self.assertIn("below", result["interpretation"])
        self.assertFalse(result["interpretive_flags"]["shaping_dependent"])

    def test_extract_shaping_audit_high_gap_interpretation_mentions_high_shaping(self) -> None:
        summary = {
            "reward_audit": {
                "comparison": {
                    "minimal_profile": "austere",
                    "deltas_vs_minimal": {
                        "classic": {
                            "scenario_success_rate_delta": LARGE_SHAPING_GAP,
                            "episode_success_rate_delta": 0.0,
                            "mean_reward_delta": 0.0,
                        }
                    },
                }
            }
        }
        result = extract_shaping_audit(summary)
        self.assertIn("High shaping dependence", result["interpretation"])

    def test_extract_shaping_audit_falls_back_to_non_minimal_profile_as_dense(self) -> None:
        # If 'classic' is not in deltas_vs_minimal, picks another non-minimal profile
        summary = {
            "reward_audit": {
                "comparison": {
                    "minimal_profile": "austere",
                    "deltas_vs_minimal": {
                        "ecological": {
                            "scenario_success_rate_delta": 0.15,
                            "episode_success_rate_delta": 0.10,
                            "mean_reward_delta": 0.5,
                        },
                        "austere": {
                            "scenario_success_rate_delta": 0.0,
                            "episode_success_rate_delta": 0.0,
                            "mean_reward_delta": 0.0,
                        },
                    },
                }
            }
        }
        result = extract_shaping_audit(summary)
        self.assertEqual(result["dense_profile"], "ecological")
        self.assertTrue(result["interpretive_flags"]["gap_available"])

    def test_extract_shaping_audit_limitation_when_no_component_classification(self) -> None:
        # reward_audit present but no reward_components → limitation listed
        summary = {
            "reward_audit": {
                "comparison": {
                    "minimal_profile": "austere",
                    "deltas_vs_minimal": {
                        "classic": {
                            "scenario_success_rate_delta": 0.1,
                            "episode_success_rate_delta": 0.0,
                            "mean_reward_delta": 0.0,
                        }
                    },
                }
            }
        }
        result = extract_shaping_audit(summary)
        self.assertIn("No reward component disposition table was available.", result["limitations"])

    def test_extract_shaping_audit_limitation_when_no_behavior_survival(self) -> None:
        summary = {
            "reward_audit": {
                "comparison": {
                    "minimal_profile": "austere",
                    "deltas_vs_minimal": {
                        "classic": {
                            "scenario_success_rate_delta": 0.1,
                            "episode_success_rate_delta": 0.0,
                            "mean_reward_delta": 0.0,
                        }
                    },
                }
            }
        }
        result = extract_shaping_audit(summary)
        self.assertIn(
            "No austere per-scenario behavior survival data was available.",
            result["limitations"],
        )

    def test_extract_shaping_audit_removed_weight_gap_computed_from_profiles(self) -> None:
        summary = {
            "reward_audit": {
                "comparison": {
                    "minimal_profile": "austere",
                    "deltas_vs_minimal": {
                        "classic": {
                            "scenario_success_rate_delta": 0.1,
                            "episode_success_rate_delta": 0.0,
                            "mean_reward_delta": 0.0,
                        }
                    },
                },
                "reward_profiles": {
                    "classic": {
                        "disposition_summary": {
                            "removed": {"total_weight_proxy": 2.5},
                        }
                    },
                    "austere": {
                        "disposition_summary": {
                            "removed": {"total_weight_proxy": 0.0},
                        }
                    },
                },
            }
        }
        result = extract_shaping_audit(summary)
        self.assertAlmostEqual(result["removed_weight_gap"], 2.5)

    def test_extract_shaping_audit_source_is_summary_reward_audit(self) -> None:
        summary = {
            "reward_audit": {
                "comparison": {
                    "deltas_vs_minimal": {
                        "classic": {
                            "scenario_success_rate_delta": 0.1,
                            "episode_success_rate_delta": 0.0,
                            "mean_reward_delta": 0.0,
                        }
                    }
                }
            }
        }
        result = extract_shaping_audit(summary)
        self.assertEqual(result["source"], "summary.reward_audit")

    def test_extract_shaping_audit_behavior_survival_normalized(self) -> None:
        summary = {
            "reward_audit": {
                "comparison": {
                    "minimal_profile": "austere",
                    "deltas_vs_minimal": {
                        "classic": {
                            "scenario_success_rate_delta": 0.1,
                            "episode_success_rate_delta": 0.0,
                            "mean_reward_delta": 0.0,
                        }
                    },
                    "behavior_survival": {
                        "available": True,
                        "minimal_profile": "austere",
                        "survival_threshold": 0.5,
                        "scenario_count": 2,
                        "surviving_scenario_count": 1,
                        "survival_rate": 0.5,
                        "scenarios": {
                            "night_rest": {"austere_success_rate": 1.0, "survives": True, "episodes": 2},
                            "open_field": {"austere_success_rate": 0.0, "survives": False, "episodes": 2},
                        },
                    },
                }
            }
        }
        result = extract_shaping_audit(summary)
        bs = result["behavior_survival"]
        self.assertTrue(bs["available"])
        self.assertEqual(bs["surviving_scenario_count"], 1)
        self.assertAlmostEqual(bs["survival_rate"], 0.5)

    def test_extract_shaping_audit_no_limitations_when_all_data_present(self) -> None:
        summary = {
            "reward_audit": {
                "comparison": {
                    "minimal_profile": "austere",
                    "deltas_vs_minimal": {
                        "classic": {
                            "scenario_success_rate_delta": 0.1,
                            "episode_success_rate_delta": 0.0,
                            "mean_reward_delta": 0.0,
                        }
                    },
                    "behavior_survival": {
                        "available": True,
                        "scenarios": {
                            "night_rest": {"austere_success_rate": 1.0, "survives": True, "episodes": 1}
                        },
                    },
                },
                "reward_components": {
                    "food_progress": {
                        "category": "progress",
                        "shaping_risk": "high",
                        "shaping_disposition": "removed",
                        "disposition_rationale": "Zeroed in austere.",
                    }
                },
            }
        }
        result = extract_shaping_audit(summary)
        self.assertEqual(result["limitations"], [])

    def test_extract_shaping_audit_write_report_no_warning_for_gap_at_threshold(self) -> None:
        summary = {
            "reward_audit": {
                "comparison": {
                    "minimal_profile": "austere",
                    "deltas_vs_minimal": {
                        "classic": {
                            "scenario_success_rate_delta": SHAPING_DEPENDENCE_WARNING_THRESHOLD,
                            "episode_success_rate_delta": 0.0,
                            "mean_reward_delta": 0.0,
                        }
                    },
                }
            }
        }
        report = build_report_data(summary=summary, trace=[], behavior_rows=[])
        with tempfile.TemporaryDirectory() as tmpdir:
            write_report(tmpdir, report)
            report_md = (Path(tmpdir) / "report.md").read_text(encoding="utf-8")
        self.assertNotIn("WARNING: High shaping dependence detected", report_md)

    def test_write_report_mentions_missing_survival_when_shaping_program_exists(self) -> None:
        summary = {
            "reward_audit": {
                "comparison": {
                    "minimal_profile": "austere",
                    "deltas_vs_minimal": {
                        "classic": {
                            "scenario_success_rate_delta": 0.0,
                            "episode_success_rate_delta": 0.0,
                            "mean_reward_delta": 0.0,
                        }
                    },
                }
            }
        }
        report = build_report_data(summary=summary, trace=[], behavior_rows=[])
        with tempfile.TemporaryDirectory() as tmpdir:
            write_report(tmpdir, report)
            report_md = (Path(tmpdir) / "report.md").read_text(encoding="utf-8")
        self.assertIn("_No shaping survival data available._", report_md)

    def test_extract_shaping_audit_write_report_warning_just_above_threshold(self) -> None:
        summary = {
            "reward_audit": {
                "comparison": {
                    "minimal_profile": "austere",
                    "deltas_vs_minimal": {
                        "classic": {
                            "scenario_success_rate_delta": SHAPING_DEPENDENCE_WARNING_THRESHOLD + 0.001,
                            "episode_success_rate_delta": 0.0,
                            "mean_reward_delta": 0.0,
                        }
                    },
                }
            }
        }
        report = build_report_data(summary=summary, trace=[], behavior_rows=[])
        with tempfile.TemporaryDirectory() as tmpdir:
            write_report(tmpdir, report)
            report_md = (Path(tmpdir) / "report.md").read_text(encoding="utf-8")
        self.assertIn("WARNING: High shaping dependence detected", report_md)

    def test_extract_shaping_audit_build_report_shaping_program_unavailable_without_reward_audit(self) -> None:
        report = build_report_data(summary={}, trace=[], behavior_rows=[])
        self.assertIn("shaping_program", report)
        self.assertFalse(report["shaping_program"]["available"])

    def test_extract_shaping_audit_behavior_survival_scenarios_as_list_normalized(self) -> None:
        summary = {
            "reward_audit": {
                "comparison": {
                    "minimal_profile": "austere",
                    "deltas_vs_minimal": {
                        "classic": {
                            "scenario_success_rate_delta": 0.1,
                            "episode_success_rate_delta": 0.0,
                            "mean_reward_delta": 0.0,
                        }
                    },
                    "behavior_survival": {
                        "available": True,
                        "scenarios": [
                            {
                                "scenario": "night_rest",
                                "austere_success_rate": 1.0,
                                "survives": True,
                                "episodes": 5,
                            }
                        ],
                    },
                }
            }
        }
        result = extract_shaping_audit(summary)
        bs = result["behavior_survival"]
        self.assertTrue(bs["available"])
        scenarios = bs["scenarios"]
        self.assertEqual(len(scenarios), 1)
        self.assertEqual(scenarios[0]["scenario"], "night_rest")


class OfflineAnalysisReflexFrequencyTest(unittest.TestCase):
    def test_extract_reflex_frequency_uses_messages_without_debug(self) -> None:
        trace = [
            {
                "messages": [
                    {
                        "sender": "alert_center",
                        "topic": "action.proposal",
                        "payload": {"reflex": {"action": "MOVE_LEFT", "reason": "threat"}},
                    }
                ]
            },
            {
                "messages": [
                    {
                        "sender": "sleep_center",
                        "topic": "action.proposal",
                        "payload": {"reflex": {"action": "STAY", "reason": "rest"}},
                    }
                ]
            },
        ]
        result = extract_reflex_frequency(trace)

        self.assertTrue(result["available"])
        modules = {item["module"]: item for item in result["modules"]}
        self.assertEqual(modules["alert_center"]["reflex_events"], 1)
        self.assertEqual(modules["sleep_center"]["reflex_events"], 1)
        self.assertFalse(result["uses_debug_reflexes"])

    def test_extract_reflex_frequency_enriches_with_debug_payload(self) -> None:
        trace = [
            {
                "messages": [],
                "debug": {
                    "reflexes": {
                        "alert_center": {
                            "reflex": {"action": "MOVE_LEFT", "reason": "threat"},
                            "module_reflex_override": True,
                            "module_reflex_dominance": 0.75,
                        }
                    }
                },
            }
        ]
        result = extract_reflex_frequency(trace)

        modules = {item["module"]: item for item in result["modules"]}
        self.assertTrue(result["uses_debug_reflexes"])
        self.assertEqual(modules["alert_center"]["debug_reflex_events"], 1)
        self.assertGreater(modules["alert_center"]["override_rate"], 0.0)
        self.assertGreater(modules["alert_center"]["mean_dominance"], 0.0)


class OfflineAnalysisOutputTest(unittest.TestCase):
    def test_run_offline_analysis_writes_expected_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "report"
            report = run_offline_analysis(
                summary_path=CHECKIN_SUMMARY,
                trace_path=CHECKIN_TRACE,
                output_dir=output_dir,
            )

            generated = report["generated_files"]
            self.assertTrue((output_dir / "report.md").exists())
            self.assertTrue((output_dir / "report.json").exists())
            self.assertTrue((output_dir / "training_eval.svg").exists())
            self.assertTrue((output_dir / "scenario_success.svg").exists())
            self.assertTrue((output_dir / "scenario_checks.csv").exists())
            self.assertTrue((output_dir / "reward_components.csv").exists())
            self.assertTrue(generated["reflex_frequency_svg"])

    def test_run_offline_analysis_report_mentions_real_variant_and_scenario(self) -> None:
        sim = SpiderSimulation(seed=5, max_steps=8)
        _, _, rows = sim.evaluate_behavior_suite(["night_rest"])
        rows.append(
            {
                "reward_profile": "classic",
                "scenario_map": "central_burrow",
                "evaluation_map": "central_burrow",
                "ablation_variant": "monolithic_policy",
                "ablation_architecture": "monolithic",
                "reflex_scale": 0.0,
                "reflex_anneal_final_scale": 0.0,
                "eval_reflex_scale": 0.0,
                "budget_profile": "smoke",
                "benchmark_strength": "quick",
                "operational_profile": "default_v1",
                "operational_profile_version": 1,
                "checkpoint_source": "final",
                "simulation_seed": 7,
                "episode_seed": 7001,
                "scenario": "night_rest",
                "scenario_description": "night rest",
                "scenario_objective": "sleep",
                "scenario_focus": "sleep",
                "episode": 0,
                "success": False,
                "failure_count": 1,
                "failures": "check_missing",
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "behavior.csv"
            SpiderSimulation.save_behavior_csv(rows, csv_path)
            output_dir = Path(tmpdir) / "report"
            run_offline_analysis(
                behavior_csv_path=csv_path,
                output_dir=output_dir,
            )
            report_md = (output_dir / "report.md").read_text(encoding="utf-8")
            report_json = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))

        self.assertIn("night_rest", report_md)
        self.assertIn("monolithic_policy", report_md)
        self.assertIn("night_rest", json.dumps(report_json, ensure_ascii=False))
        self.assertIn("monolithic_policy", json.dumps(report_json, ensure_ascii=False))


class OrderedNoiseConditionsTest(unittest.TestCase):
    """Tests for _ordered_noise_conditions helper (new in this PR)."""

    def test_canonical_conditions_kept_in_canonical_order(self) -> None:
        result = _ordered_noise_conditions(["high", "none", "medium", "low"])
        self.assertEqual(result, ["none", "low", "medium", "high"])

    def test_extra_conditions_sorted_alphabetically_after_canonicals(self) -> None:
        result = _ordered_noise_conditions(["beta", "none", "alpha"])
        self.assertEqual(result, ["none", "alpha", "beta"])

    def test_empty_input_returns_empty_list(self) -> None:
        self.assertEqual(_ordered_noise_conditions([]), [])

    def test_duplicate_names_deduplicated(self) -> None:
        result = _ordered_noise_conditions(["none", "low", "none", "low"])
        self.assertEqual(result, ["none", "low"])

    def test_falsy_names_are_excluded(self) -> None:
        result = _ordered_noise_conditions(["", "none", ""])
        self.assertEqual(result, ["none"])

    def test_non_canonical_only_returns_sorted_list(self) -> None:
        result = _ordered_noise_conditions(["zeta", "alpha"])
        self.assertEqual(result, ["alpha", "zeta"])

    def test_subset_of_canonical_conditions_preserved_in_order(self) -> None:
        result = _ordered_noise_conditions(["high", "low"])
        self.assertEqual(result, ["low", "high"])

    def test_all_canonical_conditions_are_returned_in_full_canonical_order(self) -> None:
        result = _ordered_noise_conditions(["high", "medium", "low", "none"])
        self.assertEqual(result, ["none", "low", "medium", "high"])


class NoiseRobustnessCellSummaryTest(unittest.TestCase):
    """Tests for _noise_robustness_cell_summary helper (new in this PR)."""

    def test_reads_scenario_and_episode_success_rate_from_summary(self) -> None:
        payload = {
            "summary": {
                "scenario_success_rate": 0.8,
                "episode_success_rate": 0.75,
                "mean_reward": 1.5,
                "scenario_count": 3,
                "episode_count": 12,
            }
        }
        result = _noise_robustness_cell_summary(payload)
        self.assertAlmostEqual(result["scenario_success_rate"], 0.8)
        self.assertAlmostEqual(result["episode_success_rate"], 0.75)
        self.assertAlmostEqual(result["mean_reward"], 1.5)
        self.assertEqual(result["scenario_count"], 3)
        self.assertEqual(result["episode_count"], 12)

    def test_empty_summary_returns_zeros(self) -> None:
        result = _noise_robustness_cell_summary({})
        self.assertAlmostEqual(result["scenario_success_rate"], 0.0)
        self.assertAlmostEqual(result["episode_success_rate"], 0.0)
        self.assertAlmostEqual(result["mean_reward"], 0.0)

    def test_falls_back_to_legacy_scenarios_mean_reward_when_summary_missing(self) -> None:
        payload = {
            "legacy_scenarios": {
                "night_rest": {"mean_reward": 2.0},
                "day_forage": {"mean_reward": 4.0},
            }
        }
        result = _noise_robustness_cell_summary(payload)
        self.assertAlmostEqual(result["mean_reward"], 3.0)

    def test_scenario_count_falls_back_to_legacy_scenario_count(self) -> None:
        payload = {
            "legacy_scenarios": {
                "night_rest": {"mean_reward": 1.0},
                "day_forage": {"mean_reward": 1.0},
            }
        }
        result = _noise_robustness_cell_summary(payload)
        self.assertEqual(result["scenario_count"], 2)

    def test_missing_legacy_and_summary_mean_reward_defaults_to_zero(self) -> None:
        payload = {"summary": {"scenario_success_rate": 0.5}}
        result = _noise_robustness_cell_summary(payload)
        self.assertAlmostEqual(result["mean_reward"], 0.0)

    def test_non_mapping_legacy_scenario_items_are_ignored(self) -> None:
        payload = {
            "legacy_scenarios": {
                "good": {"mean_reward": 2.0},
                "bad": "not_a_mapping",
            }
        }
        result = _noise_robustness_cell_summary(payload)
        self.assertAlmostEqual(result["mean_reward"], 2.0)


class NormalizeNoiseMarginalTest(unittest.TestCase):
    """Tests for _normalize_noise_marginals helper (new in this PR)."""

    def test_uses_payload_values_when_present(self) -> None:
        result = _normalize_noise_marginals(
            {"none": 0.9, "low": 0.7},
            conditions=["none", "low"],
            fallback={"none": 0.1, "low": 0.2},
        )
        self.assertAlmostEqual(result["none"], 0.9)
        self.assertAlmostEqual(result["low"], 0.7)

    def test_falls_back_when_condition_missing_from_payload(self) -> None:
        result = _normalize_noise_marginals(
            {"none": 0.9},
            conditions=["none", "low"],
            fallback={"none": 0.1, "low": 0.5},
        )
        self.assertAlmostEqual(result["none"], 0.9)
        self.assertAlmostEqual(result["low"], 0.5)

    def test_non_mapping_payload_uses_fallback_entirely(self) -> None:
        result = _normalize_noise_marginals(
            None,
            conditions=["none", "low"],
            fallback={"none": 0.3, "low": 0.6},
        )
        self.assertAlmostEqual(result["none"], 0.3)
        self.assertAlmostEqual(result["low"], 0.6)

    def test_empty_conditions_returns_empty_dict(self) -> None:
        result = _normalize_noise_marginals(
            {"none": 0.5},
            conditions=[],
            fallback={},
        )
        self.assertEqual(result, {})

    def test_missing_from_both_payload_and_fallback_defaults_to_zero(self) -> None:
        result = _normalize_noise_marginals(
            {},
            conditions=["none"],
            fallback={},
        )
        self.assertAlmostEqual(result["none"], 0.0)


class NoiseRobustnessMetricsTest(unittest.TestCase):
    """Tests for _noise_robustness_metrics helper (new in this PR)."""

    def _make_matrix(self) -> dict:
        return {
            "none": {
                "none": {"scenario_success_rate": 1.0},
                "high": {"scenario_success_rate": 0.5},
            },
            "low": {
                "none": {"scenario_success_rate": 0.8},
                "high": {"scenario_success_rate": 0.2},
            },
        }

    def test_robustness_score_is_mean_of_all_cells(self) -> None:
        matrix = self._make_matrix()
        result = _noise_robustness_metrics(
            matrix,
            train_conditions=["none", "low"],
            eval_conditions=["none", "high"],
        )
        expected = (1.0 + 0.5 + 0.8 + 0.2) / 4
        self.assertAlmostEqual(result["robustness_score"], expected)

    def test_diagonal_score_is_mean_of_matched_cells(self) -> None:
        matrix = self._make_matrix()
        result = _noise_robustness_metrics(
            matrix,
            train_conditions=["none", "low"],
            eval_conditions=["none", "high"],
        )
        # diagonal cells: (none, none)=1.0; (low, low) is absent
        self.assertAlmostEqual(result["diagonal_score"], 1.0)

    def test_off_diagonal_score_excludes_matched_cells(self) -> None:
        matrix = self._make_matrix()
        result = _noise_robustness_metrics(
            matrix,
            train_conditions=["none", "low"],
            eval_conditions=["none", "high"],
        )
        # off-diagonal: (none, high)=0.5, (low, none)=0.8, (low, high)=0.2
        # Only (none, none) is diagonal; "low" != "high" so (low, high) is also off-diagonal
        expected = (0.5 + 0.8 + 0.2) / 3
        self.assertAlmostEqual(result["off_diagonal_score"], expected)

    def test_train_marginals_computed_per_row(self) -> None:
        matrix = self._make_matrix()
        result = _noise_robustness_metrics(
            matrix,
            train_conditions=["none", "low"],
            eval_conditions=["none", "high"],
        )
        self.assertAlmostEqual(result["train_marginals"]["none"], 0.75)
        self.assertAlmostEqual(result["train_marginals"]["low"], 0.5)

    def test_eval_marginals_computed_per_column(self) -> None:
        matrix = self._make_matrix()
        result = _noise_robustness_metrics(
            matrix,
            train_conditions=["none", "low"],
            eval_conditions=["none", "high"],
        )
        self.assertAlmostEqual(result["eval_marginals"]["none"], 0.9)
        self.assertAlmostEqual(result["eval_marginals"]["high"], 0.35)

    def test_available_cell_count_counts_present_cells(self) -> None:
        matrix = self._make_matrix()
        result = _noise_robustness_metrics(
            matrix,
            train_conditions=["none", "low"],
            eval_conditions=["none", "high"],
        )
        self.assertEqual(result["available_cell_count"], 4)

    def test_missing_cells_are_excluded_from_available_count(self) -> None:
        matrix = {
            "none": {"none": {"scenario_success_rate": 1.0}},
            "low": {},
        }
        result = _noise_robustness_metrics(
            matrix,
            train_conditions=["none", "low"],
            eval_conditions=["none", "high"],
        )
        self.assertEqual(result["available_cell_count"], 1)

    def test_empty_matrix_returns_zero_scores(self) -> None:
        result = _noise_robustness_metrics(
            {},
            train_conditions=["none"],
            eval_conditions=["low"],
        )
        self.assertAlmostEqual(result["robustness_score"], 0.0)
        self.assertAlmostEqual(result["diagonal_score"], 0.0)
        self.assertEqual(result["available_cell_count"], 0)

    def test_fully_diagonal_matrix_returns_zero_off_diagonal(self) -> None:
        matrix = {
            "none": {"none": {"scenario_success_rate": 0.9}},
            "low": {"low": {"scenario_success_rate": 0.7}},
        }
        result = _noise_robustness_metrics(
            matrix,
            train_conditions=["none", "low"],
            eval_conditions=["none", "low"],
        )
        self.assertAlmostEqual(result["diagonal_score"], 0.8)
        self.assertAlmostEqual(result["off_diagonal_score"], 0.0)


class RenderMatrixHeatmapTest(unittest.TestCase):
    """Tests for render_matrix_heatmap (new in this PR)."""

    def test_returns_svg_string(self) -> None:
        svg = render_matrix_heatmap(
            "Test heatmap",
            train_conditions=["none", "low"],
            eval_conditions=["none", "high"],
            matrix={
                "none": {"none": {"scenario_success_rate": 1.0}, "high": {"scenario_success_rate": 0.5}},
                "low": {"none": {"scenario_success_rate": 0.8}, "high": {"scenario_success_rate": 0.2}},
            },
            train_marginals={"none": 0.75, "low": 0.5},
            eval_marginals={"none": 0.9, "high": 0.35},
        )
        self.assertTrue(svg.startswith("<svg"))
        self.assertIn("</svg>", svg)

    def test_svg_contains_title(self) -> None:
        svg = render_matrix_heatmap(
            "Noise robustness matrix",
            train_conditions=["none"],
            eval_conditions=["none"],
            matrix={"none": {"none": {"scenario_success_rate": 0.5}}},
            train_marginals={"none": 0.5},
            eval_marginals={"none": 0.5},
        )
        self.assertIn("Noise robustness matrix", svg)

    def test_empty_train_conditions_returns_placeholder_svg(self) -> None:
        svg = render_matrix_heatmap(
            "Empty heatmap",
            train_conditions=[],
            eval_conditions=["none"],
            matrix={},
            train_marginals={},
            eval_marginals={},
        )
        self.assertIn("No data available", svg)

    def test_empty_eval_conditions_returns_placeholder_svg(self) -> None:
        svg = render_matrix_heatmap(
            "Empty heatmap",
            train_conditions=["none"],
            eval_conditions=[],
            matrix={},
            train_marginals={},
            eval_marginals={},
        )
        self.assertIn("No data available", svg)

    def test_svg_contains_train_and_eval_condition_labels(self) -> None:
        svg = render_matrix_heatmap(
            "Heatmap",
            train_conditions=["none", "high"],
            eval_conditions=["low", "medium"],
            matrix={
                "none": {
                    "low": {"scenario_success_rate": 0.9},
                    "medium": {"scenario_success_rate": 0.7},
                },
                "high": {
                    "low": {"scenario_success_rate": 0.6},
                    "medium": {"scenario_success_rate": 0.4},
                },
            },
            train_marginals={"none": 0.8, "high": 0.5},
            eval_marginals={"low": 0.75, "medium": 0.55},
        )
        self.assertIn("none", svg)
        self.assertIn("high", svg)
        self.assertIn("low", svg)
        self.assertIn("medium", svg)

    def test_svg_contains_mean_row_and_column(self) -> None:
        svg = render_matrix_heatmap(
            "Heatmap",
            train_conditions=["none"],
            eval_conditions=["none"],
            matrix={"none": {"none": {"scenario_success_rate": 1.0}}},
            train_marginals={"none": 1.0},
            eval_marginals={"none": 1.0},
        )
        self.assertIn("mean", svg)

    def test_svg_includes_cell_values_as_decimals(self) -> None:
        svg = render_matrix_heatmap(
            "Heatmap",
            train_conditions=["none"],
            eval_conditions=["low"],
            matrix={"none": {"low": {"scenario_success_rate": 0.75}}},
            train_marginals={"none": 0.75},
            eval_marginals={"low": 0.75},
        )
        self.assertIn("0.75", svg)

    def test_title_special_chars_are_escaped(self) -> None:
        svg = render_matrix_heatmap(
            "Test <&> title",
            train_conditions=["none"],
            eval_conditions=["none"],
            matrix={"none": {"none": {"scenario_success_rate": 0.5}}},
            train_marginals={"none": 0.5},
            eval_marginals={"none": 0.5},
        )
        self.assertIn("&lt;", svg)
        self.assertIn("&amp;", svg)


class NormalizeBehaviorRowsNoiseFieldsTest(unittest.TestCase):
    """Tests for train/eval noise profile fields added to normalize_behavior_rows (new in this PR)."""

    def test_train_noise_profile_is_normalized_from_row(self) -> None:
        rows = normalize_behavior_rows([{"train_noise_profile": "low"}])
        self.assertEqual(rows[0]["train_noise_profile"], "low")

    def test_eval_noise_profile_is_normalized_from_row(self) -> None:
        rows = normalize_behavior_rows([{"eval_noise_profile": "high"}])
        self.assertEqual(rows[0]["eval_noise_profile"], "high")

    def test_eval_noise_profile_falls_back_to_noise_profile_key(self) -> None:
        rows = normalize_behavior_rows([{"noise_profile": "medium"}])
        self.assertEqual(rows[0]["eval_noise_profile"], "medium")

    def test_eval_noise_profile_prefers_explicit_key_over_noise_profile(self) -> None:
        rows = normalize_behavior_rows([
            {"eval_noise_profile": "low", "noise_profile": "high"}
        ])
        self.assertEqual(rows[0]["eval_noise_profile"], "low")

    def test_missing_noise_profile_fields_default_to_empty_string(self) -> None:
        rows = normalize_behavior_rows([{"scenario": "night_rest"}])
        self.assertEqual(rows[0]["train_noise_profile"], "")
        self.assertEqual(rows[0]["eval_noise_profile"], "")

    def test_none_train_noise_profile_coerced_to_empty_string(self) -> None:
        rows = normalize_behavior_rows([{"train_noise_profile": None}])
        self.assertEqual(rows[0]["train_noise_profile"], "")

    def test_none_eval_noise_profile_coerced_to_empty_string(self) -> None:
        rows = normalize_behavior_rows([{"eval_noise_profile": None}])
        self.assertEqual(rows[0]["eval_noise_profile"], "")

    def test_both_profiles_present_in_normalized_row_keys(self) -> None:
        rows = normalize_behavior_rows([{}])
        self.assertIn("train_noise_profile", rows[0])
        self.assertIn("eval_noise_profile", rows[0])

    def test_noise_profile_fields_are_strings(self) -> None:
        rows = normalize_behavior_rows([
            {"train_noise_profile": "low", "eval_noise_profile": "high"}
        ])
        self.assertIsInstance(rows[0]["train_noise_profile"], str)
        self.assertIsInstance(rows[0]["eval_noise_profile"], str)


class ExtractNoiseRobustnessEdgeCasesTest(unittest.TestCase):
    """Additional edge case and regression tests for extract_noise_robustness (new in this PR)."""

    def test_summary_without_matrix_spec_infers_conditions_from_matrix_keys(self) -> None:
        summary = {
            "behavior_evaluation": {
                "robustness_matrix": {
                    "matrix": {
                        "low": {
                            "high": {"summary": {"scenario_success_rate": 0.6}},
                        },
                    },
                }
            }
        }
        result = extract_noise_robustness(summary, [])
        self.assertIn("low", result["matrix_spec"]["train_conditions"])
        self.assertIn("high", result["matrix_spec"]["eval_conditions"])

    def test_behavior_rows_with_partial_noise_metadata_count_as_missing(self) -> None:
        rows = normalize_behavior_rows([
            {
                "scenario": "night_rest",
                "success": True,
                "noise_profile": "none",
                "train_noise_profile": "",
                "eval_noise_profile": "",
            }
        ])
        result = extract_noise_robustness({}, rows)
        self.assertFalse(result["available"])
        self.assertEqual(result["source"], "none")

    def test_diagonal_and_off_diagonal_scores_for_full_canonical_matrix(self) -> None:
        summary = {
            "behavior_evaluation": {
                "robustness_matrix": {
                    "matrix_spec": {
                        "train_conditions": ["none", "low"],
                        "eval_conditions": ["none", "low"],
                        "cell_count": 4,
                    },
                    "matrix": {
                        "none": {
                            "none": {"summary": {"scenario_success_rate": 0.9}},
                            "low": {"summary": {"scenario_success_rate": 0.5}},
                        },
                        "low": {
                            "none": {"summary": {"scenario_success_rate": 0.6}},
                            "low": {"summary": {"scenario_success_rate": 0.8}},
                        },
                    },
                    "robustness_score": 0.7,
                    "diagonal_score": 0.85,
                    "off_diagonal_score": 0.55,
                }
            }
        }
        result = extract_noise_robustness(summary, [])
        # summary-provided scores are used when present
        self.assertAlmostEqual(result["robustness_score"], 0.7)
        self.assertAlmostEqual(result["diagonal_score"], 0.85)
        self.assertAlmostEqual(result["off_diagonal_score"], 0.55)

    def test_behavior_csv_source_computes_episode_count_per_cell(self) -> None:
        rows = normalize_behavior_rows([
            {"scenario": "night_rest", "success": True, "train_noise_profile": "none", "eval_noise_profile": "none"},
            {"scenario": "night_rest", "success": True, "train_noise_profile": "none", "eval_noise_profile": "none"},
            {"scenario": "night_rest", "success": False, "train_noise_profile": "none", "eval_noise_profile": "none"},
        ])
        result = extract_noise_robustness({}, rows)
        self.assertEqual(result["matrix"]["none"]["none"]["episode_count"], 3)
        self.assertAlmostEqual(result["matrix"]["none"]["none"]["scenario_success_rate"], 2 / 3)

    def test_metadata_complete_true_when_all_cells_present(self) -> None:
        summary = {
            "behavior_evaluation": {
                "robustness_matrix": {
                    "matrix_spec": {
                        "train_conditions": ["none"],
                        "eval_conditions": ["none"],
                        "cell_count": 1,
                    },
                    "matrix": {
                        "none": {
                            "none": {"summary": {"scenario_success_rate": 1.0}},
                        }
                    },
                }
            }
        }
        result = extract_noise_robustness(summary, [])
        self.assertTrue(result["metadata"]["complete"])
        self.assertEqual(result["metadata"]["available_cell_count"], 1)
        self.assertEqual(result["metadata"]["expected_cell_count"], 1)


class NormalizeModuleResponseByPredatorTypeTest(unittest.TestCase):
    """Tests for offline_analysis._normalize_module_response_by_predator_type() - new in this PR."""

    def test_valid_nested_mapping_converts_values_to_float(self) -> None:
        value = {
            "visual": {"visual_cortex": 0.7, "sensory_cortex": 0.3},
            "olfactory": {"visual_cortex": 0.2, "sensory_cortex": 0.8},
        }
        result = _normalize_module_response_by_predator_type(value)
        self.assertAlmostEqual(result["visual"]["visual_cortex"], 0.7)
        self.assertAlmostEqual(result["olfactory"]["sensory_cortex"], 0.8)

    def test_non_mapping_input_returns_empty(self) -> None:
        result = _normalize_module_response_by_predator_type("not_a_mapping")
        self.assertEqual(result, {})

    def test_none_input_returns_empty(self) -> None:
        result = _normalize_module_response_by_predator_type(None)
        self.assertEqual(result, {})

    def test_list_input_returns_empty(self) -> None:
        result = _normalize_module_response_by_predator_type([1, 2, 3])
        self.assertEqual(result, {})

    def test_non_mapping_predator_entry_is_skipped(self) -> None:
        value = {
            "visual": "not_a_mapping",
            "olfactory": {"sensory_cortex": 0.5},
        }
        result = _normalize_module_response_by_predator_type(value)
        self.assertNotIn("visual", result)
        self.assertIn("olfactory", result)

    def test_keys_are_strings(self) -> None:
        value = {"visual": {"a": 1.0}}
        result = _normalize_module_response_by_predator_type(value)
        for key in result:
            self.assertIsInstance(key, str)
        for key in result.get("visual", {}):
            self.assertIsInstance(key, str)

    def test_values_are_floats(self) -> None:
        value = {"visual": {"module_a": 1}}
        result = _normalize_module_response_by_predator_type(value)
        for val in result.get("visual", {}).values():
            self.assertIsInstance(val, float)

    def test_empty_mapping_returns_empty(self) -> None:
        result = _normalize_module_response_by_predator_type({})
        self.assertEqual(result, {})

    def test_invalid_value_coerced_to_zero(self) -> None:
        value = {"visual": {"module_a": None}}
        result = _normalize_module_response_by_predator_type(value)
        self.assertAlmostEqual(result["visual"]["module_a"], 0.0)


class AggregateSpecializationFromScenariosTest(unittest.TestCase):
    """Tests for offline_analysis._aggregate_specialization_from_scenarios() - new in this PR."""

    def test_returns_mean_across_scenarios(self) -> None:
        scenarios = {
            "scenario_a": {
                "mean_module_response_by_predator_type": {
                    "visual": {"visual_cortex": 0.6, "sensory_cortex": 0.4},
                }
            },
            "scenario_b": {
                "mean_module_response_by_predator_type": {
                    "visual": {"visual_cortex": 0.8, "sensory_cortex": 0.2},
                }
            },
        }
        result = _aggregate_specialization_from_scenarios(scenarios)
        self.assertIn("visual", result)
        self.assertAlmostEqual(result["visual"]["visual_cortex"], 0.7)
        self.assertAlmostEqual(result["visual"]["sensory_cortex"], 0.3)

    def test_empty_scenarios_returns_empty(self) -> None:
        result = _aggregate_specialization_from_scenarios({})
        self.assertEqual(result, {})

    def test_non_mapping_scenario_payload_is_skipped(self) -> None:
        scenarios = {
            "scenario_a": "not_a_mapping",
            "scenario_b": {
                "mean_module_response_by_predator_type": {
                    "visual": {"visual_cortex": 0.5},
                }
            },
        }
        result = _aggregate_specialization_from_scenarios(scenarios)
        self.assertAlmostEqual(result["visual"]["visual_cortex"], 0.5)

    def test_falls_back_to_module_response_by_predator_type_key(self) -> None:
        scenarios = {
            "scenario_a": {
                "module_response_by_predator_type": {
                    "olfactory": {"sensory_cortex": 0.9},
                }
            }
        }
        result = _aggregate_specialization_from_scenarios(scenarios)
        self.assertIn("olfactory", result)
        self.assertAlmostEqual(result["olfactory"]["sensory_cortex"], 0.9)

    def test_aggregates_across_predator_types(self) -> None:
        scenarios = {
            "s1": {
                "mean_module_response_by_predator_type": {
                    "visual": {"vc": 0.7},
                    "olfactory": {"sc": 0.8},
                }
            }
        }
        result = _aggregate_specialization_from_scenarios(scenarios)
        self.assertIn("visual", result)
        self.assertIn("olfactory", result)


class ExtractPredatorTypeSpecializationEdgeCasesTest(unittest.TestCase):
    """Edge-case and boundary tests for extract_predator_type_specialization() - new in this PR."""

    def test_empty_summary_returns_unavailable(self) -> None:
        result = extract_predator_type_specialization({}, [], [])
        self.assertFalse(result["available"])
        self.assertEqual(result["source"], "none")

    def test_unavailable_result_has_required_keys(self) -> None:
        result = extract_predator_type_specialization({}, [], [])
        for key in ("available", "source", "predator_types", "differential_activation",
                    "type_module_correlation", "specialization_score", "interpretation", "limitations"):
            self.assertIn(key, result)

    def test_unavailable_result_has_unavailable_interpretation(self) -> None:
        result = extract_predator_type_specialization({}, [], [])
        self.assertEqual(result["interpretation"], "unavailable")

    def test_low_specialization_score_interpretation(self) -> None:
        summary = {
            "evaluation": {
                "mean_module_response_by_predator_type": {
                    "visual": {"visual_cortex": 0.5, "sensory_cortex": 0.5},
                    "olfactory": {"visual_cortex": 0.5, "sensory_cortex": 0.5},
                }
            }
        }
        result = extract_predator_type_specialization(summary, [], [])
        self.assertTrue(result["available"])
        self.assertIn(result["interpretation"], ("low", "moderate", "high"))
        # Same distributions -> specialization near 0 -> low
        self.assertEqual(result["interpretation"], "low")

    def test_high_specialization_score_interpretation(self) -> None:
        summary = {
            "evaluation": {
                "mean_module_response_by_predator_type": {
                    "visual": {"visual_cortex": 0.95, "sensory_cortex": 0.05},
                    "olfactory": {"visual_cortex": 0.05, "sensory_cortex": 0.95},
                }
            }
        }
        result = extract_predator_type_specialization(summary, [], [])
        self.assertTrue(result["available"])
        self.assertEqual(result["interpretation"], "high")
        self.assertGreater(result["specialization_score"], 0.5)

    def test_moderate_specialization_score_interpretation(self) -> None:
        summary = {
            "evaluation": {
                "mean_module_response_by_predator_type": {
                    "visual": {"visual_cortex": 0.65, "sensory_cortex": 0.35},
                    "olfactory": {"visual_cortex": 0.30, "sensory_cortex": 0.70},
                }
            }
        }
        result = extract_predator_type_specialization(summary, [], [])
        self.assertTrue(result["available"])
        self.assertEqual(result["interpretation"], "moderate")

    def test_specialization_score_bounded_between_zero_and_one(self) -> None:
        summary = {
            "evaluation": {
                "mean_module_response_by_predator_type": {
                    "visual": {"visual_cortex": 1.0, "sensory_cortex": 0.0},
                    "olfactory": {"visual_cortex": 0.0, "sensory_cortex": 1.0},
                }
            }
        }
        result = extract_predator_type_specialization(summary, [], [])
        self.assertGreaterEqual(result["specialization_score"], 0.0)
        self.assertLessEqual(result["specialization_score"], 1.0)

    def test_type_module_correlation_bounded_between_zero_and_one(self) -> None:
        summary = {
            "evaluation": {
                "mean_module_response_by_predator_type": {
                    "visual": {"visual_cortex": 0.8, "sensory_cortex": 0.2},
                    "olfactory": {"visual_cortex": 0.2, "sensory_cortex": 0.8},
                }
            }
        }
        result = extract_predator_type_specialization(summary, [], [])
        self.assertGreaterEqual(result["type_module_correlation"], 0.0)
        self.assertLessEqual(result["type_module_correlation"], 1.0)

    def test_falls_back_to_behavior_evaluation_legacy_scenarios(self) -> None:
        summary = {
            "behavior_evaluation": {
                "legacy_scenarios": {
                    "visual_olfactory_pincer": {
                        "module_response_by_predator_type": {
                            "visual": {"visual_cortex": 0.8, "sensory_cortex": 0.2},
                            "olfactory": {"visual_cortex": 0.2, "sensory_cortex": 0.8},
                        }
                    }
                }
            }
        }
        result = extract_predator_type_specialization(summary, [], [])
        self.assertTrue(result["available"])
        self.assertIn("legacy_scenarios", result["source"])

    def test_prefers_paired_suite_data_over_partial_earlier_candidate(self) -> None:
        summary = {
            "evaluation": {
                "mean_module_response_by_predator_type": {
                    "visual": {"visual_cortex": 0.9, "sensory_cortex": 0.1},
                }
            },
            "behavior_evaluation": {
                "suite": {
                    "visual_hunter_open_field": {
                        "module_response_by_predator_type": {
                            "visual": {"visual_cortex": 0.8, "sensory_cortex": 0.2},
                            "olfactory": {"visual_cortex": 0.2, "sensory_cortex": 0.8},
                        }
                    }
                }
            },
        }
        result = extract_predator_type_specialization(summary, [], [])
        self.assertTrue(result["available"])
        self.assertEqual(result["source"], "summary.behavior_evaluation.suite")
        self.assertGreater(result["specialization_score"], 0.0)
        self.assertNotEqual(result["interpretation"], "insufficient_data")

    def test_prefers_suite_over_legacy_when_both_have_paired_data(self) -> None:
        summary = {
            "behavior_evaluation": {
                "legacy_scenarios": {
                    "visual_olfactory_pincer": {
                        "module_response_by_predator_type": {
                            "visual": {"visual_cortex": 0.6, "sensory_cortex": 0.4},
                            "olfactory": {"visual_cortex": 0.5, "sensory_cortex": 0.5},
                        }
                    }
                },
                "suite": {
                    "visual_hunter_open_field": {
                        "module_response_by_predator_type": {
                            "visual": {"visual_cortex": 0.8, "sensory_cortex": 0.2},
                            "olfactory": {"visual_cortex": 0.2, "sensory_cortex": 0.8},
                        }
                    }
                },
            }
        }
        result = extract_predator_type_specialization(summary, [], [])
        self.assertTrue(result["available"])
        self.assertEqual(result["source"], "summary.behavior_evaluation.suite")

    def test_result_includes_differential_activation_keys(self) -> None:
        summary = {
            "evaluation": {
                "mean_module_response_by_predator_type": {
                    "visual": {"visual_cortex": 0.7, "sensory_cortex": 0.3},
                    "olfactory": {"visual_cortex": 0.2, "sensory_cortex": 0.8},
                }
            }
        }
        result = extract_predator_type_specialization(summary, [], [])
        self.assertIn("visual_cortex_visual_minus_olfactory", result["differential_activation"])
        self.assertIn("sensory_cortex_olfactory_minus_visual", result["differential_activation"])

    def test_result_predator_types_include_visual_and_olfactory(self) -> None:
        summary = {
            "evaluation": {
                "mean_module_response_by_predator_type": {
                    "visual": {"visual_cortex": 0.7, "sensory_cortex": 0.3},
                    "olfactory": {"visual_cortex": 0.2, "sensory_cortex": 0.8},
                }
            }
        }
        result = extract_predator_type_specialization(summary, [], [])
        self.assertIn("visual", result["predator_types"])
        self.assertIn("olfactory", result["predator_types"])

    def test_single_predator_type_returns_insufficient_data(self) -> None:
        summary = {
            "evaluation": {
                "mean_module_response_by_predator_type": {
                    "visual": {"visual_cortex": 0.8, "sensory_cortex": 0.2},
                }
            }
        }
        result = extract_predator_type_specialization(summary, [], [])
        self.assertTrue(result["available"])
        self.assertEqual(result["specialization_score"], 0.0)
        self.assertEqual(result["type_module_correlation"], 0.0)
        self.assertEqual(result["interpretation"], "insufficient_data")
        self.assertEqual(
            result["differential_activation"]["visual_cortex_visual_minus_olfactory"],
            0.0,
        )
        self.assertEqual(
            result["differential_activation"]["sensory_cortex_olfactory_minus_visual"],
            0.0,
        )
        self.assertTrue(
            any("both visual and olfactory predators" in limitation for limitation in result["limitations"])
        )

    def test_differential_activation_visual_cortex_positive_for_specialized(self) -> None:
        summary = {
            "evaluation": {
                "mean_module_response_by_predator_type": {
                    "visual": {"visual_cortex": 0.8, "sensory_cortex": 0.2},
                    "olfactory": {"visual_cortex": 0.1, "sensory_cortex": 0.9},
                }
            }
        }
        result = extract_predator_type_specialization(summary, [], [])
        differential = result["differential_activation"]
        # visual cortex responds more to visual predators -> positive
        self.assertGreater(differential["visual_cortex_visual_minus_olfactory"], 0.0)
        # sensory cortex responds more to olfactory predators -> positive
        self.assertGreater(differential["sensory_cortex_olfactory_minus_visual"], 0.0)

    def test_write_report_includes_differential_activation_table(self) -> None:
        summary = {
            "evaluation": {
                "mean_module_response_by_predator_type": {
                    "visual": {"visual_cortex": 0.75, "sensory_cortex": 0.25},
                    "olfactory": {"visual_cortex": 0.15, "sensory_cortex": 0.85},
                }
            }
        }
        report = build_report_data(summary=summary, trace=[], behavior_rows=[])
        with tempfile.TemporaryDirectory() as tmpdir:
            write_report(tmpdir, report)
            report_md = (Path(tmpdir) / "report.md").read_text(encoding="utf-8")
        self.assertIn("sensory_cortex (olfactory - visual)", report_md)
        self.assertIn("visual_cortex (visual - olfactory)", report_md)


class ExtractAblationsPredatorTypeComparisonsTest(unittest.TestCase):
    """Tests that extract_ablations includes predator_type_comparisons - new in this PR."""

    def test_extract_ablations_from_summary_includes_predator_type_comparisons(self) -> None:
        summary = {
            "behavior_evaluation": {
                "ablations": {
                    "reference_variant": "modular_full",
                    "variants": {
                        "modular_full": {
                            "suite": {
                                "visual_olfactory_pincer": {"success_rate": 0.8},
                                "olfactory_ambush": {"success_rate": 0.7},
                                "visual_hunter_open_field": {"success_rate": 0.9},
                            }
                        },
                        "drop_visual_cortex": {
                            "suite": {
                                "visual_olfactory_pincer": {"success_rate": 0.3},
                                "olfactory_ambush": {"success_rate": 0.7},
                                "visual_hunter_open_field": {"success_rate": 0.2},
                            }
                        },
                    },
                }
            }
        }
        result = extract_ablations(summary, [])
        self.assertIn("predator_type_comparisons", result)
        comparisons = result["predator_type_comparisons"]
        self.assertIn("available", comparisons)

    def test_extract_ablations_from_csv_includes_predator_type_comparisons(self) -> None:
        rows = normalize_behavior_rows([
            {
                "scenario": "visual_olfactory_pincer",
                "success": True,
                "ablation_variant": "drop_visual_cortex",
                "ablation_architecture": "modular",
                "eval_reflex_scale": 1.0,
            },
            {
                "scenario": "olfactory_ambush",
                "success": False,
                "ablation_variant": "drop_visual_cortex",
                "ablation_architecture": "modular",
                "eval_reflex_scale": 1.0,
            },
        ])
        result = extract_ablations({}, rows)
        self.assertIn("predator_type_comparisons", result)
