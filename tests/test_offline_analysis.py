import json
import tempfile
import unittest
from pathlib import Path

from spider_cortex_sim.offline_analysis import (
    build_report_data,
    extract_ablations,
    extract_reflex_frequency,
    load_behavior_csv,
    load_summary,
    load_trace,
    normalize_behavior_rows,
    run_offline_analysis,
)
from spider_cortex_sim.simulation import SpiderSimulation


REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKIN_SUMMARY = REPO_ROOT / "spider_summary.json"
CHECKIN_TRACE = REPO_ROOT / "spider_trace.jsonl"


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


class OfflineAnalysisToleranceTest(unittest.TestCase):
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
