from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from spider_cortex_sim.export import save_behavior_csv
from spider_cortex_sim.offline_analysis.ingestion import (
    load_behavior_csv,
    load_summary,
    load_trace,
    normalize_behavior_rows,
)

from .conftest import CHECKIN_SUMMARY, CHECKIN_TRACE

class OfflineAnalysisLoadersTest(unittest.TestCase):
    def test_load_summary_reads_checkin_summary(self) -> None:
        summary = load_summary(CHECKIN_SUMMARY)
        self.assertIn("training", summary)
        self.assertIn("evaluation", summary)

    def test_load_summary_rejects_non_object_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = Path(tmpdir) / "summary.json"
            summary_path.write_text(
                json.dumps(["not", "an", "object"]),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                TypeError,
                "summary must be a top-level JSON object; found list",
            ):
                load_summary(summary_path)

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
            save_behavior_csv(rows, csv_path)
            loaded = load_behavior_csv(csv_path)

        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0]["scenario"], "night_rest")

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
