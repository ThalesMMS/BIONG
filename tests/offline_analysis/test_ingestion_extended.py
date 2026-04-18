"""Additional tests for spider_cortex_sim.offline_analysis.ingestion.

Covers edge cases in normalize_behavior_rows not addressed in test_ingestion.py,
including reflex scale fallback logic, boolean field coercion, integer coercion,
evaluation_map alias resolution, and empty/None input handling.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from spider_cortex_sim.offline_analysis.ingestion import (
    load_behavior_csv,
    load_summary,
    load_trace,
    normalize_behavior_rows,
)


class LoadSummaryTest(unittest.TestCase):
    def test_none_path_returns_empty_dict(self) -> None:
        result = load_summary(None)
        self.assertEqual(result, {})

    def test_reads_json_file(self) -> None:
        data = {"training": {"mean_reward": 1.5}}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "summary.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            result = load_summary(path)
        self.assertEqual(result["training"]["mean_reward"], 1.5)

    def test_accepts_string_path(self) -> None:
        data = {"key": "value"}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "s.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            result = load_summary(str(path))
        self.assertEqual(result["key"], "value")


class LoadTraceTest(unittest.TestCase):
    def test_none_path_returns_empty_list(self) -> None:
        self.assertEqual(load_trace(None), [])

    def test_reads_jsonl_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "trace.jsonl"
            path.write_text(
                '{"step": 1, "reward": 0.5}\n{"step": 2, "reward": 0.8}\n',
                encoding="utf-8",
            )
            result = load_trace(path)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["step"], 1)

    def test_reads_jsonl_file_str_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "trace.jsonl"
            path.write_text(
                '{"step": 1, "reward": 0.5}\n{"step": 2, "reward": 0.8}\n',
                encoding="utf-8",
            )
            result = load_trace(str(path))
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["step"], 1)

    def test_skips_empty_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "trace.jsonl"
            path.write_text('{"step": 1}\n\n{"step": 2}\n', encoding="utf-8")
            result = load_trace(path)
        self.assertEqual(len(result), 2)

    def test_skips_non_dict_json_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "trace.jsonl"
            path.write_text('{"step": 1}\n[1, 2, 3]\n{"step": 3}\n', encoding="utf-8")
            result = load_trace(path)
        self.assertEqual(len(result), 2)

    def test_empty_file_returns_empty_list(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "trace.jsonl"
            path.write_text("", encoding="utf-8")
            result = load_trace(path)
        self.assertEqual(result, [])


class LoadBehaviorCsvTest(unittest.TestCase):
    def test_none_path_returns_empty_list(self) -> None:
        self.assertEqual(load_behavior_csv(None), [])

    def test_reads_csv_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "behavior.csv"
            path.write_text("scenario,success\nnight_rest,True\n", encoding="utf-8")
            result = load_behavior_csv(path)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["scenario"], "night_rest")

    def test_reads_csv_file_str_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "behavior.csv"
            path.write_text("scenario,success\nnight_rest,True\n", encoding="utf-8")
            result = load_behavior_csv(str(path))
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["scenario"], "night_rest")


class NormalizeBehaviorRowsTest(unittest.TestCase):
    def test_empty_input_returns_empty_list(self) -> None:
        self.assertEqual(normalize_behavior_rows([]), [])

    # --- evaluation_map resolution ---

    def test_evaluation_map_preferred_over_map_template(self) -> None:
        rows = normalize_behavior_rows([{
            "evaluation_map": "primary_map",
            "map_template": "fallback_map",
        }])
        self.assertEqual(rows[0]["evaluation_map"], "primary_map")
        self.assertEqual(rows[0]["map_template"], "primary_map")

    def test_map_template_used_when_evaluation_map_absent(self) -> None:
        rows = normalize_behavior_rows([{"map_template": "template_map"}])
        self.assertEqual(rows[0]["evaluation_map"], "template_map")

    def test_scenario_map_used_as_final_fallback(self) -> None:
        rows = normalize_behavior_rows([{"scenario_map": "scenario_map_val"}])
        self.assertEqual(rows[0]["evaluation_map"], "scenario_map_val")

    def test_scenario_map_distinct_from_evaluation_map(self) -> None:
        rows = normalize_behavior_rows([{
            "evaluation_map": "eval_map",
            "scenario_map": "scene_map",
        }])
        self.assertEqual(rows[0]["evaluation_map"], "eval_map")
        self.assertEqual(rows[0]["scenario_map"], "scene_map")

    def test_missing_all_map_fields_produces_empty_strings(self) -> None:
        rows = normalize_behavior_rows([{}])
        self.assertEqual(rows[0]["evaluation_map"], "")
        self.assertEqual(rows[0]["map_template"], "")
        self.assertEqual(rows[0]["scenario_map"], "")

    # --- success coercion ---

    def test_success_true_string(self) -> None:
        rows = normalize_behavior_rows([{"success": "True"}])
        self.assertTrue(rows[0]["success"])
        self.assertIsInstance(rows[0]["success"], bool)

    def test_success_false_string(self) -> None:
        rows = normalize_behavior_rows([{"success": "False"}])
        self.assertFalse(rows[0]["success"])

    def test_success_1_string(self) -> None:
        rows = normalize_behavior_rows([{"success": "1"}])
        self.assertTrue(rows[0]["success"])

    def test_success_bool_true(self) -> None:
        rows = normalize_behavior_rows([{"success": True}])
        self.assertTrue(rows[0]["success"])

    def test_success_none_becomes_false(self) -> None:
        rows = normalize_behavior_rows([{"success": None}])
        self.assertFalse(rows[0]["success"])

    # --- integer fields ---

    def test_failure_count_coerced_to_int(self) -> None:
        rows = normalize_behavior_rows([{"failure_count": "3"}])
        self.assertEqual(rows[0]["failure_count"], 3)
        self.assertIsInstance(rows[0]["failure_count"], int)

    def test_failure_count_none_becomes_zero(self) -> None:
        rows = normalize_behavior_rows([{"failure_count": None}])
        self.assertEqual(rows[0]["failure_count"], 0)

    def test_episode_coerced_to_int(self) -> None:
        rows = normalize_behavior_rows([{"episode": "5"}])
        self.assertEqual(rows[0]["episode"], 5)

    def test_simulation_seed_coerced_to_int(self) -> None:
        rows = normalize_behavior_rows([{"simulation_seed": "42"}])
        self.assertEqual(rows[0]["simulation_seed"], 42)

    def test_episode_seed_coerced_to_int(self) -> None:
        rows = normalize_behavior_rows([{"episode_seed": "1001"}])
        self.assertEqual(rows[0]["episode_seed"], 1001)

    # --- reflex scale fallback ---

    def test_eval_reflex_scale_falls_back_to_reflex_scale(self) -> None:
        rows = normalize_behavior_rows([{"reflex_scale": "0.5"}])
        self.assertAlmostEqual(rows[0]["eval_reflex_scale"], 0.5)

    def test_eval_reflex_scale_explicit_overrides_fallback(self) -> None:
        rows = normalize_behavior_rows([{
            "reflex_scale": "1.0",
            "eval_reflex_scale": "0.0",
        }])
        self.assertAlmostEqual(rows[0]["eval_reflex_scale"], 0.0)

    def test_reflex_anneal_final_scale_falls_back_to_reflex_scale(self) -> None:
        rows = normalize_behavior_rows([{"reflex_scale": "0.7"}])
        self.assertAlmostEqual(rows[0]["reflex_anneal_final_scale"], 0.7)

    def test_reflex_anneal_final_scale_explicit(self) -> None:
        rows = normalize_behavior_rows([{
            "reflex_scale": "1.0",
            "reflex_anneal_final_scale": "0.3",
        }])
        self.assertAlmostEqual(rows[0]["reflex_anneal_final_scale"], 0.3)

    # --- string fields preserved ---

    def test_reward_profile_preserved(self) -> None:
        rows = normalize_behavior_rows([{"reward_profile": "austere"}])
        self.assertEqual(rows[0]["reward_profile"], "austere")

    def test_ablation_variant_preserved(self) -> None:
        rows = normalize_behavior_rows([{"ablation_variant": "modular_full"}])
        self.assertEqual(rows[0]["ablation_variant"], "modular_full")

    def test_scenario_preserved(self) -> None:
        rows = normalize_behavior_rows([{"scenario": "night_rest"}])
        self.assertEqual(rows[0]["scenario"], "night_rest")

    def test_none_scenario_becomes_empty_string(self) -> None:
        rows = normalize_behavior_rows([{"scenario": None}])
        self.assertEqual(rows[0]["scenario"], "")

    # --- multiple rows ---

    def test_multiple_rows_all_normalized(self) -> None:
        raw = [
            {"scenario": "s1", "success": "True", "episode": "0"},
            {"scenario": "s2", "success": "False", "episode": "1"},
        ]
        rows = normalize_behavior_rows(raw)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["scenario"], "s1")
        self.assertTrue(rows[0]["success"])
        self.assertEqual(rows[1]["scenario"], "s2")
        self.assertFalse(rows[1]["success"])

    # --- extra keys preserved ---

    def test_extra_keys_preserved_in_output(self) -> None:
        rows = normalize_behavior_rows([{"scenario": "s", "custom_field": "val"}])
        self.assertEqual(rows[0]["custom_field"], "val")
