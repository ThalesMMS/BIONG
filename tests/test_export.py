import csv
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from spider_cortex_sim.curriculum import CURRICULUM_COLUMNS
from spider_cortex_sim.export import (
    compact_aggregate,
    compact_behavior_payload,
    jsonify,
    save_behavior_csv,
    save_summary,
    save_trace,
)
from spider_cortex_sim.noise import LOW_NOISE_PROFILE

NOISE_PROFILE_CONFIG_JSON = json.dumps(
    {
        "motor": {
            "action_flip_prob": 0.0,
            "fatigue_slip_factor": 0.0,
            "orientation_slip_factor": 0.0,
            "terrain_slip_factor": 0.0,
        },
        "name": "none",
        "olfactory": {
            "direction_jitter": 0.0,
            "strength_jitter": 0.0,
        },
        "predator": {
            "random_choice_prob": 0.0,
        },
        "spawn": {
            "uniform_mix": 0.0,
        },
        "visual": {
            "certainty_jitter": 0.0,
            "direction_jitter": 0.0,
            "dropout_prob": 0.0,
        },
    },
    sort_keys=True,
)
LOW_NOISE_PROFILE_CONFIG_JSON = json.dumps(LOW_NOISE_PROFILE.to_summary(), sort_keys=True)


class JsonifyTest(unittest.TestCase):
    """Tests for export.jsonify."""

    def test_primitive_int_returned_unchanged(self) -> None:
        self.assertEqual(jsonify(42), 42)

    def test_primitive_float_returned_unchanged(self) -> None:
        self.assertAlmostEqual(jsonify(3.14), 3.14)

    def test_primitive_string_returned_unchanged(self) -> None:
        self.assertEqual(jsonify("hello"), "hello")

    def test_primitive_bool_returned_unchanged(self) -> None:
        self.assertEqual(jsonify(True), True)
        self.assertEqual(jsonify(False), False)

    def test_none_returned_unchanged(self) -> None:
        self.assertIsNone(jsonify(None))

    def test_dict_keys_converted_to_strings(self) -> None:
        result = jsonify({1: "a", 2: "b"})
        self.assertIn("1", result)
        self.assertIn("2", result)
        self.assertEqual(result["1"], "a")

    def test_dict_values_recursively_converted(self) -> None:
        result = jsonify({"nested": {"key": [1, 2, 3]}})
        self.assertEqual(result["nested"]["key"], [1, 2, 3])

    def test_list_items_recursively_converted(self) -> None:
        result = jsonify([1, "two", None, True])
        self.assertEqual(result, [1, "two", None, True])

    def test_tuple_converted_to_list(self) -> None:
        result = jsonify((1, 2, 3))
        self.assertIsInstance(result, list)
        self.assertEqual(result, [1, 2, 3])

    def test_object_with_tolist_method_uses_tolist(self) -> None:
        class FakeArray:
            def tolist(self):
                return [10, 20, 30]

        result = jsonify(FakeArray())
        self.assertEqual(result, [10, 20, 30])

    def test_unknown_object_converted_to_string(self) -> None:
        class MyObj:
            def __str__(self):
                return "my_obj_str"

        result = jsonify(MyObj())
        self.assertEqual(result, "my_obj_str")

    def test_empty_dict_returns_empty_dict(self) -> None:
        self.assertEqual(jsonify({}), {})

    def test_empty_list_returns_empty_list(self) -> None:
        self.assertEqual(jsonify([]), [])

    def test_nested_mixed_types(self) -> None:
        payload = {
            "a": [1, 2.5, "x"],
            "b": {"c": None, "d": True},
        }
        result = jsonify(payload)
        self.assertEqual(result["a"], [1, 2.5, "x"])
        self.assertIsNone(result["b"]["c"])
        self.assertTrue(result["b"]["d"])

    def test_path_value_converted_to_string(self) -> None:
        checkpoint_path = Path("checkpoints") / "best"
        result = jsonify({"checkpoint": checkpoint_path})
        self.assertEqual(result, {"checkpoint": str(checkpoint_path)})


class CompactAggregateTest(unittest.TestCase):
    """Tests for export.compact_aggregate."""

    def test_removes_episodes_detail_key(self) -> None:
        data = {
            "mean_reward": 1.5,
            "episodes_detail": [{"episode": 0}, {"episode": 1}],
        }
        result = compact_aggregate(data)
        self.assertNotIn("episodes_detail", result)

    def test_preserves_other_keys(self) -> None:
        data = {
            "mean_reward": 2.0,
            "success_rate": 0.8,
            "episodes_detail": [],
        }
        result = compact_aggregate(data)
        self.assertEqual(result["mean_reward"], 2.0)
        self.assertEqual(result["success_rate"], 0.8)

    def test_no_episodes_detail_passes_through(self) -> None:
        data = {"mean_reward": 1.0, "count": 5}
        result = compact_aggregate(data)
        self.assertEqual(result, {"mean_reward": 1.0, "count": 5})

    def test_returns_shallow_copy_not_original(self) -> None:
        data = {"key": "value", "episodes_detail": []}
        result = compact_aggregate(data)
        self.assertIsNot(result, data)

    def test_original_dict_not_mutated(self) -> None:
        data = {"episodes_detail": [1, 2, 3], "other": "keep"}
        compact_aggregate(data)
        self.assertIn("episodes_detail", data)

    def test_empty_dict_returns_empty_dict(self) -> None:
        self.assertEqual(compact_aggregate({}), {})


class SaveSummaryTest(unittest.TestCase):
    """Tests for export.save_summary."""

    def test_creates_json_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "summary.json"
            save_summary({"key": "value"}, path)
            self.assertTrue(path.exists())

    def test_file_content_is_valid_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "summary.json"
            save_summary({"episodes": 100, "metric": 0.85}, path)
            content = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(content["episodes"], 100)
            self.assertAlmostEqual(content["metric"], 0.85)

    def test_nested_dict_is_preserved(self) -> None:
        summary = {"config": {"width": 12, "height": 12}, "result": {"score": 0.9}}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "summary.json"
            save_summary(summary, path)
            loaded = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(loaded["config"]["width"], 12)
            self.assertEqual(loaded["result"]["score"], 0.9)

    def test_accepts_string_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path_str = str(Path(tmpdir) / "summary.json")
            save_summary({"x": 1}, path_str)
            self.assertTrue(Path(path_str).exists())

    def test_uses_indented_formatting(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "summary.json"
            save_summary({"a": 1}, path)
            content = path.read_text(encoding="utf-8")
            # Indented JSON has newlines
            self.assertIn("\n", content)

    def test_empty_summary_writes_empty_object(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "summary.json"
            save_summary({}, path)
            content = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(content, {})

    def test_overwrites_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "summary.json"
            save_summary({"version": 1}, path)
            save_summary({"version": 2}, path)
            content = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(content["version"], 2)


class SaveTraceTest(unittest.TestCase):
    """Tests for export.save_trace."""

    def test_creates_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "trace.jsonl"
            save_trace([{"step": 0}], path)
            self.assertTrue(path.exists())

    def test_each_item_on_separate_line(self) -> None:
        trace = [{"step": 0, "reward": 1.0}, {"step": 1, "reward": 0.5}]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "trace.jsonl"
            save_trace(trace, path)
            lines = [
                line for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(lines), 2)

    def test_each_line_is_valid_json(self) -> None:
        trace = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "trace.jsonl"
            save_trace(trace, path)
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    parsed = json.loads(line)
                    self.assertIn("a", parsed)
                    self.assertIn("b", parsed)

    def test_empty_trace_creates_empty_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "trace.jsonl"
            save_trace([], path)
            content = path.read_text(encoding="utf-8").strip()
            self.assertEqual(content, "")

    def test_accepts_string_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path_str = str(Path(tmpdir) / "trace.jsonl")
            save_trace([{"x": 1}], path_str)
            self.assertTrue(Path(path_str).exists())

    def test_overwrites_existing_file(self) -> None:
        trace1 = [{"version": 1}]
        trace2 = [{"version": 2}, {"version": 3}]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "trace.jsonl"
            save_trace(trace1, path)
            save_trace(trace2, path)
            lines = [
                line for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(lines), 2)
            self.assertEqual(json.loads(lines[0])["version"], 2)


class _ListLike:
    def __init__(self, values: list[int]) -> None:
        self.values = values

    def tolist(self) -> list[int]:
        return list(self.values)


class SaveJsonOutputTest(unittest.TestCase):
    def test_save_summary_jsonifies_non_json_native_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "summary.json"
            checkpoint_path = Path("checkpoints") / "best"

            save_summary(
                {
                    "path": checkpoint_path,
                    "array": _ListLike([1, 2, 3]),
                },
                path,
            )

            payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["path"], str(checkpoint_path))
        self.assertEqual(payload["array"], [1, 2, 3])

    def test_save_trace_jsonifies_each_trace_item(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "trace.jsonl"
            trace_path = Path("trace") / "item"

            save_trace(
                [
                    {
                        "path": trace_path,
                        "array": _ListLike([4, 5]),
                    }
                ],
                path,
            )

            lines = path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 1)
        payload = json.loads(lines[0])
        self.assertEqual(payload["path"], str(trace_path))
        self.assertEqual(payload["array"], [4, 5])


class SaveBehaviorCsvTest(unittest.TestCase):
    """Tests for export.save_behavior_csv."""

    def _make_rows(self) -> list[dict[str, Any]]:
        """
        Constructs example CSV row dictionaries used by the behavior-evaluation test suite.

        Each dictionary models a single flattened CSV row including simulation and curriculum metadata, competence/reflex/budget/checkpoint/noise fields, scenario identifiers and descriptive fields, outcome fields (`success`, `failure_count`, `failures`), metric columns prefixed with `metric_...`, and per-check columns named `check_<name>_passed`, `check_<name>_value`, and `check_<name>_expected`.

        Returns:
            list[dict[str, Any]]: A list of example row dictionaries conforming to the CSV/flattened behavior-row schema used in tests.
        """
        return [
            {
                "reward_profile": "classic",
                "scenario_map": "night_rest_map",
                "evaluation_map": "central_burrow",
                "training_regime": "curriculum",
                "curriculum_profile": "ecological_v1",
                "curriculum_phase": "phase_4_corridor_food_deprivation",
                "curriculum_skill": "corridor_navigation",
                "curriculum_phase_status": "max_budget_exhausted",
                "curriculum_promotion_reason": "threshold_fallback",
                "learning_evidence_condition": "trained_final",
                "learning_evidence_policy_mode": "normal",
                "learning_evidence_training_regime": "baseline",
                "learning_evidence_train_episodes": 6,
                "learning_evidence_frozen_after_episode": "",
                "learning_evidence_checkpoint_source": "final",
                "learning_evidence_budget_profile": "smoke",
                "learning_evidence_budget_benchmark_strength": "quick",
                "reflex_scale": 1.0,
                "reflex_anneal_final_scale": 0.5,
                "competence_type": "self_sufficient",
                "is_primary_benchmark": True,
                "is_capability_probe": False,
                "eval_reflex_scale": 0.0,
                "budget_profile": "dev",
                "benchmark_strength": "quick",
                "operational_profile": "default_v1",
                "operational_profile_version": 1,
                "train_noise_profile": "low",
                "train_noise_profile_config": LOW_NOISE_PROFILE_CONFIG_JSON,
                "eval_noise_profile": "none",
                "eval_noise_profile_config": NOISE_PROFILE_CONFIG_JSON,
                "noise_profile": "none",
                "noise_profile_config": NOISE_PROFILE_CONFIG_JSON,
                "checkpoint_source": "best",
                "simulation_seed": 7,
                "episode_seed": 100000,
                "scenario": "night_rest",
                "scenario_description": "night desc",
                "scenario_objective": "night obj",
                "scenario_focus": "night focus",
                "episode": 0,
                "success": True,
                "failure_count": 0,
                "failures": "",
                "metric_food_eaten": 1,
                "check_deep_night_shelter_passed": True,
                "check_deep_night_shelter_value": 0.99,
                "check_deep_night_shelter_expected": ">= 0.95",
            }
        ]

    def test_creates_csv_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.csv"
            save_behavior_csv(self._make_rows(), path)
            self.assertTrue(path.exists())

    def test_csv_has_header_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.csv"
            save_behavior_csv(self._make_rows(), path)
            content = path.read_text(encoding="utf-8")
            lines = content.strip().splitlines()
            self.assertGreater(len(lines), 1)

    def test_preferred_columns_appear_in_header(self) -> None:
        """
        Verify the CSV header produced by save_behavior_csv includes all preferred behavior CSV columns.

        Checks presence of columns for reward/profile/map metadata, curriculum and learning-evidence fields, reflex/budget/checkpoint settings, competence and reflex evaluation fields, noise-profile configuration, scenario metadata, and outcome columns such as episode, success, failure_count, and failures.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.csv"
            save_behavior_csv(self._make_rows(), path)
            header_cols = path.read_text(encoding="utf-8").splitlines()[0].split(",")
            for col in [
                "reward_profile",
                "scenario_map",
                "evaluation_map",
                "ablation_variant",
                "ablation_architecture",
                *CURRICULUM_COLUMNS,
                "learning_evidence_condition",
                "learning_evidence_policy_mode",
                "learning_evidence_training_regime",
                "learning_evidence_train_episodes",
                "learning_evidence_frozen_after_episode",
                "learning_evidence_checkpoint_source",
                "learning_evidence_budget_profile",
                "learning_evidence_budget_benchmark_strength",
                "reflex_scale",
                "reflex_anneal_final_scale",
                "competence_type",
                "is_primary_benchmark",
                "is_capability_probe",
                "probe_type",
                "target_skill",
                "geometry_assumptions",
                "benchmark_tier",
                "acceptable_partial_progress",
                "eval_reflex_scale",
                "budget_profile",
                "benchmark_strength",
                "architecture_version",
                "architecture_fingerprint",
                "operational_profile",
                "operational_profile_version",
                "train_noise_profile",
                "train_noise_profile_config",
                "eval_noise_profile",
                "eval_noise_profile_config",
                "noise_profile",
                "noise_profile_config",
                "checkpoint_source",
                "simulation_seed",
                "episode_seed",
                "scenario",
                "scenario_description",
                "scenario_objective",
                "scenario_focus",
                "episode",
                "success",
                "failure_count",
                "failures",
            ]:
                self.assertIn(
                    col,
                    header_cols,
                    msg=f"Column {col!r} missing from CSV header",
                )

    def test_curriculum_columns_appear_in_preferred_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.csv"
            save_behavior_csv(self._make_rows(), path)
            header_cols = path.read_text(encoding="utf-8").splitlines()[0].split(",")
            curriculum_header = [
                column for column in header_cols if column in CURRICULUM_COLUMNS
            ]
            self.assertEqual(curriculum_header, list(CURRICULUM_COLUMNS))

    def test_data_rows_count(self) -> None:
        rows = self._make_rows() * 3
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.csv"
            save_behavior_csv(rows, path)
            content = path.read_text(encoding="utf-8")
            lines = [line for line in content.strip().splitlines() if line.strip()]
            self.assertEqual(len(lines), 4)  # 1 header + 3 data rows

    def test_extra_columns_sorted_after_preferred(self) -> None:
        rows = [{
            "reward_profile": "classic",
            "scenario_map": "night_rest_map",
            "evaluation_map": "central_burrow",
            "simulation_seed": 7,
            "episode_seed": 100000,
            "scenario": "s",
            "episode": 0,
            "success": True,
            "failure_count": 0,
            "failures": "",
            "zzz_metric": 1.0,
            "aaa_metric": 2.0,
        }]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.csv"
            save_behavior_csv(rows, path)
            header = path.read_text(encoding="utf-8").splitlines()[0].split(",")
            # 'failures' is last preferred column, extra columns come after
            failures_idx = header.index("failures")
            aaa_idx = header.index("aaa_metric")
            zzz_idx = header.index("zzz_metric")
            self.assertGreater(aaa_idx, failures_idx)
            self.assertLess(aaa_idx, zzz_idx)

    def test_empty_rows_creates_header_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.csv"
            save_behavior_csv([], path)
            content = path.read_text(encoding="utf-8")
            lines = [line for line in content.strip().splitlines() if line.strip()]
            self.assertEqual(len(lines), 1)  # header only

    def test_csv_is_readable_by_csv_reader(self) -> None:
        """
        Verify the CSV produced by save_behavior_csv is readable by csv.DictReader and that key fields serialize to the expected string values.

        Writes a single-row CSV to a temporary file, reads it back with csv.DictReader, asserts exactly one data row is present, and checks that metadata (including reward/profile/map fields, curriculum and learning-evidence columns, reflex/budget/checkpoint settings, seed fields, scenario metadata, outcome fields, and noise profile/config) are serialized exactly as expected.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.csv"
            rows = self._make_rows()
            save_behavior_csv(rows, path)
            with open(path, encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                data_rows = list(reader)
            self.assertEqual(len(data_rows), 1)
            expected = {
                "reward_profile": "classic",
                "scenario_map": "night_rest_map",
                "evaluation_map": "central_burrow",
                "training_regime": "curriculum",
                "curriculum_profile": "ecological_v1",
                "curriculum_phase": "phase_4_corridor_food_deprivation",
                "curriculum_skill": "corridor_navigation",
                "curriculum_phase_status": "max_budget_exhausted",
                "curriculum_promotion_reason": "threshold_fallback",
                "learning_evidence_condition": "trained_final",
                "learning_evidence_policy_mode": "normal",
                "learning_evidence_training_regime": "baseline",
                "learning_evidence_train_episodes": "6",
                "learning_evidence_frozen_after_episode": "",
                "learning_evidence_checkpoint_source": "final",
                "learning_evidence_budget_profile": "smoke",
                "learning_evidence_budget_benchmark_strength": "quick",
                "reflex_scale": "1.0",
                "reflex_anneal_final_scale": "0.5",
                "competence_type": "self_sufficient",
                "is_primary_benchmark": "True",
                "is_capability_probe": "False",
                "eval_reflex_scale": "0.0",
                "budget_profile": "dev",
                "benchmark_strength": "quick",
                "operational_profile": "default_v1",
                "operational_profile_version": "1",
                "train_noise_profile": "low",
                "train_noise_profile_config": LOW_NOISE_PROFILE_CONFIG_JSON,
                "eval_noise_profile": "none",
                "eval_noise_profile_config": NOISE_PROFILE_CONFIG_JSON,
                "noise_profile": "none",
                "noise_profile_config": NOISE_PROFILE_CONFIG_JSON,
                "checkpoint_source": "best",
                "simulation_seed": "7",
                "episode_seed": "100000",
                "scenario": "night_rest",
                "scenario_description": "night desc",
                "scenario_objective": "night obj",
                "scenario_focus": "night focus",
                "episode": "0",
                "success": "True",
                "failure_count": "0",
                "failures": "",
            }
            for field, value in expected.items():
                self.assertEqual(data_rows[0].get(field), value)

class CompactBehaviorPayloadTest(unittest.TestCase):
    """Tests for export.compact_behavior_payload."""

    def _make_payload(self):
        return {
            "summary": {"scenario_count": 1, "episode_count": 2, "scenario_success_rate": 0.5, "episode_success_rate": 0.5, "regressions": []},
            "suite": {
                "night_rest": {
                    "scenario": "night_rest",
                    "description": "desc",
                    "objective": "obj",
                    "episodes": 2,
                    "success_rate": 1.0,
                    "checks": {},
                    "behavior_metrics": {},
                    "failures": [],
                    "episodes_detail": [{"episode": 0}, {"episode": 1}],
                    "legacy_metrics": {"mean_reward": 1.0, "episodes_detail": [{"episode": 0}]},
                }
            },
            "legacy_scenarios": {
                "night_rest": {"mean_reward": 1.0, "episodes_detail": [{"episode": 0}]}
            },
        }

    def test_episodes_detail_removed_from_suite(self) -> None:
        payload = self._make_payload()
        compact = compact_behavior_payload(payload)
        self.assertNotIn("episodes_detail", compact["suite"]["night_rest"])

    def test_legacy_metrics_episodes_detail_removed(self) -> None:
        payload = self._make_payload()
        compact = compact_behavior_payload(payload)
        self.assertNotIn("episodes_detail", compact["suite"]["night_rest"]["legacy_metrics"])

    def test_legacy_scenarios_episodes_detail_removed(self) -> None:
        payload = self._make_payload()
        compact = compact_behavior_payload(payload)
        self.assertNotIn("episodes_detail", compact["legacy_scenarios"]["night_rest"])

    def test_summary_preserved(self) -> None:
        payload = self._make_payload()
        compact = compact_behavior_payload(payload)
        self.assertEqual(compact["summary"]["scenario_count"], 1)
        self.assertEqual(compact["summary"]["episode_count"], 2)

    def test_suite_fields_preserved(self) -> None:
        payload = self._make_payload()
        compact = compact_behavior_payload(payload)
        suite_item = compact["suite"]["night_rest"]
        self.assertEqual(suite_item["success_rate"], 1.0)
        self.assertEqual(suite_item["episodes"], 2)

    def test_empty_payload_does_not_raise(self) -> None:
        compact = compact_behavior_payload({})
        self.assertIn("summary", compact)
        self.assertIn("suite", compact)
        self.assertIn("legacy_scenarios", compact)

if __name__ == "__main__":
    unittest.main()
