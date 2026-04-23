from __future__ import annotations

import json
import tempfile
import unittest
from collections.abc import Mapping, Sequence
from pathlib import Path
from unittest import mock

from spider_cortex_sim.checkpointing import CheckpointSelectionConfig
from spider_cortex_sim.comparison import (
    compare_noise_robustness,
    matrix_cell_success_rate,
    robustness_aggregate_metrics,
    robustness_matrix_metadata,
)
from spider_cortex_sim.noise import RobustnessMatrixSpec
from spider_cortex_sim.simulation import SpiderSimulation


class NoiseRobustnessWorkflowTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        """
        Prepare class-level fixtures for noise-robustness tests.
        
        Calls compare_noise_robustness with a fixed smoke/classic + central_burrow configuration and stores the returned payload and rows on the class as `cls.payload` and `cls.rows` for use by test methods.
        """
        cls.payload, cls.rows = compare_noise_robustness(
            budget_profile="smoke",
            reward_profile="classic",
            map_template="central_burrow",
            seeds=(7,),
            names=("night_rest",),
            robustness_matrix=RobustnessMatrixSpec(
                train_conditions=("none", "low"),
                eval_conditions=("none", "high"),
            ),
        )

    def test_swap_eval_noise_profile_sets_and_restores_world_noise(self) -> None:
        sim = SpiderSimulation(seed=7, max_steps=10, noise_profile="low")
        self.assertEqual(sim.world.noise_profile.name, "low")

        with sim._swap_eval_noise_profile("high"):
            self.assertEqual(sim.world.noise_profile.name, "high")

        self.assertEqual(sim.world.noise_profile.name, "low")

    def test_compare_noise_robustness_produces_expected_matrix_structure(self) -> None:
        """
        Verifies the structure of the noise robustness matrix produced by compare_noise_robustness.
        
        Asserts that the matrix specification lists train conditions as ["none", "low"], eval conditions as ["none", "high"], and a cell count of 4. Also asserts that the produced matrix contains top-level keys "none" and "low", that "none" appears under matrix["none"], and that "high" appears under matrix["low"].
        """
        self.assertEqual(
            self.payload["matrix_spec"]["train_conditions"],
            ["none", "low"],
        )
        self.assertEqual(
            self.payload["matrix_spec"]["eval_conditions"],
            ["none", "high"],
        )
        self.assertEqual(self.payload["matrix_spec"]["cell_count"], 4)
        self.assertIn("none", self.payload["matrix"])
        self.assertIn("low", self.payload["matrix"])
        self.assertIn("none", self.payload["matrix"]["none"])
        self.assertIn("high", self.payload["matrix"]["low"])

    def test_compare_noise_robustness_payload_contains_required_keys(self) -> None:
        for key in (
            "matrix",
            "train_marginals",
            "eval_marginals",
            "robustness_score",
            "diagonal_score",
            "off_diagonal_score",
        ):
            self.assertIn(key, self.payload)

    def test_compare_noise_robustness_rows_include_train_and_eval_noise_columns(self) -> None:
        """
        Verify that the robustness comparison rows include non-empty train and eval noise profile fields.
        
        Asserts that the collected rows are not empty and that every row contains the keys "train_noise_profile" and "eval_noise_profile" with truthy values.
        """
        self.assertTrue(self.rows)
        for row in self.rows:
            self.assertIn("train_noise_profile", row)
            self.assertIn("eval_noise_profile", row)
            self.assertTrue(row["train_noise_profile"])
            self.assertTrue(row["eval_noise_profile"])


class ResolveCheckpointLoadDirTest(unittest.TestCase):
    """Tests for SpiderSimulation._resolve_checkpoint_load_dir (new in this PR)."""

    def test_returns_none_when_checkpoint_dir_is_none(self) -> None:
        result = SpiderSimulation._resolve_checkpoint_load_dir(
            None, checkpoint_selection="best"
        )
        self.assertIsNone(result)

    def test_returns_none_when_no_candidate_has_metadata_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = SpiderSimulation._resolve_checkpoint_load_dir(
                tmpdir, checkpoint_selection="best"
            )
        self.assertIsNone(result)

    def test_prefers_best_subdir_when_selection_is_best(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            best_dir = root / "best"
            best_dir.mkdir()
            (best_dir / "metadata.json").write_text("{}", encoding="utf-8")
            last_dir = root / "last"
            last_dir.mkdir()
            (last_dir / "metadata.json").write_text("{}", encoding="utf-8")

            result = SpiderSimulation._resolve_checkpoint_load_dir(
                tmpdir, checkpoint_selection="best"
            )
        self.assertEqual(result.name, "best")

    def test_prefers_last_subdir_when_selection_is_last(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            best_dir = root / "best"
            best_dir.mkdir()
            (best_dir / "metadata.json").write_text("{}", encoding="utf-8")
            last_dir = root / "last"
            last_dir.mkdir()
            (last_dir / "metadata.json").write_text("{}", encoding="utf-8")

            result = SpiderSimulation._resolve_checkpoint_load_dir(
                tmpdir, checkpoint_selection="last"
            )
        self.assertEqual(result.name, "last")

    def test_falls_back_to_root_when_best_missing_but_root_has_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "metadata.json").write_text("{}", encoding="utf-8")

            result = SpiderSimulation._resolve_checkpoint_load_dir(
                tmpdir, checkpoint_selection="best"
            )
        self.assertEqual(result, root)

    def test_falls_back_to_best_when_last_missing_and_best_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            best_dir = root / "best"
            best_dir.mkdir()
            (best_dir / "metadata.json").write_text("{}", encoding="utf-8")

            result = SpiderSimulation._resolve_checkpoint_load_dir(
                tmpdir, checkpoint_selection="last"
            )
        self.assertEqual(result.name, "best")

    def test_none_selection_disables_checkpoint_loading(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            best_dir = root / "best"
            best_dir.mkdir()
            (best_dir / "metadata.json").write_text("{}", encoding="utf-8")

            result = SpiderSimulation._resolve_checkpoint_load_dir(
                tmpdir, checkpoint_selection="none"
            )
        self.assertIsNone(result)

    def test_accepts_path_object_as_checkpoint_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            best_dir = root / "best"
            best_dir.mkdir()
            (best_dir / "metadata.json").write_text("{}", encoding="utf-8")

            result = SpiderSimulation._resolve_checkpoint_load_dir(
                root, checkpoint_selection="best"
            )
        self.assertEqual(result.name, "best")

    def test_candidate_dir_without_metadata_json_is_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            best_dir = root / "best"
            best_dir.mkdir()
            # No metadata.json in best - should skip
            last_dir = root / "last"
            last_dir.mkdir()
            (last_dir / "metadata.json").write_text("{}", encoding="utf-8")

            result = SpiderSimulation._resolve_checkpoint_load_dir(
                tmpdir, checkpoint_selection="best"
            )
        self.assertEqual(result.name, "last")


class CheckpointRunFingerprintTest(unittest.TestCase):
    def test_checkpoint_run_fingerprint_is_stable_and_sensitive(self) -> None:
        payload = {
            "workflow": "noise_robustness",
            "world": {"width": 12, "height": 12, "reward_profile": "classic"},
            "learning": {"gamma": 0.96},
        }

        first = SpiderSimulation._checkpoint_run_fingerprint(payload)
        second = SpiderSimulation._checkpoint_run_fingerprint(
            {
                "learning": {"gamma": 0.96},
                "world": {"reward_profile": "classic", "height": 12, "width": 12},
                "workflow": "noise_robustness",
            }
        )
        changed = SpiderSimulation._checkpoint_run_fingerprint(
            {
                **payload,
                "world": {**payload["world"], "reward_profile": "shaped"},
            }
        )

        self.assertEqual(first, second)
        self.assertNotEqual(first, changed)

    def test_checkpoint_run_fingerprint_changes_with_preload_inputs(self) -> None:
        payload = {
            "workflow": "noise_robustness",
            "world": {"width": 12, "height": 12, "reward_profile": "classic"},
            "learning": {"gamma": 0.96},
            "preload": {
                "load_brain": "brain_a",
                "load_modules": ["motor"],
            },
        }

        with_different_brain = SpiderSimulation._checkpoint_run_fingerprint(
            {
                **payload,
                "preload": {
                    "load_brain": "brain_b",
                    "load_modules": ["motor"],
                },
            }
        )
        with_different_modules = SpiderSimulation._checkpoint_run_fingerprint(
            {
                **payload,
                "preload": {
                    "load_brain": "brain_a",
                    "load_modules": ["vision", "motor"],
                },
            }
        )

        baseline = SpiderSimulation._checkpoint_run_fingerprint(payload)
        self.assertNotEqual(baseline, with_different_brain)
        self.assertNotEqual(baseline, with_different_modules)

    def test_checkpoint_preload_fingerprint_changes_when_artifact_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            load_dir = Path(tmpdir) / "load_source"
            SpiderSimulation(seed=7, max_steps=10).brain.save(load_dir)

            baseline = SpiderSimulation._checkpoint_preload_fingerprint(load_dir)
            action_weights = load_dir / "action_center.npz"
            action_weights.write_bytes(action_weights.read_bytes() + b"changed")
            changed = SpiderSimulation._checkpoint_preload_fingerprint(load_dir)

        self.assertNotEqual(baseline["module_sha256"], changed["module_sha256"])

    def test_checkpoint_preload_fingerprint_normalizes_module_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            load_dir = Path(tmpdir) / "load_source"
            SpiderSimulation(seed=7, max_steps=10).brain.save(load_dir)

            first = SpiderSimulation._checkpoint_preload_fingerprint(
                load_dir,
                ("motor_cortex", "action_center", "motor_cortex"),
            )
            second = SpiderSimulation._checkpoint_preload_fingerprint(
                load_dir,
                ("action_center", "motor_cortex"),
            )

        self.assertEqual(first, second)


class RobustnessAggregateMetricsTest(unittest.TestCase):
    """Tests for comparison.robustness_aggregate_metrics (new in this PR)."""

    def _make_spec(
        self,
        trains: Sequence[str],
        evals: Sequence[str],
    ) -> RobustnessMatrixSpec:
        """
        Create a RobustnessMatrixSpec from sequences of train and evaluation condition names.
        
        Parameters:
            trains (Sequence[str]): Iterable of training condition names.
            evals (Sequence[str]): Iterable of evaluation condition names.
        
        Returns:
            RobustnessMatrixSpec: Spec with `train_conditions` and `eval_conditions` set to the provided values as tuples.
        """
        return RobustnessMatrixSpec(
            train_conditions=tuple(trains),
            eval_conditions=tuple(evals),
        )

    def _make_cell(self, rate: float) -> dict[str, Mapping[str, float]]:
        # _matrix_cell_success_rate calls _condition_compact_summary which expects
        # a "summary" key containing the actual metrics dict
        return {"summary": {"scenario_success_rate": rate, "episode_success_rate": rate}}

    def test_robustness_score_is_mean_of_all_cells(self) -> None:
        spec = self._make_spec(["none", "low"], ["none", "high"])
        payloads = {
            "none": {"none": self._make_cell(1.0), "high": self._make_cell(0.5)},
            "low": {"none": self._make_cell(0.8), "high": self._make_cell(0.2)},
        }
        result = robustness_aggregate_metrics(
            payloads, robustness_matrix=spec
        )
        expected = (1.0 + 0.5 + 0.8 + 0.2) / 4
        self.assertAlmostEqual(result["robustness_score"], expected)

    def test_diagonal_score_uses_only_matching_cells(self) -> None:
        spec = self._make_spec(["none", "low"], ["none", "low"])
        payloads = {
            "none": {"none": self._make_cell(0.9), "low": self._make_cell(0.4)},
            "low": {"none": self._make_cell(0.3), "low": self._make_cell(0.7)},
        }
        result = robustness_aggregate_metrics(
            payloads, robustness_matrix=spec
        )
        self.assertAlmostEqual(result["diagonal_score"], 0.8)

    def test_off_diagonal_score_excludes_matching_cells(self) -> None:
        spec = self._make_spec(["none", "low"], ["none", "low"])
        payloads = {
            "none": {"none": self._make_cell(0.9), "low": self._make_cell(0.4)},
            "low": {"none": self._make_cell(0.3), "low": self._make_cell(0.7)},
        }
        result = robustness_aggregate_metrics(
            payloads, robustness_matrix=spec
        )
        self.assertAlmostEqual(result["off_diagonal_score"], 0.35)

    def test_train_marginals_averaged_across_eval_conditions(self) -> None:
        spec = self._make_spec(["none"], ["none", "high"])
        payloads = {
            "none": {"none": self._make_cell(1.0), "high": self._make_cell(0.4)},
        }
        result = robustness_aggregate_metrics(
            payloads, robustness_matrix=spec
        )
        self.assertAlmostEqual(result["train_marginals"]["none"], 0.7)

    def test_eval_marginals_averaged_across_train_conditions(self) -> None:
        spec = self._make_spec(["none", "low"], ["none"])
        payloads = {
            "none": {"none": self._make_cell(0.8)},
            "low": {"none": self._make_cell(0.6)},
        }
        result = robustness_aggregate_metrics(
            payloads, robustness_matrix=spec
        )
        self.assertAlmostEqual(result["eval_marginals"]["none"], 0.7)

    def test_empty_payloads_return_zero_scores(self) -> None:
        spec = self._make_spec(["none"], ["low"])
        result = robustness_aggregate_metrics(
            {}, robustness_matrix=spec
        )
        self.assertAlmostEqual(result["robustness_score"], 0.0)
        self.assertAlmostEqual(result["diagonal_score"], 0.0)
        self.assertAlmostEqual(result["off_diagonal_score"], 0.0)

    def test_missing_cell_treated_as_zero(self) -> None:
        spec = self._make_spec(["none", "low"], ["none"])
        payloads = {
            "none": {"none": self._make_cell(1.0)},
            # "low" train condition missing entirely
        }
        result = robustness_aggregate_metrics(
            payloads, robustness_matrix=spec
        )
        # (none,none)=1.0, (low,none)=0.0 (missing → 0.0)
        self.assertAlmostEqual(result["robustness_score"], 0.5)

    def test_uncertainty_reported_for_matrix_scores(self) -> None:
        spec = self._make_spec(["none"], ["none", "low"])
        payloads = {
            "none": {
                "none": {
                    "summary": {"scenario_success_rate": 0.7},
                    "seed_level": [
                        {"metric_name": "scenario_success_rate", "seed": 1, "value": 0.6, "condition": "none->none", "scenario": None},
                        {"metric_name": "scenario_success_rate", "seed": 2, "value": 0.8, "condition": "none->none", "scenario": None},
                    ],
                },
                "low": {
                    "summary": {"scenario_success_rate": 0.3},
                    "seed_level": [
                        {"metric_name": "scenario_success_rate", "seed": 1, "value": 0.2, "condition": "none->low", "scenario": None},
                        {"metric_name": "scenario_success_rate", "seed": 2, "value": 0.4, "condition": "none->low", "scenario": None},
                    ],
                },
            },
        }
        result = robustness_aggregate_metrics(
            payloads, robustness_matrix=spec
        )
        self.assertIn("uncertainty", result)
        self.assertIn("diagonal_minus_off_diagonal_score", result)
        self.assertEqual(result["uncertainty"]["robustness_score"]["n_seeds"], 2)
        self.assertAlmostEqual(result["uncertainty"]["robustness_score"]["mean"], 0.5)
        self.assertEqual(
            result["uncertainty"]["diagonal_minus_off_diagonal_score"]["n_seeds"],
            2,
        )


class RobustnessMatrixMetadataTest(unittest.TestCase):
    """Tests for comparison.robustness_matrix_metadata (new in this PR)."""

    def test_returns_dict_with_matrix_spec_key(self) -> None:
        spec = RobustnessMatrixSpec(
            train_conditions=("none", "low"),
            eval_conditions=("none",),
        )
        result = robustness_matrix_metadata(spec)
        self.assertIn("matrix_spec", result)

    def test_matrix_spec_contains_train_and_eval_conditions(self) -> None:
        spec = RobustnessMatrixSpec(
            train_conditions=("none", "low"),
            eval_conditions=("high",),
        )
        result = robustness_matrix_metadata(spec)
        self.assertEqual(result["matrix_spec"]["train_conditions"], ["none", "low"])
        self.assertEqual(result["matrix_spec"]["eval_conditions"], ["high"])

    def test_matrix_spec_has_correct_cell_count(self) -> None:
        spec = RobustnessMatrixSpec(
            train_conditions=("none", "low", "medium"),
            eval_conditions=("none", "high"),
        )
        result = robustness_matrix_metadata(spec)
        self.assertEqual(result["matrix_spec"]["cell_count"], 6)


class MatrixCellSuccessRateTest(unittest.TestCase):
    """Tests for comparison.matrix_cell_success_rate (new in this PR)."""

    def test_returns_scenario_success_rate_from_payload(self) -> None:
        # _matrix_cell_success_rate delegates to _condition_compact_summary which
        # looks for a nested "summary" key to find the metrics
        payload = {"summary": {"scenario_success_rate": 0.75}}
        result = matrix_cell_success_rate(payload)
        self.assertAlmostEqual(result, 0.75)

    def test_returns_zero_when_payload_is_none(self) -> None:
        result = matrix_cell_success_rate(None)
        self.assertAlmostEqual(result, 0.0)

    def test_returns_zero_when_field_missing(self) -> None:
        result = matrix_cell_success_rate({})
        self.assertAlmostEqual(result, 0.0)

    def test_returns_float_type(self) -> None:
        result = matrix_cell_success_rate(
            {"summary": {"scenario_success_rate": 1}}
        )
        self.assertIsInstance(result, float)


class SwapEvalNoiseProfileTest(unittest.TestCase):
    """Tests for SpiderSimulation._swap_eval_noise_profile context manager (new in this PR)."""

    def test_restores_profile_after_exception(self) -> None:
        sim = SpiderSimulation(seed=1, max_steps=10, noise_profile="none")
        self.assertEqual(sim.world.noise_profile.name, "none")
        try:
            with sim._swap_eval_noise_profile("high"):
                self.assertEqual(sim.world.noise_profile.name, "high")
                raise RuntimeError("forced error")
        except RuntimeError:
            pass
        self.assertEqual(sim.world.noise_profile.name, "none")

    def test_none_profile_swaps_to_none_profile(self) -> None:
        sim = SpiderSimulation(seed=1, max_steps=10, noise_profile="low")
        with sim._swap_eval_noise_profile(None):
            self.assertEqual(sim.world.noise_profile.name, "none")
        self.assertEqual(sim.world.noise_profile.name, "low")


class CompareNoiseRobustnessValidationTest(unittest.TestCase):
    def test_compare_noise_robustness_rejects_invalid_checkpoint_selection(self) -> None:
        with self.assertRaises(ValueError):
            compare_noise_robustness(
                max_steps=10,
                episodes=0,
                evaluation_episodes=0,
                reward_profile="classic",
                map_template="central_burrow",
                seeds=(7,),
                names=("night_rest",),
                robustness_matrix=RobustnessMatrixSpec(
                    train_conditions=("none",),
                    eval_conditions=("none",),
                ),
                checkpoint_selection="last",
            )

    def test_compare_noise_robustness_rejects_invalid_checkpoint_metric(self) -> None:
        """
        Verifies that compare_noise_robustness raises a ValueError when given an invalid checkpoint_metric.
        
        Asserts that calling compare_noise_robustness with checkpoint_selection set to "best" and
        checkpoint_metric set to an unrecognized value causes a ValueError to be raised.
        """
        with self.assertRaises(ValueError):
            compare_noise_robustness(
                max_steps=10,
                episodes=0,
                evaluation_episodes=0,
                reward_profile="classic",
                map_template="central_burrow",
                seeds=(7,),
                names=("night_rest",),
                robustness_matrix=RobustnessMatrixSpec(
                    train_conditions=("none",),
                    eval_conditions=("none",),
                ),
                checkpoint_selection="best",
                checkpoint_selection_config=CheckpointSelectionConfig(
                    metric="not_a_metric",
                ),
            )

    def test_compare_noise_robustness_loads_and_saves_brains_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            load_dir = root / "load_source"
            save_dir = root / "saved_runs"
            SpiderSimulation(seed=7, max_steps=10).brain.save(load_dir)

            compare_noise_robustness(
                max_steps=10,
                episodes=0,
                evaluation_episodes=0,
                reward_profile="classic",
                map_template="central_burrow",
                seeds=(7,),
                names=("night_rest",),
                episodes_per_scenario=1,
                robustness_matrix=RobustnessMatrixSpec(
                    train_conditions=("none",),
                    eval_conditions=("none",),
                ),
                load_brain=load_dir,
                save_brain=save_dir,
            )

            saved_metadata = list(save_dir.rglob("metadata.json"))
            self.assertTrue(saved_metadata)

    def test_compare_noise_robustness_preload_only_marks_rows_preloaded(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            load_dir = Path(tmpdir) / "load_source"
            SpiderSimulation(seed=7, max_steps=10).brain.save(load_dir)

            _, rows = compare_noise_robustness(
                max_steps=10,
                episodes=0,
                evaluation_episodes=0,
                reward_profile="classic",
                map_template="central_burrow",
                seeds=(7,),
                names=("night_rest",),
                episodes_per_scenario=1,
                robustness_matrix=RobustnessMatrixSpec(
                    train_conditions=("none",),
                    eval_conditions=("none",),
                ),
                load_brain=load_dir,
                checkpoint_selection="none",
            )

            self.assertTrue(rows)
            self.assertTrue(
                all(row["checkpoint_source"] == "preloaded" for row in rows)
            )

    def test_compare_noise_robustness_uses_collision_free_eval_base_indices(self) -> None:
        base_indices: list[int] = []

        def fake_execute_behavior_suite(
            sim: SpiderSimulation,
            *,
            names: Sequence[str],
            episodes_per_scenario: int,
            capture_trace: bool,
            debug_trace: bool,
            base_index: int,
        ) -> tuple[dict[str, list[object]], dict[str, list[object]], list[object]]:
            del sim, episodes_per_scenario, capture_trace, debug_trace
            base_indices.append(base_index)
            empty_stats = {name: [] for name in names}
            empty_scores = {name: [] for name in names}
            return empty_stats, empty_scores, []

        with mock.patch.object(
            SpiderSimulation,
            "_execute_behavior_suite",
            autospec=True,
            side_effect=fake_execute_behavior_suite,
        ):
            compare_noise_robustness(
                max_steps=10,
                episodes=0,
                evaluation_episodes=0,
                reward_profile="classic",
                map_template="central_burrow",
                seeds=(7,),
                names=("night_rest",),
                episodes_per_scenario=20_000,
                robustness_matrix=RobustnessMatrixSpec(
                    train_conditions=("none",),
                    eval_conditions=("none", "low"),
                ),
            )

        self.assertEqual(len(base_indices), 2)
        episodes_per_cell = 20_000
        self.assertGreaterEqual(base_indices[1] - base_indices[0], episodes_per_cell)
