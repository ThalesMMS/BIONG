import json
import tempfile
import unittest
from pathlib import Path

from spider_cortex_sim.checkpointing import (
    CheckpointPenaltyMode,
    CheckpointSelectionConfig,
    checkpoint_candidate_composite_score,
    checkpoint_candidate_sort_key,
    checkpoint_preload_fingerprint,
    checkpoint_run_fingerprint,
    file_sha256,
    jsonify_observation,
    mean_reward_from_behavior_payload,
    persist_checkpoint_pair,
    resolve_checkpoint_load_dir,
)


def _write_checkpoint_artifact(path: Path, *, label: str) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "metadata.json").write_text(
        json.dumps({"label": label}),
        encoding="utf-8",
    )
    (path / "weights.npz").write_bytes(label.encode("utf-8"))
    return path


class ResolveCheckpointLoadDirValidationTest(unittest.TestCase):
    def test_invalid_checkpoint_selection_raises_value_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(
                ValueError,
                "checkpoint_selection.*typo",
            ):
                resolve_checkpoint_load_dir(
                    Path(tmpdir),
                    checkpoint_selection="typo",
                )


class CheckpointSelectionPenaltyConfigTest(unittest.TestCase):
    def test_direct_penalty_config_normalizes_mode_and_weights(self) -> None:
        """
        Verify that CheckpointSelectionConfig normalizes the penalty mode and preserves penalty weights.

        Asserts that providing penalty_mode="direct" results in the enum value CheckpointPenaltyMode.DIRECT, that to_summary() reports "direct" for penalty_mode, and that override and dominance penalty weights are retained.
        """
        config = CheckpointSelectionConfig(
            metric="scenario_success_rate",
            override_penalty_weight=0.4,
            dominance_penalty_weight=0.2,
            penalty_mode="direct",
        )

        self.assertEqual(config.penalty_mode, CheckpointPenaltyMode.DIRECT)
        self.assertEqual(config.to_summary()["penalty_mode"], "direct")
        self.assertAlmostEqual(config.override_penalty_weight, 0.4)
        self.assertAlmostEqual(config.dominance_penalty_weight, 0.2)

    def test_direct_mode_penalizes_high_override_with_same_success_rate(self) -> None:
        low_override = {
            "scenario_success_rate": 0.8,
            "episode_success_rate": 0.6,
            "mean_reward": 1.0,
            "evaluation_summary": {
                "mean_final_reflex_override_rate": 0.1,
                "mean_reflex_dominance": 0.0,
            },
            "episode": 1,
        }
        high_override = {
            "scenario_success_rate": 0.8,
            "episode_success_rate": 0.6,
            "mean_reward": 1.0,
            "evaluation_summary": {
                "mean_final_reflex_override_rate": 0.7,
                "mean_reflex_dominance": 0.0,
            },
            "episode": 2,
        }
        config = CheckpointSelectionConfig(
            metric="scenario_success_rate",
            override_penalty_weight=1.0,
            penalty_mode=CheckpointPenaltyMode.DIRECT,
        )

        self.assertGreater(
            checkpoint_candidate_sort_key(
                low_override,
                selection_config=config,
            ),
            checkpoint_candidate_sort_key(
                high_override,
                selection_config=config,
            ),
        )

    def test_tiebreaker_mode_preserves_legacy_six_tuple(self) -> None:
        candidate = {
            "scenario_success_rate": 0.5,
            "episode_success_rate": 0.4,
            "mean_reward": 1.0,
            "evaluation_summary": {
                "mean_final_reflex_override_rate": 0.2,
                "mean_reflex_dominance": 0.3,
            },
            "episode": 7,
        }

        key = checkpoint_candidate_sort_key(
            candidate,
            selection_config=CheckpointSelectionConfig(
                metric="scenario_success_rate",
                penalty_mode=CheckpointPenaltyMode.TIEBREAKER,
            ),
        )

        self.assertEqual(key, (0.5, 0.4, 1.0, -0.2, -0.3, 7))

    def test_direct_mode_composite_score_uses_penalty_formula(self) -> None:
        candidate = {
            "scenario_success_rate": 0.8,
            "episode_success_rate": 0.4,
            "mean_reward": 1.0,
            "evaluation_summary": {
                "mean_final_reflex_override_rate": 0.25,
                "mean_reflex_dominance": 0.5,
            },
            "episode": 7,
        }

        key = checkpoint_candidate_sort_key(
            candidate,
            selection_config=CheckpointSelectionConfig(
                metric="scenario_success_rate",
                override_penalty_weight=0.4,
                dominance_penalty_weight=0.2,
                penalty_mode=CheckpointPenaltyMode.DIRECT,
            ),
        )

        self.assertAlmostEqual(key[0], 0.8 - (0.4 * 0.25) - (0.2 * 0.5))
        self.assertEqual(key[1:4], (0.8, 0.4, 1.0))

class SimulationMeanRewardFromPayloadTest(unittest.TestCase):
    """Tests for mean_reward_from_behavior_payload()."""

    def test_empty_payload_returns_zero(self) -> None:
        result = mean_reward_from_behavior_payload({})
        self.assertEqual(result, 0.0)

    def test_empty_legacy_scenarios_returns_zero(self) -> None:
        result = mean_reward_from_behavior_payload(
            {"legacy_scenarios": {}}
        )
        self.assertEqual(result, 0.0)

    def test_non_dict_legacy_scenarios_returns_zero(self) -> None:
        result = mean_reward_from_behavior_payload(
            {"legacy_scenarios": None}
        )
        self.assertEqual(result, 0.0)

    def test_single_scenario_mean_reward(self) -> None:
        payload = {
            "legacy_scenarios": {
                "night_rest": {"mean_reward": 2.5},
            }
        }
        result = mean_reward_from_behavior_payload(payload)
        self.assertAlmostEqual(result, 2.5)

    def test_multiple_scenarios_averaged(self) -> None:
        payload = {
            "legacy_scenarios": {
                "night_rest": {"mean_reward": 1.0},
                "food_deprivation": {"mean_reward": 3.0},
            }
        }
        result = mean_reward_from_behavior_payload(payload)
        self.assertAlmostEqual(result, 2.0)

    def test_non_dict_scenario_data_skipped(self) -> None:
        payload = {
            "legacy_scenarios": {
                "night_rest": {"mean_reward": 4.0},
                "bad_entry": "not_a_dict",
            }
        }
        result = mean_reward_from_behavior_payload(payload)
        self.assertAlmostEqual(result, 4.0)

    def test_missing_mean_reward_key_defaults_to_zero(self) -> None:
        payload = {
            "legacy_scenarios": {
                "night_rest": {"other_key": 99},
            }
        }
        result = mean_reward_from_behavior_payload(payload)
        self.assertAlmostEqual(result, 0.0)

class SimulationCheckpointSortKeyTest(unittest.TestCase):
    """Tests for checkpoint_candidate_sort_key()."""

    def test_invalid_metric_raises_value_error(self) -> None:
        candidate = {
            "scenario_success_rate": 0.5,
            "episode_success_rate": 0.5,
            "mean_reward": 0.5,
            "episode": 1,
        }
        with self.assertRaises(ValueError):
            checkpoint_candidate_sort_key(
                candidate, primary_metric="invalid_metric"
            )

    def test_tiebreaker_config_preserves_legacy_tuple(self) -> None:
        candidate = {
            "scenario_success_rate": 0.5,
            "episode_success_rate": 0.4,
            "mean_reward": 1.0,
            "evaluation_summary": {
                "mean_final_reflex_override_rate": 0.2,
                "mean_reflex_dominance": 0.3,
            },
            "episode": 7,
        }
        key = checkpoint_candidate_sort_key(
            candidate,
            selection_config=CheckpointSelectionConfig(
                metric="scenario_success_rate",
                penalty_mode=CheckpointPenaltyMode.TIEBREAKER,
            ),
        )
        self.assertEqual(key, (0.5, 0.4, 1.0, -0.2, -0.3, 7))

    def test_direct_penalty_mode_prepends_composite_score(self) -> None:
        candidate = {
            "scenario_success_rate": 0.8,
            "episode_success_rate": 0.4,
            "mean_reward": 1.0,
            "evaluation_summary": {
                "mean_final_reflex_override_rate": 0.25,
                "mean_reflex_dominance": 0.5,
            },
            "episode": 7,
        }
        key = checkpoint_candidate_sort_key(
            candidate,
            selection_config=CheckpointSelectionConfig(
                metric="scenario_success_rate",
                override_penalty_weight=0.4,
                dominance_penalty_weight=0.2,
                penalty_mode=CheckpointPenaltyMode.DIRECT,
            ),
        )
        self.assertAlmostEqual(key[0], 0.6)
        self.assertEqual(key[1:4], (0.8, 0.4, 1.0))
        self.assertEqual(key[4:], (-0.25, -0.5, 7))

    def test_direct_penalty_can_override_primary_metric_gap(self) -> None:
        high_dependence = {
            "scenario_success_rate": 0.8,
            "episode_success_rate": 0.5,
            "mean_reward": 0.5,
            "evaluation_summary": {
                "mean_final_reflex_override_rate": 0.5,
                "mean_reflex_dominance": 0.0,
            },
            "episode": 1,
        }
        low_dependence = {
            "scenario_success_rate": 0.7,
            "episode_success_rate": 0.5,
            "mean_reward": 0.5,
            "evaluation_summary": {
                "mean_final_reflex_override_rate": 0.0,
                "mean_reflex_dominance": 0.0,
            },
            "episode": 2,
        }
        selection_config = CheckpointSelectionConfig(
            metric="scenario_success_rate",
            override_penalty_weight=1.0,
            penalty_mode=CheckpointPenaltyMode.DIRECT,
        )

        self.assertGreater(
            checkpoint_candidate_sort_key(
                low_dependence,
                selection_config=selection_config,
            ),
            checkpoint_candidate_sort_key(
                high_dependence,
                selection_config=selection_config,
            ),
        )

    def test_scenario_success_rate_primary_metric(self) -> None:
        high = {
            "scenario_success_rate": 0.9,
            "episode_success_rate": 0.1,
            "mean_reward": 0.1,
            "episode": 1,
        }
        low = {
            "scenario_success_rate": 0.1,
            "episode_success_rate": 0.9,
            "mean_reward": 0.9,
            "episode": 2,
        }
        key_high = checkpoint_candidate_sort_key(
            high, primary_metric="scenario_success_rate"
        )
        key_low = checkpoint_candidate_sort_key(
            low, primary_metric="scenario_success_rate"
        )
        self.assertGreater(key_high, key_low)

    def test_episode_success_rate_primary_metric(self) -> None:
        a = {
            "scenario_success_rate": 0.9,
            "episode_success_rate": 0.1,
            "mean_reward": 0.5,
            "episode": 1,
        }
        b = {
            "scenario_success_rate": 0.2,
            "episode_success_rate": 0.8,
            "mean_reward": 0.5,
            "episode": 2,
        }
        key_a = checkpoint_candidate_sort_key(
            a, primary_metric="episode_success_rate"
        )
        key_b = checkpoint_candidate_sort_key(
            b, primary_metric="episode_success_rate"
        )
        self.assertGreater(key_b, key_a)

    def test_episode_tiebreaker_favors_later_episode(self) -> None:
        # Same metric scores but different episode number
        early = {
            "scenario_success_rate": 0.5,
            "episode_success_rate": 0.5,
            "mean_reward": 0.5,
            "episode": 3,
        }
        late = {
            "scenario_success_rate": 0.5,
            "episode_success_rate": 0.5,
            "mean_reward": 0.5,
            "episode": 10,
        }
        key_early = checkpoint_candidate_sort_key(
            early, primary_metric="scenario_success_rate"
        )
        key_late = checkpoint_candidate_sort_key(
            late, primary_metric="scenario_success_rate"
        )
        self.assertGreater(key_late, key_early)

    def test_reflex_dependence_penalty_precedes_episode_tiebreaker(self) -> None:
        low_reflex = {
            "scenario_success_rate": 0.5,
            "episode_success_rate": 0.5,
            "mean_reward": 0.5,
            "evaluation_summary": {
                "mean_final_reflex_override_rate": 0.1,
                "mean_reflex_dominance": 0.1,
            },
            "episode": 3,
        }
        high_reflex = {
            "scenario_success_rate": 0.5,
            "episode_success_rate": 0.5,
            "mean_reward": 0.5,
            "evaluation_summary": {
                "mean_final_reflex_override_rate": 0.9,
                "mean_reflex_dominance": 0.9,
            },
            "episode": 10,
        }
        key_low_reflex = checkpoint_candidate_sort_key(
            low_reflex, primary_metric="scenario_success_rate"
        )
        key_high_reflex = checkpoint_candidate_sort_key(
            high_reflex, primary_metric="scenario_success_rate"
        )
        self.assertGreater(key_low_reflex, key_high_reflex)

    def test_high_reflex_override_rate_lowers_checkpoint_ranking(self) -> None:
        low_override = {
            "scenario_success_rate": 0.5,
            "episode_success_rate": 0.5,
            "mean_reward": 0.5,
            "evaluation_summary": {
                "mean_final_reflex_override_rate": 0.0,
                "mean_reflex_dominance": 0.1,
            },
            "episode": 5,
        }
        high_override = {
            "scenario_success_rate": 0.5,
            "episode_success_rate": 0.5,
            "mean_reward": 0.5,
            "evaluation_summary": {
                "mean_final_reflex_override_rate": 0.8,
                "mean_reflex_dominance": 0.1,
            },
            "episode": 5,
        }

        self.assertGreater(
            checkpoint_candidate_sort_key(
                low_override,
                primary_metric="scenario_success_rate",
            ),
            checkpoint_candidate_sort_key(
                high_override,
                primary_metric="scenario_success_rate",
            ),
        )

    def test_high_reflex_dominance_lowers_checkpoint_ranking(self) -> None:
        low_dominance = {
            "scenario_success_rate": 0.5,
            "episode_success_rate": 0.5,
            "mean_reward": 0.5,
            "evaluation_summary": {
                "mean_final_reflex_override_rate": 0.1,
                "mean_reflex_dominance": 0.0,
            },
            "episode": 5,
        }
        high_dominance = {
            "scenario_success_rate": 0.5,
            "episode_success_rate": 0.5,
            "mean_reward": 0.5,
            "evaluation_summary": {
                "mean_final_reflex_override_rate": 0.1,
                "mean_reflex_dominance": 0.8,
            },
            "episode": 5,
        }

        self.assertGreater(
            checkpoint_candidate_sort_key(
                low_dominance,
                primary_metric="scenario_success_rate",
            ),
            checkpoint_candidate_sort_key(
                high_dominance,
                primary_metric="scenario_success_rate",
            ),
        )

    def test_sort_key_is_comparable_for_checkpoint_ranking(self) -> None:
        lower_success = {
            "scenario_success_rate": 0.6,
            "episode_success_rate": 0.6,
            "mean_reward": 0.6,
            "episode": 5,
        }
        higher_success = {
            "scenario_success_rate": 0.7,
            "episode_success_rate": 0.6,
            "mean_reward": 0.6,
            "episode": 5,
        }
        lower_key = checkpoint_candidate_sort_key(
            lower_success,
            primary_metric="scenario_success_rate",
        )
        higher_key = checkpoint_candidate_sort_key(
            higher_success,
            primary_metric="scenario_success_rate",
        )
        self.assertLess(lower_key, higher_key)

    def test_missing_metric_keys_default_to_zero(self) -> None:
        missing_metrics = {"episode": 1}
        explicit_zero_metrics = {
            "scenario_success_rate": 0.0,
            "episode_success_rate": 0.0,
            "mean_reward": 0.0,
            "episode": 1,
        }
        positive_metric = {
            "scenario_success_rate": 0.0,
            "episode_success_rate": 0.0,
            "mean_reward": 0.1,
            "episode": 1,
        }

        missing_key = checkpoint_candidate_sort_key(
            missing_metrics,
            primary_metric="mean_reward",
        )
        explicit_zero_key = checkpoint_candidate_sort_key(
            explicit_zero_metrics,
            primary_metric="mean_reward",
        )
        positive_key = checkpoint_candidate_sort_key(
            positive_metric,
            primary_metric="mean_reward",
        )
        self.assertEqual(missing_key, explicit_zero_key)
        self.assertLess(missing_key, positive_key)

    def test_checkpoint_metric_changes_selection_priority(self) -> None:
        candidate_a = {
            "scenario_success_rate": 0.9,
            "episode_success_rate": 0.1,
            "mean_reward": 0.2,
            "episode": 1,
        }
        candidate_b = {
            "scenario_success_rate": 0.2,
            "episode_success_rate": 0.8,
            "mean_reward": 0.7,
            "episode": 2,
        }

        scenario_choice = max(
            [candidate_a, candidate_b],
            key=lambda item: checkpoint_candidate_sort_key(
                item,
                primary_metric="scenario_success_rate",
            ),
        )
        reward_choice = max(
            [candidate_a, candidate_b],
            key=lambda item: checkpoint_candidate_sort_key(
                item,
                primary_metric="mean_reward",
            ),
        )

        self.assertIs(scenario_choice, candidate_a)
        self.assertIs(reward_choice, candidate_b)


class PersistCheckpointPairTest(unittest.TestCase):
    """Tests for persist_checkpoint_pair()."""

    def test_none_checkpoint_dir_returns_empty_dict(self) -> None:
        result = persist_checkpoint_pair(
            checkpoint_dir=None,
            best_candidate={"path": Path("/some/path"), "name": "ep1", "episode": 1},
            last_candidate={"path": Path("/some/path"), "name": "ep1", "episode": 1},
        )
        self.assertEqual(result, {})

    def test_persist_creates_best_and_last_directories(self) -> None:
        with tempfile.TemporaryDirectory() as source_dir:
            with tempfile.TemporaryDirectory() as dest_dir:
                best_path = _write_checkpoint_artifact(
                    Path(source_dir) / "best_src",
                    label="best",
                )
                last_path = _write_checkpoint_artifact(
                    Path(source_dir) / "last_src",
                    label="last",
                )

                result = persist_checkpoint_pair(
                    checkpoint_dir=Path(dest_dir),
                    best_candidate={"path": best_path, "name": "best_src", "episode": 2},
                    last_candidate={"path": last_path, "name": "last_src", "episode": 4},
                )

                self.assertIn("best", result)
                self.assertIn("last", result)
                self.assertTrue(Path(result["best"]).exists())
                self.assertTrue(Path(result["last"]).exists())
                self.assertEqual(
                    json.loads(
                        (Path(result["best"]) / "metadata.json").read_text(
                            encoding="utf-8",
                        )
                    ),
                    {"label": "best"},
                )
                self.assertEqual(
                    json.loads(
                        (Path(result["last"]) / "metadata.json").read_text(
                            encoding="utf-8",
                        )
                    ),
                    {"label": "last"},
                )

    def test_persist_creates_parent_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as source_dir:
            with tempfile.TemporaryDirectory() as base_dir:
                src_path = _write_checkpoint_artifact(
                    Path(source_dir) / "ep",
                    label="ep",
                )
                nested_dest = Path(base_dir) / "nested" / "deep"

                result = persist_checkpoint_pair(
                    checkpoint_dir=nested_dest,
                    best_candidate={"path": src_path, "name": "ep", "episode": 1},
                    last_candidate={"path": src_path, "name": "ep", "episode": 1},
                )

                self.assertTrue(nested_dest.exists())
                self.assertIn("best", result)

class CheckpointSelectionConfigValidationTest(unittest.TestCase):
    """Tests for CheckpointSelectionConfig validation in __post_init__."""

    def test_invalid_metric_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            CheckpointSelectionConfig(metric="invalid_metric")

    def test_invalid_penalty_mode_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            CheckpointSelectionConfig(
                metric="scenario_success_rate",
                penalty_mode="bogus_mode",
            )

    def test_infinite_override_weight_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            CheckpointSelectionConfig(
                metric="scenario_success_rate",
                override_penalty_weight=float("inf"),
            )

    def test_negative_override_weight_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            CheckpointSelectionConfig(
                metric="scenario_success_rate",
                override_penalty_weight=-0.1,
            )

    def test_negative_dominance_weight_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            CheckpointSelectionConfig(
                metric="scenario_success_rate",
                dominance_penalty_weight=-1.0,
            )

    def test_nan_dominance_weight_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            CheckpointSelectionConfig(
                metric="scenario_success_rate",
                dominance_penalty_weight=float("nan"),
            )

    def test_all_valid_metrics_accepted(self) -> None:
        for metric in ("scenario_success_rate", "episode_success_rate", "mean_reward"):
            with self.subTest(metric=metric):
                config = CheckpointSelectionConfig(metric=metric)
                self.assertEqual(config.metric, metric)

    def test_to_summary_returns_all_expected_keys(self) -> None:
        config = CheckpointSelectionConfig(
            metric="episode_success_rate",
            override_penalty_weight=0.3,
            dominance_penalty_weight=0.1,
            penalty_mode="tiebreaker",
        )
        summary = config.to_summary()
        self.assertEqual(summary["metric"], "episode_success_rate")
        self.assertAlmostEqual(float(summary["override_penalty_weight"]), 0.3)
        self.assertAlmostEqual(float(summary["dominance_penalty_weight"]), 0.1)
        self.assertEqual(summary["penalty_mode"], "tiebreaker")

    def test_string_penalty_mode_normalized_to_enum(self) -> None:
        config = CheckpointSelectionConfig(
            metric="scenario_success_rate",
            penalty_mode="tiebreaker",
        )
        self.assertEqual(config.penalty_mode, CheckpointPenaltyMode.TIEBREAKER)


class ResolveCheckpointLoadDirTest(unittest.TestCase):
    """Tests for resolve_checkpoint_load_dir()."""

    def test_none_checkpoint_dir_returns_none(self) -> None:
        result = resolve_checkpoint_load_dir(None, checkpoint_selection="best")
        self.assertIsNone(result)

    def test_checkpoint_selection_none_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = resolve_checkpoint_load_dir(
                tmpdir, checkpoint_selection="none"
            )
        self.assertIsNone(result)

    def test_best_selection_prefers_best_subdir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            best_dir = root / "best"
            best_dir.mkdir()
            (best_dir / "metadata.json").write_text("{}", encoding="utf-8")
            last_dir = root / "last"
            last_dir.mkdir()
            (last_dir / "metadata.json").write_text("{}", encoding="utf-8")

            result = resolve_checkpoint_load_dir(root, checkpoint_selection="best")
            self.assertEqual(result, best_dir)

    def test_last_selection_prefers_last_subdir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            best_dir = root / "best"
            best_dir.mkdir()
            (best_dir / "metadata.json").write_text("{}", encoding="utf-8")
            last_dir = root / "last"
            last_dir.mkdir()
            (last_dir / "metadata.json").write_text("{}", encoding="utf-8")

            result = resolve_checkpoint_load_dir(root, checkpoint_selection="last")
            self.assertEqual(result, last_dir)

    def test_best_selection_falls_back_to_last_when_best_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            last_dir = root / "last"
            last_dir.mkdir()
            (last_dir / "metadata.json").write_text("{}", encoding="utf-8")

            result = resolve_checkpoint_load_dir(root, checkpoint_selection="best")
            self.assertEqual(result, last_dir)

    def test_best_selection_falls_back_to_root_when_both_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "metadata.json").write_text("{}", encoding="utf-8")

            result = resolve_checkpoint_load_dir(root, checkpoint_selection="best")
            self.assertEqual(result, root)

    def test_returns_none_when_no_metadata_found(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            # Root exists as directory but has no metadata.json and no subdirs
            result = resolve_checkpoint_load_dir(root, checkpoint_selection="best")
            self.assertIsNone(result)

    def test_accepts_string_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "metadata.json").write_text("{}", encoding="utf-8")
            result = resolve_checkpoint_load_dir(str(tmpdir), checkpoint_selection="last")
            self.assertEqual(result, root)


class CheckpointRunFingerprintTest(unittest.TestCase):
    """Tests for checkpoint_run_fingerprint()."""

    def test_returns_12_character_hex_string(self) -> None:
        fingerprint = checkpoint_run_fingerprint({"key": "value"})
        self.assertEqual(len(fingerprint), 12)
        # Valid hex string
        int(fingerprint, 16)

    def test_same_payload_produces_same_fingerprint(self) -> None:
        payload = {"episodes": 100, "gamma": 0.96, "metric": "scenario_success_rate"}
        fp1 = checkpoint_run_fingerprint(payload)
        fp2 = checkpoint_run_fingerprint(payload)
        self.assertEqual(fp1, fp2)

    def test_different_payloads_produce_different_fingerprints(self) -> None:
        fp1 = checkpoint_run_fingerprint({"key": "value_a"})
        fp2 = checkpoint_run_fingerprint({"key": "value_b"})
        self.assertNotEqual(fp1, fp2)

    def test_key_order_does_not_affect_fingerprint(self) -> None:
        fp1 = checkpoint_run_fingerprint({"a": 1, "b": 2})
        fp2 = checkpoint_run_fingerprint({"b": 2, "a": 1})
        self.assertEqual(fp1, fp2)

    def test_empty_payload_returns_fingerprint(self) -> None:
        fingerprint = checkpoint_run_fingerprint({})
        self.assertEqual(len(fingerprint), 12)

    def test_nested_payload_is_stable(self) -> None:
        payload = {"config": {"width": 12, "height": 12}, "seed": 7}
        fp1 = checkpoint_run_fingerprint(payload)
        fp2 = checkpoint_run_fingerprint(payload)
        self.assertEqual(fp1, fp2)


class FileSha256Test(unittest.TestCase):
    """Tests for file_sha256()."""

    def test_returns_64_character_hex_string(self) -> None:
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"hello world")
            path = Path(f.name)
        try:
            digest = file_sha256(path)
            self.assertEqual(len(digest), 64)
            int(digest, 16)
        finally:
            path.unlink()

    def test_same_content_produces_same_digest(self) -> None:
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"consistent content")
            path = Path(f.name)
        try:
            d1 = file_sha256(path)
            d2 = file_sha256(path)
            self.assertEqual(d1, d2)
        finally:
            path.unlink()

    def test_different_content_produces_different_digest(self) -> None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".a") as fa:
            fa.write(b"content_a")
            path_a = Path(fa.name)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".b") as fb:
            fb.write(b"content_b")
            path_b = Path(fb.name)
        try:
            self.assertNotEqual(file_sha256(path_a), file_sha256(path_b))
        finally:
            path_a.unlink()
            path_b.unlink()

    def test_empty_file_returns_known_sha256(self) -> None:
        # SHA-256 of empty content is a known constant
        with tempfile.NamedTemporaryFile(delete=False) as f:
            path = Path(f.name)
        try:
            digest = file_sha256(path)
            self.assertEqual(
                digest,
                "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            )
        finally:
            path.unlink()


class CheckpointPreloadFingerprintTest(unittest.TestCase):
    """Tests for checkpoint_preload_fingerprint()."""

    def test_none_load_brain_returns_null_payload(self) -> None:
        result = checkpoint_preload_fingerprint(None)
        self.assertIsNone(result["load_brain"])
        self.assertIsNone(result["metadata_sha256"])
        self.assertIsNone(result["load_modules"])
        self.assertIsNone(result["module_sha256"])

    def test_file_path_returns_artifact_fingerprint(self) -> None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as f:
            f.write(b"brain data")
            path = Path(f.name)
        try:
            result = checkpoint_preload_fingerprint(path)
            self.assertEqual(result["load_brain"], str(path))
            self.assertIsNotNone(result["artifact_sha256"])
            self.assertIsNone(result["module_sha256"])
        finally:
            path.unlink()

    def test_file_path_with_modules_normalizes_module_list(self) -> None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as f:
            f.write(b"data")
            path = Path(f.name)
        try:
            result = checkpoint_preload_fingerprint(
                path, load_modules=["visual_cortex", "sensory_cortex", "visual_cortex"]
            )
            # Modules should be sorted and de-duplicated
            self.assertEqual(
                result["load_modules"], ["sensory_cortex", "visual_cortex"]
            )
        finally:
            path.unlink()

    def test_file_path_with_no_modules_has_none_load_modules(self) -> None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as f:
            f.write(b"data")
            path = Path(f.name)
        try:
            result = checkpoint_preload_fingerprint(path, load_modules=None)
            self.assertIsNone(result["load_modules"])
        finally:
            path.unlink()

    def test_directory_path_reads_metadata_and_hashes_modules(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            metadata = {"modules": {"visual_cortex": {}, "sensory_cortex": {}}}
            (root / "metadata.json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )
            # Create fake .npz files
            (root / "visual_cortex.npz").write_bytes(b"visual_data")
            (root / "sensory_cortex.npz").write_bytes(b"sensory_data")

            result = checkpoint_preload_fingerprint(root)
            self.assertEqual(result["load_brain"], str(root))
            self.assertIsNotNone(result["metadata_sha256"])
            self.assertIn("visual_cortex", result["module_sha256"])
            self.assertIn("sensory_cortex", result["module_sha256"])
            self.assertIsNone(result.get("artifact_sha256"))

    def test_directory_path_with_specified_modules_only_hashes_those(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            metadata = {"modules": {"visual_cortex": {}, "sensory_cortex": {}, "motor": {}}}
            (root / "metadata.json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )
            (root / "visual_cortex.npz").write_bytes(b"v")
            (root / "sensory_cortex.npz").write_bytes(b"s")
            (root / "motor.npz").write_bytes(b"m")

            result = checkpoint_preload_fingerprint(
                root, load_modules=["visual_cortex"]
            )
            self.assertEqual(result["load_modules"], ["visual_cortex"])
            self.assertIn("visual_cortex", result["module_sha256"])
            self.assertNotIn("sensory_cortex", result["module_sha256"])


class JsonifyObservationTest(unittest.TestCase):
    """Tests for jsonify_observation()."""

    def test_string_values_preserved(self) -> None:
        result = jsonify_observation({"key": "value"})
        self.assertEqual(result["key"], "value")

    def test_numeric_values_preserved(self) -> None:
        result = jsonify_observation({"int_val": 42, "float_val": 3.14})
        self.assertEqual(result["int_val"], 42)
        self.assertAlmostEqual(result["float_val"], 3.14)

    def test_none_values_preserved(self) -> None:
        result = jsonify_observation({"key": None})
        self.assertIsNone(result["key"])

    def test_dict_values_are_recursively_converted(self) -> None:
        result = jsonify_observation({"nested": {"a": 1, "b": 2}})
        self.assertEqual(result["nested"], {"a": 1, "b": 2})

    def test_list_values_are_recursively_converted(self) -> None:
        result = jsonify_observation({"items": [1, 2, 3]})
        self.assertEqual(result["items"], [1, 2, 3])

    def test_empty_observation_returns_empty_dict(self) -> None:
        result = jsonify_observation({})
        self.assertEqual(result, {})

    def test_keys_are_preserved(self) -> None:
        obs = {"sensor_a": 0.5, "sensor_b": 0.7, "sensor_c": 0.0}
        result = jsonify_observation(obs)
        self.assertEqual(set(result.keys()), set(obs.keys()))


class CheckpointCandidateCompositeScoreTest(unittest.TestCase):
    """Tests for checkpoint_candidate_composite_score()."""

    def test_zero_penalty_weights_returns_primary_metric_value(self) -> None:
        candidate = {
            "scenario_success_rate": 0.75,
            "episode_success_rate": 0.5,
            "mean_reward": 1.0,
            "evaluation_summary": {
                "mean_final_reflex_override_rate": 0.5,
                "mean_reflex_dominance": 0.5,
            },
            "episode": 5,
        }
        config = CheckpointSelectionConfig(
            metric="scenario_success_rate",
            override_penalty_weight=0.0,
            dominance_penalty_weight=0.0,
            penalty_mode=CheckpointPenaltyMode.TIEBREAKER,
        )
        score = checkpoint_candidate_composite_score(candidate, config)
        self.assertAlmostEqual(score, 0.75)

    def test_composite_score_penalizes_override_rate(self) -> None:
        candidate = {
            "scenario_success_rate": 0.8,
            "episode_success_rate": 0.5,
            "mean_reward": 1.0,
            "evaluation_summary": {
                "mean_final_reflex_override_rate": 0.4,
                "mean_reflex_dominance": 0.0,
            },
            "episode": 5,
        }
        config = CheckpointSelectionConfig(
            metric="scenario_success_rate",
            override_penalty_weight=0.5,
            dominance_penalty_weight=0.0,
            penalty_mode=CheckpointPenaltyMode.TIEBREAKER,
        )
        # 0.8 - (0.5 * 0.4) = 0.8 - 0.2 = 0.6
        score = checkpoint_candidate_composite_score(candidate, config)
        self.assertAlmostEqual(score, 0.6)

    def test_composite_score_penalizes_dominance(self) -> None:
        candidate = {
            "scenario_success_rate": 0.9,
            "episode_success_rate": 0.5,
            "mean_reward": 1.0,
            "evaluation_summary": {
                "mean_final_reflex_override_rate": 0.0,
                "mean_reflex_dominance": 0.5,
            },
            "episode": 5,
        }
        config = CheckpointSelectionConfig(
            metric="scenario_success_rate",
            override_penalty_weight=0.0,
            dominance_penalty_weight=0.2,
            penalty_mode=CheckpointPenaltyMode.TIEBREAKER,
        )
        # 0.9 - (0.2 * 0.5) = 0.9 - 0.1 = 0.8
        score = checkpoint_candidate_composite_score(candidate, config)
        self.assertAlmostEqual(score, 0.8)

    def test_composite_score_uses_primary_metric_regardless_of_mode(self) -> None:
        candidate = {
            "scenario_success_rate": 0.6,
            "episode_success_rate": 0.9,
            "mean_reward": 1.5,
            "evaluation_summary": {},
            "episode": 1,
        }
        config_tiebreaker = CheckpointSelectionConfig(
            metric="episode_success_rate",
            penalty_mode=CheckpointPenaltyMode.TIEBREAKER,
        )
        config_direct = CheckpointSelectionConfig(
            metric="episode_success_rate",
            penalty_mode=CheckpointPenaltyMode.DIRECT,
        )
        # Both configs should produce composite score based on episode_success_rate
        score_tb = checkpoint_candidate_composite_score(candidate, config_tiebreaker)
        score_direct = checkpoint_candidate_composite_score(candidate, config_direct)
        self.assertAlmostEqual(score_tb, 0.9)
        self.assertAlmostEqual(score_direct, 0.9)

    def test_tiebreaker_mode_is_overridden_for_composite_calculation(self) -> None:
        """composite_score always uses direct penalty mode internally."""
        candidate = {
            "scenario_success_rate": 0.7,
            "episode_success_rate": 0.5,
            "mean_reward": 1.0,
            "evaluation_summary": {
                "mean_final_reflex_override_rate": 0.3,
                "mean_reflex_dominance": 0.1,
            },
            "episode": 1,
        }
        config = CheckpointSelectionConfig(
            metric="scenario_success_rate",
            override_penalty_weight=1.0,
            dominance_penalty_weight=1.0,
            penalty_mode=CheckpointPenaltyMode.TIEBREAKER,
        )
        # 0.7 - (1.0 * 0.3) - (1.0 * 0.1) = 0.7 - 0.3 - 0.1 = 0.3
        score = checkpoint_candidate_composite_score(candidate, config)
        self.assertAlmostEqual(score, 0.3)


if __name__ == "__main__":
    unittest.main()
