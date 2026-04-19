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

from tests.fixtures.checkpointing import _write_checkpoint_artifact

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
