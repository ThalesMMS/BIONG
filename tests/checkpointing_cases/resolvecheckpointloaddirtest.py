from __future__ import annotations

from .shared import *


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
