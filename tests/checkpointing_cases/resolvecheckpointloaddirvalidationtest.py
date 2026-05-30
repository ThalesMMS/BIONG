from __future__ import annotations

from .shared import *


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
