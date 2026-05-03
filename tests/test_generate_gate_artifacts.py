from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from spider_cortex_sim.scenarios import generate_gate_artifacts


class GenerateGateArtifactsTest(unittest.TestCase):
    def test_run_gate_writes_report_under_gate_report_subdirectory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.object(
            generate_gate_artifacts,
            "_run",
        ) as run_mock:
            output_dir = Path(tmpdir)

            generate_gate_artifacts._run_gate(
                gate="night_rest",
                label="before",
                seeds=[0],
                episodes=1,
                output_dir=output_dir,
            )

        self.assertEqual(run_mock.call_count, 2)
        report_cmd = run_mock.call_args_list[1].args[0]
        output_index = report_cmd.index("--output-dir") + 1
        self.assertEqual(
            Path(report_cmd[output_index]),
            output_dir / "night_rest" / "before" / "gate_report",
        )

    def test_run_gate_rejects_path_traversal_segments(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            for field, kwargs in (
                ("gate", {"gate": "../night_rest", "label": "before"}),
                ("label", {"gate": "night_rest", "label": "before/after"}),
            ):
                with self.subTest(field=field):
                    with self.assertRaises(ValueError):
                        generate_gate_artifacts._run_gate(
                            **kwargs,
                            seeds=[0],
                            episodes=1,
                            output_dir=output_dir,
                        )


if __name__ == "__main__":
    unittest.main()
