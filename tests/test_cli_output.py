"""Tests for spider_cortex_sim.cli.output module.

Covers _build_and_record_benchmark_package extracted in the PR refactor.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, call


class BuildAndRecordBenchmarkPackageTest(unittest.TestCase):
    """Tests for cli.output._build_and_record_benchmark_package."""

    def _import_function(self):
        from spider_cortex_sim.cli.output import _build_and_record_benchmark_package
        return _build_and_record_benchmark_package

    def test_calls_assemble_benchmark_package(self) -> None:
        _build_and_record_benchmark_package = self._import_function()
        manifest = MagicMock()
        manifest.package_version = "1.0"
        summary: dict = {}
        with patch(
            "spider_cortex_sim.cli.output.assemble_benchmark_package",
            return_value=manifest,
        ) as mock_assemble, patch(
            "spider_cortex_sim.cli.output._benchmark_package_manifest_summary",
            return_value={"package_version": "1.0"},
        ):
            _build_and_record_benchmark_package(
                output_dir=Path("/tmp/pkg"),
                summary=summary,
                behavior_csv=None,
                behavior_rows=[],
                command_metadata={"argv": []},
            )

        mock_assemble.assert_called_once()

    def test_manifest_summary_added_to_summary_dict(self) -> None:
        _build_and_record_benchmark_package = self._import_function()
        manifest = MagicMock()
        manifest_summary = {"package_version": "2.0", "seed_count": 3}
        summary: dict = {}

        with patch(
            "spider_cortex_sim.cli.output.assemble_benchmark_package",
            return_value=manifest,
        ), patch(
            "spider_cortex_sim.cli.output._benchmark_package_manifest_summary",
            return_value=manifest_summary,
        ):
            _build_and_record_benchmark_package(
                output_dir=Path("/tmp/pkg"),
                summary=summary,
                behavior_csv=None,
                behavior_rows=[],
                command_metadata={"argv": []},
            )

        self.assertIn("benchmark_package", summary)
        self.assertEqual(summary["benchmark_package"]["package_version"], "2.0")

    def test_returns_manifest(self) -> None:
        _build_and_record_benchmark_package = self._import_function()
        manifest = MagicMock()
        summary: dict = {}

        with patch(
            "spider_cortex_sim.cli.output.assemble_benchmark_package",
            return_value=manifest,
        ), patch(
            "spider_cortex_sim.cli.output._benchmark_package_manifest_summary",
            return_value={},
        ):
            result = _build_and_record_benchmark_package(
                output_dir=Path("/tmp/pkg"),
                summary=summary,
                behavior_csv=None,
                behavior_rows=[],
                command_metadata={"argv": []},
            )

        self.assertIs(result, manifest)

    def test_trace_default_is_empty_sequence(self) -> None:
        """Test that trace defaults to empty sequence and is passed correctly."""
        _build_and_record_benchmark_package = self._import_function()
        manifest = MagicMock()
        summary: dict = {}
        captured_kwargs = {}

        def mock_assemble(output_dir, summary_arg, behavior_csv, **kwargs):
            captured_kwargs.update(kwargs)
            return manifest

        with patch(
            "spider_cortex_sim.cli.output.assemble_benchmark_package",
            side_effect=mock_assemble,
        ), patch(
            "spider_cortex_sim.cli.output._benchmark_package_manifest_summary",
            return_value={},
        ):
            _build_and_record_benchmark_package(
                output_dir=Path("/tmp/pkg"),
                summary=summary,
                behavior_csv=None,
                behavior_rows=[],
                command_metadata={"argv": []},
            )

        self.assertIn("trace", captured_kwargs)
        self.assertEqual(list(captured_kwargs["trace"]), [])

    def test_behavior_rows_passed_to_assemble(self) -> None:
        _build_and_record_benchmark_package = self._import_function()
        manifest = MagicMock()
        summary: dict = {}
        behavior_rows = [{"scenario": "night_rest", "success": 1}]
        captured_kwargs = {}

        def mock_assemble(output_dir, summary_arg, behavior_csv, **kwargs):
            captured_kwargs.update(kwargs)
            return manifest

        with patch(
            "spider_cortex_sim.cli.output.assemble_benchmark_package",
            side_effect=mock_assemble,
        ), patch(
            "spider_cortex_sim.cli.output._benchmark_package_manifest_summary",
            return_value={},
        ):
            _build_and_record_benchmark_package(
                output_dir=Path("/tmp/pkg"),
                summary=summary,
                behavior_csv=None,
                behavior_rows=behavior_rows,
                command_metadata={"argv": []},
            )

        self.assertEqual(captured_kwargs.get("behavior_rows"), behavior_rows)

    def test_command_metadata_passed_to_assemble(self) -> None:
        _build_and_record_benchmark_package = self._import_function()
        manifest = MagicMock()
        summary: dict = {}
        cmd_meta = {"argv": ["--budget-profile", "paper"]}
        captured_kwargs = {}

        def mock_assemble(output_dir, summary_arg, behavior_csv, **kwargs):
            captured_kwargs.update(kwargs)
            return manifest

        with patch(
            "spider_cortex_sim.cli.output.assemble_benchmark_package",
            side_effect=mock_assemble,
        ), patch(
            "spider_cortex_sim.cli.output._benchmark_package_manifest_summary",
            return_value={},
        ):
            _build_and_record_benchmark_package(
                output_dir=Path("/tmp/pkg"),
                summary=summary,
                behavior_csv=None,
                behavior_rows=[],
                command_metadata=cmd_meta,
            )

        self.assertEqual(captured_kwargs.get("command_metadata"), cmd_meta)

    def test_summary_dict_mutated_in_place(self) -> None:
        """The function mutates the provided summary dict (does not return a new one)."""
        _build_and_record_benchmark_package = self._import_function()
        manifest = MagicMock()
        original_summary = {"config": {"world": {}}}
        summary = original_summary

        with patch(
            "spider_cortex_sim.cli.output.assemble_benchmark_package",
            return_value=manifest,
        ), patch(
            "spider_cortex_sim.cli.output._benchmark_package_manifest_summary",
            return_value={"key": "value"},
        ):
            _build_and_record_benchmark_package(
                output_dir=Path("/tmp/pkg"),
                summary=summary,
                behavior_csv=None,
                behavior_rows=[],
                command_metadata={},
            )

        # The original dict is mutated in place
        self.assertIn("benchmark_package", original_summary)


if __name__ == "__main__":
    unittest.main()