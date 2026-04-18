from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from spider_cortex_sim.offline_analysis.writers import (
    _markdown_table,
    _markdown_value,
    _table,
    _table_rows,
    _write_csv,
    _write_svg,
)


class WritersFormatterTest(unittest.TestCase):
    def test_markdown_value_formats_float_with_requested_digits(self) -> None:
        self.assertEqual(_markdown_value(0.12345, digits=3), "0.123")

    def test_markdown_table_renders_headers_and_rows(self) -> None:
        markdown = _markdown_table([["night_rest", 1.0]], ["scenario", "success"])

        self.assertIn("| scenario | success |", markdown)
        self.assertIn("| night_rest | 1.0 |", markdown)

    def test_markdown_table_escapes_pipes_and_newlines(self) -> None:
        markdown = _markdown_table(
            [["night|rest", "line one\nline two"]],
            ["scenario|name", "notes"],
        )

        self.assertIn("| scenario\\|name | notes |", markdown)
        self.assertIn("| night\\|rest | line one<br>line two |", markdown)

    def test_table_helpers_preserve_columns_and_rows(self) -> None:
        table = _table(["scenario", "success"], [{"scenario": "night_rest", "success": 1.0}])

        self.assertEqual(table["columns"], ["scenario", "success"])
        self.assertEqual(_table_rows(table), [{"scenario": "night_rest", "success": 1.0}])

    def test_table_rows_filters_invalid_payloads(self) -> None:
        rows = _table_rows({"rows": [{"scenario": "night_rest"}, "invalid"]})

        self.assertEqual(rows, [{"scenario": "night_rest"}])
        self.assertEqual(_table_rows({"rows": "invalid"}), [])

    def test_write_csv_and_svg_write_expected_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "rows.csv"
            svg_path = Path(tmpdir) / "chart.svg"

            _write_csv(csv_path, [{"scenario": "night_rest", "success": 1.0}], ["scenario", "success"])
            _write_svg(svg_path, "<svg></svg>")

            self.assertIn("night_rest", csv_path.read_text(encoding="utf-8"))
            self.assertEqual(svg_path.read_text(encoding="utf-8"), "<svg></svg>")
