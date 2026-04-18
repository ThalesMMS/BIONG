"""Additional edge-case tests for spider_cortex_sim.offline_analysis.writers.

The basic happy-path tests live in test_writers.py. This file focuses on
boundary conditions and less-exercised paths in the newly extracted module.
"""
from __future__ import annotations

import csv
import io
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


class MarkdownTableEdgeCasesTest(unittest.TestCase):
    def test_empty_rows_returns_no_data_message(self) -> None:
        result = _markdown_table([], ["col_a", "col_b"])
        self.assertEqual(result, "_No data available._")

    def test_single_row_single_col(self) -> None:
        result = _markdown_table([["value"]], ["header"])
        self.assertIn("| header |", result)
        self.assertIn("| value |", result)

    def test_separator_row_uses_triple_dashes(self) -> None:
        result = _markdown_table([["x"]], ["h"])
        lines = result.split("\n")
        self.assertEqual(lines[1], "| --- |")

    def test_multiple_columns_separator_correct(self) -> None:
        result = _markdown_table([["a", "b"]], ["col1", "col2"])
        lines = result.split("\n")
        self.assertEqual(lines[1], "| --- | --- |")

    def test_values_coerced_to_string(self) -> None:
        result = _markdown_table([[1, 2.5, None]], ["a", "b", "c"])
        self.assertIn("1", result)
        self.assertIn("2.5", result)
        self.assertIn("None", result)

    def test_multiple_rows_all_present(self) -> None:
        rows = [["row1val"], ["row2val"], ["row3val"]]
        result = _markdown_table(rows, ["col"])
        for val in ["row1val", "row2val", "row3val"]:
            self.assertIn(val, result)

    def test_result_is_single_string_with_newlines(self) -> None:
        result = _markdown_table([["a", "b"]], ["x", "y"])
        self.assertIsInstance(result, str)
        self.assertIn("\n", result)


class MarkdownValueEdgeCasesTest(unittest.TestCase):
    def test_none_returns_dash(self) -> None:
        self.assertEqual(_markdown_value(None), "—")

    def test_bool_true_returns_yes(self) -> None:
        self.assertEqual(_markdown_value(True), "yes")

    def test_bool_false_returns_no(self) -> None:
        self.assertEqual(_markdown_value(False), "no")

    def test_float_formatted_to_two_digits(self) -> None:
        self.assertEqual(_markdown_value(0.5), "0.50")

    def test_int_formatted_as_float(self) -> None:
        # int 3 -> _coerce_optional_float(3) = 3.0, not isinstance(3, str) -> formats
        self.assertEqual(_markdown_value(3), "3.00")

    def test_string_returned_as_is(self) -> None:
        self.assertEqual(_markdown_value("hello"), "hello")

    def test_empty_string_returns_dash(self) -> None:
        self.assertEqual(_markdown_value(""), "—")

    def test_custom_digits(self) -> None:
        self.assertEqual(_markdown_value(1.23456789, digits=4), "1.2346")

    def test_zero_digits(self) -> None:
        self.assertEqual(_markdown_value(1.7, digits=0), "2")

    def test_negative_float(self) -> None:
        self.assertEqual(_markdown_value(-0.5), "-0.50")

    def test_string_that_looks_like_number_not_formatted(self) -> None:
        # str type bypasses float formatting even if parseable as float
        self.assertEqual(_markdown_value("1.5"), "1.5")


class TableHelpersEdgeCasesTest(unittest.TestCase):
    def test_table_with_empty_rows(self) -> None:
        result = _table(["col"], [])
        self.assertEqual(result["columns"], ["col"])
        self.assertEqual(result["rows"], [])

    def test_table_columns_coerced_to_strings(self) -> None:
        result = _table([1, 2], [])
        self.assertEqual(result["columns"], ["1", "2"])

    def test_table_rows_are_dicts(self) -> None:
        result = _table(["a"], [{"a": 1}])
        self.assertIsInstance(result["rows"][0], dict)

    def test_table_rows_on_non_list_returns_empty(self) -> None:
        bad_table = {"columns": ["a"], "rows": "not-a-list"}
        self.assertEqual(_table_rows(bad_table), [])

    def test_table_rows_on_missing_key_returns_empty(self) -> None:
        self.assertEqual(_table_rows({"columns": ["a"]}), [])

    def test_table_rows_on_valid_list(self) -> None:
        rows = [{"a": 1}, {"a": 2}]
        result = _table_rows({"columns": ["a"], "rows": rows})
        self.assertEqual(result, rows)


class WriteCsvEdgeCasesTest(unittest.TestCase):
    def test_empty_rows_writes_header_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.csv"
            _write_csv(path, [], ["col_a", "col_b"])
            content = path.read_text(encoding="utf-8")
        self.assertIn("col_a", content)
        self.assertIn("col_b", content)

    def test_missing_field_written_as_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.csv"
            _write_csv(path, [{"col_a": "val"}], ["col_a", "col_b"])
            content = path.read_text(encoding="utf-8")
        rows = list(csv.reader(io.StringIO(content)))
        self.assertEqual(len(rows), 2)
        self.assertIn("val", rows[1])

    def test_multiple_rows_written_correctly(self) -> None:
        rows = [{"name": f"row{i}"} for i in range(5)]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.csv"
            _write_csv(path, rows, ["name"])
            content = path.read_text(encoding="utf-8")
        parsed_rows = list(csv.reader(io.StringIO(content)))
        self.assertEqual(len(parsed_rows), 6)


class WriteSvgTest(unittest.TestCase):
    def test_writes_expected_content(self) -> None:
        svg_content = '<svg xmlns="http://www.w3.org/2000/svg"><text>hello</text></svg>'
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "chart.svg"
            _write_svg(path, svg_content)
            result = path.read_text(encoding="utf-8")
        self.assertEqual(result, svg_content)

    def test_overwrites_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "chart.svg"
            _write_svg(path, "<svg>old</svg>")
            _write_svg(path, "<svg>new</svg>")
            result = path.read_text(encoding="utf-8")
        self.assertEqual(result, "<svg>new</svg>")
