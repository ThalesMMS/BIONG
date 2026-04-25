"""Tests for spider_cortex_sim.cli.validation module.

Covers the _argument_error helper function introduced in the PR refactor
of the CLI into submodules.
"""

from __future__ import annotations

import argparse
import io
import sys
import unittest
from unittest.mock import MagicMock, patch

from spider_cortex_sim.cli.validation import _argument_error


class ArgumentErrorWithParserTest(unittest.TestCase):
    """Tests for _argument_error when a _parser attribute is present."""

    def test_delegates_to_parser_error_when_parser_present(self) -> None:
        parser = argparse.ArgumentParser()
        args = parser.parse_args([])
        args._parser = parser

        with self.assertRaises(SystemExit) as ctx:
            with patch("sys.stderr", io.StringIO()):
                _argument_error(args, "test error message")

        # argparse.error exits with code 2
        self.assertEqual(ctx.exception.code, 2)

    def test_parser_error_message_in_stderr_when_parser_present(self) -> None:
        parser = argparse.ArgumentParser(prog="myapp")
        args = parser.parse_args([])
        args._parser = parser
        stderr = io.StringIO()

        with self.assertRaises(SystemExit):
            with patch("sys.stderr", stderr):
                _argument_error(args, "custom failure message")

        self.assertIn("custom failure message", stderr.getvalue())

    def test_mock_parser_error_is_called_with_message(self) -> None:
        mock_parser = MagicMock()
        mock_parser.error.side_effect = SystemExit(2)
        args = argparse.Namespace(_parser=mock_parser)

        with self.assertRaises(SystemExit):
            _argument_error(args, "an argument error")

        mock_parser.error.assert_called_once_with("an argument error")


class ArgumentErrorWithoutParserTest(unittest.TestCase):
    """Tests for _argument_error when no _parser attribute is present."""

    def test_raises_system_exit_when_no_parser(self) -> None:
        args = argparse.Namespace()  # no _parser
        stderr = io.StringIO()

        with self.assertRaises(SystemExit) as ctx:
            with patch("sys.stderr", stderr):
                _argument_error(args, "some error")

        self.assertEqual(ctx.exception.code, 2)

    def test_prints_error_prefix_to_stderr(self) -> None:
        args = argparse.Namespace()
        stderr = io.StringIO()

        with self.assertRaises(SystemExit):
            with patch("sys.stderr", stderr):
                _argument_error(args, "missing argument")

        output = stderr.getvalue()
        self.assertIn("error:", output)

    def test_prints_message_to_stderr(self) -> None:
        args = argparse.Namespace()
        stderr = io.StringIO()

        with self.assertRaises(SystemExit):
            with patch("sys.stderr", stderr):
                _argument_error(args, "specific message here")

        self.assertIn("specific message here", stderr.getvalue())

    def test_exit_code_is_2_without_parser(self) -> None:
        args = argparse.Namespace()
        stderr = io.StringIO()

        with self.assertRaises(SystemExit) as ctx:
            with patch("sys.stderr", stderr):
                _argument_error(args, "error occurred")

        self.assertEqual(ctx.exception.code, 2)

    def test_parser_none_attribute_falls_back_to_stderr(self) -> None:
        """When _parser is None (falsy), fall back to printing to stderr."""
        args = argparse.Namespace(_parser=None)
        stderr = io.StringIO()

        with self.assertRaises(SystemExit) as ctx:
            with patch("sys.stderr", stderr):
                _argument_error(args, "null parser fallback")

        self.assertEqual(ctx.exception.code, 2)
        self.assertIn("null parser fallback", stderr.getvalue())

    def test_empty_message_is_handled(self) -> None:
        args = argparse.Namespace()
        stderr = io.StringIO()

        with self.assertRaises(SystemExit) as ctx:
            with patch("sys.stderr", stderr):
                _argument_error(args, "")

        self.assertEqual(ctx.exception.code, 2)

    def test_long_message_is_fully_printed(self) -> None:
        args = argparse.Namespace()
        long_msg = "x" * 500
        stderr = io.StringIO()

        with self.assertRaises(SystemExit):
            with patch("sys.stderr", stderr):
                _argument_error(args, long_msg)

        self.assertIn(long_msg, stderr.getvalue())


if __name__ == "__main__":
    unittest.main()