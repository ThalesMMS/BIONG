"""Tests for the CLI argument parser defaults.

The PR changed day_length and night_length defaults:
  day_length:  36 → 18
  night_length: 24 → 12
"""

import unittest

from spider_cortex_sim.cli import build_parser


class BuildParserDefaultsTest(unittest.TestCase):
    """Verify that build_parser() carries the PR-updated cycle length defaults."""

    def _defaults(self) -> dict:
        parser = build_parser()
        return vars(parser.parse_args([]))

    def test_day_length_default_is_18(self) -> None:
        defaults = self._defaults()
        self.assertEqual(defaults["day_length"], 18)

    def test_night_length_default_is_12(self) -> None:
        defaults = self._defaults()
        self.assertEqual(defaults["night_length"], 12)

    def test_day_length_can_be_overridden(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--day-length", "36"])
        self.assertEqual(args.day_length, 36)

    def test_night_length_can_be_overridden(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--night-length", "24"])
        self.assertEqual(args.night_length, 24)

    def test_day_length_accepts_integer(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--day-length", "10"])
        self.assertIsInstance(args.day_length, int)

    def test_night_length_accepts_integer(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--night-length", "8"])
        self.assertIsInstance(args.night_length, int)

    def test_day_length_default_shorter_than_old_default(self) -> None:
        """The new default (18) is strictly less than the old default (36)."""
        defaults = self._defaults()
        self.assertLess(defaults["day_length"], 36)

    def test_night_length_default_shorter_than_old_default(self) -> None:
        """The new default (12) is strictly less than the old default (24)."""
        defaults = self._defaults()
        self.assertLess(defaults["night_length"], 24)

    def test_day_length_default_has_3_to_2_ratio_with_night_length_default(self) -> None:
        """The defaults should preserve a 3:2 day-to-night ratio via day_length * 2 == night_length * 3."""
        defaults = self._defaults()
        self.assertEqual(defaults["day_length"] * 2, defaults["night_length"] * 3)

    def test_width_and_height_defaults_unchanged(self) -> None:
        """Sanity check: unrelated defaults were not altered by the PR."""
        defaults = self._defaults()
        self.assertEqual(defaults["width"], 12)
        self.assertEqual(defaults["height"], 12)

    def test_food_count_default_unchanged(self) -> None:
        defaults = self._defaults()
        self.assertEqual(defaults["food_count"], 4)


if __name__ == "__main__":
    unittest.main()
