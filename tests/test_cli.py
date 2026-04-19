"""Tests for the CLI argument parser defaults.

The PR changed day_length and night_length defaults:
  day_length:  36 → 18
  night_length: 24 → 12
"""

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from spider_cortex_sim.cli import build_parser, main


class PublicReexportsTest(unittest.TestCase):
    """Verify package-level entrypoint re-exports remain import-compatible."""

    def test_cli_reexports_build_parser_and_main(self) -> None:
        from spider_cortex_sim.cli import build_parser as exported_build_parser
        from spider_cortex_sim.cli import main as exported_main

        self.assertIs(exported_build_parser, build_parser)
        self.assertIs(exported_main, main)

    def test_gui_reexports_run_gui_and_spider_gui(self) -> None:
        from spider_cortex_sim.gui import SpiderGUI, run_gui

        self.assertTrue(callable(run_gui))
        self.assertEqual(SpiderGUI.__name__, "SpiderGUI")


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


class CheckpointSelectionValidationTest(unittest.TestCase):
    def _minimal_summary(self) -> dict[str, object]:
        seed_level = [
            {
                "metric_name": "scenario_success_rate",
                "seed": 7,
                "value": 0.5,
                "condition": "modular_full",
            },
            {
                "metric_name": "scenario_success_rate",
                "seed": 17,
                "value": 1.0,
                "condition": "modular_full",
            },
        ]
        uncertainty = {
            "mean": 0.75,
            "ci_lower": 0.5,
            "ci_upper": 1.0,
            "std_error": 0.1,
            "n_seeds": 2,
            "confidence_level": 0.95,
            "seed_values": [0.5, 1.0],
        }
        return {
            "config": {
                "world": {"reward_profile": "classic", "map_template": "central_burrow"},
                "budget": {
                    "profile": "paper",
                    "benchmark_strength": "paper",
                    "resolved": {"episodes": 10, "evaluation_episodes": 2},
                },
                "training_regime": {"mode": "flat"},
                "operational_profile": {"name": "default_v1"},
                "noise_profile": {"name": "none"},
            },
            "checkpointing": {
                "selection": "best",
                "metric": "scenario_success_rate",
            },
            "training_last_window": {
                "mean_reward": 0.0,
                "mean_night_shelter_occupancy_rate": 0.0,
                "mean_night_stillness_rate": 0.0,
            },
            "evaluation": {
                "mean_reward": 0.0,
                "mean_food": 0.0,
                "mean_sleep": 0.0,
                "mean_sleep_debt": 0.0,
                "mean_predator_contacts": 0.0,
                "mean_predator_escapes": 0.0,
                "mean_night_shelter_occupancy_rate": 0.0,
                "mean_night_stillness_rate": 0.0,
                "mean_predator_response_events": 0.0,
                "mean_predator_response_latency": 0.0,
                "mean_predator_mode_transitions": 0.0,
                "dominant_predator_state": "idle",
                "mean_food_distance_delta": 0.0,
                "mean_shelter_distance_delta": 0.0,
                "survival_rate": 0.0,
                "mean_night_role_distribution": {},
            },
            "behavior_evaluation": {
                "summary": {
                    "scenario_success_rate": 0.75,
                    "episode_success_rate": 0.75,
                    "mean_reward": 3.0,
                },
                "suite": {
                    "night_rest": {
                        "success_rate": 0.75,
                        "uncertainty": {"success_rate": uncertainty},
                        "seed_level": [
                            {**row, "scenario": "night_rest"}
                            for row in seed_level
                        ],
                    }
                },
                "ablations": {
                    "reference_variant": "modular_full",
                    "variants": {
                        "modular_full": {
                            "summary": {
                                "scenario_success_rate": 0.75,
                                "episode_success_rate": 0.75,
                                "mean_reward": 3.0,
                            },
                            "seed_level": seed_level,
                            "uncertainty": {
                                "scenario_success_rate": uncertainty,
                                "episode_success_rate": uncertainty,
                            },
                            "suite": {
                                "night_rest": {
                                    "success_rate": 0.75,
                                    "uncertainty": {"success_rate": uncertainty},
                                }
                            },
                        }
                    },
                },
            },
        }

    def test_paper_budget_profile_requires_checkpoint_selection(self) -> None:
        stderr = io.StringIO()

        with patch(
            "sys.argv",
            ["spider_cortex_sim", "--budget-profile", "paper"],
        ), patch("sys.stderr", stderr), patch(
            "spider_cortex_sim.cli.SpiderSimulation"
        ) as simulation_mock:
            with self.assertRaises(SystemExit) as ctx:
                main()

        self.assertEqual(ctx.exception.code, 2)
        error_output = stderr.getvalue()
        self.assertIn("paper", error_output)
        self.assertIn("--checkpoint-selection best", error_output)
        simulation_mock.assert_not_called()

    def test_paper_error_message_mentions_requires_checkpoint_selection(self) -> None:
        stderr = io.StringIO()

        with patch(
            "sys.argv",
            ["spider_cortex_sim", "--budget-profile", "paper"],
        ), patch("sys.stderr", stderr), patch(
            "spider_cortex_sim.cli.SpiderSimulation"
        ):
            with self.assertRaises(SystemExit):
                main()

        error_output = stderr.getvalue()
        self.assertIn("requires checkpoint selection", error_output)

    def test_paper_with_checkpoint_selection_none_still_rejected(self) -> None:
        stderr = io.StringIO()

        with patch(
            "sys.argv",
            [
                "spider_cortex_sim",
                "--budget-profile",
                "paper",
                "--checkpoint-selection",
                "none",
            ],
        ), patch("sys.stderr", stderr), patch(
            "spider_cortex_sim.cli.SpiderSimulation"
        ) as simulation_mock:
            with self.assertRaises(SystemExit) as ctx:
                main()

        self.assertEqual(ctx.exception.code, 2)
        simulation_mock.assert_not_called()

    def test_checkpoint_selection_parser_default_is_none_string(self) -> None:
        parser = build_parser()
        args = parser.parse_args([])
        self.assertEqual(args.checkpoint_selection, "none")

    def test_checkpoint_selection_best_is_accepted_by_parser(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--checkpoint-selection", "best"])
        self.assertEqual(args.checkpoint_selection, "best")

    def test_budget_profile_paper_is_accepted_by_parser(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--budget-profile", "paper"])
        self.assertEqual(args.budget_profile, "paper")

    def test_benchmark_package_requires_paper_budget_profile(self) -> None:
        stderr = io.StringIO()

        with patch(
            "sys.argv",
            [
                "spider_cortex_sim",
                "--budget-profile",
                "smoke",
                "--checkpoint-selection",
                "best",
                "--benchmark-package",
                "package",
            ],
        ), patch("sys.stderr", stderr), patch(
            "spider_cortex_sim.cli.SpiderSimulation"
        ) as simulation_mock:
            with self.assertRaises(SystemExit) as ctx:
                main()

        self.assertEqual(ctx.exception.code, 2)
        self.assertIn("--budget-profile paper", stderr.getvalue())
        simulation_mock.assert_not_called()

    def test_benchmark_package_requires_best_checkpoint_selection(self) -> None:
        stderr = io.StringIO()

        with patch(
            "sys.argv",
            [
                "spider_cortex_sim",
                "--budget-profile",
                "paper",
                "--benchmark-package",
                "package",
            ],
        ), patch("sys.stderr", stderr), patch(
            "spider_cortex_sim.cli.SpiderSimulation"
        ) as simulation_mock:
            with self.assertRaises(SystemExit) as ctx:
                main()

        self.assertEqual(ctx.exception.code, 2)
        self.assertIn("--checkpoint-selection best", stderr.getvalue())
        simulation_mock.assert_not_called()

    def test_benchmark_package_invokes_assembler_and_prints_summary(self) -> None:
        sim = MagicMock()
        sim.train.return_value = (self._minimal_summary(), [])
        manifest = MagicMock()
        manifest.package_version = "1.0"
        manifest.created_at = "2026-04-15T00:00:00+00:00"
        manifest.seed_count = 2
        manifest.confidence_level = 0.95
        manifest.contents = [{"path": "benchmark_manifest.json"}]
        manifest.limitations = []

        with tempfile.TemporaryDirectory() as tmpdir:
            stdout = io.StringIO()
            package_dir = Path(tmpdir) / "package"
            with patch(
                "sys.argv",
                [
                    "spider_cortex_sim",
                    "--budget-profile",
                    "paper",
                    "--checkpoint-selection",
                    "best",
                    "--benchmark-package",
                    str(package_dir),
                    "--full-summary",
                ],
            ), patch("sys.stdout", stdout), patch(
                "sys.stderr", io.StringIO()
            ), patch(
                "spider_cortex_sim.cli.SpiderSimulation",
                return_value=sim,
            ), patch(
                "spider_cortex_sim.cli.assemble_benchmark_package",
                return_value=manifest,
            ) as assemble_mock:
                main()

        assemble_mock.assert_called_once()
        payload = json.loads(stdout.getvalue())
        self.assertIn("benchmark_package", payload)
        self.assertEqual(payload["benchmark_package"]["package_version"], "1.0")
        self.assertEqual(payload["benchmark_package"]["seed_count"], 2)

    def test_benchmark_package_happy_path_writes_manifest_structure(self) -> None:
        sim = MagicMock()
        sim.train.return_value = (self._minimal_summary(), [])

        with tempfile.TemporaryDirectory() as tmpdir:
            stdout = io.StringIO()
            package_dir = Path(tmpdir) / "package"
            with patch(
                "sys.argv",
                [
                    "spider_cortex_sim",
                    "--budget-profile",
                    "paper",
                    "--checkpoint-selection",
                    "best",
                    "--benchmark-package",
                    str(package_dir),
                    "--full-summary",
                ],
            ), patch("sys.stdout", stdout), patch(
                "sys.stderr",
                io.StringIO(),
            ), patch(
                "spider_cortex_sim.cli.SpiderSimulation",
                return_value=sim,
            ):
                main()

            manifest_path = package_dir / "benchmark_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload = json.loads(stdout.getvalue())

        self.assertEqual(manifest["package_version"], "1.0")
        self.assertEqual(manifest["budget_profile"]["profile"], "paper")
        self.assertEqual(manifest["checkpoint_selection"]["selection"], "best")
        self.assertEqual(manifest["seed_count"], 2)
        content_paths = {item["path"] for item in manifest["contents"]}
        self.assertIn("aggregate_benchmark_tables.json", content_paths)
        self.assertIn("seed_level_rows.csv", content_paths)
        self.assertIn("report.md", content_paths)
        self.assertIn("benchmark_package", payload)
        self.assertEqual(payload["benchmark_package"]["seed_count"], 2)


if __name__ == "__main__":
    unittest.main()
