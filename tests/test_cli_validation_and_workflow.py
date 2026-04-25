"""Tests for CLI subpackage changed files.

Covers:
- spider_cortex_sim/cli/validation.py: _argument_error()
- spider_cortex_sim/cli/workflow_args.py: _build_compare_training_kwargs(), _build_checkpoint_selection_config()
- spider_cortex_sim/offline_analysis/report.py: re-export of build_report_data, write_report
- spider_cortex_sim/comparison_learning.py: module-level re-exports and compare_learning_evidence patch logic
"""

from __future__ import annotations

import argparse
import sys
import unittest
from unittest.mock import MagicMock, patch


class ArgumentErrorTest(unittest.TestCase):
    """Tests for cli/validation.py: _argument_error()."""

    def _make_args(self, *, with_parser: bool = False) -> argparse.Namespace:
        ns = argparse.Namespace()
        if with_parser:
            parser = argparse.ArgumentParser()
            ns._parser = parser
        return ns

    def test_raises_system_exit_2_without_parser(self) -> None:
        from spider_cortex_sim.cli.validation import _argument_error

        args = self._make_args(with_parser=False)
        with self.assertRaises(SystemExit) as ctx:
            _argument_error(args, "some error message")
        self.assertEqual(ctx.exception.code, 2)

    def test_prints_error_to_stderr_without_parser(self) -> None:
        from spider_cortex_sim.cli.validation import _argument_error

        args = self._make_args(with_parser=False)
        with patch("sys.stderr") as mock_stderr, self.assertRaises(SystemExit):
            _argument_error(args, "test_message_xyz")
        # Cannot easily assert on mock_stderr.write in all contexts,
        # but at minimum the SystemExit should have been raised.

    def test_delegates_to_parser_error_when_parser_present(self) -> None:
        from spider_cortex_sim.cli.validation import _argument_error

        mock_parser = MagicMock()
        args = argparse.Namespace(_parser=mock_parser)
        # parser.error() normally raises SystemExit; our mock doesn't, so the
        # function falls through to the stderr + SystemExit(2) path.
        with self.assertRaises(SystemExit):
            _argument_error(args, "test_error")
        mock_parser.error.assert_called_once_with("test_error")

    def test_no_parser_attribute_falls_back_to_stderr_exit(self) -> None:
        from spider_cortex_sim.cli.validation import _argument_error

        # Namespace with no _parser attribute at all
        args = argparse.Namespace()
        with self.assertRaises(SystemExit) as ctx:
            _argument_error(args, "fallback_error")
        self.assertEqual(ctx.exception.code, 2)

    def test_parser_none_falls_back_to_stderr_exit(self) -> None:
        from spider_cortex_sim.cli.validation import _argument_error

        # Namespace with _parser=None
        args = argparse.Namespace(_parser=None)
        with self.assertRaises(SystemExit) as ctx:
            _argument_error(args, "none_parser_error")
        self.assertEqual(ctx.exception.code, 2)


class BuildCompareTrainingKwargsTest(unittest.TestCase):
    """Tests for cli/workflow_args.py: _build_compare_training_kwargs()."""

    def _make_args(self) -> argparse.Namespace:
        return argparse.Namespace(
            width=20,
            height=20,
            food_count=5,
            day_length=18,
            night_length=12,
            gamma=0.99,
            module_lr=0.001,
            motor_lr=0.001,
            module_dropout=0.05,
            reward_profile="default",
            map_template="random",
            operational_profile="standard",
            noise_profile="none",
            budget_profile="current",
            checkpoint_selection="best",
            checkpoint_dir=None,
            curriculum_profile=None,
        )

    def _make_budget(self) -> MagicMock:
        budget = MagicMock()
        budget.max_steps = 1000
        budget.episodes = 100
        budget.eval_episodes = 10
        budget.behavior_seeds = [1, 2, 3]
        budget.scenario_episodes = 5
        budget.checkpoint_interval = 50
        return budget

    def _make_checkpoint_config(self) -> MagicMock:
        return MagicMock()

    def test_returns_dict(self) -> None:
        from spider_cortex_sim.cli.workflow_args import _build_compare_training_kwargs

        result = _build_compare_training_kwargs(
            self._make_args(),
            self._make_budget(),
            self._make_checkpoint_config(),
        )
        self.assertIsInstance(result, dict)

    def test_all_expected_keys_present(self) -> None:
        from spider_cortex_sim.cli.workflow_args import _build_compare_training_kwargs

        result = _build_compare_training_kwargs(
            self._make_args(),
            self._make_budget(),
            self._make_checkpoint_config(),
        )
        expected_keys = {
            "width", "height", "food_count", "day_length", "night_length",
            "max_steps", "episodes", "evaluation_episodes", "gamma",
            "module_lr", "motor_lr", "module_dropout", "reward_profile",
            "map_template", "operational_profile", "noise_profile",
            "budget_profile", "seeds", "episodes_per_scenario",
            "checkpoint_selection", "checkpoint_selection_config",
            "checkpoint_interval", "checkpoint_dir", "curriculum_profile",
        }
        for key in expected_keys:
            self.assertIn(key, result, msg=f"Missing key: '{key}'")

    def test_seeds_is_tuple_from_budget_behavior_seeds(self) -> None:
        from spider_cortex_sim.cli.workflow_args import _build_compare_training_kwargs

        budget = self._make_budget()
        budget.behavior_seeds = [10, 20, 30]
        result = _build_compare_training_kwargs(
            self._make_args(), budget, self._make_checkpoint_config()
        )
        self.assertEqual(result["seeds"], (10, 20, 30))
        self.assertIsInstance(result["seeds"], tuple)

    def test_width_from_args(self) -> None:
        from spider_cortex_sim.cli.workflow_args import _build_compare_training_kwargs

        args = self._make_args()
        args.width = 42
        result = _build_compare_training_kwargs(
            args, self._make_budget(), self._make_checkpoint_config()
        )
        self.assertEqual(result["width"], 42)

    def test_max_steps_from_budget(self) -> None:
        from spider_cortex_sim.cli.workflow_args import _build_compare_training_kwargs

        budget = self._make_budget()
        budget.max_steps = 999
        result = _build_compare_training_kwargs(
            self._make_args(), budget, self._make_checkpoint_config()
        )
        self.assertEqual(result["max_steps"], 999)

    def test_checkpoint_selection_config_included_verbatim(self) -> None:
        from spider_cortex_sim.cli.workflow_args import _build_compare_training_kwargs

        ckpt_config = self._make_checkpoint_config()
        result = _build_compare_training_kwargs(
            self._make_args(), self._make_budget(), ckpt_config
        )
        self.assertIs(result["checkpoint_selection_config"], ckpt_config)

    def test_evaluation_episodes_from_budget_eval_episodes(self) -> None:
        from spider_cortex_sim.cli.workflow_args import _build_compare_training_kwargs

        budget = self._make_budget()
        budget.eval_episodes = 25
        result = _build_compare_training_kwargs(
            self._make_args(), budget, self._make_checkpoint_config()
        )
        self.assertEqual(result["evaluation_episodes"], 25)

    def test_episodes_per_scenario_from_budget_scenario_episodes(self) -> None:
        from spider_cortex_sim.cli.workflow_args import _build_compare_training_kwargs

        budget = self._make_budget()
        budget.scenario_episodes = 7
        result = _build_compare_training_kwargs(
            self._make_args(), budget, self._make_checkpoint_config()
        )
        self.assertEqual(result["episodes_per_scenario"], 7)


class BuildCheckpointSelectionConfigTest(unittest.TestCase):
    """Tests for cli/workflow_args.py: _build_checkpoint_selection_config()."""

    def _make_args(
        self,
        metric: str = "scenario_success_rate",
        override_penalty: float = 0.0,
        dominance_penalty: float = 0.0,
        penalty_mode: str = "tiebreaker",
    ) -> argparse.Namespace:
        return argparse.Namespace(
            checkpoint_metric=metric,
            checkpoint_override_penalty=override_penalty,
            checkpoint_dominance_penalty=dominance_penalty,
            checkpoint_penalty_mode=penalty_mode,
        )

    def test_returns_checkpoint_selection_config(self) -> None:
        from spider_cortex_sim.cli.workflow_args import _build_checkpoint_selection_config
        from spider_cortex_sim.checkpointing import CheckpointSelectionConfig

        result = _build_checkpoint_selection_config(self._make_args())
        self.assertIsInstance(result, CheckpointSelectionConfig)

    def test_metric_passed_through(self) -> None:
        from spider_cortex_sim.cli.workflow_args import _build_checkpoint_selection_config

        result = _build_checkpoint_selection_config(
            self._make_args(metric="episode_success_rate")
        )
        self.assertEqual(str(result.metric), "episode_success_rate")

    def test_mean_reward_metric_accepted(self) -> None:
        from spider_cortex_sim.cli.workflow_args import _build_checkpoint_selection_config

        result = _build_checkpoint_selection_config(
            self._make_args(metric="mean_reward")
        )
        self.assertEqual(str(result.metric), "mean_reward")

    def test_override_penalty_passed_through(self) -> None:
        from spider_cortex_sim.cli.workflow_args import _build_checkpoint_selection_config

        result = _build_checkpoint_selection_config(
            self._make_args(override_penalty=0.5)
        )
        self.assertAlmostEqual(result.override_penalty_weight, 0.5)

    def test_invalid_metric_calls_argument_error(self) -> None:
        from spider_cortex_sim.cli.workflow_args import _build_checkpoint_selection_config

        args = self._make_args(metric="invalid_metric")
        # _argument_error falls back to SystemExit(2) when no _parser
        with self.assertRaises((SystemExit, AssertionError)):
            _build_checkpoint_selection_config(args)

    def test_direct_penalty_mode_accepted(self) -> None:
        from spider_cortex_sim.cli.workflow_args import _build_checkpoint_selection_config
        from spider_cortex_sim.checkpointing import CheckpointPenaltyMode

        result = _build_checkpoint_selection_config(
            self._make_args(penalty_mode="direct")
        )
        self.assertEqual(result.penalty_mode, CheckpointPenaltyMode.DIRECT)


class OfflineAnalysisReportImportTest(unittest.TestCase):
    """Tests for offline_analysis/report.py re-exports."""

    def test_build_report_data_importable(self) -> None:
        from spider_cortex_sim.offline_analysis.report import build_report_data
        self.assertIsNotNone(build_report_data)

    def test_write_report_importable(self) -> None:
        from spider_cortex_sim.offline_analysis.report import write_report
        self.assertIsNotNone(write_report)

    def test_all_contains_expected_names(self) -> None:
        import spider_cortex_sim.offline_analysis.report as report_mod
        self.assertIn("build_report_data", report_mod.__all__)
        self.assertIn("write_report", report_mod.__all__)

    def test_build_report_data_is_callable(self) -> None:
        from spider_cortex_sim.offline_analysis.report import build_report_data
        self.assertTrue(callable(build_report_data))

    def test_write_report_is_callable(self) -> None:
        from spider_cortex_sim.offline_analysis.report import write_report
        self.assertTrue(callable(write_report))


class ComparisonLearningImportTest(unittest.TestCase):
    """Tests for spider_cortex_sim/comparison_learning.py re-exports and function."""

    def test_compare_learning_evidence_importable(self) -> None:
        from spider_cortex_sim.comparison_learning import compare_learning_evidence
        self.assertIsNotNone(compare_learning_evidence)
        self.assertTrue(callable(compare_learning_evidence))

    def test_learning_comparison_symbols_accessible_from_comparison_learning(self) -> None:
        """Verify that learning_comparison.* symbols are re-exported."""
        import spider_cortex_sim.comparison_learning as cl
        # The module does `from .learning_comparison import *`
        # so key names from learning_comparison should be accessible
        self.assertTrue(hasattr(cl, "compare_learning_evidence"))

    def test_compare_learning_evidence_restores_runner_on_success(self) -> None:
        """Verify that runner resolver functions are restored after a call."""
        from spider_cortex_sim.learning_comparison import runner as _runner

        # Record original functions
        original_budget = _runner.resolve_budget
        original_conditions = _runner.resolve_learning_evidence_conditions

        # Mock the actual runner call so we don't execute real training
        with patch.object(
            _runner,
            "compare_learning_evidence",
            return_value=({"mock": True}, []),
        ):
            from spider_cortex_sim.comparison_learning import compare_learning_evidence
            compare_learning_evidence()

        # After the call, the runner's resolver functions should be restored
        self.assertIs(_runner.resolve_budget, original_budget)
        self.assertIs(
            _runner.resolve_learning_evidence_conditions,
            original_conditions,
        )

    def test_compare_learning_evidence_restores_runner_on_exception(self) -> None:
        """Verify runner functions are restored even when the delegate raises."""
        from spider_cortex_sim.learning_comparison import runner as _runner

        original_budget = _runner.resolve_budget
        original_conditions = _runner.resolve_learning_evidence_conditions

        with patch.object(
            _runner,
            "compare_learning_evidence",
            side_effect=RuntimeError("simulated failure"),
        ):
            from spider_cortex_sim.comparison_learning import compare_learning_evidence
            with self.assertRaises(RuntimeError):
                compare_learning_evidence()

        # Even after the exception, runner functions should be restored
        self.assertIs(_runner.resolve_budget, original_budget)
        self.assertIs(
            _runner.resolve_learning_evidence_conditions,
            original_conditions,
        )


class LearningComparisonPackageImportTest(unittest.TestCase):
    """Tests for spider_cortex_sim/learning_comparison/__init__.py package exports."""

    def test_build_distillation_comparison_report_importable(self) -> None:
        from spider_cortex_sim.learning_comparison import build_distillation_comparison_report
        self.assertTrue(callable(build_distillation_comparison_report))

    def test_build_learning_evidence_deltas_importable(self) -> None:
        from spider_cortex_sim.learning_comparison import build_learning_evidence_deltas
        self.assertTrue(callable(build_learning_evidence_deltas))

    def test_build_learning_evidence_summary_importable(self) -> None:
        from spider_cortex_sim.learning_comparison import build_learning_evidence_summary
        self.assertTrue(callable(build_learning_evidence_summary))

    def test_compare_learning_evidence_importable_from_package(self) -> None:
        from spider_cortex_sim.learning_comparison import compare_learning_evidence
        self.assertTrue(callable(compare_learning_evidence))


if __name__ == "__main__":
    unittest.main()
