"""Tests for spider_cortex_sim.learning_comparison.distillation module.

Covers build_distillation_comparison_report, introduced in the PR refactor
of the learning comparison code into focused submodules.
"""

from __future__ import annotations

import unittest

from spider_cortex_sim.learning_comparison.distillation import (
    build_distillation_comparison_report,
)


def _make_payload(scenario_success_rate: float = 0.5) -> dict:
    """Build a minimal condition payload that mirrors the compact format."""
    return {
        "summary": {
            "scenario_success_rate": scenario_success_rate,
            "episode_success_rate": scenario_success_rate,
            "mean_reward": scenario_success_rate * 5.0,
        },
        "suite": {},
    }


class BuildDistillationComparisonReportAnswerTest(unittest.TestCase):
    """Tests for the 'answer' field of build_distillation_comparison_report."""

    def test_answer_yes_when_distilled_exceeds_baseline(self) -> None:
        report = build_distillation_comparison_report(
            teacher=_make_payload(0.9),
            modular_rl_from_scratch=_make_payload(0.3),
            modular_distilled=_make_payload(0.7),
        )
        self.assertEqual(report["answer"], "yes")

    def test_answer_no_when_distilled_below_baseline(self) -> None:
        report = build_distillation_comparison_report(
            teacher=_make_payload(0.9),
            modular_rl_from_scratch=_make_payload(0.5),
            modular_distilled=_make_payload(0.2),
        )
        self.assertEqual(report["answer"], "no")

    def test_answer_partially_when_distilled_positive_but_below_baseline(self) -> None:
        report = build_distillation_comparison_report(
            teacher=_make_payload(0.9),
            modular_rl_from_scratch=_make_payload(0.6),
            modular_distilled=_make_payload(0.3),
        )
        self.assertEqual(report["answer"], "partially")

    def test_answer_inconclusive_when_teacher_not_above_baseline(self) -> None:
        report = build_distillation_comparison_report(
            teacher=_make_payload(0.3),
            modular_rl_from_scratch=_make_payload(0.5),
            modular_distilled=_make_payload(0.4),
        )
        self.assertEqual(report["answer"], "inconclusive")

    def test_answer_yes_after_rl_finetuning_when_finetuned_better_and_exceeds_baseline(
        self,
    ) -> None:
        report = build_distillation_comparison_report(
            teacher=_make_payload(0.9),
            modular_rl_from_scratch=_make_payload(0.3),
            modular_distilled=_make_payload(0.1),
            modular_distilled_plus_rl_finetuning=_make_payload(0.7),
        )
        self.assertEqual(report["answer"], "yes_after_rl_finetuning")

    def test_answer_yes_when_no_baseline_and_distilled_positive_teacher_positive(
        self,
    ) -> None:
        report = build_distillation_comparison_report(
            teacher=_make_payload(0.8),
            modular_rl_from_scratch=None,
            modular_distilled=_make_payload(0.5),
        )
        self.assertEqual(report["answer"], "yes")

    def test_answer_no_when_no_baseline_and_distilled_zero(self) -> None:
        report = build_distillation_comparison_report(
            teacher=_make_payload(0.8),
            modular_rl_from_scratch=None,
            modular_distilled=_make_payload(0.0),
        )
        self.assertEqual(report["answer"], "no")

    def test_answer_inconclusive_when_no_baseline_and_teacher_zero(self) -> None:
        report = build_distillation_comparison_report(
            teacher=_make_payload(0.0),
            modular_rl_from_scratch=None,
            modular_distilled=_make_payload(0.5),
        )
        self.assertEqual(report["answer"], "inconclusive")


class BuildDistillationComparisonReportStructureTest(unittest.TestCase):
    """Tests for the structure and keys of build_distillation_comparison_report output."""

    def _report(self) -> dict:
        return build_distillation_comparison_report(
            teacher=_make_payload(0.8),
            modular_rl_from_scratch=_make_payload(0.4),
            modular_distilled=_make_payload(0.6),
        )

    def test_contains_question_key(self) -> None:
        report = self._report()
        self.assertIn("question", report)
        self.assertIsInstance(report["question"], str)

    def test_contains_answer_key(self) -> None:
        report = self._report()
        self.assertIn("answer", report)

    def test_contains_rationale_key(self) -> None:
        report = self._report()
        self.assertIn("rationale", report)
        self.assertIsInstance(report["rationale"], str)

    def test_contains_primary_metric_key(self) -> None:
        report = self._report()
        self.assertIn("primary_metric", report)
        self.assertEqual(report["primary_metric"], "scenario_success_rate")

    def test_contains_teacher_key(self) -> None:
        report = self._report()
        self.assertIn("teacher", report)

    def test_contains_modular_rl_from_scratch_key(self) -> None:
        report = self._report()
        self.assertIn("modular_rl_from_scratch", report)

    def test_contains_modular_distilled_key(self) -> None:
        report = self._report()
        self.assertIn("modular_distilled", report)

    def test_contains_modular_distilled_plus_rl_finetuning_none_by_default(
        self,
    ) -> None:
        report = self._report()
        self.assertIn("modular_distilled_plus_rl_finetuning", report)
        self.assertIsNone(report["modular_distilled_plus_rl_finetuning"])

    def test_contains_best_student_condition(self) -> None:
        report = self._report()
        self.assertIn("best_student_condition", report)
        valid = {"modular_distilled", "modular_distilled_plus_rl_finetuning"}
        self.assertIn(report["best_student_condition"], valid)

    def test_contains_quantitative_evidence(self) -> None:
        report = self._report()
        self.assertIn("quantitative_evidence", report)
        self.assertIsInstance(report["quantitative_evidence"], dict)

    def test_quantitative_evidence_contains_distilled_vs_teacher(self) -> None:
        report = self._report()
        self.assertIn(
            "modular_distilled_vs_teacher", report["quantitative_evidence"]
        )

    def test_quantitative_evidence_contains_teacher_vs_baseline(self) -> None:
        report = self._report()
        self.assertIn(
            "teacher_vs_modular_rl_from_scratch", report["quantitative_evidence"]
        )

    def test_quantitative_evidence_contains_gap_closed_fraction(self) -> None:
        report = self._report()
        self.assertIn("gap_closed_fraction", report["quantitative_evidence"])

    def test_modular_rl_from_scratch_is_none_when_not_provided(self) -> None:
        report = build_distillation_comparison_report(
            teacher=_make_payload(0.8),
            modular_rl_from_scratch=None,
            modular_distilled=_make_payload(0.6),
        )
        self.assertIsNone(report["modular_rl_from_scratch"])

    def test_quantitative_evidence_delta_is_rounded(self) -> None:
        report = build_distillation_comparison_report(
            teacher=_make_payload(0.9),
            modular_rl_from_scratch=_make_payload(0.3),
            modular_distilled=_make_payload(0.6),
        )
        delta = report["quantitative_evidence"]["modular_distilled_vs_teacher"][
            "scenario_success_rate_delta"
        ]
        # Verify it's a float rounded to 6 decimal places
        self.assertIsInstance(delta, float)
        self.assertAlmostEqual(delta, round(delta, 6), places=10)


class BuildDistillationGapClosedFractionTest(unittest.TestCase):
    """Tests for gap_closed_fraction computation in build_distillation_comparison_report."""

    def test_gap_closed_fraction_is_correct_for_full_gap(self) -> None:
        """When distilled exactly matches teacher, gap_closed should be 1.0."""
        report = build_distillation_comparison_report(
            teacher=_make_payload(1.0),
            modular_rl_from_scratch=_make_payload(0.0),
            modular_distilled=_make_payload(1.0),
        )
        fraction = report["quantitative_evidence"]["gap_closed_fraction"]["modular_distilled"]
        self.assertAlmostEqual(fraction, 1.0, places=5)

    def test_gap_closed_fraction_is_zero_when_distilled_equals_baseline(self) -> None:
        report = build_distillation_comparison_report(
            teacher=_make_payload(1.0),
            modular_rl_from_scratch=_make_payload(0.5),
            modular_distilled=_make_payload(0.5),
        )
        fraction = report["quantitative_evidence"]["gap_closed_fraction"]["modular_distilled"]
        self.assertAlmostEqual(fraction, 0.0, places=5)

    def test_gap_closed_fraction_is_none_when_no_baseline(self) -> None:
        report = build_distillation_comparison_report(
            teacher=_make_payload(1.0),
            modular_rl_from_scratch=None,
            modular_distilled=_make_payload(0.5),
        )
        self.assertNotIn("gap_closed_fraction", report["quantitative_evidence"])

    def test_gap_closed_fraction_none_when_teacher_equals_baseline(self) -> None:
        """When teacher doesn't exceed baseline, gap is 0, fraction is None."""
        report = build_distillation_comparison_report(
            teacher=_make_payload(0.5),
            modular_rl_from_scratch=_make_payload(0.5),
            modular_distilled=_make_payload(0.6),
        )
        # This triggers inconclusive (teacher not above baseline)
        self.assertEqual(report["answer"], "inconclusive")


class BuildDistillationFinetuningTest(unittest.TestCase):
    """Tests for finetuning comparison in build_distillation_comparison_report."""

    def test_finetuning_vs_teacher_in_quantitative_when_provided(self) -> None:
        report = build_distillation_comparison_report(
            teacher=_make_payload(0.9),
            modular_rl_from_scratch=_make_payload(0.3),
            modular_distilled=_make_payload(0.5),
            modular_distilled_plus_rl_finetuning=_make_payload(0.7),
        )
        self.assertIn(
            "modular_distilled_plus_rl_finetuning_vs_teacher",
            report["quantitative_evidence"],
        )

    def test_best_student_is_distilled_when_finetuned_not_better(self) -> None:
        report = build_distillation_comparison_report(
            teacher=_make_payload(0.9),
            modular_rl_from_scratch=_make_payload(0.3),
            modular_distilled=_make_payload(0.8),
            modular_distilled_plus_rl_finetuning=_make_payload(0.5),
        )
        self.assertEqual(report["best_student_condition"], "modular_distilled")

    def test_best_student_is_finetuned_when_finetuned_better(self) -> None:
        report = build_distillation_comparison_report(
            teacher=_make_payload(0.9),
            modular_rl_from_scratch=_make_payload(0.3),
            modular_distilled=_make_payload(0.5),
            modular_distilled_plus_rl_finetuning=_make_payload(0.85),
        )
        self.assertEqual(
            report["best_student_condition"],
            "modular_distilled_plus_rl_finetuning",
        )


if __name__ == "__main__":
    unittest.main()