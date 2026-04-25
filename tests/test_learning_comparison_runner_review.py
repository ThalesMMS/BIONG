from __future__ import annotations

import unittest
from unittest import mock

from spider_cortex_sim.learning_comparison.runner import compare_learning_evidence
from spider_cortex_sim.learning_evidence import LearningEvidenceConditionSpec


class LearningComparisonRunnerReviewRegressionTest(unittest.TestCase):
    def test_explicit_empty_names_raise_clear_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "scenario name"):
            compare_learning_evidence(
                episodes=0,
                evaluation_episodes=0,
                max_steps=1,
                names=[],
                seeds=(1,),
                condition_names=("trained_without_reflex_support", "random_init"),
                episodes_per_scenario=1,
                long_budget_profile="smoke",
            )

    def test_behavior_indices_start_after_resolved_training_total(self) -> None:
        condition = LearningEvidenceConditionSpec(
            name="late_finetuning_review",
            description="Review regression condition.",
            train_budget="base",
            training_regime="late_finetuning",
            eval_reflex_scale=0.0,
        )
        with mock.patch(
            "spider_cortex_sim.learning_comparison.runner.resolve_learning_evidence_conditions",
            return_value=[condition],
        ):
            payload, rows = compare_learning_evidence(
                episodes=2,
                evaluation_episodes=0,
                max_steps=1,
                names=("night_rest",),
                seeds=(1,),
                condition_names=("late_finetuning_review",),
                episodes_per_scenario=1,
                long_budget_profile="smoke",
            )

        train_total = payload["conditions"]["late_finetuning_review"][
            "train_episodes"
        ]
        self.assertEqual(train_total, 12)
        self.assertEqual(rows[0]["episode"], train_total + 1)

    def test_freeze_half_indices_start_after_full_base_budget(self) -> None:
        condition = LearningEvidenceConditionSpec(
            name="freeze_half_review",
            description="Review regression freeze condition.",
            train_budget="freeze_half",
            eval_reflex_scale=0.0,
        )
        with mock.patch(
            "spider_cortex_sim.learning_comparison.runner.resolve_learning_evidence_conditions",
            return_value=[condition],
        ):
            payload, rows = compare_learning_evidence(
                episodes=6,
                evaluation_episodes=0,
                max_steps=1,
                names=("night_rest",),
                seeds=(1,),
                condition_names=("freeze_half_review",),
                episodes_per_scenario=1,
                long_budget_profile="smoke",
            )

        self.assertEqual(
            payload["conditions"]["freeze_half_review"]["train_episodes"],
            3,
        )
        self.assertEqual(rows[0]["episode"], 7)


if __name__ == "__main__":
    unittest.main()
