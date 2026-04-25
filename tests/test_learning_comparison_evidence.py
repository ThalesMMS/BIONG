"""Tests for spider_cortex_sim.learning_comparison.evidence module.

Covers build_learning_evidence_deltas and build_learning_evidence_summary,
extracted into a focused submodule in the PR refactor.
"""

from __future__ import annotations

import unittest

from spider_cortex_sim.learning_comparison.evidence import (
    build_learning_evidence_deltas,
    build_learning_evidence_summary,
)


def _make_condition_payload(
    scenario_success_rate: float = 0.5,
    episode_success_rate: float = 0.5,
    mean_reward: float = 2.0,
    scenario_names: tuple[str, ...] = ("night_rest",),
    per_scenario_rate: float = 0.5,
) -> dict:
    """Build a minimal condition payload in compact format."""
    return {
        "summary": {
            "scenario_success_rate": scenario_success_rate,
            "episode_success_rate": episode_success_rate,
            "mean_reward": mean_reward,
        },
        "suite": {name: {"success_rate": per_scenario_rate} for name in scenario_names},
    }


class BuildLearningEvidenceDeltasBasicTest(unittest.TestCase):
    """Tests for build_learning_evidence_deltas basic behavior."""

    def test_returns_dict(self) -> None:
        conditions = {
            "trained_without_reflex_support": _make_condition_payload(0.7),
            "random_init": _make_condition_payload(0.3),
        }
        result = build_learning_evidence_deltas(
            conditions,
            reference_condition="trained_without_reflex_support",
            scenario_names=["night_rest"],
        )
        self.assertIsInstance(result, dict)

    def test_condition_keys_in_result(self) -> None:
        conditions = {
            "trained_without_reflex_support": _make_condition_payload(0.8),
            "random_init": _make_condition_payload(0.2),
        }
        result = build_learning_evidence_deltas(
            conditions,
            reference_condition="trained_without_reflex_support",
            scenario_names=["night_rest"],
        )
        self.assertIn("trained_without_reflex_support", result)
        self.assertIn("random_init", result)

    def test_delta_summary_keys_present(self) -> None:
        conditions = {
            "trained_without_reflex_support": _make_condition_payload(0.8),
            "random_init": _make_condition_payload(0.3),
        }
        result = build_learning_evidence_deltas(
            conditions,
            reference_condition="trained_without_reflex_support",
            scenario_names=["night_rest"],
        )
        delta_payload = result["random_init"]
        self.assertIn("summary", delta_payload)
        summary_keys = delta_payload["summary"]
        self.assertIn("scenario_success_rate_delta", summary_keys)
        self.assertIn("episode_success_rate_delta", summary_keys)
        self.assertIn("mean_reward_delta", summary_keys)

    def test_delta_is_correct_for_scenario_success_rate(self) -> None:
        conditions = {
            "trained_without_reflex_support": _make_condition_payload(0.8),
            "random_init": _make_condition_payload(0.3),
        }
        result = build_learning_evidence_deltas(
            conditions,
            reference_condition="trained_without_reflex_support",
            scenario_names=["night_rest"],
        )
        delta = result["random_init"]["summary"]["scenario_success_rate_delta"]
        # 0.3 - 0.8 = -0.5
        self.assertAlmostEqual(delta, -0.5, places=5)

    def test_reference_condition_delta_is_zero_for_self(self) -> None:
        conditions = {
            "trained_without_reflex_support": _make_condition_payload(0.7),
        }
        result = build_learning_evidence_deltas(
            conditions,
            reference_condition="trained_without_reflex_support",
            scenario_names=["night_rest"],
        )
        delta = result["trained_without_reflex_support"]["summary"][
            "scenario_success_rate_delta"
        ]
        self.assertAlmostEqual(delta, 0.0, places=10)

    def test_deltas_rounded_to_6_decimal_places(self) -> None:
        conditions = {
            "trained_without_reflex_support": _make_condition_payload(1 / 3),
            "random_init": _make_condition_payload(1 / 7),
        }
        result = build_learning_evidence_deltas(
            conditions,
            reference_condition="trained_without_reflex_support",
            scenario_names=["night_rest"],
        )
        delta = result["random_init"]["summary"]["scenario_success_rate_delta"]
        self.assertAlmostEqual(delta, round(delta, 6), places=10)

    def test_skipped_conditions_get_skipped_payload(self) -> None:
        conditions = {
            "trained_without_reflex_support": _make_condition_payload(0.7),
            "skipped_condition": {"skipped": True, "reason": "not applicable"},
        }
        result = build_learning_evidence_deltas(
            conditions,
            reference_condition="trained_without_reflex_support",
            scenario_names=["night_rest"],
        )
        skipped = result["skipped_condition"]
        self.assertTrue(skipped.get("skipped"))

    def test_reference_missing_marks_all_as_skipped(self) -> None:
        conditions = {
            "random_init": _make_condition_payload(0.3),
            "reflex_only": _make_condition_payload(0.4),
        }
        result = build_learning_evidence_deltas(
            conditions,
            reference_condition="trained_without_reflex_support",
            scenario_names=["night_rest"],
        )
        for payload in result.values():
            self.assertTrue(payload.get("skipped"), f"Expected skipped: {payload}")

    def test_per_scenario_delta_computed(self) -> None:
        conditions = {
            "trained_without_reflex_support": _make_condition_payload(
                0.8, per_scenario_rate=0.8
            ),
            "random_init": _make_condition_payload(0.3, per_scenario_rate=0.3),
        }
        result = build_learning_evidence_deltas(
            conditions,
            reference_condition="trained_without_reflex_support",
            scenario_names=["night_rest"],
        )
        scenario_delta = result["random_init"]["scenarios"]["night_rest"][
            "success_rate_delta"
        ]
        self.assertAlmostEqual(scenario_delta, -0.5, places=5)


class BuildLearningEvidenceSummaryBasicTest(unittest.TestCase):
    """Tests for build_learning_evidence_summary basic behavior."""

    def _make_conditions(
        self,
        *,
        reference_rate: float = 0.7,
        random_rate: float = 0.2,
        reflex_rate: float = 0.4,
    ) -> dict:
        return {
            "trained_without_reflex_support": _make_condition_payload(reference_rate),
            "random_init": _make_condition_payload(random_rate),
            "reflex_only": _make_condition_payload(reflex_rate),
        }

    def test_returns_dict(self) -> None:
        conditions = self._make_conditions()
        result = build_learning_evidence_summary(
            conditions,
            reference_condition="trained_without_reflex_support",
        )
        self.assertIsInstance(result, dict)

    def test_contains_required_top_level_keys(self) -> None:
        conditions = self._make_conditions()
        result = build_learning_evidence_summary(
            conditions,
            reference_condition="trained_without_reflex_support",
        )
        required_keys = [
            "reference_condition",
            "primary_gate_metric",
            "supports_primary_evidence",
            "has_learning_evidence",
            "primary_condition",
            "trained_final",
            "random_init",
            "reflex_only",
            "trained_vs_random_init",
            "trained_vs_reflex_only",
            "seed_level",
            "uncertainty",
            "notes",
        ]
        for key in required_keys:
            self.assertIn(key, result, f"Missing key: {key}")

    def test_primary_gate_metric_is_scenario_success_rate(self) -> None:
        conditions = self._make_conditions()
        result = build_learning_evidence_summary(
            conditions,
            reference_condition="trained_without_reflex_support",
        )
        self.assertEqual(result["primary_gate_metric"], "scenario_success_rate")

    def test_has_learning_evidence_true_when_reference_exceeds_both(self) -> None:
        conditions = self._make_conditions(
            reference_rate=0.8, random_rate=0.2, reflex_rate=0.4
        )
        result = build_learning_evidence_summary(
            conditions,
            reference_condition="trained_without_reflex_support",
        )
        self.assertTrue(result["has_learning_evidence"])
        self.assertTrue(result["supports_primary_evidence"])

    def test_has_learning_evidence_false_when_reference_below_random(self) -> None:
        conditions = self._make_conditions(
            reference_rate=0.1, random_rate=0.5, reflex_rate=0.2
        )
        result = build_learning_evidence_summary(
            conditions,
            reference_condition="trained_without_reflex_support",
        )
        self.assertFalse(result["has_learning_evidence"])

    def test_has_learning_evidence_false_when_reference_below_reflex(self) -> None:
        conditions = self._make_conditions(
            reference_rate=0.3, random_rate=0.1, reflex_rate=0.5
        )
        result = build_learning_evidence_summary(
            conditions,
            reference_condition="trained_without_reflex_support",
        )
        self.assertFalse(result["has_learning_evidence"])

    def test_trained_vs_random_init_contains_correct_keys(self) -> None:
        conditions = self._make_conditions()
        result = build_learning_evidence_summary(
            conditions,
            reference_condition="trained_without_reflex_support",
        )
        vs_random = result["trained_vs_random_init"]
        self.assertIn("scenario_success_rate_delta", vs_random)
        self.assertIn("episode_success_rate_delta", vs_random)
        self.assertIn("mean_reward_delta", vs_random)

    def test_trained_vs_random_delta_is_reference_minus_comparator(self) -> None:
        conditions = self._make_conditions(
            reference_rate=0.8, random_rate=0.3
        )
        result = build_learning_evidence_summary(
            conditions,
            reference_condition="trained_without_reflex_support",
        )
        # reference - random = 0.8 - 0.3 = 0.5
        delta = result["trained_vs_random_init"]["scenario_success_rate_delta"]
        self.assertAlmostEqual(delta, 0.5, places=5)

    def test_supports_primary_evidence_false_when_reference_missing(self) -> None:
        conditions = {
            "random_init": _make_condition_payload(0.3),
            "reflex_only": _make_condition_payload(0.4),
        }
        result = build_learning_evidence_summary(
            conditions,
            reference_condition="trained_without_reflex_support",
        )
        self.assertFalse(result["supports_primary_evidence"])
        self.assertFalse(result["has_learning_evidence"])

    def test_notes_is_list_of_strings(self) -> None:
        conditions = self._make_conditions()
        result = build_learning_evidence_summary(
            conditions,
            reference_condition="trained_without_reflex_support",
        )
        notes = result["notes"]
        self.assertIsInstance(notes, list)
        for note in notes:
            self.assertIsInstance(note, str)

    def test_reference_condition_field_matches_input(self) -> None:
        conditions = self._make_conditions()
        result = build_learning_evidence_summary(
            conditions,
            reference_condition="trained_without_reflex_support",
        )
        self.assertEqual(
            result["reference_condition"], "trained_without_reflex_support"
        )

    def test_uncertainty_contains_conditions_key(self) -> None:
        conditions = self._make_conditions()
        result = build_learning_evidence_summary(
            conditions,
            reference_condition="trained_without_reflex_support",
        )
        self.assertIn("conditions", result["uncertainty"])

    def test_missing_reflex_only_adds_note(self) -> None:
        """When reflex_only is not in conditions, a note is added."""
        conditions = {
            "trained_without_reflex_support": _make_condition_payload(0.7),
            "random_init": _make_condition_payload(0.2),
        }
        result = build_learning_evidence_summary(
            conditions,
            reference_condition="trained_without_reflex_support",
        )
        self.assertFalse(result["has_learning_evidence"])
        # A note about reflex_only unavailability should be added
        note_text = " ".join(result["notes"])
        self.assertIn("reflex_only", note_text)


if __name__ == "__main__":
    unittest.main()