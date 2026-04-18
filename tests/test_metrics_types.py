"""Focused metrics and behavior-evaluation tests."""

from __future__ import annotations

import unittest
from dataclasses import asdict

from spider_cortex_sim.metrics import (
    BehaviorCheckResult,
    BehaviorCheckSpec,
    BehavioralEpisodeScore,
    EpisodeStats,
    PREDATOR_TYPE_NAMES,
    competence_label_from_eval_reflex_scale,
)
from tests.fixtures.shared_builders import (
    make_behavior_check_result,
    make_behavioral_episode_score,
    make_episode_stats,
)


class BehaviorCheckSpecTest(unittest.TestCase):
    """Tests for the BehaviorCheckSpec frozen dataclass."""

    def test_creation_stores_fields(self) -> None:
        spec = BehaviorCheckSpec("my_check", "some description", ">= 0.5")
        self.assertEqual(spec.name, "my_check")
        self.assertEqual(spec.description, "some description")
        self.assertEqual(spec.expected, ">= 0.5")

    def test_frozen_raises_on_assignment(self) -> None:
        spec = BehaviorCheckSpec("check", "desc", "val")
        with self.assertRaises((AttributeError, TypeError)):
            spec.name = "new_name"  # type: ignore[misc]

    def test_equality_with_same_values(self) -> None:
        a = BehaviorCheckSpec("x", "d", "e")
        b = BehaviorCheckSpec("x", "d", "e")
        self.assertEqual(a, b)

    def test_inequality_with_different_values(self) -> None:
        a = BehaviorCheckSpec("x", "d", "e1")
        b = BehaviorCheckSpec("x", "d", "e2")
        self.assertNotEqual(a, b)


class BehaviorCheckResultTest(unittest.TestCase):
    """Tests for the BehaviorCheckResult frozen dataclass."""

    def test_creation_stores_all_fields(self) -> None:
        result = BehaviorCheckResult("chk", "description", ">= 1", True, 1.5)
        self.assertEqual(result.name, "chk")
        self.assertEqual(result.description, "description")
        self.assertEqual(result.expected, ">= 1")
        self.assertTrue(result.passed)
        self.assertEqual(result.value, 1.5)

    def test_passed_false_is_stored(self) -> None:
        result = BehaviorCheckResult("chk", "desc", "true", False, False)
        self.assertFalse(result.passed)

    def test_value_can_be_any_type(self) -> None:
        for value in [0, 1.5, True, "string", [1, 2], {"a": 1}]:
            result = BehaviorCheckResult("chk", "desc", "exp", True, value)
            self.assertEqual(result.value, value)


class BehavioralEpisodeScoreTest(unittest.TestCase):
    """Tests for the BehavioralEpisodeScore dataclass."""

    def test_creation_stores_fields(self) -> None:
        check = make_behavior_check_result()
        score = BehavioralEpisodeScore(
            episode=3,
            seed=7,
            scenario="night_rest",
            objective="test_obj",
            success=True,
            checks={"check_a": check},
            behavior_metrics={"rate": 0.9},
            failures=[],
        )
        self.assertEqual(score.episode, 3)
        self.assertEqual(score.seed, 7)
        self.assertEqual(score.scenario, "night_rest")
        self.assertTrue(score.success)
        self.assertIn("check_a", score.checks)
        self.assertEqual(score.behavior_metrics["rate"], 0.9)
        self.assertEqual(score.failures, [])

    def test_can_be_converted_to_dict_with_asdict(self) -> None:
        score = make_behavioral_episode_score(behavior_metrics={"x": 1.0})
        d = asdict(score)
        self.assertIn("episode", d)
        self.assertIn("seed", d)
        self.assertIn("success", d)


class CompetenceLabelFromEvalReflexScaleTest(unittest.TestCase):
    def test_near_zero_scale_is_self_sufficient(self) -> None:
        self.assertEqual(
            competence_label_from_eval_reflex_scale(1e-12),
            "self_sufficient",
        )


class EpisodeStatsSeedFieldTest(unittest.TestCase):
    """Test that EpisodeStats correctly stores the new seed field."""

    def test_seed_stored(self) -> None:
        stats = make_episode_stats(seed=999)
        self.assertEqual(stats.seed, 999)

    def test_seed_zero(self) -> None:
        stats = make_episode_stats(seed=0)
        self.assertEqual(stats.seed, 0)

    def test_seed_in_asdict(self) -> None:
        """
        Verify that an EpisodeStats instance includes its `seed` field when converted to a dict and that the value is preserved.
        
        Constructs an EpisodeStats with a specific seed, converts it with `dataclasses.asdict`, and asserts the resulting mapping contains the "seed" key with the same integer value.
        """
        stats = make_episode_stats(seed=123)
        d = asdict(stats)
        self.assertIn("seed", d)
        self.assertEqual(d["seed"], 123)


class PredatorTypeNamesConstantTest(unittest.TestCase):
    """Tests for metrics.PREDATOR_TYPE_NAMES - new in this PR."""

    def test_contains_visual(self) -> None:
        self.assertIn("visual", PREDATOR_TYPE_NAMES)

    def test_contains_olfactory(self) -> None:
        self.assertIn("olfactory", PREDATOR_TYPE_NAMES)

    def test_has_two_entries(self) -> None:
        self.assertEqual(len(PREDATOR_TYPE_NAMES), 2)


class EpisodeStatsNewFieldsTest(unittest.TestCase):
    """Tests for new EpisodeStats fields - predator_contacts_by_type, predator_escapes_by_type, etc."""

    def _make_stats(self, **overrides) -> EpisodeStats:
        """
        Build an EpisodeStats instance using the shared test defaults.
        """
        return make_episode_stats(**overrides)

    def test_default_predator_contacts_by_type_is_empty_dict(self) -> None:
        stats = self._make_stats()
        self.assertEqual(stats.predator_contacts_by_type, {})

    def test_default_predator_escapes_by_type_is_empty_dict(self) -> None:
        stats = self._make_stats()
        self.assertEqual(stats.predator_escapes_by_type, {})

    def test_default_predator_response_latency_by_type_is_empty_dict(self) -> None:
        stats = self._make_stats()
        self.assertEqual(stats.predator_response_latency_by_type, {})

    def test_default_module_response_by_predator_type_is_empty_dict(self) -> None:
        stats = self._make_stats()
        self.assertEqual(stats.module_response_by_predator_type, {})

    def test_predator_contacts_by_type_can_be_set(self) -> None:
        stats = self._make_stats(predator_contacts_by_type={"visual": 3, "olfactory": 1})
        self.assertEqual(stats.predator_contacts_by_type["visual"], 3)
        self.assertEqual(stats.predator_contacts_by_type["olfactory"], 1)

    def test_predator_escapes_by_type_can_be_set(self) -> None:
        stats = self._make_stats(predator_escapes_by_type={"visual": 2, "olfactory": 0})
        self.assertEqual(stats.predator_escapes_by_type["visual"], 2)

    def test_predator_response_latency_by_type_can_be_set(self) -> None:
        stats = self._make_stats(predator_response_latency_by_type={"visual": 3.5, "olfactory": 0.0})
        self.assertAlmostEqual(stats.predator_response_latency_by_type["visual"], 3.5)

    def test_module_response_by_predator_type_can_be_set(self) -> None:
        """
        Verifies that EpisodeStats.module_response_by_predator_type accepts a mapping and preserves nested module weights.
        
        Constructs an EpisodeStats with a provided nested mapping for predator types to module response values and asserts that the stored value for the 'visual' predator's 'visual_cortex' response is approximately 0.7.
        """
        module_response = {
            "visual": {"visual_cortex": 0.7, "sensory_cortex": 0.3},
            "olfactory": {"visual_cortex": 0.2, "sensory_cortex": 0.8},
        }
        stats = self._make_stats(module_response_by_predator_type=module_response)
        self.assertAlmostEqual(
            stats.module_response_by_predator_type["visual"]["visual_cortex"], 0.7
        )


if __name__ == "__main__":
    unittest.main()
