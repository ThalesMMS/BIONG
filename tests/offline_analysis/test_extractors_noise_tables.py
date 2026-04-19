from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from spider_cortex_sim.offline_analysis.constants import (
    DEFAULT_MINIMAL_SHAPING_SURVIVAL_THRESHOLD,
    SHAPING_DEPENDENCE_WARNING_THRESHOLD,
)
from spider_cortex_sim.offline_analysis.extractors import (
    _aggregate_specialization_from_scenarios,
    _noise_robustness_cell_summary,
    _noise_robustness_metrics,
    _normalize_module_response_by_predator_type,
    _normalize_noise_marginals,
    _ordered_noise_conditions,
    extract_ablations,
    extract_noise_robustness,
    extract_predator_type_specialization,
    extract_reflex_frequency,
    extract_representation_specialization,
    extract_shaping_audit,
)
from spider_cortex_sim.offline_analysis.ingestion import load_summary, normalize_behavior_rows
from spider_cortex_sim.offline_analysis.report import build_report_data, write_report
from spider_cortex_sim.offline_analysis.tables import build_diagnostics
from spider_cortex_sim.simulation import SpiderSimulation

from .conftest import (
    CHECKIN_SUMMARY,
    EPISODE_SHAPING_GAP,
    LARGE_SHAPING_GAP,
    MEAN_REWARD_SHAPING_GAP,
    SHAPING_GAP_EPSILON,
    SMALL_SHAPING_GAP,
)

class NoiseRobustnessMetricsTest(unittest.TestCase):
    """Tests for _noise_robustness_metrics helper (new in this PR)."""

    def _make_matrix(self) -> dict:
        return {
            "none": {
                "none": {"scenario_success_rate": 1.0},
                "high": {"scenario_success_rate": 0.5},
            },
            "low": {
                "none": {"scenario_success_rate": 0.8},
                "high": {"scenario_success_rate": 0.2},
            },
        }

    def test_robustness_score_is_mean_of_all_cells(self) -> None:
        matrix = self._make_matrix()
        result = _noise_robustness_metrics(
            matrix,
            train_conditions=["none", "low"],
            eval_conditions=["none", "high"],
        )
        expected = (1.0 + 0.5 + 0.8 + 0.2) / 4
        self.assertAlmostEqual(result["robustness_score"], expected)

    def test_diagonal_score_is_mean_of_matched_cells(self) -> None:
        matrix = self._make_matrix()
        result = _noise_robustness_metrics(
            matrix,
            train_conditions=["none", "low"],
            eval_conditions=["none", "high"],
        )
        # diagonal cells: (none, none)=1.0; (low, low) is absent
        self.assertAlmostEqual(result["diagonal_score"], 1.0)

    def test_off_diagonal_score_excludes_matched_cells(self) -> None:
        matrix = self._make_matrix()
        result = _noise_robustness_metrics(
            matrix,
            train_conditions=["none", "low"],
            eval_conditions=["none", "high"],
        )
        # off-diagonal: (none, high)=0.5, (low, none)=0.8, (low, high)=0.2
        # Only (none, none) is diagonal; "low" != "high" so (low, high) is also off-diagonal
        expected = (0.5 + 0.8 + 0.2) / 3
        self.assertAlmostEqual(result["off_diagonal_score"], expected)

    def test_train_marginals_computed_per_row(self) -> None:
        matrix = self._make_matrix()
        result = _noise_robustness_metrics(
            matrix,
            train_conditions=["none", "low"],
            eval_conditions=["none", "high"],
        )
        self.assertAlmostEqual(result["train_marginals"]["none"], 0.75)
        self.assertAlmostEqual(result["train_marginals"]["low"], 0.5)

    def test_eval_marginals_computed_per_column(self) -> None:
        matrix = self._make_matrix()
        result = _noise_robustness_metrics(
            matrix,
            train_conditions=["none", "low"],
            eval_conditions=["none", "high"],
        )
        self.assertAlmostEqual(result["eval_marginals"]["none"], 0.9)
        self.assertAlmostEqual(result["eval_marginals"]["high"], 0.35)

    def test_available_cell_count_counts_present_cells(self) -> None:
        matrix = self._make_matrix()
        result = _noise_robustness_metrics(
            matrix,
            train_conditions=["none", "low"],
            eval_conditions=["none", "high"],
        )
        self.assertEqual(result["available_cell_count"], 4)

    def test_missing_cells_are_excluded_from_available_count(self) -> None:
        matrix = {
            "none": {"none": {"scenario_success_rate": 1.0}},
            "low": {},
        }
        result = _noise_robustness_metrics(
            matrix,
            train_conditions=["none", "low"],
            eval_conditions=["none", "high"],
        )
        self.assertEqual(result["available_cell_count"], 1)

    def test_empty_matrix_returns_zero_scores(self) -> None:
        result = _noise_robustness_metrics(
            {},
            train_conditions=["none"],
            eval_conditions=["low"],
        )
        self.assertAlmostEqual(result["robustness_score"], 0.0)
        self.assertAlmostEqual(result["diagonal_score"], 0.0)
        self.assertEqual(result["available_cell_count"], 0)

    def test_fully_diagonal_matrix_returns_zero_off_diagonal(self) -> None:
        matrix = {
            "none": {"none": {"scenario_success_rate": 0.9}},
            "low": {"low": {"scenario_success_rate": 0.7}},
        }
        result = _noise_robustness_metrics(
            matrix,
            train_conditions=["none", "low"],
            eval_conditions=["none", "low"],
        )
        self.assertAlmostEqual(result["diagonal_score"], 0.8)
        self.assertAlmostEqual(result["off_diagonal_score"], 0.0)

class NoiseRobustnessCellSummaryTest(unittest.TestCase):
    """Tests for _noise_robustness_cell_summary helper (new in this PR)."""

    def test_reads_scenario_and_episode_success_rate_from_summary(self) -> None:
        payload = {
            "summary": {
                "scenario_success_rate": 0.8,
                "episode_success_rate": 0.75,
                "mean_reward": 1.5,
                "scenario_count": 3,
                "episode_count": 12,
            }
        }
        result = _noise_robustness_cell_summary(payload)
        self.assertAlmostEqual(result["scenario_success_rate"], 0.8)
        self.assertAlmostEqual(result["episode_success_rate"], 0.75)
        self.assertAlmostEqual(result["mean_reward"], 1.5)
        self.assertEqual(result["scenario_count"], 3)
        self.assertEqual(result["episode_count"], 12)

    def test_empty_summary_returns_zeros(self) -> None:
        result = _noise_robustness_cell_summary({})
        self.assertAlmostEqual(result["scenario_success_rate"], 0.0)
        self.assertAlmostEqual(result["episode_success_rate"], 0.0)
        self.assertAlmostEqual(result["mean_reward"], 0.0)

    def test_falls_back_to_legacy_scenarios_mean_reward_when_summary_missing(self) -> None:
        payload = {
            "legacy_scenarios": {
                "night_rest": {"mean_reward": 2.0},
                "day_forage": {"mean_reward": 4.0},
            }
        }
        result = _noise_robustness_cell_summary(payload)
        self.assertAlmostEqual(result["mean_reward"], 3.0)

    def test_scenario_count_falls_back_to_legacy_scenario_count(self) -> None:
        payload = {
            "legacy_scenarios": {
                "night_rest": {"mean_reward": 1.0},
                "day_forage": {"mean_reward": 1.0},
            }
        }
        result = _noise_robustness_cell_summary(payload)
        self.assertEqual(result["scenario_count"], 2)

    def test_missing_legacy_and_summary_mean_reward_defaults_to_zero(self) -> None:
        payload = {"summary": {"scenario_success_rate": 0.5}}
        result = _noise_robustness_cell_summary(payload)
        self.assertAlmostEqual(result["mean_reward"], 0.0)

    def test_non_mapping_legacy_scenario_items_are_ignored(self) -> None:
        payload = {
            "legacy_scenarios": {
                "good": {"mean_reward": 2.0},
                "bad": "not_a_mapping",
            }
        }
        result = _noise_robustness_cell_summary(payload)
        self.assertAlmostEqual(result["mean_reward"], 2.0)

class NormalizeNoiseMarginalTest(unittest.TestCase):
    """Tests for _normalize_noise_marginals helper (new in this PR)."""

    def test_uses_payload_values_when_present(self) -> None:
        result = _normalize_noise_marginals(
            {"none": 0.9, "low": 0.7},
            conditions=["none", "low"],
            fallback={"none": 0.1, "low": 0.2},
        )
        self.assertAlmostEqual(result["none"], 0.9)
        self.assertAlmostEqual(result["low"], 0.7)

    def test_falls_back_when_condition_missing_from_payload(self) -> None:
        result = _normalize_noise_marginals(
            {"none": 0.9},
            conditions=["none", "low"],
            fallback={"none": 0.1, "low": 0.5},
        )
        self.assertAlmostEqual(result["none"], 0.9)
        self.assertAlmostEqual(result["low"], 0.5)

    def test_non_mapping_payload_uses_fallback_entirely(self) -> None:
        result = _normalize_noise_marginals(
            None,
            conditions=["none", "low"],
            fallback={"none": 0.3, "low": 0.6},
        )
        self.assertAlmostEqual(result["none"], 0.3)
        self.assertAlmostEqual(result["low"], 0.6)

    def test_empty_conditions_returns_empty_dict(self) -> None:
        result = _normalize_noise_marginals(
            {"none": 0.5},
            conditions=[],
            fallback={},
        )
        self.assertEqual(result, {})

    def test_missing_from_both_payload_and_fallback_defaults_to_zero(self) -> None:
        result = _normalize_noise_marginals(
            {},
            conditions=["none"],
            fallback={},
        )
        self.assertAlmostEqual(result["none"], 0.0)

class OrderedNoiseConditionsTest(unittest.TestCase):
    """Tests for _ordered_noise_conditions helper (new in this PR)."""

    def test_canonical_conditions_kept_in_canonical_order(self) -> None:
        result = _ordered_noise_conditions(["high", "none", "medium", "low"])
        self.assertEqual(result, ["none", "low", "medium", "high"])

    def test_extra_conditions_sorted_alphabetically_after_canonicals(self) -> None:
        result = _ordered_noise_conditions(["beta", "none", "alpha"])
        self.assertEqual(result, ["none", "alpha", "beta"])

    def test_empty_input_returns_empty_list(self) -> None:
        self.assertEqual(_ordered_noise_conditions([]), [])

    def test_duplicate_names_deduplicated(self) -> None:
        result = _ordered_noise_conditions(["none", "low", "none", "low"])
        self.assertEqual(result, ["none", "low"])

    def test_falsy_names_are_excluded(self) -> None:
        result = _ordered_noise_conditions(["", "none", ""])
        self.assertEqual(result, ["none"])

    def test_non_canonical_only_returns_sorted_list(self) -> None:
        result = _ordered_noise_conditions(["zeta", "alpha"])
        self.assertEqual(result, ["alpha", "zeta"])

    def test_subset_of_canonical_conditions_preserved_in_order(self) -> None:
        result = _ordered_noise_conditions(["high", "low"])
        self.assertEqual(result, ["low", "high"])

    def test_all_canonical_conditions_are_returned_in_full_canonical_order(self) -> None:
        result = _ordered_noise_conditions(["high", "medium", "low", "none"])
        self.assertEqual(result, ["none", "low", "medium", "high"])
