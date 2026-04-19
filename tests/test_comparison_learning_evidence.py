import unittest
from unittest import mock

import numpy as np

from spider_cortex_sim.ablations import canonical_ablation_configs
from spider_cortex_sim.comparison import (
    aggregate_with_uncertainty,
    behavior_metric_seed_rows,
    build_ablation_deltas,
    build_learning_evidence_deltas,
    build_learning_evidence_summary,
    build_predator_type_specialization_summary,
    build_reward_audit,
    build_reward_audit_comparison,
    austere_survival_gate_passed,
    compare_ablation_suite,
    compare_behavior_suite,
    compare_configurations,
    compare_learning_evidence,
    compare_noise_robustness,
    compare_reward_profiles,
    compare_training_regimes,
    condition_compact_summary,
    condition_mean_reward,
    module_response_by_predator_type_from_payload,
    profile_comparison_metrics,
    representation_specialization_from_payload,
    safe_float,
    visual_minus_olfactory_seed_rows,
)
from spider_cortex_sim.curriculum import CURRICULUM_FOCUS_SCENARIOS
from spider_cortex_sim.learning_evidence import LearningEvidenceConditionSpec
from spider_cortex_sim.noise import RobustnessMatrixSpec
from spider_cortex_sim.reward import SCENARIO_AUSTERE_REQUIREMENTS, SHAPING_GAP_POLICY
from spider_cortex_sim.simulation import SpiderSimulation

class LearningEvidenceComparisonHelpersTest(unittest.TestCase):
    def test_condition_compact_summary_none_returns_zeros(self) -> None:
            result = condition_compact_summary(None)
            self.assertEqual(result["scenario_success_rate"], 0.0)
            self.assertEqual(result["episode_success_rate"], 0.0)
            self.assertEqual(result["mean_reward"], 0.0)

    def test_condition_compact_summary_non_dict_returns_zeros(self) -> None:
            result = condition_compact_summary("not_a_dict")  # type: ignore[arg-type]
            self.assertEqual(result["scenario_success_rate"], 0.0)
            self.assertEqual(result["episode_success_rate"], 0.0)
            self.assertEqual(result["mean_reward"], 0.0)

    def test_condition_compact_summary_missing_summary_key_returns_zeros(self) -> None:
            result = condition_compact_summary({"policy_mode": "normal"})
            self.assertEqual(result["scenario_success_rate"], 0.0)
            self.assertEqual(result["episode_success_rate"], 0.0)
            self.assertEqual(result["mean_reward"], 0.0)

    def test_condition_compact_summary_extracts_values(self) -> None:
            payload = {
                "summary": {
                    "scenario_success_rate": 0.8,
                    "episode_success_rate": 0.75,
                    "mean_reward": 12.5,
                }
            }
            result = condition_compact_summary(payload)
            self.assertAlmostEqual(result["scenario_success_rate"], 0.8)
            self.assertAlmostEqual(result["episode_success_rate"], 0.75)
            self.assertAlmostEqual(result["mean_reward"], 12.5)

    def test_condition_compact_summary_partial_keys_default_to_zero(self) -> None:
            payload = {"summary": {"scenario_success_rate": 0.5}}
            result = condition_compact_summary(payload)
            self.assertAlmostEqual(result["scenario_success_rate"], 0.5)
            self.assertAlmostEqual(result["episode_success_rate"], 0.0)
            self.assertAlmostEqual(result["mean_reward"], 0.0)

    def test_condition_compact_summary_non_dict_summary_returns_zeros(self) -> None:
            payload = {"summary": "not_a_dict"}
            result = condition_compact_summary(payload)
            self.assertEqual(result["scenario_success_rate"], 0.0)

    def test_compare_learning_evidence_empty_seeds_raise_clear_error(self) -> None:
            with self.assertRaisesRegex(ValueError, "behavior seed"):
                compare_learning_evidence(
                    budget_profile="smoke",
                    long_budget_profile="smoke",
                    names=("night_rest",),
                    seeds=(),
                    condition_names=("trained_final",),
                )

    def test_safe_float_rejects_non_finite_values(self) -> None:
            self.assertEqual(safe_float(float("nan")), 0.0)
            self.assertEqual(safe_float(float("inf")), 0.0)
            self.assertEqual(safe_float(float("-inf")), 0.0)

    def test_condition_mean_reward_rejects_non_finite_values(self) -> None:
            self.assertEqual(
                condition_mean_reward({"summary": {"mean_reward": float("nan")}}),
                0.0,
            )
            self.assertEqual(
                condition_mean_reward({"mean_reward": float("inf")}),
                0.0,
            )

    def test_aggregate_with_uncertainty_skips_malformed_items(self) -> None:
            result = aggregate_with_uncertainty(
                [
                    (1, 0.2),
                    "bad",
                    (2, 0.4, 0.6),
                    {"seed": 3, "value": 0.6},
                ],
                n_resamples=10,
            )
            self.assertEqual(result["n_seeds"], 2)

    def test_aggregate_with_uncertainty_validates_parameters_without_rows(self) -> None:
            with self.assertRaises(ValueError):
                aggregate_with_uncertainty([], confidence_level=1.0)
            with self.assertRaises(ValueError):
                aggregate_with_uncertainty([], n_resamples=0)

    def test_behavior_metric_seed_rows_skip_missing_metrics(self) -> None:
            rows = behavior_metric_seed_rows(
                [
                    (1, {"summary": {"scenario_success_rate": 0.5}, "suite": {}}),
                    (2, {"summary": {}, "suite": {}}),
                    (
                        3,
                        {
                            "summary": {"scenario_success_rate": float("nan")},
                            "suite": {},
                        },
                    ),
                ],
                metric_name="scenario_success_rate",
                condition="condition",
            )

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].seed, 1)

    def test_behavior_metric_seed_rows_skip_missing_scenario_success(self) -> None:
            rows = behavior_metric_seed_rows(
                [
                    (1, {"suite": {"night_rest": {"success_rate": 0.75}}}),
                    (2, {"suite": {"night_rest": {}}}),
                    (3, {"suite": {"night_rest": {"success_rate": float("inf")}}}),
                ],
                metric_name="scenario_success_rate",
                condition="condition",
                scenario="night_rest",
            )

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].seed, 1)

    def test_build_learning_evidence_deltas_skipped_condition(self) -> None:
            conditions = {
                "trained_final": {
                    "summary": {
                        "scenario_success_rate": 0.8,
                        "episode_success_rate": 0.7,
                        "mean_reward": 10.0,
                    },
                    "suite": {},
                },
                "random_init": {
                    "skipped": True,
                    "reason": "architecture mismatch",
                },
            }
            deltas = build_learning_evidence_deltas(
                conditions,
                reference_condition="trained_final",
                scenario_names=["night_rest"],
            )
            self.assertTrue(deltas["random_init"]["skipped"])
            self.assertEqual(deltas["random_init"]["reason"], "architecture mismatch")

    def test_build_learning_evidence_deltas_reference_has_zero_delta(self) -> None:
            conditions = {
                "trained_final": {
                    "summary": {
                        "scenario_success_rate": 0.8,
                        "episode_success_rate": 0.7,
                        "mean_reward": 10.0,
                    },
                    "suite": {"night_rest": {"success_rate": 0.9}},
                },
            }
            deltas = build_learning_evidence_deltas(
                conditions,
                reference_condition="trained_final",
                scenario_names=["night_rest"],
            )
            ref_delta = deltas["trained_final"]
            self.assertAlmostEqual(ref_delta["summary"]["scenario_success_rate_delta"], 0.0)
            self.assertAlmostEqual(ref_delta["summary"]["episode_success_rate_delta"], 0.0)
            self.assertAlmostEqual(ref_delta["summary"]["mean_reward_delta"], 0.0)

    def test_build_learning_evidence_deltas_correct_delta_values(self) -> None:
            conditions = {
                "trained_final": {
                    "summary": {
                        "scenario_success_rate": 1.0,
                        "episode_success_rate": 0.9,
                    },
                    "suite": {"night_rest": {"success_rate": 1.0}},
                    "legacy_scenarios": {"night_rest": {"mean_reward": 20.0}},
                },
                "random_init": {
                    "summary": {
                        "scenario_success_rate": 0.4,
                        "episode_success_rate": 0.3,
                    },
                    "suite": {"night_rest": {"success_rate": 0.4}},
                    "legacy_scenarios": {"night_rest": {"mean_reward": 5.0}},
                },
            }
            deltas = build_learning_evidence_deltas(
                conditions,
                reference_condition="trained_final",
                scenario_names=["night_rest"],
            )
            random_delta = deltas["random_init"]
            self.assertAlmostEqual(
                random_delta["summary"]["scenario_success_rate_delta"], 0.4 - 1.0
            )
            self.assertAlmostEqual(
                random_delta["summary"]["mean_reward_delta"], 5.0 - 20.0
            )
            self.assertAlmostEqual(
                random_delta["scenarios"]["night_rest"]["success_rate_delta"], 0.4 - 1.0
            )

    def test_compare_learning_evidence_skips_reflex_only_when_reflexes_disabled(self) -> None:
            payload, _ = compare_learning_evidence(
                budget_profile="smoke",
                long_budget_profile="smoke",
                names=("night_rest",),
                seeds=(7,),
                brain_config=canonical_ablation_configs()["no_module_reflexes"],
                condition_names=("reflex_only",),
            )

            self.assertIn("trained_without_reflex_support", payload["conditions"])
            self.assertIn("reflex_only", payload["conditions"])
            self.assertTrue(payload["conditions"]["reflex_only"]["skipped"])
            self.assertIn(
                "reflexes disabled",
                payload["conditions"]["reflex_only"]["reason"],
            )

    def test_build_learning_evidence_deltas_condition_without_summary_is_skipped(self) -> None:
            conditions = {
                "trained_final": {
                    "summary": {"scenario_success_rate": 0.8, "episode_success_rate": 0.7, "mean_reward": 5.0},
                    "suite": {},
                },
                "random_init": {"policy_mode": "normal"},  # no "summary" key
            }
            deltas = build_learning_evidence_deltas(
                conditions,
                reference_condition="trained_final",
                scenario_names=[],
            )
            self.assertTrue(deltas["random_init"]["skipped"])

    def test_build_learning_evidence_deltas_missing_reference_skips_all_conditions(self) -> None:
            conditions = {
                "random_init": {
                    "summary": {
                        "scenario_success_rate": 0.4,
                        "episode_success_rate": 0.3,
                        "mean_reward": 5.0,
                    },
                    "suite": {"night_rest": {"success_rate": 0.4}},
                },
            }
            deltas = build_learning_evidence_deltas(
                conditions,
                reference_condition="trained_final",
                scenario_names=["night_rest"],
            )

            self.assertTrue(deltas["random_init"]["skipped"])
            self.assertIn("missing or skipped", deltas["random_init"]["reason"])

    def test_build_learning_evidence_deltas_skipped_reference_skips_all_conditions(self) -> None:
            conditions = {
                "trained_final": {
                    "skipped": True,
                    "reason": "architecture mismatch",
                },
                "random_init": {
                    "summary": {
                        "scenario_success_rate": 0.4,
                        "episode_success_rate": 0.3,
                        "mean_reward": 5.0,
                    },
                    "suite": {"night_rest": {"success_rate": 0.4}},
                },
            }
            deltas = build_learning_evidence_deltas(
                conditions,
                reference_condition="trained_final",
                scenario_names=["night_rest"],
            )

            self.assertTrue(deltas["trained_final"]["skipped"])
            self.assertTrue(deltas["random_init"]["skipped"])
            self.assertIn("missing or skipped", deltas["random_init"]["reason"])

    def test_build_learning_evidence_summary_has_learning_evidence_true(self) -> None:
            conditions = {
                "trained_final": {
                    "summary": {"scenario_success_rate": 0.9, "episode_success_rate": 0.8, "mean_reward": 15.0},
                },
                "random_init": {
                    "summary": {"scenario_success_rate": 0.3, "episode_success_rate": 0.2, "mean_reward": 2.0},
                },
                "reflex_only": {
                    "summary": {"scenario_success_rate": 0.4, "episode_success_rate": 0.3, "mean_reward": 3.0},
                },
                "trained_without_reflex_support": {
                    "summary": {"scenario_success_rate": 0.7, "episode_success_rate": 0.6, "mean_reward": 10.0},
                },
            }
            result = build_learning_evidence_summary(
                conditions, reference_condition="trained_without_reflex_support"
            )
            self.assertTrue(result["has_learning_evidence"])
            self.assertTrue(result["supports_primary_evidence"])
            self.assertEqual(result["primary_gate_metric"], "scenario_success_rate")
            self.assertEqual(result["reference_condition"], "trained_without_reflex_support")
            self.assertEqual(
                result["primary_condition"],
                result["trained_without_reflex_support"],
            )
            self.assertAlmostEqual(
                result["trained_vs_random_init"]["scenario_success_rate_delta"],
                0.7 - 0.3,
            )

    def test_build_learning_evidence_summary_has_learning_evidence_false_when_trained_not_better(self) -> None:
            conditions = {
                "trained_final": {
                    "summary": {"scenario_success_rate": 0.9, "episode_success_rate": 0.8, "mean_reward": 15.0},
                },
                "trained_without_reflex_support": {
                    "summary": {"scenario_success_rate": 0.3, "episode_success_rate": 0.2, "mean_reward": 2.0},
                },
                "random_init": {
                    "summary": {"scenario_success_rate": 0.9, "episode_success_rate": 0.8, "mean_reward": 15.0},
                },
                "reflex_only": {
                    "summary": {"scenario_success_rate": 0.4, "episode_success_rate": 0.3, "mean_reward": 3.0},
                },
            }
            result = build_learning_evidence_summary(
                conditions, reference_condition="trained_without_reflex_support"
            )
            self.assertFalse(result["has_learning_evidence"])

    def test_build_learning_evidence_summary_reflex_only_not_available_sets_primary_supported_false(self) -> None:
            conditions = {
                "trained_final": {
                    "summary": {"scenario_success_rate": 0.9, "episode_success_rate": 0.8, "mean_reward": 15.0},
                },
                "trained_without_reflex_support": {
                    "summary": {"scenario_success_rate": 0.7, "episode_success_rate": 0.6, "mean_reward": 10.0},
                },
                "random_init": {
                    "summary": {"scenario_success_rate": 0.3, "episode_success_rate": 0.2, "mean_reward": 2.0},
                },
                "reflex_only": {
                    "skipped": True,
                    "reason": "monolithic architecture",
                },
            }
            result = build_learning_evidence_summary(
                conditions, reference_condition="trained_without_reflex_support"
            )
            self.assertFalse(result["supports_primary_evidence"])
            self.assertFalse(result["has_learning_evidence"])

    def test_build_learning_evidence_summary_notes_include_gate_note(self) -> None:
            conditions = {
                "trained_final": {
                    "summary": {"scenario_success_rate": 0.9, "episode_success_rate": 0.8, "mean_reward": 15.0},
                },
                "trained_without_reflex_support": {
                    "summary": {"scenario_success_rate": 0.7, "episode_success_rate": 0.6, "mean_reward": 10.0},
                },
                "random_init": {
                    "summary": {"scenario_success_rate": 0.3, "episode_success_rate": 0.2, "mean_reward": 2.0},
                },
                "reflex_only": {
                    "summary": {"scenario_success_rate": 0.4, "episode_success_rate": 0.3, "mean_reward": 3.0},
                },
            }
            result = build_learning_evidence_summary(
                conditions, reference_condition="trained_without_reflex_support"
            )
            notes = result["notes"]
            self.assertTrue(any("trained_without_reflex_support" in note for note in notes))
            self.assertTrue(any("scenario_success_rate" in note for note in notes))

    def test_build_learning_evidence_summary_contains_delta_blocks(self) -> None:
            conditions = {
                "trained_final": {
                    "summary": {"scenario_success_rate": 0.9, "episode_success_rate": 0.8, "mean_reward": 15.0},
                },
                "trained_without_reflex_support": {
                    "summary": {"scenario_success_rate": 0.7, "episode_success_rate": 0.6, "mean_reward": 10.0},
                },
                "random_init": {
                    "summary": {"scenario_success_rate": 0.3, "episode_success_rate": 0.2, "mean_reward": 2.0},
                },
                "reflex_only": {
                    "summary": {"scenario_success_rate": 0.4, "episode_success_rate": 0.3, "mean_reward": 3.0},
                },
            }
            result = build_learning_evidence_summary(
                conditions, reference_condition="trained_without_reflex_support"
            )
            self.assertIn("trained_vs_random_init", result)
            self.assertIn("trained_vs_reflex_only", result)
            self.assertIn("scenario_success_rate_delta", result["trained_vs_random_init"])

    def test_build_learning_evidence_summary_reports_uncertainty(self) -> None:
            conditions = {
                "trained_without_reflex_support": {
                    "summary": {"scenario_success_rate": 0.7, "episode_success_rate": 0.6, "mean_reward": 10.0},
                    "seed_level": [
                        {"metric_name": "scenario_success_rate", "seed": 1, "value": 0.6, "condition": "trained_without_reflex_support", "scenario": None},
                        {"metric_name": "scenario_success_rate", "seed": 2, "value": 0.8, "condition": "trained_without_reflex_support", "scenario": None},
                    ],
                    "uncertainty": {
                        "scenario_success_rate": {
                            "mean": 0.7,
                            "ci_lower": 0.6,
                            "ci_upper": 0.8,
                            "std_error": 0.1,
                            "n_seeds": 2,
                            "confidence_level": 0.95,
                            "seed_values": [0.6, 0.8],
                        }
                    },
                },
                "random_init": {
                    "summary": {"scenario_success_rate": 0.3, "episode_success_rate": 0.2, "mean_reward": 2.0},
                    "seed_level": [
                        {"metric_name": "scenario_success_rate", "seed": 1, "value": 0.2, "condition": "random_init", "scenario": None},
                        {"metric_name": "scenario_success_rate", "seed": 2, "value": 0.4, "condition": "random_init", "scenario": None},
                    ],
                    "uncertainty": {
                        "scenario_success_rate": {
                            "mean": 0.3,
                            "ci_lower": 0.2,
                            "ci_upper": 0.4,
                            "std_error": 0.1,
                            "n_seeds": 2,
                            "confidence_level": 0.95,
                            "seed_values": [0.2, 0.4],
                        }
                    },
                },
                "reflex_only": {
                    "summary": {"scenario_success_rate": 0.4, "episode_success_rate": 0.3, "mean_reward": 3.0},
                },
            }
            result = build_learning_evidence_summary(
                conditions, reference_condition="trained_without_reflex_support"
            )
            self.assertIn("seed_level", result)
            self.assertIn("uncertainty", result)
            delta_uncertainty = result["uncertainty"]["trained_vs_random_init"][
                "scenario_success_rate_delta"
            ]
            self.assertEqual(delta_uncertainty["n_seeds"], 2)
            self.assertAlmostEqual(delta_uncertainty["mean"], 0.4)

    def test_compare_learning_evidence_conditions_have_required_fields(self) -> None:
            payload, _rows = compare_learning_evidence(
                budget_profile="smoke",
                long_budget_profile="smoke",
                names=("night_rest",),
                seeds=(3,),
                condition_names=(
                    "trained_final",
                    "trained_reflex_annealed",
                    "random_init",
                ),
            )
            for cond_name in ("trained_final", "trained_reflex_annealed", "random_init"):
                cond = payload["conditions"][cond_name]
                self.assertIn("policy_mode", cond)
                self.assertIn("training_regime", cond)
                self.assertIn("train_episodes", cond)
                self.assertIn("checkpoint_source", cond)
                self.assertIn("budget_profile", cond)
                self.assertIn("skipped", cond)
            self.assertEqual(
                payload["conditions"]["trained_reflex_annealed"]["training_regime"],
                "reflex_annealed",
            )

    def test_compare_learning_evidence_random_init_has_zero_train_episodes(self) -> None:
            payload, _ = compare_learning_evidence(
                budget_profile="smoke",
                long_budget_profile="smoke",
                names=("night_rest",),
                seeds=(3,),
                condition_names=("random_init",),
            )
            self.assertEqual(payload["conditions"]["random_init"]["train_episodes"], 0)

    def test_compare_learning_evidence_rows_contain_all_learning_evidence_columns(self) -> None:
            _, rows = compare_learning_evidence(
                budget_profile="smoke",
                long_budget_profile="smoke",
                names=("night_rest",),
                seeds=(3,),
                condition_names=("trained_final", "trained_reflex_annealed"),
            )
            self.assertTrue(rows)
            expected_keys = [
                "learning_evidence_condition",
                "learning_evidence_policy_mode",
                "learning_evidence_training_regime",
                "learning_evidence_train_episodes",
                "learning_evidence_frozen_after_episode",
                "learning_evidence_checkpoint_source",
                "learning_evidence_budget_profile",
                "learning_evidence_budget_benchmark_strength",
            ]
            for key in expected_keys:
                self.assertIn(key, rows[0])
            regime_rows = [
                row
                for row in rows
                if row["learning_evidence_condition"] == "trained_reflex_annealed"
            ]
            self.assertTrue(regime_rows)
            self.assertTrue(
                all(
                    row["learning_evidence_training_regime"] == "reflex_annealed"
                    for row in regime_rows
                )
            )

    def test_compare_learning_evidence_evidence_summary_has_gate_fields(self) -> None:
            payload, _ = compare_learning_evidence(
                budget_profile="smoke",
                long_budget_profile="smoke",
                names=("night_rest",),
                seeds=(3,),
                condition_names=("trained_final", "random_init", "reflex_only"),
            )
            ev = payload["evidence_summary"]
            self.assertIn("has_learning_evidence", ev)
            self.assertIn("supports_primary_evidence", ev)
            self.assertIn("primary_gate_metric", ev)
            self.assertEqual(ev["primary_gate_metric"], "scenario_success_rate")

class BuildAblationDeltasTest(unittest.TestCase):
    """Tests for comparison.build_ablation_deltas (new in this PR)."""

    def _make_variants(self) -> dict:
        def _variant(scenario_sr: float, episode_sr: float, scenario_success: float):
            return {
                "summary": {
                    "scenario_success_rate": scenario_sr,
                    "episode_success_rate": episode_sr,
                },
                "suite": {
                    "night_rest": {"success_rate": scenario_success},
                },
            }
        return {
            "modular_full": _variant(0.50, 0.60, 0.50),
            "no_module_reflexes": _variant(0.30, 0.40, 0.20),
        }

    def test_reference_variant_delta_is_zero(self) -> None:
        variants = self._make_variants()
        deltas = build_ablation_deltas(
            variants, reference_variant="modular_full", scenario_names=["night_rest"]
        )
        ref_delta = deltas["modular_full"]["summary"]
        self.assertAlmostEqual(ref_delta["scenario_success_rate_delta"], 0.0)
        self.assertAlmostEqual(ref_delta["episode_success_rate_delta"], 0.0)

    def test_nonreference_variant_delta_is_computed_correctly(self) -> None:
        variants = self._make_variants()
        deltas = build_ablation_deltas(
            variants, reference_variant="modular_full", scenario_names=["night_rest"]
        )
        d = deltas["no_module_reflexes"]["summary"]
        self.assertAlmostEqual(d["scenario_success_rate_delta"], 0.30 - 0.50, places=5)
        self.assertAlmostEqual(d["episode_success_rate_delta"], 0.40 - 0.60, places=5)

    def test_scenario_level_delta_computed(self) -> None:
        variants = self._make_variants()
        deltas = build_ablation_deltas(
            variants, reference_variant="modular_full", scenario_names=["night_rest"]
        )
        scenario_delta = deltas["no_module_reflexes"]["scenarios"]["night_rest"]["success_rate_delta"]
        self.assertAlmostEqual(scenario_delta, 0.20 - 0.50, places=5)

    def test_reference_scenario_delta_is_zero(self) -> None:
        variants = self._make_variants()
        deltas = build_ablation_deltas(
            variants, reference_variant="modular_full", scenario_names=["night_rest"]
        )
        self.assertAlmostEqual(
            deltas["modular_full"]["scenarios"]["night_rest"]["success_rate_delta"], 0.0
        )

    def test_all_variants_present_in_deltas(self) -> None:
        variants = self._make_variants()
        deltas = build_ablation_deltas(
            variants, reference_variant="modular_full", scenario_names=["night_rest"]
        )
        self.assertIn("modular_full", deltas)
        self.assertIn("no_module_reflexes", deltas)

    def test_partial_variant_payloads_are_skipped_without_keyerror(self) -> None:
        variants = self._make_variants()
        variants["summary_only"] = {
            "summary": {"scenario_success_rate": 0.4},
        }
        variants["empty"] = {}

        deltas = build_ablation_deltas(
            variants,
            reference_variant="modular_full",
            scenario_names=["night_rest", "missing"],
        )

        self.assertIn("summary_only", deltas)
        self.assertEqual(
            deltas["summary_only"]["summary"],
            {"scenario_success_rate_delta": -0.1},
        )
        self.assertEqual(deltas["summary_only"]["scenarios"], {})
        self.assertNotIn("empty", deltas)

    def test_uncertainty_and_seed_level_deltas_are_reported(self) -> None:
        variants = self._make_variants()
        variants["modular_full"]["seed_level"] = [
            {
                "metric_name": "scenario_success_rate",
                "seed": 1,
                "value": 0.4,
                "condition": "modular_full",
                "scenario": None,
            },
            {
                "metric_name": "scenario_success_rate",
                "seed": 2,
                "value": 0.6,
                "condition": "modular_full",
                "scenario": None,
            },
        ]
        variants["no_module_reflexes"]["seed_level"] = [
            {
                "metric_name": "scenario_success_rate",
                "seed": 1,
                "value": 0.2,
                "condition": "no_module_reflexes",
                "scenario": None,
            },
            {
                "metric_name": "scenario_success_rate",
                "seed": 2,
                "value": 0.4,
                "condition": "no_module_reflexes",
                "scenario": None,
            },
        ]
        deltas = build_ablation_deltas(
            variants, reference_variant="modular_full", scenario_names=["night_rest"]
        )
        result = deltas["no_module_reflexes"]
        self.assertIn("uncertainty", result)
        self.assertIn("seed_level", result)
        uncertainty = result["uncertainty"]["scenario_success_rate_delta"]
        self.assertEqual(uncertainty["n_seeds"], 2)
        self.assertAlmostEqual(uncertainty["mean"], -0.2)

class PredatorTypeSpecializationUncertaintyTest(unittest.TestCase):
    def _variant(self, condition: str, values: dict[str, tuple[float, float]]) -> dict:
        suite = {
            scenario: {"success_rate": sum(seed_values) / len(seed_values)}
            for scenario, seed_values in values.items()
        }
        seed_level: list[dict[str, object]] = [
            {
                "metric_name": "specialization_score",
                "seed": 1,
                "value": 0.2,
                "condition": condition,
                "scenario": None,
            },
            {
                "metric_name": "specialization_score",
                "seed": 2,
                "value": 0.4,
                "condition": condition,
                "scenario": None,
            },
        ]
        for scenario, seed_values in values.items():
            self.assertEqual(2, len(seed_values))
            scenario_rows = [
                {
                    "metric_name": "scenario_success_rate",
                    "seed": seed,
                    "value": value,
                    "condition": condition,
                    "scenario": scenario,
                }
                for seed, value in zip((1, 2), seed_values)
            ]
            suite[scenario]["seed_level"] = scenario_rows
            seed_level.extend(scenario_rows)
        return {
            "summary": {
                "scenario_success_rate": 0.5,
                "episode_success_rate": 0.5,
            },
            "suite": suite,
            "seed_level": seed_level,
        }

    def test_visual_minus_olfactory_rows_skip_malformed_seeds(self) -> None:
        rows = visual_minus_olfactory_seed_rows(
            {
                "seed_level": [
                    {
                        "metric_name": "scenario_success_rate",
                        "seed": None,
                        "value": 1.0,
                        "scenario": "visual_hunter_open_field",
                    },
                    {
                        "metric_name": "scenario_success_rate",
                        "seed": 1,
                        "value": 0.8,
                        "scenario": "visual_hunter_open_field",
                    },
                    {
                        "metric_name": "scenario_success_rate",
                        "seed": 1,
                        "value": 0.5,
                        "scenario": "olfactory_ambush",
                    },
                ]
            },
            condition="variant",
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].seed, 1)
        self.assertAlmostEqual(rows[0].value, 0.3)

    def test_visual_minus_olfactory_rows_count_shared_scenario_in_both_buckets(self) -> None:
        rows = visual_minus_olfactory_seed_rows(
            {
                "seed_level": [
                    {
                        "metric_name": "scenario_success_rate",
                        "seed": 1,
                        "value": 0.6,
                        "scenario": "visual_olfactory_pincer",
                    },
                    {
                        "metric_name": "scenario_success_rate",
                        "seed": 1,
                        "value": 1.0,
                        "scenario": "visual_hunter_open_field",
                    },
                    {
                        "metric_name": "scenario_success_rate",
                        "seed": 1,
                        "value": 0.2,
                        "scenario": "olfactory_ambush",
                    },
                ]
            },
            condition="variant",
        )

        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0].value, 0.4)

    def test_representation_score_ignores_missing_score_sentinel(self) -> None:
        result = representation_specialization_from_payload(
            {
                "suite": {
                    "proposer_only": {
                        "proposer_divergence_by_module": {
                            "visual_cortex": 0.2,
                        },
                    },
                    "scored": {
                        "representation_specialization_score": 0.8,
                    },
                }
            }
        )

        self.assertTrue(result["available"])
        self.assertAlmostEqual(result["representation_specialization_score"], 0.8)

    def test_predator_type_summary_includes_custom_variant_names(self) -> None:
        variants = {
            "modular_full": self._variant(
                "modular_full",
                {
                    "visual_olfactory_pincer": (0.5, 0.5),
                    "visual_hunter_open_field": (0.6, 0.8),
                    "olfactory_ambush": (0.4, 0.2),
                },
            ),
            "drop_alert_center": self._variant(
                "drop_alert_center",
                {
                    "visual_olfactory_pincer": (0.2, 0.2),
                    "visual_hunter_open_field": (0.1, 0.3),
                    "olfactory_ambush": (0.7, 0.5),
                },
            ),
        }
        deltas = build_ablation_deltas(
            variants,
            reference_variant="modular_full",
            scenario_names=[
                "visual_olfactory_pincer",
                "visual_hunter_open_field",
                "olfactory_ambush",
            ],
        )

        result = build_predator_type_specialization_summary(
            variants,
            reference_variant="modular_full",
            deltas_vs_reference=deltas,
        )

        self.assertIn("drop_alert_center", result["comparisons"])

    def test_specialization_uncertainty_uses_aggregate_fallback_row(self) -> None:
        payload = {
            "legacy_scenarios": {
                "visual": {
                    "module_response_by_predator_type": {
                        "visual": {"visual_cortex": 0.9, "sensory_cortex": 0.1},
                        "olfactory": {"visual_cortex": 0.1, "sensory_cortex": 0.9},
                    }
                }
            }
        }
        variants = {
            "modular_full": payload,
            "custom_variant": payload,
        }

        result = build_predator_type_specialization_summary(
            variants,
            reference_variant="modular_full",
            deltas_vs_reference={},
        )

        self.assertEqual(
            result["uncertainty"]["custom_variant"]["specialization_score"]["n_seeds"],
            1,
        )

    def test_module_response_skips_malformed_values(self) -> None:
        result = module_response_by_predator_type_from_payload(
            {
                "legacy_scenarios": {
                    "scenario": {
                        "module_response_by_predator_type": {
                            "visual": {
                                "visual_cortex": "bad",
                                "sensory_cortex": 0.5,
                            }
                        }
                    }
                }
            }
        )

        self.assertNotIn("visual_cortex", result["visual"])
        self.assertAlmostEqual(result["visual"]["sensory_cortex"], 0.5)

    def test_predator_type_specialization_summary_reports_uncertainty(self) -> None:
        variants = {
            "modular_full": self._variant(
                "modular_full",
                {
                    "visual_olfactory_pincer": (0.5, 0.5),
                    "visual_hunter_open_field": (0.6, 0.8),
                    "olfactory_ambush": (0.4, 0.2),
                },
            ),
            "drop_visual_cortex": self._variant(
                "drop_visual_cortex",
                {
                    "visual_olfactory_pincer": (0.2, 0.2),
                    "visual_hunter_open_field": (0.1, 0.3),
                    "olfactory_ambush": (0.7, 0.5),
                },
            ),
        }
        deltas = build_ablation_deltas(
            variants,
            reference_variant="modular_full",
            scenario_names=[
                "visual_olfactory_pincer",
                "visual_hunter_open_field",
                "olfactory_ambush",
            ],
        )

        result = build_predator_type_specialization_summary(
            variants,
            reference_variant="modular_full",
            deltas_vs_reference=deltas,
        )

        self.assertTrue(result["available"])
        uncertainty = result["uncertainty"]["drop_visual_cortex"]
        self.assertEqual(
            uncertainty["visual_minus_olfactory_success_rate"]["n_seeds"],
            2,
        )
        self.assertEqual(uncertainty["specialization_score"]["n_seeds"], 2)
