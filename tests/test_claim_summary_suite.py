import json
import unittest

from spider_cortex_sim.claim_tests import (
    ClaimTestSpec,
    ScaffoldAssessment,
    ScaffoldSupportLevel,
    WARM_START_MINIMAL_THRESHOLD,
    assess_scaffold_support,
    canonical_claim_tests,
    claim_test_names,
    primary_claim_test_names,
    resolve_claim_tests,
)
from spider_cortex_sim.claim_evaluation import (
    SPECIALIZATION_ENGAGEMENT_CHECKS,
    build_claim_test_summary,
    claim_count_threshold,
    claim_leakage_audit_summary,
    claim_noise_subset_scores,
    claim_payload_config_summary,
    claim_payload_eval_reflex_scale,
    claim_registry_entry,
    claim_skip_result,
    claim_subset_scenario_success_rate,
    claim_test_source,
    claim_threshold_from_operator,
    claim_threshold_from_phrase,
    evaluate_claim_test,
    extract_claim_config_for_scaffold_assessment,
    run_claim_test_suite,
)
from spider_cortex_sim.comparison import (
    compare_noise_robustness,
    metric_seed_values_from_payload,
)
from spider_cortex_sim.noise import RobustnessMatrixSpec
from spider_cortex_sim.reward import SCENARIO_AUSTERE_REQUIREMENTS
from spider_cortex_sim.simulation import CAPABILITY_PROBE_SCENARIOS

from tests.fixtures.claim_tests import (
    _behavior_payload,
    _representation_metrics,
    _scaffold_config_summary,
    _add_seed_level_success_rates,
    _austere_payload,
)

class BuildClaimTestSummaryTest(unittest.TestCase):
    def test_aggregation_counts_passed_failed_and_skipped(self) -> None:
        claim_results = {
            "learning_without_privileged_signals": {
                "status": "passed",
                "passed": True,
                "scaffold_support_level": "minimal_manual",
                "benchmark_of_record_eligible": True,
            },
            "escape_without_reflex_support": {
                "status": "failed",
                "passed": False,
                "scaffold_support_level": "standard_constrained",
                "benchmark_of_record_eligible": False,
            },
            "memory_improves_shelter_return": {
                "status": "skipped",
                "passed": False,
                "scaffold_support_level": "scaffolded_runtime",
                "benchmark_of_record_eligible": False,
            },
            "noise_preserves_threat_valence": {
                "status": "passed",
                "passed": True,
                "scaffold_support_level": "minimal_manual",
                "benchmark_of_record_eligible": True,
            },
            "specialization_emerges_with_multiple_predators": {
                "status": "passed",
                "passed": True,
                "scaffold_support_level": "minimal_manual",
                "benchmark_of_record_eligible": True,
            },
        }

        summary = build_claim_test_summary(claim_results)

        self.assertEqual(summary["claims_passed"], 3)
        self.assertEqual(summary["claims_failed"], 1)
        self.assertEqual(summary["claims_skipped"], 1)
        self.assertEqual(summary["claims_at_minimal_manual"], 3)
        self.assertEqual(summary["claims_at_standard_constrained"], 1)
        self.assertEqual(summary["claims_at_scaffolded_runtime"], 1)
        self.assertEqual(summary["benchmark_of_record_claims"], 3)
        self.assertFalse(summary["all_primary_claims_passed"])
        self.assertFalse(summary["all_primary_claims_benchmark_eligible"])

    def test_benchmark_of_record_claims_counts_only_passing_claims(self) -> None:
        claim_results = {
            "learning_without_privileged_signals": {
                "status": "passed",
                "passed": True,
                "scaffold_support_level": "minimal_manual",
                "benchmark_of_record_eligible": True,
            },
            "escape_without_reflex_support": {
                "status": "failed",
                "passed": False,
                "scaffold_support_level": "minimal_manual",
                "benchmark_of_record_eligible": True,
            },
        }

        summary = build_claim_test_summary(claim_results)

        self.assertEqual(summary["benchmark_of_record_claims"], 1)

    def test_primary_benchmark_eligible_requires_all_executed_primary_claims_to_pass(self) -> None:
        claim_results = {
            "learning_without_privileged_signals": {
                "status": "passed",
                "passed": True,
                "scaffold_support_level": "minimal_manual",
                "benchmark_of_record_eligible": True,
            },
            "escape_without_reflex_support": {
                "status": "failed",
                "passed": False,
                "scaffold_support_level": "minimal_manual",
                "benchmark_of_record_eligible": True,
            },
        }

        summary = build_claim_test_summary(claim_results)

        self.assertFalse(summary["all_primary_claims_benchmark_eligible"])

    def test_all_primary_claims_passed_true_when_all_primary_claims_pass(self) -> None:
        claim_results = {
            "learning_without_privileged_signals": {
                "status": "passed",
                "passed": True,
                "scaffold_support_level": "minimal_manual",
                "benchmark_of_record_eligible": True,
            },
            "escape_without_reflex_support": {
                "status": "passed",
                "passed": True,
                "scaffold_support_level": "minimal_manual",
                "benchmark_of_record_eligible": True,
            },
            "memory_improves_shelter_return": {
                "status": "failed",
                "passed": False,
                "scaffold_support_level": "standard_constrained",
                "benchmark_of_record_eligible": False,
            },
            "noise_preserves_threat_valence": {
                "status": "passed",
                "passed": True,
                "scaffold_support_level": "minimal_manual",
                "benchmark_of_record_eligible": True,
            },
            "specialization_emerges_with_multiple_predators": {
                "status": "passed",
                "passed": True,
                "scaffold_support_level": "minimal_manual",
                "benchmark_of_record_eligible": True,
            },
        }

        summary = build_claim_test_summary(claim_results)

        self.assertTrue(summary["all_primary_claims_passed"])
        self.assertTrue(summary["all_primary_claims_benchmark_eligible"])

    def test_partial_primary_run_only_checks_executed_primary_claims(self) -> None:
        claim_results = {
            "learning_without_privileged_signals": {
                "status": "passed",
                "passed": True,
                "scaffold_support_level": "minimal_manual",
                "benchmark_of_record_eligible": True,
            },
            "memory_improves_shelter_return": {
                "status": "failed",
                "passed": False,
                "scaffold_support_level": "standard_constrained",
                "benchmark_of_record_eligible": False,
            },
        }

        summary = build_claim_test_summary(claim_results)

        self.assertTrue(summary["all_primary_claims_passed"])
        self.assertTrue(summary["all_primary_claims_benchmark_eligible"])

    def test_no_primary_claims_executed_sets_primary_gate_false(self) -> None:
        claim_results = {
            "memory_improves_shelter_return": {
                "status": "passed",
                "passed": True,
                "scaffold_support_level": "minimal_manual",
                "benchmark_of_record_eligible": True,
            },
            "noise_preserves_threat_valence": {
                "status": "passed",
                "passed": True,
                "scaffold_support_level": "minimal_manual",
                "benchmark_of_record_eligible": True,
            },
        }

        summary = build_claim_test_summary(claim_results)

        self.assertFalse(summary["all_primary_claims_passed"])
        self.assertFalse(summary["all_primary_claims_benchmark_eligible"])

    def test_primary_claim_benchmark_eligibility_requires_minimal_manual_support(self) -> None:
        claim_results = {
            "learning_without_privileged_signals": {
                "status": "passed",
                "passed": True,
                "scaffold_support_level": "standard_constrained",
                "benchmark_of_record_eligible": False,
            },
            "escape_without_reflex_support": {
                "status": "failed",
                "passed": False,
                "scaffold_support_level": "scaffolded_runtime",
                "benchmark_of_record_eligible": False,
            },
            "specialization_emerges_with_multiple_predators": {
                "status": "passed",
                "passed": True,
                "scaffold_support_level": "minimal_manual",
                "benchmark_of_record_eligible": True,
            },
        }

        summary = build_claim_test_summary(claim_results)

        self.assertEqual(summary["claims_at_minimal_manual"], 1)
        self.assertEqual(summary["claims_at_standard_constrained"], 1)
        self.assertEqual(summary["claims_at_scaffolded_runtime"], 1)
        self.assertEqual(summary["benchmark_of_record_claims"], 1)
        self.assertFalse(summary["all_primary_claims_benchmark_eligible"])

class RunClaimTestSuiteTest(unittest.TestCase):
    def test_rows_include_scaffold_metadata_columns(self) -> None:
        spec = resolve_claim_tests(["learning_without_privileged_signals"])[0]
        learning_evidence_payload = {
            "conditions": {
                "random_init": _behavior_payload(
                    spec.scenarios,
                    {"night_rest"},
                    aggregate_metrics={
                        "config": _scaffold_config_summary(warm_start_scale=0.5),
                    },
                ),
                "trained_without_reflex_support": _behavior_payload(
                    spec.scenarios,
                    set(spec.scenarios),
                    aggregate_metrics={"eval_reflex_scale": 0.0},
                ),
            }
        }

        payload, rows = run_claim_test_suite(
            claim_tests=[spec.name],
            learning_evidence_payload=learning_evidence_payload,
            austere_survival_payload=_austere_payload(
                spec.scenarios,
                set(spec.scenarios),
            ),
        )

        self.assertEqual(payload["claims"][spec.name]["claim_severity"], "qualified")
        self.assertEqual(len(rows), 1)
        self.assertEqual(
            rows[0]["claim_test_scaffold_support_level"],
            "standard_constrained",
        )
        self.assertEqual(
            json.loads(rows[0]["claim_test_scaffold_findings"]),
            ["warm_start_prior_active"],
        )
        self.assertFalse(rows[0]["claim_test_benchmark_of_record_eligible"])
        self.assertEqual(rows[0]["claim_test_severity"], "qualified")

    def test_rows_serialize_structured_reference_values_as_json(self) -> None:
        spec = resolve_claim_tests(
            ["specialization_emerges_with_multiple_predators"]
        )[0]
        ablation_payload = {
            "variants": {
                "modular_full": _behavior_payload(
                    spec.scenarios,
                    set(spec.scenarios),
                    check_pass_rates={
                        "visual_olfactory_pincer": {
                            "type_specific_response": 1.0,
                        },
                        "olfactory_ambush": {
                            "sensory_cortex_engaged": 1.0,
                        },
                        "visual_hunter_open_field": {
                            "visual_cortex_engaged": 1.0,
                        },
                    },
                    aggregate_metrics={
                        "config": _scaffold_config_summary(),
                        **_representation_metrics(score=0.15),
                    },
                ),
                "drop_visual_cortex": _behavior_payload(
                    spec.scenarios,
                    {"olfactory_ambush"},
                ),
                "drop_sensory_cortex": _behavior_payload(
                    spec.scenarios,
                    {"visual_hunter_open_field"},
                ),
            }
        }

        payload, rows = run_claim_test_suite(
            claim_tests=[spec.name],
            ablation_payload=ablation_payload,
            austere_survival_payload=_austere_payload(
                spec.scenarios,
                set(spec.scenarios),
            ),
        )

        self.assertEqual(payload["claims"][spec.name]["status"], "passed")
        self.assertEqual(len(rows), 1)
        reference_value = json.loads(rows[0]["claim_test_reference_value"])
        self.assertEqual(
            reference_value["type_specific_cortex_engagement_count"],
            3,
        )
        self.assertIn(
            "visual_minus_olfactory_success_rate",
            reference_value,
        )
        self.assertAlmostEqual(
            reference_value["representation_specialization_score"],
            0.15,
        )
        self.assertTrue(reference_value["representation_tier_passed"])
        self.assertTrue(rows[0]["claim_test_austere_survival_required"])
        self.assertTrue(rows[0]["claim_test_austere_survival_passed"])
        austere_gate = json.loads(rows[0]["claim_test_austere_survival_gate"])
        self.assertTrue(austere_gate["passed"])
        self.assertTrue(payload["metadata"]["sources"]["austere_survival"]["reused"])
        self.assertIn("austere_survival", payload["metadata"]["required_sources"])

    def test_reused_payload_metadata_coerces_invalid_sequences_to_empty_lists(self) -> None:
        spec = resolve_claim_tests(["memory_improves_shelter_return"])[0]
        ablation_payload = {
            "variants": {
                "modular_full": _behavior_payload(
                    spec.scenarios,
                    {"night_rest"},
                ),
                "modular_recurrent": _behavior_payload(
                    spec.scenarios,
                    set(spec.scenarios),
                ),
            },
            "seeds": None,
            "scenario_names": "night_rest",
        }

        payload, _ = run_claim_test_suite(
            claim_tests=[spec.name],
            ablation_payload=ablation_payload,
        )

        source_metadata = payload["metadata"]["sources"]["ablation"]
        self.assertEqual(source_metadata["seeds"], [])
        self.assertEqual(source_metadata["scenario_names"], [])
        self.assertEqual(payload["metadata"]["seeds"], [])

    def test_reused_non_dict_payload_metadata_is_ignored_safely(self) -> None:
        spec = resolve_claim_tests(["memory_improves_shelter_return"])[0]

        payload, rows = run_claim_test_suite(
            claim_tests=[spec.name],
            ablation_payload="not_a_payload",  # type: ignore[arg-type]
        )

        source_metadata = payload["metadata"]["sources"]["ablation"]
        self.assertEqual(source_metadata["seeds"], [])
        self.assertEqual(source_metadata["scenario_names"], [])
        self.assertIsNone(source_metadata["budget_profile"])
        self.assertEqual(payload["metadata"]["seeds"], [])
        self.assertEqual(payload["metadata"]["noise_profiles"], {})
        self.assertEqual(rows[0]["claim_test_status"], "skipped")

class ClaimSubsetScenarioSuccessRateTest(unittest.TestCase):
    def test_uses_actual_success_rates_not_binary_pass_flags(self) -> None:
        score, reason = claim_subset_scenario_success_rate(
            {
                "suite": {
                    "a": {"success_rate": 0.25},
                    "b": {"success_rate": 0.75},
                }
            },
            scenarios=("a", "b"),
        )

        self.assertIsNone(reason)
        self.assertEqual(score, 0.5)

    def test_none_payload_returns_error(self) -> None:
        score, reason = claim_subset_scenario_success_rate(None, scenarios=("a",))
        self.assertIsNone(score)
        self.assertIsNotNone(reason)

    def test_missing_scenario_returns_error(self) -> None:
        score, reason = claim_subset_scenario_success_rate(
            {"suite": {"a": {"success_rate": 1.0}}},
            scenarios=("a", "missing_scenario"),
        )
        self.assertIsNone(score)
        self.assertIn("missing_scenario", reason)

    def test_empty_scenarios_returns_zero(self) -> None:
        score, reason = claim_subset_scenario_success_rate(
            {"suite": {"a": {"success_rate": 1.0}}},
            scenarios=(),
        )
        self.assertEqual(score, 0.0)
        self.assertIsNone(reason)

    def test_single_scenario_returns_its_rate(self) -> None:
        score, reason = claim_subset_scenario_success_rate(
            {"suite": {"a": {"success_rate": 0.6}}},
            scenarios=("a",),
        )
        self.assertIsNone(reason)
        self.assertAlmostEqual(score, 0.6)

    def test_non_dict_suite_returns_error(self) -> None:
        score, reason = claim_subset_scenario_success_rate(
            {"suite": "not_a_dict"},
            scenarios=("a",),
        )
        self.assertIsNone(score)
        self.assertIsNotNone(reason)

    def test_non_mapping_scenario_entry_is_reported_missing(self) -> None:
        score, reason = claim_subset_scenario_success_rate(
            {
                "suite": {
                    "a": "not_a_mapping",
                    "b": {"success_rate": 0.75},
                }
            },
            scenarios=("a", "b"),
        )

        self.assertIsNone(score)
        self.assertEqual(reason, "Missing required scenarios: ['a'].")

class ClaimPayloadConfigSummaryTest(unittest.TestCase):
    """Tests for claim_payload_config_summary()."""

    def test_extracts_config_dict(self) -> None:
        payload = {"config": {"use_learned_arbitration": True, "warm_start_scale": 0.0}}
        result = claim_payload_config_summary(payload)
        self.assertEqual(result["use_learned_arbitration"], True)
        self.assertEqual(result["warm_start_scale"], 0.0)

    def test_none_payload_returns_empty_dict(self) -> None:
        result = claim_payload_config_summary(None)
        self.assertEqual(result, {})

    def test_non_dict_payload_returns_empty_dict(self) -> None:
        result = claim_payload_config_summary("not_a_dict")
        self.assertEqual(result, {})

    def test_missing_config_key_returns_empty_dict(self) -> None:
        result = claim_payload_config_summary({"suite": {}})
        self.assertEqual(result, {})

    def test_non_dict_config_returns_empty_dict(self) -> None:
        result = claim_payload_config_summary({"config": "not_a_dict"})
        self.assertEqual(result, {})

    def test_returns_shallow_copy(self) -> None:
        original_config = {"key": "value"}
        payload = {"config": original_config}
        result = claim_payload_config_summary(payload)
        self.assertIsNot(result, original_config)

class ClaimPayloadEvalReflexScaleTest(unittest.TestCase):
    """Tests for claim_payload_eval_reflex_scale()."""

    def test_extracts_top_level_eval_reflex_scale(self) -> None:
        payload = {"eval_reflex_scale": 0.5}
        result = claim_payload_eval_reflex_scale(payload)
        self.assertAlmostEqual(result, 0.5)

    def test_falls_back_to_summary_eval_reflex_scale(self) -> None:
        payload = {"summary": {"eval_reflex_scale": 0.3}}
        result = claim_payload_eval_reflex_scale(payload)
        self.assertAlmostEqual(result, 0.3)

    def test_top_level_takes_precedence_over_summary(self) -> None:
        payload = {
            "eval_reflex_scale": 0.7,
            "summary": {"eval_reflex_scale": 0.3},
        }
        result = claim_payload_eval_reflex_scale(payload)
        self.assertAlmostEqual(result, 0.7)

    def test_none_payload_returns_none(self) -> None:
        result = claim_payload_eval_reflex_scale(None)
        self.assertIsNone(result)

    def test_non_dict_payload_returns_none(self) -> None:
        result = claim_payload_eval_reflex_scale("not_a_dict")
        self.assertIsNone(result)

    def test_missing_eval_reflex_scale_returns_none(self) -> None:
        result = claim_payload_eval_reflex_scale({"config": {}})
        self.assertIsNone(result)

    def test_zero_value_is_returned(self) -> None:
        result = claim_payload_eval_reflex_scale({"eval_reflex_scale": 0.0})
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result, 0.0)

    def test_inf_value_returns_none(self) -> None:
        result = claim_payload_eval_reflex_scale({"eval_reflex_scale": float("inf")})
        self.assertIsNone(result)

    def test_nan_value_returns_none(self) -> None:
        result = claim_payload_eval_reflex_scale({"eval_reflex_scale": float("nan")})
        self.assertIsNone(result)

    def test_string_numeric_value_is_parsed(self) -> None:
        result = claim_payload_eval_reflex_scale({"eval_reflex_scale": "0.25"})
        self.assertAlmostEqual(result, 0.25)

class ClaimLeakageAuditSummaryTest(unittest.TestCase):
    """Tests for claim_leakage_audit_summary()."""

    def test_returns_dict_with_finding_count_and_findings(self) -> None:
        result = claim_leakage_audit_summary()
        self.assertIn("finding_count", result)
        self.assertIn("findings", result)

    def test_finding_count_is_integer(self) -> None:
        result = claim_leakage_audit_summary()
        self.assertIsInstance(result["finding_count"], int)

    def test_findings_is_list(self) -> None:
        result = claim_leakage_audit_summary()
        self.assertIsInstance(result["findings"], list)

    def test_finding_count_matches_findings_length(self) -> None:
        result = claim_leakage_audit_summary()
        self.assertEqual(result["finding_count"], len(result["findings"]))

    def test_findings_formatted_as_audit_colon_signal(self) -> None:
        result = claim_leakage_audit_summary()
        for finding in result["findings"]:
            self.assertIn(":", finding, msg=f"Finding {finding!r} missing colon separator")

    def test_result_is_deterministic(self) -> None:
        result1 = claim_leakage_audit_summary()
        result2 = claim_leakage_audit_summary()
        self.assertEqual(result1["finding_count"], result2["finding_count"])
        self.assertEqual(result1["findings"], result2["findings"])

class ClaimNoiseSubsetScoresTest(unittest.TestCase):
    """Tests for claim_noise_subset_scores()."""

    def _make_cell_payload(self, success_rate: float) -> dict:
        return {"suite": {"test_scenario": {"success_rate": success_rate}}}

    def test_none_payload_returns_error(self) -> None:
        diag, off_diag, reason = claim_noise_subset_scores(None, scenarios=("test_scenario",))
        self.assertIsNone(diag)
        self.assertIsNone(off_diag)
        self.assertIsNotNone(reason)

    def test_missing_matrix_returns_error(self) -> None:
        payload = {"scenario_names": ["test_scenario"]}
        diag, off_diag, reason = claim_noise_subset_scores(
            payload, scenarios=("test_scenario",)
        )
        self.assertIsNone(diag)
        self.assertIsNone(off_diag)
        self.assertIsNotNone(reason)

    def test_diagonal_and_off_diagonal_computed_correctly(self) -> None:
        payload = {
            "matrix": {
                "none": {
                    "none": self._make_cell_payload(0.9),
                    "low": self._make_cell_payload(0.7),
                },
                "low": {
                    "none": self._make_cell_payload(0.6),
                    "low": self._make_cell_payload(0.8),
                },
            }
        }
        diag, off_diag, reason = claim_noise_subset_scores(
            payload, scenarios=("test_scenario",)
        )
        self.assertIsNone(reason)
        self.assertAlmostEqual(diag, (0.9 + 0.8) / 2, places=5)
        self.assertAlmostEqual(off_diag, (0.7 + 0.6) / 2, places=5)

    def test_no_diagonal_cells_returns_error(self) -> None:
        payload = {
            "matrix": {
                "none": {"low": self._make_cell_payload(0.7)},
                "low": {"none": self._make_cell_payload(0.6)},
            }
        }
        diag, off_diag, reason = claim_noise_subset_scores(
            payload, scenarios=("test_scenario",)
        )
        self.assertIsNone(diag)
        self.assertIsNone(off_diag)
        self.assertIsNotNone(reason)

    def test_malformed_row_returns_error(self) -> None:
        payload = {
            "matrix": {
                "none": "not_a_dict",
            }
        }
        diag, off_diag, reason = claim_noise_subset_scores(
            payload, scenarios=("test_scenario",)
        )
        self.assertIsNone(diag)
        self.assertIsNone(off_diag)
        self.assertIsNotNone(reason)
        self.assertIn("none", reason)

    def test_returns_rounded_values(self) -> None:
        payload = {
            "matrix": {
                "cond": {
                    "cond": self._make_cell_payload(1.0 / 3.0),
                    "other": self._make_cell_payload(2.0 / 3.0),
                },
                "other": {
                    "cond": self._make_cell_payload(2.0 / 3.0),
                    "other": self._make_cell_payload(1.0 / 3.0),
                },
            }
        }
        diag, off_diag, reason = claim_noise_subset_scores(
            payload, scenarios=("test_scenario",)
        )
        self.assertIsNone(reason)
        # Values should be rounded to 6 decimal places
        self.assertEqual(diag, round(diag, 6))
        self.assertEqual(off_diag, round(off_diag, 6))
