import json
import unittest

from spider_cortex_sim.claim_tests import (
    ClaimTestSpec,
    canonical_claim_tests,
    claim_test_names,
    primary_claim_test_names,
    resolve_claim_tests,
)
from spider_cortex_sim.simulation import SpiderSimulation


def _behavior_payload(
    scenarios: tuple[str, ...],
    passed_scenarios: set[str],
    *,
    check_pass_rates: dict[str, dict[str, float]] | None = None,
) -> dict[str, object]:
    """
    Create a synthetic behavior payload mapping each scenario to a success rate and optional per-check pass rates.
    
    Parameters:
        scenarios (tuple[str, ...]): Ordered sequence of scenario names to include in the payload.
        passed_scenarios (set[str]): Set of scenario names considered successful (will receive success_rate 1.0).
        check_pass_rates (dict[str, dict[str, float]] | None): Optional mapping from scenario name to a mapping of
            check name -> pass rate (float between 0.0 and 1.0). When provided, each check is included under the
            scenario's "checks" key as {"check_name": {"pass_rate": <float>}}.
    
    Returns:
        dict[str, object]: A payload containing:
            - "suite": dict mapping each scenario name to an object with:
                - "success_rate": 1.0 for scenarios in `passed_scenarios`, 0.0 otherwise.
                - optional "checks": mapping of check names to {"pass_rate": float} when `check_pass_rates` is supplied.
            - "summary": object with "scenario_success_rate": the average success rate across `scenarios`.
    """
    suite: dict[str, object] = {}
    for scenario_name in scenarios:
        suite_item: dict[str, object] = {
            "success_rate": 1.0 if scenario_name in passed_scenarios else 0.0,
        }
        scenario_checks = (check_pass_rates or {}).get(scenario_name)
        if scenario_checks is not None:
            suite_item["checks"] = {
                check_name: {"pass_rate": pass_rate}
                for check_name, pass_rate in scenario_checks.items()
            }
        suite[scenario_name] = suite_item
    return {
        "suite": suite,
        "summary": {
            "scenario_success_rate": (
                sum(1.0 if name in passed_scenarios else 0.0 for name in scenarios)
                / max(1, len(scenarios))
            ),
        },
    }


class ClaimTestSpecTest(unittest.TestCase):
    def test_dataclass_fields_are_accessible(self) -> None:
        spec = ClaimTestSpec(
            name="x",
            hypothesis="h",
            description="d",
            protocol="p",
            reference_condition="r",
            comparison_conditions=("c",),
            primary_metric="m",
            success_criterion="s",
            effect_size_metric="e",
            scenarios=("scenario",),
            primary=True,
        )
        self.assertEqual(spec.name, "x")
        self.assertEqual(spec.hypothesis, "h")
        self.assertEqual(spec.description, "d")
        self.assertEqual(spec.protocol, "p")
        self.assertEqual(spec.reference_condition, "r")
        self.assertEqual(spec.comparison_conditions, ("c",))
        self.assertEqual(spec.primary_metric, "m")
        self.assertEqual(spec.success_criterion, "s")
        self.assertEqual(spec.effect_size_metric, "e")
        self.assertEqual(spec.scenarios, ("scenario",))
        self.assertTrue(spec.primary)

    def test_frozen_dataclass_is_immutable(self) -> None:
        spec = ClaimTestSpec(
            name="x",
            hypothesis="h",
            description="d",
            protocol="p",
            reference_condition="r",
            comparison_conditions=("c",),
            primary_metric="m",
            success_criterion="s",
            effect_size_metric="e",
            scenarios=("scenario",),
        )
        with self.assertRaises((AttributeError, TypeError)):
            spec.name = "z"  # type: ignore[misc]

    def test_sequence_fields_are_stored_as_tuples_of_strings(self) -> None:
        spec = ClaimTestSpec(
            name="x",
            hypothesis="h",
            description="d",
            protocol="p",
            reference_condition="r",
            comparison_conditions=["one", "two"],  # type: ignore[arg-type]
            primary_metric="m",
            success_criterion="s",
            effect_size_metric="e",
            scenarios=["a", "b"],  # type: ignore[arg-type]
        )
        self.assertEqual(spec.comparison_conditions, ("one", "two"))
        self.assertEqual(spec.scenarios, ("a", "b"))


class CanonicalClaimTestsTest(unittest.TestCase):
    EXPECTED_NAMES = (
        "learning_without_privileged_signals",
        "escape_without_reflex_support",
        "memory_improves_shelter_return",
        "noise_preserves_threat_valence",
        "specialization_emerges_with_multiple_predators",
    )

    def test_returns_exactly_five_specs(self) -> None:
        specs = canonical_claim_tests()
        self.assertEqual(len(specs), 5)
        self.assertEqual(tuple(spec.name for spec in specs), self.EXPECTED_NAMES)

    def test_returns_claim_test_specs(self) -> None:
        for spec in canonical_claim_tests():
            self.assertIsInstance(spec, ClaimTestSpec)

    def test_learning_without_privileged_signals_uses_learning_evidence_reference(self) -> None:
        spec = canonical_claim_tests()[0]
        self.assertEqual(spec.reference_condition, "random_init")
        self.assertEqual(spec.comparison_conditions, ("trained_without_reflex_support",))
        self.assertEqual(spec.primary_metric, "scenario_success_rate")
        self.assertIn("leakage audit", spec.protocol.lower())
        self.assertIn("0.15", spec.success_criterion)

    def test_escape_without_reflex_support_uses_predator_scenarios(self) -> None:
        spec = canonical_claim_tests()[1]
        self.assertEqual(spec.reference_condition, "reflex_only")
        self.assertEqual(spec.comparison_conditions, ("trained_without_reflex_support",))
        self.assertEqual(
            spec.scenarios,
            ("predator_edge", "entrance_ambush", "shelter_blockade"),
        )
        self.assertEqual(spec.primary_metric, "predator_response_scenario_success_rate")

    def test_memory_improves_shelter_return_uses_ablation_pair(self) -> None:
        spec = canonical_claim_tests()[2]
        self.assertEqual(spec.reference_condition, "modular_full")
        self.assertEqual(spec.comparison_conditions, ("modular_recurrent",))
        self.assertEqual(spec.scenarios, ("night_rest", "two_shelter_tradeoff"))
        self.assertIn("0.10", spec.success_criterion)

    def test_noise_preserves_threat_valence_uses_diagonal_vs_off_diagonal(self) -> None:
        spec = canonical_claim_tests()[3]
        self.assertEqual(spec.reference_condition, "diagonal")
        self.assertEqual(spec.comparison_conditions, ("off_diagonal",))
        self.assertEqual(spec.effect_size_metric, "diagonal_minus_off_diagonal_score")
        self.assertIn("0.60", spec.success_criterion)
        self.assertIn("0.15", spec.success_criterion)

    def test_specialization_emerges_with_multiple_predators_uses_predator_type_ablation(self) -> None:
        spec = canonical_claim_tests()[4]
        self.assertEqual(spec.reference_condition, "modular_full")
        self.assertEqual(
            spec.comparison_conditions,
            ("drop_visual_cortex", "drop_sensory_cortex"),
        )
        self.assertEqual(
            spec.scenarios,
            (
                "visual_olfactory_pincer",
                "olfactory_ambush",
                "visual_hunter_open_field",
            ),
        )
        self.assertEqual(spec.primary_metric, "visual_minus_olfactory_success_rate")

    def test_returns_new_list_each_call(self) -> None:
        self.assertIsNot(canonical_claim_tests(), canonical_claim_tests())

    def test_primary_claims_are_marked_in_registry(self) -> None:
        self.assertEqual(
            primary_claim_test_names(),
            (
                "learning_without_privileged_signals",
                "escape_without_reflex_support",
                "specialization_emerges_with_multiple_predators",
            ),
        )


class ClaimTestNamesTest(unittest.TestCase):
    def test_returns_list(self) -> None:
        names = claim_test_names()
        self.assertIsInstance(names, list)

    def test_matches_canonical_spec_order(self) -> None:
        self.assertEqual(claim_test_names(), [spec.name for spec in canonical_claim_tests()])


class ResolveClaimTestsTest(unittest.TestCase):
    def test_none_returns_all_claim_tests_in_order(self) -> None:
        specs = resolve_claim_tests(None)
        self.assertEqual([spec.name for spec in specs], claim_test_names())

    def test_empty_sequence_returns_empty(self) -> None:
        self.assertEqual(resolve_claim_tests([]), [])

    def test_multiple_names_preserve_order(self) -> None:
        names = [
            "specialization_emerges_with_multiple_predators",
            "learning_without_privileged_signals",
        ]
        specs = resolve_claim_tests(names)
        self.assertEqual([spec.name for spec in specs], names)

    def test_duplicate_names_return_duplicate_specs(self) -> None:
        specs = resolve_claim_tests(
            ["memory_improves_shelter_return", "memory_improves_shelter_return"]
        )
        self.assertEqual([spec.name for spec in specs], [
            "memory_improves_shelter_return",
            "memory_improves_shelter_return",
        ])

    def test_invalid_name_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            resolve_claim_tests(["not_a_claim_test"])

    def test_invalid_name_message_contains_requested_name(self) -> None:
        with self.assertRaisesRegex(ValueError, "not_a_claim_test"):
            resolve_claim_tests(["not_a_claim_test"])

    def test_names_are_coerced_to_str(self) -> None:
        class StrLike:
            def __str__(self) -> str:
                """Return canonical claim test name."""
                return "noise_preserves_threat_valence"

        specs = resolve_claim_tests([StrLike()])  # type: ignore[list-item]
        self.assertEqual([spec.name for spec in specs], ["noise_preserves_threat_valence"])


class EvaluateClaimTestTest(unittest.TestCase):
    def test_passing_case_uses_synthetic_learning_payload(self) -> None:
        spec = resolve_claim_tests(["learning_without_privileged_signals"])[0]
        payloads = {
            "learning_evidence": {
                "conditions": {
                    "random_init": _behavior_payload(
                        spec.scenarios,
                        {"night_rest"},
                    ),
                    "trained_without_reflex_support": _behavior_payload(
                        spec.scenarios,
                        {
                            "night_rest",
                            "predator_edge",
                            "entrance_ambush",
                            "shelter_blockade",
                        },
                    ),
                }
            }
        }

        result = SpiderSimulation._evaluate_claim_test(spec, payloads)

        self.assertEqual(result["status"], "passed")
        self.assertTrue(result["passed"])
        self.assertAlmostEqual(result["reference_value"], 0.2)
        self.assertAlmostEqual(
            result["comparison_values"]["trained_without_reflex_support"],
            0.8,
        )
        self.assertAlmostEqual(
            result["delta"]["trained_without_reflex_support"],
            0.6,
        )

    def test_failing_case_uses_synthetic_learning_payload(self) -> None:
        spec = resolve_claim_tests(["escape_without_reflex_support"])[0]
        payloads = {
            "learning_evidence": {
                "conditions": {
                    "reflex_only": _behavior_payload(spec.scenarios, set()),
                    "trained_without_reflex_support": _behavior_payload(
                        spec.scenarios,
                        {"predator_edge"},
                    ),
                }
            }
        }

        result = SpiderSimulation._evaluate_claim_test(spec, payloads)

        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["passed"])
        self.assertAlmostEqual(
            result["comparison_values"]["trained_without_reflex_support"],
            1.0 / 3.0,
            places=6,
        )
        self.assertAlmostEqual(
            result["delta"]["trained_without_reflex_support"],
            1.0 / 3.0,
            places=6,
        )

    def test_missing_condition_returns_skipped_result(self) -> None:
        spec = resolve_claim_tests(["learning_without_privileged_signals"])[0]
        payloads = {
            "learning_evidence": {
                "conditions": {
                    "random_init": _behavior_payload(spec.scenarios, {"night_rest"}),
                }
            }
        }

        result = SpiderSimulation._evaluate_claim_test(spec, payloads)

        self.assertEqual(result["status"], "skipped")
        self.assertIn("trained_without_reflex_support", result["reason"])

    def test_zero_delta_returns_failed_result(self) -> None:
        """
        Verifies that an evaluated claim test fails when ablation variants produce no improvement (zero delta).
        
        Asserts the returned result has status "failed", passed is False, and the computed delta for the "modular_recurrent" variant is 0.0.
        """
        spec = resolve_claim_tests(["memory_improves_shelter_return"])[0]
        shared_payload = _behavior_payload(
            spec.scenarios,
            {"night_rest"},
        )
        payloads = {
            "ablation": {
                "variants": {
                    "modular_full": shared_payload,
                    "modular_recurrent": shared_payload,
                }
            }
        }

        result = SpiderSimulation._evaluate_claim_test(spec, payloads)

        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["passed"])
        self.assertEqual(result["delta"]["modular_recurrent"], 0.0)

    def test_missing_scenario_key_returns_skipped_result(self) -> None:
        """
        Verifies that evaluating a claim test returns a skipped result when a payload omits a required scenario.
        
        Constructs a payload where one condition's suite is missing a scenario from the claim spec, calls SpiderSimulation._evaluate_claim_test, and asserts the result has status "skipped" and the reason contains "Missing required scenarios".
        """
        spec = resolve_claim_tests(["learning_without_privileged_signals"])[0]
        incomplete_scenarios = tuple(spec.scenarios[:-1])
        payloads = {
            "learning_evidence": {
                "conditions": {
                    "random_init": _behavior_payload(
                        spec.scenarios,
                        {"night_rest"},
                    ),
                    "trained_without_reflex_support": _behavior_payload(
                        incomplete_scenarios,
                        {
                            "night_rest",
                            "predator_edge",
                            "entrance_ambush",
                        },
                    ),
                }
            }
        }

        result = SpiderSimulation._evaluate_claim_test(spec, payloads)

        self.assertEqual(result["status"], "skipped")
        self.assertIn("Missing required scenarios", result["reason"])


class BuildClaimTestSummaryTest(unittest.TestCase):
    def test_aggregation_counts_passed_failed_and_skipped(self) -> None:
        claim_results = {
            "learning_without_privileged_signals": {"status": "passed", "passed": True},
            "escape_without_reflex_support": {"status": "failed", "passed": False},
            "memory_improves_shelter_return": {"status": "skipped", "passed": False},
            "noise_preserves_threat_valence": {"status": "passed", "passed": True},
            "specialization_emerges_with_multiple_predators": {
                "status": "passed",
                "passed": True,
            },
        }

        summary = SpiderSimulation._build_claim_test_summary(claim_results)

        self.assertEqual(summary["claims_passed"], 3)
        self.assertEqual(summary["claims_failed"], 1)
        self.assertEqual(summary["claims_skipped"], 1)
        self.assertFalse(summary["all_primary_claims_passed"])

    def test_all_primary_claims_passed_true_when_all_primary_claims_pass(self) -> None:
        claim_results = {
            "learning_without_privileged_signals": {"status": "passed", "passed": True},
            "escape_without_reflex_support": {"status": "passed", "passed": True},
            "memory_improves_shelter_return": {"status": "failed", "passed": False},
            "noise_preserves_threat_valence": {"status": "passed", "passed": True},
            "specialization_emerges_with_multiple_predators": {
                "status": "passed",
                "passed": True,
            },
        }

        summary = SpiderSimulation._build_claim_test_summary(claim_results)

        self.assertTrue(summary["all_primary_claims_passed"])

    def test_partial_primary_run_only_checks_executed_primary_claims(self) -> None:
        claim_results = {
            "learning_without_privileged_signals": {"status": "passed", "passed": True},
            "memory_improves_shelter_return": {"status": "failed", "passed": False},
        }

        summary = SpiderSimulation._build_claim_test_summary(claim_results)

        self.assertTrue(summary["all_primary_claims_passed"])

    def test_no_primary_claims_executed_sets_primary_gate_false(self) -> None:
        claim_results = {
            "memory_improves_shelter_return": {"status": "passed", "passed": True},
            "noise_preserves_threat_valence": {"status": "passed", "passed": True},
        }

        summary = SpiderSimulation._build_claim_test_summary(claim_results)

        self.assertFalse(summary["all_primary_claims_passed"])


class RunClaimTestSuiteTest(unittest.TestCase):
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

        payload, rows = SpiderSimulation.run_claim_test_suite(
            claim_tests=[spec.name],
            ablation_payload=ablation_payload,
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

        payload, _ = SpiderSimulation.run_claim_test_suite(
            claim_tests=[spec.name],
            ablation_payload=ablation_payload,
        )

        source_metadata = payload["metadata"]["sources"]["ablation"]
        self.assertEqual(source_metadata["seeds"], [])
        self.assertEqual(source_metadata["scenario_names"], [])
        self.assertEqual(payload["metadata"]["seeds"], [])


class ClaimSubsetScenarioSuccessRateTest(unittest.TestCase):
    def test_uses_actual_success_rates_not_binary_pass_flags(self) -> None:
        score, reason = SpiderSimulation._claim_subset_scenario_success_rate(
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
