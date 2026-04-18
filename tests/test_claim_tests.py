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


def _behavior_payload(
    scenarios: tuple[str, ...],
    passed_scenarios: set[str],
    *,
    check_pass_rates: dict[str, dict[str, float]] | None = None,
    aggregate_metrics: dict[str, object] | None = None,
) -> dict[str, object]:
    """
    Create a synthetic behavior payload mapping each scenario to a success rate and optional per-check pass rates.
    
    Parameters:
        scenarios (tuple[str, ...]): Ordered sequence of scenario names to include in the payload.
        passed_scenarios (set[str]): Set of scenario names considered successful (will receive success_rate 1.0).
        check_pass_rates (dict[str, dict[str, float]] | None): Optional mapping from scenario name to a mapping of
            check name -> pass rate (float between 0.0 and 1.0). When provided, each check is included under the
            scenario's "checks" key as {"check_name": {"pass_rate": <float>}}.
        aggregate_metrics (dict[str, object] | None): Optional top-level aggregate
            metrics merged into the payload. Tests use this to inject
            representation-specialization evidence the same way compact behavior
            payloads expose it in production.
    
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
    payload = {
        "suite": suite,
        "summary": {
            "scenario_success_rate": (
                sum(1.0 if name in passed_scenarios else 0.0 for name in scenarios)
                / max(1, len(scenarios))
            ),
        },
    }
    if aggregate_metrics:
        payload.update(dict(aggregate_metrics))
    return payload


def _representation_metrics(
    *,
    score: float,
    proposer_divergence: dict[str, float] | None = None,
    gate_differential: dict[str, float] | None = None,
    contribution_differential: dict[str, float] | None = None,
) -> dict[str, object]:
    return {
        "mean_proposer_divergence_by_module": proposer_divergence
        or {
            "visual_cortex": 0.18,
            "sensory_cortex": 0.12,
        },
        "mean_action_center_gate_differential": gate_differential
        or {
            "visual_cortex": 0.3,
            "sensory_cortex": -0.25,
        },
        "mean_action_center_contribution_differential": contribution_differential
        or {
            "visual_cortex": 0.2,
            "sensory_cortex": -0.15,
        },
        "mean_representation_specialization_score": score,
    }


def _scaffold_config_summary(
    *,
    use_learned_arbitration: bool = True,
    enable_deterministic_guards: bool = False,
    enable_food_direction_bias: bool = False,
    warm_start_scale: float = 0.0,
) -> dict[str, object]:
    return {
        "use_learned_arbitration": use_learned_arbitration,
        "enable_deterministic_guards": enable_deterministic_guards,
        "enable_food_direction_bias": enable_food_direction_bias,
        "warm_start_scale": warm_start_scale,
    }


def _add_seed_level_success_rates(
    payload: dict[str, object],
    *,
    condition: str,
    scenarios: tuple[str, ...],
    seed_values: dict[int, float],
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    suite = payload["suite"]
    assert isinstance(suite, dict)
    for scenario_name in scenarios:
        scenario_payload = suite[scenario_name]
        assert isinstance(scenario_payload, dict)
        scenario_rows = [
            {
                "metric_name": "scenario_success_rate",
                "seed": seed,
                "value": value,
                "condition": condition,
                "scenario": scenario_name,
            }
            for seed, value in seed_values.items()
        ]
        scenario_payload["seed_level"] = scenario_rows
        rows.extend(scenario_rows)
    payload["seed_level"] = rows
    return payload


def _austere_payload(
    scenarios: tuple[str, ...],
    surviving_scenarios: set[str],
) -> dict[str, object]:
    scenario_payloads = {
        scenario_name: {
            "austere_success_rate": 1.0
            if scenario_name in surviving_scenarios
            else 0.0,
            "survives": scenario_name in surviving_scenarios,
            "episodes": 1,
        }
        for scenario_name in scenarios
    }
    scenario_count = len(scenario_payloads)
    surviving_count = sum(
        1 for payload in scenario_payloads.values() if payload["survives"]
    )
    return {
        "scenario_names": list(scenarios),
        "episodes_per_scenario": 1,
        "reward_audit": {
            "comparison": {
                "minimal_profile": "austere",
                "behavior_survival": {
                    "available": True,
                    "minimal_profile": "austere",
                    "survival_threshold": 0.5,
                    "scenario_count": scenario_count,
                    "surviving_scenario_count": surviving_count,
                    "survival_rate": (
                        surviving_count / scenario_count if scenario_count else 0.0
                    ),
                    "scenarios": scenario_payloads,
                },
                "gap_policy_check": {"violations": [], "warnings": []},
            }
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
        self.assertFalse(spec.austere_survival_required)

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

    def test_austere_survival_required_is_coerced_to_bool(self) -> None:
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
            austere_survival_required=1,  # type: ignore[arg-type]
        )
        self.assertTrue(spec.austere_survival_required)


class ScaffoldAssessmentSerializationTest(unittest.TestCase):
    def test_to_dict_serializes_enum_and_findings(self) -> None:
        assessment = ScaffoldAssessment(
            support_level=ScaffoldSupportLevel.STANDARD_CONSTRAINED,
            findings=["warm_start_prior_active"],
            benchmark_of_record_eligible=False,
        )

        self.assertEqual(
            assessment.to_dict(),
            {
                "support_level": "standard_constrained",
                "findings": ["warm_start_prior_active"],
                "benchmark_of_record_eligible": False,
            },
        )

    def test_findings_are_normalized_to_immutable_tuple(self) -> None:
        assessment = ScaffoldAssessment(
            support_level=ScaffoldSupportLevel.STANDARD_CONSTRAINED,
            findings=["warm_start_prior_active"],
            benchmark_of_record_eligible=False,
        )

        self.assertEqual(assessment.findings, ("warm_start_prior_active",))


class ScaffoldAssessmentTest(unittest.TestCase):
    def test_assess_scaffold_support_returns_minimal_manual_when_all_scaffolds_disabled(self) -> None:
        assessment = assess_scaffold_support(
            {
                "enable_deterministic_guards": False,
                "enable_food_direction_bias": False,
                "use_learned_arbitration": True,
                "warm_start_scale": 0.0,
            },
            eval_reflex_scale=0.0,
        )

        self.assertEqual(
            assessment.support_level,
            ScaffoldSupportLevel.MINIMAL_MANUAL,
        )
        self.assertEqual(assessment.findings, ())
        self.assertTrue(assessment.benchmark_of_record_eligible)

    def test_returns_minimal_manual_at_threshold_without_runtime_support(self) -> None:
        assessment = assess_scaffold_support(
            {
                "enable_deterministic_guards": False,
                "enable_food_direction_bias": False,
                "use_learned_arbitration": True,
                "warm_start_scale": WARM_START_MINIMAL_THRESHOLD,
            },
            eval_reflex_scale=0.0,
        )

        self.assertEqual(
            assessment.support_level,
            ScaffoldSupportLevel.MINIMAL_MANUAL,
        )
        self.assertEqual(assessment.findings, ())
        self.assertTrue(assessment.benchmark_of_record_eligible)

    def test_assess_scaffold_support_returns_scaffolded_runtime_when_deterministic_guards_enabled(self) -> None:
        assessment = assess_scaffold_support(
            {
                "enable_deterministic_guards": True,
                "enable_food_direction_bias": False,
                "use_learned_arbitration": True,
                "warm_start_scale": 0.0,
            },
            eval_reflex_scale=0.0,
        )

        self.assertEqual(
            assessment.support_level,
            ScaffoldSupportLevel.SCAFFOLDED_RUNTIME,
        )
        self.assertEqual(assessment.findings, ("deterministic_guards_enabled",))

    def test_assess_scaffold_support_returns_scaffolded_runtime_when_food_direction_bias_enabled(self) -> None:
        assessment = assess_scaffold_support(
            {
                "enable_deterministic_guards": False,
                "enable_food_direction_bias": True,
                "use_learned_arbitration": True,
                "warm_start_scale": 0.0,
            },
            eval_reflex_scale=0.0,
        )

        self.assertEqual(
            assessment.support_level,
            ScaffoldSupportLevel.SCAFFOLDED_RUNTIME,
        )
        self.assertEqual(assessment.findings, ("food_direction_bias_enabled",))

    def test_returns_standard_constrained_for_warm_start_prior(self) -> None:
        assessment = assess_scaffold_support(
            {
                "use_learned_arbitration": True,
                "warm_start_scale": WARM_START_MINIMAL_THRESHOLD + 0.01,
            },
            eval_reflex_scale=0.0,
        )

        self.assertEqual(
            assessment.support_level,
            ScaffoldSupportLevel.STANDARD_CONSTRAINED,
        )
        self.assertEqual(assessment.findings, ("warm_start_prior_active",))
        self.assertFalse(assessment.benchmark_of_record_eligible)

    def test_assess_scaffold_support_returns_standard_constrained_when_warm_start_exceeds_threshold(self) -> None:
        assessment = assess_scaffold_support(
            {
                "enable_deterministic_guards": False,
                "enable_food_direction_bias": False,
                "use_learned_arbitration": True,
                "warm_start_scale": 0.3,
            },
            eval_reflex_scale=0.0,
        )

        self.assertEqual(
            assessment.support_level,
            ScaffoldSupportLevel.STANDARD_CONSTRAINED,
        )
        self.assertEqual(assessment.findings, ("warm_start_prior_active",))

    def test_returns_standard_constrained_for_fixed_arbitration(self) -> None:
        assessment = assess_scaffold_support(
            {
                "use_learned_arbitration": False,
                "warm_start_scale": 0.0,
            },
            eval_reflex_scale=None,
        )

        self.assertEqual(
            assessment.support_level,
            ScaffoldSupportLevel.STANDARD_CONSTRAINED,
        )
        self.assertEqual(assessment.findings, ("fixed_arbitration_runtime",))
        self.assertFalse(assessment.benchmark_of_record_eligible)

    def test_runtime_scaffolding_takes_precedence_over_constrained_flags(self) -> None:
        assessment = assess_scaffold_support(
            {
                "enable_deterministic_guards": True,
                "enable_food_direction_bias": False,
                "use_learned_arbitration": False,
                "warm_start_scale": 0.9,
            },
            eval_reflex_scale=0.0,
        )

        self.assertEqual(
            assessment.support_level,
            ScaffoldSupportLevel.SCAFFOLDED_RUNTIME,
        )
        self.assertEqual(
            assessment.findings,
            (
                "deterministic_guards_enabled",
                "warm_start_prior_active",
                "fixed_arbitration_runtime",
            ),
        )
        self.assertFalse(assessment.benchmark_of_record_eligible)

    def test_positive_eval_reflex_scale_marks_scaffolded_runtime(self) -> None:
        assessment = assess_scaffold_support(
            {
                "enable_deterministic_guards": False,
                "enable_food_direction_bias": False,
                "use_learned_arbitration": True,
                "warm_start_scale": 0.0,
            },
            eval_reflex_scale=0.5,
        )

        self.assertEqual(
            assessment.support_level,
            ScaffoldSupportLevel.SCAFFOLDED_RUNTIME,
        )
        self.assertEqual(assessment.findings, ("reflex_support_at_eval",))
        self.assertFalse(assessment.benchmark_of_record_eligible)

    def test_findings_list_contains_expected_labels_for_each_detected_scaffold(self) -> None:
        assessment = assess_scaffold_support(
            {
                "enable_deterministic_guards": True,
                "enable_food_direction_bias": True,
                "use_learned_arbitration": False,
                "warm_start_scale": 0.6,
            },
            eval_reflex_scale=1.0,
        )

        self.assertEqual(
            assessment.findings,
            (
                "deterministic_guards_enabled",
                "food_direction_bias_enabled",
                "warm_start_prior_active",
                "fixed_arbitration_runtime",
                "reflex_support_at_eval",
            ),
        )

    def test_string_typed_summary_values_are_coerced_before_classification(self) -> None:
        assessment = assess_scaffold_support(
            {
                "enable_deterministic_guards": "false",
                "enable_food_direction_bias": "true",
                "use_learned_arbitration": "false",
                "warm_start_scale": "0.5",
            },
            eval_reflex_scale=0.0,
        )

        self.assertEqual(
            assessment.support_level,
            ScaffoldSupportLevel.SCAFFOLDED_RUNTIME,
        )
        self.assertEqual(
            assessment.findings,
            (
                "food_direction_bias_enabled",
                "warm_start_prior_active",
                "fixed_arbitration_runtime",
            ),
        )
        self.assertFalse(assessment.benchmark_of_record_eligible)

    def test_unrecognized_string_flags_fall_back_to_defaults(self) -> None:
        assessment = assess_scaffold_support(
            {
                "enable_deterministic_guards": "maybe",
                "enable_food_direction_bias": "unknown",
                "use_learned_arbitration": "perhaps",
                "warm_start_scale": 0.0,
            },
            eval_reflex_scale=0.0,
        )

        self.assertEqual(
            assessment.support_level,
            ScaffoldSupportLevel.MINIMAL_MANUAL,
        )
        self.assertEqual(assessment.findings, ())
        self.assertTrue(assessment.benchmark_of_record_eligible)

    def test_invalid_numeric_scaffold_inputs_fall_back_without_crashing(self) -> None:
        assessment = assess_scaffold_support(
            {
                "use_learned_arbitration": True,
                "warm_start_scale": "not-a-number",
            },
            eval_reflex_scale="not-a-number",  # type: ignore[arg-type]
        )

        self.assertEqual(
            assessment.support_level,
            ScaffoldSupportLevel.MINIMAL_MANUAL,
        )
        self.assertEqual(assessment.findings, ())
        self.assertTrue(assessment.benchmark_of_record_eligible)


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
        self.assertIn("representation", spec.protocol.lower())
        self.assertIn("representation_specialization_score >= 0.10", spec.success_criterion)

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

    def test_primary_claims_require_austere_survival(self) -> None:
        expected = set(primary_claim_test_names())
        for spec in canonical_claim_tests():
            self.assertEqual(spec.austere_survival_required, spec.name in expected)

    def test_primary_austere_claim_scenarios_are_all_gates(self) -> None:
        for spec in canonical_claim_tests():
            if not spec.austere_survival_required:
                continue
            for scenario_name in spec.scenarios:
                with self.subTest(claim=spec.name, scenario=scenario_name):
                    self.assertEqual(
                        SCENARIO_AUSTERE_REQUIREMENTS[scenario_name][
                            "requirement_level"
                        ],
                        "gate",
                    )

    def test_capability_probes_not_in_claim_test_registry(self) -> None:
        claim_scenarios = {
            scenario
            for spec in canonical_claim_tests()
            for scenario in spec.scenarios
        }
        self.assertTrue(CAPABILITY_PROBE_SCENARIOS)
        self.assertTrue(set(CAPABILITY_PROBE_SCENARIOS).isdisjoint(claim_scenarios))


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
    @staticmethod
    def _specialization_payloads(
        spec: ClaimTestSpec,
        *,
        representation_score: float,
    ) -> dict[str, object]:
        return {
            "ablation": {
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
                        aggregate_metrics=_representation_metrics(
                            score=representation_score,
                        ),
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
            },
            "austere_survival": _austere_payload(
                spec.scenarios,
                set(spec.scenarios),
            ),
        }

    def test_passing_case_uses_synthetic_learning_payload(self) -> None:
        spec = resolve_claim_tests(["learning_without_privileged_signals"])[0]
        payloads = {
            "learning_evidence": {
                "conditions": {
                    "random_init": _behavior_payload(
                        spec.scenarios,
                        {"night_rest"},
                        aggregate_metrics={"config": _scaffold_config_summary()},
                    ),
                    "trained_without_reflex_support": _behavior_payload(
                        spec.scenarios,
                        {
                            "night_rest",
                            "predator_edge",
                            "entrance_ambush",
                            "shelter_blockade",
                        },
                        aggregate_metrics={"eval_reflex_scale": 0.0},
                    ),
                }
            },
            "austere_survival": _austere_payload(
                spec.scenarios,
                set(spec.scenarios),
            ),
        }

        result = evaluate_claim_test(spec, payloads)

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
        self.assertEqual(result["scaffold_support_level"], "minimal_manual")
        self.assertEqual(result["claim_severity"], "full")
        self.assertTrue(result["benchmark_of_record_eligible"])

    def test_extract_scaffold_config_for_learning_claim_uses_reference_and_comparison(self) -> None:
        spec = resolve_claim_tests(["learning_without_privileged_signals"])[0]
        expected_config = _scaffold_config_summary(warm_start_scale=0.1)
        payloads = {
            "learning_evidence": {
                "conditions": {
                    "random_init": _behavior_payload(
                        spec.scenarios,
                        {"night_rest"},
                        aggregate_metrics={"config": expected_config},
                    ),
                    "trained_without_reflex_support": _behavior_payload(
                        spec.scenarios,
                        set(spec.scenarios),
                        aggregate_metrics={"eval_reflex_scale": 0.5},
                    ),
                }
            }
        }

        config_summary, eval_reflex_scale = (
            extract_claim_config_for_scaffold_assessment(
                spec,
                payloads,
            )
        )

        self.assertEqual(config_summary, expected_config)
        self.assertEqual(eval_reflex_scale, 0.5)

    def test_extract_scaffold_config_for_learning_claim_uses_verdict_determining_comparison(self) -> None:
        spec = ClaimTestSpec(
            name="custom_learning_claim",
            hypothesis="h",
            description="d",
            protocol="Compose the learning-evidence workflow for validation.",
            reference_condition="random_init",
            comparison_conditions=(
                "trained_without_reflex_support",
                "reflex_only",
            ),
            primary_metric="scenario_success_rate",
            success_criterion="s",
            effect_size_metric="scenario_success_rate_delta",
            scenarios=("predator_edge",),
        )
        expected_config = _scaffold_config_summary(warm_start_scale=0.1)
        payloads = {
            "learning_evidence": {
                "conditions": {
                    "random_init": _behavior_payload(
                        spec.scenarios,
                        set(),
                        aggregate_metrics={"config": expected_config},
                    ),
                    "trained_without_reflex_support": _behavior_payload(
                        spec.scenarios,
                        set(spec.scenarios),
                        aggregate_metrics={"eval_reflex_scale": 0.0},
                    ),
                    "reflex_only": _behavior_payload(
                        spec.scenarios,
                        set(spec.scenarios),
                        aggregate_metrics={"eval_reflex_scale": 1.0},
                    ),
                }
            }
        }

        config_summary, eval_reflex_scale = (
            extract_claim_config_for_scaffold_assessment(
                spec,
                payloads,
            )
        )

        self.assertEqual(config_summary, expected_config)
        self.assertEqual(eval_reflex_scale, 0.0)

    def test_extract_scaffold_config_for_noise_claim_uses_baseline_condition(self) -> None:
        spec = resolve_claim_tests(["noise_preserves_threat_valence"])[0]
        expected_config = _scaffold_config_summary(warm_start_scale=0.1)
        payloads = {
            "noise_robustness": {
                "matrix_spec": {
                    "train_conditions": ["none", "low"],
                    "eval_conditions": ["none", "low"],
                },
                "matrix": {
                    "none": {
                        "none": {
                            "config": expected_config,
                            "eval_reflex_scale": 0.0,
                        },
                        "low": {},
                    },
                    "low": {
                        "none": {},
                        "low": {},
                    },
                },
            }
        }

        config_summary, eval_reflex_scale = (
            extract_claim_config_for_scaffold_assessment(
                spec,
                payloads,
            )
        )

        self.assertEqual(config_summary, expected_config)
        self.assertEqual(eval_reflex_scale, 0.0)

    def test_compare_noise_robustness_includes_cell_config_for_scaffold_assessment(self) -> None:
        payload, _ = compare_noise_robustness(
            episodes=0,
            evaluation_episodes=0,
            max_steps=5,
            seeds=(0,),
            names=("predator_edge",),
            episodes_per_scenario=1,
            robustness_matrix=RobustnessMatrixSpec(
                train_conditions=("none",),
                eval_conditions=("none",),
            ),
        )

        cell_payload = payload["matrix"]["none"]["none"]

        self.assertIsInstance(cell_payload.get("config"), dict)
        self.assertIn("use_learned_arbitration", cell_payload["config"])
        self.assertIn("eval_reflex_scale", cell_payload)
        self.assertIsNotNone(cell_payload["eval_reflex_scale"])

    def test_extract_scaffold_config_returns_empty_tuple_when_config_missing(self) -> None:
        spec = resolve_claim_tests(["memory_improves_shelter_return"])[0]
        payloads = {
            "ablation": {
                "variants": {
                    "modular_full": _behavior_payload(spec.scenarios, {"night_rest"}),
                    "modular_recurrent": _behavior_payload(
                        spec.scenarios,
                        set(spec.scenarios),
                    ),
                }
            }
        }

        config_summary, eval_reflex_scale = (
            extract_claim_config_for_scaffold_assessment(
                spec,
                payloads,
            )
        )

        self.assertEqual(config_summary, {})
        self.assertIsNone(eval_reflex_scale)

    def test_learning_claim_reports_uncertainty_and_cohens_d(self) -> None:
        spec = resolve_claim_tests(["learning_without_privileged_signals"])[0]
        random_payload = _add_seed_level_success_rates(
            _behavior_payload(
                spec.scenarios,
                {"night_rest"},
                aggregate_metrics={"config": _scaffold_config_summary()},
            ),
            condition="random_init",
            scenarios=spec.scenarios,
            seed_values={1: 0.1, 2: 0.3},
        )
        trained_payload = _add_seed_level_success_rates(
            _behavior_payload(
                spec.scenarios,
                {
                    "night_rest",
                    "predator_edge",
                    "entrance_ambush",
                    "shelter_blockade",
                },
                aggregate_metrics={"eval_reflex_scale": 0.0},
            ),
            condition="trained_without_reflex_support",
            scenarios=spec.scenarios,
            seed_values={1: 0.7, 2: 0.9},
        )
        payloads = {
            "learning_evidence": {
                "conditions": {
                    "random_init": random_payload,
                    "trained_without_reflex_support": trained_payload,
                }
            },
            "austere_survival": _austere_payload(
                spec.scenarios,
                set(spec.scenarios),
            ),
        }

        result = evaluate_claim_test(spec, payloads)

        comparison_name = "trained_without_reflex_support"
        self.assertEqual(result["reference_uncertainty"]["n_seeds"], 2)
        self.assertEqual(
            result["comparison_uncertainty"][comparison_name]["n_seeds"],
            2,
        )
        self.assertEqual(
            result["delta_uncertainty"][comparison_name]["n_seeds"],
            2,
        )
        self.assertEqual(
            result["effect_size_uncertainty"][comparison_name]["n_seeds"],
            2,
        )
        self.assertIn(comparison_name, result["cohens_d"])
        self.assertIn(comparison_name, result["effect_magnitude"])
        self.assertAlmostEqual(result["reference_uncertainty"]["mean"], 0.2)
        self.assertAlmostEqual(
            result["comparison_uncertainty"][comparison_name]["mean"],
            0.8,
        )
        self.assertAlmostEqual(
            result["delta_uncertainty"][comparison_name]["mean"],
            0.6,
        )
        self.assertAlmostEqual(
            result["effect_size_uncertainty"][comparison_name]["mean"],
            4.242641,
            places=6,
        )
        self.assertAlmostEqual(
            result["cohens_d"][comparison_name],
            4.242641,
            places=6,
        )
        self.assertEqual(result["effect_magnitude"][comparison_name], "large")

    def test_failing_case_uses_synthetic_learning_payload(self) -> None:
        spec = resolve_claim_tests(["escape_without_reflex_support"])[0]
        payloads = {
            "learning_evidence": {
                "conditions": {
                    "reflex_only": _behavior_payload(
                        spec.scenarios,
                        set(),
                        aggregate_metrics={"config": _scaffold_config_summary()},
                    ),
                    "trained_without_reflex_support": _behavior_payload(
                        spec.scenarios,
                        {"predator_edge"},
                        aggregate_metrics={"eval_reflex_scale": 0.0},
                    ),
                }
            },
            "austere_survival": _austere_payload(
                spec.scenarios,
                set(spec.scenarios),
            ),
        }

        result = evaluate_claim_test(spec, payloads)

        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["passed"])

    def test_metric_seed_values_from_payload_does_not_fabricate_fallback_rows(self) -> None:
        rows = metric_seed_values_from_payload(
            {
                "summary": {"scenario_success_rate": 0.8},
                "suite": {"night_rest": {"success_rate": 0.9}},
            },
            metric_name="scenario_success_rate",
            scenario="night_rest",
            fallback_value=0.9,
        )

        self.assertEqual(rows, [])

    def test_missing_condition_returns_skipped_result(self) -> None:
        spec = resolve_claim_tests(["learning_without_privileged_signals"])[0]
        payloads = {
            "learning_evidence": {
                "conditions": {
                    "random_init": _behavior_payload(
                        spec.scenarios,
                        {"night_rest"},
                        aggregate_metrics={"config": _scaffold_config_summary()},
                    ),
                }
            }
        }

        result = evaluate_claim_test(spec, payloads)

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

        result = evaluate_claim_test(spec, payloads)

        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["passed"])
        self.assertEqual(result["delta"]["modular_recurrent"], 0.0)

    def test_missing_scenario_key_returns_skipped_result(self) -> None:
        """
        Verifies that evaluating a claim test returns a skipped result when a payload omits a required scenario.
        
        Constructs a payload where one condition's suite is missing a scenario from the claim spec, calls evaluate_claim_test, and asserts the result has status "skipped" and the reason contains "Missing required scenarios".
        """
        spec = resolve_claim_tests(["learning_without_privileged_signals"])[0]
        incomplete_scenarios = tuple(spec.scenarios[:-1])
        payloads = {
            "learning_evidence": {
                "conditions": {
                    "random_init": _behavior_payload(
                        spec.scenarios,
                        {"night_rest"},
                        aggregate_metrics={"config": _scaffold_config_summary()},
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

        result = evaluate_claim_test(spec, payloads)

        self.assertEqual(result["status"], "skipped")
        self.assertIn("Missing required scenarios", result["reason"])

    def test_required_austere_survival_failure_fails_claim(self) -> None:
        spec = resolve_claim_tests(["learning_without_privileged_signals"])[0]
        payloads = {
            "learning_evidence": {
                "conditions": {
                    "random_init": _behavior_payload(
                        spec.scenarios,
                        set(),
                        aggregate_metrics={"config": _scaffold_config_summary()},
                    ),
                    "trained_without_reflex_support": _behavior_payload(
                        spec.scenarios,
                        set(spec.scenarios),
                        aggregate_metrics={"eval_reflex_scale": 0.0},
                    ),
                }
            },
            "austere_survival": _austere_payload(
                spec.scenarios,
                set(spec.scenarios) - {"shelter_blockade"},
            ),
        }

        result = evaluate_claim_test(spec, payloads)

        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["passed"])
        self.assertIn("Austere survival gate failed", result["reason"])
        self.assertIn(
            "shelter_blockade",
            result["austere_survival_gate"]["failed_scenarios"],
        )

    def test_missing_scaffold_inputs_mark_result_as_non_benchmark(self) -> None:
        spec = resolve_claim_tests(["learning_without_privileged_signals"])[0]
        payloads = {
            "learning_evidence": {
                "conditions": {
                    "random_init": _behavior_payload(
                        spec.scenarios,
                        {"night_rest"},
                        aggregate_metrics={"config": _scaffold_config_summary()},
                    ),
                    "trained_without_reflex_support": _behavior_payload(
                        spec.scenarios,
                        set(spec.scenarios),
                    ),
                }
            },
            "austere_survival": _austere_payload(
                spec.scenarios,
                set(spec.scenarios),
            ),
        }

        result = evaluate_claim_test(spec, payloads)

        self.assertEqual(result["status"], "passed")
        self.assertTrue(result["passed"])
        self.assertEqual(result["scaffold_support_level"], "missing_inputs")
        self.assertEqual(
            result["scaffold_findings"],
            ["scaffold_eval_reflex_scale_missing"],
        )
        self.assertEqual(result["claim_severity"], "non_benchmark")
        self.assertFalse(result["benchmark_of_record_eligible"])
        self.assertIn(
            "scaffold_eval_reflex_scale_missing",
            " ".join(result["notes"]),
        )

    def test_primary_pass_under_scaffolded_runtime_becomes_non_benchmark_failure(self) -> None:
        spec = resolve_claim_tests(["learning_without_privileged_signals"])[0]
        payloads = {
            "learning_evidence": {
                "conditions": {
                    "random_init": _behavior_payload(
                        spec.scenarios,
                        {"night_rest"},
                        aggregate_metrics={"config": _scaffold_config_summary()},
                    ),
                    "trained_without_reflex_support": _behavior_payload(
                        spec.scenarios,
                        set(spec.scenarios),
                        aggregate_metrics={"eval_reflex_scale": 0.5},
                    ),
                }
            },
            "austere_survival": _austere_payload(
                spec.scenarios,
                set(spec.scenarios),
            ),
        }

        result = evaluate_claim_test(spec, payloads)

        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["passed"])
        self.assertEqual(result["scaffold_support_level"], "scaffolded_runtime")
        self.assertEqual(result["claim_severity"], "non_benchmark")
        self.assertFalse(result["benchmark_of_record_eligible"])
        self.assertIn("scaffolded runtime conditions", result["reason"])
        self.assertIn("reflex_support_at_eval", " ".join(result["notes"]))

    def test_primary_pass_under_standard_constrained_is_qualified(self) -> None:
        spec = resolve_claim_tests(["learning_without_privileged_signals"])[0]
        payloads = {
            "learning_evidence": {
                "conditions": {
                    "random_init": _behavior_payload(
                        spec.scenarios,
                        {"night_rest"},
                        aggregate_metrics={
                            "config": _scaffold_config_summary(
                                warm_start_scale=0.5,
                            )
                        },
                    ),
                    "trained_without_reflex_support": _behavior_payload(
                        spec.scenarios,
                        set(spec.scenarios),
                        aggregate_metrics={"eval_reflex_scale": 0.0},
                    ),
                }
            },
            "austere_survival": _austere_payload(
                spec.scenarios,
                set(spec.scenarios),
            ),
        }

        result = evaluate_claim_test(spec, payloads)

        self.assertEqual(result["status"], "passed")
        self.assertTrue(result["passed"])
        self.assertEqual(result["scaffold_support_level"], "standard_constrained")
        self.assertEqual(result["claim_severity"], "qualified")
        self.assertFalse(result["benchmark_of_record_eligible"])
        self.assertIn("warm_start_prior_active", " ".join(result["notes"]))

    def test_non_required_claim_can_pass_without_austere_survival(self) -> None:
        spec = resolve_claim_tests(["memory_improves_shelter_return"])[0]
        payloads = {
            "ablation": {
                "variants": {
                    "modular_full": _behavior_payload(
                        spec.scenarios,
                        set(),
                        aggregate_metrics={
                            "config": _scaffold_config_summary(),
                            "eval_reflex_scale": 0.0,
                        },
                    ),
                    "modular_recurrent": _behavior_payload(
                        spec.scenarios,
                        set(spec.scenarios),
                    ),
                }
            }
        }

        result = evaluate_claim_test(spec, payloads)

        self.assertEqual(result["status"], "passed")
        self.assertFalse(result["austere_survival_required"])
        self.assertTrue(result["austere_survival_gate"]["passed"])
        self.assertEqual(result["scaffold_support_level"], "minimal_manual")
        self.assertEqual(result["claim_severity"], "full")

    def test_non_primary_claim_keeps_pass_status_under_scaffolded_runtime(self) -> None:
        spec = resolve_claim_tests(["memory_improves_shelter_return"])[0]
        payloads = {
            "ablation": {
                "variants": {
                    "modular_full": _behavior_payload(
                        spec.scenarios,
                        {"night_rest"},
                        aggregate_metrics={
                            "config": _scaffold_config_summary(
                                enable_deterministic_guards=True,
                            ),
                            "eval_reflex_scale": 0.0,
                        },
                    ),
                    "modular_recurrent": _behavior_payload(
                        spec.scenarios,
                        set(spec.scenarios),
                    ),
                }
            }
        }

        result = evaluate_claim_test(spec, payloads)

        self.assertEqual(result["status"], "passed")
        self.assertTrue(result["passed"])
        self.assertEqual(result["scaffold_support_level"], "scaffolded_runtime")
        self.assertEqual(result["claim_severity"], "non_benchmark")
        self.assertFalse(result["benchmark_of_record_eligible"])
        self.assertIn(
            "deterministic_guards_enabled",
            " ".join(result["notes"]),
        )

    def test_specialization_claim_includes_representation_metrics_in_reference_value(self) -> None:
        spec = resolve_claim_tests(
            ["specialization_emerges_with_multiple_predators"]
        )[0]
        payloads = self._specialization_payloads(
            spec,
            representation_score=0.15,
        )

        result = evaluate_claim_test(spec, payloads)

        self.assertEqual(result["status"], "passed")
        reference_value = result["reference_value"]
        self.assertAlmostEqual(
            reference_value["representation_specialization_score"],
            0.15,
        )
        self.assertTrue(reference_value["representation_tier_passed"])
        self.assertIn("proposer_divergence_by_module", reference_value)
        self.assertIn("action_center_gate_differential", reference_value)
        self.assertIn(
            "action_center_contribution_differential",
            reference_value,
        )

    def test_specialization_claim_fails_when_representation_score_is_below_threshold(self) -> None:
        spec = resolve_claim_tests(
            ["specialization_emerges_with_multiple_predators"]
        )[0]
        payloads = self._specialization_payloads(
            spec,
            representation_score=0.05,
        )

        result = evaluate_claim_test(spec, payloads)

        # High behavioral specialization without emerging representation
        # separation should fail the added representation tier.
        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["passed"])
        self.assertFalse(result["reference_value"]["representation_tier_passed"])
        self.assertIn("downstream gating", " ".join(result["notes"]))

    def test_specialization_claim_skips_when_representation_metrics_are_missing(self) -> None:
        spec = resolve_claim_tests(
            ["specialization_emerges_with_multiple_predators"]
        )[0]
        payloads = {
            "ablation": {
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
            },
            "austere_survival": _austere_payload(
                spec.scenarios,
                set(spec.scenarios),
            ),
        }

        result = evaluate_claim_test(spec, payloads)

        self.assertEqual(result["status"], "skipped")
        self.assertIn("representation specialization evidence", result["reason"])


class ClaimScaffoldSeverityTest(unittest.TestCase):
    @staticmethod
    def _primary_learning_payloads(
        spec: ClaimTestSpec,
        *,
        config: dict[str, object],
        eval_reflex_scale: float,
    ) -> dict[str, object]:
        return {
            "learning_evidence": {
                "conditions": {
                    "random_init": _behavior_payload(
                        spec.scenarios,
                        {"night_rest"},
                        aggregate_metrics={"config": config},
                    ),
                    "trained_without_reflex_support": _behavior_payload(
                        spec.scenarios,
                        set(spec.scenarios),
                        aggregate_metrics={"eval_reflex_scale": eval_reflex_scale},
                    ),
                }
            },
            "austere_survival": _austere_payload(
                spec.scenarios,
                set(spec.scenarios),
            ),
        }

    def test_primary_claim_fails_when_level_is_scaffolded_runtime_even_if_metrics_pass(self) -> None:
        spec = resolve_claim_tests(["learning_without_privileged_signals"])[0]
        result = evaluate_claim_test(
            spec,
            self._primary_learning_payloads(
                spec,
                config=_scaffold_config_summary(),
                eval_reflex_scale=0.5,
            ),
        )

        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["passed"])
        self.assertEqual(result["claim_severity"], "non_benchmark")
        self.assertEqual(result["scaffold_support_level"], "scaffolded_runtime")

    def test_primary_claim_passes_with_qualified_severity_under_standard_constrained(self) -> None:
        spec = resolve_claim_tests(["learning_without_privileged_signals"])[0]
        result = evaluate_claim_test(
            spec,
            self._primary_learning_payloads(
                spec,
                config=_scaffold_config_summary(warm_start_scale=0.5),
                eval_reflex_scale=0.0,
            ),
        )

        self.assertEqual(result["status"], "passed")
        self.assertTrue(result["passed"])
        self.assertEqual(result["claim_severity"], "qualified")
        self.assertEqual(result["scaffold_support_level"], "standard_constrained")
        self.assertFalse(result["benchmark_of_record_eligible"])

    def test_primary_claim_passes_with_full_severity_and_benchmark_eligibility_under_minimal_manual(self) -> None:
        spec = resolve_claim_tests(["learning_without_privileged_signals"])[0]
        result = evaluate_claim_test(
            spec,
            self._primary_learning_payloads(
                spec,
                config=_scaffold_config_summary(),
                eval_reflex_scale=0.0,
            ),
        )

        self.assertEqual(result["status"], "passed")
        self.assertTrue(result["passed"])
        self.assertEqual(result["claim_severity"], "full")
        self.assertEqual(result["scaffold_support_level"], "minimal_manual")
        self.assertTrue(result["benchmark_of_record_eligible"])

    def test_non_primary_claims_do_not_change_pass_fail_status_based_on_scaffold_level(self) -> None:
        spec = resolve_claim_tests(["memory_improves_shelter_return"])[0]
        result = evaluate_claim_test(
            spec,
            {
                "ablation": {
                    "variants": {
                        "modular_full": _behavior_payload(
                            spec.scenarios,
                            {"night_rest"},
                            aggregate_metrics={
                                "config": _scaffold_config_summary(
                                    enable_food_direction_bias=True,
                                ),
                                "eval_reflex_scale": 0.0,
                            },
                        ),
                        "modular_recurrent": _behavior_payload(
                            spec.scenarios,
                            set(spec.scenarios),
                        ),
                    }
                }
            },
        )

        self.assertEqual(result["status"], "passed")
        self.assertTrue(result["passed"])
        self.assertEqual(result["claim_severity"], "non_benchmark")
        self.assertEqual(result["scaffold_support_level"], "scaffolded_runtime")

    def test_scaffold_metadata_appears_correctly_in_claim_result_dict(self) -> None:
        spec = resolve_claim_tests(["learning_without_privileged_signals"])[0]
        result = evaluate_claim_test(
            spec,
            self._primary_learning_payloads(
                spec,
                config=_scaffold_config_summary(warm_start_scale=0.5),
                eval_reflex_scale=0.0,
            ),
        )

        self.assertEqual(result["scaffold_support_level"], "standard_constrained")
        self.assertEqual(result["scaffold_findings"], ["warm_start_prior_active"])
        self.assertFalse(result["benchmark_of_record_eligible"])
        self.assertEqual(result["claim_severity"], "qualified")

    def test_scaffold_fields_appear_in_csv_row_output(self) -> None:
        spec = resolve_claim_tests(["learning_without_privileged_signals"])[0]
        payload, rows = run_claim_test_suite(
            claim_tests=[spec.name],
            learning_evidence_payload=self._primary_learning_payloads(
                spec,
                config=_scaffold_config_summary(warm_start_scale=0.5),
                eval_reflex_scale=0.0,
            )["learning_evidence"],
            austere_survival_payload=_austere_payload(
                spec.scenarios,
                set(spec.scenarios),
            ),
        )

        self.assertEqual(payload["claims"][spec.name]["claim_severity"], "qualified")
        self.assertEqual(rows[0]["claim_test_scaffold_support_level"], "standard_constrained")
        self.assertEqual(
            json.loads(rows[0]["claim_test_scaffold_findings"]),
            ["warm_start_prior_active"],
        )
        self.assertFalse(rows[0]["claim_test_benchmark_of_record_eligible"])
        self.assertEqual(rows[0]["claim_test_severity"], "qualified")


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
        self.assertTrue(summary["all_primary_claims_benchmark_eligible"])

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


class ClaimTestSourceTest(unittest.TestCase):
    """Tests for claim_test_source()."""

    def _make_spec(self, protocol: str) -> ClaimTestSpec:
        return ClaimTestSpec(
            name="test",
            hypothesis="h",
            description="d",
            protocol=protocol,
            reference_condition="ref",
            comparison_conditions=("comp",),
            primary_metric="scenario_success_rate",
            success_criterion="s",
            effect_size_metric="e",
            scenarios=("scenario",),
        )

    def test_learning_evidence_protocol_returns_learning_evidence(self) -> None:
        spec = self._make_spec("learning-evidence: compare training conditions")
        self.assertEqual(claim_test_source(spec), "learning_evidence")

    def test_noise_robustness_protocol_returns_noise_robustness(self) -> None:
        spec = self._make_spec("noise-robustness: robustness matrix evaluation")
        self.assertEqual(claim_test_source(spec), "noise_robustness")

    def test_ablation_protocol_returns_ablation(self) -> None:
        spec = self._make_spec("ablation: compare modular variants")
        self.assertEqual(claim_test_source(spec), "ablation")

    def test_unknown_protocol_returns_none(self) -> None:
        spec = self._make_spec("some-other-protocol: unknown")
        self.assertIsNone(claim_test_source(spec))

    def test_protocol_matching_is_case_insensitive(self) -> None:
        spec = self._make_spec("Learning-Evidence: upper case check")
        self.assertEqual(claim_test_source(spec), "learning_evidence")

    def test_ablation_in_middle_of_protocol_matches(self) -> None:
        spec = self._make_spec("variant-ablation-comparison")
        self.assertEqual(claim_test_source(spec), "ablation")


class ClaimSkipResultTest(unittest.TestCase):
    """Tests for claim_skip_result()."""

    def _make_spec(self) -> ClaimTestSpec:
        return ClaimTestSpec(
            name="test_claim",
            hypothesis="h",
            description="d",
            protocol="p",
            reference_condition="ref",
            comparison_conditions=("comp",),
            primary_metric="scenario_success_rate",
            success_criterion="score >= 0.7",
            effect_size_metric="e",
            scenarios=("scenario_a", "scenario_b"),
        )

    def test_status_is_skipped(self) -> None:
        result = claim_skip_result(self._make_spec(), "test reason")
        self.assertEqual(result["status"], "skipped")

    def test_passed_is_false(self) -> None:
        result = claim_skip_result(self._make_spec(), "test reason")
        self.assertFalse(result["passed"])

    def test_reason_is_stored(self) -> None:
        result = claim_skip_result(self._make_spec(), "cannot find payload")
        self.assertEqual(result["reason"], "cannot find payload")

    def test_reference_value_is_none(self) -> None:
        result = claim_skip_result(self._make_spec(), "reason")
        self.assertIsNone(result["reference_value"])

    def test_comparison_values_is_empty_dict(self) -> None:
        result = claim_skip_result(self._make_spec(), "reason")
        self.assertEqual(result["comparison_values"], {})

    def test_delta_is_empty_dict(self) -> None:
        result = claim_skip_result(self._make_spec(), "reason")
        self.assertEqual(result["delta"], {})

    def test_effect_size_is_none(self) -> None:
        result = claim_skip_result(self._make_spec(), "reason")
        self.assertIsNone(result["effect_size"])

    def test_primary_metric_from_spec(self) -> None:
        result = claim_skip_result(self._make_spec(), "reason")
        self.assertEqual(result["primary_metric"], "scenario_success_rate")

    def test_scenarios_evaluated_from_spec(self) -> None:
        result = claim_skip_result(self._make_spec(), "reason")
        self.assertEqual(result["scenarios_evaluated"], ["scenario_a", "scenario_b"])

    def test_notes_contains_success_criterion(self) -> None:
        result = claim_skip_result(self._make_spec(), "reason")
        self.assertIn("score >= 0.7", result["notes"])


class ClaimThresholdFromOperatorTest(unittest.TestCase):
    """Tests for claim_threshold_from_operator()."""

    def test_gte_operator_extracts_value(self) -> None:
        value = claim_threshold_from_operator("score >= 0.75", ">=")
        self.assertAlmostEqual(value, 0.75)

    def test_lte_operator_extracts_value(self) -> None:
        value = claim_threshold_from_operator("gap <= 0.15", "<=")
        self.assertAlmostEqual(value, 0.15)

    def test_gt_operator_extracts_value(self) -> None:
        value = claim_threshold_from_operator("delta > 0.10", ">")
        self.assertAlmostEqual(value, 0.10)

    def test_no_match_returns_none(self) -> None:
        value = claim_threshold_from_operator("score is good", ">=")
        self.assertIsNone(value)

    def test_integer_value_is_parsed(self) -> None:
        value = claim_threshold_from_operator("count >= 3", ">=")
        self.assertAlmostEqual(value, 3.0)

    def test_negative_value_is_parsed(self) -> None:
        value = claim_threshold_from_operator("delta >= -0.05", ">=")
        self.assertAlmostEqual(value, -0.05)

    def test_operator_with_spaces_is_parsed(self) -> None:
        value = claim_threshold_from_operator("score >=  0.60", ">=")
        self.assertAlmostEqual(value, 0.60)

    def test_first_match_is_returned_when_multiple(self) -> None:
        value = claim_threshold_from_operator("a >= 0.5 and b >= 0.8", ">=")
        self.assertAlmostEqual(value, 0.5)


class ClaimThresholdFromPhraseTest(unittest.TestCase):
    """Tests for claim_threshold_from_phrase()."""

    def test_by_at_least_phrase_extracts_value(self) -> None:
        value = claim_threshold_from_phrase(
            "trained agent improves by at least 0.15", "by at least"
        )
        self.assertAlmostEqual(value, 0.15)

    def test_no_match_returns_none(self) -> None:
        value = claim_threshold_from_phrase("score is good", "by at least")
        self.assertIsNone(value)

    def test_custom_phrase_extracts_value(self) -> None:
        value = claim_threshold_from_phrase(
            "representation_specialization_score >= 0.10", "representation_specialization_score >="
        )
        self.assertAlmostEqual(value, 0.10)

    def test_negative_threshold_is_parsed(self) -> None:
        value = claim_threshold_from_phrase("delta changes by at least -0.05", "by at least")
        self.assertAlmostEqual(value, -0.05)

    def test_phrase_with_spaces_after_matched(self) -> None:
        value = claim_threshold_from_phrase("improves by at least  0.20", "by at least")
        self.assertAlmostEqual(value, 0.20)


class ClaimCountThresholdTest(unittest.TestCase):
    """Tests for claim_count_threshold()."""

    def test_extracts_count_from_at_least_phrase(self) -> None:
        count = claim_count_threshold("at least 2 of the 3 scenarios must pass")
        self.assertEqual(count, 2)

    def test_no_match_returns_none(self) -> None:
        count = claim_count_threshold("all scenarios must pass")
        self.assertIsNone(count)

    def test_larger_count_is_parsed(self) -> None:
        count = claim_count_threshold("at least 5 of the 7 checks must be engaged")
        self.assertEqual(count, 5)

    def test_count_of_one_is_parsed(self) -> None:
        count = claim_count_threshold("at least 1 of the 3 checks engaged")
        self.assertEqual(count, 1)

    def test_partial_phrase_no_of_the_returns_none(self) -> None:
        count = claim_count_threshold("at least 2 checks engaged")
        self.assertIsNone(count)


class ClaimRegistryEntryTest(unittest.TestCase):
    """Tests for claim_registry_entry()."""

    def test_valid_entry_returned_with_no_error(self) -> None:
        payload = {
            "variants": {
                "modular_full": {"config": {}, "suite": {}}
            }
        }
        entry, error = claim_registry_entry(
            payload, registry_key="variants", entry_name="modular_full"
        )
        self.assertIsNotNone(entry)
        self.assertIsNone(error)

    def test_none_payload_returns_error(self) -> None:
        entry, error = claim_registry_entry(
            None, registry_key="variants", entry_name="modular_full"
        )
        self.assertIsNone(entry)
        self.assertIsNotNone(error)
        self.assertIn("variants", error)

    def test_missing_registry_key_returns_error(self) -> None:
        payload = {"conditions": {"ref": {}}}
        entry, error = claim_registry_entry(
            payload, registry_key="variants", entry_name="modular_full"
        )
        self.assertIsNone(entry)
        self.assertIsNotNone(error)

    def test_missing_entry_name_returns_error(self) -> None:
        payload = {"variants": {"other_variant": {}}}
        entry, error = claim_registry_entry(
            payload, registry_key="variants", entry_name="modular_full"
        )
        self.assertIsNone(entry)
        self.assertIsNotNone(error)
        self.assertIn("modular_full", error)

    def test_skipped_entry_returns_none_with_skip_reason(self) -> None:
        payload = {
            "variants": {
                "skipped_variant": {
                    "skipped": True,
                    "reason": "Not enough data",
                }
            }
        }
        entry, error = claim_registry_entry(
            payload, registry_key="variants", entry_name="skipped_variant"
        )
        self.assertIsNone(entry)
        self.assertEqual(error, "Not enough data")

    def test_skipped_without_reason_returns_default_message(self) -> None:
        payload = {
            "variants": {
                "skipped_variant": {"skipped": True}
            }
        }
        entry, error = claim_registry_entry(
            payload, registry_key="variants", entry_name="skipped_variant"
        )
        self.assertIsNone(entry)
        self.assertIsNotNone(error)

    def test_non_dict_registry_value_returns_error(self) -> None:
        payload = {"variants": "not_a_dict"}
        entry, error = claim_registry_entry(
            payload, registry_key="variants", entry_name="modular_full"
        )
        self.assertIsNone(entry)
        self.assertIsNotNone(error)

    def test_conditions_registry_key_works(self) -> None:
        payload = {
            "conditions": {
                "random_init": {"config": {}, "suite": {}}
            }
        }
        entry, error = claim_registry_entry(
            payload, registry_key="conditions", entry_name="random_init"
        )
        self.assertIsNotNone(entry)
        self.assertIsNone(error)


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
