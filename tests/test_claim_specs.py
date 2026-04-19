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
