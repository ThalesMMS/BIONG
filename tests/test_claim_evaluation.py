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
