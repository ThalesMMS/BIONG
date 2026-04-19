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
