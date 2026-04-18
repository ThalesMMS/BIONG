from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

from .ablations import (
    BrainAblationConfig,
    compare_predator_type_ablation_performance,
    default_brain_config,
    resolve_ablation_configs,
    resolve_ablation_scenario_group,
)
from .benchmark_types import SeedLevelResult, UncertaintyEstimate
from .budget_profiles import BudgetProfile, resolve_budget
from .checkpointing import (
    CheckpointPenaltyMode,
    CheckpointSelectionConfig,
    checkpoint_candidate_sort_key,
    checkpoint_preload_fingerprint,
    checkpoint_run_fingerprint,
    mean_reward_from_behavior_payload,
    resolve_checkpoint_load_dir,
)
from .curriculum import (
    CURRICULUM_FOCUS_SCENARIOS,
    empty_curriculum_summary,
    validate_curriculum_profile,
)
from .export import compact_aggregate, compact_behavior_payload
from .learning_evidence import resolve_learning_evidence_conditions
from .memory import memory_leakage_audit
from .metrics import (
    EpisodeStats,
    aggregate_episode_stats,
    competence_label_from_eval_reflex_scale,
    flatten_behavior_rows,
)
from .noise import (
    NoiseConfig,
    RobustnessMatrixSpec,
    canonical_robustness_matrix,
    resolve_noise_profile,
)
from .operational_profiles import OperationalProfile
from .perception import observation_leakage_audit
from .reward import (
    MINIMAL_SHAPING_SURVIVAL_THRESHOLD,
    REWARD_PROFILES,
    SCENARIO_AUSTERE_REQUIREMENTS,
    SHAPING_GAP_POLICY,
    SHAPING_REDUCTION_ROADMAP,
    reward_component_audit,
    reward_profile_audit,
    validate_gap_policy,
)
from .scenarios import SCENARIO_NAMES, get_scenario
from .simulation import EXPERIMENT_OF_RECORD_REGIME, SpiderSimulation
from .statistics import bootstrap_confidence_interval, cohens_d
from .training_regimes import resolve_training_regime


def noise_profile_metadata(noise_profile: NoiseConfig) -> Dict[str, object]:
    """Return aggregate-safe metadata for a resolved noise profile."""
    return {
        "noise_profile": noise_profile.name,
        "noise_profile_config": noise_profile.to_summary(),
    }


def noise_profile_csv_value(noise_profile: NoiseConfig) -> str:
    """Serialize a resolved noise profile as stable JSON for CSV exports."""
    return json.dumps(
        noise_profile.to_summary(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def with_noise_profile_metadata(
    payload: Dict[str, object],
    noise_profile: NoiseConfig,
) -> Dict[str, object]:
    """
    Merge noise-profile metadata into a payload.

    Parameters:
        payload: Base payload to augment.
        noise_profile: Resolved noise profile to attach.

    Returns:
        A shallow copy with noise-profile metadata.
    """
    enriched = dict(payload)
    enriched.update(noise_profile_metadata(noise_profile))
    return enriched


def robustness_matrix_metadata(
    robustness_matrix: RobustnessMatrixSpec,
) -> Dict[str, object]:
    """
    Build serializable robustness-matrix metadata.

    Parameters:
        robustness_matrix: Matrix specification to summarize.

    Returns:
        A mapping containing the `"matrix_spec"` summary.
    """
    return {
        "matrix_spec": robustness_matrix.to_summary(),
    }


def matrix_cell_success_rate(payload: Dict[str, object] | None) -> float:
    """
    Extracts the scenario success rate from a compact matrix-cell payload.

    Parameters:
        payload (dict | None): Compact matrix-cell payload (or None) produced by robustness matrix evaluations.

    Returns:
        float: The `scenario_success_rate` value from the payload, or `0.0` if the payload is None or the field is missing.
    """
    return condition_compact_summary(payload).get(
        "scenario_success_rate",
        0.0,
    )


def robustness_aggregate_metrics(
    matrix_payloads: Dict[str, Dict[str, object]],
    *,
    robustness_matrix: RobustnessMatrixSpec,
) -> Dict[str, object]:
    """
    Compute aggregate scores for a train x eval noise matrix.

    Parameters:
        matrix_payloads: Nested train->eval matrix-cell payloads.
        robustness_matrix: Matrix axes and iteration order.

    Returns:
        Train/eval marginals and overall, diagonal, and off-diagonal scores.
    """
    train_marginals: Dict[str, float] = {}
    eval_marginals: Dict[str, float] = {}
    all_scores: List[float] = []
    diagonal_scores: List[float] = []
    off_diagonal_scores: List[float] = []
    all_scores_by_seed: Dict[int, List[float]] = {}
    diagonal_scores_by_seed: Dict[int, List[float]] = {}
    off_diagonal_scores_by_seed: Dict[int, List[float]] = {}

    for train_condition in robustness_matrix.train_conditions:
        train_scores = [
            matrix_cell_success_rate(
                matrix_payloads.get(train_condition, {}).get(eval_condition)
            )
            for eval_condition in robustness_matrix.eval_conditions
        ]
        train_marginals[train_condition] = safe_float(
            sum(train_scores) / len(train_scores) if train_scores else 0.0
        )

    for eval_condition in robustness_matrix.eval_conditions:
        eval_scores = [
            matrix_cell_success_rate(
                matrix_payloads.get(train_condition, {}).get(eval_condition)
            )
            for train_condition in robustness_matrix.train_conditions
        ]
        eval_marginals[eval_condition] = safe_float(
            sum(eval_scores) / len(eval_scores) if eval_scores else 0.0
        )

    for train_condition, eval_condition in robustness_matrix.cells():
        cell_payload = matrix_payloads.get(train_condition, {}).get(
            eval_condition
        )
        score = matrix_cell_success_rate(cell_payload)
        all_scores.append(score)
        for seed, seed_score in metric_seed_values_from_payload(
            cell_payload,
            metric_name="scenario_success_rate",
            fallback_value=score,
        ):
            all_scores_by_seed.setdefault(seed, []).append(seed_score)
            if train_condition == eval_condition:
                diagonal_scores_by_seed.setdefault(seed, []).append(seed_score)
            else:
                off_diagonal_scores_by_seed.setdefault(seed, []).append(seed_score)
        if train_condition == eval_condition:
            diagonal_scores.append(score)
        else:
            off_diagonal_scores.append(score)

    robustness_score = safe_float(
        sum(all_scores) / len(all_scores) if all_scores else 0.0
    )
    diagonal_score = safe_float(
        sum(diagonal_scores) / len(diagonal_scores)
        if diagonal_scores
        else 0.0
    )
    off_diagonal_score = safe_float(
        sum(off_diagonal_scores) / len(off_diagonal_scores)
        if off_diagonal_scores
        else 0.0
    )
    diagonal_minus_off_diagonal_score = round(
        float(diagonal_score - off_diagonal_score),
        6,
    )

    def _mean_rows(
        values_by_seed: Dict[int, List[float]],
        *,
        metric_name: str,
        fallback_value: float,
    ) -> list[SeedLevelResult]:
        rows = [
            SeedLevelResult(
                metric_name=metric_name,
                seed=seed,
                value=round(sum(values) / len(values), 6),
                condition="noise_robustness",
            )
            for seed, values in sorted(values_by_seed.items())
            if values
        ]
        if rows:
            return rows
        return [
            SeedLevelResult(
                metric_name=metric_name,
                seed=0,
                value=fallback_value,
                condition="noise_robustness",
            )
        ]

    robustness_rows = _mean_rows(
        all_scores_by_seed,
        metric_name="robustness_score",
        fallback_value=robustness_score,
    )
    diagonal_rows = _mean_rows(
        diagonal_scores_by_seed,
        metric_name="diagonal_score",
        fallback_value=diagonal_score,
    )
    off_diagonal_rows = _mean_rows(
        off_diagonal_scores_by_seed,
        metric_name="off_diagonal_score",
        fallback_value=off_diagonal_score,
    )
    common_gap_seeds = sorted(
        set(diagonal_scores_by_seed) & set(off_diagonal_scores_by_seed)
    )
    gap_rows = [
        SeedLevelResult(
            metric_name="diagonal_minus_off_diagonal_score",
            seed=seed,
            value=round(
                (sum(diagonal_scores_by_seed[seed]) / len(diagonal_scores_by_seed[seed]))
                - (
                    sum(off_diagonal_scores_by_seed[seed])
                    / len(off_diagonal_scores_by_seed[seed])
                ),
                6,
            ),
            condition="noise_robustness",
        )
        for seed in common_gap_seeds
        if diagonal_scores_by_seed[seed] and off_diagonal_scores_by_seed[seed]
    ]
    if not gap_rows:
        gap_rows = [
            SeedLevelResult(
                metric_name="diagonal_minus_off_diagonal_score",
                seed=0,
                value=diagonal_minus_off_diagonal_score,
                condition="noise_robustness",
            )
        ]

    return {
        "train_marginals": train_marginals,
        "eval_marginals": eval_marginals,
        "robustness_score": robustness_score,
        "diagonal_score": diagonal_score,
        "off_diagonal_score": off_diagonal_score,
        "diagonal_minus_off_diagonal_score": diagonal_minus_off_diagonal_score,
        "seed_level": seed_level_dicts(
            [
                *robustness_rows,
                *diagonal_rows,
                *off_diagonal_rows,
                *gap_rows,
            ]
        ),
        "uncertainty": {
            "robustness_score": aggregate_with_uncertainty(
                robustness_rows
            ),
            "diagonal_score": aggregate_with_uncertainty(diagonal_rows),
            "off_diagonal_score": aggregate_with_uncertainty(
                off_diagonal_rows
            ),
            "diagonal_minus_off_diagonal_score": (
                aggregate_with_uncertainty(gap_rows)
            ),
        },
    }


def profile_comparison_metrics(
    payload: Dict[str, object] | None,
) -> Dict[str, float]:
    """
    Extract three numeric comparison metrics from a behavior-suite or aggregate comparison payload.

    Parameters:
        payload (dict | None): A behavior-suite payload (with a top-level "summary") or an aggregate comparison payload; may be None or malformed.

    Returns:
        dict: A mapping with keys:
            - "scenario_success_rate": float, success rate aggregated per scenario.
            - "episode_success_rate": float, success rate aggregated per episode.
            - "mean_reward": float, mean reward (computed from payload summary or legacy scenario entries).
    """
    if not isinstance(payload, dict):
        return {
            "scenario_success_rate": 0.0,
            "episode_success_rate": 0.0,
            "mean_reward": 0.0,
        }
    summary = payload.get("summary")
    if isinstance(summary, dict):
        return {
            "scenario_success_rate": safe_float(
                summary.get("scenario_success_rate")
            ),
            "episode_success_rate": safe_float(
                summary.get("episode_success_rate")
            ),
            "mean_reward": condition_mean_reward(payload),
        }
    fallback_success_rate = (
        payload.get("survival_rate")
        if "survival_rate" in payload
        else payload.get("scenario_success_rate")
    )
    return {
        "scenario_success_rate": safe_float(
            fallback_success_rate
        ),
        "episode_success_rate": safe_float(
            payload.get("episode_success_rate", fallback_success_rate)
        ),
        "mean_reward": safe_float(payload.get("mean_reward")),
    }


def comparison_suite_from_payload(
    payload: Dict[str, object] | None,
) -> Dict[str, object]:
    """Extract or derive scenario-level comparison entries from a payload."""
    if not isinstance(payload, dict):
        return {}
    suite = payload.get("suite")
    if isinstance(suite, dict) and suite:
        return suite
    legacy_scenarios = payload.get("legacy_scenarios")
    if isinstance(legacy_scenarios, dict) and legacy_scenarios:
        derived_from_legacy: Dict[str, object] = {}
        for scenario_name, scenario_payload in sorted(legacy_scenarios.items()):
            if not isinstance(scenario_payload, dict):
                continue
            success_rate = safe_float(
                scenario_payload.get(
                    "success_rate",
                    scenario_payload.get("survival_rate"),
                )
            )
            derived_from_legacy[str(scenario_name)] = {
                "success_rate": round(float(success_rate), 6),
                "episodes": int(
                    safe_float(scenario_payload.get("episodes"))
                ),
                "mean_reward": round(
                    safe_float(scenario_payload.get("mean_reward")),
                    6,
                ),
                "success_basis": "episode_survival_rate",
            }
        if derived_from_legacy:
            return derived_from_legacy
    episodes_detail = payload.get("episodes_detail")
    if not isinstance(episodes_detail, list):
        return {}
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for item in episodes_detail:
        if not isinstance(item, dict):
            continue
        scenario_name = item.get("scenario")
        if scenario_name is None or str(scenario_name) == "":
            continue
        grouped.setdefault(str(scenario_name), []).append(item)
    derived: Dict[str, object] = {}
    for scenario_name, items in sorted(grouped.items()):
        episode_count = len(items)
        if episode_count == 0:
            continue
        surviving_count = sum(1 for item in items if bool(item.get("alive")))
        mean_reward = sum(
            safe_float(item.get("total_reward")) for item in items
        ) / episode_count
        derived[scenario_name] = {
            "success_rate": round(float(surviving_count / episode_count), 6),
            "episodes": episode_count,
            "mean_reward": round(float(mean_reward), 6),
            "success_basis": "episode_survival_rate",
        }
    return derived


def episode_history_reward_payload(
    history: Sequence[EpisodeStats],
) -> Dict[str, object]:
    """Build an audit comparison payload from raw episode stats."""
    history_list = list(history)
    payload = dict(aggregate_episode_stats(history_list))
    scenario_groups: Dict[str, List[EpisodeStats]] = {}
    for stats in history_list:
        if stats.scenario is None or str(stats.scenario) == "":
            continue
        scenario_groups.setdefault(str(stats.scenario), []).append(stats)
    suite: Dict[str, object] = {}
    legacy_scenarios: Dict[str, object] = {}
    for scenario_name, group in sorted(scenario_groups.items()):
        legacy_metrics = aggregate_episode_stats(group)
        success_rate = safe_float(legacy_metrics.get("survival_rate"))
        suite[scenario_name] = {
            "success_rate": round(float(success_rate), 6),
            "episodes": int(safe_float(legacy_metrics.get("episodes"))),
            "mean_reward": round(
                safe_float(legacy_metrics.get("mean_reward")),
                6,
            ),
            "success_basis": "episode_survival_rate",
            "legacy_metrics": compact_aggregate(legacy_metrics),
        }
        legacy_scenarios[scenario_name] = legacy_metrics
    if suite:
        payload["suite"] = suite
        payload["legacy_scenarios"] = legacy_scenarios
        scenario_success_rate = sum(
            safe_float(dict(item).get("success_rate"))
            for item in suite.values()
            if isinstance(item, dict)
        ) / len(suite)
    else:
        scenario_success_rate = safe_float(payload.get("survival_rate"))
    episode_success_rate = safe_float(payload.get("survival_rate"))
    payload["summary"] = {
        "scenario_success_rate": round(float(scenario_success_rate), 6),
        "episode_success_rate": round(float(episode_success_rate), 6),
        "mean_reward": round(safe_float(payload.get("mean_reward")), 6),
        "scenario_count": len(suite),
        "episode_count": int(safe_float(payload.get("episodes"))),
        "success_basis": "episode_survival_rate",
    }
    return payload


def find_austere_comparison_payload(
    payload: Dict[str, object] | None,
) -> Dict[str, object] | None:
    """Extract a reward-audit comparison payload from common wrapper shapes."""
    if not isinstance(payload, dict):
        return None
    if "behavior_survival" in payload or "deltas_vs_minimal" in payload:
        return payload
    comparison = payload.get("comparison")
    if isinstance(comparison, dict):
        return comparison
    reward_audit = payload.get("reward_audit")
    if isinstance(reward_audit, dict):
        comparison = reward_audit.get("comparison")
        if isinstance(comparison, dict):
            return comparison
    return None


def austere_comparison_from_payloads(
    payloads: Dict[str, Dict[str, object]],
) -> Dict[str, object] | None:
    """Find austere comparison data in a claim-test payload bundle."""
    preferred = payloads.get("austere_survival")
    comparison = find_austere_comparison_payload(preferred)
    if comparison is not None:
        return comparison
    for payload in payloads.values():
        comparison = find_austere_comparison_payload(payload)
        if comparison is not None:
            return comparison
    return None


def austere_survival_summary(
    behavior_survival: Dict[str, object],
    gap_policy_check: Dict[str, object],
) -> Dict[str, object]:
    """Promote gate/warning austere survival outcomes into a compact summary."""
    scenarios = behavior_survival.get("scenarios", {})
    if not isinstance(scenarios, dict):
        scenarios = {}
    expected_gate_count = sum(
        1
        for requirement in SCENARIO_AUSTERE_REQUIREMENTS.values()
        if requirement.get("requirement_level") == "gate"
    )
    observed_gate_count = 0
    gate_pass_count = 0
    gate_fail_count = 0
    warning_scenarios: list[Dict[str, object]] = []
    for scenario_name, scenario_payload in sorted(scenarios.items()):
        requirement = SCENARIO_AUSTERE_REQUIREMENTS.get(str(scenario_name))
        if requirement is None:
            continue
        if not isinstance(scenario_payload, dict):
            scenario_payload = {}
        requirement_level = str(requirement.get("requirement_level", ""))
        survives = bool(scenario_payload.get("survives", False))
        success_rate = round(
            safe_float(scenario_payload.get("austere_success_rate")),
            6,
        )
        if requirement_level == "gate":
            observed_gate_count += 1
            if survives:
                gate_pass_count += 1
            else:
                gate_fail_count += 1
        elif requirement_level == "warning" and not survives:
            warning_scenarios.append(
                {
                    "scenario": str(scenario_name),
                    "austere_success_rate": success_rate,
                    "survival_threshold": MINIMAL_SHAPING_SURVIVAL_THRESHOLD,
                }
            )
    return {
        "available": bool(behavior_survival.get("available", False)),
        "overall_survival_rate": round(
            safe_float(behavior_survival.get("survival_rate")),
            6,
        ),
        "expected_gate_count": expected_gate_count,
        "observed_gate_count": observed_gate_count,
        "gate_coverage_complete": observed_gate_count == expected_gate_count,
        "gate_pass_count": gate_pass_count,
        "gate_fail_count": gate_fail_count,
        "warning_scenarios": warning_scenarios,
        "gap_policy_violations": list(gap_policy_check.get("violations", [])),
    }


def shaping_dependent_behaviors(
    profile_payloads: Dict[str, object],
    *,
    minimal_profile: str | None,
) -> list[Dict[str, object]]:
    """List per-scenario dense-profile wins that exceed shaping-gap policy limits."""
    if minimal_profile is None:
        return []
    minimal_payload = profile_payloads.get(minimal_profile)
    if not isinstance(minimal_payload, dict):
        return []
    minimal_suite = comparison_suite_from_payload(minimal_payload)
    if not minimal_suite:
        return []
    success_limits = SHAPING_GAP_POLICY.get("max_scenario_success_rate_delta", {})
    if not isinstance(success_limits, dict):
        return []
    dependent: list[Dict[str, object]] = []
    for profile_name, profile_payload in sorted(profile_payloads.items()):
        if profile_name == minimal_profile or not isinstance(profile_payload, dict):
            continue
        profile_key = f"{profile_name}_minus_austere"
        if profile_key not in success_limits:
            continue
        limit = safe_float(success_limits.get(profile_key))
        profile_suite = comparison_suite_from_payload(profile_payload)
        if not profile_suite:
            continue
        for scenario_name, minimal_scenario in sorted(minimal_suite.items()):
            if not isinstance(minimal_scenario, dict):
                continue
            profile_scenario = profile_suite.get(scenario_name)
            if not isinstance(profile_scenario, dict):
                continue
            austere_success = safe_float(
                minimal_scenario.get("success_rate")
            )
            profile_success = safe_float(
                profile_scenario.get("success_rate")
            )
            delta = round(float(profile_success - austere_success), 6)
            if delta > limit:
                dependent.append(
                    {
                        "scenario": str(scenario_name),
                        "profile": str(profile_name),
                        "comparison": profile_key,
                        "success_rate_delta": delta,
                        "limit": round(float(limit), 6),
                        "profile_success_rate": round(float(profile_success), 6),
                        "austere_success_rate": round(float(austere_success), 6),
                    }
                )
    return dependent


def austere_survival_gate_passed(
    comparison: Dict[str, object] | None,
) -> bool | None:
    """Return the aggregate austere gate result when comparison data is available."""
    if not isinstance(comparison, dict):
        return None
    summary = comparison.get("austere_survival_summary")
    if not isinstance(summary, dict) or not bool(summary.get("available", False)):
        return None
    gate_coverage_complete = bool(summary.get("gate_coverage_complete", False))
    return bool(
        gate_coverage_complete
        and int(summary.get("gate_fail_count", 0)) == 0
    )


def shaping_reduction_status(
    reward_audit: Dict[str, object],
) -> Dict[str, object]:
    """Summarize roadmap risk and gap-policy status for top-level reports."""
    comparison = reward_audit.get("comparison")
    gap_policy_check = (
        comparison.get("gap_policy_check", {})
        if isinstance(comparison, dict)
        else {}
    )
    violations = (
        list(gap_policy_check.get("violations", []))
        if isinstance(gap_policy_check, dict)
        else []
    )
    terms_at_risk = [
        component_name
        for component_name, roadmap_entry in sorted(
            SHAPING_REDUCTION_ROADMAP.items()
        )
        if roadmap_entry.get("target_disposition") == "under_investigation"
        or roadmap_entry.get("reduction_priority") == "high"
    ]
    gate_passed = austere_survival_gate_passed(
        comparison if isinstance(comparison, dict) else None
    )
    return {
        "roadmap_target_count": len(SHAPING_REDUCTION_ROADMAP),
        "terms_at_risk": terms_at_risk,
        "terms_at_risk_count": len(terms_at_risk),
        "gap_policy_violation_count": len(violations),
        "gap_policy_violations": violations,
        "austere_survival_available": gate_passed is not None,
        "austere_survival_gate_passed": gate_passed,
    }


def build_reward_audit_comparison(
    comparison_payload: Dict[str, object] | None,
) -> Dict[str, object] | None:
    """
    Summarize reward-profile metrics and compute per-profile deltas relative to the "austere" baseline when available.

    Parameters:
        comparison_payload (dict | None): A payload containing a "reward_profiles" mapping from profile
            name to per-profile payloads. Each per-profile payload may include a "suite" mapping of
            per-scenario aggregates and summary/metrics used for comparison.

    Returns:
        dict | None: A compact comparison dictionary, or `None` if input is missing or invalid. When
        returned the dict contains:
          - "minimal_profile": name of the austere baseline profile if present, else `None`.
          - "profiles": mapping profile_name -> metrics (`scenario_success_rate`,
            `episode_success_rate`, `mean_reward`) with values rounded to 6 decimals.
          - "deltas_vs_minimal": mapping profile_name -> deltas vs the minimal profile for the same
            three metrics (rounded to 6 decimals). Empty if no austere baseline is available.
          - "behavior_survival": availability flag, minimal_profile name, configured survival threshold,
            per-scenario austere success rates, per-scenario episode counts, and per-scenario boolean
            `survives` indicating whether austere success_rate >= threshold.
          - "survival_rate": overall fraction of scenarios that survive under the austere profile
            (rounded to 6 decimals).
          - "notes": brief notes describing how metrics and survival were derived.
    """
    if not isinstance(comparison_payload, dict):
        return None
    profile_payloads = comparison_payload.get("reward_profiles")
    if not isinstance(profile_payloads, dict) or not profile_payloads:
        return None
    profiles = {
        name: profile_comparison_metrics(dict(payload))
        for name, payload in profile_payloads.items()
        if isinstance(payload, dict)
    }
    if not profiles:
        return None
    minimal_profile = "austere" if "austere" in profiles else None
    deltas_vs_minimal: Dict[str, object] = {}
    if minimal_profile is not None:
        minimal_metrics = profiles[minimal_profile]
        for name, metrics in profiles.items():
            deltas_vs_minimal[name] = {
                "scenario_success_rate_delta": round(
                    float(metrics["scenario_success_rate"])
                    - float(minimal_metrics["scenario_success_rate"]),
                    6,
                ),
                "episode_success_rate_delta": round(
                    float(metrics["episode_success_rate"])
                    - float(minimal_metrics["episode_success_rate"]),
                    6,
                ),
                "mean_reward_delta": round(
                    float(metrics["mean_reward"])
                    - float(minimal_metrics["mean_reward"]),
                    6,
                ),
            }
    behavior_survival: Dict[str, object] = {
        "available": False,
        "minimal_profile": minimal_profile,
        "survival_threshold": MINIMAL_SHAPING_SURVIVAL_THRESHOLD,
        "scenario_count": 0,
        "surviving_scenario_count": 0,
        "survival_rate": 0.0,
        "scenarios": {},
    }
    if minimal_profile is not None:
        minimal_payload = profile_payloads.get(minimal_profile, {})
        minimal_suite = (
            comparison_suite_from_payload(minimal_payload)
            if isinstance(minimal_payload, dict)
            else {}
        )
        if isinstance(minimal_suite, dict) and minimal_suite:
            scenario_names = sorted(
                str(scenario_name)
                for scenario_name in minimal_suite.keys()
            )
            scenario_payloads: Dict[str, object] = {}
            for scenario_name in scenario_names:
                minimal_scenario = minimal_suite.get(scenario_name, {})
                if not isinstance(minimal_scenario, dict):
                    minimal_scenario = {}
                austere_success_rate = safe_float(
                    minimal_scenario.get("success_rate")
                )
                scenario_payloads[scenario_name] = {
                    "austere_success_rate": round(austere_success_rate, 6),
                    "survives": austere_success_rate >= MINIMAL_SHAPING_SURVIVAL_THRESHOLD,
                    "episodes": int(
                        safe_float(minimal_scenario.get("episodes"))
                    ),
                }
            surviving_count = sum(
                1
                for payload in scenario_payloads.values()
                if payload.get("survives")
            )
            scenario_count = len(scenario_payloads)
            behavior_survival = {
                "available": scenario_count > 0,
                "minimal_profile": minimal_profile,
                "survival_threshold": MINIMAL_SHAPING_SURVIVAL_THRESHOLD,
                "scenario_count": scenario_count,
                "surviving_scenario_count": surviving_count,
                "survival_rate": round(
                    float(surviving_count / scenario_count)
                    if scenario_count
                    else 0.0,
                    6,
                ),
                "scenarios": scenario_payloads,
            }
    comparison_result: Dict[str, object] = {
        "minimal_profile": minimal_profile,
        "behavior_survival": behavior_survival,
        "deltas_vs_minimal": deltas_vs_minimal,
    }
    gap_policy_check = validate_gap_policy(comparison_result)
    survival_summary = austere_survival_summary(
        behavior_survival,
        gap_policy_check,
    )
    dependent_behaviors = shaping_dependent_behaviors(
        profile_payloads,
        minimal_profile=minimal_profile,
    )
    comparison_result.update(
        {
            "austere_survival_summary": survival_summary,
            "gap_policy_check": gap_policy_check,
            "shaping_dependent_behaviors": dependent_behaviors,
            "profiles": {
                name: {
                    key: round(float(value), 6)
                    for key, value in sorted(metrics.items())
                }
                for name, metrics in sorted(profiles.items())
            },
        }
    )
    comparison_result.update({
        "survival_rate": behavior_survival["survival_rate"],
        "notes": [
            "scenario_success_rate and episode_success_rate mirror the existing comparison payload.",
            "mean_reward is derived from the summary when available or from the corresponding compact aggregate.",
            "behavior_survival treats a scenario as surviving minimal shaping when austere success_rate reaches the configured survival threshold.",
            "gap_policy_check evaluates dense-vs-austere deltas against the shaping-reduction policy.",
        ],
    })
    return comparison_result


def build_reward_audit(
    *,
    current_profile: str | None = None,
    comparison_payload: Dict[str, object] | None = None,
) -> Dict[str, object]:
    """
    Build a structured audit of reward shaping and leakage signals for the simulation.

    This produces a dictionary summarizing the current reward profile, a minimal baseline
    marker (when available), per-component reward audits, observation- and memory-leakage
    signals, per-profile audits for every known reward profile, and human-readable notes.
    If a comparison payload is supplied, a `comparison` block with computed deltas is
    included.

    Parameters:
        current_profile (str | None): Optional name of the currently selected reward profile
            to record in the audit. May be None.
        comparison_payload (dict | None): Optional payload used to compute comparison deltas.
            When provided, a `comparison` entry will be added to the returned audit.

    Returns:
        dict: Audit payload containing keys:
          - "current_profile": the provided `current_profile` value
          - "minimal_profile": name of an austere baseline when available, else None
          - "reward_components": per-component audit results
          - "observation_signals": observation-leakage audit results
          - "memory_signals": memory-leakage audit results
          - "reward_profiles": per-profile audit entries for all known profiles
          - "notes": explanatory notes
          - "comparison" (optional): computed comparison block when `comparison_payload` was provided
    """
    audit = {
        "current_profile": current_profile,
        "minimal_profile": None,
        "reward_components": reward_component_audit(),
        "observation_signals": observation_leakage_audit(),
        "memory_signals": memory_leakage_audit(),
        "reward_profiles": {
            name: reward_profile_audit(name)
            for name in sorted(REWARD_PROFILES)
        },
        "notes": [
            "The audit distinguishes event shaping, progress shaping, and internal pressure.",
            "Observation and memory signals classified as privileged/world_derived should be interpreted cautiously in benchmarks.",
        ],
    }
    comparison = build_reward_audit_comparison(comparison_payload)
    if comparison is not None:
        audit["minimal_profile"] = comparison.get("minimal_profile")
        audit["comparison"] = comparison
    elif "austere" in REWARD_PROFILES:
        audit["minimal_profile"] = "austere"
    return audit


def safe_float(value: object) -> float:
    """
    Safely convert a value to float, returning 0.0 for invalid inputs.

    Parameters:
        value: Any value to attempt conversion to float.

    Returns:
        float: The converted float value, or 0.0 if conversion fails.
    """
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0


def aggregate_with_uncertainty(
    seed_results: Sequence[SeedLevelResult | Dict[str, object] | tuple[int, float]],
    *,
    confidence_level: float = 0.95,
    n_resamples: int = 1000,
) -> Dict[str, object]:
    """Aggregate seed-level values into a JSON-safe uncertainty estimate."""
    parsed: list[tuple[int, float]] = []
    for item in seed_results:
        seed: object
        value: object
        if isinstance(item, SeedLevelResult):
            seed = item.seed
            value = item.value
        elif isinstance(item, dict):
            seed = item.get("seed")
            value = item.get("value")
        else:
            seed, value = item
        try:
            numeric_value = float(value)
            if not math.isfinite(numeric_value):
                continue
            parsed.append((int(seed), numeric_value))
        except (TypeError, ValueError):
            continue
    if not parsed:
        return UncertaintyEstimate(
            mean=0.0,
            ci_lower=0.0,
            ci_upper=0.0,
            std_error=0.0,
            n_seeds=0,
            confidence_level=confidence_level,
            seed_values=(),
        ).to_dict()
    values = tuple(value for _, value in parsed)
    mean, ci_lower, ci_upper, std_error = bootstrap_confidence_interval(
        values,
        confidence_level=confidence_level,
        n_resamples=n_resamples,
    )
    return UncertaintyEstimate(
        mean=mean,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        std_error=std_error,
        n_seeds=len(parsed),
        confidence_level=confidence_level,
        seed_values=values,
    ).to_dict()


def metric_seed_values_from_payload(
    payload: Dict[str, object] | None,
    *,
    metric_name: str,
    scenario: str | None = None,
    fallback_value: object | None = None,
) -> list[tuple[int, float]]:
    """Read seed-level metric values from a compact behavior payload."""
    if not isinstance(payload, dict):
        return []
    rows: list[tuple[int, float]] = []
    row_sources: list[object] = []
    if scenario is not None:
        suite = payload.get("suite", {})
        if isinstance(suite, dict):
            scenario_payload = suite.get(scenario, {})
            if isinstance(scenario_payload, dict):
                row_sources.append(scenario_payload.get("seed_level", []))
    row_sources.append(payload.get("seed_level", []))
    for source in row_sources:
        if not isinstance(source, list):
            continue
        for item in source:
            if not isinstance(item, dict):
                continue
            if str(item.get("metric_name", "")) != metric_name:
                continue
            item_scenario = item.get("scenario")
            if scenario is None and item_scenario is not None:
                continue
            if scenario is not None and str(item_scenario) != str(scenario):
                continue
            try:
                value = float(item.get("value"))
                if not math.isfinite(value):
                    continue
                rows.append((int(item.get("seed")), value))
            except (TypeError, ValueError):
                continue
    if rows:
        deduped: dict[int, float] = {}
        for seed, value in rows:
            deduped[seed] = value
        return sorted(deduped.items())
    return []


def fallback_seed_values(value: object | None) -> list[tuple[int, float]]:
    if value is None:
        return []
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return []
    if not math.isfinite(numeric_value):
        return []
    return [(0, numeric_value)]


def paired_seed_delta_rows(
    reference_values: Sequence[tuple[int, float]],
    comparison_values: Sequence[tuple[int, float]],
    *,
    metric_name: str,
    condition: str,
    fallback_delta: object | None = None,
    scenario: str | None = None,
) -> list[SeedLevelResult]:
    reference_by_seed = {int(seed): float(value) for seed, value in reference_values}
    comparison_by_seed = {
        int(seed): float(value) for seed, value in comparison_values
    }
    common_seeds = sorted(set(reference_by_seed) & set(comparison_by_seed))
    rows = [
        SeedLevelResult(
            metric_name=metric_name,
            seed=seed,
            value=round(comparison_by_seed[seed] - reference_by_seed[seed], 6),
            condition=condition,
            scenario=scenario,
        )
        for seed in common_seeds
    ]
    if rows:
        return rows
    return [
        SeedLevelResult(
            metric_name=metric_name,
            seed=seed,
            value=value,
            condition=condition,
            scenario=scenario,
        )
        for seed, value in fallback_seed_values(fallback_delta)
    ]


def paired_seed_effect_size_rows(
    reference_values: Sequence[tuple[int, float]],
    comparison_values: Sequence[tuple[int, float]],
    *,
    condition: str,
    point_effect_size: float,
) -> list[SeedLevelResult]:
    reference_by_seed = {int(seed): float(value) for seed, value in reference_values}
    comparison_by_seed = {
        int(seed): float(value) for seed, value in comparison_values
    }
    common_seeds = sorted(set(reference_by_seed) & set(comparison_by_seed))
    if not common_seeds:
        return [
            SeedLevelResult(
                metric_name="cohens_d",
                seed=seed,
                value=value,
                condition=condition,
            )
            for seed, value in fallback_seed_values(point_effect_size)
        ]

    reference_group = values_only(reference_values)
    comparison_group = values_only(comparison_values)
    if not reference_group or not comparison_group:
        return []
    mean_delta = (
        sum(comparison_group) / len(comparison_group)
        - sum(reference_group) / len(reference_group)
    )
    if point_effect_size == 0.0 or mean_delta == 0.0:
        standardized_values = {seed: 0.0 for seed in common_seeds}
    else:
        pooled_std = abs(mean_delta / point_effect_size)
        if pooled_std == 0.0 or not math.isfinite(pooled_std):
            standardized_values = {seed: 0.0 for seed in common_seeds}
        else:
            standardized_values = {
                seed: round(
                    (comparison_by_seed[seed] - reference_by_seed[seed])
                    / pooled_std,
                    6,
                )
                for seed in common_seeds
            }
    return [
        SeedLevelResult(
            metric_name="cohens_d",
            seed=seed,
            value=standardized_values[seed],
            condition=condition,
        )
        for seed in common_seeds
    ]


def values_only(seed_values: Sequence[tuple[int, float]]) -> list[float]:
    return [float(value) for _, value in seed_values]


def seed_level_dicts(rows: Sequence[SeedLevelResult]) -> list[Dict[str, object]]:
    return [row.to_dict() for row in rows]


def behavior_metric_seed_rows(
    seed_payloads: Sequence[tuple[int, Dict[str, object]]],
    *,
    metric_name: str,
    condition: str,
    scenario: str | None = None,
) -> list[SeedLevelResult]:
    rows: list[SeedLevelResult] = []
    for seed, payload in seed_payloads:
        if scenario is None:
            summary = condition_compact_summary(payload)
            if metric_name not in summary:
                continue
            value = summary[metric_name]
        else:
            suite = payload.get("suite", {})
            if not isinstance(suite, dict):
                continue
            scenario_payload = suite.get(scenario, {})
            if not isinstance(scenario_payload, dict):
                continue
            value = safe_float(scenario_payload.get("success_rate"))
        rows.append(
            SeedLevelResult(
                metric_name=metric_name,
                seed=seed,
                value=value,
                condition=condition,
                scenario=scenario,
            )
        )
    return rows


def attach_behavior_seed_statistics(
    payload: Dict[str, object],
    seed_payloads: Sequence[tuple[int, Dict[str, object]]],
    *,
    condition: str,
    scenario_names: Sequence[str],
) -> None:
    """Attach seed-level rows and uncertainty estimates to a behavior payload."""
    seed_level_rows: list[SeedLevelResult] = []
    uncertainty: Dict[str, object] = {}
    for metric_name in (
        "scenario_success_rate",
        "episode_success_rate",
        "mean_reward",
    ):
        metric_rows = behavior_metric_seed_rows(
            seed_payloads,
            metric_name=metric_name,
            condition=condition,
        )
        seed_level_rows.extend(metric_rows)
        uncertainty[metric_name] = aggregate_with_uncertainty(metric_rows)
    specialization_rows = [
        SeedLevelResult(
            metric_name="specialization_score",
            seed=seed,
            value=predator_type_specialization_score(seed_payload),
            condition=condition,
        )
        for seed, seed_payload in seed_payloads
    ]
    if specialization_rows:
        seed_level_rows.extend(specialization_rows)
        uncertainty["specialization_score"] = aggregate_with_uncertainty(
            specialization_rows
        )
    suite = payload.get("suite", {})
    if isinstance(suite, dict):
        for scenario_name in scenario_names:
            scenario_payload = suite.get(scenario_name)
            if not isinstance(scenario_payload, dict):
                continue
            scenario_rows = behavior_metric_seed_rows(
                seed_payloads,
                metric_name="scenario_success_rate",
                condition=condition,
                scenario=scenario_name,
            )
            seed_level_rows.extend(scenario_rows)
            scenario_payload["seed_level"] = seed_level_dicts(scenario_rows)
            scenario_payload["uncertainty"] = {
                "success_rate": aggregate_with_uncertainty(scenario_rows),
            }
    payload["seed_level"] = seed_level_dicts(seed_level_rows)
    payload["uncertainty"] = uncertainty


def module_response_by_predator_type_from_payload(
    payload: Dict[str, object] | None,
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate module response mappings by predator type from suite or legacy payloads.

    Accepts compact behavior payloads with legacy scenario summaries or suite
    entries and returns rounded mean module responses per predator type.
    """
    if not isinstance(payload, dict):
        return {}
    sources: list[object] = []
    legacy_scenarios = payload.get("legacy_scenarios", {})
    if isinstance(legacy_scenarios, dict):
        sources.extend(legacy_scenarios.values())
    suite = payload.get("suite", {})
    if isinstance(suite, dict):
        for scenario_payload in suite.values():
            if isinstance(scenario_payload, dict):
                legacy_metrics = scenario_payload.get("legacy_metrics")
                if isinstance(legacy_metrics, dict):
                    sources.append(legacy_metrics)
                sources.append(scenario_payload)
    grouped: Dict[str, Dict[str, list[float]]] = {}
    for source in sources:
        if not isinstance(source, dict):
            continue
        response = (
            source.get("mean_module_response_by_predator_type")
            or source.get("module_response_by_predator_type")
        )
        if not isinstance(response, dict):
            continue
        for predator_type, module_values in response.items():
            if not isinstance(module_values, dict):
                continue
            predator_key = str(predator_type)
            grouped.setdefault(predator_key, {})
            for module_name, value in module_values.items():
                numeric_value = safe_float(value)
                grouped[predator_key].setdefault(str(module_name), []).append(
                    numeric_value
                )
    return {
        predator_type: {
            module_name: round(sum(values) / len(values), 6)
            for module_name, values in sorted(module_values.items())
            if values
        }
        for predator_type, module_values in sorted(grouped.items())
    }


def representation_specialization_from_payload(
    payload: Dict[str, object] | None,
) -> Dict[str, object]:
    """
    Extract reference-side representation specialization evidence from a payload.

    Top-level aggregate metrics are preferred. When they are absent, this
    falls back to arithmetic means across suite-level legacy metrics. Cross-
    tier mismatches matter during interpretation: strong behavioral
    specialization with weak representation separation suggests downstream
    arbitration or scenario asymmetry, while emerging representation
    separation with weak behavior suggests internal differentiation that has
    not yet stabilized into robust policy outcomes.
    """

    def _mapping_metric(
        source: Dict[str, object],
        mean_key: str,
        raw_key: str,
    ) -> tuple[Dict[str, float], bool]:
        for key in (mean_key, raw_key):
            if key not in source:
                continue
            value = source.get(key)
            if value is None:
                continue
            if not isinstance(value, Mapping) or len(value) == 0:
                continue
            normalized = {
                str(name): round(safe_float(metric), 6)
                for name, metric in value.items()
            }
            if normalized:
                return normalized, True
        return {}, False

    def _scalar_metric(
        source: Dict[str, object],
        mean_key: str,
        raw_key: str,
    ) -> tuple[float, bool]:
        for key in (mean_key, raw_key):
            if key in source:
                value = source.get(key)
                if value is None:
                    continue
                return round(safe_float(value), 6), True
        return 0.0, False

    def _from_source(source: Dict[str, object]) -> Dict[str, object]:
        proposer_divergence, has_proposer = _mapping_metric(
            source,
            "mean_proposer_divergence_by_module",
            "proposer_divergence_by_module",
        )
        gate_differential, has_gate = _mapping_metric(
            source,
            "mean_action_center_gate_differential",
            "action_center_gate_differential",
        )
        contribution_differential, has_contribution = _mapping_metric(
            source,
            "mean_action_center_contribution_differential",
            "action_center_contribution_differential",
        )
        score, has_score = _scalar_metric(
            source,
            "mean_representation_specialization_score",
            "representation_specialization_score",
        )
        available = bool(
            has_proposer or has_gate or has_contribution or has_score
        )
        return {
            "available": available,
            "proposer_divergence_by_module": proposer_divergence,
            "action_center_gate_differential": gate_differential,
            "action_center_contribution_differential": (
                contribution_differential
            ),
            "representation_specialization_score": score,
        }

    if not isinstance(payload, dict):
        return {
            "available": False,
            "proposer_divergence_by_module": {},
            "action_center_gate_differential": {},
            "action_center_contribution_differential": {},
            "representation_specialization_score": 0.0,
        }

    top_level = _from_source(payload)
    if bool(top_level["available"]):
        return top_level

    sources: list[Dict[str, object]] = []
    legacy_scenarios = payload.get("legacy_scenarios", {})
    if isinstance(legacy_scenarios, dict):
        for source in legacy_scenarios.values():
            if isinstance(source, dict):
                sources.append(source)
    suite = payload.get("suite", {})
    if isinstance(suite, dict):
        for scenario_payload in suite.values():
            if not isinstance(scenario_payload, dict):
                continue
            legacy_metrics = scenario_payload.get("legacy_metrics")
            if isinstance(legacy_metrics, dict):
                sources.append(legacy_metrics)
            sources.append(scenario_payload)

    proposer_grouped: Dict[str, list[float]] = {}
    gate_grouped: Dict[str, list[float]] = {}
    contribution_grouped: Dict[str, list[float]] = {}
    scores: list[float] = []
    for source in sources:
        source_metrics = _from_source(source)
        if not bool(source_metrics["available"]):
            continue
        for name, value in dict(
            source_metrics["proposer_divergence_by_module"]
        ).items():
            proposer_grouped.setdefault(str(name), []).append(
                safe_float(value)
            )
        for name, value in dict(
            source_metrics["action_center_gate_differential"]
        ).items():
            gate_grouped.setdefault(str(name), []).append(safe_float(value))
        for name, value in dict(
            source_metrics["action_center_contribution_differential"]
        ).items():
            contribution_grouped.setdefault(str(name), []).append(
                safe_float(value)
            )
        scores.append(
            safe_float(
                source_metrics["representation_specialization_score"]
            )
        )

    if not proposer_grouped and not gate_grouped and not contribution_grouped and not scores:
        return top_level

    return {
        "available": True,
        "proposer_divergence_by_module": {
            name: round(sum(values) / len(values), 6)
            for name, values in sorted(proposer_grouped.items())
            if values
        },
        "action_center_gate_differential": {
            name: round(sum(values) / len(values), 6)
            for name, values in sorted(gate_grouped.items())
            if values
        },
        "action_center_contribution_differential": {
            name: round(sum(values) / len(values), 6)
            for name, values in sorted(contribution_grouped.items())
            if values
        },
        "representation_specialization_score": round(
            sum(scores) / len(scores),
            6,
        )
        if scores
        else 0.0,
    }


def predator_type_specialization_score(
    payload: Dict[str, object] | None,
) -> float:
    """
    Score visual-vs-olfactory response specialization from module responses.

    Returns 0.0 when either predator type lacks positive response data; otherwise
    returns a bounded rounded score based on visual and sensory cortex contrast.
    """
    response = module_response_by_predator_type_from_payload(payload)
    visual_modules = response.get("visual", {})
    olfactory_modules = response.get("olfactory", {})
    has_visual_data = any(safe_float(value) > 0.0 for value in visual_modules.values())
    has_olfactory_data = any(
        safe_float(value) > 0.0 for value in olfactory_modules.values()
    )
    if not has_visual_data or not has_olfactory_data:
        return 0.0
    visual_visual = safe_float(visual_modules.get("visual_cortex"))
    visual_sensory = safe_float(visual_modules.get("sensory_cortex"))
    olfactory_visual = safe_float(olfactory_modules.get("visual_cortex"))
    olfactory_sensory = safe_float(olfactory_modules.get("sensory_cortex"))
    return round(
        max(
            0.0,
            min(
                1.0,
                (
                    max(0.0, visual_visual - olfactory_visual)
                    + max(0.0, olfactory_sensory - visual_sensory)
                )
                / 2.0,
            ),
        ),
        6,
    )


def visual_minus_olfactory_seed_rows(
    payload: Dict[str, object] | None,
    *,
    condition: str,
    metric_name: str = "visual_minus_olfactory_delta",
    fallback_value: object | None = None,
) -> list[SeedLevelResult]:
    """
    Build per-seed visual-minus-olfactory success deltas from scenario seed rows.

    Reads scenario_success_rate seed_level entries, groups them by visual and
    olfactory predator scenario families, and falls back only when no real rows exist.
    """
    if not isinstance(payload, dict):
        return [
            SeedLevelResult(
                metric_name=metric_name,
                seed=seed,
                value=value,
                condition=condition,
            )
            for seed, value in fallback_seed_values(fallback_value)
        ]
    visual_scenarios = set(resolve_ablation_scenario_group("visual_predator_scenarios"))
    olfactory_scenarios = set(
        resolve_ablation_scenario_group("olfactory_predator_scenarios")
    )
    by_seed: Dict[int, Dict[str, list[float]]] = {}
    for item in payload.get("seed_level", []):
        if not isinstance(item, dict):
            continue
        if str(item.get("metric_name", "")) != "scenario_success_rate":
            continue
        scenario = item.get("scenario")
        if scenario is None:
            continue
        scenario_name = str(scenario)
        bucket = None
        if scenario_name in visual_scenarios:
            bucket = "visual"
        if scenario_name in olfactory_scenarios:
            by_seed.setdefault(int(item.get("seed", 0)), {}).setdefault(
                "olfactory",
                [],
            )
            try:
                value = float(item.get("value"))
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                by_seed[int(item.get("seed", 0))]["olfactory"].append(value)
        if bucket is None:
            continue
        try:
            value = float(item.get("value"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            by_seed.setdefault(int(item.get("seed", 0)), {}).setdefault(
                bucket,
                [],
            ).append(value)
    rows = []
    for seed, buckets in sorted(by_seed.items()):
        visual_values = buckets.get("visual", [])
        olfactory_values = buckets.get("olfactory", [])
        if not visual_values or not olfactory_values:
            continue
        rows.append(
            SeedLevelResult(
                metric_name=metric_name,
                seed=seed,
                value=round(
                    (sum(visual_values) / len(visual_values))
                    - (sum(olfactory_values) / len(olfactory_values)),
                    6,
                ),
                condition=condition,
            )
        )
    if rows:
        return rows
    return [
        SeedLevelResult(
            metric_name=metric_name,
            seed=seed,
            value=value,
            condition=condition,
        )
        for seed, value in fallback_seed_values(fallback_value)
    ]


def build_predator_type_specialization_summary(
    variants: Dict[str, Dict[str, object]],
    *,
    reference_variant: str,
    deltas_vs_reference: Dict[str, object],
) -> Dict[str, object]:
    """
    Return predator-type specialization comparison, seed rows, and uncertainty.

    Composes the ablation comparison summary with visual-minus-olfactory
    deltas, specialization scores, and uncertainty blocks per variant.
    """
    comparison_summary = compare_predator_type_ablation_performance(
        {
            "variants": variants,
            "deltas_vs_reference": deltas_vs_reference,
        }
    )
    comparisons = comparison_summary.get("comparisons", {})
    uncertainty: Dict[str, object] = {}
    seed_level: list[Dict[str, object]] = []
    reference_payload = variants.get(reference_variant)
    reference_rows = visual_minus_olfactory_seed_rows(
        reference_payload,
        condition=reference_variant,
        metric_name="visual_minus_olfactory_success_rate",
        fallback_value=(
            dict(comparisons.get(reference_variant, {})).get(
                "visual_minus_olfactory_success_rate"
            )
            if isinstance(comparisons, dict)
            else None
        ),
    )
    for variant_name, payload in variants.items():
        comparison_payload = (
            dict(comparisons.get(variant_name, {}))
            if isinstance(comparisons, dict)
            and isinstance(comparisons.get(variant_name), dict)
            else {}
        )
        visual_rows = visual_minus_olfactory_seed_rows(
            payload,
            condition=variant_name,
            metric_name="visual_minus_olfactory_success_rate",
            fallback_value=comparison_payload.get(
                "visual_minus_olfactory_success_rate"
            ),
        )
        specialization_rows = [
            SeedLevelResult(
                metric_name="specialization_score",
                seed=seed,
                value=value,
                condition=variant_name,
            )
            for seed, value in metric_seed_values_from_payload(
                payload,
                metric_name="specialization_score",
                fallback_value=predator_type_specialization_score(payload),
            )
        ]
        delta_rows = paired_seed_delta_rows(
            [(row.seed, row.value) for row in reference_rows],
            [(row.seed, row.value) for row in visual_rows],
            metric_name="visual_minus_olfactory_success_rate_delta",
            condition=variant_name,
            fallback_delta=comparison_payload.get(
                "visual_minus_olfactory_success_rate_delta"
            ),
        )
        seed_level.extend(
            seed_level_dicts([*visual_rows, *specialization_rows, *delta_rows])
        )
        uncertainty[variant_name] = {
            "visual_minus_olfactory_success_rate": (
                aggregate_with_uncertainty(visual_rows)
            ),
            "visual_minus_olfactory_success_rate_delta": (
                aggregate_with_uncertainty(delta_rows)
            ),
            "specialization_score": aggregate_with_uncertainty(
                specialization_rows
            ),
        }
    return {
        **comparison_summary,
        "reference_variant": reference_variant,
        "seed_level": seed_level,
        "uncertainty": uncertainty,
    }


def condition_compact_summary(
    payload: Dict[str, object] | None,
) -> Dict[str, float]:
    """
    Extract a compact numeric summary of scenario/episode success rates and mean reward from a behavior-suite payload.

    Parameters:
        payload (Dict[str, object] | None): A behavior-suite payload (expected to contain a top-level "summary" mapping) or None.

    Returns:
        Dict[str, float]: A mapping with keys `"scenario_success_rate"`, `"episode_success_rate"`, and `"mean_reward"`, each cast to float. Missing or malformed input yields zeros for all three fields.
    """
    if not isinstance(payload, dict):
        return {
            "scenario_success_rate": 0.0,
            "episode_success_rate": 0.0,
            "mean_reward": 0.0,
        }
    summary = payload.get("summary", {})
    if not isinstance(summary, dict):
        return {
            "scenario_success_rate": 0.0,
            "episode_success_rate": 0.0,
            "mean_reward": 0.0,
        }
    return {
        "scenario_success_rate": safe_float(
            summary.get("scenario_success_rate", 0.0)
        ),
        "episode_success_rate": safe_float(
            summary.get("episode_success_rate", 0.0)
        ),
        "mean_reward": condition_mean_reward(payload),
    }


def condition_mean_reward(payload: Dict[str, object] | None) -> float:
    """
    Resolve a condition payload's mean reward, preferring `summary["mean_reward"]`
    and falling back to the compact payload's `legacy_scenarios` aggregate.
    """
    if not isinstance(payload, dict):
        return 0.0
    summary = payload.get("summary", {})
    if isinstance(summary, dict):
        summary_mean_reward = summary.get("mean_reward")
        if summary_mean_reward is not None:
            try:
                return float(summary_mean_reward)
            except (TypeError, ValueError):
                pass
    try:
        return float(mean_reward_from_behavior_payload(payload))
    except (TypeError, ValueError):
        return 0.0


def build_learning_evidence_deltas(
    conditions: Dict[str, Dict[str, object]],
    *,
    reference_condition: str,
    scenario_names: Sequence[str],
) -> Dict[str, object]:
    """
    Compute numeric deltas for each learning-evidence condition relative to a reference condition.

    Calculates deltas for overall suite summary fields (`scenario_success_rate`, `episode_success_rate`, `mean_reward`)
    and per-scenario `success_rate`. All numeric deltas are (condition - reference) and rounded to 6 decimal places.
    Conditions marked as skipped (or missing a `summary`) are represented with `{"skipped": True, "reason": <str>}`.

    Parameters:
        conditions (Dict[str, Dict[str, object]]): Mapping from condition name to its evaluation payload. Each payload
            is expected to contain a `summary` dict with top-level metrics and a `suite` dict keyed by scenario name.
        reference_condition (str): Name of the condition to use as the reference baseline for delta computation.
        scenario_names (Sequence[str]): Ordered list of scenario names to include for per-scenario `success_rate` deltas.

    Returns:
        Dict[str, object]: Mapping from condition name to a delta payload. For non-skipped conditions the payload has:
            {
                "summary": {
                    "scenario_success_rate_delta": float,
                    "episode_success_rate_delta": float,
                    "mean_reward_delta": float
                },
                "scenarios": {
                    "<scenario_name>": {"success_rate_delta": float}, ...
                }
            }
        Skipped conditions return:
            {"skipped": True, "reason": <string>}
    """
    deltas: Dict[str, object] = {}
    reference = conditions.get(reference_condition)
    reference_missing_or_skipped = (
        not isinstance(reference, dict) or bool(reference.get("skipped"))
    )
    if reference_missing_or_skipped:
        for condition_name in conditions:
            deltas[condition_name] = {
                "skipped": True,
                "reason": (
                    f"reference condition {reference_condition!r} missing or skipped"
                ),
            }
        return deltas
    reference_summary = dict(reference.get("summary", {}))
    reference_suite = dict(reference.get("suite", {}))
    reference_mean_reward = condition_mean_reward(reference)
    for condition_name, payload in conditions.items():
        if bool(payload.get("skipped")) or "summary" not in payload:
            deltas[condition_name] = {
                "skipped": True,
                "reason": str(payload.get("reason", "")),
            }
            continue
        summary = dict(payload.get("summary", {}))
        suite = dict(payload.get("suite", {}))
        mean_reward = condition_mean_reward(payload)
        summary_deltas = {
            "scenario_success_rate_delta": round(
                float(summary.get("scenario_success_rate", 0.0))
                - float(reference_summary.get("scenario_success_rate", 0.0)),
                6,
            ),
            "episode_success_rate_delta": round(
                float(summary.get("episode_success_rate", 0.0))
                - float(reference_summary.get("episode_success_rate", 0.0)),
                6,
            ),
            "mean_reward_delta": round(
                float(mean_reward) - float(reference_mean_reward),
                6,
            ),
        }
        summary_uncertainty: Dict[str, object] = {}
        summary_seed_level: list[Dict[str, object]] = []
        for metric_name, delta_name in (
            ("scenario_success_rate", "scenario_success_rate_delta"),
            ("episode_success_rate", "episode_success_rate_delta"),
            ("mean_reward", "mean_reward_delta"),
        ):
            reference_seed_values = metric_seed_values_from_payload(
                reference,
                metric_name=metric_name,
                fallback_value=reference_summary.get(metric_name)
                if metric_name != "mean_reward"
                else reference_mean_reward,
            )
            comparison_seed_values = metric_seed_values_from_payload(
                payload,
                metric_name=metric_name,
                fallback_value=summary.get(metric_name)
                if metric_name != "mean_reward"
                else mean_reward,
            )
            seed_rows = paired_seed_delta_rows(
                reference_seed_values,
                comparison_seed_values,
                metric_name=delta_name,
                condition=condition_name,
                fallback_delta=summary_deltas[delta_name],
            )
            summary_seed_level.extend(seed_level_dicts(seed_rows))
            summary_uncertainty[delta_name] = (
                aggregate_with_uncertainty(seed_rows)
            )
        scenario_deltas: Dict[str, object] = {}
        for scenario_name in scenario_names:
            scenario_payload = dict(suite.get(scenario_name, {}))
            reference_scenario_payload = dict(reference_suite.get(scenario_name, {}))
            delta_value = round(
                float(scenario_payload.get("success_rate", 0.0))
                - float(reference_scenario_payload.get("success_rate", 0.0)),
                6,
            )
            scenario_seed_rows = paired_seed_delta_rows(
                metric_seed_values_from_payload(
                    reference,
                    metric_name="scenario_success_rate",
                    scenario=scenario_name,
                    fallback_value=reference_scenario_payload.get("success_rate"),
                ),
                metric_seed_values_from_payload(
                    payload,
                    metric_name="scenario_success_rate",
                    scenario=scenario_name,
                    fallback_value=scenario_payload.get("success_rate"),
                ),
                metric_name="success_rate_delta",
                condition=condition_name,
                fallback_delta=delta_value,
                scenario=scenario_name,
            )
            scenario_deltas[scenario_name] = {
                "success_rate_delta": delta_value,
                "seed_level": seed_level_dicts(
                    scenario_seed_rows
                ),
                "uncertainty": aggregate_with_uncertainty(
                    scenario_seed_rows
                ),
            }
        deltas[condition_name] = {
            "summary": summary_deltas,
            "scenarios": scenario_deltas,
            "seed_level": summary_seed_level,
            "uncertainty": summary_uncertainty,
        }
    return deltas


def build_learning_evidence_summary(
    conditions: Dict[str, Dict[str, object]],
    *,
    reference_condition: str,
) -> Dict[str, object]:
    """
    Build a compact summary comparing learning-evidence conditions against a reference condition.

    Parameters:
        conditions (Dict[str, Dict[str, object]]): Mapping from condition name to its compact payload; a payload may include a `"skipped"` flag.
        reference_condition (str): Name of the condition to use as the primary trained reference. For the canonical workflow this is `trained_without_reflex_support`.

    Returns:
        Dict[str, object]: Summary containing:
            - reference_condition: the provided reference name.
            - primary_gate_metric: the metric used for the primary evidence gate ("scenario_success_rate").
            - supports_primary_evidence: `true` if `reference_condition`, `random_init`, and `reflex_only` are present and not skipped.
            - has_learning_evidence: `true` if `scenario_success_rate` for the no-reflex reference exceeds both `random_init` and `reflex_only`.
            - trained_final, random_init, reflex_only, trained_without_reflex_support: compact summaries for each condition (zeroed defaults when missing).
            - trained_vs_random_init, trained_vs_reflex_only: per-metric deltas (`scenario_success_rate`, `episode_success_rate`, `mean_reward`) computed as no-reflex reference minus comparator and rounded to 6 decimals.
            - notes: list of explanatory messages about gating, supporting metrics, and reflex-only availability.
    """
    reference = condition_compact_summary(conditions.get(reference_condition))
    trained_final = condition_compact_summary(conditions.get("trained_final"))
    random_init = condition_compact_summary(conditions.get("random_init"))
    reflex_only = condition_compact_summary(conditions.get("reflex_only"))
    trained_without_reflex_support = condition_compact_summary(
        conditions.get("trained_without_reflex_support")
    )
    reference_available = (
        reference_condition in conditions
        and not bool(conditions[reference_condition].get("skipped"))
    )
    random_init_available = (
        "random_init" in conditions
        and not bool(conditions["random_init"].get("skipped"))
    )
    reflex_only_available = (
        "reflex_only" in conditions
        and not bool(conditions["reflex_only"].get("skipped"))
    )
    primary_supported = (
        reference_available
        and random_init_available
        and reflex_only_available
    )
    trained_vs_random = {
        "scenario_success_rate_delta": round(
            reference["scenario_success_rate"]
            - random_init["scenario_success_rate"],
            6,
        ),
        "episode_success_rate_delta": round(
            reference["episode_success_rate"]
            - random_init["episode_success_rate"],
            6,
        ),
        "mean_reward_delta": round(
            reference["mean_reward"] - random_init["mean_reward"],
            6,
        ),
    }
    trained_vs_reflex = {
        "scenario_success_rate_delta": round(
            reference["scenario_success_rate"]
            - reflex_only["scenario_success_rate"],
            6,
        ),
        "episode_success_rate_delta": round(
            reference["episode_success_rate"]
            - reflex_only["episode_success_rate"],
            6,
        ),
        "mean_reward_delta": round(
            reference["mean_reward"] - reflex_only["mean_reward"],
            6,
        ),
    }
    notes = [
        f"Primary evidence uses {reference_condition} and only scenario_success_rate as the gate.",
        "episode_success_rate and mean_reward are included only as supporting documentation.",
        "trained_final is retained as a default-runtime diagnostic and does not drive the primary gate.",
    ]
    if not reflex_only_available:
        notes.append(
            "The reflex_only condition is not available for the current architecture."
        )
    has_learning_evidence = (
        primary_supported
        and reference["scenario_success_rate"] > random_init["scenario_success_rate"]
        and reference["scenario_success_rate"] > reflex_only["scenario_success_rate"]
    )
    condition_seed_level = {
        condition_name: list(payload.get("seed_level", []))
        for condition_name, payload in conditions.items()
        if isinstance(payload, dict)
    }
    condition_uncertainty = {
        condition_name: dict(payload.get("uncertainty", {}))
        for condition_name, payload in conditions.items()
        if isinstance(payload, dict)
    }

    def _delta_uncertainty(
        comparison_name: str,
        comparison_summary: Dict[str, float],
        deltas: Dict[str, float],
    ) -> Dict[str, object]:
        comparison_payload = conditions.get(comparison_name)
        reference_payload = conditions.get(reference_condition)
        result: Dict[str, object] = {}
        for metric_name, delta_name in (
            ("scenario_success_rate", "scenario_success_rate_delta"),
            ("episode_success_rate", "episode_success_rate_delta"),
            ("mean_reward", "mean_reward_delta"),
        ):
            reference_seed_values = metric_seed_values_from_payload(
                reference_payload,
                metric_name=metric_name,
                fallback_value=reference.get(metric_name),
            )
            comparison_seed_values = metric_seed_values_from_payload(
                comparison_payload,
                metric_name=metric_name,
                fallback_value=comparison_summary.get(metric_name),
            )
            seed_rows = paired_seed_delta_rows(
                comparison_seed_values,
                reference_seed_values,
                metric_name=delta_name,
                condition=comparison_name,
                fallback_delta=deltas.get(delta_name),
            )
            result[delta_name] = aggregate_with_uncertainty(seed_rows)
        return result

    return {
        "reference_condition": reference_condition,
        "primary_gate_metric": "scenario_success_rate",
        "supports_primary_evidence": bool(primary_supported),
        "has_learning_evidence": bool(has_learning_evidence),
        "primary_condition": reference,
        "trained_final": trained_final,
        "random_init": random_init,
        "reflex_only": reflex_only,
        "trained_without_reflex_support": trained_without_reflex_support,
        "trained_vs_random_init": trained_vs_random,
        "trained_vs_reflex_only": trained_vs_reflex,
        "seed_level": condition_seed_level,
        "uncertainty": {
            "conditions": condition_uncertainty,
            "trained_vs_random_init": _delta_uncertainty(
                "random_init",
                random_init,
                trained_vs_random,
            ),
            "trained_vs_reflex_only": _delta_uncertainty(
                "reflex_only",
                reflex_only,
                trained_vs_reflex,
            ),
        },
        "notes": notes,
    }


def build_ablation_deltas(
    variants: Dict[str, Dict[str, object]],
    *,
    reference_variant: str,
    scenario_names: Sequence[str],
) -> Dict[str, object]:
    """
    Compute per-variant deltas of success rates against a reference ablation variant.

    Parameters:
        variants (Dict[str, Dict[str, object]]): Mapping from variant name to its behavior payload containing at least `"summary"` and `"suite"` entries.
        reference_variant (str): Name of the variant to use as the reference baseline.
        scenario_names (Sequence[str]): Sequence of scenario names for which per-scenario deltas will be computed.

    Returns:
        Dict[str, object]: Mapping from variant name to a dict with two keys:
            - "summary": contains
                - "scenario_success_rate_delta": difference between the variant's overall scenario success rate and the reference's, rounded to 6 decimals.
                - "episode_success_rate_delta": difference between the variant's episode success rate and the reference's, rounded to 6 decimals.
            - "scenarios": mapping of each scenario name to
                - "success_rate_delta": difference between the variant's per-scenario success rate and the reference's, rounded to 6 decimals.
    """
    reference = variants[reference_variant]
    reference_summary = reference["summary"]
    reference_suite = reference["suite"]
    deltas: Dict[str, object] = {}
    for variant_name, payload in variants.items():
        summary = payload["summary"]
        suite = payload["suite"]
        summary_deltas = {
            "scenario_success_rate_delta": round(
                float(summary["scenario_success_rate"] - reference_summary["scenario_success_rate"]),
                6,
            ),
            "episode_success_rate_delta": round(
                float(summary["episode_success_rate"] - reference_summary["episode_success_rate"]),
                6,
            ),
        }
        summary_uncertainty: Dict[str, object] = {}
        summary_seed_level: list[Dict[str, object]] = []
        for metric_name, delta_name in (
            ("scenario_success_rate", "scenario_success_rate_delta"),
            ("episode_success_rate", "episode_success_rate_delta"),
        ):
            seed_rows = paired_seed_delta_rows(
                metric_seed_values_from_payload(
                    reference,
                    metric_name=metric_name,
                    fallback_value=reference_summary[metric_name],
                ),
                metric_seed_values_from_payload(
                    payload,
                    metric_name=metric_name,
                    fallback_value=summary[metric_name],
                ),
                metric_name=delta_name,
                condition=variant_name,
                fallback_delta=summary_deltas[delta_name],
            )
            summary_seed_level.extend(seed_level_dicts(seed_rows))
            summary_uncertainty[delta_name] = (
                aggregate_with_uncertainty(seed_rows)
            )
        scenario_deltas: Dict[str, object] = {}
        for scenario_name in scenario_names:
            delta_value = round(
                float(suite[scenario_name]["success_rate"] - reference_suite[scenario_name]["success_rate"]),
                6,
            )
            seed_rows = paired_seed_delta_rows(
                metric_seed_values_from_payload(
                    reference,
                    metric_name="scenario_success_rate",
                    scenario=scenario_name,
                    fallback_value=reference_suite[scenario_name]["success_rate"],
                ),
                metric_seed_values_from_payload(
                    payload,
                    metric_name="scenario_success_rate",
                    scenario=scenario_name,
                    fallback_value=suite[scenario_name]["success_rate"],
                ),
                metric_name="success_rate_delta",
                condition=variant_name,
                fallback_delta=delta_value,
                scenario=scenario_name,
            )
            scenario_deltas[scenario_name] = {
                "success_rate_delta": delta_value,
                "seed_level": seed_level_dicts(seed_rows),
                "uncertainty": aggregate_with_uncertainty(
                    seed_rows
                ),
            }
        deltas[variant_name] = {
            "summary": summary_deltas,
            "scenarios": scenario_deltas,
            "seed_level": summary_seed_level,
            "uncertainty": summary_uncertainty,
        }
    return deltas


def compare_configurations(
    *,
    width: int = 12,
    height: int = 12,
    food_count: int = 4,
    day_length: int = 18,
    night_length: int = 12,
    max_steps: int | None = None,
    episodes: int | None = None,
    evaluation_episodes: int | None = None,
    gamma: float = 0.96,
    module_lr: float = 0.010,
    motor_lr: float = 0.012,
    module_dropout: float = 0.05,
    operational_profile: str | OperationalProfile | None = None,
    noise_profile: str | NoiseConfig | None = None,
    budget_profile: str | BudgetProfile | None = None,
    reward_profiles: Sequence[str] | None = None,
    map_templates: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
) -> Dict[str, object]:
    """
    Compare agent performance across reward profiles and map templates by training and evaluating simulations and aggregating episode statistics.

    Runs independent simulation instances for each (reward_profile, map_template) pair across the provided seeds using a resolved budget (episodes/evaluation_episodes/max_steps). Each instance is trained and then evaluated; per-seed results are aggregated into compact episode-stat summaries.

    Parameters:
        episodes (int | None): Training episodes override used to resolve the runtime budget; if None the budget profile or defaults are used.
        evaluation_episodes (int | None): Evaluation episodes override used to resolve the runtime budget.
        operational_profile (str | OperationalProfile | None): Operational profile or its name applied to each simulation.
        noise_profile (str | NoiseConfig | None): Noise profile or its name applied to each simulation.
        reward_profiles (Sequence[str] | None): Iterable of reward profile names to compare; defaults to ("classic",) when None.
        map_templates (Sequence[str] | None): Iterable of map template names to compare; defaults to ("central_burrow",) when None.
        seeds (Sequence[int] | None): Random seeds used to instantiate independent simulation runs; if None the budget's comparison seeds are used.
        budget_profile (str | BudgetProfile | None): Budget profile or its name used to resolve episodes/eval/max_steps when overrides are not provided.
        width, height, food_count, day_length, night_length, max_steps, gamma, module_lr, motor_lr, module_dropout: World and agent hyperparameters used to construct each simulation instance.

    Returns:
        Dict[str, object]: A mapping containing:
          - "budget_profile": resolved budget profile name.
          - "benchmark_strength": resolved benchmark strength from the budget.
          - "seeds": list of seeds actually used.
          - "reward_profiles": mapping from reward profile name to compact aggregated episode statistics across all maps and seeds.
          - "map_templates": mapping from map template name to compact aggregated episode statistics across all profiles and seeds.
          - "matrix": nested mapping matrix[profile][map_name] containing compact aggregated episode statistics for each (profile, map) pair.
    """
    resolved_noise_profile = resolve_noise_profile(noise_profile)
    budget = resolve_budget(
        profile=budget_profile,
        episodes=episodes,
        eval_episodes=evaluation_episodes,
        max_steps=max_steps,
        scenario_episodes=None,
        checkpoint_interval=None,
        behavior_seeds=seeds,
        ablation_seeds=seeds,
    )
    profile_names = list(reward_profiles or ("classic",))
    map_names = list(map_templates or ("central_burrow",))
    seed_values = tuple(seeds) if seeds is not None else budget.comparison_seeds
    eval_runs = max(1, int(budget.eval_episodes))
    matrix: Dict[str, Dict[str, object]] = {}
    profile_histories: Dict[str, List[EpisodeStats]] = {
        profile: [] for profile in profile_names
    }
    map_histories: Dict[str, List[EpisodeStats]] = {
        map_name: [] for map_name in map_names
    }

    for profile in profile_names:
        matrix[profile] = {}
        for map_name in map_names:
            combined_history: List[EpisodeStats] = []
            for seed in seed_values:
                sim = SpiderSimulation(
                    width=width,
                    height=height,
                    food_count=food_count,
                    day_length=day_length,
                    night_length=night_length,
                    max_steps=budget.max_steps,
                    seed=seed,
                    gamma=gamma,
                    module_lr=module_lr,
                    motor_lr=motor_lr,
                    module_dropout=module_dropout,
                    operational_profile=operational_profile,
                    noise_profile=resolved_noise_profile,
                    reward_profile=profile,
                    map_template=map_name,
                )
                _, evaluation_history, _ = sim._train_histories(
                    episodes=budget.episodes,
                    evaluation_episodes=eval_runs,
                    render_last_evaluation=False,
                    capture_evaluation_trace=False,
                    debug_trace=False,
                )
                combined_history.extend(evaluation_history)
            matrix[profile][map_name] = with_noise_profile_metadata(
                compact_aggregate(aggregate_episode_stats(combined_history)),
                resolved_noise_profile,
            )
            profile_histories[profile].extend(combined_history)
            map_histories[map_name].extend(combined_history)

    reward_profile_payloads = {
        profile: with_noise_profile_metadata(
            episode_history_reward_payload(history),
            resolved_noise_profile,
        )
        for profile, history in profile_histories.items()
    }
    reward_audit = build_reward_audit(
        current_profile=profile_names[0] if profile_names else None,
        comparison_payload={"reward_profiles": reward_profile_payloads},
    )
    result: Dict[str, object] = {
        "budget_profile": budget.profile,
        "benchmark_strength": budget.benchmark_strength,
        "seeds": list(seed_values),
        "reward_profiles": reward_profile_payloads,
        "map_templates": {
            map_name: with_noise_profile_metadata(
                episode_history_reward_payload(history),
                resolved_noise_profile,
            )
            for map_name, history in map_histories.items()
        },
        "matrix": matrix,
        "reward_audit": reward_audit,
        "shaping_reduction_status": shaping_reduction_status(reward_audit),
        **noise_profile_metadata(resolved_noise_profile),
    }
    comparison = reward_audit.get("comparison")
    if isinstance(comparison, dict):
        result["behavior_survival"] = comparison.get("behavior_survival")
        result["austere_survival_summary"] = comparison.get(
            "austere_survival_summary"
        )
        result["gap_policy_check"] = comparison.get("gap_policy_check")
        result["shaping_dependent_behaviors"] = comparison.get(
            "shaping_dependent_behaviors",
            [],
        )
        gate_passed = austere_survival_gate_passed(comparison)
        if gate_passed is not None:
            result["austere_survival_gate_passed"] = gate_passed
    return result


def compare_reward_profiles(**kwargs: object) -> Dict[str, object]:
    """Compatibility wrapper for reward-profile comparison reports."""
    return compare_configurations(**kwargs)


def compare_behavior_suite(
    *,
    width: int = 12,
    height: int = 12,
    food_count: int = 4,
    day_length: int = 18,
    night_length: int = 12,
    max_steps: int | None = None,
    episodes: int | None = None,
    evaluation_episodes: int | None = None,
    gamma: float = 0.96,
    module_lr: float = 0.010,
    motor_lr: float = 0.012,
    module_dropout: float = 0.05,
    operational_profile: str | OperationalProfile | None = None,
    noise_profile: str | NoiseConfig | None = None,
    budget_profile: str | BudgetProfile | None = None,
    reward_profiles: Sequence[str] | None = None,
    map_templates: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
    names: Sequence[str] | None = None,
    episodes_per_scenario: int | None = None,
    checkpoint_selection: str = "none",
    checkpoint_metric: str = "scenario_success_rate",
    checkpoint_override_penalty: float = 0.0,
    checkpoint_dominance_penalty: float = 0.0,
    checkpoint_penalty_mode: CheckpointPenaltyMode | str = (
        CheckpointPenaltyMode.TIEBREAKER
    ),
    checkpoint_interval: int | None = None,
    checkpoint_dir: str | Path | None = None,
) -> tuple[Dict[str, object], List[Dict[str, object]]]:
    """
    Compare behavior-suite performance across reward profiles, map templates, and random seeds.

    Runs (optionally trains) simulations for each combination of reward profile, map template, and seed, executes the behavior suite per scenario, aggregates behavioral scores and legacy episode statistics into a compact payload matrix, and collects flattened per-episode behavior rows suitable for CSV export.

    Parameters:
        width (int): World width in grid cells.
        height (int): World height in grid cells.
        food_count (int): Number of food items placed in the world.
        day_length (int): Duration of daytime ticks.
        night_length (int): Duration of nighttime ticks.
        max_steps (int): Maximum steps per episode.
        episodes (int): Number of training episodes to run before behavior evaluation for each simulation.
        evaluation_episodes (int): Number of evaluation episodes (performed during training phase) to run.
        gamma (float): Discount factor for the agent's learning configuration.
        module_lr (float): Module learning rate passed to the agent.
        motor_lr (float): Motor learning rate passed to the agent.
        module_dropout (float): Module dropout probability used when constructing simulations.
        operational_profile (str | OperationalProfile | None): Registered operational profile name, explicit
            `OperationalProfile` instance, or `None` to use the default profile for each simulation run.
            Invalid profile names raise `ValueError` during resolution.
        reward_profiles (Sequence[str] | None): Iterable of reward profile names to evaluate; defaults to ("classic",) when None.
        map_templates (Sequence[str] | None): Iterable of map template names to evaluate; defaults to ("central_burrow",) when None.
        seeds (Sequence[int]): Sequence of RNG seeds to instantiate independent simulations.
        names (Sequence[str] | None): Sequence of scenario names to run; defaults to the global SCENARIO_NAMES when None.
        episodes_per_scenario (int): Number of runs to execute per scenario (controls how many episodes are aggregated per scenario).

    Returns:
        tuple:
            - payload (Dict[str, object]): A dictionary containing:
                - "seeds": list of seeds used,
                - "scenario_names": list of scenario names evaluated,
                - "episodes_per_scenario": number of runs per scenario,
                - "reward_profiles": per-profile compact behavior payloads,
                - "map_templates": per-map compact behavior payloads,
                - "matrix": nested mapping reward_profile -> map_template -> compact behavior payload.
            - rows (List[Dict[str, object]]): Flattened per-episode behavior rows collected across all runs, suitable for CSV export.
    """
    resolved_noise_profile = resolve_noise_profile(noise_profile)
    budget = resolve_budget(
        profile=budget_profile,
        episodes=episodes,
        eval_episodes=evaluation_episodes,
        max_steps=max_steps,
        scenario_episodes=episodes_per_scenario,
        checkpoint_interval=checkpoint_interval,
        behavior_seeds=seeds,
        ablation_seeds=seeds,
    )
    scenario_names = list(names or SCENARIO_NAMES)
    profile_names = list(reward_profiles or ("classic",))
    map_names = list(map_templates or ("central_burrow",))
    run_count = max(1, int(budget.scenario_episodes))
    seed_values = tuple(seeds) if seeds is not None else budget.behavior_seeds
    matrix: Dict[str, Dict[str, object]] = {}
    rows: List[Dict[str, object]] = []
    profile_stats = {
        profile: {name: [] for name in scenario_names}
        for profile in profile_names
    }
    profile_scores = {
        profile: {name: [] for name in scenario_names}
        for profile in profile_names
    }
    map_stats = {
        map_name: {name: [] for name in scenario_names}
        for map_name in map_names
    }
    map_scores = {
        map_name: {name: [] for name in scenario_names}
        for map_name in map_names
    }

    for profile in profile_names:
        matrix[profile] = {}
        for map_name in map_names:
            combined_stats = {
                name: [] for name in scenario_names
            }
            combined_scores = {
                name: [] for name in scenario_names
            }
            sim: SpiderSimulation | None = None
            for seed in seed_values:
                sim_budget = budget.to_summary()
                sim_budget["resolved"]["scenario_episodes"] = run_count
                sim_budget["resolved"]["behavior_seeds"] = list(seed_values)
                sim_budget["resolved"]["ablation_seeds"] = list(budget.ablation_seeds)
                sim = SpiderSimulation(
                    width=width,
                    height=height,
                    food_count=food_count,
                    day_length=day_length,
                    night_length=night_length,
                    max_steps=budget.max_steps,
                    seed=seed,
                    gamma=gamma,
                    module_lr=module_lr,
                    motor_lr=motor_lr,
                    module_dropout=module_dropout,
                    operational_profile=operational_profile,
                    noise_profile=resolved_noise_profile,
                    reward_profile=profile,
                    map_template=map_name,
                    budget_profile_name=budget.profile,
                    benchmark_strength=budget.benchmark_strength,
                    budget_summary=sim_budget,
                )
                if (
                    checkpoint_selection == "best"
                    or budget.episodes > 0
                    or budget.eval_episodes > 0
                ):
                    run_checkpoint_dir = None
                    if checkpoint_dir is not None:
                        run_checkpoint_dir = (
                            Path(checkpoint_dir)
                            / "behavior_compare"
                            / f"{profile}__{map_name}__seed_{seed}"
                        )
                    sim.train(
                        budget.episodes,
                        evaluation_episodes=0,
                        render_last_evaluation=False,
                        capture_evaluation_trace=False,
                        debug_trace=False,
                        checkpoint_selection=checkpoint_selection,
                        checkpoint_metric=checkpoint_metric,
                        checkpoint_override_penalty=checkpoint_override_penalty,
                        checkpoint_dominance_penalty=checkpoint_dominance_penalty,
                        checkpoint_penalty_mode=checkpoint_penalty_mode,
                        checkpoint_interval=budget.checkpoint_interval,
                        checkpoint_dir=run_checkpoint_dir,
                        checkpoint_scenario_names=scenario_names,
                        selection_scenario_episodes=budget.selection_scenario_episodes,
                    )
                stats_histories, behavior_histories, _ = sim._execute_behavior_suite(
                    names=scenario_names,
                    episodes_per_scenario=run_count,
                    capture_trace=False,
                    debug_trace=False,
                    base_index=200_000,
                )
                for name in scenario_names:
                    combined_stats[name].extend(stats_histories[name])
                    combined_scores[name].extend(behavior_histories[name])
                    profile_stats[profile][name].extend(stats_histories[name])
                    profile_scores[profile][name].extend(behavior_histories[name])
                    map_stats[map_name][name].extend(stats_histories[name])
                    map_scores[map_name][name].extend(behavior_histories[name])
                    rows.extend(
                        sim._annotate_behavior_rows(
                            flatten_behavior_rows(
                                behavior_histories[name],
                                reward_profile=profile,
                                scenario_map=get_scenario(name).map_template,
                                simulation_seed=seed,
                                scenario_description=get_scenario(name).description,
                                scenario_objective=get_scenario(name).objective,
                                scenario_focus=get_scenario(name).diagnostic_focus,
                                evaluation_map=map_name,
                            )
                        )
                    )
            if sim is None:
                continue
            matrix[profile][map_name] = with_noise_profile_metadata(
                compact_behavior_payload(
                    sim._build_behavior_payload(
                        stats_histories=combined_stats,
                        behavior_histories=combined_scores,
                    )
                ),
                resolved_noise_profile,
            )

    reward_profile_payloads: Dict[str, object] = {}
    for profile in profile_names:
        sim = SpiderSimulation(
            width=width,
            height=height,
            food_count=food_count,
            day_length=day_length,
            night_length=night_length,
            max_steps=budget.max_steps,
            seed=seed_values[0] if seed_values else 0,
            gamma=gamma,
            module_lr=module_lr,
            motor_lr=motor_lr,
            module_dropout=module_dropout,
            operational_profile=operational_profile,
            noise_profile=resolved_noise_profile,
            reward_profile=profile,
            map_template=map_names[0] if map_names else "central_burrow",
        )
        reward_profile_payloads[profile] = with_noise_profile_metadata(
            compact_behavior_payload(
                sim._build_behavior_payload(
                    stats_histories=profile_stats[profile],
                    behavior_histories=profile_scores[profile],
                )
            ),
            resolved_noise_profile,
        )

    map_payloads: Dict[str, object] = {}
    for map_name in map_names:
        sim = SpiderSimulation(
            width=width,
            height=height,
            food_count=food_count,
            day_length=day_length,
            night_length=night_length,
            max_steps=budget.max_steps,
            seed=seed_values[0] if seed_values else 0,
            gamma=gamma,
            module_lr=module_lr,
            motor_lr=motor_lr,
            module_dropout=module_dropout,
            operational_profile=operational_profile,
            noise_profile=resolved_noise_profile,
            reward_profile=profile_names[0] if profile_names else "classic",
            map_template=map_name,
        )
        map_payloads[map_name] = with_noise_profile_metadata(
            compact_behavior_payload(
                sim._build_behavior_payload(
                    stats_histories=map_stats[map_name],
                    behavior_histories=map_scores[map_name],
                )
            ),
            resolved_noise_profile,
        )

    reward_audit = build_reward_audit(
        current_profile=profile_names[0] if profile_names else None,
        comparison_payload={"reward_profiles": reward_profile_payloads},
    )
    result: Dict[str, object] = {
        "budget_profile": budget.profile,
        "benchmark_strength": budget.benchmark_strength,
        "checkpoint_selection": checkpoint_selection,
        "checkpoint_metric": checkpoint_metric,
        "checkpoint_penalty_config": CheckpointSelectionConfig(
            metric=checkpoint_metric,
            override_penalty_weight=checkpoint_override_penalty,
            dominance_penalty_weight=checkpoint_dominance_penalty,
            penalty_mode=checkpoint_penalty_mode,
        ).to_summary(),
        "seeds": list(seed_values),
        "scenario_names": scenario_names,
        "episodes_per_scenario": run_count,
        "reward_profiles": reward_profile_payloads,
        "map_templates": map_payloads,
        "matrix": matrix,
        "reward_audit": reward_audit,
        "shaping_reduction_status": shaping_reduction_status(reward_audit),
        **noise_profile_metadata(resolved_noise_profile),
    }
    comparison = reward_audit.get("comparison")
    if isinstance(comparison, dict):
        result["behavior_survival"] = comparison.get("behavior_survival")
        result["austere_survival_summary"] = comparison.get(
            "austere_survival_summary"
        )
        result["gap_policy_check"] = comparison.get("gap_policy_check")
        result["shaping_dependent_behaviors"] = comparison.get(
            "shaping_dependent_behaviors",
            [],
        )
        gate_passed = austere_survival_gate_passed(comparison)
        if gate_passed is not None:
            result["austere_survival_gate_passed"] = gate_passed
    return result, rows


def _compare_named_training_regimes(
    *,
    regime_names: Sequence[str],
    width: int = 12,
    height: int = 12,
    food_count: int = 4,
    day_length: int = 18,
    night_length: int = 12,
    max_steps: int | None = None,
    episodes: int | None = None,
    evaluation_episodes: int | None = None,
    gamma: float = 0.96,
    module_lr: float = 0.010,
    motor_lr: float = 0.012,
    module_dropout: float = 0.05,
    reward_profile: str = "classic",
    map_template: str = "central_burrow",
    operational_profile: str | OperationalProfile | None = None,
    noise_profile: str | NoiseConfig | None = None,
    budget_profile: str | BudgetProfile | None = None,
    seeds: Sequence[int] | None = None,
    names: Sequence[str] | None = None,
    episodes_per_scenario: int | None = None,
    checkpoint_selection: str = "none",
    checkpoint_metric: str = "scenario_success_rate",
    checkpoint_override_penalty: float = 0.0,
    checkpoint_dominance_penalty: float = 0.0,
    checkpoint_penalty_mode: CheckpointPenaltyMode | str = (
        CheckpointPenaltyMode.TIEBREAKER
    ),
    checkpoint_interval: int | None = None,
    checkpoint_dir: str | Path | None = None,
) -> tuple[Dict[str, object], List[Dict[str, object]]]:
    """
    Compare specified training regimes by training and evaluating each under both self-sufficient (no reflex support) and scaffolded (reflex-supported) conditions.

    For each regime and seed this runs training according to the resolved budget and regime spec, collects per-scenario episode statistics and behavior scores for scaffolded and self-sufficient evaluations, aggregates results across seeds, computes competence gaps (scaffolded vs self-sufficient), and flattens per-episode/seed rows suitable for CSV export.

    Parameters:
        regime_names (Sequence[str]): Names of training regimes to compare. `"baseline"` will be ensured as the first regime if not present.
        width, height, food_count, day_length, night_length, max_steps: Environment layout and episode length overrides used to construct each simulation.
        episodes (int | None): Total training episodes (resolved via the budget profile if None).
        evaluation_episodes (int | None): Evaluation episodes per evaluation pass (resolved via the budget profile if None).
        seeds (Sequence[int] | None): Random seeds to run; at least one seed is required.
        names (Sequence[str] | None): Scenario names to evaluate; defaults to all known scenarios.
        episodes_per_scenario (int | None): Number of evaluation episodes per scenario (overrides budget per-scenario setting).
        checkpoint_selection (str): One of `"none"` or `"best"`, controls whether checkpoints are captured/selected during training.
        checkpoint_metric (str): Primary metric label used when ranking checkpoint candidates.
        checkpoint_override_penalty (float): Weight applied to override-rate penalty when using direct composite scoring.
        checkpoint_dominance_penalty (float): Weight applied to reflex dominance penalty when using direct composite scoring.
        checkpoint_penalty_mode (CheckpointPenaltyMode | str): Penalty interpretation mode; `TIEBREAKER` preserves legacy tuple ordering, `DIRECT` uses a penalized composite score.
        checkpoint_interval, checkpoint_dir: Checkpoint capture cadence and optional persistence directory.

    Returns:
        tuple[Dict[str, object], List[Dict[str, object]]]: A pair (payload, rows) where:
          - payload: a comparison summary containing per-regime payloads (`regimes`), competence gaps, deltas vs baseline, checkpoint penalty configuration, and noise/budget metadata.
          - rows: flattened annotated behavior rows (one row per evaluated episode/scenario/seed) suitable for CSV export.
    """
    requested_regime_names = [str(name) for name in regime_names]
    if not requested_regime_names:
        requested_regime_names = [
            "baseline",
            "reflex_annealed",
            EXPERIMENT_OF_RECORD_REGIME,
        ]
    if "baseline" not in requested_regime_names:
        requested_regime_names.insert(0, "baseline")
    deduped_regime_names = list(dict.fromkeys(requested_regime_names))
    regime_specs = {
        name: resolve_training_regime(name)
        for name in deduped_regime_names
    }
    resolved_noise_profile = resolve_noise_profile(noise_profile)
    budget = resolve_budget(
        profile=budget_profile,
        episodes=episodes,
        eval_episodes=evaluation_episodes,
        max_steps=max_steps,
        scenario_episodes=episodes_per_scenario,
        checkpoint_interval=checkpoint_interval,
        behavior_seeds=seeds,
        ablation_seeds=seeds,
    )
    scenario_names = list(names or SCENARIO_NAMES)
    run_count = max(1, int(budget.scenario_episodes))
    seed_values = tuple(seeds) if seeds is not None else budget.behavior_seeds
    if not seed_values:
        raise ValueError(
            "compare_training_regimes() requires at least one seed."
        )

    rows: List[Dict[str, object]] = []
    regime_payloads: Dict[str, Dict[str, object]] = {}

    for regime_index, (regime_name, regime_spec) in enumerate(regime_specs.items()):
        self_sufficient_stats = {name: [] for name in scenario_names}
        self_sufficient_scores = {name: [] for name in scenario_names}
        scaffolded_stats = {name: [] for name in scenario_names}
        scaffolded_scores = {name: [] for name in scenario_names}
        training_summaries: list[Dict[str, object]] = []
        training_metadata: list[Dict[str, object]] = []
        exemplar_sim: SpiderSimulation | None = None

        for seed in seed_values:
            sim_budget = budget.to_summary()
            sim_budget["resolved"]["scenario_episodes"] = run_count
            sim_budget["resolved"]["behavior_seeds"] = list(seed_values)
            sim_budget["resolved"]["ablation_seeds"] = list(seed_values)
            sim = SpiderSimulation(
                width=width,
                height=height,
                food_count=food_count,
                day_length=day_length,
                night_length=night_length,
                max_steps=budget.max_steps,
                seed=seed,
                gamma=gamma,
                module_lr=module_lr,
                motor_lr=motor_lr,
                module_dropout=module_dropout,
                operational_profile=operational_profile,
                noise_profile=resolved_noise_profile,
                reward_profile=reward_profile,
                map_template=map_template,
                budget_profile_name=budget.profile,
                benchmark_strength=budget.benchmark_strength,
                budget_summary=sim_budget,
            )
            exemplar_sim = sim
            run_checkpoint_dir = None
            if checkpoint_dir is not None:
                run_checkpoint_dir = (
                    Path(checkpoint_dir)
                    / "training_regime_compare"
                    / f"{regime_name}__seed_{seed}"
                )
            training_summary, _ = sim.train(
                budget.episodes,
                evaluation_episodes=budget.eval_episodes,
                render_last_evaluation=False,
                capture_evaluation_trace=False,
                debug_trace=False,
                checkpoint_selection=checkpoint_selection,
                checkpoint_metric=checkpoint_metric,
                checkpoint_override_penalty=checkpoint_override_penalty,
                checkpoint_dominance_penalty=checkpoint_dominance_penalty,
                checkpoint_penalty_mode=checkpoint_penalty_mode,
                checkpoint_interval=budget.checkpoint_interval,
                checkpoint_dir=run_checkpoint_dir,
                checkpoint_scenario_names=scenario_names,
                selection_scenario_episodes=budget.selection_scenario_episodes,
                training_regime=regime_spec,
            )
            training_summaries.append(deepcopy(training_summary))
            seed_training_metadata = deepcopy(sim._latest_training_regime_summary)
            seed_training_metadata["seed"] = int(seed)
            training_metadata.append(seed_training_metadata)

            scaffolded_eval_scale = sim._effective_reflex_scale(
                sim.brain.current_reflex_scale
            )
            previous_reflex_scale = float(sim.brain.current_reflex_scale)
            sim.brain.set_runtime_reflex_scale(scaffolded_eval_scale)
            try:
                scaffolded_stats_histories, scaffolded_behavior_histories, _ = (
                    sim._execute_behavior_suite(
                        names=scenario_names,
                        episodes_per_scenario=run_count,
                        capture_trace=False,
                        debug_trace=False,
                        base_index=600_000 + regime_index * 20_000,
                    )
                )
            finally:
                sim.brain.set_runtime_reflex_scale(previous_reflex_scale)

            previous_reflex_scale = float(sim.brain.current_reflex_scale)
            sim.brain.set_runtime_reflex_scale(0.0)
            try:
                self_stats_histories, self_behavior_histories, _ = (
                    sim._execute_behavior_suite(
                        names=scenario_names,
                        episodes_per_scenario=run_count,
                        capture_trace=False,
                        debug_trace=False,
                        base_index=600_000 + regime_index * 20_000,
                    )
                )
            finally:
                sim.brain.set_runtime_reflex_scale(previous_reflex_scale)

            for scenario_name in scenario_names:
                scenario = get_scenario(scenario_name)
                scaffolded_stats[scenario_name].extend(
                    scaffolded_stats_histories[scenario_name]
                )
                scaffolded_scores[scenario_name].extend(
                    scaffolded_behavior_histories[scenario_name]
                )
                self_sufficient_stats[scenario_name].extend(
                    self_stats_histories[scenario_name]
                )
                self_sufficient_scores[scenario_name].extend(
                    self_behavior_histories[scenario_name]
                )
                rows.extend(
                    sim._annotate_behavior_rows(
                        flatten_behavior_rows(
                            scaffolded_behavior_histories[scenario_name],
                            reward_profile=reward_profile,
                            scenario_map=scenario.map_template,
                            simulation_seed=seed,
                            scenario_description=scenario.description,
                            scenario_objective=scenario.objective,
                            scenario_focus=scenario.diagnostic_focus,
                            evaluation_map=map_template,
                            eval_reflex_scale=scaffolded_eval_scale,
                            competence_label="scaffolded",
                        ),
                        eval_reflex_scale=scaffolded_eval_scale,
                    )
                )
                rows.extend(
                    sim._annotate_behavior_rows(
                        flatten_behavior_rows(
                            self_behavior_histories[scenario_name],
                            reward_profile=reward_profile,
                            scenario_map=scenario.map_template,
                            simulation_seed=seed,
                            scenario_description=scenario.description,
                            scenario_objective=scenario.objective,
                            scenario_focus=scenario.diagnostic_focus,
                            evaluation_map=map_template,
                            eval_reflex_scale=0.0,
                            competence_label="self_sufficient",
                        ),
                        eval_reflex_scale=0.0,
                    )
                )

        if exemplar_sim is None:
            continue

        self_payload = with_noise_profile_metadata(
            compact_behavior_payload(
                exemplar_sim._build_behavior_payload(
                    stats_histories=self_sufficient_stats,
                    behavior_histories=self_sufficient_scores,
                    competence_label="self_sufficient",
                )
            ),
            resolved_noise_profile,
        )
        self_payload["summary"]["eval_reflex_scale"] = 0.0
        self_payload["eval_reflex_scale"] = 0.0
        self_payload["competence_type"] = "self_sufficient"

        scaffolded_payload = with_noise_profile_metadata(
            compact_behavior_payload(
                exemplar_sim._build_behavior_payload(
                    stats_histories=scaffolded_stats,
                    behavior_histories=scaffolded_scores,
                    competence_label="scaffolded",
                )
            ),
            resolved_noise_profile,
        )
        scaffolded_eval_scale = exemplar_sim._effective_reflex_scale(
            exemplar_sim.brain.current_reflex_scale
        )
        scaffolded_payload["summary"]["eval_reflex_scale"] = scaffolded_eval_scale
        scaffolded_payload["eval_reflex_scale"] = scaffolded_eval_scale
        scaffolded_payload["competence_type"] = "scaffolded"

        competence_gap = SpiderSimulation._evaluation_competence_gap(
            self_sufficient=self_payload["summary"],
            scaffolded=scaffolded_payload["summary"],
        )
        regime_summary = regime_spec.to_summary()
        regime_summary["is_experiment_of_record"] = (
            regime_name == EXPERIMENT_OF_RECORD_REGIME
        )
        regime_payloads[regime_name] = {
            "regime": regime_name,
            "training_regime": regime_summary,
            "is_experiment_of_record": regime_name == EXPERIMENT_OF_RECORD_REGIME,
            "training_regimes": training_metadata,
            "training_summaries": training_summaries,
            "primary_evaluation": "self_sufficient",
            "summary": deepcopy(self_payload["summary"]),
            "success_rates": {
                "self_sufficient": float(
                    self_payload["summary"].get("scenario_success_rate", 0.0)
                ),
                "scaffolded": float(
                    scaffolded_payload["summary"].get(
                        "scenario_success_rate",
                        0.0,
                    )
                ),
            },
            "episode_success_rates": {
                "self_sufficient": float(
                    self_payload["summary"].get("episode_success_rate", 0.0)
                ),
                "scaffolded": float(
                    scaffolded_payload["summary"].get(
                        "episode_success_rate",
                        0.0,
                    )
                ),
            },
            "competence_gap": competence_gap,
            "self_sufficient": self_payload,
            "scaffolded": scaffolded_payload,
            "primary_benchmark": self_payload,
            "episode_allocation": {
                "main_training_episodes": int(budget.episodes),
                "evaluation_episodes": int(budget.eval_episodes),
                "episodes_per_scenario": int(run_count),
            },
            **noise_profile_metadata(resolved_noise_profile),
        }

    baseline_payload = regime_payloads.get("baseline", {})
    baseline_self = baseline_payload.get("self_sufficient", {})
    baseline_scaffolded = baseline_payload.get("scaffolded", {})
    baseline_self_summary = (
        baseline_self.get("summary", {})
        if isinstance(baseline_self, dict)
        else {}
    )
    baseline_scaffolded_summary = (
        baseline_scaffolded.get("summary", {})
        if isinstance(baseline_scaffolded, dict)
        else {}
    )
    baseline_gap = baseline_payload.get("competence_gap", {})
    deltas_vs_baseline: Dict[str, Dict[str, float]] = {}
    for regime_name, payload in regime_payloads.items():
        self_summary = payload["self_sufficient"]["summary"]
        scaffolded_summary = payload["scaffolded"]["summary"]
        competence_gap = payload["competence_gap"]
        deltas_vs_baseline[regime_name] = {
            "scenario_success_rate_delta": round(
                float(self_summary.get("scenario_success_rate", 0.0))
                - float(baseline_self_summary.get("scenario_success_rate", 0.0)),
                6,
            ),
            "episode_success_rate_delta": round(
                float(self_summary.get("episode_success_rate", 0.0))
                - float(baseline_self_summary.get("episode_success_rate", 0.0)),
                6,
            ),
            "scaffolded_scenario_success_rate_delta": round(
                float(scaffolded_summary.get("scenario_success_rate", 0.0))
                - float(
                    baseline_scaffolded_summary.get(
                        "scenario_success_rate",
                        0.0,
                    )
                ),
                6,
            ),
            "competence_gap_delta": round(
                float(competence_gap.get("scenario_success_rate_delta", 0.0))
                - float(baseline_gap.get("scenario_success_rate_delta", 0.0)),
                6,
            ),
        }

    return {
        "comparison_type": "training_regimes",
        "budget_profile": budget.profile,
        "benchmark_strength": budget.benchmark_strength,
        "checkpoint_selection": checkpoint_selection,
        "checkpoint_metric": checkpoint_metric,
        "checkpoint_penalty_config": CheckpointSelectionConfig(
            metric=checkpoint_metric,
            override_penalty_weight=checkpoint_override_penalty,
            dominance_penalty_weight=checkpoint_dominance_penalty,
            penalty_mode=checkpoint_penalty_mode,
        ).to_summary(),
        "reference_regime": "baseline",
        "experiment_of_record_regime": EXPERIMENT_OF_RECORD_REGIME,
        "regime_names": deduped_regime_names,
        "seeds": list(seed_values),
        "scenario_names": scenario_names,
        "episodes_per_scenario": run_count,
        "regimes": regime_payloads,
        "competence_gaps": {
            regime_name: payload["competence_gap"]
            for regime_name, payload in regime_payloads.items()
        },
        "deltas_vs_baseline": deltas_vs_baseline,
        **noise_profile_metadata(resolved_noise_profile),
    }, rows


def compare_training_regimes(
    regime_names: Sequence[str] | None = None,
    *,
    width: int = 12,
    height: int = 12,
    food_count: int = 4,
    day_length: int = 18,
    night_length: int = 12,
    max_steps: int | None = None,
    episodes: int | None = None,
    evaluation_episodes: int | None = None,
    gamma: float = 0.96,
    module_lr: float = 0.010,
    motor_lr: float = 0.012,
    module_dropout: float = 0.05,
    reward_profile: str = "classic",
    map_template: str = "central_burrow",
    operational_profile: str | OperationalProfile | None = None,
    noise_profile: str | NoiseConfig | None = None,
    budget_profile: str | BudgetProfile | None = None,
    seeds: Sequence[int] | None = None,
    names: Sequence[str] | None = None,
    episodes_per_scenario: int | None = None,
    checkpoint_selection: str = "none",
    checkpoint_metric: str = "scenario_success_rate",
    checkpoint_override_penalty: float = 0.0,
    checkpoint_dominance_penalty: float = 0.0,
    checkpoint_penalty_mode: CheckpointPenaltyMode | str = (
        CheckpointPenaltyMode.TIEBREAKER
    ),
    checkpoint_interval: int | None = None,
    checkpoint_dir: str | Path | None = None,
    curriculum_profile: str = "ecological_v1",
) -> tuple[Dict[str, object], List[Dict[str, object]]]:
    """
    Compare 'flat' and curriculum training regimes by training and evaluating both under a shared budget and seed set.

    Trains a flat regime and a curriculum regime (using the provided curriculum_profile) for each seed, optionally capturing checkpoints per the checkpoint parameters, evaluates both regimes across the same scenario suite, and aggregates per-scenario episode statistics, behavioral scores, training/curriculum metadata, and computed deltas comparing curriculum versus flat.

    Returns:
        result_payload (Dict[str, object]): Aggregated comparison payload containing budget and seed metadata, per-regime compact behavior payloads under "regimes", computed deltas versus the flat reference under "deltas_vs_flat", focus summaries, and noise profile metadata.
        rows (List[Dict[str, object]]): A flattened, annotated list of per-episode/behavior rows suitable for CSV export.
    """
    if regime_names is not None:
        return _compare_named_training_regimes(
            regime_names=regime_names,
            width=width,
            height=height,
            food_count=food_count,
            day_length=day_length,
            night_length=night_length,
            max_steps=max_steps,
            episodes=episodes,
            evaluation_episodes=evaluation_episodes,
            gamma=gamma,
            module_lr=module_lr,
            motor_lr=motor_lr,
            module_dropout=module_dropout,
            reward_profile=reward_profile,
            map_template=map_template,
            operational_profile=operational_profile,
            noise_profile=noise_profile,
            budget_profile=budget_profile,
            seeds=seeds,
            names=names,
            episodes_per_scenario=episodes_per_scenario,
            checkpoint_selection=checkpoint_selection,
            checkpoint_metric=checkpoint_metric,
            checkpoint_override_penalty=checkpoint_override_penalty,
            checkpoint_dominance_penalty=checkpoint_dominance_penalty,
            checkpoint_penalty_mode=checkpoint_penalty_mode,
            checkpoint_interval=checkpoint_interval,
            checkpoint_dir=checkpoint_dir,
        )
    profile_name = validate_curriculum_profile(curriculum_profile)
    if profile_name == "none":
        raise ValueError(
            "compare_training_regimes() requires an active curriculum."
        )
    resolved_noise_profile = resolve_noise_profile(noise_profile)
    budget = resolve_budget(
        profile=budget_profile,
        episodes=episodes,
        eval_episodes=evaluation_episodes,
        max_steps=max_steps,
        scenario_episodes=episodes_per_scenario,
        checkpoint_interval=checkpoint_interval,
        behavior_seeds=seeds,
        ablation_seeds=seeds,
    )
    scenario_names = list(names or SCENARIO_NAMES)
    focus_scenarios = [
        name for name in CURRICULUM_FOCUS_SCENARIOS if name in scenario_names
    ]
    run_count = max(1, int(budget.scenario_episodes))
    seed_values = tuple(seeds) if seeds is not None else budget.behavior_seeds
    if not seed_values:
        raise ValueError(
            "compare_training_regimes() requires at least one seed."
        )
    regimes = ("flat", "curriculum")
    regime_stats = {
        regime: {name: [] for name in scenario_names}
        for regime in regimes
    }
    regime_scores = {
        regime: {name: [] for name in scenario_names}
        for regime in regimes
    }
    regime_training_metadata: Dict[str, list[Dict[str, object]]] = {
        regime: [] for regime in regimes
    }
    regime_curriculum_metadata: Dict[str, list[Dict[str, object]]] = {
        regime: [] for regime in regimes
    }
    rows: List[Dict[str, object]] = []
    sim_budget = budget.to_summary()
    sim_budget["resolved"]["scenario_episodes"] = run_count
    sim_budget["resolved"]["behavior_seeds"] = list(seed_values)
    sim_budget["resolved"]["ablation_seeds"] = list(budget.ablation_seeds)

    for regime in regimes:
        for seed in seed_values:
            sim = SpiderSimulation(
                width=width,
                height=height,
                food_count=food_count,
                day_length=day_length,
                night_length=night_length,
                max_steps=budget.max_steps,
                seed=seed,
                gamma=gamma,
                module_lr=module_lr,
                motor_lr=motor_lr,
                module_dropout=module_dropout,
                operational_profile=operational_profile,
                noise_profile=resolved_noise_profile,
                reward_profile=reward_profile,
                map_template=map_template,
                budget_profile_name=budget.profile,
                benchmark_strength=budget.benchmark_strength,
                budget_summary=sim_budget,
            )
            if regime == "curriculum":
                sim._set_training_regime_metadata(
                    curriculum_profile=profile_name,
                    episodes=int(budget.episodes),
                    curriculum_summary=empty_curriculum_summary(
                        profile_name,
                        int(budget.episodes),
                    ) if int(budget.episodes) <= 0 else None,
                )
            else:
                sim._set_training_regime_metadata(
                    curriculum_profile="none",
                    episodes=int(budget.episodes),
                )
            if (
                checkpoint_selection == "best"
                or budget.episodes > 0
                or budget.eval_episodes > 0
            ):
                run_checkpoint_dir = None
                if checkpoint_dir is not None:
                    run_checkpoint_dir = (
                        Path(checkpoint_dir)
                        / "curriculum_compare"
                        / f"{regime}__seed_{seed}"
                    )
                sim.train(
                    budget.episodes,
                    evaluation_episodes=0,
                    render_last_evaluation=False,
                    capture_evaluation_trace=False,
                    debug_trace=False,
                    checkpoint_selection=checkpoint_selection,
                    checkpoint_metric=checkpoint_metric,
                    checkpoint_override_penalty=checkpoint_override_penalty,
                    checkpoint_dominance_penalty=checkpoint_dominance_penalty,
                    checkpoint_penalty_mode=checkpoint_penalty_mode,
                    checkpoint_interval=budget.checkpoint_interval,
                    checkpoint_dir=run_checkpoint_dir,
                    checkpoint_scenario_names=scenario_names,
                    selection_scenario_episodes=budget.selection_scenario_episodes,
                    curriculum_profile=(
                        profile_name if regime == "curriculum" else "none"
                    ),
                )
            seed_identifier = int(getattr(sim, "seed", seed))
            training_metadata = deepcopy(sim._latest_training_regime_summary)
            training_metadata["seed"] = seed_identifier
            regime_training_metadata[regime].append(training_metadata)
            if sim._latest_curriculum_summary is not None:
                curriculum_metadata = deepcopy(sim._latest_curriculum_summary)
                curriculum_metadata["seed"] = seed_identifier
                regime_curriculum_metadata[regime].append(
                    curriculum_metadata
                )
            stats_histories, behavior_histories, _ = sim._execute_behavior_suite(
                names=scenario_names,
                episodes_per_scenario=run_count,
                capture_trace=False,
                debug_trace=False,
                base_index=300_000,
            )
            for name in scenario_names:
                regime_stats[regime][name].extend(stats_histories[name])
                regime_scores[regime][name].extend(behavior_histories[name])
                rows.extend(
                    sim._annotate_behavior_rows(
                        flatten_behavior_rows(
                            behavior_histories[name],
                            reward_profile=reward_profile,
                            scenario_map=get_scenario(name).map_template,
                            simulation_seed=seed,
                            scenario_description=get_scenario(name).description,
                            scenario_objective=get_scenario(name).objective,
                            scenario_focus=get_scenario(name).diagnostic_focus,
                            evaluation_map=map_template,
                        )
                    )
                )

    regime_payloads = {
        regime: with_noise_profile_metadata(
            compact_behavior_payload(
                sim._build_behavior_payload(
                    stats_histories=regime_stats[regime],
                    behavior_histories=regime_scores[regime],
                )
            ),
            resolved_noise_profile,
        )
        for regime in regimes
    }
    for regime in regimes:
        regime_payloads[regime]["training_regimes"] = deepcopy(
            regime_training_metadata.get(regime, [])
        )
        latest_training_regime = (
            regime_training_metadata[regime][-1]
            if regime_training_metadata.get(regime)
            else {}
        )
        regime_payloads[regime]["training_regime"] = deepcopy(
            latest_training_regime
        )
        regime_payloads[regime]["episode_allocation"] = {
            "total_training_episodes": int(budget.episodes),
            "evaluation_episodes": int(budget.eval_episodes),
            "episodes_per_scenario": int(run_count),
        }
        curriculum_metadata = regime_curriculum_metadata.get(regime, [])
        if curriculum_metadata:
            regime_payloads[regime]["curriculum_runs"] = deepcopy(
                curriculum_metadata
            )
            regime_payloads[regime]["curriculum"] = deepcopy(
                curriculum_metadata[-1]
            )
    deltas = build_learning_evidence_deltas(
        regime_payloads,
        reference_condition="flat",
        scenario_names=scenario_names,
    )
    return {
        "budget_profile": budget.profile,
        "benchmark_strength": budget.benchmark_strength,
        "checkpoint_selection": checkpoint_selection,
        "checkpoint_metric": checkpoint_metric,
        "checkpoint_penalty_config": CheckpointSelectionConfig(
            metric=checkpoint_metric,
            override_penalty_weight=checkpoint_override_penalty,
            dominance_penalty_weight=checkpoint_dominance_penalty,
            penalty_mode=checkpoint_penalty_mode,
        ).to_summary(),
        "curriculum_profile": profile_name,
        "reference_regime": "flat",
        "seeds": list(seed_values),
        "scenario_names": scenario_names,
        "episodes_per_scenario": run_count,
        "focus_scenarios": focus_scenarios,
        "regimes": regime_payloads,
        "deltas_vs_flat": deltas,
        "focus_summary": {
            regime: {
                name: {
                    "success_rate": float(
                        regime_payloads[regime]["suite"]
                        .get(name, {})
                        .get("success_rate", 0.0)
                    )
                }
                for name in focus_scenarios
            }
            for regime in regimes
        },
        **noise_profile_metadata(resolved_noise_profile),
    }, rows


def compare_noise_robustness(
    *,
    width: int = 12,
    height: int = 12,
    food_count: int = 4,
    day_length: int = 18,
    night_length: int = 12,
    max_steps: int | None = None,
    episodes: int | None = None,
    evaluation_episodes: int | None = None,
    gamma: float = 0.96,
    module_lr: float = 0.010,
    motor_lr: float = 0.012,
    module_dropout: float = 0.05,
    reward_profile: str = "classic",
    map_template: str = "central_burrow",
    operational_profile: str | OperationalProfile | None = None,
    budget_profile: str | BudgetProfile | None = None,
    seeds: Sequence[int] | None = None,
    names: Sequence[str] | None = None,
    episodes_per_scenario: int | None = None,
    robustness_matrix: RobustnessMatrixSpec | None = None,
    checkpoint_selection: str = "none",
    checkpoint_metric: str = "scenario_success_rate",
    checkpoint_override_penalty: float = 0.0,
    checkpoint_dominance_penalty: float = 0.0,
    checkpoint_penalty_mode: CheckpointPenaltyMode | str = (
        CheckpointPenaltyMode.TIEBREAKER
    ),
    checkpoint_interval: int | None = None,
    checkpoint_dir: str | Path | None = None,
    load_brain: str | Path | None = None,
    load_modules: Sequence[str] | None = None,
    save_brain: str | Path | None = None,
) -> tuple[Dict[str, object], List[Dict[str, object]]]:
    """
    Train each robustness-matrix row and evaluate it across every column.

    When `robustness_matrix` is omitted, uses the canonical 4x4 protocol.
    Raises `ValueError` if no seeds are available after budget resolution.
    If `checkpoint_dir` is provided, per-train-condition checkpoints are
    stored and loaded from train/seed-specific subdirectories keyed by a
    stable config fingerprint. `checkpoint_selection` controls whether
    checkpoints are ignored (`"none"`) or selected (`"best"`) using
    `checkpoint_metric`. Returns `(payload, rows)`, where
    `payload["matrix"]` is the nested train->eval summary mapping and
    `rows` is the flattened per-episode behavior export.
    """
    if robustness_matrix is None:
        robustness_matrix = canonical_robustness_matrix()
    if checkpoint_selection not in {"none", "best"}:
        raise ValueError(
            "Invalid checkpoint_selection. Use 'none' or 'best'."
        )
    checkpoint_selection_config: CheckpointSelectionConfig | None = None
    if checkpoint_selection == "best":
        checkpoint_selection_config = CheckpointSelectionConfig(
            metric=checkpoint_metric,
            override_penalty_weight=checkpoint_override_penalty,
            dominance_penalty_weight=checkpoint_dominance_penalty,
            penalty_mode=checkpoint_penalty_mode,
        )
        checkpoint_candidate_sort_key(
            {},
            selection_config=checkpoint_selection_config,
        )
    budget = resolve_budget(
        profile=budget_profile,
        episodes=episodes,
        eval_episodes=evaluation_episodes,
        max_steps=max_steps,
        scenario_episodes=episodes_per_scenario,
        checkpoint_interval=checkpoint_interval,
        behavior_seeds=seeds,
        ablation_seeds=seeds,
    )
    scenario_names = list(names or SCENARIO_NAMES)
    run_count = max(1, int(budget.scenario_episodes))
    seed_values = tuple(seeds) if seeds is not None else budget.behavior_seeds
    if not seed_values:
        raise ValueError(
            "compare_noise_robustness() requires at least one seed."
        )

    rows: List[Dict[str, object]] = []
    matrix_payloads: Dict[str, Dict[str, object]] = {
        train_condition: {}
        for train_condition in robustness_matrix.train_conditions
    }

    for train_index, train_condition in enumerate(
        robustness_matrix.train_conditions
    ):
        resolved_train_noise_profile = resolve_noise_profile(train_condition)
        train_brain_config_summary: Dict[str, object] = {}
        train_eval_reflex_scale = 0.0
        combined_stats_by_eval = {
            eval_condition: {name: [] for name in scenario_names}
            for eval_condition in robustness_matrix.eval_conditions
        }
        combined_scores_by_eval = {
            eval_condition: {name: [] for name in scenario_names}
            for eval_condition in robustness_matrix.eval_conditions
        }
        seed_payloads_by_eval: dict[str, list[tuple[int, Dict[str, object]]]] = {
            eval_condition: []
            for eval_condition in robustness_matrix.eval_conditions
        }

        for seed in seed_values:
            sim_budget = budget.to_summary()
            sim_budget["resolved"]["scenario_episodes"] = run_count
            sim_budget["resolved"]["behavior_seeds"] = list(seed_values)
            sim_budget["resolved"]["ablation_seeds"] = list(seed_values)
            fingerprint_budget_resolved = {
                "episodes": budget.episodes,
                "eval_episodes": budget.eval_episodes,
                "max_steps": budget.max_steps,
                "scenario_episodes": run_count,
                "checkpoint_interval": budget.checkpoint_interval,
                "selection_scenario_episodes": (
                    budget.selection_scenario_episodes
                ),
            }
            preload_fingerprint = checkpoint_preload_fingerprint(
                load_brain,
                load_modules,
            )
            sim = SpiderSimulation(
                width=width,
                height=height,
                food_count=food_count,
                day_length=day_length,
                night_length=night_length,
                max_steps=budget.max_steps,
                seed=seed,
                gamma=gamma,
                module_lr=module_lr,
                motor_lr=motor_lr,
                module_dropout=module_dropout,
                operational_profile=operational_profile,
                noise_profile=resolved_train_noise_profile,
                reward_profile=reward_profile,
                map_template=map_template,
                budget_profile_name=budget.profile,
                benchmark_strength=budget.benchmark_strength,
                budget_summary=sim_budget,
            )

            brain_config_summary = sim.brain.config.to_summary()
            train_brain_config_summary = dict(brain_config_summary)
            architecture_fingerprint = sim.brain._architecture_fingerprint()
            run_fingerprint = checkpoint_run_fingerprint(
                {
                    "workflow": "noise_robustness",
                    "scenario_names": scenario_names,
                    "episodes_per_scenario": run_count,
                    "budget_profile": budget.profile,
                    "budget_benchmark_strength": budget.benchmark_strength,
                    "budget_resolved": fingerprint_budget_resolved,
                    "world": {
                        "width": width,
                        "height": height,
                        "food_count": food_count,
                        "day_length": day_length,
                        "night_length": night_length,
                        "max_steps": budget.max_steps,
                        "reward_profile": reward_profile,
                        "map_template": map_template,
                        "train_noise_profile": resolved_train_noise_profile.name,
                    },
                    "learning": {
                        "gamma": gamma,
                        "module_lr": module_lr,
                        "motor_lr": motor_lr,
                        "module_dropout": module_dropout,
                    },
                    "operational_profile": sim.operational_profile.to_summary(),
                    "architecture": brain_config_summary,
                    "architecture_fingerprint": architecture_fingerprint,
                    "checkpoint_selection": checkpoint_selection,
                    "checkpoint_metric": checkpoint_metric,
                    "checkpoint_penalty_config": (
                        checkpoint_selection_config.to_summary()
                        if checkpoint_selection_config is not None
                        else {
                            "metric": checkpoint_metric,
                            "override_penalty_weight": float(
                                checkpoint_override_penalty
                            ),
                            "dominance_penalty_weight": float(
                                checkpoint_dominance_penalty
                            ),
                            "penalty_mode": (
                                checkpoint_penalty_mode.value
                                if isinstance(
                                    checkpoint_penalty_mode,
                                    CheckpointPenaltyMode,
                                )
                                else str(checkpoint_penalty_mode)
                            ),
                        }
                    ),
                    "checkpoint_interval": budget.checkpoint_interval,
                    "selection_scenario_episodes": budget.selection_scenario_episodes,
                    "preload": preload_fingerprint,
                }
            )
            run_checkpoint_dir = None
            if checkpoint_dir is not None:
                run_checkpoint_dir = (
                    Path(checkpoint_dir)
                    / "noise_robustness"
                    / f"{train_condition}__seed_{seed}__{run_fingerprint}"
                )
            checkpoint_load_dir = resolve_checkpoint_load_dir(
                run_checkpoint_dir,
                checkpoint_selection=checkpoint_selection,
            )
            if checkpoint_load_dir is not None:
                sim.brain.load(checkpoint_load_dir)
                sim.checkpoint_source = (
                    checkpoint_load_dir.name
                    if checkpoint_load_dir.name in {"best", "last"}
                    else "checkpoint"
                )
            else:
                if load_brain is not None:
                    sim.brain.load(load_brain, modules=load_modules)
                should_train = (
                    checkpoint_selection == "best"
                    or budget.episodes > 0
                )
                if load_brain is not None and not should_train:
                    sim.checkpoint_source = "preloaded"
                if should_train:
                    sim.train(
                        budget.episodes,
                        evaluation_episodes=0,
                        render_last_evaluation=False,
                        capture_evaluation_trace=False,
                        debug_trace=False,
                        checkpoint_selection=checkpoint_selection,
                        checkpoint_metric=checkpoint_metric,
                        checkpoint_override_penalty=checkpoint_override_penalty,
                        checkpoint_dominance_penalty=checkpoint_dominance_penalty,
                        checkpoint_penalty_mode=checkpoint_penalty_mode,
                        checkpoint_interval=budget.checkpoint_interval,
                        checkpoint_dir=run_checkpoint_dir,
                        checkpoint_scenario_names=scenario_names,
                        selection_scenario_episodes=budget.selection_scenario_episodes,
                    )
            if save_brain is not None:
                save_root = (
                    Path(save_brain)
                    / "noise_robustness"
                    / f"{train_condition}__seed_{seed}__{run_fingerprint}"
                )
                sim.brain.save(save_root)
            train_eval_reflex_scale = float(sim._effective_reflex_scale())

            episodes_per_cell = max(
                1,
                run_count * max(1, len(scenario_names)),
            )
            eval_stride = episodes_per_cell
            train_stride = eval_stride * max(
                1,
                len(robustness_matrix.eval_conditions),
            )
            for eval_index, eval_condition in enumerate(
                robustness_matrix.eval_conditions
            ):
                with sim._swap_eval_noise_profile(eval_condition):
                    stats_histories, behavior_histories, _ = sim._execute_behavior_suite(
                        names=scenario_names,
                        episodes_per_scenario=run_count,
                        capture_trace=False,
                        debug_trace=False,
                        base_index=500_000
                        + train_index * train_stride
                        + eval_index * eval_stride,
                    )
                    seed_cell_payload = compact_behavior_payload(
                        sim._build_behavior_payload(
                            stats_histories=stats_histories,
                            behavior_histories=behavior_histories,
                        )
                    )
                    seed_payloads_by_eval[eval_condition].append(
                        (int(seed), seed_cell_payload)
                    )
                    for name in scenario_names:
                        combined_stats_by_eval[eval_condition][name].extend(
                            stats_histories[name]
                        )
                        combined_scores_by_eval[eval_condition][name].extend(
                            behavior_histories[name]
                        )
                        rows.extend(
                            sim._annotate_behavior_rows(
                                flatten_behavior_rows(
                                    behavior_histories[name],
                                    reward_profile=reward_profile,
                                    scenario_map=get_scenario(name).map_template,
                                    simulation_seed=seed,
                                    scenario_description=get_scenario(name).description,
                                    scenario_objective=get_scenario(name).objective,
                                    scenario_focus=get_scenario(name).diagnostic_focus,
                                    evaluation_map=map_template,
                                ),
                                train_noise_profile=resolved_train_noise_profile,
                            )
                        )

        for eval_condition in robustness_matrix.eval_conditions:
            resolved_eval_noise_profile = resolve_noise_profile(eval_condition)
            cell_payload = with_noise_profile_metadata(
                compact_behavior_payload(
                    sim._build_behavior_payload(
                        stats_histories=combined_stats_by_eval[eval_condition],
                        behavior_histories=combined_scores_by_eval[eval_condition],
                    )
                ),
                resolved_eval_noise_profile,
            )
            cell_payload["train_noise_profile"] = resolved_train_noise_profile.name
            cell_payload["train_noise_profile_config"] = (
                resolved_train_noise_profile.to_summary()
            )
            cell_payload["config"] = dict(train_brain_config_summary)
            cell_payload["summary"]["eval_reflex_scale"] = train_eval_reflex_scale
            cell_payload["eval_reflex_scale"] = train_eval_reflex_scale
            cell_payload["eval_noise_profile"] = resolved_eval_noise_profile.name
            cell_payload["eval_noise_profile_config"] = (
                resolved_eval_noise_profile.to_summary()
            )
            attach_behavior_seed_statistics(
                cell_payload,
                seed_payloads_by_eval[eval_condition],
                condition=f"{train_condition}->{eval_condition}",
                scenario_names=scenario_names,
            )
            matrix_payloads[train_condition][eval_condition] = cell_payload

    aggregate_metrics = robustness_aggregate_metrics(
        matrix_payloads,
        robustness_matrix=robustness_matrix,
    )
    return {
        "budget_profile": budget.profile,
        "benchmark_strength": budget.benchmark_strength,
        "checkpoint_selection": checkpoint_selection,
        "checkpoint_metric": checkpoint_metric,
        "checkpoint_penalty_config": (
            checkpoint_selection_config.to_summary()
            if checkpoint_selection_config is not None
            else {
                "metric": checkpoint_metric,
                "override_penalty_weight": float(checkpoint_override_penalty),
                "dominance_penalty_weight": float(checkpoint_dominance_penalty),
                "penalty_mode": (
                    checkpoint_penalty_mode.value
                    if isinstance(checkpoint_penalty_mode, CheckpointPenaltyMode)
                    else str(checkpoint_penalty_mode)
                ),
            }
        ),
        "reward_profile": reward_profile,
        "map_template": map_template,
        "seeds": list(seed_values),
        "scenario_names": scenario_names,
        "episodes_per_scenario": run_count,
        "matrix": matrix_payloads,
        **aggregate_metrics,
        **robustness_matrix_metadata(robustness_matrix),
    }, rows


def compare_ablation_suite(
    *,
    width: int = 12,
    height: int = 12,
    food_count: int = 4,
    day_length: int = 18,
    night_length: int = 12,
    max_steps: int | None = None,
    episodes: int | None = None,
    evaluation_episodes: int | None = None,
    gamma: float = 0.96,
    module_lr: float = 0.010,
    motor_lr: float = 0.012,
    module_dropout: float = 0.05,
    reward_profile: str = "classic",
    map_template: str = "central_burrow",
    operational_profile: str | OperationalProfile | None = None,
    noise_profile: str | NoiseConfig | None = None,
    budget_profile: str | BudgetProfile | None = None,
    seeds: Sequence[int] | None = None,
    names: Sequence[str] | None = None,
    variant_names: Sequence[str] | None = None,
    episodes_per_scenario: int | None = None,
    checkpoint_selection: str = "none",
    checkpoint_metric: str = "scenario_success_rate",
    checkpoint_override_penalty: float = 0.0,
    checkpoint_dominance_penalty: float = 0.0,
    checkpoint_penalty_mode: CheckpointPenaltyMode | str = (
        CheckpointPenaltyMode.TIEBREAKER
    ),
    checkpoint_interval: int | None = None,
    checkpoint_dir: str | Path | None = None,
) -> tuple[Dict[str, object], List[Dict[str, object]]]:
    """
    Compare ablation variants by optionally training them and evaluating behavior suites, returning per-variant aggregated payloads and flattened CSV-ready rows.

    Each variant (including a default reference) is executed across the requested seeds and scenarios. For each seed the simulation may be trained (controlled by `checkpoint_selection` and budget), then evaluated twice: once with the runtime reflex support preserved (diagnostic/scaffolded) and once with reflexes disabled (primary/self-sufficient). The primary evaluation recorded in the returned payloads is the no-reflex result; reflex-enabled results are included for diagnostics and delta computations versus the reference.

    Returns:
        tuple:
            payload (dict): Summary payload containing keys including:
                - "reference_variant": name of the reference variant,
                - "scenario_names": list of evaluated scenario names,
                - "episodes_per_scenario": number of runs per scenario,
                - "variants": mapping from variant name to a compact behavior-suite payload where
                  the primary evaluation is the no-reflex result and reflex-enabled results appear under
                  "with_reflex_support" / "without_reflex_support",
                - "deltas_vs_reference": per-variant delta metrics versus the reference,
                - checkpoint selection metadata and resolved noise profile metadata.
            rows (list[dict]): Flattened, annotated rows for every evaluated episode (suitable for CSV export),
                including ablation, reflex/evaluation metadata, seed and scenario details.
    """
    resolved_noise_profile = resolve_noise_profile(noise_profile)
    budget = resolve_budget(
        profile=budget_profile,
        episodes=episodes,
        eval_episodes=evaluation_episodes,
        max_steps=max_steps,
        scenario_episodes=episodes_per_scenario,
        checkpoint_interval=checkpoint_interval,
        behavior_seeds=seeds,
        ablation_seeds=seeds,
    )
    scenario_names = list(names or SCENARIO_NAMES)
    requested_configs = resolve_ablation_configs(
        variant_names,
        module_dropout=module_dropout,
    )
    reference_config = default_brain_config(module_dropout=module_dropout)
    configs: List[BrainAblationConfig] = [reference_config]
    for config in requested_configs:
        if config.name != reference_config.name:
            configs.append(config)

    run_count = max(1, int(budget.scenario_episodes))
    seed_values = tuple(seeds) if seeds is not None else budget.ablation_seeds
    rows: List[Dict[str, object]] = []
    variants: Dict[str, Dict[str, object]] = {}

    for config in configs:
        combined_stats = {name: [] for name in scenario_names}
        combined_scores = {name: [] for name in scenario_names}
        combined_stats_without_reflex = {name: [] for name in scenario_names}
        combined_scores_without_reflex = {name: [] for name in scenario_names}
        seed_payloads_with_reflex: list[tuple[int, Dict[str, object]]] = []
        seed_payloads_without_reflex: list[tuple[int, Dict[str, object]]] = []
        sim: SpiderSimulation | None = None
        behavior_base_index = 300_000
        for seed in seed_values:
            sim_budget = budget.to_summary()
            sim_budget["resolved"]["scenario_episodes"] = run_count
            sim_budget["resolved"]["behavior_seeds"] = list(budget.behavior_seeds)
            sim_budget["resolved"]["ablation_seeds"] = list(seed_values)
            sim = SpiderSimulation(
                width=width,
                height=height,
                food_count=food_count,
                day_length=day_length,
                night_length=night_length,
                max_steps=budget.max_steps,
                seed=seed,
                gamma=gamma,
                module_lr=module_lr,
                motor_lr=motor_lr,
                module_dropout=config.module_dropout,
                operational_profile=operational_profile,
                noise_profile=resolved_noise_profile,
                reward_profile=reward_profile,
                map_template=map_template,
                brain_config=config,
                budget_profile_name=budget.profile,
                benchmark_strength=budget.benchmark_strength,
                budget_summary=sim_budget,
            )
            if (
                checkpoint_selection == "best"
                or budget.episodes > 0
                or budget.eval_episodes > 0
            ):
                run_checkpoint_dir = None
                if checkpoint_dir is not None:
                    run_checkpoint_dir = (
                        Path(checkpoint_dir)
                        / "ablation_compare"
                        / f"{config.name}__seed_{seed}"
                    )
                sim.train(
                    budget.episodes,
                    evaluation_episodes=0,
                    render_last_evaluation=False,
                    capture_evaluation_trace=False,
                    debug_trace=False,
                    checkpoint_selection=checkpoint_selection,
                    checkpoint_metric=checkpoint_metric,
                    checkpoint_override_penalty=checkpoint_override_penalty,
                    checkpoint_dominance_penalty=checkpoint_dominance_penalty,
                    checkpoint_penalty_mode=checkpoint_penalty_mode,
                    checkpoint_interval=budget.checkpoint_interval,
                    checkpoint_dir=run_checkpoint_dir,
                    checkpoint_scenario_names=scenario_names,
                    selection_scenario_episodes=budget.selection_scenario_episodes,
                )
            stats_histories, behavior_histories, _ = sim._execute_behavior_suite(
                names=scenario_names,
                episodes_per_scenario=run_count,
                capture_trace=False,
                debug_trace=False,
                base_index=behavior_base_index,
            )
            previous_reflex_scale = float(sim.brain.current_reflex_scale)
            sim.brain.set_runtime_reflex_scale(0.0)
            try:
                no_reflex_stats_histories, no_reflex_behavior_histories, _ = sim._execute_behavior_suite(
                    names=scenario_names,
                    episodes_per_scenario=run_count,
                    capture_trace=False,
                    debug_trace=False,
                    base_index=behavior_base_index,
                )
            finally:
                sim.brain.set_runtime_reflex_scale(previous_reflex_scale)
            seed_with_reflex_payload = compact_behavior_payload(
                sim._build_behavior_payload(
                    stats_histories=stats_histories,
                    behavior_histories=behavior_histories,
                    competence_label="scaffolded",
                )
            )
            seed_with_reflex_payload["summary"]["eval_reflex_scale"] = float(
                previous_reflex_scale
            )
            seed_with_reflex_payload["summary"]["competence_type"] = "scaffolded"
            seed_payloads_with_reflex.append((int(seed), seed_with_reflex_payload))
            seed_without_reflex_payload = compact_behavior_payload(
                sim._build_behavior_payload(
                    stats_histories=no_reflex_stats_histories,
                    behavior_histories=no_reflex_behavior_histories,
                    competence_label="self_sufficient",
                )
            )
            seed_without_reflex_payload["summary"]["eval_reflex_scale"] = 0.0
            seed_without_reflex_payload["summary"]["competence_type"] = (
                "self_sufficient"
            )
            seed_payloads_without_reflex.append(
                (int(seed), seed_without_reflex_payload)
            )
            for name in scenario_names:
                combined_stats[name].extend(stats_histories[name])
                combined_scores[name].extend(behavior_histories[name])
                combined_stats_without_reflex[name].extend(no_reflex_stats_histories[name])
                combined_scores_without_reflex[name].extend(no_reflex_behavior_histories[name])
                rows.extend(
                    sim._annotate_behavior_rows(
                        flatten_behavior_rows(
                            behavior_histories[name],
                            reward_profile=reward_profile,
                            scenario_map=get_scenario(name).map_template,
                            simulation_seed=seed,
                            scenario_description=get_scenario(name).description,
                            scenario_objective=get_scenario(name).objective,
                            scenario_focus=get_scenario(name).diagnostic_focus,
                            evaluation_map=map_template,
                            eval_reflex_scale=previous_reflex_scale,
                            competence_label="scaffolded",
                        ),
                        eval_reflex_scale=previous_reflex_scale,
                    )
                )
                rows.extend(
                    sim._annotate_behavior_rows(
                        flatten_behavior_rows(
                            no_reflex_behavior_histories[name],
                            reward_profile=reward_profile,
                            scenario_map=get_scenario(name).map_template,
                            simulation_seed=seed,
                            scenario_description=get_scenario(name).description,
                            scenario_objective=get_scenario(name).objective,
                            scenario_focus=get_scenario(name).diagnostic_focus,
                            evaluation_map=map_template,
                            eval_reflex_scale=0.0,
                            competence_label="self_sufficient",
                        ),
                        eval_reflex_scale=0.0,
                    )
                )
        if sim is None:
            continue
        with_reflex_payload = compact_behavior_payload(
            sim._build_behavior_payload(
                stats_histories=combined_stats,
                behavior_histories=combined_scores,
                competence_label="scaffolded",
            )
        )
        with_reflex_payload["summary"]["eval_reflex_scale"] = float(
            previous_reflex_scale
        )
        with_reflex_payload["summary"]["competence_type"] = "scaffolded"
        with_reflex_payload["eval_reflex_scale"] = float(previous_reflex_scale)
        with_reflex_payload["competence_type"] = "scaffolded"
        with_reflex_payload["config"] = config.to_summary()
        attach_behavior_seed_statistics(
            with_reflex_payload,
            seed_payloads_with_reflex,
            condition=config.name,
            scenario_names=scenario_names,
        )
        with_reflex_payload.update(
            noise_profile_metadata(resolved_noise_profile)
        )
        without_reflex_payload = with_noise_profile_metadata(
            compact_behavior_payload(
                sim._build_behavior_payload(
                    stats_histories=combined_stats_without_reflex,
                    behavior_histories=combined_scores_without_reflex,
                    competence_label="self_sufficient",
                )
            ),
            resolved_noise_profile,
        )
        without_reflex_payload["summary"]["eval_reflex_scale"] = 0.0
        without_reflex_payload["summary"]["competence_type"] = "self_sufficient"
        without_reflex_payload["eval_reflex_scale"] = 0.0
        without_reflex_payload["competence_type"] = "self_sufficient"
        without_reflex_payload["config"] = config.to_summary()
        attach_behavior_seed_statistics(
            without_reflex_payload,
            seed_payloads_without_reflex,
            condition=config.name,
            scenario_names=scenario_names,
        )
        compact_payload = dict(without_reflex_payload)
        compact_payload["primary_evaluation"] = "without_reflex_support"
        compact_payload["with_reflex_support"] = with_reflex_payload
        compact_payload["without_reflex_support"] = without_reflex_payload
        variants[config.name] = compact_payload

    reference_variant = reference_config.name
    deltas_vs_reference = build_ablation_deltas(
        variants,
        reference_variant=reference_variant,
        scenario_names=scenario_names,
    )
    return {
        "budget_profile": budget.profile,
        "benchmark_strength": budget.benchmark_strength,
        "checkpoint_selection": checkpoint_selection,
        "checkpoint_metric": checkpoint_metric,
        "checkpoint_penalty_config": CheckpointSelectionConfig(
            metric=checkpoint_metric,
            override_penalty_weight=checkpoint_override_penalty,
            dominance_penalty_weight=checkpoint_dominance_penalty,
            penalty_mode=checkpoint_penalty_mode,
        ).to_summary(),
        "primary_evaluation": "without_reflex_support",
        "reference_variant": reference_variant,
        "reference_eval_reflex_scale": 0.0,
        "seeds": list(seed_values),
        "scenario_names": scenario_names,
        "episodes_per_scenario": run_count,
        "variants": variants,
        "deltas_vs_reference": deltas_vs_reference,
        "predator_type_specialization": build_predator_type_specialization_summary(
            variants,
            reference_variant=reference_variant,
            deltas_vs_reference=deltas_vs_reference,
        ),
        **noise_profile_metadata(resolved_noise_profile),
    }, rows


def compare_learning_evidence(
    *,
    width: int = 12,
    height: int = 12,
    food_count: int = 4,
    day_length: int = 18,
    night_length: int = 12,
    max_steps: int | None = None,
    episodes: int | None = None,
    evaluation_episodes: int | None = None,
    gamma: float = 0.96,
    module_lr: float = 0.010,
    motor_lr: float = 0.012,
    module_dropout: float = 0.05,
    reward_profile: str = "classic",
    map_template: str = "central_burrow",
    brain_config: BrainAblationConfig | None = None,
    operational_profile: str | OperationalProfile | None = None,
    noise_profile: str | NoiseConfig | None = None,
    budget_profile: str | BudgetProfile | None = None,
    long_budget_profile: str | BudgetProfile | None = "report",
    seeds: Sequence[int] | None = None,
    names: Sequence[str] | None = None,
    condition_names: Sequence[str] | None = None,
    episodes_per_scenario: int | None = None,
    checkpoint_selection: str = "none",
    checkpoint_metric: str = "scenario_success_rate",
    checkpoint_override_penalty: float = 0.0,
    checkpoint_dominance_penalty: float = 0.0,
    checkpoint_penalty_mode: CheckpointPenaltyMode | str = (
        CheckpointPenaltyMode.TIEBREAKER
    ),
    checkpoint_interval: int | None = None,
    checkpoint_dir: str | Path | None = None,
) -> tuple[Dict[str, object], List[Dict[str, object]]]:
    """
    Run a registry of learning-evidence conditions across seeds and scenarios and produce compact comparison results.

    For each resolved condition this method trains or initializes simulations according to the condition's specified budget (base, long, freeze_half, or initial), optionally persists checkpoints, executes the behavior suite with the condition's evaluation policy mode, and aggregates per-condition behavior-suite payloads and flattened CSV rows. Conditions that are incompatible with the base architecture are marked as skipped. The returned payload contains budget and noise metadata, per-condition compact behavior summaries under `"conditions"`, deltas versus the no-reflex reference condition (`"trained_without_reflex_support"`), and a synthesized evidence summary.

    Returns:
        tuple: A pair (payload, rows) where
            payload (dict): Compact comparison payload including:
                - budget and long_budget identifiers and benchmark strengths
                - checkpoint selection settings and reference condition
                - list of seeds and scenario names
                - aggregated `conditions` mapping (compact per-condition behavior payloads)
                - `deltas_vs_reference` and `evidence_summary` computed from conditions
                - noise profile metadata
            rows (list[dict]): Flattened, annotated per-episode CSV rows for all runs with learning-evidence metadata fields added.
    """
    resolved_noise_profile = resolve_noise_profile(noise_profile)
    base_budget = resolve_budget(
        profile=budget_profile,
        episodes=episodes,
        eval_episodes=evaluation_episodes,
        max_steps=max_steps,
        scenario_episodes=episodes_per_scenario,
        checkpoint_interval=checkpoint_interval,
        behavior_seeds=seeds,
        ablation_seeds=seeds,
    )
    long_budget = resolve_budget(
        profile=long_budget_profile,
        episodes=None,
        eval_episodes=None,
        max_steps=max_steps if max_steps is not None else base_budget.max_steps,
        scenario_episodes=episodes_per_scenario,
        checkpoint_interval=checkpoint_interval,
        behavior_seeds=seeds,
        ablation_seeds=seeds,
    )
    scenario_names = list(names or SCENARIO_NAMES)
    condition_specs = resolve_learning_evidence_conditions(condition_names)
    primary_reference_condition = "trained_without_reflex_support"
    if condition_names is not None and not any(
        spec.name == primary_reference_condition for spec in condition_specs
    ):
        condition_specs = resolve_learning_evidence_conditions(
            (primary_reference_condition, *tuple(condition_names))
        )
    seed_values = tuple(seeds) if seeds is not None else base_budget.behavior_seeds
    run_count = max(1, int(base_budget.scenario_episodes))
    base_config = (
        brain_config
        if brain_config is not None
        else default_brain_config(module_dropout=module_dropout)
    )

    rows: List[Dict[str, object]] = []
    conditions: Dict[str, Dict[str, object]] = {}

    for condition_index, condition in enumerate(condition_specs):
        if base_config.architecture not in condition.supports_architectures:
            conditions[condition.name] = {
                "condition": condition.name,
                "description": condition.description,
                "policy_mode": condition.policy_mode,
                "train_budget": condition.train_budget,
                "training_regime": condition.training_regime,
                "checkpoint_source": condition.checkpoint_source,
                "budget_profile": (
                    long_budget.profile
                    if condition.train_budget == "long"
                    else base_budget.profile
                ),
                "benchmark_strength": (
                    long_budget.benchmark_strength
                    if condition.train_budget == "long"
                    else base_budget.benchmark_strength
                ),
                "config": base_config.to_summary(),
                "skipped": True,
                "reason": (
                    f"Condition incompatible with architecture {base_config.architecture!r}."
                ),
                **noise_profile_metadata(resolved_noise_profile),
            }
            continue
        if (
            condition.policy_mode in {"reflex_only", "reflex-only"}
            and not base_config.enable_reflexes
        ):
            conditions[condition.name] = {
                "condition": condition.name,
                "description": condition.description,
                "policy_mode": condition.policy_mode,
                "train_budget": condition.train_budget,
                "training_regime": condition.training_regime,
                "checkpoint_source": condition.checkpoint_source,
                "budget_profile": (
                    long_budget.profile
                    if condition.train_budget == "long"
                    else base_budget.profile
                ),
                "benchmark_strength": (
                    long_budget.benchmark_strength
                    if condition.train_budget == "long"
                    else base_budget.benchmark_strength
                ),
                "config": base_config.to_summary(),
                "skipped": True,
                "reason": (
                    f"Condition {condition.name!r} requires reflexes to be enabled, "
                    "but the base configuration has reflexes disabled."
                ),
                **noise_profile_metadata(resolved_noise_profile),
            }
            continue

        combined_stats = {name: [] for name in scenario_names}
        combined_scores = {name: [] for name in scenario_names}
        exemplar_sim: SpiderSimulation | None = None
        condition_budget = long_budget if condition.train_budget == "long" else base_budget
        seed_payloads: list[tuple[int, Dict[str, object]]] = []
        train_episodes = 0
        frozen_after_episode: int | None = None
        observed_checkpoint_source = condition.checkpoint_source
        observed_eval_reflex_scale: float | None = None

        for seed in seed_values:
            sim_budget = condition_budget.to_summary()
            sim_budget["resolved"]["scenario_episodes"] = run_count
            sim_budget["resolved"]["behavior_seeds"] = list(seed_values)
            sim_budget["resolved"]["ablation_seeds"] = list(seed_values)
            sim = SpiderSimulation(
                width=width,
                height=height,
                food_count=food_count,
                day_length=day_length,
                night_length=night_length,
                max_steps=condition_budget.max_steps,
                seed=seed,
                gamma=gamma,
                module_lr=module_lr,
                motor_lr=motor_lr,
                module_dropout=base_config.module_dropout,
                reward_profile=reward_profile,
                map_template=map_template,
                brain_config=base_config,
                operational_profile=operational_profile,
                noise_profile=resolved_noise_profile,
                budget_profile_name=condition_budget.profile,
                benchmark_strength=condition_budget.benchmark_strength,
                budget_summary=sim_budget,
            )
            exemplar_sim = sim

            if condition.train_budget == "base":
                run_checkpoint_dir = None
                if checkpoint_dir is not None:
                    run_checkpoint_dir = (
                        Path(checkpoint_dir)
                        / "learning_evidence"
                        / f"{condition.name}__seed_{seed}"
                    )
                sim.train(
                    condition_budget.episodes,
                    evaluation_episodes=0,
                    render_last_evaluation=False,
                    capture_evaluation_trace=False,
                    debug_trace=False,
                    checkpoint_selection=checkpoint_selection,
                    checkpoint_metric=checkpoint_metric,
                    checkpoint_override_penalty=checkpoint_override_penalty,
                    checkpoint_dominance_penalty=checkpoint_dominance_penalty,
                    checkpoint_penalty_mode=checkpoint_penalty_mode,
                    checkpoint_interval=condition_budget.checkpoint_interval,
                    checkpoint_dir=run_checkpoint_dir,
                    checkpoint_scenario_names=scenario_names,
                    selection_scenario_episodes=condition_budget.selection_scenario_episodes,
                    training_regime=condition.training_regime,
                )
                train_episodes = int(
                    sim._latest_training_regime_summary.get(
                        "resolved_budget",
                        {},
                    ).get(
                        "total_training_episodes",
                        condition_budget.episodes,
                    )
                )
                observed_checkpoint_source = str(sim.checkpoint_source)
            elif condition.train_budget == "long":
                run_checkpoint_dir = None
                if checkpoint_dir is not None:
                    run_checkpoint_dir = (
                        Path(checkpoint_dir)
                        / "learning_evidence"
                        / f"{condition.name}__seed_{seed}"
                    )
                sim.train(
                    long_budget.episodes,
                    evaluation_episodes=0,
                    render_last_evaluation=False,
                    capture_evaluation_trace=False,
                    debug_trace=False,
                    checkpoint_selection=checkpoint_selection,
                    checkpoint_metric=checkpoint_metric,
                    checkpoint_override_penalty=checkpoint_override_penalty,
                    checkpoint_dominance_penalty=checkpoint_dominance_penalty,
                    checkpoint_penalty_mode=checkpoint_penalty_mode,
                    checkpoint_interval=long_budget.checkpoint_interval,
                    checkpoint_dir=run_checkpoint_dir,
                    checkpoint_scenario_names=scenario_names,
                    selection_scenario_episodes=long_budget.selection_scenario_episodes,
                    training_regime=condition.training_regime,
                )
                train_episodes = int(
                    sim._latest_training_regime_summary.get(
                        "resolved_budget",
                        {},
                    ).get(
                        "total_training_episodes",
                        long_budget.episodes,
                    )
                )
                observed_checkpoint_source = str(sim.checkpoint_source)
            elif condition.train_budget == "freeze_half":
                train_episodes = max(0, int(base_budget.episodes) // 2)
                run_checkpoint_dir = None
                if checkpoint_dir is not None:
                    run_checkpoint_dir = (
                        Path(checkpoint_dir)
                        / "learning_evidence"
                        / f"{condition.name}__seed_{seed}"
                    )
                sim.train(
                    train_episodes,
                    evaluation_episodes=0,
                    render_last_evaluation=False,
                    capture_evaluation_trace=False,
                    debug_trace=False,
                    checkpoint_selection=checkpoint_selection,
                    checkpoint_metric=checkpoint_metric,
                    checkpoint_override_penalty=checkpoint_override_penalty,
                    checkpoint_dominance_penalty=checkpoint_dominance_penalty,
                    checkpoint_penalty_mode=checkpoint_penalty_mode,
                    checkpoint_interval=base_budget.checkpoint_interval,
                    checkpoint_dir=run_checkpoint_dir,
                    checkpoint_scenario_names=scenario_names,
                    selection_scenario_episodes=base_budget.selection_scenario_episodes,
                    training_regime=condition.training_regime,
                )
                train_episodes = int(
                    sim._latest_training_regime_summary.get(
                        "resolved_budget",
                        {},
                    ).get(
                        "total_training_episodes",
                        train_episodes,
                    )
                )
                frozen_after_episode = train_episodes
                observed_checkpoint_source = str(sim.checkpoint_source)
                remaining_episodes = max(0, int(base_budget.episodes) - train_episodes)
                if remaining_episodes:
                    sim._consume_episodes_without_learning(
                        episodes=remaining_episodes,
                        episode_start=train_episodes,
                        policy_mode="normal",
                    )
            else:
                train_episodes = 0
                sim.checkpoint_source = "initial"
                observed_checkpoint_source = "initial"

            previous_reflex_scale = float(sim.brain.current_reflex_scale)
            effective_eval_reflex_scale = (
                float(condition.eval_reflex_scale)
                if condition.eval_reflex_scale is not None
                else sim._effective_reflex_scale(sim.brain.current_reflex_scale)
            )
            observed_eval_reflex_scale = effective_eval_reflex_scale
            sim.brain.set_runtime_reflex_scale(effective_eval_reflex_scale)
            try:
                stats_histories, behavior_histories, _ = sim._execute_behavior_suite(
                    names=scenario_names,
                    episodes_per_scenario=run_count,
                    capture_trace=False,
                    debug_trace=False,
                    base_index=400_000 + condition_index * 10_000,
                    policy_mode=condition.policy_mode,
                )
            finally:
                sim.brain.set_runtime_reflex_scale(previous_reflex_scale)

            seed_compact_payload = compact_behavior_payload(
                sim._build_behavior_payload(
                    stats_histories=stats_histories,
                    behavior_histories=behavior_histories,
                    competence_label=competence_label_from_eval_reflex_scale(
                        effective_eval_reflex_scale
                    ),
                )
            )
            seed_compact_payload["summary"]["eval_reflex_scale"] = (
                effective_eval_reflex_scale
            )
            seed_payloads.append((int(seed), seed_compact_payload))

            extra_row_metadata = {
                "learning_evidence_condition": condition.name,
                "learning_evidence_policy_mode": condition.policy_mode,
                "learning_evidence_training_regime": (
                    "" if condition.training_regime is None else condition.training_regime
                ),
                "learning_evidence_train_episodes": int(train_episodes),
                "learning_evidence_frozen_after_episode": (
                    "" if frozen_after_episode is None else int(frozen_after_episode)
                ),
                "learning_evidence_checkpoint_source": (
                    condition.checkpoint_source
                    if condition.train_budget == "freeze_half"
                    else observed_checkpoint_source
                ),
                "learning_evidence_budget_profile": condition_budget.profile,
                "learning_evidence_budget_benchmark_strength": condition_budget.benchmark_strength,
            }
            for name in scenario_names:
                combined_stats[name].extend(stats_histories[name])
                combined_scores[name].extend(behavior_histories[name])
                rows.extend(
                    sim._annotate_behavior_rows(
                        flatten_behavior_rows(
                            behavior_histories[name],
                            reward_profile=reward_profile,
                            scenario_map=get_scenario(name).map_template,
                            simulation_seed=seed,
                            scenario_description=get_scenario(name).description,
                            scenario_objective=get_scenario(name).objective,
                            scenario_focus=get_scenario(name).diagnostic_focus,
                            evaluation_map=map_template,
                            eval_reflex_scale=effective_eval_reflex_scale,
                            competence_label=competence_label_from_eval_reflex_scale(
                                effective_eval_reflex_scale
                            ),
                        ),
                        eval_reflex_scale=effective_eval_reflex_scale,
                        extra_metadata=extra_row_metadata,
                    )
                )

        if exemplar_sim is None:
            continue

        compact_payload = compact_behavior_payload(
            exemplar_sim._build_behavior_payload(
                stats_histories=combined_stats,
                behavior_histories=combined_scores,
                competence_label=competence_label_from_eval_reflex_scale(
                    observed_eval_reflex_scale
                ),
            )
        )
        compact_payload["summary"]["eval_reflex_scale"] = (
            observed_eval_reflex_scale
        )
        compact_payload["eval_reflex_scale"] = observed_eval_reflex_scale
        attach_behavior_seed_statistics(
            compact_payload,
            seed_payloads,
            condition=condition.name,
            scenario_names=scenario_names,
        )
        compact_payload.update(
            {
                "condition": condition.name,
                "description": condition.description,
                "policy_mode": condition.policy_mode,
                "train_budget": condition.train_budget,
                "training_regime": condition.training_regime,
                "train_episodes": int(train_episodes),
                "frozen_after_episode": frozen_after_episode,
                "checkpoint_source": (
                    condition.checkpoint_source
                    if condition.train_budget == "freeze_half"
                    else observed_checkpoint_source
                ),
                "budget_profile": condition_budget.profile,
                "benchmark_strength": condition_budget.benchmark_strength,
                "config": base_config.to_summary(),
                "skipped": False,
            }
        )
        compact_payload.update(noise_profile_metadata(resolved_noise_profile))
        conditions[condition.name] = compact_payload

    reference_condition = primary_reference_condition
    return {
        "budget_profile": base_budget.profile,
        "benchmark_strength": base_budget.benchmark_strength,
        "long_budget_profile": long_budget.profile,
        "long_budget_benchmark_strength": long_budget.benchmark_strength,
        "checkpoint_selection": checkpoint_selection,
        "checkpoint_metric": checkpoint_metric,
        "checkpoint_penalty_config": CheckpointSelectionConfig(
            metric=checkpoint_metric,
            override_penalty_weight=checkpoint_override_penalty,
            dominance_penalty_weight=checkpoint_dominance_penalty,
            penalty_mode=checkpoint_penalty_mode,
        ).to_summary(),
        "reference_condition": reference_condition,
        "seeds": list(seed_values),
        "scenario_names": scenario_names,
        "episodes_per_scenario": run_count,
        "conditions": conditions,
        "deltas_vs_reference": build_learning_evidence_deltas(
            conditions,
            reference_condition=reference_condition,
            scenario_names=scenario_names,
        ),
        "evidence_summary": build_learning_evidence_summary(
            conditions,
            reference_condition=reference_condition,
        ),
        **noise_profile_metadata(resolved_noise_profile),
    }, rows
