"""Cross-run comparison and condensed reporting helpers.

``condense_robustness_summary`` is the public home for the CLI helper formerly
named ``_short_robustness_matrix_summary``.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

from .export import compact_aggregate
from .memory import memory_leakage_audit
from .metrics import (
    EpisodeStats,
    aggregate_episode_stats,
)
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

from .comparison_noise import profile_comparison_metrics
from .comparison_utils import safe_float

def comparison_suite_from_payload(
    payload: Dict[str, object] | None,
) -> Dict[str, object]:
    """
    Derive a mapping of per-scenario comparison entries from a payload containing one of several recognized shapes.
    
    Accepts a payload that may already include a "suite" mapping, a legacy "legacy_scenarios" mapping, or an "episodes_detail" list. Returns a per-scenario dictionary where each value contains keys such as "success_rate", "episodes", "mean_reward", and "success_basis". If the payload is not a dict or contains no recognizable structure, returns an empty dict.
    
    Parameters:
        payload (dict | None): Payload potentially containing comparison data in one of the supported shapes.
    
    Returns:
        dict: Mapping of scenario names to derived scenario entries, or an empty dict when no usable suite can be extracted.
    """
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
    """
    Produce a structured reward-audit payload derived from a sequence of episode statistics.
    
    Parameters:
        history (Sequence[EpisodeStats]): Iterable of per-episode statistics; episodes lacking a scenario identifier are ignored when building per-scenario entries.
    
    Returns:
        Dict[str, object]: A payload containing aggregated metrics. When per-scenario data is present the payload includes:
            - "suite": mapping of scenario name -> { "success_rate", "episodes", "mean_reward", "success_basis", "legacy_metrics" } with numeric values rounded to six decimals where applicable.
            - "legacy_scenarios": mapping of scenario name -> raw aggregated metrics (uncompacted).
          Always includes:
            - "summary": { "scenario_success_rate", "episode_success_rate", "mean_reward", "scenario_count", "episode_count", "success_basis" } with numeric values rounded to six decimals.
          If no per-scenario entries are found the payload contains only global aggregates and the summary is derived from those global values.
    """
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
        scenario_success_values = [
            safe_float(dict(item).get("success_rate"))
            for item in suite.values()
            if isinstance(item, dict)
        ]
        scenario_success_rate = (
            sum(scenario_success_values) / len(scenario_success_values)
            if scenario_success_values
            else 0.0
        )
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
    """
    Locate and return an austere comparison dictionary from a mapping of payloads.
    
    Parameters:
        payloads (Dict[str, Dict[str, object]]): Mapping from profile names to payload dictionaries to search for an austere comparison block.
    
    Returns:
        Dict[str, object] | None: The discovered comparison dictionary if found, otherwise `None`.
    """
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
    """
    Summarize austere survival results into gate and warning counters plus related metadata.
    
    Parameters:
        behavior_survival (dict): Behavior survival block that may contain "scenarios" (mapping of scenario -> payload),
            "available" (bool), and "survival_rate" (numeric).
        gap_policy_check (dict): Gap policy check object that may contain a "violations" sequence.
    
    Returns:
        dict: Summary containing:
            - "available" (bool): whether behavior_survival is marked available.
            - "overall_survival_rate" (float): rounded survival_rate from behavior_survival.
            - "expected_gate_count" (int): number of scenarios defined as gate requirements.
            - "observed_gate_count" (int): number of gate scenarios observed in the payload.
            - "gate_coverage_complete" (bool): whether observed_gate_count equals expected_gate_count.
            - "gate_pass_count" (int): count of gate scenarios that survived.
            - "gate_fail_count" (int): count of gate scenarios that failed.
            - "warning_scenarios" (list): entries for warning-level scenarios that failed with keys
                "scenario", "austere_success_rate", and "survival_threshold".
            - "gap_policy_violations" (list): violations extracted from gap_policy_check.
    """
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
    """
    Identify scenarios where non-minimal reward profiles exceed the minimal (austere) profile's scenario success rate by more than configured limits.
    
    Parameters:
    	profile_payloads (Dict[str, object]): Mapping of profile names to their comparison payloads.
    	minimal_profile (str | None): Name of the minimal (austere) profile to compare against.
    
    Returns:
    	list[Dict[str, object]]: A list of records for each detected excess, where each record contains:
    		- "scenario": scenario identifier (str)
    		- "profile": profile name that exceeds the limit (str)
    		- "comparison": comparison key of the form "{profile}_minus_austere" (str)
    		- "success_rate_delta": profile success rate minus austere success rate (rounded float)
    		- "limit": configured maximum allowed delta for that comparison (rounded float)
    		- "profile_success_rate": profile's scenario success rate (rounded float)
    		- "austere_success_rate": austere profile's scenario success rate (rounded float)
    """
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
    """
    Determine whether austere survival gating conditions passed for the provided comparison payload.
    
    Returns:
        `True` if an austere survival summary is available, gate coverage is complete, and `gate_fail_count` is 0; `False` if an available summary indicates gates did not pass; `None` if `comparison` is not a dict or the austere survival summary is unavailable.
    """
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
    """
    Summarize which reward-reduction roadmap terms are at risk and the current gap-policy and austere-survival status.
    
    Parameters:
        reward_audit (dict): Top-level reward audit payload; may contain a `"comparison"` mapping with `gap_policy_check` and austere-survival data.
    
    Returns:
        dict: Summary containing:
            - `roadmap_target_count` (int): Number of entries in the shaping reduction roadmap.
            - `terms_at_risk` (list[str]): Roadmap component names flagged as `under_investigation` or high priority for reduction.
            - `terms_at_risk_count` (int): Length of `terms_at_risk`.
            - `gap_policy_violation_count` (int): Number of gap-policy violations found (from `comparison["gap_policy_check"]["violations"]`).
            - `gap_policy_violations` (list): List of gap-policy violation entries.
            - `austere_survival_available` (bool): `True` if austere-survival data was present, `False` otherwise.
            - `austere_survival_gate_passed` (bool or None): `True` if austere gates passed, `False` if failed, `None` if data unavailable.
    """
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
