"""Aggregate episode statistics and flatten behavior-suite outputs."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict
from statistics import mean
from typing import Any, Callable, Dict, List, Sequence

from ..ablations import PROPOSAL_SOURCE_NAMES, REFLEX_MODULE_NAMES
from .types import (
    BehaviorCheckResult,
    BehavioralEpisodeScore,
    BehaviorCheckSpec,
    EpisodeStats,
    PREDATOR_TYPE_NAMES,
    PRIMARY_REPRESENTATION_READOUT_MODULES,
    SHELTER_ROLES,
    competence_label_from_eval_reflex_scale,
    normalize_competence_label,
)

def _extra_module_names(history: List[EpisodeStats], attr: str) -> List[str]:
    """Return sorted unique module names for an attribute across history, excluding PROPOSAL_SOURCE_NAMES."""
    return sorted(
        {name for stats in history for name in getattr(stats, attr) if name not in PROPOSAL_SOURCE_NAMES}
    )


def _mean_map(
    history: List[EpisodeStats],
    names: Sequence[str],
    getter: "Callable[[EpisodeStats, str], float]",
) -> Dict[str, float]:
    """
    Compute mean values per name from a history of EpisodeStats.
    
    Parameters:
        history (List[EpisodeStats]): Episodes to aggregate.
        names (Sequence[str]): Keys for which to compute means.
        getter (Callable[[EpisodeStats, str], float]): Function that returns the value for a given episode and name.
    
    Returns:
        Dict[str, float]: Mapping each name to its mean value; when `history` is empty each name maps to `0.0`.
    """
    if not history:
        return {name: 0.0 for name in names}
    return {name: mean(getter(stats, name) for stats in history) for name in names}


def _mean_scalar(
    history: List[EpisodeStats],
    getter: "Callable[[EpisodeStats], float]",
) -> float:
    """
    Compute the mean value produced by applying `getter` to each EpisodeStats in `history`. If `history` is empty, returns 0.0.
    
    Parameters:
        history (List[EpisodeStats]): Sequence of episode statistics to aggregate.
        getter (Callable[[EpisodeStats], float]): Function that extracts a scalar from an EpisodeStats.
    
    Returns:
        float: Arithmetic mean of extracted values, or 0.0 when `history` is empty.
    """
    if not history:
        return 0.0
    return float(mean(getter(stats) for stats in history))


def aggregate_episode_stats(history: List[EpisodeStats]) -> Dict[str, object]:
    """
    Aggregate suite-level statistics from a sequence of EpisodeStats into a dictionary of numeric summaries.
    
    If `history` is empty the function returns zero-valued aggregates (e.g., 0.0 for means, empty maps where appropriate) and `"episodes"` equal to 0; `episodes_detail` will be an empty list.
    
    Returns:
        dict: Mapping of aggregated metrics computed across the provided episodes. Notable keys include:
            - "episodes": total number of episodes aggregated.
            - Overall means: "mean_reward", "mean_food", "mean_sleep", survival and event means.
            - Predator metrics: "mean_predator_contacts", "mean_predator_escapes",
              "mean_predator_response_events", "mean_predator_response_latency",
              plus per-type maps ("mean_predator_contacts_by_type", "mean_predator_escapes_by_type",
              "mean_predator_response_latency_by_type").
            - Night metrics: "mean_night_shelter_occupancy_rate", "mean_night_stillness_rate",
              "mean_night_role_ticks", "mean_night_role_distribution".
            - Reward and state maps: "mean_reward_components", "mean_predator_state_ticks",
              "mean_predator_state_occupancy", "mean_predator_mode_transitions", "dominant_predator_state".
            - Reflex/module metrics: mean reflex usage/override/dominance and per-module maps
              ("mean_module_reflex_usage_rate", "mean_module_reflex_override_rate",
              "mean_module_reflex_dominance", "mean_module_contribution_share"),
              proposer/divergence and action-center differentials ("mean_proposer_divergence_by_module",
              "mean_action_center_gate_differential", "mean_action_center_contribution_differential"),
              credit/gradient summaries ("mean_module_credit_weights", "module_gradient_norm_means"),
              dominant-module statistics ("dominant_module", "dominant_module_distribution", "mean_dominant_module_share").
            - Motor/terrain metrics: "mean_motor_slip_rate", "mean_orientation_alignment",
              "mean_terrain_difficulty", "mean_terrain_slip_rates".
            - Misc: distance and sleep deltas ("mean_food_distance_delta", "mean_shelter_distance_delta"),
              "mean_sleep_debt", "mean_representation_specialization_score", effective module counts and agreement rates.
            - "episodes_detail": list of per-episode dictionaries produced by asdict(stats).
    """
    reward_component_names = (
        list(history[0].reward_component_totals.keys())
        if history
        else []
    )
    predator_states = (
        list(history[0].predator_state_ticks.keys())
        if history
        else []
    )
    reward_component_means = _mean_map(
        history, reward_component_names,
        lambda s, n: s.reward_component_totals[n],
    )
    predator_state_means = _mean_map(
        history, predator_states,
        lambda s, n: s.predator_state_ticks[n],
    )
    night_role_tick_means = _mean_map(
        history, SHELTER_ROLES,
        lambda s, n: s.night_role_ticks.get(n, 0),
    )
    night_role_distribution = _mean_map(
        history, SHELTER_ROLES,
        lambda s, n: s.night_role_distribution.get(n, 0.0),
    )
    steps_mean = mean(stats.steps for stats in history) if history else 1.0
    predator_state_rates = {
        name: (predator_state_means[name] / steps_mean if steps_mean else 0.0)
        for name in predator_state_means
    }
    module_reflex_usage_rate_means = _mean_map(
        history, REFLEX_MODULE_NAMES,
        lambda s, n: s.module_reflex_usage_rates.get(n, 0.0),
    )
    module_reflex_override_rate_means = _mean_map(
        history, REFLEX_MODULE_NAMES,
        lambda s, n: s.module_reflex_override_rates.get(n, 0.0),
    )
    module_reflex_dominance_means = _mean_map(
        history, REFLEX_MODULE_NAMES,
        lambda s, n: s.module_reflex_dominance.get(n, 0.0),
    )
    module_contribution_share_means = _mean_map(
        history, PROPOSAL_SOURCE_NAMES,
        lambda s, n: s.module_contribution_share.get(n, 0.0),
    )
    proposer_divergence_names = sorted(
        {
            *PRIMARY_REPRESENTATION_READOUT_MODULES,
            *(name for stats in history for name in stats.proposer_divergence_by_module),
        }
    )
    mean_proposer_divergence_by_module = _mean_map(
        history,
        proposer_divergence_names,
        lambda s, n: s.proposer_divergence_by_module.get(n, 0.0),
    )
    mean_action_center_gate_differential = _mean_map(
        history,
        PROPOSAL_SOURCE_NAMES,
        lambda s, n: s.action_center_gate_differential.get(n, 0.0),
    )
    mean_action_center_contribution_differential = _mean_map(
        history,
        PROPOSAL_SOURCE_NAMES,
        lambda s, n: s.action_center_contribution_differential.get(n, 0.0),
    )
    module_credit_names = list(PROPOSAL_SOURCE_NAMES) + (_extra_module_names(history, "mean_module_credit_weights") if history else [])
    module_gradient_names = list(PROPOSAL_SOURCE_NAMES) + (_extra_module_names(history, "module_gradient_norm_means") if history else [])
    counterfactual_credit_names = list(PROPOSAL_SOURCE_NAMES) + (_extra_module_names(history, "mean_counterfactual_credit_weights") if history else [])
    module_credit_weight_means = _mean_map(
        history, module_credit_names,
        lambda s, n: s.mean_module_credit_weights.get(n, 0.0),
    )
    module_gradient_norm_means = _mean_map(
        history, module_gradient_names,
        lambda s, n: s.module_gradient_norm_means.get(n, 0.0),
    )
    mean_counterfactual_credit_weights = _mean_map(
        history,
        counterfactual_credit_names,
        lambda s, n: s.mean_counterfactual_credit_weights.get(n, 0.0),
    )
    terrain_slip_names = sorted(
        {
            terrain
            for stats in history
            for terrain in stats.terrain_slip_rates
        }
    )
    terrain_slip_rate_means = _mean_map(
        history, terrain_slip_names,
        lambda s, n: s.terrain_slip_rates.get(n, 0.0),
    )
    predator_type_names = sorted(
        {
            *PREDATOR_TYPE_NAMES,
            *(
                predator_type
                for stats in history
                for predator_type in (
                    set(stats.predator_contacts_by_type)
                    | set(stats.predator_escapes_by_type)
                    | set(stats.predator_response_latency_by_type)
                    | set(stats.module_response_by_predator_type)
                )
            ),
        }
    )
    mean_predator_contacts_by_type = _mean_map(
        history, predator_type_names,
        lambda s, n: s.predator_contacts_by_type.get(n, 0),
    )
    mean_predator_escapes_by_type = _mean_map(
        history, predator_type_names,
        lambda s, n: s.predator_escapes_by_type.get(n, 0),
    )
    mean_predator_response_latency_by_type = _mean_map(
        history, predator_type_names,
        lambda s, n: s.predator_response_latency_by_type.get(n, 0.0),
    )
    mean_module_response_by_predator_type = {
        pt: _mean_map(
            history,
            PROPOSAL_SOURCE_NAMES,
            lambda s, n, _pt=pt: s.module_response_by_predator_type.get(_pt, {}).get(n, 0.0),
        )
        for pt in predator_type_names
    }
    dominant_module_counter = Counter(
        stats.dominant_module
        for stats in history
        if stats.dominant_module in PROPOSAL_SOURCE_NAMES
    )
    dominant_module_distribution = {
        name: (
            float(dominant_module_counter.get(name, 0) / len(history))
            if history
            else 0.0
        )
        for name in PROPOSAL_SOURCE_NAMES
    }
    dominant_module = (
        max(dominant_module_counter, key=dominant_module_counter.get)
        if dominant_module_counter
        else ""
    )
    dominant_predator_state = (
        max(predator_state_means, key=predator_state_means.get)
        if any(value > 0 for value in predator_state_means.values())
        else "PATROL"
    )
    ms = _mean_scalar  # shorthand for readability
    return {
        "episodes": len(history),
        "mean_reward": ms(history, lambda s: s.total_reward),
        "mean_food": ms(history, lambda s: s.food_eaten),
        "mean_sleep": ms(history, lambda s: s.sleep_events),
        "mean_predator_contacts": ms(history, lambda s: s.predator_contacts),
        "mean_predator_contacts_by_type": mean_predator_contacts_by_type,
        "mean_predator_escapes": ms(history, lambda s: s.predator_escapes),
        "mean_predator_escapes_by_type": mean_predator_escapes_by_type,
        "mean_night_shelter_occupancy_rate": ms(history, lambda s: s.night_shelter_occupancy_rate),
        "mean_night_stillness_rate": ms(history, lambda s: s.night_stillness_rate),
        "mean_night_role_ticks": night_role_tick_means,
        "mean_night_role_distribution": night_role_distribution,
        "mean_predator_response_events": ms(history, lambda s: s.predator_response_events),
        "mean_predator_response_latency": ms(history, lambda s: s.mean_predator_response_latency),
        "mean_predator_response_latency_by_type": mean_predator_response_latency_by_type,
        "mean_sleep_debt": ms(history, lambda s: s.mean_sleep_debt),
        "mean_food_distance_delta": ms(history, lambda s: s.food_distance_delta),
        "mean_shelter_distance_delta": ms(history, lambda s: s.shelter_distance_delta),
        "survival_rate": ms(history, lambda s: 1.0 if s.alive else 0.0),
        "mean_reward_components": reward_component_means,
        "mean_predator_state_ticks": predator_state_means,
        "mean_predator_state_occupancy": predator_state_rates,
        "mean_predator_mode_transitions": ms(history, lambda s: s.predator_mode_transitions),
        "dominant_predator_state": dominant_predator_state,
        "mean_reflex_usage_rate": ms(history, lambda s: s.reflex_usage_rate),
        "mean_final_reflex_override_rate": ms(history, lambda s: s.final_reflex_override_rate),
        "mean_reflex_dominance": ms(history, lambda s: s.mean_reflex_dominance),
        "mean_module_reflex_usage_rate": module_reflex_usage_rate_means,
        "mean_module_reflex_override_rate": module_reflex_override_rate_means,
        "mean_module_reflex_dominance": module_reflex_dominance_means,
        "mean_module_contribution_share": module_contribution_share_means,
        "mean_module_response_by_predator_type": mean_module_response_by_predator_type,
        "mean_proposer_divergence_by_module": mean_proposer_divergence_by_module,
        "mean_action_center_gate_differential": mean_action_center_gate_differential,
        "mean_action_center_contribution_differential": mean_action_center_contribution_differential,
        "mean_representation_specialization_score": ms(history, lambda s: s.representation_specialization_score),
        "mean_module_credit_weights": module_credit_weight_means,
        "module_gradient_norm_means": module_gradient_norm_means,
        "mean_counterfactual_credit_weights": mean_counterfactual_credit_weights,
        "mean_motor_slip_rate": ms(history, lambda s: s.motor_slip_rate),
        "mean_orientation_alignment": ms(history, lambda s: s.mean_orientation_alignment),
        "mean_terrain_difficulty": ms(history, lambda s: s.mean_terrain_difficulty),
        "mean_terrain_slip_rates": terrain_slip_rate_means,
        "dominant_module": dominant_module,
        "dominant_module_distribution": dominant_module_distribution,
        "mean_dominant_module_share": ms(history, lambda s: s.dominant_module_share),
        "mean_effective_module_count": ms(history, lambda s: s.effective_module_count),
        "mean_module_agreement_rate": ms(history, lambda s: s.module_agreement_rate),
        "mean_module_disagreement_rate": ms(history, lambda s: s.module_disagreement_rate),
        "episodes_detail": [asdict(stats) for stats in history],
    }


def build_behavior_check(spec: BehaviorCheckSpec, *, passed: bool, value: Any) -> BehaviorCheckResult:
    """
    Create a BehaviorCheckResult from a BehaviorCheckSpec and an observed outcome.
    
    Parameters:
        spec (BehaviorCheckSpec): Specification of the check (name, description, expected outcome).
        passed (bool): Whether the observed outcome satisfies the spec; will be stored as `bool`.
        value (Any): Observed value associated with the check result.
    
    Returns:
        BehaviorCheckResult: Result populated with `name`, `description`, and `expected` from `spec`,
        and with `passed` coerced to `bool` and `value` set to the provided observation.
    """
    return BehaviorCheckResult(
        name=spec.name,
        description=spec.description,
        expected=spec.expected,
        passed=bool(passed),
        value=value,
    )


def build_behavior_score(
    *,
    stats: EpisodeStats,
    objective: str,
    checks: Sequence[BehaviorCheckResult],
    behavior_metrics: Mapping[str, Any],
) -> BehavioralEpisodeScore:
    """
    Builds a BehavioralEpisodeScore for an episode from episode stats, check results, and behavior metrics.
    
    Parameters:
        stats (EpisodeStats): Source of episode identifier, seed, and scenario (used for the score's episode/seed/scenario).
        objective (str): The objective name associated with this behavior score.
        checks (Sequence[BehaviorCheckResult]): Sequence of check results; they are indexed by `name` into the score's `checks` map.
        behavior_metrics (Mapping[str, Any]): Arbitrary per-episode behavior metrics copied into the score.
    
    Returns:
        BehavioralEpisodeScore: Score populated with:
            - `episode` and `seed` from `stats`
            - `scenario` from `stats.scenario` or `"default"` if falsy
            - `objective` as provided
            - `success` set to `true` if there are no failed checks, `false` otherwise
            - `checks` as a mapping of check name → BehaviorCheckResult
            - `behavior_metrics` as a plain dict copy of the provided mapping
            - `failures` as a list of names for checks that did not pass
    """
    check_map = {
        check.name: check
        for check in checks
    }
    failures = [
        name
        for name, check in check_map.items()
        if not check.passed
    ]
    return BehavioralEpisodeScore(
        episode=stats.episode,
        seed=stats.seed,
        scenario=stats.scenario or "default",
        objective=objective,
        success=not failures,
        checks=check_map,
        behavior_metrics=dict(behavior_metrics),
        failures=failures,
    )


def aggregate_behavior_scores(
    scores: Sequence[BehavioralEpisodeScore],
    *,
    scenario: str,
    description: str,
    objective: str,
    check_specs: Sequence[BehaviorCheckSpec],
    diagnostic_focus: str | None = None,
    success_interpretation: str | None = None,
    failure_interpretation: str | None = None,
    budget_note: str | None = None,
    legacy_metrics: Mapping[str, Any] | None = None,
) -> Dict[str, object]:
    """
    Aggregate episode-level behavioral scores into a scenario-level summary dictionary.
    
    Produces a mapping with per-check pass rates and mean values, aggregated behavior metrics (numeric mean or most-common), diagnostic summaries (primary outcome, outcome distribution, optional failure-mode diagnostics, partial progress and died-without-contact aggregates), the episode success rate, sorted unique failure names, per-episode detail records, and a plain-copy of any provided legacy metrics.
    
    Parameters:
        scores (Sequence[BehavioralEpisodeScore]): Episode-level behavioral scores to aggregate.
        scenario (str): Scenario identifier for the aggregation.
        description (str): Human-readable description of the scenario.
        objective (str): Objective name associated with the aggregated scores.
        check_specs (Sequence[BehaviorCheckSpec]): Specifications for checks to include; used to supply each check's description and expected value.
        diagnostic_focus (str | None): Optional diagnostic focus label to include in the output (empty string when None).
        success_interpretation (str | None): Optional textual interpretation of success included in the output (empty string when None).
        failure_interpretation (str | None): Optional textual interpretation of failures included in the output (empty string when None).
        budget_note (str | None): Optional budget/note string to include in the output (empty string when None).
        legacy_metrics (Mapping[str, Any] | None): Optional additional metrics preserved unchanged under the `legacy_metrics` key.
    
    Returns:
        Dict[str, object]: Aggregated summary containing at least the keys:
            - "scenario", "description", "objective": echoed input metadata.
            - "diagnostic_focus", "success_interpretation", "failure_interpretation", "budget_note": optional metadata strings.
            - "episodes" (int): number of episodes aggregated.
            - "success_rate" (float): mean of per-episode success indicators (`1.0` for success, `0.0` otherwise).
            - "checks" (dict): mapping check name -> { "description", "expected", "pass_rate", "mean_value" }.
            - "behavior_metrics" (dict): aggregated metrics (numeric mean or most-common value).
            - "diagnostics" (dict): includes "primary_outcome", "outcome_distribution", optional "primary_failure_mode" and "failure_mode_distribution", and aggregated diagnostic rates for "partial_progress" and "died_without_contact" when present.
            - "failures" (list[str]): sorted unique failure names observed across episodes.
            - "episodes_detail" (list[dict]): per-episode records converted to plain dicts.
            - "legacy_metrics" (dict): plain dict copy of `legacy_metrics` or an empty dict.
    """
    score_list = list(scores)
    check_index = {
        spec.name: spec
        for spec in check_specs
    }
    aggregated_checks: Dict[str, object] = {}
    for name, spec in check_index.items():
        results = [
            score.checks[name]
            for score in score_list
            if name in score.checks
        ]
        values = [result.value for result in results]
        aggregated_checks[name] = {
            "description": spec.description,
            "expected": spec.expected,
            "pass_rate": (
                mean(1.0 if result.passed else 0.0 for result in results)
                if results
                else 0.0
            ),
            "mean_value": _mean_like(values),
        }

    metric_names = sorted(
        {
            metric_name
            for score in score_list
            for metric_name in score.behavior_metrics
        }
    )
    behavior_metrics = {
        name: _aggregate_values(
            [score.behavior_metrics[name] for score in score_list if name in score.behavior_metrics]
        )
        for name in metric_names
    }
    outcome_labels = [
        str(score.behavior_metrics["outcome_band"])
        for score in score_list
        if "outcome_band" in score.behavior_metrics
    ]
    outcome_counter = Counter(outcome_labels)
    outcome_total = max(1, len(outcome_labels))
    failure_mode_labels = [
        str(score.behavior_metrics["failure_mode"])
        for score in score_list
        if "failure_mode" in score.behavior_metrics
    ]
    failure_mode_counter = Counter(failure_mode_labels)
    failure_mode_total = max(1, len(failure_mode_labels))
    partial_progress_values = [
        score.behavior_metrics["partial_progress"]
        for score in score_list
        if "partial_progress" in score.behavior_metrics
    ]
    died_without_contact_values = [
        score.behavior_metrics["died_without_contact"]
        for score in score_list
        if "died_without_contact" in score.behavior_metrics
    ]
    diagnostics = {
        "primary_outcome": (
            outcome_counter.most_common(1)[0][0]
            if outcome_counter
            else "not_available"
        ),
        "outcome_distribution": {
            label: float(count / outcome_total)
            for label, count in sorted(outcome_counter.items())
        },
        "partial_progress_rate": (
            _aggregate_values(partial_progress_values)
            if partial_progress_values
            else None
        ),
        "died_without_contact_rate": (
            _aggregate_values(died_without_contact_values)
            if died_without_contact_values
            else None
        ),
    }
    if failure_mode_counter:
        diagnostics["primary_failure_mode"] = failure_mode_counter.most_common(1)[0][0]
        diagnostics["failure_mode_distribution"] = {
            label: float(count / failure_mode_total)
            for label, count in sorted(failure_mode_counter.items())
        }
    failures = sorted(
        {
            failure
            for score in score_list
            for failure in score.failures
        }
    )
    return {
        "scenario": scenario,
        "description": description,
        "objective": objective,
        "diagnostic_focus": diagnostic_focus or "",
        "success_interpretation": success_interpretation or "",
        "failure_interpretation": failure_interpretation or "",
        "budget_note": budget_note or "",
        "episodes": len(score_list),
        "success_rate": (
            mean(1.0 if score.success else 0.0 for score in score_list)
            if score_list
            else 0.0
        ),
        "checks": aggregated_checks,
        "behavior_metrics": behavior_metrics,
        "diagnostics": diagnostics,
        "failures": failures,
        "episodes_detail": [asdict(score) for score in score_list],
        "legacy_metrics": dict(legacy_metrics or {}),
    }


def summarize_behavior_suite(
    suite: Mapping[str, Mapping[str, Any]],
    *,
    competence_label: str = "mixed",
) -> Dict[str, object]:
    """
    Summarizes per-scenario behavior aggregation results into suite-level counts and rates.
    
    Parameters:
        suite (Mapping[str, Mapping[str, Any]]): Mapping from scenario name to its aggregated data. Each scenario mapping is expected to include numeric "episodes" and "success_rate" keys and may include a "failures" iterable.
        competence_label (str): Competence context for this evaluation; one of "self_sufficient", "scaffolded", or "mixed".
    
    Returns:
        Dict[str, object]: Summary dictionary with keys:
            - "scenario_count": number of scenarios in the suite.
            - "episode_count": total number of episodes across all scenarios.
            - "scenario_success_rate": mean of per-scenario indicators where a scenario's `success_rate` is at least 1.0 (1.0 if fully successful, 0.0 otherwise).
            - "episode_success_rate": overall fraction of episodes considered successful (episodes-weighted success rate).
            - "regressions": list of regression entries (one per scenario that reported failures), each of the form {"scenario": <name>, "failures": [<failure names>]}.
            - "competence_type": the validated competence label.
    """
    competence_type = normalize_competence_label(competence_label)
    scenario_items = list(suite.items())
    total_episodes = sum(int(data.get("episodes", 0)) for _, data in scenario_items)
    successful_episodes = sum(
        float(data.get("success_rate", 0.0)) * int(data.get("episodes", 0))
        for _, data in scenario_items
    )
    regressions = [
        {
            "scenario": name,
            "failures": list(data.get("failures", [])),
        }
        for name, data in scenario_items
        if data.get("failures")
    ]
    return {
        "scenario_count": len(scenario_items),
        "episode_count": total_episodes,
        "scenario_success_rate": (
            mean(1.0 if float(data.get("success_rate", 0.0)) >= 1.0 else 0.0 for _, data in scenario_items)
            if scenario_items
            else 0.0
        ),
        "episode_success_rate": (
            float(successful_episodes / total_episodes)
            if total_episodes
            else 0.0
        ),
        "competence_type": competence_type,
        "regressions": regressions,
    }


def flatten_behavior_rows(
    scores: Sequence[BehavioralEpisodeScore],
    *,
    reward_profile: str,
    scenario_map: str,
    simulation_seed: int,
    scenario_description: str,
    scenario_objective: str,
    scenario_focus: str,
    evaluation_map: str | None = None,
    eval_reflex_scale: float | None = None,
    competence_label: str | None = None,
) -> List[Dict[str, object]]:
    """
    Flatten BehavioralEpisodeScore records into tabular row dictionaries.
    
    Each row contains fixed metadata columns (reward_profile, scenario_map, evaluation_map,
    competence_type, is_primary_benchmark, eval_reflex_scale, simulation_seed, episode_seed,
    scenario, scenario_description, scenario_objective, scenario_focus, episode, success,
    failure_count, failures), plus one `metric_{name}` entry per behavior metric and for each
    check three columns: `check_{name}_passed`, `check_{name}_value`, and `check_{name}_expected`.
    
    Parameters:
        scores (Sequence[BehavioralEpisodeScore]): Episode-level behavioral scores to flatten.
        reward_profile (str): Identifier for the reward configuration applied to all rows.
        scenario_map (str): Map template used by the scenario.
        simulation_seed (int): Global simulation seed applied to all rows.
        scenario_description (str): Human-readable scenario description to include in each row.
        scenario_objective (str): Scenario objective string to include in each row.
        scenario_focus (str): Scenario focus/category to include in each row.
        evaluation_map (str | None): Optional outer sweep or default map context for the run.
        eval_reflex_scale (float | None): Optional evaluation-time reflex scale used to derive competence when `competence_label` is not provided.
        competence_label (str | None): Optional explicit competence label override; when omitted, competence is derived from `eval_reflex_scale` and validated.
    
    Returns:
        List[Dict[str, object]]: A list of per-episode row dictionaries ready for tabular export. Each row includes the computed `competence_type` and `is_primary_benchmark` fields.
    """
    rows: List[Dict[str, object]] = []
    competence_type = normalize_competence_label(
        competence_label
        if competence_label is not None
        else competence_label_from_eval_reflex_scale(eval_reflex_scale)
    )
    for score in scores:
        row: Dict[str, object] = {
            "reward_profile": reward_profile,
            "scenario_map": scenario_map,
            "evaluation_map": evaluation_map,
            "competence_type": competence_type,
            "is_primary_benchmark": competence_type == "self_sufficient",
            "eval_reflex_scale": eval_reflex_scale,
            "simulation_seed": simulation_seed,
            "episode_seed": score.seed,
            "scenario": score.scenario,
            "scenario_description": scenario_description,
            "scenario_objective": scenario_objective,
            "scenario_focus": scenario_focus,
            "episode": score.episode,
            "success": bool(score.success),
            "failure_count": len(score.failures),
            "failures": ",".join(score.failures),
        }
        for metric_name, value in sorted(score.behavior_metrics.items()):
            row[f"metric_{metric_name}"] = value
        for check_name, result in sorted(score.checks.items()):
            row[f"check_{check_name}_passed"] = bool(result.passed)
            row[f"check_{check_name}_value"] = result.value
            row[f"check_{check_name}_expected"] = result.expected
        rows.append(row)
    return rows


def _aggregate_values(values: Sequence[Any]) -> Any:
    """
    Aggregate a sequence of values into a single representative value.
    
    If `values` is empty returns 0.0. If all elements are numeric-like (ints, floats, or bools)
    returns their arithmetic mean as a float. Otherwise returns the most common string
    representation among the elements.
    
    Parameters:
        values (Sequence[Any]): Values to aggregate.
    
    Returns:
        Any: `0.0` for empty input, a `float` mean when all elements are numeric-like,
        or the most common stringified value otherwise.
    """
    if not values:
        return 0.0
    mean_value = _mean_like(values)
    if mean_value is not None:
        return mean_value
    counter = Counter(str(value) for value in values)
    return counter.most_common(1)[0][0]


def _mean_like(values: Sequence[Any]) -> float | None:
    """
    Compute the arithmetic mean of the sequence when every element is numeric-like.
    
    Returns:
        float: The mean of the values when all elements are instances of int, float, or bool; `0.0` if `values` is empty.
        None: If any element is not numeric-like.
    """
    if not values:
        return 0.0
    if all(isinstance(value, (int, float, bool)) for value in values):
        return float(mean(float(value) for value in values))
    return None
