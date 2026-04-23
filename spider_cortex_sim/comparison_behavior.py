"""Cross-run comparison and condensed reporting helpers.

``condense_robustness_summary`` is the public home for the CLI helper formerly
named ``_short_robustness_matrix_summary``.
"""

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
    CheckpointSelectionConfig,
    checkpoint_candidate_sort_key,
    checkpoint_preload_fingerprint,
    checkpoint_run_fingerprint,
    mean_reward_from_behavior_payload,
    resolve_checkpoint_selection_config,
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

from .comparison_noise import with_noise_profile_metadata
from .comparison_reward import austere_survival_gate_passed, build_reward_audit, episode_history_reward_payload, shaping_reduction_status
from .comparison_utils import aggregate_with_uncertainty, metric_seed_values_from_payload, noise_profile_metadata, paired_seed_delta_rows, safe_float, seed_level_dicts

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
    reference = variants.get(reference_variant)
    if not isinstance(reference, dict):
        return {}
    reference_summary = reference.get("summary", {})
    reference_suite = reference.get("suite", {})
    if not isinstance(reference_summary, dict):
        reference_summary = {}
    if not isinstance(reference_suite, dict):
        reference_suite = {}
    deltas: Dict[str, object] = {}
    for variant_name, payload in variants.items():
        if not isinstance(payload, dict):
            continue
        summary = payload.get("summary", {})
        suite = payload.get("suite", {})
        if not isinstance(summary, dict):
            summary = {}
        if not isinstance(suite, dict):
            suite = {}
        summary_deltas: Dict[str, float] = {}
        for metric_name, delta_name in (
            ("scenario_success_rate", "scenario_success_rate_delta"),
            ("episode_success_rate", "episode_success_rate_delta"),
        ):
            if metric_name not in summary or metric_name not in reference_summary:
                continue
            summary_deltas[delta_name] = round(
                safe_float(summary.get(metric_name))
                - safe_float(reference_summary.get(metric_name)),
                6,
            )
        summary_uncertainty: Dict[str, object] = {}
        summary_seed_level: list[Dict[str, object]] = []
        for metric_name, delta_name in (
            ("scenario_success_rate", "scenario_success_rate_delta"),
            ("episode_success_rate", "episode_success_rate_delta"),
        ):
            if delta_name not in summary_deltas:
                continue
            seed_rows = paired_seed_delta_rows(
                metric_seed_values_from_payload(
                    reference,
                    metric_name=metric_name,
                    fallback_value=reference_summary.get(metric_name),
                ),
                metric_seed_values_from_payload(
                    payload,
                    metric_name=metric_name,
                    fallback_value=summary.get(metric_name),
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
            reference_scenario = reference_suite.get(scenario_name, {})
            scenario_payload = suite.get(scenario_name, {})
            if not isinstance(reference_scenario, dict) or not isinstance(
                scenario_payload,
                dict,
            ):
                continue
            if (
                "success_rate" not in reference_scenario
                or "success_rate" not in scenario_payload
            ):
                continue
            delta_value = round(
                safe_float(scenario_payload.get("success_rate"))
                - safe_float(reference_scenario.get("success_rate")),
                6,
            )
            seed_rows = paired_seed_delta_rows(
                metric_seed_values_from_payload(
                    reference,
                    metric_name="scenario_success_rate",
                    scenario=scenario_name,
                    fallback_value=reference_scenario.get("success_rate"),
                ),
                metric_seed_values_from_payload(
                    payload,
                    metric_name="scenario_success_rate",
                    scenario=scenario_name,
                    fallback_value=scenario_payload.get("success_rate"),
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
        if not summary_deltas and not scenario_deltas:
            continue
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
    """
    Run a configuration comparison using the provided keyword arguments and return the assembled comparison report.
    
    Returns:
        result (Dict[str, object]): A dictionary containing the comparison report payload (resolved budget and seed fields, reward and map summaries, the matrix of aggregated evaluation stats, reward audit and shaping-reduction status, and noise-profile metadata).
    """
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
    checkpoint_selection_config: CheckpointSelectionConfig | None = None,
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
    checkpoint_selection_config = resolve_checkpoint_selection_config(
        checkpoint_selection_config
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
                        checkpoint_selection_config=checkpoint_selection_config,
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
        "checkpoint_metric": checkpoint_selection_config.metric,
        "checkpoint_penalty_config": checkpoint_selection_config.to_summary(),
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
