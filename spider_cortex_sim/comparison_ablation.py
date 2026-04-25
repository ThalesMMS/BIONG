"""Cross-run comparison and condensed reporting helpers.

``condense_robustness_summary`` is the public home for the CLI helper formerly
named ``_short_robustness_matrix_summary``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

from .ablations import (
    BrainAblationConfig,
    default_brain_config,
    resolve_ablation_configs,
)
from .budget_profiles import BudgetProfile, resolve_budget
from .capacity_profiles import CapacityProfile
from .checkpointing import (
    CheckpointSelectionConfig,
    resolve_checkpoint_selection_config,
)
from .export import compact_behavior_payload
from .metrics import flatten_behavior_rows
from .noise import (
    NoiseConfig,
    resolve_noise_profile,
)
from .operational_profiles import OperationalProfile
from .scenarios import SCENARIO_NAMES, get_scenario
from .simulation import SpiderSimulation

from .comparison_behavior import build_ablation_deltas
from .comparison_noise import with_noise_profile_metadata
from .comparison_representation import build_predator_type_specialization_summary
from .comparison_utils import (
    attach_behavior_seed_statistics,
    build_checkpoint_path,
    build_run_budget_summary,
    create_comparison_simulation,
    noise_profile_metadata,
)

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
    capacity_profile: str | CapacityProfile | None = None,
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
    checkpoint_selection_config: CheckpointSelectionConfig | None = None,
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
    if seeds is not None and len(seeds) == 0:
        raise ValueError("compare_ablation_suite() requires non-empty seeds.")
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
    requested_configs = resolve_ablation_configs(
        variant_names,
        module_dropout=module_dropout,
        capacity_profile=capacity_profile,
    )
    reference_config = default_brain_config(
        module_dropout=module_dropout,
        capacity_profile=capacity_profile,
    )
    configs: List[BrainAblationConfig] = [reference_config]
    for config in requested_configs:
        if config.name != reference_config.name:
            configs.append(config)

    run_count = max(1, int(budget.scenario_episodes))
    seed_values = tuple(seeds) if seeds is not None else budget.ablation_seeds
    rows: List[Dict[str, object]] = []
    variants: Dict[str, Dict[str, object]] = {}

    for config in configs:
        summary = config.to_summary()
        combined_stats = {name: [] for name in scenario_names}
        combined_scores = {name: [] for name in scenario_names}
        combined_stats_without_reflex = {name: [] for name in scenario_names}
        combined_scores_without_reflex = {name: [] for name in scenario_names}
        seed_payloads_with_reflex: list[tuple[int, Dict[str, object]]] = []
        seed_payloads_without_reflex: list[tuple[int, Dict[str, object]]] = []
        sim: SpiderSimulation | None = None
        previous_reflex_scale = 0.0
        behavior_base_index = 300_000
        for seed in seed_values:
            sim_budget = build_run_budget_summary(
                budget,
                scenario_episodes=run_count,
                behavior_seeds=budget.behavior_seeds,
                ablation_seeds=seed_values,
            )
            sim = create_comparison_simulation(
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
                    run_checkpoint_dir = build_checkpoint_path(
                        checkpoint_dir,
                        "ablation_compare",
                        f"{config.name}__seed_{seed}",
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
            seed_with_reflex_payload["summary"]["architecture_description"] = (
                summary["architecture_description"]
            )
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
            seed_without_reflex_payload["summary"]["architecture_description"] = (
                summary["architecture_description"]
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
        with_reflex_payload["summary"]["architecture_description"] = (
            summary["architecture_description"]
        )
        with_reflex_payload["eval_reflex_scale"] = float(previous_reflex_scale)
        with_reflex_payload["competence_type"] = "scaffolded"
        with_reflex_payload["config"] = summary
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
        without_reflex_payload["summary"]["architecture_description"] = (
            summary["architecture_description"]
        )
        without_reflex_payload["eval_reflex_scale"] = 0.0
        without_reflex_payload["competence_type"] = "self_sufficient"
        without_reflex_payload["config"] = summary
        attach_behavior_seed_statistics(
            without_reflex_payload,
            seed_payloads_without_reflex,
            condition=config.name,
            scenario_names=scenario_names,
        )
        parameter_counts_by_network = sim.brain.count_parameters()
        total_trainable_parameters = int(sum(parameter_counts_by_network.values()))
        approximate_compute_cost = sim.brain.estimate_compute_cost()
        parameter_counts = {
            "per_network": parameter_counts_by_network,
            "total": total_trainable_parameters,
            "proportions": {
                name: (
                    0.0
                    if total_trainable_parameters <= 0
                    else float(count / total_trainable_parameters)
                )
                for name, count in parameter_counts_by_network.items()
            },
        }
        with_reflex_payload["parameter_counts"] = dict(parameter_counts)
        without_reflex_payload["parameter_counts"] = dict(parameter_counts)
        with_reflex_payload["approximate_compute_cost"] = dict(approximate_compute_cost)
        without_reflex_payload["approximate_compute_cost"] = dict(approximate_compute_cost)
        compact_payload = dict(without_reflex_payload)
        compact_payload["primary_evaluation"] = "without_reflex_support"
        compact_payload["with_reflex_support"] = with_reflex_payload
        compact_payload["without_reflex_support"] = without_reflex_payload
        compact_payload["parameter_counts"] = dict(parameter_counts)
        compact_payload["approximate_compute_cost"] = dict(approximate_compute_cost)
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
        "checkpoint_metric": checkpoint_selection_config.metric,
        "checkpoint_penalty_config": checkpoint_selection_config.to_summary(),
        "primary_evaluation": "without_reflex_support",
        "reference_variant": reference_variant,
        "reference_eval_reflex_scale": 0.0,
        "seeds": list(seed_values),
        "scenario_names": scenario_names,
        "episodes_per_scenario": run_count,
        "capacity_profile": reference_config.capacity_profile.to_summary(),
        "variants": variants,
        "deltas_vs_reference": deltas_vs_reference,
        "predator_type_specialization": build_predator_type_specialization_summary(
            variants,
            reference_variant=reference_variant,
            deltas_vs_reference=deltas_vs_reference,
        ),
        **noise_profile_metadata(resolved_noise_profile),
    }, rows
