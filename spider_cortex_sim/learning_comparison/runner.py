from __future__ import annotations

from .common import (
    BrainAblationConfig,
    BudgetProfile,
    CheckpointSelectionConfig,
    Dict,
    DistillationConfig,
    List,
    NoiseConfig,
    OperationalProfile,
    Path,
    SCENARIO_NAMES,
    Sequence,
    SpiderSimulation,
    TrainingRegimeSpec,
    attach_behavior_seed_statistics,
    build_checkpoint_path,
    build_run_budget_summary,
    compact_behavior_payload,
    competence_label_from_eval_reflex_scale,
    create_comparison_simulation,
    default_brain_config,
    flatten_behavior_rows,
    get_scenario,
    noise_profile_metadata,
    replace,
    resolve_budget,
    resolve_checkpoint_selection_config,
    resolve_learning_evidence_conditions,
    resolve_noise_profile,
    resolve_training_regime,
)
from .evidence import build_learning_evidence_deltas, build_learning_evidence_summary
from ..learning_evidence import LearningEvidenceConditionSpec


def _build_skipped_condition(
    *,
    condition: LearningEvidenceConditionSpec,
    policy_mode: str,
    training_regime_name: str,
    resolved_training_regime_spec: TrainingRegimeSpec | None,
    budget: BudgetProfile,
    base_config: BrainAblationConfig,
    resolved_noise_profile: NoiseConfig,
    reason: str,
) -> Dict[str, object]:
    return {
        "condition": condition.name,
        "description": condition.description,
        "policy_mode": policy_mode,
        "train_budget": condition.train_budget,
        "training_regime": training_regime_name,
        "checkpoint_source": condition.checkpoint_source,
        "budget_profile": budget.profile,
        "benchmark_strength": budget.benchmark_strength,
        "config": base_config.to_summary(),
        "training_regime_summary": (
            None
            if resolved_training_regime_spec is None
            else resolved_training_regime_spec.to_summary()
        ),
        "skipped": True,
        "reason": reason,
        **noise_profile_metadata(resolved_noise_profile),
    }


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
    checkpoint_selection_config: CheckpointSelectionConfig | None = None,
    checkpoint_interval: int | None = None,
    checkpoint_dir: str | Path | None = None,
    distillation_config: DistillationConfig | None = None,
    distillation_training_regime: TrainingRegimeSpec | None = None,
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
    if seeds is not None and len(seeds) == 0:
        raise ValueError(
            "compare_learning_evidence() requires at least one behavior seed."
        )
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
    checkpoint_selection_config = resolve_checkpoint_selection_config(
        checkpoint_selection_config
    )
    scenario_names = list(names) if names is not None else list(SCENARIO_NAMES)
    if names is not None and not scenario_names:
        raise ValueError(
            "compare_learning_evidence() requires at least one scenario name."
        )
    condition_specs = resolve_learning_evidence_conditions(condition_names)
    primary_reference_condition = "trained_without_reflex_support"
    if condition_names is not None and not any(
        spec.name == primary_reference_condition for spec in condition_specs
    ):
        condition_specs = resolve_learning_evidence_conditions(
            (primary_reference_condition, *tuple(condition_names))
        )
    seed_values = tuple(seeds) if seeds is not None else base_budget.behavior_seeds
    if not seed_values:
        raise ValueError(
            "compare_learning_evidence() requires at least one behavior seed."
        )
    run_count = max(1, int(base_budget.scenario_episodes))
    behavior_suite_span = max(1, len(scenario_names) * run_count)
    seed_index_stride = behavior_suite_span
    condition_index_stride = seed_index_stride * len(seed_values)
    resolved_training_totals: list[int] = []
    for condition in condition_specs:
        budget = long_budget if condition.train_budget == "long" else base_budget
        training_episodes = (
            max(0, int(base_budget.episodes) // 2)
            if condition.train_budget == "freeze_half"
            else max(0, int(budget.episodes))
        )
        regime = condition.training_regime
        regime_spec = (
            regime
            if isinstance(regime, TrainingRegimeSpec)
            else resolve_training_regime(str(regime)) if regime is not None else None
        )
        if (
            regime_spec is not None
            and bool(regime_spec.distillation_enabled)
            and distillation_training_regime is not None
        ):
            regime_spec = replace(
                distillation_training_regime,
                finetuning_episodes=regime_spec.finetuning_episodes,
                finetuning_reflex_scale=regime_spec.finetuning_reflex_scale,
            )
        resolved_training_episodes = training_episodes + (
            max(0, int(regime_spec.finetuning_episodes))
            if regime_spec is not None
            else 0
        )
        if condition.train_budget == "freeze_half":
            resolved_training_episodes = max(
                resolved_training_episodes,
                max(0, int(base_budget.episodes)),
            )
        resolved_training_totals.append(resolved_training_episodes)
    training_index_floor = max(resolved_training_totals, default=0) + 1
    base_index_offset = max(
        condition_index_stride * len(condition_specs),
        training_index_floor,
    )
    base_config = (
        brain_config
        if brain_config is not None
        else default_brain_config(module_dropout=module_dropout)
    )

    rows: List[Dict[str, object]] = []
    conditions: Dict[str, Dict[str, object]] = {}

    for condition_index, condition in enumerate(condition_specs):
        policy_mode = condition.policy_mode.replace("-", "_")
        training_regime = condition.training_regime
        training_regime_name = ""
        resolved_training_regime: str | TrainingRegimeSpec | None = None
        resolved_training_regime_spec: TrainingRegimeSpec | None = None
        if isinstance(training_regime, TrainingRegimeSpec):
            resolved_training_regime = training_regime
            resolved_training_regime_spec = training_regime
            training_regime_name = training_regime.name
        elif training_regime is not None:
            training_regime_name = str(training_regime)
            resolved_training_regime = training_regime_name
            resolved_training_regime_spec = resolve_training_regime(
                training_regime_name
            )
        if (
            resolved_training_regime_spec is not None
            and bool(resolved_training_regime_spec.distillation_enabled)
            and distillation_training_regime is not None
        ):
            resolved_training_regime_spec = replace(
                distillation_training_regime,
                finetuning_episodes=resolved_training_regime_spec.finetuning_episodes,
                finetuning_reflex_scale=resolved_training_regime_spec.finetuning_reflex_scale,
            )
            resolved_training_regime = resolved_training_regime_spec
            training_regime_name = resolved_training_regime_spec.name
        condition_budget = long_budget if condition.train_budget == "long" else base_budget
        if base_config.architecture not in condition.supports_architectures:
            conditions[condition.name] = _build_skipped_condition(
                condition=condition,
                policy_mode=policy_mode,
                training_regime_name=training_regime_name,
                resolved_training_regime_spec=resolved_training_regime_spec,
                budget=condition_budget,
                base_config=base_config,
                resolved_noise_profile=resolved_noise_profile,
                reason=(
                    f"Condition incompatible with architecture {base_config.architecture!r}."
                ),
            )
            continue
        if (
            resolved_training_regime_spec is not None
            and bool(resolved_training_regime_spec.distillation_enabled)
            and distillation_config is None
        ):
            conditions[condition.name] = _build_skipped_condition(
                condition=condition,
                policy_mode=policy_mode,
                training_regime_name=training_regime_name,
                resolved_training_regime_spec=resolved_training_regime_spec,
                budget=condition_budget,
                base_config=base_config,
                resolved_noise_profile=resolved_noise_profile,
                reason=(
                    f"Condition {condition.name!r} requires a distillation teacher checkpoint."
                ),
            )
            continue
        if (
            policy_mode == "reflex_only"
            and not base_config.enable_reflexes
        ):
            conditions[condition.name] = _build_skipped_condition(
                condition=condition,
                policy_mode=policy_mode,
                training_regime_name=training_regime_name,
                resolved_training_regime_spec=resolved_training_regime_spec,
                budget=condition_budget,
                base_config=base_config,
                resolved_noise_profile=resolved_noise_profile,
                reason=(
                    f"Condition {condition.name!r} requires reflexes to be enabled, "
                    "but the base configuration has reflexes disabled."
                ),
            )
            continue

        combined_stats = {name: [] for name in scenario_names}
        combined_scores = {name: [] for name in scenario_names}
        exemplar_sim: SpiderSimulation | None = None
        seed_payloads: list[tuple[int, Dict[str, object]]] = []
        train_episodes = 0
        frozen_after_episode: int | None = None
        observed_checkpoint_source = condition.checkpoint_source
        observed_eval_reflex_scale: float | None = None

        for seed_index, seed in enumerate(seed_values):
            behavior_base_index = (
                base_index_offset
                + condition_index * condition_index_stride
                + seed_index * seed_index_stride
            )
            sim_budget = build_run_budget_summary(
                condition_budget,
                scenario_episodes=run_count,
                behavior_seeds=seed_values,
                ablation_seeds=seed_values,
            )
            sim = create_comparison_simulation(
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

            def run_training(
                *,
                sim: SpiderSimulation,
                condition: LearningEvidenceConditionSpec,
                seed: int,
                training_regime: str | TrainingRegimeSpec | None,
                budget_episodes: int,
                run_budget: BudgetProfile,
            ) -> int:
                run_checkpoint_dir = None
                if checkpoint_dir is not None:
                    run_checkpoint_dir = build_checkpoint_path(
                        checkpoint_dir,
                        "learning_evidence",
                        f"{condition.name}__seed_{seed}",
                    )
                sim.train(
                    budget_episodes,
                    evaluation_episodes=0,
                    render_last_evaluation=False,
                    capture_evaluation_trace=False,
                    debug_trace=False,
                    checkpoint_selection=checkpoint_selection,
                    checkpoint_selection_config=checkpoint_selection_config,
                    checkpoint_interval=run_budget.checkpoint_interval,
                    checkpoint_dir=run_checkpoint_dir,
                    checkpoint_scenario_names=scenario_names,
                    selection_scenario_episodes=run_budget.selection_scenario_episodes,
                    training_regime=training_regime,
                    teacher_checkpoint=(
                        None
                        if distillation_config is None
                        else distillation_config.teacher_checkpoint
                    ),
                    distillation_config=distillation_config,
                )
                return int(
                    sim._latest_training_regime_summary.get(
                        "resolved_budget",
                        {},
                    ).get(
                        "total_training_episodes",
                        budget_episodes,
                    )
                )

            if condition.train_budget == "base":
                train_episodes = run_training(
                    sim=sim,
                    condition=condition,
                    seed=seed,
                    training_regime=resolved_training_regime,
                    budget_episodes=condition_budget.episodes,
                    run_budget=condition_budget,
                )
                observed_checkpoint_source = str(sim.checkpoint_source)
            elif condition.train_budget == "long":
                train_episodes = run_training(
                    sim=sim,
                    condition=condition,
                    seed=seed,
                    training_regime=resolved_training_regime,
                    budget_episodes=long_budget.episodes,
                    run_budget=long_budget,
                )
                observed_checkpoint_source = str(sim.checkpoint_source)
            elif condition.train_budget == "freeze_half":
                train_episodes = max(0, int(base_budget.episodes) // 2)
                train_episodes = run_training(
                    sim=sim,
                    condition=condition,
                    seed=seed,
                    training_regime=resolved_training_regime,
                    budget_episodes=train_episodes,
                    run_budget=base_budget,
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
                    base_index=behavior_base_index,
                    policy_mode=policy_mode,
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
                "learning_evidence_policy_mode": policy_mode,
                "learning_evidence_training_regime": training_regime_name,
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
                "policy_mode": policy_mode,
                "train_budget": condition.train_budget,
                "training_regime": training_regime_name,
                "training_regime_summary": (
                    None
                    if resolved_training_regime_spec is None
                    else resolved_training_regime_spec.to_summary()
                ),
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
        "checkpoint_metric": checkpoint_selection_config.metric,
        "checkpoint_penalty_config": checkpoint_selection_config.to_summary(),
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

__all__ = ["compare_learning_evidence"]
