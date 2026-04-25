"""Cross-run comparison and condensed reporting helpers.

``condense_robustness_summary`` is the public home for the CLI helper formerly
named ``_short_robustness_matrix_summary``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

from .benchmark_types import SeedLevelResult
from .budget_profiles import BudgetProfile, resolve_budget
from .checkpointing import (
    CheckpointSelectionConfig,
    checkpoint_candidate_sort_key,
    checkpoint_preload_fingerprint,
    checkpoint_run_fingerprint,
    resolve_checkpoint_selection_config,
    resolve_checkpoint_load_dir,
)
from .export import compact_behavior_payload
from .metrics import flatten_behavior_rows
from .noise import (
    NoiseConfig,
    RobustnessMatrixSpec,
    canonical_robustness_matrix,
    resolve_noise_profile,
)
from .operational_profiles import OperationalProfile
from .scenarios import SCENARIO_NAMES, get_scenario
from .simulation import SpiderSimulation

from .comparison_utils import (
    aggregate_with_uncertainty,
    attach_behavior_seed_statistics,
    build_run_budget_summary,
    condition_compact_summary,
    condition_mean_reward,
    create_comparison_simulation,
    metric_seed_values_from_payload,
    noise_profile_metadata,
    safe_float,
    seed_level_dicts,
)

def with_noise_profile_metadata(
    payload: Dict[str, object],
    noise_profile: NoiseConfig,
) -> Dict[str, object]:
    """
    Attach noise-profile metadata to a copy of the given payload.
    
    Parameters:
        payload: Mapping to be enriched with noise-profile metadata.
        noise_profile: Resolved noise profile whose metadata will be merged into the payload.
    
    Returns:
        A shallow copy of `payload` enriched with metadata extracted from `noise_profile`.
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

def condense_robustness_summary(
    robustness_payload: object,
) -> dict[str, object]:
    """
    Condense a robustness-matrix payload into a compact, display-friendly summary.
    
    Accepts an arbitrary payload and safely extracts a matrix specification, train/eval condition counts,
    aggregate robustness scores, and marginals. If the input is not a dict or expected fields are missing,
    defaults to empty mappings/lists so the returned summary remains serializable.
    
    Parameters:
        robustness_payload: The robustness-matrix payload to condense (may be any object).
    
    Returns:
        A dict containing:
          - "matrix_dimensions": string formatted as "<num_train>x<num_eval>".
          - "matrix_spec": the extracted matrix specification dict (empty dict if absent).
          - "robustness_score": overall robustness score from the payload (or None if absent).
          - "diagonal_score": mean score on the matrix diagonal (or None if absent).
          - "off_diagonal_score": mean score off the matrix diagonal (or None if absent).
          - "train_marginals": per-train-condition marginals from the payload (or None if absent).
          - "eval_marginals": per-eval-condition marginals from the payload (or None if absent).
    """
    payload = robustness_payload if isinstance(robustness_payload, dict) else {}
    matrix_spec = payload.get("matrix_spec")
    matrix_spec_dict = matrix_spec if isinstance(matrix_spec, dict) else {}
    train_conditions = matrix_spec_dict.get("train_conditions")
    eval_conditions = matrix_spec_dict.get("eval_conditions")
    train_names = list(train_conditions) if isinstance(train_conditions, list) else []
    eval_names = list(eval_conditions) if isinstance(eval_conditions, list) else []
    return {
        "matrix_dimensions": f"{len(train_names)}x{len(eval_names)}",
        "matrix_spec": matrix_spec_dict,
        "robustness_score": payload.get("robustness_score"),
        "diagonal_score": payload.get("diagonal_score"),
        "off_diagonal_score": payload.get("off_diagonal_score"),
        "train_marginals": payload.get("train_marginals"),
        "eval_marginals": payload.get("eval_marginals"),
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
        """
        Create per-seed SeedLevelResult rows by averaging each seed's values for noise robustness.
        
        Parameters:
            values_by_seed (Dict[int, List[float]]): Mapping from seed id to a list of numeric values to average.
            metric_name (str): Metric label to assign to each SeedLevelResult.
            fallback_value (float): Value to use when no seed has any values.
        
        Returns:
            list[SeedLevelResult]: One row per seed with the mean of that seed's values rounded to 6 decimals and condition "noise_robustness".
            If no seed has values, returns a single row with `seed=0` and `value=fallback_value`.
        """
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

def _train_brain_for_noise(
    *,
    width: int,
    height: int,
    food_count: int,
    day_length: int,
    night_length: int,
    gamma: float,
    module_lr: float,
    motor_lr: float,
    module_dropout: float,
    operational_profile: str | OperationalProfile | None,
    reward_profile: str,
    map_template: str,
    budget: BudgetProfile,
    run_count: int,
    seed_values: Sequence[int],
    seed: int,
    scenario_names: Sequence[str],
    train_condition: str,
    resolved_train_noise_profile: NoiseConfig,
    checkpoint_selection: str,
    checkpoint_selection_config: CheckpointSelectionConfig,
    checkpoint_dir: str | Path | None,
    load_brain: str | Path | None,
    load_modules: Sequence[str] | None,
    save_brain: str | Path | None,
) -> tuple[SpiderSimulation, Dict[str, object], float]:
    sim_budget = build_run_budget_summary(
        budget,
        scenario_episodes=run_count,
        behavior_seeds=seed_values,
        ablation_seeds=seed_values,
    )
    fingerprint_budget_resolved = {
        "episodes": budget.episodes,
        "eval_episodes": budget.eval_episodes,
        "max_steps": budget.max_steps,
        "scenario_episodes": run_count,
        "checkpoint_interval": budget.checkpoint_interval,
        "selection_scenario_episodes": budget.selection_scenario_episodes,
    }
    preload_fingerprint = checkpoint_preload_fingerprint(load_brain, load_modules)
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
            "checkpoint_metric": checkpoint_selection_config.metric,
            "checkpoint_penalty_config": checkpoint_selection_config.to_summary(),
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
        should_train = checkpoint_selection == "best" or budget.episodes > 0
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
                checkpoint_selection_config=checkpoint_selection_config,
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
    return sim, dict(brain_config_summary), float(sim._effective_reflex_scale())

def _evaluate_robustness_matrix(
    *,
    sim: SpiderSimulation,
    robustness_matrix: RobustnessMatrixSpec,
    train_index: int,
    resolved_train_noise_profile: NoiseConfig,
    scenario_names: Sequence[str],
    run_count: int,
    seed: int,
    reward_profile: str,
    map_template: str,
    combined_stats_by_eval: Dict[str, Dict[str, list]],
    combined_scores_by_eval: Dict[str, Dict[str, list]],
    seed_payloads_by_eval: dict[str, list[tuple[int, Dict[str, object]]]],
    rows: List[Dict[str, object]],
) -> None:
    episodes_per_cell = max(1, run_count * max(1, len(scenario_names)))
    eval_stride = episodes_per_cell
    train_stride = eval_stride * max(1, len(robustness_matrix.eval_conditions))
    for eval_index, eval_condition in enumerate(robustness_matrix.eval_conditions):
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
    checkpoint_selection_config: CheckpointSelectionConfig | None = None,
    checkpoint_interval: int | None = None,
    checkpoint_dir: str | Path | None = None,
    load_brain: str | Path | None = None,
    load_modules: Sequence[str] | None = None,
    save_brain: str | Path | None = None,
) -> tuple[Dict[str, object], List[Dict[str, object]]]:
    """
    Orchestrate training per train-noise condition and evaluate each trained agent across all eval-noise conditions to produce a nested robustness matrix and flattened per-episode rows.
    
    Per train condition this function optionally loads or trains a SpiderSimulation for each seed, evaluates it across every eval-noise condition, builds compact per-cell behavior payloads enriched with noise-profile metadata and seed statistics, and computes aggregate robustness metrics (marginals, overall/diagonal/off-diagonal scores, seed-level rows, and uncertainty). If `robustness_matrix` is omitted, a canonical matrix is used. Checkpoint behavior is controlled by `checkpoint_selection`, `checkpoint_selection_config`, `checkpoint_dir`, and related checkpoint parameters.
    
    Parameters:
        (See function signature for parameters; each controls world configuration, learning hyperparameters, budget/seed selection, checkpointing, and I/O for loading/saving brains.)
    
    Returns:
        A tuple (payload, rows) where:
        - payload (Dict[str, object]): top-level result containing budget/checkpoint/reward/map metadata, `seeds`, `scenario_names`, `episodes_per_scenario`, the nested `"matrix"` mapping train_condition -> eval_condition -> compact cell payload, aggregate robustness metrics, and `"matrix_spec"`.
        - rows (List[Dict[str, object]]): flattened per-episode behavior export rows annotated with train-noise metadata.
    
    Raises:
        ValueError: if `checkpoint_selection` is not "none" or "best", or if no seed values are available after budget resolution.
    """
    if robustness_matrix is None:
        robustness_matrix = canonical_robustness_matrix()
    if checkpoint_selection not in {"none", "best"}:
        raise ValueError(
            "Invalid checkpoint_selection. Use 'none' or 'best'."
        )
    checkpoint_selection_config = resolve_checkpoint_selection_config(
        checkpoint_selection_config
    )
    if checkpoint_selection == "best":
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
        exemplar_sim: SpiderSimulation | None = None

        for seed in seed_values:
            sim, train_brain_config_summary, train_eval_reflex_scale = (
                _train_brain_for_noise(
                    width=width,
                    height=height,
                    food_count=food_count,
                    day_length=day_length,
                    night_length=night_length,
                    gamma=gamma,
                    module_lr=module_lr,
                    motor_lr=motor_lr,
                    module_dropout=module_dropout,
                    operational_profile=operational_profile,
                    reward_profile=reward_profile,
                    map_template=map_template,
                    budget=budget,
                    run_count=run_count,
                    seed_values=seed_values,
                    seed=seed,
                    scenario_names=scenario_names,
                    train_condition=train_condition,
                    resolved_train_noise_profile=resolved_train_noise_profile,
                    checkpoint_selection=checkpoint_selection,
                    checkpoint_selection_config=checkpoint_selection_config,
                    checkpoint_dir=checkpoint_dir,
                    load_brain=load_brain,
                    load_modules=load_modules,
                    save_brain=save_brain,
                )
            )
            exemplar_sim = sim
            _evaluate_robustness_matrix(
                sim=sim,
                robustness_matrix=robustness_matrix,
                train_index=train_index,
                resolved_train_noise_profile=resolved_train_noise_profile,
                scenario_names=scenario_names,
                run_count=run_count,
                seed=seed,
                reward_profile=reward_profile,
                map_template=map_template,
                combined_stats_by_eval=combined_stats_by_eval,
                combined_scores_by_eval=combined_scores_by_eval,
                seed_payloads_by_eval=seed_payloads_by_eval,
                rows=rows,
            )

        if exemplar_sim is None:
            raise ValueError(
                "compare_noise_robustness() requires at least one seed."
            )
        for eval_condition in robustness_matrix.eval_conditions:
            resolved_eval_noise_profile = resolve_noise_profile(eval_condition)
            cell_payload = with_noise_profile_metadata(
                compact_behavior_payload(
                    exemplar_sim._build_behavior_payload(
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
        "checkpoint_metric": checkpoint_selection_config.metric,
        "checkpoint_penalty_config": checkpoint_selection_config.to_summary(),
        "reward_profile": reward_profile,
        "map_template": map_template,
        "seeds": list(seed_values),
        "scenario_names": scenario_names,
        "episodes_per_scenario": run_count,
        "matrix": matrix_payloads,
        **aggregate_metrics,
        **robustness_matrix_metadata(robustness_matrix),
    }, rows
