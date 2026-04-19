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

from .comparison_learning import build_learning_evidence_deltas
from .comparison_noise import with_noise_profile_metadata
from .comparison_utils import noise_profile_metadata

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
