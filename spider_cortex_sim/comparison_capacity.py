from __future__ import annotations

from pathlib import Path
from collections import Counter
from typing import Dict, List, Sequence

from .ablations import resolve_ablation_configs
from .budget_profiles import BudgetProfile, resolve_budget
from .capacity_profiles import (
    CapacityProfile,
    canonical_capacity_profile_names,
    resolve_capacity_profile,
)
from .checkpointing import (
    CheckpointSelectionConfig,
    resolve_checkpoint_selection_config,
)
from .comparison_ablation import compare_ablation_suite
from .noise import NoiseConfig
from .operational_profiles import OperationalProfile
from .scenarios import SCENARIO_NAMES


CAPACITY_SWEEP_VARIANT_NAMES: tuple[str, ...] = (
    "true_monolithic_policy",
    "monolithic_policy",
    "three_center_modular",
    "four_center_modular",
    "modular_full",
)


def compare_capacity_sweep(
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
    architecture_variants: Sequence[str] | None = None,
    capacity_profiles: Sequence[str] | None = None,
    episodes_per_scenario: int | None = None,
    checkpoint_selection: str = "none",
    checkpoint_selection_config: CheckpointSelectionConfig | None = None,
    checkpoint_interval: int | None = None,
    checkpoint_dir: str | Path | None = None,
) -> tuple[Dict[str, object], List[Dict[str, object]]]:
    if seeds is not None and len(seeds) == 0:
        raise ValueError("compare_capacity_sweep() requires non-empty seeds.")
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
    scenario_names = list(names) if names is not None else list(SCENARIO_NAMES)
    variant_names = (
        list(architecture_variants)
        if architecture_variants is not None
        else list(CAPACITY_SWEEP_VARIANT_NAMES)
    )
    if variant_names:
        resolved_variant_configs = resolve_ablation_configs(
            variant_names,
            module_dropout=module_dropout,
        )
        duplicate_variant_names = sorted(
            name
            for name, count in Counter(
                config.name for config in resolved_variant_configs
            ).items()
            if count > 1
        )
        if duplicate_variant_names:
            raise ValueError(
                "compare_capacity_sweep() requires unique architecture variants; "
                f"duplicates would overwrite payloads in compare_ablation_suite(): "
                f"{duplicate_variant_names}."
            )
        variant_names = [config.name for config in resolved_variant_configs]
    profile_names = (
        list(capacity_profiles)
        if capacity_profiles is not None
        else list(canonical_capacity_profile_names())
    )
    resolved_profiles: list[CapacityProfile] = []
    seen_profile_names: set[str] = set()
    for profile_name in profile_names:
        resolved_profile = resolve_capacity_profile(profile_name)
        if resolved_profile.name in seen_profile_names:
            raise ValueError(
                "compare_capacity_sweep() requires unique capacity profiles; "
                f"duplicate resolved profile name: {resolved_profile.name!r}."
            )
        seen_profile_names.add(resolved_profile.name)
        resolved_profiles.append(resolved_profile)

    rows: List[Dict[str, object]] = []
    profile_payloads: Dict[str, Dict[str, object]] = {}
    performance_matrices: Dict[str, Dict[str, Dict[str, float | int | None]]] = {
        "scenario_success_rate": {},
        "episode_success_rate": {},
        "capability_probe_success_rate": {},
        "total_trainable": {},
        "approximate_compute_cost_total": {},
    }
    if not scenario_names:
        raise ValueError("No scenario names resolved; expected at least one scenario")
    if not variant_names or not resolved_profiles:
        raise ValueError(
            "No architecture variants or capacity profiles resolved; expected at least one of each"
        )

    for capacity_profile in resolved_profiles:
        run_checkpoint_dir = (
            Path(checkpoint_dir) / f"capacity_{capacity_profile.name}"
            if checkpoint_dir is not None
            else None
        )
        payload, profile_rows = compare_ablation_suite(
            width=width,
            height=height,
            food_count=food_count,
            day_length=day_length,
            night_length=night_length,
            max_steps=budget.max_steps,
            episodes=budget.episodes,
            evaluation_episodes=budget.eval_episodes,
            gamma=gamma,
            module_lr=module_lr,
            motor_lr=motor_lr,
            module_dropout=module_dropout,
            capacity_profile=capacity_profile,
            reward_profile=reward_profile,
            map_template=map_template,
            operational_profile=operational_profile,
            noise_profile=noise_profile,
            budget_profile=budget_profile,
            seeds=tuple(seeds) if seeds is not None else tuple(budget.ablation_seeds),
            names=scenario_names,
            variant_names=variant_names,
            episodes_per_scenario=budget.scenario_episodes,
            checkpoint_selection=checkpoint_selection,
            checkpoint_selection_config=checkpoint_selection_config,
            checkpoint_interval=budget.checkpoint_interval,
            checkpoint_dir=run_checkpoint_dir,
        )
        payload["capacity_profile"] = capacity_profile.to_summary()
        profile_payloads[capacity_profile.name] = payload
        rows.extend(profile_rows)
        for variant_name, variant_payload in payload["variants"].items():
            summary = variant_payload.get("summary", {})
            if not isinstance(summary, dict):
                summary = {}
            parameter_counts = variant_payload.get("parameter_counts", {})
            if not isinstance(parameter_counts, dict):
                parameter_counts = {}
            compute_cost = variant_payload.get("approximate_compute_cost", {})
            if not isinstance(compute_cost, dict):
                compute_cost = {}
            performance_matrices["scenario_success_rate"].setdefault(
                str(variant_name),
                {},
            )[capacity_profile.name] = float(
                summary.get("scenario_success_rate", 0.0)
            )
            performance_matrices["episode_success_rate"].setdefault(
                str(variant_name),
                {},
            )[capacity_profile.name] = float(
                summary.get("episode_success_rate", 0.0)
            )
            probe_success_rate = None
            suite = variant_payload.get("suite", {})
            if isinstance(suite, dict):
                probe_rates = [
                    float(item.get("success_rate", 0.0))
                    for item in suite.values()
                    if isinstance(item, dict) and bool(item.get("is_capability_probe"))
                ]
                if probe_rates:
                    probe_success_rate = sum(probe_rates) / len(probe_rates)
            performance_matrices["capability_probe_success_rate"].setdefault(
                str(variant_name),
                {},
            )[capacity_profile.name] = probe_success_rate
            performance_matrices["total_trainable"].setdefault(
                str(variant_name),
                {},
            )[capacity_profile.name] = int(parameter_counts.get("total", 0))
            performance_matrices["approximate_compute_cost_total"].setdefault(
                str(variant_name),
                {},
            )[capacity_profile.name] = int(compute_cost.get("total", 0))

    return {
        "budget_profile": budget.profile,
        "benchmark_strength": budget.benchmark_strength,
        "checkpoint_selection": checkpoint_selection,
        "checkpoint_metric": checkpoint_selection_config.metric,
        "checkpoint_penalty_config": checkpoint_selection_config.to_summary(),
        "capacity_profiles": [profile.to_summary() for profile in resolved_profiles],
        "architecture_variants": variant_names,
        "scenario_names": scenario_names,
        "episodes_per_scenario": max(1, int(budget.scenario_episodes)),
        "seeds": list(seeds) if seeds is not None else list(budget.ablation_seeds),
        "performance_matrices": performance_matrices,
        "profiles": profile_payloads,
    }, rows


def compare_capacity_sweeps(**kwargs) -> tuple[Dict[str, object], List[Dict[str, object]]]:
    return compare_capacity_sweep(**kwargs)


__all__ = [
    "CAPACITY_SWEEP_VARIANT_NAMES",
    "compare_capacity_sweep",
    "compare_capacity_sweeps",
]
