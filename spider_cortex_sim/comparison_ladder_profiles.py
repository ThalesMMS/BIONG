"""Cross-profile architectural ladder comparison helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Mapping, Sequence

from .ablations import MONOLITHIC_POLICY_NAME, TRUE_MONOLITHIC_POLICY_NAME
from .benchmark_types import SeedLevelResult
from .budget_profiles import BudgetProfile, resolve_budget
from .checkpointing import (
    CheckpointSelectionConfig,
    resolve_checkpoint_selection_config,
)
from .comparison_ablation import compare_ablation_suite
from .comparison_utils import (
    aggregate_with_uncertainty,
    metric_seed_values_from_payload,
    paired_seed_delta_rows,
    paired_seed_effect_size_rows,
    safe_float,
    seed_level_dicts,
    values_only,
)
from .noise import NoiseConfig
from .operational_profiles import OperationalProfile
from .reward import (
    MINIMAL_SHAPING_SURVIVAL_THRESHOLD,
    REWARD_PROFILES,
    SHAPING_GAP_POLICY,
)
from .statistics import cohens_d

LADDER_VARIANT_NAMES: tuple[str, ...] = (
    TRUE_MONOLITHIC_POLICY_NAME,
    MONOLITHIC_POLICY_NAME,
    "three_center_modular",
    "four_center_modular",
    "modular_full",
)
LADDER_RUNG_BY_VARIANT: dict[str, str] = {
    TRUE_MONOLITHIC_POLICY_NAME: "A0",
    MONOLITHIC_POLICY_NAME: "A1",
    "three_center_modular": "A2",
    "four_center_modular": "A3",
    "modular_full": "A4",
}
LADDER_PROTOCOL_NAMES: dict[str, str] = {
    "A0": "A0_true_monolithic",
    "A1": "A1_monolithic_with_action_motor",
    "A2": "A2_three_center",
    "A3": "A3_four_center",
    "A4": "A4_current_full_modular",
}
_DELTA_METRICS: tuple[str, ...] = (
    "scenario_success_rate",
    "episode_success_rate",
    "mean_reward",
)


def _metric_summary(payload: Mapping[str, object] | None) -> dict[str, float]:
    summary = payload.get("summary", {}) if isinstance(payload, Mapping) else {}
    if not isinstance(summary, Mapping):
        summary = {}
    return {
        "scenario_success_rate": round(
            safe_float(summary.get("scenario_success_rate")),
            6,
        ),
        "episode_success_rate": round(
            safe_float(summary.get("episode_success_rate")),
            6,
        ),
        "mean_reward": round(safe_float(summary.get("mean_reward")), 6),
    }


def _metric_uncertainty(
    payload: Mapping[str, object] | None,
    metric_name: str,
) -> dict[str, object]:
    if not isinstance(payload, Mapping):
        return aggregate_with_uncertainty(())
    uncertainty = payload.get("uncertainty", {})
    if not isinstance(uncertainty, Mapping):
        return aggregate_with_uncertainty(())
    metric_uncertainty = uncertainty.get(metric_name, {})
    return (
        dict(metric_uncertainty)
        if isinstance(metric_uncertainty, Mapping)
        else aggregate_with_uncertainty(())
    )


def _competence_gap_from_variant_payload(
    variant_name: str,
    payload: Mapping[str, object] | None,
) -> dict[str, object]:
    without_reflex = (
        payload.get("without_reflex_support", {})
        if isinstance(payload, Mapping)
        else {}
    )
    with_reflex = (
        payload.get("with_reflex_support", {})
        if isinstance(payload, Mapping)
        else {}
    )
    without_summary = _metric_summary(
        without_reflex if isinstance(without_reflex, Mapping) else payload
    )
    with_summary = _metric_summary(
        with_reflex if isinstance(with_reflex, Mapping) else payload
    )
    summary = {"basis": "scaffolded_minus_self_sufficient"}
    uncertainty: dict[str, object] = {}
    seed_level: list[dict[str, object]] = []
    for metric_name in _DELTA_METRICS:
        delta_name = f"{metric_name}_delta"
        delta_value = round(
            with_summary[metric_name] - without_summary[metric_name],
            6,
        )
        seed_rows = paired_seed_delta_rows(
            metric_seed_values_from_payload(
                without_reflex if isinstance(without_reflex, Mapping) else payload,
                metric_name=metric_name,
                fallback_value=without_summary[metric_name],
            ),
            metric_seed_values_from_payload(
                with_reflex if isinstance(with_reflex, Mapping) else payload,
                metric_name=metric_name,
                fallback_value=with_summary[metric_name],
            ),
            metric_name=delta_name,
            condition=f"{variant_name}:competence_gap",
            fallback_delta=delta_value,
        )
        summary[delta_name] = delta_value
        uncertainty[delta_name] = aggregate_with_uncertainty(seed_rows)
        seed_level.extend(seed_level_dicts(seed_rows))
    return {
        **summary,
        "uncertainty": uncertainty,
        "seed_level": seed_level,
    }


def build_ladder_profile_deltas(
    profile_payloads: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    """Build per-rung reward-profile deltas versus austere."""
    minimal_profile = str(SHAPING_GAP_POLICY.get("minimal_profile") or "austere")
    austere_payload = profile_payloads.get(minimal_profile, {})
    austere_variants = (
        austere_payload.get("variants", {})
        if isinstance(austere_payload, Mapping)
        else {}
    )
    if not isinstance(austere_variants, Mapping):
        austere_variants = {}

    deltas_vs_austere: dict[str, object] = {}
    for variant_name, austere_variant_payload in sorted(austere_variants.items()):
        if not isinstance(austere_variant_payload, Mapping):
            continue
        profile_deltas: dict[str, object] = {}
        austere_summary = _metric_summary(austere_variant_payload)
        for profile_name in REWARD_PROFILES.keys():
            if profile_name == minimal_profile:
                continue
            profile_payload = profile_payloads.get(profile_name, {})
            variants = (
                profile_payload.get("variants", {})
                if isinstance(profile_payload, Mapping)
                else {}
            )
            if not isinstance(variants, Mapping):
                continue
            profile_variant_payload = variants.get(variant_name)
            if not isinstance(profile_variant_payload, Mapping):
                continue
            profile_summary = _metric_summary(profile_variant_payload)
            metric_blocks: dict[str, object] = {}
            comparison_seed_level: list[dict[str, object]] = []
            for metric_name in _DELTA_METRICS:
                profile_values = metric_seed_values_from_payload(
                    profile_variant_payload,
                    metric_name=metric_name,
                    fallback_value=profile_summary[metric_name],
                )
                austere_values = metric_seed_values_from_payload(
                    austere_variant_payload,
                    metric_name=metric_name,
                    fallback_value=austere_summary[metric_name],
                )
                delta_value = round(
                    profile_summary[metric_name] - austere_summary[metric_name],
                    6,
                )
                delta_rows = paired_seed_delta_rows(
                    austere_values,
                    profile_values,
                    metric_name=f"{metric_name}_delta",
                    condition=f"{variant_name}:{profile_name}_minus_{minimal_profile}",
                    fallback_delta=delta_value,
                )
                comparison_seed_level.extend(seed_level_dicts(delta_rows))
                profile_numeric = values_only(profile_values)
                austere_numeric = values_only(austere_values)
                if profile_numeric and austere_numeric:
                    effect_size, magnitude = cohens_d(
                        profile_numeric,
                        austere_numeric,
                    )
                    effect_size_value = round(float(effect_size), 6)
                else:
                    effect_size_value = None
                    magnitude = ""
                effect_rows: list[SeedLevelResult] = []
                effect_uncertainty = aggregate_with_uncertainty(())
                if effect_size_value is not None:
                    effect_rows = paired_seed_effect_size_rows(
                        austere_values,
                        profile_values,
                        condition=(
                            f"{variant_name}:{profile_name}_minus_{minimal_profile}"
                        ),
                        point_effect_size=effect_size_value,
                    )
                    comparison_seed_level.extend(seed_level_dicts(effect_rows))
                    effect_uncertainty = aggregate_with_uncertainty(effect_rows)
                metric_blocks[metric_name] = {
                    "profile_value": profile_summary[metric_name],
                    "austere_value": austere_summary[metric_name],
                    "delta": delta_value,
                    "delta_uncertainty": aggregate_with_uncertainty(delta_rows),
                    "cohens_d": effect_size_value,
                    "effect_magnitude": magnitude,
                    "effect_size_uncertainty": effect_uncertainty,
                    "seed_level": {
                        "delta": seed_level_dicts(delta_rows),
                        "cohens_d": seed_level_dicts(effect_rows),
                    },
                }
            profile_deltas[profile_name] = {
                "comparison": f"{profile_name}_minus_{minimal_profile}",
                "profile": profile_name,
                "minimal_profile": minimal_profile,
                "summary": {
                    f"{metric_name}_delta": metric_blocks[metric_name]["delta"]
                    for metric_name in _DELTA_METRICS
                },
                "metrics": metric_blocks,
                "seed_level": comparison_seed_level,
            }
        if profile_deltas:
            deltas_vs_austere[variant_name] = profile_deltas
    return {
        "minimal_profile": minimal_profile,
        "deltas_vs_austere": deltas_vs_austere,
    }


def _summary_scenario_success_delta(
    deltas_vs_austere: Mapping[str, object],
    profile_name: str,
) -> object | None:
    profile_delta = deltas_vs_austere.get(profile_name)
    if not isinstance(profile_delta, Mapping):
        return None
    summary = profile_delta.get("summary")
    if not isinstance(summary, Mapping):
        return None
    return summary.get("scenario_success_rate_delta")


def classify_architecture_learning(
    *,
    variant_name: str,
    protocol_name: str,
    profiles: Mapping[str, Mapping[str, object]],
    deltas_vs_austere: Mapping[str, object],
) -> dict[str, object]:
    """Classify whether a ladder rung survives austere or depends on shaping."""
    survival_threshold = float(
        SHAPING_GAP_POLICY.get("min_austere_survival_rate")
        or MINIMAL_SHAPING_SURVIVAL_THRESHOLD
    )
    success_limits = SHAPING_GAP_POLICY.get("max_scenario_success_rate_delta", {})
    classic_limit = safe_float(success_limits.get("classic_minus_austere", 0.20))
    ecological_limit = safe_float(success_limits.get("ecological_minus_austere", 0.15))

    classic_success = safe_float(
        _metric_summary(profiles.get("classic", {})).get("scenario_success_rate")
    )
    ecological_success = safe_float(
        _metric_summary(profiles.get("ecological", {})).get("scenario_success_rate")
    )
    austere_success = safe_float(
        _metric_summary(profiles.get("austere", {})).get("scenario_success_rate")
    )
    classic_gap = safe_float(
        _summary_scenario_success_delta(deltas_vs_austere, "classic")
    )
    ecological_gap = safe_float(
        _summary_scenario_success_delta(deltas_vs_austere, "ecological")
    )

    if austere_success >= survival_threshold:
        label = "austere_survivor"
        reason = (
            f"austere no-reflex scenario_success_rate reached the survival gate "
            f"({austere_success:.2f} >= {survival_threshold:.2f})."
        )
    elif classic_success < survival_threshold:
        label = "fails_with_shaping"
        reason = (
            f"classic no-reflex scenario_success_rate stayed below the survival gate "
            f"({classic_success:.2f} < {survival_threshold:.2f})."
        )
    elif classic_gap > classic_limit or ecological_gap > ecological_limit:
        label = "shaping_dependent"
        if classic_gap > classic_limit:
            reason = (
                "classic no-reflex scenario_success_rate exceeded the austere gap "
                f"policy ({classic_gap:.2f} > {classic_limit:.2f})."
            )
        else:
            reason = (
                "ecological no-reflex scenario_success_rate exceeded the austere gap "
                f"policy ({ecological_gap:.2f} > {ecological_limit:.2f})."
            )
    else:
        label = "mixed_or_borderline"
        reason = (
            "performance improved with denser reward support, but not enough to trip "
            "the shaping-gap or survival gates cleanly."
        )
    return {
        "variant": variant_name,
        "rung": LADDER_RUNG_BY_VARIANT.get(variant_name, ""),
        "protocol_name": protocol_name,
        "label": label,
        "reason": reason,
        "austere_success_rate": round(austere_success, 6),
        "classic_success_rate": round(classic_success, 6),
        "ecological_success_rate": round(ecological_success, 6),
        "austere_survival_threshold": round(survival_threshold, 6),
        "classic_minus_austere": round(classic_gap, 6),
        "ecological_minus_austere": round(ecological_gap, 6),
        "classic_gap_limit": round(classic_limit, 6),
        "ecological_gap_limit": round(ecological_limit, 6),
        "austere_survival_passed": austere_success >= survival_threshold,
    }


def build_ladder_classification_summary(
    variant_rows: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    """Build a compact classification table for cross-profile ladder runs."""
    rows: list[dict[str, object]] = []
    counts: dict[str, int] = {}
    for variant_name, payload in sorted(variant_rows.items()):
        if not isinstance(payload, Mapping):
            continue
        classification = payload.get("classification", {})
        if not isinstance(classification, Mapping):
            continue
        label = str(classification.get("label") or "")
        counts[label] = counts.get(label, 0) + 1
        shaping_gap = payload.get("shaping_gap", {})
        if not isinstance(shaping_gap, Mapping):
            shaping_gap = {}
        classic_gap = shaping_gap.get("classic", {})
        ecological_gap = shaping_gap.get("ecological", {})
        classic_gap_summary = (
            classic_gap.get("summary", {}) if isinstance(classic_gap, Mapping) else {}
        )
        ecological_gap_summary = (
            ecological_gap.get("summary", {})
            if isinstance(ecological_gap, Mapping)
            else {}
        )
        classic_gap_metrics = (
            classic_gap.get("metrics", {})
            if isinstance(classic_gap, Mapping)
            else {}
        )
        ecological_gap_metrics = (
            ecological_gap.get("metrics", {})
            if isinstance(ecological_gap, Mapping)
            else {}
        )
        classic_scenario_metric = (
            classic_gap_metrics.get("scenario_success_rate", {})
            if isinstance(classic_gap_metrics, Mapping)
            else {}
        )
        ecological_scenario_metric = (
            ecological_gap_metrics.get("scenario_success_rate", {})
            if isinstance(ecological_gap_metrics, Mapping)
            else {}
        )
        profiles = payload.get("profiles", {})
        if not isinstance(profiles, Mapping):
            profiles = {}
        rows.append(
            {
                "variant": variant_name,
                "rung": classification.get("rung"),
                "protocol_name": classification.get("protocol_name"),
                "classification": label,
                "reason": classification.get("reason"),
                "classic_success_rate": _metric_summary(
                    profiles.get("classic", {})
                )["scenario_success_rate"],
                "ecological_success_rate": _metric_summary(
                    profiles.get("ecological", {})
                )["scenario_success_rate"],
                "austere_success_rate": _metric_summary(
                    profiles.get("austere", {})
                )["scenario_success_rate"],
                "austere_survival_passed": classification.get(
                    "austere_survival_passed"
                ),
                "classic_minus_austere": classic_gap_summary.get(
                    "scenario_success_rate_delta"
                ),
                "classic_gap_ci_lower": (
                    classic_scenario_metric.get("delta_uncertainty", {})
                    if isinstance(classic_scenario_metric, Mapping)
                    else {}
                ).get("ci_lower"),
                "classic_gap_ci_upper": (
                    classic_scenario_metric.get("delta_uncertainty", {})
                    if isinstance(classic_scenario_metric, Mapping)
                    else {}
                ).get("ci_upper"),
                "classic_gap_n_seeds": (
                    classic_scenario_metric.get("delta_uncertainty", {})
                    if isinstance(classic_scenario_metric, Mapping)
                    else {}
                ).get("n_seeds"),
                "classic_gap_effect_size": (
                    classic_scenario_metric.get("cohens_d")
                    if isinstance(classic_scenario_metric, Mapping)
                    else None
                ),
                "ecological_minus_austere": ecological_gap_summary.get(
                    "scenario_success_rate_delta"
                ),
                "ecological_gap_ci_lower": (
                    ecological_scenario_metric.get("delta_uncertainty", {})
                    if isinstance(ecological_scenario_metric, Mapping)
                    else {}
                ).get("ci_lower"),
                "ecological_gap_ci_upper": (
                    ecological_scenario_metric.get("delta_uncertainty", {})
                    if isinstance(ecological_scenario_metric, Mapping)
                    else {}
                ).get("ci_upper"),
                "ecological_gap_n_seeds": (
                    ecological_scenario_metric.get("delta_uncertainty", {})
                    if isinstance(ecological_scenario_metric, Mapping)
                    else {}
                ).get("n_seeds"),
                "ecological_gap_effect_size": (
                    ecological_scenario_metric.get("cohens_d")
                    if isinstance(ecological_scenario_metric, Mapping)
                    else None
                ),
                "austere_competence_gap": (
                    payload.get("competence_gaps", {})
                    if isinstance(payload.get("competence_gaps", {}), Mapping)
                    else {}
                ).get("austere", {}).get("scenario_success_rate_delta")
                if isinstance(
                    (
                        payload.get("competence_gaps", {})
                        if isinstance(payload.get("competence_gaps", {}), Mapping)
                        else {}
                    ).get("austere", {}),
                    Mapping,
                )
                else None,
            }
        )
    return {
        "available": bool(rows),
        "counts": counts,
        "rows": rows,
    }


def compare_ladder_under_profiles(
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
    capacity_profile: str | None = None,
    map_template: str = "central_burrow",
    operational_profile: str | OperationalProfile | None = None,
    noise_profile: str | NoiseConfig | None = None,
    budget_profile: str | BudgetProfile | None = None,
    seeds: Sequence[int] | None = None,
    names: Sequence[str] | None = None,
    episodes_per_scenario: int | None = None,
    checkpoint_selection: str = "none",
    checkpoint_selection_config: CheckpointSelectionConfig | None = None,
    checkpoint_interval: int | None = None,
    checkpoint_dir: str | Path | None = None,
    enforce_benchmark_policy: bool = False,
) -> tuple[Dict[str, object], List[Dict[str, object]]]:
    """Run the A0-A4 architectural ladder under all reward profiles."""
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
    if enforce_benchmark_policy:
        if budget.profile != "paper":
            raise ValueError(
                "compare_ladder_under_profiles() benchmark workflows require "
                "budget_profile='paper'."
            )
        if checkpoint_selection != "best":
            raise ValueError(
                "compare_ladder_under_profiles() benchmark workflows require "
                "checkpoint_selection='best'."
            )

    resolved_checkpoint_config = resolve_checkpoint_selection_config(
        checkpoint_selection_config
    )
    profile_payloads: dict[str, dict[str, object]] = {}
    combined_rows: list[dict[str, object]] = []
    resolved_scenario_names: list[str] = []
    for profile_name in REWARD_PROFILES.keys():
        profile_payload, profile_rows = compare_ablation_suite(
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
            reward_profile=profile_name,
            map_template=map_template,
            operational_profile=operational_profile,
            noise_profile=noise_profile,
            budget_profile=budget.profile,
            seeds=tuple(budget.ablation_seeds),
            names=names,
            variant_names=LADDER_VARIANT_NAMES,
            episodes_per_scenario=budget.scenario_episodes,
            checkpoint_selection=checkpoint_selection,
            checkpoint_selection_config=resolved_checkpoint_config,
            checkpoint_interval=budget.checkpoint_interval,
            checkpoint_dir=checkpoint_dir,
        )
        profile_payload = dict(profile_payload)
        profile_payload["reward_profile"] = profile_name
        if not resolved_scenario_names:
            resolved = profile_payload.get("scenario_names", [])
            if isinstance(resolved, list):
                resolved_scenario_names = [str(item) for item in resolved]
        profile_payloads[profile_name] = profile_payload
        combined_rows.extend(profile_rows)

    deltas_payload = build_ladder_profile_deltas(profile_payloads)
    deltas_vs_austere = deltas_payload.get("deltas_vs_austere", {})
    if not isinstance(deltas_vs_austere, Mapping):
        deltas_vs_austere = {}

    variant_summaries: dict[str, object] = {}
    for variant_name, rung in sorted(
        LADDER_RUNG_BY_VARIANT.items(),
        key=lambda item: item[1],
    ):
        per_profile_payloads = {
            profile_name: (
                profile_payloads.get(profile_name, {}).get("variants", {})
                if isinstance(profile_payloads.get(profile_name, {}), Mapping)
                else {}
            ).get(variant_name, {})
            for profile_name in REWARD_PROFILES.keys()
        }
        if not any(
            isinstance(payload, Mapping) and payload
            for payload in per_profile_payloads.values()
        ):
            continue
        competence_gaps = {
            profile_name: _competence_gap_from_variant_payload(
                variant_name,
                payload if isinstance(payload, Mapping) else {},
            )
            for profile_name, payload in per_profile_payloads.items()
            if isinstance(payload, Mapping) and payload
        }
        classification = classify_architecture_learning(
            variant_name=variant_name,
            protocol_name=LADDER_PROTOCOL_NAMES.get(rung, rung),
            profiles=per_profile_payloads,
            deltas_vs_austere=(
                deltas_vs_austere.get(variant_name, {})
                if isinstance(deltas_vs_austere.get(variant_name, {}), Mapping)
                else {}
            ),
        )
        variant_summaries[variant_name] = {
            "variant": variant_name,
            "rung": rung,
            "protocol_name": LADDER_PROTOCOL_NAMES.get(rung, rung),
            "profiles": {
                profile_name: {
                    "summary": _metric_summary(
                        payload if isinstance(payload, Mapping) else {}
                    ),
                    "uncertainty": {
                        metric_name: _metric_uncertainty(
                            payload if isinstance(payload, Mapping) else {},
                            metric_name,
                        )
                        for metric_name in _DELTA_METRICS
                    },
                }
                for profile_name, payload in per_profile_payloads.items()
                if isinstance(payload, Mapping) and payload
            },
            "austere_survival": {
                "success_rate": _metric_summary(
                    per_profile_payloads.get("austere", {})
                )["scenario_success_rate"],
                "survival_threshold": safe_float(
                    classification.get("austere_survival_threshold")
                )
                or float(
                    SHAPING_GAP_POLICY.get("min_austere_survival_rate")
                    or MINIMAL_SHAPING_SURVIVAL_THRESHOLD
                ),
                "passed": bool(classification.get("austere_survival_passed", False)),
            },
            "shaping_gap": (
                deltas_vs_austere.get(variant_name, {})
                if isinstance(deltas_vs_austere.get(variant_name, {}), Mapping)
                else {}
            ),
            "competence_gaps": competence_gaps,
            "classification": classification,
        }

    classification_summary = build_ladder_classification_summary(variant_summaries)
    return {
        "available": bool(profile_payloads),
        "budget_profile": budget.profile,
        "benchmark_strength": budget.benchmark_strength,
        "checkpoint_selection": checkpoint_selection,
        "checkpoint_metric": resolved_checkpoint_config.metric,
        "checkpoint_penalty_config": resolved_checkpoint_config.to_summary(),
        "primary_evaluation": "without_reflex_support",
        "profile_order": list(REWARD_PROFILES.keys()),
        "minimal_profile": str(SHAPING_GAP_POLICY.get("minimal_profile") or "austere"),
        "seeds": list(budget.ablation_seeds),
        "scenario_names": resolved_scenario_names,
        "episodes_per_scenario": budget.scenario_episodes,
        "benchmark_policy": {
            "enforced": bool(enforce_benchmark_policy),
            "budget_profile": budget.profile,
            "checkpoint_selection": checkpoint_selection,
            "paper_budget_required": True,
            "checkpoint_selection_required": "best",
            "passed": (
                budget.profile == "paper" and checkpoint_selection == "best"
            ),
        },
        "profiles": profile_payloads,
        "variants": variant_summaries,
        "deltas_vs_austere": dict(deltas_payload),
        "classification_summary": classification_summary,
    }, combined_rows


__all__ = [
    "build_ladder_classification_summary",
    "build_ladder_profile_deltas",
    "classify_architecture_learning",
    "compare_ladder_under_profiles",
]
