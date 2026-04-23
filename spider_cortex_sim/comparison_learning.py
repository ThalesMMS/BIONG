"""Cross-run comparison and condensed reporting helpers.

``condense_robustness_summary`` is the public home for the CLI helper formerly
named ``_short_robustness_matrix_summary``.
"""

from __future__ import annotations

from dataclasses import replace
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
from .distillation import DistillationConfig
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
from .training_regimes import TrainingRegimeSpec, resolve_training_regime

from .comparison_utils import aggregate_with_uncertainty, attach_behavior_seed_statistics, condition_compact_summary, condition_mean_reward, metric_seed_values_from_payload, noise_profile_metadata, paired_seed_delta_rows, seed_level_dicts


def build_distillation_comparison_report(
    *,
    teacher: Mapping[str, object],
    modular_rl_from_scratch: Mapping[str, object] | None,
    modular_distilled: Mapping[str, object],
    modular_distilled_plus_rl_finetuning: Mapping[str, object] | None = None,
) -> Dict[str, object]:
    def _summary(summary: Mapping[str, object] | None) -> Dict[str, float]:
        raw_payload = dict(summary or {})
        payload = (
            raw_payload
            if "summary" in raw_payload or "suite" in raw_payload
            else {"summary": raw_payload}
        )
        return condition_compact_summary(payload)

    def _metric_delta(
        candidate_summary: Dict[str, float],
        reference_summary: Dict[str, float],
    ) -> Dict[str, float]:
        return {
            "scenario_success_rate_delta": round(
                candidate_summary["scenario_success_rate"]
                - reference_summary["scenario_success_rate"],
                6,
            ),
            "episode_success_rate_delta": round(
                candidate_summary["episode_success_rate"]
                - reference_summary["episode_success_rate"],
                6,
            ),
            "mean_reward_delta": round(
                candidate_summary["mean_reward"]
                - reference_summary["mean_reward"],
                6,
            ),
        }

    def _gap_closed_fraction(
        *,
        baseline_value: float,
        teacher_value: float,
        candidate_value: float,
    ) -> float | None:
        teacher_gap = float(teacher_value) - float(baseline_value)
        if teacher_gap <= 0.0 or math.isclose(
            teacher_gap,
            0.0,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            return None
        return round(
            (float(candidate_value) - float(baseline_value)) / teacher_gap,
            6,
        )

    teacher_summary = _summary(teacher)
    baseline_summary = (
        None
        if modular_rl_from_scratch is None
        else _summary(modular_rl_from_scratch)
    )
    distilled_summary = _summary(modular_distilled)
    finetuned_summary = (
        None
        if modular_distilled_plus_rl_finetuning is None
        else _summary(modular_distilled_plus_rl_finetuning)
    )

    baseline_success = (
        None
        if baseline_summary is None
        else baseline_summary["scenario_success_rate"]
    )
    distilled_success = distilled_summary["scenario_success_rate"]
    finetuned_success = (
        None
        if finetuned_summary is None
        else finetuned_summary["scenario_success_rate"]
    )
    teacher_success = teacher_summary["scenario_success_rate"]
    best_condition_name = "modular_distilled"
    best_summary = distilled_summary
    if baseline_success is not None and distilled_summary["scenario_success_rate"] <= baseline_success:
        if (
            finetuned_summary is not None
            and finetuned_summary["scenario_success_rate"]
            > best_summary["scenario_success_rate"]
        ):
            best_condition_name = "modular_distilled_plus_rl_finetuning"
            best_summary = finetuned_summary

    if baseline_success is None:
        if teacher_success <= 0.0:
            answer = "inconclusive"
            rationale = (
                "The teacher did not establish a positive no-reflex reference, so expressivity remains unproven."
            )
        elif distilled_success > 0.0:
            answer = "yes"
            rationale = (
                f"modular_distilled achieved scenario_success_rate "
                f"{distilled_success:.3f} against a teacher reference of {teacher_success:.3f}."
            )
        elif finetuned_summary is not None and finetuned_success is not None and finetuned_success > 0.0:
            best_condition_name = "modular_distilled_plus_rl_finetuning"
            best_summary = finetuned_summary
            answer = "yes_after_rl_finetuning"
            rationale = (
                f"modular_distilled_plus_rl_finetuning achieved scenario_success_rate "
                f"{finetuned_success:.3f} against a teacher reference of {teacher_success:.3f}."
            )
        else:
            answer = "no"
            rationale = (
                "The modular student did not achieve positive no-reflex success after distillation."
            )
    elif teacher_success <= baseline_success:
        answer = "inconclusive"
        rationale = (
            "The teacher did not outperform the modular RL baseline, so expressivity remains unproven."
        )
    elif best_summary["scenario_success_rate"] > baseline_success:
        answer = (
            "yes"
            if best_condition_name == "modular_distilled"
            else "yes_after_rl_finetuning"
        )
        rationale = (
            f"{best_condition_name} improved scenario_success_rate from "
            f"{baseline_success:.3f} to {best_summary['scenario_success_rate']:.3f} "
            f"against a teacher reference of {teacher_success:.3f}."
        )
    elif best_summary["scenario_success_rate"] > 0.0:
        answer = "partially"
        rationale = (
            f"The modular student achieved non-zero scenario_success_rate "
            f"({best_summary['scenario_success_rate']:.3f}) but did not surpass the RL-only modular baseline ({baseline_success:.3f})."
        )
    else:
        answer = "no"
        rationale = (
            "The modular student did not achieve positive no-reflex success after distillation."
        )

    quantitative_evidence = {
        "modular_distilled_vs_teacher": _metric_delta(
            distilled_summary,
            teacher_summary,
        ),
    }
    if baseline_summary is not None and baseline_success is not None:
        quantitative_evidence["teacher_vs_modular_rl_from_scratch"] = _metric_delta(
            teacher_summary,
            baseline_summary,
        )
        quantitative_evidence[
            "modular_distilled_vs_modular_rl_from_scratch"
        ] = _metric_delta(
            distilled_summary,
            baseline_summary,
        )
        quantitative_evidence["gap_closed_fraction"] = {
            "modular_distilled": _gap_closed_fraction(
                baseline_value=baseline_success,
                teacher_value=teacher_success,
                candidate_value=distilled_success,
            ),
            "modular_distilled_plus_rl_finetuning": (
                None
                if finetuned_success is None
                else _gap_closed_fraction(
                    baseline_value=baseline_success,
                    teacher_value=teacher_success,
                    candidate_value=finetuned_success,
                )
            ),
        }
    if finetuned_summary is not None:
        quantitative_evidence[
            "modular_distilled_plus_rl_finetuning_vs_teacher"
        ] = _metric_delta(finetuned_summary, teacher_summary)
        if baseline_summary is not None:
            quantitative_evidence[
                "modular_distilled_plus_rl_finetuning_vs_modular_rl_from_scratch"
            ] = _metric_delta(finetuned_summary, baseline_summary)

    return {
        "question": "Can modular architecture express an effective policy?",
        "answer": answer,
        "rationale": rationale,
        "primary_metric": "scenario_success_rate",
        "teacher": teacher_summary,
        "modular_rl_from_scratch": baseline_summary,
        "modular_distilled": distilled_summary,
        "modular_distilled_plus_rl_finetuning": finetuned_summary,
        "best_student_condition": best_condition_name,
        "quantitative_evidence": quantitative_evidence,
    }

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
    Build a compact evidence summary comparing a reference condition against standard comparators.
    
    Provides availability flags, per-metric deltas (reference minus comparator) for
    `scenario_success_rate`, `episode_success_rate`, and `mean_reward`, aggregated
    seed-level payloads, and uncertainty summaries for the comparisons.
    
    Parameters:
        conditions (Dict[str, Dict[str, object]]): Mapping from condition name to its compact payload; payloads may include a `"skipped"` flag and may contain `seed_level` and `uncertainty` entries.
        reference_condition (str): Condition name used as the primary trained reference (typically `"trained_without_reflex_support"`).
    
    Returns:
        Dict[str, object]: Summary with keys including:
            - reference_condition: the provided reference name.
            - primary_gate_metric: the metric used for the primary evidence gate (`"scenario_success_rate"`).
            - supports_primary_evidence: `true` if reference, `random_init`, and `reflex_only` are present and not skipped.
            - has_learning_evidence: `true` if the reference's `scenario_success_rate` exceeds both `random_init` and `reflex_only`.
            - primary_condition, trained_final, random_init, reflex_only, trained_without_reflex_support: compact summaries for each condition (zeroed defaults when missing).
            - trained_vs_random_init, trained_vs_reflex_only: dicts with `scenario_success_rate_delta`, `episode_success_rate_delta`, and `mean_reward_delta` (computed as reference minus comparator and rounded to 6 decimals).
            - seed_level: mapping of condition name to its list of seed-level payloads.
            - uncertainty: mapping that contains per-condition uncertainties and aggregated delta uncertainties for the two comparisons.
            - notes: human-readable messages describing gating and availability.
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
        """
        Compute aggregated uncertainty for the per-metric deltas between a reference condition and a named comparison condition.
        
        Looks up the comparison and reference payloads from the module-level `conditions` mapping using `comparison_name` and `reference_condition`. For each metric (`scenario_success_rate`, `episode_success_rate`, `mean_reward`) this function:
        - extracts per-seed metric values (falling back to the provided summaries when seed lists are absent),
        - pairs comparison and reference seed values into seed-level delta rows (using values from `deltas` as a fallback delta),
        - aggregates those seed-level rows into an uncertainty summary.
        
        Parameters:
            comparison_name (str): Name of the comparison condition to look up in the module-level `conditions`.
            comparison_summary (Dict[str, float]): Compact summary values for the comparison condition used as fallbacks when seed-level values are missing.
            deltas (Dict[str, float]): Precomputed metric deltas used as fallback delta values for paired seed rows.
        
        Returns:
            Dict[str, object]: Mapping of delta metric names to their aggregated uncertainty summaries, e.g.
                {
                    "scenario_success_rate_delta": <aggregated uncertainty object>,
                    "episode_success_rate_delta": <aggregated uncertainty object>,
                    "mean_reward_delta": <aggregated uncertainty object>
                }
        """
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
    if not seed_values:
        raise ValueError(
            "compare_learning_evidence() requires at least one behavior seed."
        )
    run_count = max(1, int(base_budget.scenario_episodes))
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
        if base_config.architecture not in condition.supports_architectures:
            conditions[condition.name] = {
                "condition": condition.name,
                "description": condition.description,
                "policy_mode": policy_mode,
                "train_budget": condition.train_budget,
                "training_regime": training_regime_name,
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
                "training_regime_summary": (
                    None
                    if resolved_training_regime_spec is None
                    else resolved_training_regime_spec.to_summary()
                ),
                "skipped": True,
                "reason": (
                    f"Condition incompatible with architecture {base_config.architecture!r}."
                ),
                **noise_profile_metadata(resolved_noise_profile),
            }
            continue
        if (
            resolved_training_regime_spec is not None
            and bool(resolved_training_regime_spec.distillation_enabled)
            and distillation_config is None
        ):
            conditions[condition.name] = {
                "condition": condition.name,
                "description": condition.description,
                "policy_mode": policy_mode,
                "train_budget": condition.train_budget,
                "training_regime": training_regime_name,
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
                "training_regime_summary": (
                    None
                    if resolved_training_regime_spec is None
                    else resolved_training_regime_spec.to_summary()
                ),
                "skipped": True,
                "reason": (
                    f"Condition {condition.name!r} requires a distillation teacher checkpoint."
                ),
                **noise_profile_metadata(resolved_noise_profile),
            }
            continue
        if (
            policy_mode == "reflex_only"
            and not base_config.enable_reflexes
        ):
            conditions[condition.name] = {
                "condition": condition.name,
                "description": condition.description,
                "policy_mode": policy_mode,
                "train_budget": condition.train_budget,
                "training_regime": training_regime_name,
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
                "training_regime_summary": (
                    None
                    if resolved_training_regime_spec is None
                    else resolved_training_regime_spec.to_summary()
                ),
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
                    checkpoint_selection_config=checkpoint_selection_config,
                    checkpoint_interval=condition_budget.checkpoint_interval,
                    checkpoint_dir=run_checkpoint_dir,
                    checkpoint_scenario_names=scenario_names,
                    selection_scenario_episodes=condition_budget.selection_scenario_episodes,
                    training_regime=resolved_training_regime,
                    teacher_checkpoint=(
                        None
                        if distillation_config is None
                        else distillation_config.teacher_checkpoint
                    ),
                    distillation_config=distillation_config,
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
                    checkpoint_selection_config=checkpoint_selection_config,
                    checkpoint_interval=long_budget.checkpoint_interval,
                    checkpoint_dir=run_checkpoint_dir,
                    checkpoint_scenario_names=scenario_names,
                    selection_scenario_episodes=long_budget.selection_scenario_episodes,
                    training_regime=resolved_training_regime,
                    teacher_checkpoint=(
                        None
                        if distillation_config is None
                        else distillation_config.teacher_checkpoint
                    ),
                    distillation_config=distillation_config,
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
                    checkpoint_selection_config=checkpoint_selection_config,
                    checkpoint_interval=base_budget.checkpoint_interval,
                    checkpoint_dir=run_checkpoint_dir,
                    checkpoint_scenario_names=scenario_names,
                    selection_scenario_episodes=base_budget.selection_scenario_episodes,
                    training_regime=resolved_training_regime,
                    teacher_checkpoint=(
                        None
                        if distillation_config is None
                        else distillation_config.teacher_checkpoint
                    ),
                    distillation_config=distillation_config,
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
