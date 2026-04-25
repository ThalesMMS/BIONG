from __future__ import annotations

from typing import Dict, Sequence

from .common import (
    aggregate_with_uncertainty,
    condition_compact_summary,
    condition_mean_reward,
    metric_seed_values_from_payload,
    paired_seed_delta_rows,
    seed_level_dicts,
)

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

    def _condition_available(condition_name: str) -> bool:
        payload = conditions.get(condition_name)
        if not isinstance(payload, dict):
            return False
        if bool(payload.get("skipped")) or bool(payload.get("unavailable")):
            return False
        return isinstance(payload.get("summary"), dict)

    def _summary_delta(
        comparison_summary: Dict[str, float],
        *,
        comparison_available: bool,
    ) -> Dict[str, object]:
        if not reference_available or not comparison_available:
            return {"unavailable": True}
        return {
            "scenario_success_rate_delta": round(
                reference["scenario_success_rate"]
                - comparison_summary["scenario_success_rate"],
                6,
            ),
            "episode_success_rate_delta": round(
                reference["episode_success_rate"]
                - comparison_summary["episode_success_rate"],
                6,
            ),
            "mean_reward_delta": round(
                reference["mean_reward"] - comparison_summary["mean_reward"],
                6,
            ),
        }

    reference_available = (
        _condition_available(reference_condition)
    )
    random_init_available = (
        _condition_available("random_init")
    )
    reflex_only_available = (
        _condition_available("reflex_only")
    )
    primary_supported = (
        reference_available
        and random_init_available
        and reflex_only_available
    )
    trained_vs_random = _summary_delta(
        random_init,
        comparison_available=random_init_available,
    )
    trained_vs_reflex = _summary_delta(
        reflex_only,
        comparison_available=reflex_only_available,
    )
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
        deltas: Dict[str, object],
    ) -> Dict[str, object]:
        """
        Compute aggregated uncertainty for reference-vs-comparison deltas for the three primary metrics.
        
        For each metric ("scenario_success_rate", "episode_success_rate", "mean_reward"), this extracts per-seed values for the reference and comparison (using provided summaries as fallbacks), forms paired per-seed delta rows (using values from `deltas` as a fallback), and returns the aggregated uncertainty object for each delta metric.
        
        Parameters:
            comparison_name (str): Name of the comparison condition to use when looking up per-seed values.
            comparison_summary (Dict[str, float]): Compact summary values for the comparison condition used as fallbacks when seed-level lists are absent.
            deltas (Dict[str, float]): Precomputed delta values used as fallback delta entries for paired seed rows.
        
        Returns:
            Dict[str, object]: Mapping from delta metric name to its aggregated uncertainty object, e.g.
                {
                    "scenario_success_rate_delta": <aggregated uncertainty object>,
                    "episode_success_rate_delta": <aggregated uncertainty object>,
                    "mean_reward_delta": <aggregated uncertainty object>
                }
        """
        if not reference_available or not _condition_available(comparison_name):
            return {"unavailable": True}
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

__all__ = ["build_learning_evidence_deltas", "build_learning_evidence_summary"]
