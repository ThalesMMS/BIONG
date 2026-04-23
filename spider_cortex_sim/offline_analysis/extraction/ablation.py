from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence

from ...ablations import compare_predator_type_ablation_performance
from ..constants import (
    CANONICAL_NOISE_CONDITIONS,
    DEFAULT_MINIMAL_SHAPING_SURVIVAL_THRESHOLD,
    DEFAULT_MODULE_NAMES,
    LADDER_ACTIVE_RUNGS,
    LADDER_ADJACENT_COMPARISONS,
    LADDER_PRIMARY_VARIANT_BY_RUNG,
    LADDER_PROTOCOL_NAMES,
    LADDER_RUNG_DESCRIPTIONS,
    REFLEX_DOMINANCE_WARNING_THRESHOLD,
    REFLEX_OVERRIDE_WARNING_THRESHOLD,
    SHAPING_DEPENDENCE_WARNING_THRESHOLD,
)
from ..uncertainty import _payload_has_zero_reflex_scale, _payload_uncertainty
from ..utils import (
    _coerce_bool,
    _coerce_float,
    _coerce_optional_float,
    _dominant_module_by_score,
    _mapping_or_empty,
    _mean,
    _safe_divide,
)

from .shaping import _ablation_deltas_vs_reference, _rows_with_minimal_reflex_support, _variant_with_minimal_reflex_support

def _extract_ladder_rungs(
    variants: Mapping[str, object],
) -> dict[str, object]:
    recognized_rungs = [
        rung
        for rung in LADDER_ACTIVE_RUNGS
        if LADDER_PRIMARY_VARIANT_BY_RUNG.get(rung, "") in variants
    ]
    applicable_comparisons = [
        {
            "baseline_rung": baseline_rung,
            "comparison_rung": comparison_rung,
        }
        for baseline_rung, comparison_rung in LADDER_ADJACENT_COMPARISONS
        if baseline_rung in recognized_rungs and comparison_rung in recognized_rungs
    ]
    return {
        "recognized_rungs": recognized_rungs,
        "descriptions": {
            rung: LADDER_RUNG_DESCRIPTIONS.get(rung, "")
            for rung in recognized_rungs
        },
        "applicable_comparisons": applicable_comparisons,
    }


def extract_ladder_comparison(
    variants: Mapping[str, object],
    *,
    source: str,
) -> dict[str, object]:
    rung_to_variant = {
        rung: variant_name
        for rung, variant_name in LADDER_PRIMARY_VARIANT_BY_RUNG.items()
        if variant_name in variants
    }
    comparisons: list[dict[str, object]] = []
    limitations: list[str] = []
    for baseline_rung, comparison_rung in LADDER_ADJACENT_COMPARISONS:
        baseline_variant = rung_to_variant.get(baseline_rung, "")
        comparison_variant = rung_to_variant.get(comparison_rung, "")
        if not baseline_variant or not comparison_variant:
            limitations.append(
                f"Architectural ladder comparison {baseline_rung} vs {comparison_rung} "
                "was unavailable because one of the required variants was missing."
            )
            continue
        deltas = _ablation_deltas_vs_reference(
            {
                baseline_variant: variants[baseline_variant],
                comparison_variant: variants[comparison_variant],
            },
            reference_variant=baseline_variant,
        )
        baseline_payload = _mapping_or_empty(variants.get(baseline_variant))
        comparison_payload = _mapping_or_empty(variants.get(comparison_variant))
        baseline_summary = _mapping_or_empty(baseline_payload.get("summary"))
        comparison_summary = _mapping_or_empty(comparison_payload.get("summary"))
        comparison_delta = _mapping_or_empty(deltas.get(comparison_variant))
        summary_delta = _mapping_or_empty(comparison_delta.get("summary"))
        comparisons.append(
            {
                "baseline_rung": baseline_rung,
                "comparison_rung": comparison_rung,
                "metrics": {
                    "baseline_variant": baseline_variant,
                    "comparison_variant": comparison_variant,
                    "baseline_protocol_name": LADDER_PROTOCOL_NAMES.get(
                        baseline_rung,
                        baseline_rung,
                    ),
                    "comparison_protocol_name": LADDER_PROTOCOL_NAMES.get(
                        comparison_rung,
                        comparison_rung,
                    ),
                    "baseline": {
                        "scenario_success_rate": _coerce_float(
                            baseline_summary.get("scenario_success_rate")
                        ),
                        "episode_success_rate": _coerce_float(
                            baseline_summary.get("episode_success_rate")
                        ),
                        "mean_reward": (
                            _coerce_optional_float(baseline_summary.get("mean_reward"))
                            if "mean_reward" in baseline_summary
                            else None
                        ),
                    },
                    "comparison": {
                        "scenario_success_rate": _coerce_float(
                            comparison_summary.get("scenario_success_rate")
                        ),
                        "episode_success_rate": _coerce_float(
                            comparison_summary.get("episode_success_rate")
                        ),
                        "mean_reward": (
                            _coerce_optional_float(comparison_summary.get("mean_reward"))
                            if "mean_reward" in comparison_summary
                            else None
                        ),
                    },
                },
                "deltas": {
                    "scenario_success_rate_delta": _coerce_float(
                        summary_delta.get("scenario_success_rate_delta")
                    ),
                    "episode_success_rate_delta": _coerce_float(
                        summary_delta.get("episode_success_rate_delta")
                    ),
                    "mean_reward_delta": (
                        _coerce_optional_float(summary_delta.get("mean_reward_delta"))
                        if (
                            "mean_reward" in baseline_summary
                            and baseline_summary.get("mean_reward") is not None
                            and "mean_reward" in comparison_summary
                            and comparison_summary.get("mean_reward") is not None
                            and "mean_reward_delta" in summary_delta
                        )
                        else None
                    ),
                },
                "summary": dict(summary_delta),
                "scenarios": dict(_mapping_or_empty(comparison_delta.get("scenarios"))),
                "source": f"{source}.ladder.{baseline_rung}_vs_{comparison_rung}",
            }
        )

    return {
        "available": bool(comparisons),
        "comparisons": comparisons,
        "explanatory_notes": dict(LADDER_RUNG_DESCRIPTIONS),
        "limitations": limitations,
    }


def extract_ladder_profile_comparison(
    summary: Mapping[str, object],
) -> dict[str, object]:
    """Backward-compatible alias for `extract_reward_profile_ladder()`."""
    return extract_reward_profile_ladder(summary)


def extract_reward_profile_ladder(
    summary: Mapping[str, object],
) -> dict[str, object]:
    """Extract and normalize the cross-profile architectural ladder payload."""
    behavior_evaluation = _mapping_or_empty(summary.get("behavior_evaluation"))
    payload = _mapping_or_empty(behavior_evaluation.get("ladder_under_profiles"))
    if not payload:
        payload = _mapping_or_empty(
            behavior_evaluation.get("ladder_profile_comparison")
        )
    if not payload:
        return {
            "available": False,
            "profiles": {},
            "cross_profile_deltas": {},
            "classifications": {},
            "limitations": [],
        }

    profile_payloads = _mapping_or_empty(payload.get("profiles"))
    profiles: dict[str, object] = {}
    for profile_name, profile_payload in sorted(profile_payloads.items()):
        if not isinstance(profile_payload, Mapping):
            continue
        variants_payload = _mapping_or_empty(profile_payload.get("variants"))
        normalized_variants = {
            str(variant_name): _variant_with_minimal_reflex_support(variant_payload)
            for variant_name, variant_payload in variants_payload.items()
            if isinstance(variant_payload, Mapping)
        }
        profiles[str(profile_name)] = {
            "reward_profile": str(profile_payload.get("reward_profile") or profile_name),
            "reference_variant": str(profile_payload.get("reference_variant") or ""),
            "scenario_names": list(profile_payload.get("scenario_names", []))
            if isinstance(profile_payload.get("scenario_names"), list)
            else [],
            "variants": normalized_variants,
        }

    variants_payload = _mapping_or_empty(payload.get("variants"))
    classifications = {
        str(variant_name): {
            "classification": dict(
                _mapping_or_empty(variant_payload.get("classification"))
            ),
            "austere_survival": dict(
                _mapping_or_empty(variant_payload.get("austere_survival"))
            ),
            "protocol_name": str(variant_payload.get("protocol_name") or ""),
            "rung": str(variant_payload.get("rung") or ""),
        }
        for variant_name, variant_payload in sorted(variants_payload.items())
        if isinstance(variant_payload, Mapping)
    }

    limitations = payload.get("limitations", [])
    return {
        "available": bool(payload.get("available", True)),
        "profiles": profiles,
        "minimal_profile": str(
            _mapping_or_empty(payload.get("deltas_vs_austere")).get(
                "minimal_profile",
                payload.get("minimal_profile", "austere"),
            )
            or "austere"
        ),
        "cross_profile_deltas": dict(
            _mapping_or_empty(
                _mapping_or_empty(payload.get("deltas_vs_austere")).get(
                    "deltas_vs_austere"
                )
            )
        ),
        "classifications": classifications,
        "classification_summary": dict(
            _mapping_or_empty(payload.get("classification_summary"))
        ),
        "raw_payload": dict(payload),
        "limitations": (
            [str(item) for item in limitations if item]
            if isinstance(limitations, list)
            else []
        ),
    }

def extract_ablations(
    summary: Mapping[str, object],
    behavior_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """
    Extract ablation variant payloads and compute deltas versus a reference variant.
    
    When ablation data exists in summary["behavior_evaluation"]["ablations"], this function normalizes each variant to minimal-reflex-support form and computes deltas versus a chosen reference variant. When summary ablations are absent, it reconstructs per-variant summaries from behavior_csv rows by grouping on `ablation_variant` and using the rows with minimal `eval_reflex_scale` per variant.
    
    Parameters:
        summary (Mapping[str, object]): Parsed report summary that may contain `behavior_evaluation.ablations`.
        behavior_rows (Sequence[Mapping[str, object]]): Normalized rows from the behavior CSV used as a fallback when summary ablations are not present.
    
    Returns:
        dict[str, object]: A mapping with the following keys:
            available (bool): True when any variant data was extracted or reconstructed.
            source (str): `"summary.behavior_evaluation.ablations"` when sourced from the summary, otherwise `"behavior_csv"`.
            reference_variant (str): The chosen reference variant name (e.g., `"modular_full"`) or empty string if none.
            variants (dict[str, object]): Mapping of variant names to normalized payloads. Each payload contains at least `config` and `summary` with `eval_reflex_scale` and success-rate metrics.
            deltas_vs_reference (dict[str, object]): Per-variant and per-scenario deltas computed against the selected reference variant.
            limitations (list[str]): Human-readable notes describing fallbacks, missing fields, or reconstruction caveats.
    """
    behavior_evaluation = summary.get("behavior_evaluation", {})
    ablations = (
        behavior_evaluation.get("ablations", {})
        if isinstance(behavior_evaluation, Mapping)
        else {}
    )
    limitations: list[str] = []
    if isinstance(ablations, Mapping) and ablations:
        variants_payload = ablations.get("variants", {})
        variants = (
            {
                str(variant): _variant_with_minimal_reflex_support(payload)
                for variant, payload in variants_payload.items()
                if isinstance(payload, Mapping)
            }
            if isinstance(variants_payload, Mapping)
            else {}
        )
        for variant_name, payload in variants.items():
            payload["ladder_rung"] = next(
                (
                    rung
                    for rung, canonical_variant in LADDER_PRIMARY_VARIANT_BY_RUNG.items()
                    if canonical_variant == str(variant_name)
                ),
                None,
            )
        reference_variant = (
            "modular_full"
            if "modular_full" in variants
            else str(ablations.get("reference_variant") or "")
        )
        if reference_variant and reference_variant in variants:
            reference_payload = variants[reference_variant]
            if not isinstance(reference_payload.get("without_reflex_support"), Mapping):
                limitations.append(
                    f"Reference variant {reference_variant!r} did not include a without_reflex_support block; using the available summary."
                )
        deltas = _ablation_deltas_vs_reference(
            variants,
            reference_variant=reference_variant,
        )
        ladder_rungs = _extract_ladder_rungs(variants)
        ladder_comparison = extract_ladder_comparison(
            variants,
            source="summary.behavior_evaluation.ablations",
        )
        limitations.extend(
            str(item) for item in ladder_comparison.get("limitations", []) if item
        )
        predator_type_comparisons = compare_predator_type_ablation_performance(
            {
                "variants": variants,
                "deltas_vs_reference": deltas,
            }
        )
        return {
            "available": bool(variants),
            "source": "summary.behavior_evaluation.ablations",
            "reference_variant": reference_variant,
            "variants": variants,
            "deltas_vs_reference": deltas,
            "ladder_rungs": ladder_rungs,
            "ladder_comparison": ladder_comparison,
            "ladder": {
                "available": bool(ladder_rungs.get("recognized_rungs")),
                "active_rungs": list(LADDER_ACTIVE_RUNGS),
                "rungs": {
                    rung: {
                        "rung": rung,
                        "protocol_name": LADDER_PROTOCOL_NAMES.get(rung, rung),
                        "technical_variant": LADDER_PRIMARY_VARIANT_BY_RUNG.get(rung, ""),
                        "description": LADDER_RUNG_DESCRIPTIONS.get(rung, ""),
                        "summary": dict(
                            _mapping_or_empty(
                                variants.get(
                                    LADDER_PRIMARY_VARIANT_BY_RUNG.get(rung, ""),
                                    {},
                                )
                            ).get("summary", {})
                        ),
                    }
                    for rung in ladder_rungs.get("recognized_rungs", [])
                },
                "adjacent_comparisons": [
                    {
                        "baseline_rung": item.get("baseline_rung"),
                        "comparison_rung": item.get("comparison_rung"),
                        "baseline_variant": _mapping_or_empty(item.get("metrics")).get(
                            "baseline_variant"
                        ),
                        "comparison_variant": _mapping_or_empty(item.get("metrics")).get(
                            "comparison_variant"
                        ),
                        "summary": dict(_mapping_or_empty(item.get("summary"))),
                        "scenarios": dict(_mapping_or_empty(item.get("scenarios"))),
                        "source": item.get("source"),
                    }
                    for item in ladder_comparison.get("comparisons", [])
                    if isinstance(item, Mapping)
                ],
                "limitations": list(ladder_comparison.get("limitations", [])),
            },
            "predator_type_comparisons": predator_type_comparisons,
            "limitations": limitations,
        }

    filtered_rows = _rows_with_minimal_reflex_support(behavior_rows)
    by_variant: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in filtered_rows:
        variant = str(row.get("ablation_variant") or "")
        if variant:
            by_variant[variant].append(row)
    variants: dict[str, object] = {}
    for variant, items in sorted(by_variant.items()):
        success_values = [1.0 if _coerce_bool(row.get("success")) else 0.0 for row in items]
        by_scenario: dict[str, list[Mapping[str, object]]] = defaultdict(list)
        for row in items:
            by_scenario[str(row.get("scenario") or "")].append(row)
        suite = {
            scenario_name: {
                "scenario": scenario_name,
                "episodes": len(scenario_items),
                "success_rate": _mean(
                    1.0 if _coerce_bool(row.get("success")) else 0.0
                    for row in scenario_items
                ),
            }
            for scenario_name, scenario_items in sorted(by_scenario.items())
            if scenario_name
        }
        variants[variant] = {
            "config": {
                "architecture": str(items[0].get("ablation_architecture") or ""),
                "architecture_description": str(
                    items[0].get("ablation_architecture_description") or ""
                ),
            },
            "summary": {
                "scenario_success_rate": _mean(
                    scenario_payload["success_rate"]
                    for scenario_payload in suite.values()
                    if isinstance(scenario_payload, Mapping)
                ),
                "episode_success_rate": _mean(success_values),
                "eval_reflex_scale": min(
                    _coerce_float(row.get("eval_reflex_scale"), 0.0)
                    for row in items
                ),
                "architecture_description": str(
                    items[0].get("ablation_architecture_description") or ""
                ),
            },
            "suite": suite,
            "legacy_scenarios": dict(suite),
            "ladder_rung": next(
                (
                    rung
                    for rung, canonical_variant in LADDER_PRIMARY_VARIANT_BY_RUNG.items()
                    if canonical_variant == variant
                ),
                None,
            ),
        }
    if variants:
        reference_variant = "modular_full" if "modular_full" in variants else ""
        limitations.append(
            "Ablation comparisons were reconstructed from behavior_csv using the lowest eval_reflex_scale rows per variant."
        )
    else:
        reference_variant = ""
        limitations.append("No ablation comparison data was available.")
    deltas = _ablation_deltas_vs_reference(
        variants,
        reference_variant=reference_variant,
    )
    ladder_rungs = _extract_ladder_rungs(variants)
    ladder_comparison = extract_ladder_comparison(
        variants,
        source="behavior_csv",
    )
    limitations.extend(
        str(item) for item in ladder_comparison.get("limitations", []) if item
    )
    return {
        "available": bool(variants),
        "source": "behavior_csv",
        "reference_variant": reference_variant,
        "variants": variants,
        "deltas_vs_reference": deltas,
        "ladder_rungs": ladder_rungs,
        "ladder_comparison": ladder_comparison,
        "ladder": {
            "available": bool(ladder_rungs.get("recognized_rungs")),
            "active_rungs": list(LADDER_ACTIVE_RUNGS),
            "rungs": {
                rung: {
                    "rung": rung,
                    "protocol_name": LADDER_PROTOCOL_NAMES.get(rung, rung),
                    "technical_variant": LADDER_PRIMARY_VARIANT_BY_RUNG.get(rung, ""),
                    "description": LADDER_RUNG_DESCRIPTIONS.get(rung, ""),
                    "summary": dict(
                        _mapping_or_empty(
                            variants.get(
                                LADDER_PRIMARY_VARIANT_BY_RUNG.get(rung, ""),
                                {},
                            )
                        ).get("summary", {})
                    ),
                }
                for rung in ladder_rungs.get("recognized_rungs", [])
            },
            "adjacent_comparisons": [
                {
                    "baseline_rung": item.get("baseline_rung"),
                    "comparison_rung": item.get("comparison_rung"),
                    "baseline_variant": _mapping_or_empty(item.get("metrics")).get(
                        "baseline_variant"
                    ),
                    "comparison_variant": _mapping_or_empty(item.get("metrics")).get(
                        "comparison_variant"
                    ),
                    "summary": dict(_mapping_or_empty(item.get("summary"))),
                    "scenarios": dict(_mapping_or_empty(item.get("scenarios"))),
                    "source": item.get("source"),
                }
                for item in ladder_comparison.get("comparisons", [])
                if isinstance(item, Mapping)
            ],
            "limitations": list(ladder_comparison.get("limitations", [])),
        },
        "predator_type_comparisons": compare_predator_type_ablation_performance(
            {
                "variants": variants,
                "deltas_vs_reference": deltas,
            }
        ),
        "limitations": limitations,
    }

def extract_reflex_frequency(trace: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """
    Summarizes per-module reflex activity observed in a trace of runtime items.
    
    Analyzes a sequence of trace items to count reflex-related events per module using two sources:
    1) messages with topic "action.proposal" whose payload contains a `reflex` entry (counts as message-based reflex events), and
    2) debug reflex entries under each item's `debug["reflexes"]` mapping (counts, override occurrences, and dominance values).
    For each discovered module returns aggregated event counts, events-per-tick, the debug override rate, and the mean dominance value. If no debug reflex payloads are present a limitation note is included.
    
    Parameters:
        trace (Sequence[Mapping[str, object]]): Ordered trace items where each item may include a `messages` list (message mappings with keys like `topic`, `sender`, `payload`) and/or a `debug` mapping containing a `reflexes` mapping per module.
    
    Returns:
        dict[str, object]: A dictionary with the following keys:
            - "available": `true` if any module rows were produced, `false` otherwise.
            - "tick_count": Number of trace items processed.
            - "modules": List of per-module dictionaries with keys:
                - "module": Module name (string).
                - "reflex_events": Count of reflex events (uses message-based count if available, otherwise debug-based).
                - "events_per_tick": `reflex_events` divided by `tick_count`.
                - "message_reflex_events": Count of reflex messages observed.
                - "debug_reflex_events": Count of reflex entries observed in debug payloads.
                - "override_rate": Fraction of debug reflex entries that indicated a module reflex override (0.0-1.0).
                - "mean_dominance": Average `module_reflex_dominance` over considered debug entries.
            - "uses_debug_reflexes": `true` if any debug reflex payloads were observed, `false` otherwise.
            - "limitations": List of human-readable limitation strings describing any data gaps (for example, missing trace or missing debug reflex payloads).
    """
    if not trace:
        return {
            "available": False,
            "tick_count": 0,
            "modules": [],
            "uses_debug_reflexes": False,
            "limitations": ["No trace file was provided."],
        }

    message_counts: Counter[str] = Counter()
    debug_counts: Counter[str] = Counter()
    debug_override_counts: Counter[str] = Counter()
    debug_dominance_sum: defaultdict[str, float] = defaultdict(float)
    module_names: set[str] = set(DEFAULT_MODULE_NAMES)
    uses_debug_reflexes = False

    for item in trace:
        for message in item.get("messages", []) if isinstance(item.get("messages"), list) else []:
            if not isinstance(message, Mapping):
                continue
            payload = message.get("payload")
            sender = str(message.get("sender") or "")
            if sender:
                module_names.add(sender)
            if (
                isinstance(payload, Mapping)
                and payload.get("reflex")
                and sender
                and str(message.get("topic") or "") == "action.proposal"
            ):
                message_counts[sender] += 1
        debug = item.get("debug")
        reflexes = debug.get("reflexes") if isinstance(debug, Mapping) else None
        if isinstance(reflexes, Mapping):
            uses_debug_reflexes = True
            for module_name, payload in reflexes.items():
                module_name = str(module_name)
                module_names.add(module_name)
                if not isinstance(payload, Mapping):
                    continue
                raw_reflex = payload.get("reflex")
                reflex = (
                    bool(raw_reflex)
                    if isinstance(raw_reflex, Mapping)
                    else _coerce_bool(raw_reflex)
                )
                if reflex:
                    debug_counts[module_name] += 1
                    if _coerce_bool(payload.get("module_reflex_override")):
                        debug_override_counts[module_name] += 1
                    debug_dominance_sum[module_name] += _coerce_float(
                        payload.get("module_reflex_dominance"),
                        0.0,
                    )

    tick_count = len(trace)
    module_rows: list[dict[str, object]] = []
    for module_name in sorted(module_names):
        message_count = int(message_counts[module_name])
        debug_count = int(debug_counts[module_name])
        reflex_events = message_count if message_count > 0 else debug_count
        debug_denominator = debug_count if debug_count > 0 else reflex_events
        module_rows.append(
            {
                "module": module_name,
                "reflex_events": reflex_events,
                "events_per_tick": _safe_divide(reflex_events, float(tick_count)),
                "message_reflex_events": message_count,
                "debug_reflex_events": debug_count,
                "override_rate": _safe_divide(
                    float(debug_override_counts[module_name]),
                    float(debug_denominator),
                ),
                "mean_dominance": _safe_divide(
                    float(debug_dominance_sum[module_name]),
                    float(debug_denominator),
                ),
            }
        )

    limitations: list[str] = []
    if not uses_debug_reflexes:
        limitations.append(
            "Debug reflex payloads were not present; override_rate and mean_dominance may stay at zero even when reflexes occurred."
        )
    return {
        "available": bool(module_rows),
        "tick_count": tick_count,
        "modules": module_rows,
        "uses_debug_reflexes": uses_debug_reflexes,
        "limitations": limitations,
    }
