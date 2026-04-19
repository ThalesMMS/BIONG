from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence

from ...ablations import compare_predator_type_ablation_performance
from ..constants import (
    CANONICAL_NOISE_CONDITIONS,
    DEFAULT_MINIMAL_SHAPING_SURVIVAL_THRESHOLD,
    DEFAULT_MODULE_NAMES,
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
            },
            "suite": suite,
            "legacy_scenarios": dict(suite),
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
    return {
        "available": bool(variants),
        "source": "behavior_csv",
        "reference_variant": reference_variant,
        "variants": variants,
        "deltas_vs_reference": deltas,
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
