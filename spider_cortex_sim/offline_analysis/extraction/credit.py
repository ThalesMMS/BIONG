from __future__ import annotations

from collections.abc import Mapping, Sequence

from ..constants import LADDER_RUNG_MAPPING, MODULAR_CREDIT_RUNGS
from ..utils import _coerce_optional_float, _mapping_or_empty
from .representation import _normalize_float_map
from .shaping import _variant_with_minimal_reflex_support

CREDIT_STRATEGY_KEYS: frozenset[str] = frozenset(
    ("broadcast", "local_only", "counterfactual", "route_mask")
)


def _credit_payload_from_summary_block(
    payload: Mapping[str, object],
) -> dict[str, object] | None:
    """
    Extract credit-related fields from a summary block and return a normalized credit payload.
    
    Parameters:
        payload (Mapping[str, object]): A summary-block mapping that may contain any of the following keys:
            - "mean_module_credit_weights": mapping of module -> numeric credit weight
            - "module_gradient_norm_means": mapping of module -> numeric gradient-norm mean
            - "mean_counterfactual_credit_weights": mapping of module -> numeric counterfactual weight
            - "route_credit_weights": mapping of route/module -> numeric route credit weight
            - "credit_strategy": optional strategy name
    
    Returns:
        dict[str, object] | None: A mapping with keys:
            - "strategy" (str): the credit strategy name (empty string if absent)
            - "weights" (dict[str, float]): normalized module credit weights
            - "gradient_norms" (dict[str, float]): normalized module gradient norms
            - "counterfactual_weights" (dict[str, float]): normalized counterfactual weights
            - "route_credit_weights" (dict[str, float]): normalized route credit weights
        Returns `None` if none of the credit-related maps are present or non-empty.
    """
    weights = _normalize_float_map(payload.get("mean_module_credit_weights"))
    gradient_norms = _normalize_float_map(payload.get("module_gradient_norm_means"))
    counterfactual_weights = _normalize_float_map(
        payload.get("mean_counterfactual_credit_weights")
    )
    route_credit_weights = _normalize_float_map(payload.get("route_credit_weights"))
    if not (weights or gradient_norms or counterfactual_weights or route_credit_weights):
        return None
    return {
        "strategy": str(payload.get("credit_strategy") or ""),
        "weights": weights,
        "gradient_norms": gradient_norms,
        "counterfactual_weights": counterfactual_weights,
        "route_credit_weights": route_credit_weights,
    }


def _effective_module_count(weights: Mapping[str, float]) -> float | None:
    positive = [max(0.0, float(value)) for value in weights.values() if float(value) > 0.0]
    total = sum(positive)
    if total <= 0.0:
        return None
    normalized = [value / total for value in positive]
    squared_total = sum(value * value for value in normalized)
    if squared_total <= 0.0:
        return None
    return float(1.0 / squared_total)


def _max_map_entry(values: Mapping[str, float]) -> tuple[str, float]:
    if not values:
        return "", 0.0
    key = max(values, key=values.get)
    return str(key), float(values.get(key, 0.0))


def _mean_positive(values: Mapping[str, float]) -> float | None:
    positives = [float(value) for value in values.values() if float(value) > 0.0]
    if not positives:
        return None
    return float(sum(positives) / len(positives))


def _interpret_credit_failure(
    variant_credit_data: Mapping[str, object],
    reference_credit_data: Mapping[str, object],
) -> dict[str, object]:
    """
    Heuristically diagnose credit-related failure modes for one strategy.
    """
    strategy = str(variant_credit_data.get("strategy") or "broadcast")
    weights = _normalize_float_map(variant_credit_data.get("weights"))
    gradient_norms = _normalize_float_map(variant_credit_data.get("gradient_norms"))
    counterfactual_weights = _normalize_float_map(
        variant_credit_data.get("counterfactual_weights")
    )
    success_rate = _coerce_optional_float(variant_credit_data.get("scenario_success_rate"))
    reference_success = _coerce_optional_float(
        reference_credit_data.get("scenario_success_rate")
    )
    reference_weights = _normalize_float_map(reference_credit_data.get("weights"))
    findings: list[dict[str, object]] = []
    limitations: list[str] = []

    effective_module_count = _coerce_optional_float(
        variant_credit_data.get("mean_effective_module_count")
    )
    if effective_module_count is None:
        effective_module_count = _effective_module_count(weights)
    reference_effective_module_count = _coerce_optional_float(
        reference_credit_data.get("mean_effective_module_count")
    )
    if reference_effective_module_count is None:
        reference_effective_module_count = _effective_module_count(reference_weights)

    dominant_module, dominant_weight = _max_map_entry(weights)
    mean_gradient_norm = _mean_positive(gradient_norms)

    if strategy == "broadcast":
        active_module_count = sum(1 for value in weights.values() if float(value) > 0.0)
        if (
            effective_module_count is not None
            and effective_module_count >= 3.0
            and dominant_weight <= 0.45
            and mean_gradient_norm is not None
            and mean_gradient_norm <= 1.0
        ):
            findings.append(
                {
                    "pattern": "excessive global credit",
                    "evidence": {
                        "mean_effective_module_count": round(effective_module_count, 6),
                        "active_module_count": active_module_count,
                        "dominant_credit_weight": round(dominant_weight, 6),
                        "mean_gradient_norm": round(mean_gradient_norm, 6),
                    },
                    "interpretation": (
                        "Broadcast credit is spread across many modules while per-module "
                        "gradient norms remain weak, which is consistent with excessive "
                        "global credit dilution."
                    ),
                }
            )
    elif strategy == "local_only":
        total_credit = float(sum(abs(value) for value in weights.values()))
        if reference_success is None:
            limitations.append(
                "Local-only credit could not be compared against a reference success rate."
            )
        elif total_credit <= 1e-6 and success_rate is not None and success_rate <= reference_success - 0.05:
            findings.append(
                {
                    "pattern": "insufficient local credit",
                    "evidence": {
                        "total_credit_weight": round(total_credit, 6),
                        "scenario_success_rate": round(success_rate, 6),
                        "reference_scenario_success_rate": round(reference_success, 6),
                    },
                    "interpretation": (
                        "Local-only credit removes global policy-gradient contribution and "
                        "the run also underperforms its broadcast reference, which is "
                        "consistent with insufficient local credit."
                    ),
                }
            )
    elif strategy == "counterfactual":
        dominant_counterfactual_module, dominant_counterfactual_weight = _max_map_entry(
            counterfactual_weights
        )
        if reference_success is None:
            limitations.append(
                "Counterfactual credit could not be compared against a reference success rate."
            )
        elif (
            success_rate is not None
            and success_rate >= reference_success + 0.05
            and (
                (
                    effective_module_count is not None
                    and reference_effective_module_count is not None
                    and effective_module_count <= reference_effective_module_count - 0.1
                )
                or dominant_counterfactual_weight >= 0.5
            )
        ):
            findings.append(
                {
                    "pattern": "counterfactual improvement",
                    "evidence": {
                        "scenario_success_rate": round(success_rate, 6),
                        "reference_scenario_success_rate": round(reference_success, 6),
                        "dominant_counterfactual_module": dominant_counterfactual_module,
                        "dominant_counterfactual_weight": round(
                            dominant_counterfactual_weight,
                            6,
                        ),
                        "mean_effective_module_count": (
                            round(effective_module_count, 6)
                            if effective_module_count is not None
                            else None
                        ),
                    },
                    "interpretation": (
                        "Counterfactual credit concentrates weight on the modules that most "
                        "changed the chosen action and also improves success relative to "
                        "broadcast, which is consistent with a counterfactual credit gain."
                    ),
                }
            )

    if not findings and not limitations:
        limitations.append(
            f"No strong credit-assignment diagnostic pattern was detected for strategy {strategy!r}."
        )
    return {
        "strategy": strategy,
        "findings": findings,
        "limitations": limitations,
        "dominant_module": dominant_module,
        "dominant_weight": dominant_weight,
        "mean_effective_module_count": effective_module_count,
        "mean_gradient_norm": mean_gradient_norm,
    }


def _compare_credit_across_architectures(
    variants_by_rung: Mapping[str, Mapping[str, Mapping[str, object]]]
    | Mapping[str, Mapping[str, object]],
    last_rung_variants: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, object]:
    """
    Compare strategy behavior across the modular architecture ladder.
    """
    findings: list[dict[str, object]] = []
    limitations: list[str] = []

    def _mapping_keys(payload: Mapping[str, object]) -> set[str]:
        return {str(key) for key in payload.keys()}

    def _normalize_strategy_mapping(
        payload: Mapping[str, object],
        *,
        label: str,
    ) -> dict[str, Mapping[str, object]]:
        keys = _mapping_keys(payload)
        invalid_keys = sorted(key for key in keys if key not in CREDIT_STRATEGY_KEYS)
        if invalid_keys:
            raise ValueError(
                f"{label} must be a per-strategy mapping with keys drawn from "
                f"{sorted(CREDIT_STRATEGY_KEYS)}; got invalid keys {invalid_keys}."
            )
        normalized: dict[str, Mapping[str, object]] = {}
        for key, value in payload.items():
            strategy = str(key)
            if not isinstance(value, Mapping):
                raise ValueError(
                    f"{label}[{strategy!r}] must be a mapping of credit metrics."
                )
            normalized[strategy] = value
        return normalized

    def _normalize_rung_mapping(
        payload: Mapping[str, object],
    ) -> dict[str, Mapping[str, Mapping[str, object]]]:
        keys = _mapping_keys(payload)
        invalid_keys = sorted(key for key in keys if key not in MODULAR_CREDIT_RUNGS)
        if invalid_keys:
            if keys and keys.issubset(CREDIT_STRATEGY_KEYS):
                raise ValueError(
                    "variants_by_rung looks like a single-rung strategy mapping. "
                    "Pass a rung-keyed mapping or provide last_rung_variants for the "
                    "legacy two-argument form."
                )
            raise ValueError(
                "variants_by_rung must be keyed by modular rungs "
                f"{list(MODULAR_CREDIT_RUNGS)}; got invalid keys {invalid_keys}."
            )
        normalized: dict[str, Mapping[str, Mapping[str, object]]] = {}
        for key, value in payload.items():
            rung = str(key)
            if not isinstance(value, Mapping):
                raise ValueError(
                    f"variants_by_rung[{rung!r}] must be a mapping of strategies."
                )
            normalized[rung] = _normalize_strategy_mapping(
                value,
                label=f"variants_by_rung[{rung!r}]",
            )
        return normalized

    if last_rung_variants is not None:
        normalized_variants_by_rung = {
            MODULAR_CREDIT_RUNGS[0]: _normalize_strategy_mapping(
                variants_by_rung,
                label="variants_by_rung",
            ),
            MODULAR_CREDIT_RUNGS[-1]: _normalize_strategy_mapping(
                last_rung_variants,
                label="last_rung_variants",
            ),
        }
    else:
        normalized_variants_by_rung = _normalize_rung_mapping(variants_by_rung)

    def _success_delta(
        variants: Mapping[str, Mapping[str, object]],
        strategy: str,
    ) -> float | None:
        baseline = variants.get("broadcast")
        comparison = variants.get(strategy)
        if baseline is None or comparison is None:
            return None
        baseline_success = _coerce_optional_float(baseline.get("scenario_success_rate"))
        comparison_success = _coerce_optional_float(
            comparison.get("scenario_success_rate")
        )
        if baseline_success is None or comparison_success is None:
            return None
        return float(comparison_success - baseline_success)

    first_rung = MODULAR_CREDIT_RUNGS[0]
    last_rung = MODULAR_CREDIT_RUNGS[-1]
    first_variants = normalized_variants_by_rung.get(first_rung, {})
    last_variants = normalized_variants_by_rung.get(last_rung, {})

    first_local_delta = _success_delta(first_variants, "local_only")
    last_local_delta = _success_delta(last_variants, "local_only")
    if first_local_delta is None or last_local_delta is None:
        limitations.append(
            "Local-only architecture comparison was incomplete because an endpoint rung lacked a broadcast/local_only pair."
        )
    elif last_local_delta <= first_local_delta - 0.05:
        findings.append(
            {
                "pattern": "local_only differential failure",
                "evidence": {
                    f"{first_rung.lower()}_local_only_delta_vs_broadcast": round(
                        first_local_delta,
                        6,
                    ),
                    f"{last_rung.lower()}_local_only_delta_vs_broadcast": round(
                        last_local_delta,
                        6,
                    ),
                },
                "interpretation": (
                    f"`local_only` degrades more in {last_rung} than {first_rung}, which suggests that "
                    "purely local credit becomes less adequate as modularity increases."
                ),
            }
        )

    first_counterfactual_delta = _success_delta(first_variants, "counterfactual")
    last_counterfactual_delta = _success_delta(last_variants, "counterfactual")
    if first_counterfactual_delta is None or last_counterfactual_delta is None:
        limitations.append(
            "Counterfactual architecture comparison was incomplete because an endpoint rung lacked a broadcast/counterfactual pair."
        )
    elif last_counterfactual_delta >= first_counterfactual_delta + 0.05:
        findings.append(
            {
                "pattern": "counterfactual benefit scales with module count",
                "evidence": {
                    f"{first_rung.lower()}_counterfactual_delta_vs_broadcast": round(
                        first_counterfactual_delta,
                        6,
                    ),
                    f"{last_rung.lower()}_counterfactual_delta_vs_broadcast": round(
                        last_counterfactual_delta,
                        6,
                    ),
                },
                "interpretation": (
                    f"`counterfactual` gains are larger in {last_rung} than {first_rung}, which is "
                    "consistent with counterfactual credit becoming more useful when "
                    "there are more modules competing for credit."
                ),
            }
        )

    return {
        "findings": findings,
        "limitations": limitations,
        "comparisons": {
            rung: {
                "local_only_delta_vs_broadcast": _success_delta(
                    normalized_variants_by_rung.get(rung, {}),
                    "local_only",
                ),
                "counterfactual_delta_vs_broadcast": _success_delta(
                    normalized_variants_by_rung.get(rung, {}),
                    "counterfactual",
                ),
            }
            for rung in MODULAR_CREDIT_RUNGS
        },
    }


def extract_credit_metrics(
    summary: Mapping[str, object],
    behavior_rows: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    """
    Extract per-variant credit metrics from summary blocks.

    The preferred source is `summary.behavior_evaluation.ablations.variants.*.summary`.
    When ablation variants are unavailable, the function falls back to the current-run
    evaluation/training blocks using the configured brain name (or `current_run`).
    `behavior_rows` is accepted for API consistency but is not currently used because
    credit-weight metrics are not exported in the behavior CSV.
    """
    del behavior_rows

    extracted: dict[str, dict[str, object]] = {}
    behavior_evaluation = _mapping_or_empty(summary.get("behavior_evaluation"))
    ablations = _mapping_or_empty(behavior_evaluation.get("ablations"))
    variants = _mapping_or_empty(ablations.get("variants"))
    for variant_name, payload in variants.items():
        if not isinstance(payload, Mapping):
            continue
        normalized_payload = _variant_with_minimal_reflex_support(payload)
        summary_payload = _mapping_or_empty(normalized_payload.get("summary"))
        config_payload = _mapping_or_empty(normalized_payload.get("config"))
        credit_payload = _credit_payload_from_summary_block(summary_payload)
        if credit_payload is None:
            continue
        if not str(credit_payload.get("strategy") or ""):
            credit_payload["strategy"] = str(
                config_payload.get("credit_strategy") or "broadcast"
            )
        credit_payload["scenario_success_rate"] = _coerce_optional_float(
            summary_payload.get("scenario_success_rate")
        )
        credit_payload["mean_effective_module_count"] = _coerce_optional_float(
            summary_payload.get("mean_effective_module_count")
        )
        credit_payload["architecture_rung"] = str(LADDER_RUNG_MAPPING.get(str(variant_name)) or "")
        credit_payload["source"] = (
            f"summary.behavior_evaluation.ablations.variants.{variant_name}.summary"
        )
        extracted[str(variant_name)] = credit_payload

    if extracted:
        return extracted

    config = _mapping_or_empty(summary.get("config"))
    brain_config = _mapping_or_empty(config.get("brain"))
    variant_name = str(
        brain_config.get("name")
        or config.get("name")
        or "current_run"
    )
    summary_blocks = [
        _mapping_or_empty(summary.get("evaluation_without_reflex_support")).get("summary"),
        summary.get("evaluation"),
        behavior_evaluation.get("summary"),
        summary.get("training"),
    ]
    config_credit_strategy = str(
        brain_config.get("credit_strategy")
        or config.get("credit_strategy")
        or "broadcast"
    )
    for summary_block in summary_blocks:
        payload = _mapping_or_empty(summary_block)
        credit_payload = _credit_payload_from_summary_block(payload)
        if credit_payload is None:
            continue
        if not str(credit_payload.get("strategy") or ""):
            credit_payload["strategy"] = config_credit_strategy
        credit_payload["scenario_success_rate"] = _coerce_optional_float(
            payload.get("scenario_success_rate")
        )
        credit_payload["mean_effective_module_count"] = _coerce_optional_float(
            payload.get("mean_effective_module_count")
        )
        credit_payload["architecture_rung"] = str(
            LADDER_RUNG_MAPPING.get(variant_name) or ""
        )
        credit_payload["source"] = "summary"
        extracted[variant_name] = credit_payload
        break
    return extracted
