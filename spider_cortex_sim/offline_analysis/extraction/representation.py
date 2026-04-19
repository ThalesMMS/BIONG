from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence

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

def _normalize_module_response_by_predator_type(
    value: object,
) -> dict[str, dict[str, float]]:
    """
    Normalize a nested predator-type → module response mapping into a canonical string-keyed mapping with finite float values.
    
    Converts inputs like {predator_type: {module_name: value, ...}, ...} into {str(predator_type): {str(module_name): float, ...}, ...}. Non-mapping inputs return an empty dict; predator entries that are not mappings are skipped. Non-finite or non-numeric module values are coerced to finite floats (defaults to 0.0).
    
    Parameters:
        value (object): Expected mapping from predator type to a mapping of module names to numeric-like values; other shapes are tolerated and skipped.
    
    Returns:
        dict[str, dict[str, float]]: A mapping of predator type and module name (both stringified) to coerced finite float values.
    """
    if not isinstance(value, Mapping):
        return {}
    normalized: dict[str, dict[str, float]] = {}
    for predator_type, payload in value.items():
        if not isinstance(payload, Mapping):
            continue
        normalized[str(predator_type)] = {
            str(module_name): _coerce_float(module_value)
            for module_name, module_value in payload.items()
        }
    return normalized

def _normalize_float_map(value: object) -> dict[str, float]:
    """
    Normalize a mapping-like input into a dict of string keys to finite floats.
    
    Parameters:
        value (object): A mapping whose keys will be stringified and whose values are coerced to finite floats. If `value` is not a mapping, it is treated as absent.
    
    Returns:
        dict[str, float]: A dictionary mapping each key (converted to `str`) to a finite `float`. Keys whose values cannot be coerced are omitted; if input is not a mapping, returns an empty dict.
    """
    if not isinstance(value, Mapping):
        return {}
    return {
        str(name): _coerce_float(item_value)
        for name, item_value in value.items()
    }

def _extract_first_present_float_map(
    payload: Mapping[str, object],
    keys: Sequence[str],
) -> tuple[dict[str, float], bool]:
    """
    Select the first mapping found under the provided alias keys that contains at least one coercible numeric entry and return it as a normalized float map.
    
    Parameters:
        payload (Mapping[str, object]): Mapping to search for alias keys.
        keys (Sequence[str]): Ordered alias keys to check in priority order.
    
    Returns:
        tuple[dict[str, float], bool]: A tuple where the first element is a dict mapping stringified keys to coerced float values (empty if none found), and the second element is `True` if a mapping with at least one coercible value was found, `False` otherwise.
    """
    for key in keys:
        value = payload.get(key)
        if not isinstance(value, Mapping) or len(value) == 0:
            continue
        normalized: dict[str, float] = {}
        for name, item_value in value.items():
            coerced = _coerce_optional_float(item_value)
            if coerced is None:
                continue
            normalized[str(name)] = coerced
        if normalized:
            return normalized, True
    return {}, False

def _extract_first_present_float(
    payload: Mapping[str, object],
    keys: Sequence[str],
) -> tuple[float, bool]:
    """
    Find the first key in `keys` whose value can be coerced to a finite float.
    
    Returns:
        tuple[float, bool]: The coerced finite float and `True` if a valid value was found; `0.0, False` otherwise.
    """
    for key in keys:
        value = payload.get(key)
        if value is None:
            continue
        coerced = _coerce_optional_float(value)
        if coerced is None:
            continue
        return coerced, True
    return 0.0, False

def _representation_interpretation(
    score: float,
    *,
    insufficient_data: bool = False,
) -> str:
    """
    Map a representation specialization score to a canonical interpretation label.
    
    Parameters:
        insufficient_data (bool): When True, return "insufficient_data" regardless of `score`.
    
    Returns:
        str: `'high'` if `score` >= 0.50, `'moderate'` if `score` >= 0.20, `'low'` otherwise; returns `'insufficient_data'` when `insufficient_data` is True.
    """
    if insufficient_data:
        return "insufficient_data"
    if score >= 0.50:
        return "high"
    if score >= 0.20:
        return "moderate"
    return "low"

def _representation_payload(
    payload: Mapping[str, object],
) -> dict[str, object]:
    """
    Normalize and canonicalize representation-specialization fields from a summary payload.
    
    Parameters:
        payload (Mapping[str, object]): A mapping potentially containing representation-specialization metrics under multiple alias keys
            (e.g., proposer divergence, action-center differentials, or representation specialization score).
    
    Returns:
        dict[str, object]: A dictionary with normalized fields:
            - "proposer_divergence" (dict[str, float]): Module -> coerced float proposer divergence values (empty if absent).
            - "action_center_gate_differential" (dict[str, float]): Module -> coerced float gate differential values (empty if absent).
            - "action_center_contribution_differential" (dict[str, float]): Module -> coerced float contribution differential values (empty if absent).
            - "representation_specialization_score" (float | None): Bounded score in [0.0, 1.0] if available or derivable from proposer divergence; otherwise None.
            - "has_any_field" (bool): True when any of the underlying fields or aliases were present and yielded data.
    """
    proposer_divergence, has_proposer_divergence = _extract_first_present_float_map(
        payload,
        (
            "mean_proposer_divergence_by_module",
            "proposer_divergence_by_module",
        ),
    )
    action_center_gate_differential, has_gate = _extract_first_present_float_map(
        payload,
        (
            "mean_action_center_gate_differential",
            "action_center_gate_differential",
        ),
    )
    action_center_contribution_differential, has_contribution = (
        _extract_first_present_float_map(
            payload,
            (
                "mean_action_center_contribution_differential",
                "action_center_contribution_differential",
            ),
        )
    )
    representation_specialization_score, has_score = _extract_first_present_float(
        payload,
        (
            "mean_representation_specialization_score",
            "representation_specialization_score",
        ),
    )
    score_value: float | None = None
    if has_score:
        score_value = representation_specialization_score
    elif proposer_divergence:
        score_value = _mean(proposer_divergence.values())
    if score_value is not None:
        score_value = max(
            0.0,
            min(1.0, score_value),
        )
    return {
        "proposer_divergence": proposer_divergence,
        "action_center_gate_differential": action_center_gate_differential,
        "action_center_contribution_differential": (
            action_center_contribution_differential
        ),
        "representation_specialization_score": score_value,
        "has_any_field": any(
            (
                has_proposer_divergence,
                has_gate,
                has_contribution,
                has_score,
            )
        ),
    }

def _aggregate_representation_from_scenarios(
    scenarios: Mapping[str, object],
) -> dict[str, object]:
    """
    Aggregate representation-specialization metrics across scenario payloads into per-module means and a bounded overall score.
    
    Parameters:
        scenarios (Mapping[str, object]): Mapping of scenario identifiers to payload objects; payloads that are not mappings are ignored. The function also checks for a legacy `legacy_metrics` mapping inside each payload when primary fields are absent.
    
    Returns:
        dict[str, object]: A normalized summary with the following keys:
            - "proposer_divergence" (dict[str, float]): Mean proposer-divergence per module across scenarios.
            - "action_center_gate_differential" (dict[str, float]): Mean gate-differential per module across scenarios.
            - "action_center_contribution_differential" (dict[str, float]): Mean contribution-differential per module across scenarios.
            - "representation_specialization_score" (float): Overall specialization score clamped to the range [0.0, 1.0]. This is the mean of per-scenario scores when available; otherwise, if no per-scenario scores exist but proposer-divergence values are present, it is the mean of proposer-divergence values.
            - "has_any_field" (bool): True if at least one scenario contributed any representation-specialization field, False otherwise.
    """
    proposer_divergence_values: dict[str, list[float]] = defaultdict(list)
    gate_differential_values: dict[str, list[float]] = defaultdict(list)
    contribution_differential_values: dict[str, list[float]] = defaultdict(list)
    scores: list[float] = []
    has_any_field = False

    for payload in scenarios.values():
        if not isinstance(payload, Mapping):
            continue
        representation = _representation_payload(payload)
        if not bool(representation.get("has_any_field")):
            legacy_metrics = payload.get("legacy_metrics", {})
            if isinstance(legacy_metrics, Mapping):
                representation = _representation_payload(legacy_metrics)
        if not bool(representation.get("has_any_field")):
            continue
        has_any_field = True
        for module_name, value in _normalize_float_map(
            representation.get("proposer_divergence")
        ).items():
            proposer_divergence_values[module_name].append(value)
        for module_name, value in _normalize_float_map(
            representation.get("action_center_gate_differential")
        ).items():
            gate_differential_values[module_name].append(value)
        for module_name, value in _normalize_float_map(
            representation.get("action_center_contribution_differential")
        ).items():
            contribution_differential_values[module_name].append(value)
        score_value = representation.get("representation_specialization_score")
        if score_value is not None:
            scores.append(_coerce_float(score_value))

    proposer_divergence = {
        module_name: _mean(values)
        for module_name, values in sorted(proposer_divergence_values.items())
    }
    action_center_gate_differential = {
        module_name: _mean(values)
        for module_name, values in sorted(gate_differential_values.items())
    }
    action_center_contribution_differential = {
        module_name: _mean(values)
        for module_name, values in sorted(contribution_differential_values.items())
    }
    representation_specialization_score = _mean(scores)
    if not scores and proposer_divergence:
        representation_specialization_score = _mean(proposer_divergence.values())
    return {
        "proposer_divergence": proposer_divergence,
        "action_center_gate_differential": action_center_gate_differential,
        "action_center_contribution_differential": (
            action_center_contribution_differential
        ),
        "representation_specialization_score": max(
            0.0,
            min(1.0, representation_specialization_score),
        ),
        "has_any_field": has_any_field,
    }

def _aggregate_specialization_from_scenarios(
    scenarios: Mapping[str, object],
) -> dict[str, dict[str, float]]:
    """
    Aggregate module response values across scenarios by predator type and compute per-module means.
    
    Parameters:
        scenarios (Mapping[str, object]): Mapping of scenario identifiers to payloads. Each payload may include
            `mean_module_response_by_predator_type` or `module_response_by_predator_type`, a mapping from
            predator type to module-to-value mappings. Non-mapping payloads are ignored.
    
    Returns:
        dict[str, dict[str, float]]: Mapping from predator type to a mapping of module name to the arithmetic
        mean of that module's response values across all scenarios (0.0 for modules present but with no finite values).
    """
    per_type: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for payload in scenarios.values():
        if not isinstance(payload, Mapping):
            continue
        response = _normalize_module_response_by_predator_type(
            payload.get("mean_module_response_by_predator_type")
            or payload.get("module_response_by_predator_type")
        )
        for predator_type, module_values in response.items():
            for module_name, module_value in module_values.items():
                per_type[predator_type][module_name].append(_coerce_float(module_value))
    return {
        predator_type: {
            module_name: _mean(values)
            for module_name, values in module_values.items()
        }
        for predator_type, module_values in per_type.items()
    }

def extract_predator_type_specialization(
    summary: Mapping[str, object],
    behavior_rows: Sequence[Mapping[str, object]],
    trace: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """
    Compute predator-type module specialization and related metrics from per-type module response data.
    
    When available, selects the first source that provides per-predator-type module response values and computes:
    - per-predator-type module distributions and dominant module,
    - differential activation for visual cortex (visual minus olfactory) and sensory cortex (olfactory minus visual),
    - a bounded specialization score in [0.0, 1.0],
    - a type-module correlation proxy and an interpretation bucket.
    
    Returns:
        dict: Analysis payload with keys:
            - available (bool): True when usable per-type metrics were found.
            - source (str): The chosen data source name or "none".
            - predator_types (dict): Per-type entries for "visual" and "olfactory", each containing:
                - module_distribution (dict[str, float]): Raw module response values.
                - visual_cortex_activation (float)
                - sensory_cortex_activation (float)
                - dominant_module (str)
            - differential_activation (dict): {
                "visual_cortex_visual_minus_olfactory": float,
                "sensory_cortex_olfactory_minus_visual": float
              }
            - type_module_correlation (float): Correlation proxy in [0.0, 1.0].
            - specialization_score (float): Bounded score in [0.0, 1.0].
            - interpretation (str): One of "high", "moderate", "low", "insufficient_data", or "unavailable".
            - limitations (list[str]): Notes about missing or reconstructed data when unavailable.
    """
    del behavior_rows
    del trace

    candidate_sources: list[tuple[str, dict[str, dict[str, float]]]] = []
    evaluation = _mapping_or_empty(summary.get("evaluation"))
    candidate_sources.append(
        (
            "summary.evaluation",
            _normalize_module_response_by_predator_type(
                evaluation.get("mean_module_response_by_predator_type")
                or evaluation.get("module_response_by_predator_type")
            ),
        )
    )
    evaluation_without_reflex = _mapping_or_empty(
        _mapping_or_empty(summary.get("evaluation_without_reflex_support")).get("summary")
    )
    candidate_sources.append(
        (
            "summary.evaluation_without_reflex_support.summary",
            _normalize_module_response_by_predator_type(
                evaluation_without_reflex.get("mean_module_response_by_predator_type")
                or evaluation_without_reflex.get("module_response_by_predator_type")
            ),
        )
    )
    behavior_evaluation = _mapping_or_empty(summary.get("behavior_evaluation"))
    candidate_sources.append(
        (
            "summary.behavior_evaluation.suite",
            _aggregate_specialization_from_scenarios(
                _mapping_or_empty(behavior_evaluation.get("suite"))
            ),
        )
    )
    candidate_sources.append(
        (
            "summary.behavior_evaluation.legacy_scenarios",
            _aggregate_specialization_from_scenarios(
                _mapping_or_empty(behavior_evaluation.get("legacy_scenarios"))
            ),
        )
    )

    source_name = "none"
    module_response_by_predator_type: dict[str, dict[str, float]] = {}
    fallback_source_name = "none"
    fallback_candidate: dict[str, dict[str, float]] = {}
    for candidate_name, candidate in candidate_sources:
        has_any_data = any(sum(module_values.values()) > 0.0 for module_values in candidate.values())
        if not has_any_data:
            continue
        visual_modules = candidate.get("visual", {})
        olfactory_modules = candidate.get("olfactory", {})
        has_visual_data = any(_coerce_float(value) > 0.0 for value in visual_modules.values())
        has_olfactory_data = any(
            _coerce_float(value) > 0.0 for value in olfactory_modules.values()
        )
        if has_visual_data and has_olfactory_data:
            source_name = candidate_name
            module_response_by_predator_type = candidate
            break
        if not fallback_candidate:
            fallback_source_name = candidate_name
            fallback_candidate = candidate

    if not module_response_by_predator_type and fallback_candidate:
        source_name = fallback_source_name
        module_response_by_predator_type = fallback_candidate

    if not module_response_by_predator_type:
        return {
            "available": False,
            "source": "none",
            "predator_types": {},
            "differential_activation": {},
            "type_module_correlation": 0.0,
            "specialization_score": 0.0,
            "interpretation": "unavailable",
            "limitations": [
                "No per-predator-type module-response metrics were available.",
            ],
        }

    visual_modules = module_response_by_predator_type.get("visual", {})
    olfactory_modules = module_response_by_predator_type.get("olfactory", {})
    visual_visual = _coerce_float(visual_modules.get("visual_cortex"))
    visual_sensory = _coerce_float(visual_modules.get("sensory_cortex"))
    olfactory_visual = _coerce_float(olfactory_modules.get("visual_cortex"))
    olfactory_sensory = _coerce_float(olfactory_modules.get("sensory_cortex"))
    limitations: list[str] = []
    has_visual_data = any(
        _coerce_float(value) > 0.0 for value in visual_modules.values()
    )
    has_olfactory_data = any(
        _coerce_float(value) > 0.0 for value in olfactory_modules.values()
    )

    if has_visual_data and has_olfactory_data:
        visual_dominant = _dominant_module_by_score(visual_modules)
        olfactory_dominant = _dominant_module_by_score(olfactory_modules)
        visual_cortex_differential = round(visual_visual - olfactory_visual, 6)
        sensory_cortex_differential = round(olfactory_sensory - visual_sensory, 6)
        specialization_score = max(
            0.0,
            min(
                1.0,
                (
                    max(0.0, visual_cortex_differential)
                    + max(0.0, sensory_cortex_differential)
                ) / 2.0,
            ),
        )
        dominant_alignment = _mean(
            [
                1.0 if visual_dominant == "visual_cortex" else 0.0,
                1.0 if olfactory_dominant == "sensory_cortex" else 0.0,
            ]
        )
        type_module_correlation = round(
            max(0.0, min(1.0, (dominant_alignment + specialization_score) / 2.0)),
            6,
        )
        if specialization_score >= 0.50:
            interpretation = "high"
        elif specialization_score >= 0.20:
            interpretation = "moderate"
        else:
            interpretation = "low"
    else:
        visual_dominant = _dominant_module_by_score(visual_modules) if has_visual_data else ""
        olfactory_dominant = (
            _dominant_module_by_score(olfactory_modules) if has_olfactory_data else ""
        )
        visual_cortex_differential = 0.0
        sensory_cortex_differential = 0.0
        specialization_score = 0.0
        type_module_correlation = 0.0
        interpretation = "insufficient_data"
        limitations.append(
            "Predator type specialization requires non-zero module-response data for both visual and olfactory predators."
        )

    return {
        "available": True,
        "source": source_name,
        "predator_types": {
            "visual": {
                "module_distribution": dict(visual_modules),
                "visual_cortex_activation": visual_visual,
                "sensory_cortex_activation": visual_sensory,
                "dominant_module": visual_dominant,
            },
            "olfactory": {
                "module_distribution": dict(olfactory_modules),
                "visual_cortex_activation": olfactory_visual,
                "sensory_cortex_activation": olfactory_sensory,
                "dominant_module": olfactory_dominant,
            },
        },
        "differential_activation": {
            "visual_cortex_visual_minus_olfactory": visual_cortex_differential,
            "sensory_cortex_olfactory_minus_visual": sensory_cortex_differential,
        },
        "type_module_correlation": type_module_correlation,
        "specialization_score": round(specialization_score, 6),
        "interpretation": interpretation,
        "limitations": limitations,
    }

def _build_representation_specialization_result(
    payload: Mapping[str, object],
    source: str,
) -> dict[str, object]:
    """
    Build a standardized representation-specialization result dictionary from a normalized metrics payload.
    
    Parameters:
        payload (Mapping[str, object]): Canonicalized representation-specialization fields (module maps and optional score).
        source (str): Identifier of the data source used to produce the payload (e.g., "summary.evaluation" or "summary.behavior_evaluation.suite").
    
    Returns:
        dict[str, object]: Result containing:
            - available (bool): Always True for constructed results.
            - source (str): The provided source identifier.
            - proposer_divergence (dict[str, float]): Per-module proposer divergence values.
            - action_center_gate_differential (dict[str, float]): Per-module gate differential values.
            - action_center_contribution_differential (dict[str, float]): Per-module contribution differential values.
            - representation_specialization_score (float): Bounded score in [0.0, 1.0], or 0.0 when absent.
            - interpretation (str): One of "high", "moderate", "low", or "insufficient_data".
            - limitations (list[str]): Human-readable notes about aggregation or missing paired evidence when applicable.
    """
    proposer_divergence = _normalize_float_map(payload.get("proposer_divergence"))
    score = _coerce_float(payload.get("representation_specialization_score"))
    insufficient_data = not proposer_divergence and score <= 0.0
    limitations: list[str] = []

    if source == "summary.behavior_evaluation.suite":
        limitations.append(
            "Representation specialization was aggregated from suite-level scenario summaries because summary.evaluation did not expose these metrics."
        )
        if insufficient_data:
            limitations.append(
                "Suite-level representation specialization did not include paired proposer divergence evidence."
            )
    elif insufficient_data:
        limitations.append(
            "Representation specialization metrics were present but did not include paired proposer divergence evidence."
        )

    return {
        "available": True,
        "source": source,
        "proposer_divergence": proposer_divergence,
        "action_center_gate_differential": _normalize_float_map(
            payload.get("action_center_gate_differential")
        ),
        "action_center_contribution_differential": _normalize_float_map(
            payload.get("action_center_contribution_differential")
        ),
        "representation_specialization_score": score,
        "interpretation": _representation_interpretation(
            score,
            insufficient_data=insufficient_data,
        ),
        "limitations": limitations,
    }

def extract_representation_specialization(
    summary: Mapping[str, object],
    behavior_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """
    Extract representation-specialization aggregates from available summary tiers.
    
    Prefers metrics in summary["evaluation"]; if absent, aggregates arithmetic means from summary["behavior_evaluation"]["suite"]. The `behavior_rows` argument is ignored.
    
    Returns:
        A dictionary with the following keys:
        - available (bool): Whether representation-specialization data was found.
        - source (str): Chosen source identifier ("summary.evaluation", "summary.behavior_evaluation.suite", or "none").
        - proposer_divergence (dict[str, float]): Per-module proposer divergence values.
        - action_center_gate_differential (dict[str, float]): Per-module gate differential values.
        - action_center_contribution_differential (dict[str, float]): Per-module contribution differential values.
        - representation_specialization_score (float): Bounded score in [0.0, 1.0].
        - interpretation (str): One of "high", "moderate", "low", or "insufficient_data".
        - limitations (list[str]): Human-readable notes about missing data or fallback behavior.
    """
    del behavior_rows

    evaluation = _mapping_or_empty(summary.get("evaluation"))
    evaluation_payload = _representation_payload(evaluation)
    if bool(evaluation_payload.get("has_any_field")):
        return _build_representation_specialization_result(
            evaluation_payload,
            "summary.evaluation",
        )

    behavior_evaluation = _mapping_or_empty(summary.get("behavior_evaluation"))
    suite_payload = _aggregate_representation_from_scenarios(
        _mapping_or_empty(behavior_evaluation.get("suite"))
    )
    if bool(suite_payload.get("has_any_field")):
        return _build_representation_specialization_result(
            suite_payload,
            "summary.behavior_evaluation.suite",
        )
    fallback_payload = _aggregate_representation_from_scenarios(
        _mapping_or_empty(behavior_evaluation.get("legacy_scenarios"))
    )
    if bool(fallback_payload.get("has_any_field")):
        return _build_representation_specialization_result(
            fallback_payload,
            "summary.behavior_evaluation.legacy_scenarios",
        )

    return {
        "available": False,
        "source": "none",
        "proposer_divergence": {},
        "action_center_gate_differential": {},
        "action_center_contribution_differential": {},
        "representation_specialization_score": 0.0,
        "interpretation": "insufficient_data",
        "limitations": [
            "No representation-specialization aggregates were available in summary.evaluation, summary.behavior_evaluation.suite, or summary.behavior_evaluation.legacy_scenarios.",
        ],
    }
