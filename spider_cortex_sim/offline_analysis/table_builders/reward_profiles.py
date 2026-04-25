from __future__ import annotations

from .common import *

def build_reward_profile_ladder_tables(
    summary: Mapping[str, object] | None = None,
    *,
    ladder_profile_comparison: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """
    Build four normalized tables and metadata from a cross-profile reward-profile architectural ladder payload.
    
    The function selects ladder data from `ladder_profile_comparison` when provided (must be a Mapping) or extracts it from `summary`. If the ladder is missing or not marked available, the function returns an "unavailable" result with empty row lists, formatted empty tables, and any ladder limitations (or a default limitation message). When available, it produces four row collections by deterministically iterating the ladder's mappings: per-profile variant summaries (`ladder_by_profile`), cross-profile shaping gaps by variant (`shaping_gap_by_variant`), architecture classifications (`architecture_classifications`), and austere survival outcomes by variant (`austere_survival_by_variant`). Numeric fields and confidence-interval bounds are coerced to floats where possible; protocol names and classification labels are normalized; survival thresholds fall back to the module default when absent.
    
    Parameters:
        summary (Mapping[str, object] | None): Source data from which to extract a ladder when
            `ladder_profile_comparison` is not supplied.
        ladder_profile_comparison (Mapping[str, object] | None): Explicit ladder payload to use
            instead of extracting from `summary`. If provided it must be a Mapping and will be used directly.
    
    Returns:
        dict[str, object]: A dictionary with the following keys:
            - "available": `True` if any of the four row lists is non-empty, `False` otherwise.
            - "confidence_level": fixed confidence level constant for the ladder.
            - "rows": mapping containing raw row lists for:
                - "ladder_by_profile"
                - "shaping_gap_by_variant"
                - "architecture_classifications"
                - "austere_survival_by_variant"
            - four rendered table objects with the same names as above (each produced by `_table(headers, rows)`).
            - "limitations": list of limitation messages collected from the ladder and from any empty row sets.
    """
    ladder = (
        dict(ladder_profile_comparison)
        if isinstance(ladder_profile_comparison, Mapping)
        else extract_reward_profile_ladder(summary or {})
    )
    if not ladder.get("available"):
        limitations = list(ladder.get("limitations", []))
        if not limitations:
            limitations.append(
                "No cross-profile architectural ladder comparison was available."
            )
        return {
            "available": False,
            "confidence_level": REWARD_PROFILE_LADDER_CONFIDENCE_LEVEL,
            "rows": {
                "ladder_by_profile": [],
                "shaping_gap_by_variant": [],
                "architecture_classifications": [],
                "austere_survival_by_variant": [],
            },
            "ladder_by_profile": _table(
                (
                    "variant",
                    "protocol_name",
                    "profile",
                    "scenario_success_rate",
                    "scenario_success_rate_ci_lower",
                    "scenario_success_rate_ci_upper",
                    "episode_success_rate",
                    "episode_success_rate_ci_lower",
                    "episode_success_rate_ci_upper",
                    "mean_reward",
                    "mean_reward_ci_lower",
                    "mean_reward_ci_upper",
                    "n_seeds",
                ),
                (),
            ),
            "shaping_gap_by_variant": _table(
                (
                    "variant",
                    "protocol_name",
                    "classic_vs_austere_delta",
                    "classic_vs_austere_ci_lower",
                    "classic_vs_austere_ci_upper",
                    "classic_vs_austere_effect_size",
                    "ecological_vs_austere_delta",
                    "ecological_vs_austere_ci_lower",
                    "ecological_vs_austere_ci_upper",
                    "ecological_vs_austere_effect_size",
                    "n_seeds",
                ),
                (),
            ),
            "architecture_classifications": _table(
                (
                    "variant",
                    "protocol_name",
                    "classification",
                    "austere_success_rate",
                    "classic_success_rate",
                    "ecological_success_rate",
                    "classic_minus_austere",
                    "ecological_minus_austere",
                    "austere_survival_threshold",
                    "classic_gap_limit",
                    "ecological_gap_limit",
                    "reason",
                ),
                (),
            ),
            "austere_survival_by_variant": _table(
                (
                    "variant",
                    "protocol_name",
                    "survives",
                    "austere_success_rate",
                    "threshold",
                ),
                (),
            ),
            "limitations": limitations,
        }

    profiles = _mapping_or_empty(ladder.get("profiles"))
    cross_profile_deltas = _mapping_or_empty(ladder.get("cross_profile_deltas"))
    classifications = _mapping_or_empty(ladder.get("classifications"))
    raw_payload = _mapping_or_empty(ladder.get("raw_payload"))
    raw_variants = _mapping_or_empty(raw_payload.get("variants"))

    ladder_by_profile_rows: list[dict[str, object]] = []
    for profile_name, profile_payload in sorted(profiles.items()):
        if not isinstance(profile_payload, Mapping):
            continue
        variants = _mapping_or_empty(profile_payload.get("variants"))
        for variant_name, variant_payload in sorted(variants.items()):
            if not isinstance(variant_payload, Mapping):
                continue
            summary_payload = _mapping_or_empty(variant_payload.get("summary"))
            uncertainty = _mapping_or_empty(variant_payload.get("uncertainty"))
            scenario_uncertainty = _mapping_or_empty(
                uncertainty.get("scenario_success_rate")
            )
            episode_uncertainty = _mapping_or_empty(
                uncertainty.get("episode_success_rate")
            )
            reward_uncertainty = _mapping_or_empty(uncertainty.get("mean_reward"))
            protocol_name = str(
                _mapping_or_empty(raw_variants.get(variant_name)).get("protocol_name")
                or LADDER_PROTOCOL_NAMES.get(LADDER_RUNG_MAPPING.get(variant_name, ""), "")
            )
            ladder_by_profile_rows.append(
                {
                    "variant": str(variant_name),
                    "protocol_name": protocol_name,
                    "profile": str(profile_name),
                    "scenario_success_rate": _coerce_float(
                        summary_payload.get("scenario_success_rate"),
                    ),
                    "scenario_success_rate_ci_lower": _coerce_optional_float(
                        scenario_uncertainty.get("ci_lower")
                    ),
                    "scenario_success_rate_ci_upper": _coerce_optional_float(
                        scenario_uncertainty.get("ci_upper")
                    ),
                    "episode_success_rate": _coerce_float(
                        summary_payload.get("episode_success_rate"),
                    ),
                    "episode_success_rate_ci_lower": _coerce_optional_float(
                        episode_uncertainty.get("ci_lower")
                    ),
                    "episode_success_rate_ci_upper": _coerce_optional_float(
                        episode_uncertainty.get("ci_upper")
                    ),
                    "mean_reward": _coerce_optional_float(
                        summary_payload.get("mean_reward")
                    ),
                    "mean_reward_ci_lower": _coerce_optional_float(
                        reward_uncertainty.get("ci_lower")
                    ),
                    "mean_reward_ci_upper": _coerce_optional_float(
                        reward_uncertainty.get("ci_upper")
                    ),
                    "n_seeds": int(
                        _coerce_float(scenario_uncertainty.get("n_seeds"), 0.0)
                    ),
                }
            )

    shaping_gap_rows: list[dict[str, object]] = []
    for variant_name, deltas_payload in sorted(cross_profile_deltas.items()):
        if not isinstance(deltas_payload, Mapping):
            continue
        classic_payload = _mapping_or_empty(deltas_payload.get("classic"))
        ecological_payload = _mapping_or_empty(deltas_payload.get("ecological"))
        classic_metric = _mapping_or_empty(
            _mapping_or_empty(classic_payload.get("metrics")).get(
                "scenario_success_rate"
            )
        )
        ecological_metric = _mapping_or_empty(
            _mapping_or_empty(ecological_payload.get("metrics")).get(
                "scenario_success_rate"
            )
        )
        classic_uncertainty = _mapping_or_empty(classic_metric.get("delta_uncertainty"))
        ecological_uncertainty = _mapping_or_empty(
            ecological_metric.get("delta_uncertainty")
        )
        protocol_name = str(
            _mapping_or_empty(raw_variants.get(variant_name)).get("protocol_name")
            or LADDER_PROTOCOL_NAMES.get(LADDER_RUNG_MAPPING.get(variant_name, ""), "")
        )
        shaping_gap_rows.append(
            {
                "variant": str(variant_name),
                "protocol_name": protocol_name,
                "classic_vs_austere_delta": _coerce_optional_float(
                    _mapping_or_empty(classic_payload.get("summary")).get(
                        "scenario_success_rate_delta"
                    )
                ),
                "classic_vs_austere_ci_lower": _coerce_optional_float(
                    classic_uncertainty.get("ci_lower")
                ),
                "classic_vs_austere_ci_upper": _coerce_optional_float(
                    classic_uncertainty.get("ci_upper")
                ),
                "classic_vs_austere_effect_size": _coerce_optional_float(
                    classic_metric.get("cohens_d")
                ),
                "ecological_vs_austere_delta": _coerce_optional_float(
                    _mapping_or_empty(ecological_payload.get("summary")).get(
                        "scenario_success_rate_delta"
                    )
                ),
                "ecological_vs_austere_ci_lower": _coerce_optional_float(
                    ecological_uncertainty.get("ci_lower")
                ),
                "ecological_vs_austere_ci_upper": _coerce_optional_float(
                    ecological_uncertainty.get("ci_upper")
                ),
                "ecological_vs_austere_effect_size": _coerce_optional_float(
                    ecological_metric.get("cohens_d")
                ),
                "n_seeds": max(
                    int(_coerce_float(classic_uncertainty.get("n_seeds"), 0.0)),
                    int(_coerce_float(ecological_uncertainty.get("n_seeds"), 0.0)),
                ),
            }
        )

    classification_rows: list[dict[str, object]] = []
    austere_survival_rows: list[dict[str, object]] = []
    for variant_name, classification_payload in sorted(classifications.items()):
        if not isinstance(classification_payload, Mapping):
            continue
        classification = _mapping_or_empty(classification_payload.get("classification"))
        austere_survival = _mapping_or_empty(
            classification_payload.get("austere_survival")
        )
        label = str(classification.get("label") or "")
        if not label:
            label = "unknown"
        elif label not in CLASSIFICATION_LABELS:
            label = f"nonstandard:{label}"
        protocol_name = str(
            classification_payload.get("protocol_name")
            or LADDER_PROTOCOL_NAMES.get(LADDER_RUNG_MAPPING.get(variant_name, ""), "")
        )
        austere_survival_threshold = _coerce_optional_float(
            classification.get("austere_survival_threshold")
        )
        if austere_survival_threshold is None:
            austere_survival_threshold = AUSTERE_SURVIVAL_THRESHOLD
        survival_threshold = _coerce_optional_float(
            austere_survival.get("survival_threshold")
        )
        if survival_threshold is None:
            survival_threshold = AUSTERE_SURVIVAL_THRESHOLD
        classification_rows.append(
            {
                "variant": str(variant_name),
                "protocol_name": protocol_name,
                "classification": label,
                "austere_success_rate": _coerce_optional_float(
                    classification.get("austere_success_rate")
                ),
                "classic_success_rate": _coerce_optional_float(
                    classification.get("classic_success_rate")
                ),
                "ecological_success_rate": _coerce_optional_float(
                    classification.get("ecological_success_rate")
                ),
                "classic_minus_austere": _coerce_optional_float(
                    classification.get("classic_minus_austere")
                ),
                "ecological_minus_austere": _coerce_optional_float(
                    classification.get("ecological_minus_austere")
                ),
                "austere_survival_threshold": austere_survival_threshold,
                "classic_gap_limit": _coerce_optional_float(
                    classification.get("classic_gap_limit")
                ),
                "ecological_gap_limit": _coerce_optional_float(
                    classification.get("ecological_gap_limit")
                ),
                "reason": str(classification.get("reason") or ""),
            }
        )
        austere_survival_rows.append(
            {
                "variant": str(variant_name),
                "protocol_name": protocol_name,
                "survives": bool(austere_survival.get("passed", False)),
                "austere_success_rate": _coerce_optional_float(
                    austere_survival.get("success_rate")
                ),
                "threshold": survival_threshold,
            }
        )

    limitations = list(ladder.get("limitations", []))
    if not ladder_by_profile_rows:
        limitations.append("No reward-profile ladder rows were available.")
    if not shaping_gap_rows:
        limitations.append("No shaping-gap rows were available for the ladder.")
    if not classification_rows:
        limitations.append("No architecture classification rows were available.")
    if not austere_survival_rows:
        limitations.append("No austere survival rows were available.")

    return {
        "available": bool(
            ladder_by_profile_rows
            or shaping_gap_rows
            or classification_rows
            or austere_survival_rows
        ),
        "confidence_level": REWARD_PROFILE_LADDER_CONFIDENCE_LEVEL,
        "rows": {
            "ladder_by_profile": ladder_by_profile_rows,
            "shaping_gap_by_variant": shaping_gap_rows,
            "architecture_classifications": classification_rows,
            "austere_survival_by_variant": austere_survival_rows,
        },
        "ladder_by_profile": _table(
            (
                "variant",
                "protocol_name",
                "profile",
                "scenario_success_rate",
                "scenario_success_rate_ci_lower",
                "scenario_success_rate_ci_upper",
                "episode_success_rate",
                "episode_success_rate_ci_lower",
                "episode_success_rate_ci_upper",
                "mean_reward",
                "mean_reward_ci_lower",
                "mean_reward_ci_upper",
                "n_seeds",
            ),
            ladder_by_profile_rows,
        ),
        "shaping_gap_by_variant": _table(
            (
                "variant",
                "protocol_name",
                "classic_vs_austere_delta",
                "classic_vs_austere_ci_lower",
                "classic_vs_austere_ci_upper",
                "classic_vs_austere_effect_size",
                "ecological_vs_austere_delta",
                "ecological_vs_austere_ci_lower",
                "ecological_vs_austere_ci_upper",
                "ecological_vs_austere_effect_size",
                "n_seeds",
            ),
            shaping_gap_rows,
        ),
        "architecture_classifications": _table(
            (
                "variant",
                "protocol_name",
                "classification",
                "austere_success_rate",
                "classic_success_rate",
                "ecological_success_rate",
                "classic_minus_austere",
                "ecological_minus_austere",
                "austere_survival_threshold",
                "classic_gap_limit",
                "ecological_gap_limit",
                "reason",
            ),
            classification_rows,
        ),
        "austere_survival_by_variant": _table(
            (
                "variant",
                "protocol_name",
                "survives",
                "austere_success_rate",
                "threshold",
            ),
            austere_survival_rows,
        ),
        "limitations": limitations,
    }

__all__ = [name for name in globals() if not name.startswith("__")]
