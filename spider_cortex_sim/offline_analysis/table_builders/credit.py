from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping

from .common import (
    CREDIT_ASSIGNMENT_STRATEGY_ORDER,
    CREDIT_ASSIGNMENT_SUCCESS_DELTA_THRESHOLD,
    CREDIT_ASSIGNMENT_VARIANT_CANDIDATES,
    LADDER_PROTOCOL_NAMES,
    LADDER_RUNG_MAPPING,
    MODULAR_CREDIT_RUNGS,
    _coerce_float,
    _coerce_optional_float,
    _credit_assignment_rung,
    _dominant_module_by_score,
    _mapping_or_empty,
    _table,
    _variant_with_minimal_reflex_support,
    extract_credit_metrics,
)

def _credit_assignment_strategy(
    variant_name: str,
    payload: Mapping[str, object],
) -> str:
    """
    Determine the credit-assignment strategy name for a given variant.
    
    Parameters:
        variant_name (str): Variant identifier used to infer a default strategy when no valid strategy is provided in the payload.
        payload (Mapping[str, object]): Variant payload that may contain a "config" mapping with a "credit_strategy" override.
    
    Returns:
        str: The chosen credit-assignment strategy name. This will be a configured strategy from the payload if valid, otherwise one of the canonical strategy names such as "local_only", "counterfactual", or "broadcast".
    """
    config = _mapping_or_empty(payload.get("config"))
    strategy = str(config.get("credit_strategy") or "")
    if strategy in CREDIT_ASSIGNMENT_STRATEGY_ORDER:
        return strategy
    if variant_name in {
        "three_center_modular_local_credit",
        "three_center_local_credit",
        "four_center_modular_local_credit",
        "local_credit_only",
    }:
        return "local_only"
    if variant_name in {
        "three_center_modular_counterfactual",
        "three_center_counterfactual_credit",
        "four_center_modular_counterfactual",
        "counterfactual_credit",
    }:
        return "counterfactual"
    return "broadcast"


def _credit_assignment_variant_rows(
    ablations: Mapping[str, object],
) -> tuple[
    dict[tuple[str, str], tuple[str, Mapping[str, object]]],
    list[str],
]:
    """
    Selects one representative variant payload for each (rung, credit strategy) comparison and reports any missing comparisons.
    
    Parameters:
        ablations (Mapping[str, object]): Ablation data containing a "variants" mapping of variant names to payloads; payloads are normalized before selection.
    
    Returns:
        selected (dict[tuple[str, str], tuple[str, Mapping[str, object]]]): Mapping from (rung, credit_strategy) to the chosen (variant_name, payload).
        limitations (list[str]): Human-readable messages for any (rung, strategy) comparisons that could not be satisfied.
    """
    variants = {
        str(variant_name): _variant_with_minimal_reflex_support(payload)
        for variant_name, payload in _mapping_or_empty(ablations.get("variants")).items()
        if isinstance(payload, Mapping)
    }
    selected: dict[tuple[str, str], tuple[str, Mapping[str, object]]] = {}
    limitations: list[str] = []
    for key, candidates in CREDIT_ASSIGNMENT_VARIANT_CANDIDATES.items():
        selected_variant: tuple[str, Mapping[str, object]] | None = None
        for candidate in candidates:
            payload = variants.get(candidate)
            if isinstance(payload, Mapping):
                selected_variant = (candidate, payload)
                break
        if selected_variant is None:
            rung, strategy = key
            for variant_name, payload in sorted(variants.items()):
                if (
                    _credit_assignment_rung(variant_name) == rung
                    and _credit_assignment_strategy(variant_name, payload) == strategy
                ):
                    selected_variant = (variant_name, payload)
                    break
        if selected_variant is None:
            limitations.append(
                f"Credit-assignment comparison is missing {key[0]} {key[1]}."
            )
            continue
        selected[key] = selected_variant
    return selected, limitations


def _credit_assignment_interpretations(
    strategy_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """
    Generate human-readable interpretation rows from per-(rung, credit_strategy) metric rows.
    
    Parameters:
        strategy_rows (list[dict[str, object]]): List of metric rows where each row is a mapping that should include at least the keys
            `"rung"`, `"credit_strategy"`, and `"scenario_success_delta_vs_broadcast"` (values may be missing or None).
    
    Returns:
        list[dict[str, object]]: A list of interpretation dictionaries. Each interpretation contains:
            - "scope": the rung(s) or rung-range the interpretation applies to,
            - "finding": a short title of the finding,
            - "evidence": a numeric summary (rounded) supporting the finding,
            - "interpretation": a human-readable explanation of the finding.
    """
    rows_by_key = {
        (str(row.get("rung") or ""), str(row.get("credit_strategy") or "")): row
        for row in strategy_rows
    }
    interpretations: list[dict[str, object]] = []

    local_failures: list[str] = []
    global_credit_failures: list[str] = []
    counterfactual_deltas: dict[str, float] = {}

    for rung in MODULAR_CREDIT_RUNGS:
        broadcast = rows_by_key.get((rung, "broadcast"))
        local_only = rows_by_key.get((rung, "local_only"))
        counterfactual = rows_by_key.get((rung, "counterfactual"))
        if isinstance(broadcast, Mapping) and isinstance(local_only, Mapping):
            local_delta = _coerce_float(local_only.get("scenario_success_delta_vs_broadcast"))
            if local_delta <= -CREDIT_ASSIGNMENT_SUCCESS_DELTA_THRESHOLD:
                local_failures.append(rung)
                interpretations.append(
                    {
                        "scope": rung,
                        "finding": "Failure by local credit insufficiency",
                        "evidence": round(local_delta, 6),
                        "interpretation": (
                            f"`local_only` trails broadcast by {local_delta:.2f} on {rung}, "
                            "so purely local gradients look insufficient at this rung."
                        ),
                    }
                )
            elif local_delta >= CREDIT_ASSIGNMENT_SUCCESS_DELTA_THRESHOLD:
                global_credit_failures.append(rung)
                interpretations.append(
                    {
                        "scope": rung,
                        "finding": "Failure by excessive global credit",
                        "evidence": round(local_delta, 6),
                        "interpretation": (
                            f"`local_only` beats broadcast by {local_delta:.2f} on {rung}, "
                            "so uniform global broadcast looks over-distributed here."
                        ),
                    }
                )
        if isinstance(broadcast, Mapping) and isinstance(counterfactual, Mapping):
            raw_counterfactual_delta = counterfactual.get(
                "scenario_success_delta_vs_broadcast"
            )
            counterfactual_delta = (
                _coerce_float(raw_counterfactual_delta)
                if raw_counterfactual_delta is not None
                else None
            )
            if counterfactual_delta is not None:
                counterfactual_deltas[rung] = counterfactual_delta
            if (
                counterfactual_delta is not None
                and counterfactual_delta >= CREDIT_ASSIGNMENT_SUCCESS_DELTA_THRESHOLD
            ):
                interpretations.append(
                    {
                        "scope": rung,
                        "finding": "Counterfactual credit improvement",
                        "evidence": round(counterfactual_delta, 6),
                        "interpretation": (
                            f"`counterfactual` beats broadcast by {counterfactual_delta:.2f} "
                            f"on {rung}, suggesting targeted global credit is more useful than "
                            "uniform broadcast at this architecture."
                        ),
                    }
                )

    if local_failures:
        if set(local_failures) == set(MODULAR_CREDIT_RUNGS):
            scope_text = "`local_only` fails across every modular rung that was evaluated."
        elif len(local_failures) == 1:
            scope_text = (
                f"`local_only` fails primarily at {local_failures[0]} in the modular ladder."
            )
        else:
            scope_text = (
                "`local_only` shows mixed failures across the modular ladder: "
                + ", ".join(local_failures)
                + "."
            )
        interpretations.append(
            {
                "scope": "/".join(MODULAR_CREDIT_RUNGS),
                "finding": "Local-only failure scope",
                "evidence": ", ".join(local_failures),
                "interpretation": scope_text,
            }
        )

    if global_credit_failures:
        interpretations.append(
            {
                "scope": "/".join(MODULAR_CREDIT_RUNGS),
                "finding": "Global-credit failure scope",
                "evidence": ", ".join(global_credit_failures),
                "interpretation": (
                    "Uniform broadcast underperforms a local-only baseline at "
                    + ", ".join(global_credit_failures)
                    + "."
                ),
            }
        )

    available_counterfactual_rungs = [
        rung for rung in MODULAR_CREDIT_RUNGS if rung in counterfactual_deltas
    ]
    if len(available_counterfactual_rungs) >= 2:
        first_rung = available_counterfactual_rungs[0]
        last_rung = available_counterfactual_rungs[-1]
        cf_first = counterfactual_deltas[first_rung]
        cf_last = counterfactual_deltas[last_rung]
        diff = cf_last - cf_first
        if diff >= CREDIT_ASSIGNMENT_SUCCESS_DELTA_THRESHOLD:
            text = (
                f"`counterfactual` helps more in {last_rung} than {first_rung} "
                f"({cf_last:.2f} vs {cf_first:.2f} "
                "delta vs broadcast), consistent with larger gains when there are more modules."
            )
        elif diff <= -CREDIT_ASSIGNMENT_SUCCESS_DELTA_THRESHOLD:
            text = (
                f"`counterfactual` helps more in {first_rung} than {last_rung} "
                f"({cf_first:.2f} vs {cf_last:.2f} "
                "delta vs broadcast), so extra modularity did not amplify its benefit here."
            )
        else:
            text = (
                f"`counterfactual` has similar impact across {first_rung} and {last_rung} "
                f"({cf_first:.2f} vs {cf_last:.2f} delta vs broadcast)."
            )
        interpretations.append(
            {
                "scope": f"{first_rung}/{last_rung}",
                "finding": "Counterfactual scaling across the ladder",
                "evidence": round(diff, 6),
                "interpretation": text,
            }
        )
    return interpretations


def build_credit_assignment_tables(
    ablations: Mapping[str, object],
) -> dict[str, object]:
    """
    Build credit-assignment tables for the modular ladder stages.
    
    Parameters:
        ablations (Mapping[str, object]): Ablation payload containing variant entries under
            the "variants" key; each variant's payload is expected to include `summary`
            and `suite` mappings used to populate the tables.
    
    Returns:
        dict[str, object]: A dictionary with the following keys:
            - "available": bool indicating whether any strategy rows were produced.
            - "strategy_summary": table-like object summarizing one row per rung and credit
              strategy (fields include rung, protocol_name, variant, architecture,
              credit_strategy, scenario_success_rate, episode_success_rate,
              scenario_success_delta_vs_broadcast, dominant_module,
              mean_dominant_module_share, mean_effective_module_count, source).
            - "module_credit": table-like object with one row per rung/strategy/module
              (fields include rung, protocol_name, variant, credit_strategy, module,
              mean_module_credit_weight, mean_module_gradient_norm,
              mean_counterfactual_credit_weight, mean_module_contribution_share,
              dominant_module_rate, scenario_success_rate).
            - "scenario_success": table-like object with one row per rung/strategy/scenario
              (fields include rung, protocol_name, variant, credit_strategy, scenario,
              success_rate, source).
            - "interpretations": list of human-readable interpretation rows derived from
              strategy-level comparisons.
            - "limitations": list of limitation messages describing missing or incomplete
              comparisons or data found while building the tables.
    """
    selected, limitations = _credit_assignment_variant_rows(ablations)
    strategy_rows: list[dict[str, object]] = []
    module_rows: list[dict[str, object]] = []
    scenario_rows: list[dict[str, object]] = []

    for rung in MODULAR_CREDIT_RUNGS:
        broadcast_payload_tuple = selected.get((rung, "broadcast"))
        broadcast_summary = (
            _mapping_or_empty(broadcast_payload_tuple[1].get("summary"))
            if broadcast_payload_tuple is not None
            else {}
        )
        broadcast_success = _coerce_optional_float(
            broadcast_summary.get("scenario_success_rate")
        )
        for strategy in CREDIT_ASSIGNMENT_STRATEGY_ORDER:
            selected_payload = selected.get((rung, strategy))
            if selected_payload is None:
                continue
            variant_name, payload = selected_payload
            summary = _mapping_or_empty(payload.get("summary"))
            suite = _mapping_or_empty(payload.get("suite"))
            scenario_success = (
                _coerce_float(summary.get("scenario_success_rate"))
                if summary.get("scenario_success_rate") is not None
                else None
            )
            episode_success = (
                _coerce_float(summary.get("episode_success_rate"))
                if summary.get("episode_success_rate") is not None
                else None
            )
            summary_row = {
                "rung": rung,
                "protocol_name": LADDER_PROTOCOL_NAMES.get(rung, rung),
                "variant": variant_name,
                "architecture": str(
                    _mapping_or_empty(payload.get("config")).get("architecture") or ""
                ),
                "credit_strategy": strategy,
                "scenario_success_rate": (
                    round(scenario_success, 6)
                    if scenario_success is not None
                    else None
                ),
                "episode_success_rate": (
                    round(episode_success, 6)
                    if episode_success is not None
                    else None
                ),
                "scenario_success_delta_vs_broadcast": (
                    round(scenario_success - broadcast_success, 6)
                    if scenario_success is not None and broadcast_success is not None
                    else None
                ),
                "dominant_module": str(
                    summary.get("dominant_module")
                    or _dominant_module_by_score(
                        _mapping_or_empty(summary.get("dominant_module_distribution"))
                    )
                    or ""
                ),
                "mean_dominant_module_share": round(
                    _coerce_float(summary.get("mean_dominant_module_share")),
                    6,
                ),
                "mean_effective_module_count": round(
                    _coerce_float(summary.get("mean_effective_module_count")),
                    6,
                ),
                "source": f"ablations.variants.{variant_name}.summary",
            }
            strategy_rows.append(summary_row)

            module_credit_weights = _mapping_or_empty(
                summary.get("mean_module_credit_weights")
            )
            module_gradient_norms = _mapping_or_empty(
                summary.get("module_gradient_norm_means")
            )
            counterfactual_credit_weights = _mapping_or_empty(
                summary.get("mean_counterfactual_credit_weights")
            )
            module_contribution_share = _mapping_or_empty(
                summary.get("mean_module_contribution_share")
            )
            dominant_module_distribution = _mapping_or_empty(
                summary.get("dominant_module_distribution")
            )
            dominant_module_name = str(summary.get("dominant_module") or "")
            module_names = sorted(
                {
                    *module_credit_weights.keys(),
                    *module_gradient_norms.keys(),
                    *counterfactual_credit_weights.keys(),
                    *module_contribution_share.keys(),
                    *dominant_module_distribution.keys(),
                    *((dominant_module_name,) if dominant_module_name else ()),
                }
            )
            for module_name in module_names:
                module_rows.append(
                    {
                        "rung": rung,
                        "protocol_name": summary_row["protocol_name"],
                        "variant": variant_name,
                        "credit_strategy": strategy,
                        "module": str(module_name),
                        "mean_module_credit_weight": round(
                            _coerce_float(module_credit_weights.get(module_name)),
                            6,
                        ),
                        "mean_module_gradient_norm": round(
                            _coerce_float(module_gradient_norms.get(module_name)),
                            6,
                        ),
                        "mean_counterfactual_credit_weight": round(
                            _coerce_float(counterfactual_credit_weights.get(module_name)),
                            6,
                        ),
                        "mean_module_contribution_share": round(
                            _coerce_float(module_contribution_share.get(module_name)),
                            6,
                        ),
                        "dominant_module_rate": round(
                            _coerce_float(dominant_module_distribution.get(module_name)),
                            6,
                        ),
                        "scenario_success_rate": summary_row["scenario_success_rate"],
                    }
                )

            for scenario_name, scenario_payload in sorted(suite.items()):
                if not isinstance(scenario_payload, Mapping):
                    continue
                scenario_rows.append(
                    {
                        "rung": rung,
                        "protocol_name": summary_row["protocol_name"],
                        "variant": variant_name,
                        "credit_strategy": strategy,
                        "scenario": str(scenario_name),
                        "success_rate": round(
                            _coerce_float(scenario_payload.get("success_rate")),
                            6,
                        ),
                        "source": f"ablations.variants.{variant_name}.suite.{scenario_name}",
                    }
                )

    interpretation_rows = _credit_assignment_interpretations(strategy_rows)
    if not strategy_rows:
        limitations.append(
            "No modular-ladder credit-assignment variants were available in the ablation payload."
        )
    return {
        "available": bool(strategy_rows),
        "strategy_summary": _table(
            (
                "rung",
                "protocol_name",
                "variant",
                "architecture",
                "credit_strategy",
                "scenario_success_rate",
                "episode_success_rate",
                "scenario_success_delta_vs_broadcast",
                "dominant_module",
                "mean_dominant_module_share",
                "mean_effective_module_count",
                "source",
            ),
            strategy_rows,
        ),
        "module_credit": _table(
            (
                "rung",
                "protocol_name",
                "variant",
                "credit_strategy",
                "module",
                "mean_module_credit_weight",
                "mean_module_gradient_norm",
                "mean_counterfactual_credit_weight",
                "mean_module_contribution_share",
                "dominant_module_rate",
                "scenario_success_rate",
            ),
            module_rows,
        ),
        "scenario_success": _table(
            (
                "rung",
                "protocol_name",
                "variant",
                "credit_strategy",
                "scenario",
                "success_rate",
                "source",
            ),
            scenario_rows,
        ),
        "interpretations": list(interpretation_rows),
        "limitations": limitations,
    }


def build_credit_table(
    summary: Mapping[str, object],
) -> dict[str, object]:
    """
    Build a flat, machine-readable table of per-variant, per-module credit metrics extracted from a benchmark summary payload.
    
    Parameters:
        summary (Mapping[str, object]): Benchmark summary payload containing fields like
            `config`, `behavior_evaluation`, `evaluation`, and `training` from which
            credit metrics and per-variant summaries are extracted.
    
    Returns:
        dict[str, object]: A dictionary with the following keys:
            - "available": `True` if any per-module rows were produced, `False` otherwise.
            - "table": Tabular rows of per-variant, per-module credit metrics including
              `variant`, `architecture_rung`, `credit_strategy`, `module_name`,
              `credit_weight`, `gradient_norm`, `counterfactual_weight`, and
              `scenario_success_rate`.
            - "summary_statistics": A mapping with:
                - "mean_credit_per_module_by_strategy": table of average credit weight
                  per (strategy, module) and `variant_count`.
                - "credit_concentration": table of average `mean_effective_module_count`
                  per strategy and `variant_count`.
            - "limitations": List of human-readable limitation strings describing missing
              or unavailable input data (may include a notice when no credit metrics exist).
    """
    credit_metrics = extract_credit_metrics(summary, ())
    behavior_evaluation = _mapping_or_empty(summary.get("behavior_evaluation"))
    ablations = _mapping_or_empty(behavior_evaluation.get("ablations"))
    ablation_variants = _mapping_or_empty(ablations.get("variants"))
    config = _mapping_or_empty(summary.get("config"))
    brain_config = _mapping_or_empty(config.get("brain"))

    rows: list[dict[str, object]] = []
    mean_credit_by_strategy_module: list[dict[str, object]] = []
    concentration_rows: list[dict[str, object]] = []
    limitations: list[str] = []

    credit_means_by_strategy_module: defaultdict[tuple[str, str], list[float]] = defaultdict(list)
    effective_module_count_by_strategy: defaultdict[str, list[float]] = defaultdict(list)

    def variant_summary_payload(variant_name: str) -> Mapping[str, object]:
        """
        Retrieve the summary payload for a named variant from ablation variants or, when the name matches the current run, from the run's evaluation/training blocks.
        
        Parameters:
        	variant_name (str): The variant identifier to look up.
        
        Returns:
        	The variant's summary mapping if available, otherwise an empty mapping.
        """
        ablation_payload = _mapping_or_empty(ablation_variants.get(variant_name))
        if ablation_payload:
            normalized_payload = _variant_with_minimal_reflex_support(ablation_payload)
            return _mapping_or_empty(normalized_payload.get("summary"))
        current_name = str(
            brain_config.get("name")
            or config.get("name")
            or "current_run"
        )
        if variant_name != current_name:
            return {}
        for block in (
            _mapping_or_empty(summary.get("evaluation_without_reflex_support")).get("summary"),
            summary.get("evaluation"),
            behavior_evaluation.get("summary"),
            summary.get("training"),
        ):
            payload = _mapping_or_empty(block)
            if payload:
                return payload
        return {}

    for variant_name, payload in sorted(credit_metrics.items()):
        summary_payload = variant_summary_payload(variant_name)
        raw_ablation_payload = ablation_variants.get(variant_name)
        normalized_ablation_payload = (
            _variant_with_minimal_reflex_support(raw_ablation_payload)
            if isinstance(raw_ablation_payload, Mapping)
            else {}
        )
        config_payload = _mapping_or_empty(
            normalized_ablation_payload.get("config")
        )
        brain_variant_name = str(brain_config.get("name") or config.get("name") or "")
        architecture_rung = str(
            payload.get("architecture_rung")
            or _credit_assignment_rung(variant_name)
            or LADDER_RUNG_MAPPING.get(variant_name)
            or _credit_assignment_rung(brain_variant_name)
            or LADDER_RUNG_MAPPING.get(brain_variant_name)
            or ""
        )
        inferred_strategy = ""
        if config_payload or _credit_assignment_rung(variant_name):
            inferred_strategy = _credit_assignment_strategy(
                variant_name,
                {"config": config_payload},
            )
        elif brain_variant_name:
            inferred_strategy = _credit_assignment_strategy(
                brain_variant_name,
                {"config": brain_config},
            )
        strategy = ""
        for strategy_variant_name, strategy_value in (
            (variant_name, payload.get("strategy")),
            (variant_name, config_payload.get("credit_strategy")),
            (brain_variant_name or variant_name, config.get("credit_strategy")),
            (brain_variant_name or variant_name, brain_config.get("credit_strategy")),
        ):
            if strategy_value:
                strategy = _credit_assignment_strategy(
                    strategy_variant_name,
                    {"config": {"credit_strategy": strategy_value}},
                )
                break
        if not strategy:
            strategy = inferred_strategy or "broadcast"
        scenario_success_rate = _coerce_optional_float(
            summary_payload.get("scenario_success_rate")
        )
        mean_effective_module_count = _coerce_optional_float(
            summary_payload.get("mean_effective_module_count")
        )
        weights = _mapping_or_empty(payload.get("weights"))
        gradient_norms = _mapping_or_empty(payload.get("gradient_norms"))
        counterfactual_weights = _mapping_or_empty(
            payload.get("counterfactual_weights")
        )
        raw_weight_keys = set(weights.keys())
        module_names = sorted(
            {
                *weights.keys(),
                *gradient_norms.keys(),
                *counterfactual_weights.keys(),
            }
        )
        for module_name in module_names:
            credit_weight = _coerce_float(weights.get(module_name))
            gradient_norm = _coerce_float(gradient_norms.get(module_name))
            counterfactual_weight = _coerce_float(
                counterfactual_weights.get(module_name)
            )
            rows.append(
                {
                    "variant": variant_name,
                    "architecture_rung": architecture_rung,
                    "credit_strategy": strategy,
                    "module_name": str(module_name),
                    "credit_weight": round(credit_weight, 6),
                    "gradient_norm": round(gradient_norm, 6),
                    "counterfactual_weight": round(counterfactual_weight, 6),
                    "scenario_success_rate": (
                        round(float(scenario_success_rate), 6)
                        if scenario_success_rate is not None
                        else None
                    ),
                }
            )
            if module_name in raw_weight_keys:
                credit_means_by_strategy_module[(strategy, str(module_name))].append(
                    credit_weight
                )
        if mean_effective_module_count is not None:
            effective_module_count_by_strategy[strategy].append(
                float(mean_effective_module_count)
            )

    for (strategy, module_name), values in sorted(credit_means_by_strategy_module.items()):
        mean_credit_by_strategy_module.append(
            {
                "credit_strategy": strategy,
                "module_name": module_name,
                "mean_credit_weight": round(
                    sum(values) / len(values),
                    6,
                ),
                "variant_count": len(values),
            }
        )
    for strategy, values in sorted(effective_module_count_by_strategy.items()):
        concentration_rows.append(
            {
                "credit_strategy": strategy,
                "mean_effective_module_count": round(
                    sum(values) / len(values),
                    6,
                ),
                "variant_count": len(values),
            }
        )

    if not credit_metrics:
        limitations.append("No credit metrics were available in the summary payload.")

    return {
        "available": bool(rows),
        "table": _table(
            (
                "variant",
                "architecture_rung",
                "credit_strategy",
                "module_name",
                "credit_weight",
                "gradient_norm",
                "counterfactual_weight",
                "scenario_success_rate",
            ),
            rows,
        ),
        "summary_statistics": {
            "mean_credit_per_module_by_strategy": _table(
                ("credit_strategy", "module_name", "mean_credit_weight", "variant_count"),
                mean_credit_by_strategy_module,
            ),
            "credit_concentration": _table(
                ("credit_strategy", "mean_effective_module_count", "variant_count"),
                concentration_rows,
            ),
        },
        "limitations": limitations,
    }


def build_scenario_checks_rows(
    scenario_success: Mapping[str, object],
) -> list[dict[str, object]]:
    """
    Builds a list of per-scenario check result rows from a scenario-success payload.
    
    Parameters:
        scenario_success (Mapping[str, object]): Payload containing a "scenarios" list (each scenario is a mapping) and an optional top-level "source" string.
    
    Returns:
        list[dict[str, object]]: A list of rows; each row is a mapping with the keys:
            - `scenario` (str): scenario identifier or empty string.
            - `check_name` (str): name of the check.
            - `pass_rate` (float): check pass rate coerced to float and rounded to 6 decimals.
            - `mean_value` (float): check mean value coerced to float and rounded to 6 decimals.
            - `expected` (str): expected value string (empty if missing).
            - `description` (str): description string (empty if missing).
            - `source` (str): source string taken from `scenario_success["source"]` (empty if missing).
    """
    rows: list[dict[str, object]] = []
    scenarios_list = scenario_success.get("scenarios")
    for scenario in (scenarios_list if isinstance(scenarios_list, list) else []):
        if not isinstance(scenario, Mapping):
            continue
        checks = scenario.get("checks", {})
        if not isinstance(checks, Mapping):
            continue
        for check_name, payload in sorted(checks.items()):
            if not isinstance(payload, Mapping):
                continue
            scenario_name = scenario.get("scenario")
            expected = payload.get("expected")
            description = payload.get("description")
            source = scenario_success.get("source")
            rows.append(
                {
                    "scenario": (
                        "" if scenario_name is None else str(scenario_name)
                    ),
                    "check_name": str(check_name),
                    "pass_rate": round(_coerce_float(payload.get("pass_rate")), 6),
                    "mean_value": round(_coerce_float(payload.get("mean_value")), 6),
                    "expected": "" if expected is None else str(expected),
                    "description": "" if description is None else str(description),
                    "source": "" if source is None else str(source),
                }
            )
    return rows

__all__ = [
    "build_credit_assignment_tables",
    "build_credit_table",
    "build_scenario_checks_rows",
]
