from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from html import escape
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_MODULE_NAMES: tuple[str, ...] = (
    "visual_cortex",
    "sensory_cortex",
    "hunger_center",
    "sleep_center",
    "alert_center",
    "monolithic_policy",
)


def _coerce_float(value: object, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return float(value)
    if value is None:
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(result):
        return default
    return result


def _coerce_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "on"}


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return 0.0
    return float(numerator / denominator)


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    if not items:
        return 0.0
    return float(sum(items) / len(items))


def load_summary(path: str | Path | None) -> dict[str, object]:
    if path is None:
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_trace(path: str | Path | None) -> list[dict[str, object]]:
    if path is None:
        return []
    items: list[dict[str, object]] = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                items.append(payload)
    return items


def load_behavior_csv(path: str | Path | None) -> list[dict[str, object]]:
    if path is None:
        return []
    with Path(path).open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def normalize_behavior_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    normalized: list[dict[str, object]] = []
    for row in rows:
        item = {str(key): value for key, value in dict(row).items()}
        evaluation_map = str(
            item.get("evaluation_map")
            or item.get("map_template")
            or item.get("scenario_map")
            or ""
        )
        scenario_map = str(item.get("scenario_map") or evaluation_map)
        reward_profile = str(item.get("reward_profile") or "")
        ablation_variant = str(item.get("ablation_variant") or "")
        ablation_architecture = str(item.get("ablation_architecture") or "")
        operational_profile = str(item.get("operational_profile") or "")
        budget_profile = str(item.get("budget_profile") or "")
        item["evaluation_map"] = evaluation_map
        item["map_template"] = evaluation_map
        item["scenario_map"] = scenario_map
        item["reward_profile"] = reward_profile
        item["ablation_variant"] = ablation_variant
        item["ablation_architecture"] = ablation_architecture
        item["operational_profile"] = operational_profile
        item["budget_profile"] = budget_profile
        item["scenario"] = str(item.get("scenario") or "")
        item["success"] = _coerce_bool(item.get("success"))
        item["failure_count"] = int(_coerce_float(item.get("failure_count"), 0.0))
        item["simulation_seed"] = int(_coerce_float(item.get("simulation_seed"), 0.0))
        item["episode_seed"] = int(_coerce_float(item.get("episode_seed"), 0.0))
        item["episode"] = int(_coerce_float(item.get("episode"), 0.0))
        item["reflex_scale"] = _coerce_float(item.get("reflex_scale"), 0.0)
        item["reflex_anneal_final_scale"] = _coerce_float(
            item.get("reflex_anneal_final_scale"),
            item["reflex_scale"],
        )
        item["eval_reflex_scale"] = _coerce_float(
            item.get("eval_reflex_scale"),
            item["reflex_scale"],
        )
        normalized.append(item)
    return normalized


def _extract_reward_from_episode(detail: Mapping[str, object]) -> float:
    if "total_reward" in detail:
        return _coerce_float(detail.get("total_reward"))
    if "reward" in detail:
        return _coerce_float(detail.get("reward"))
    return 0.0


def extract_training_eval_series(summary: Mapping[str, object]) -> dict[str, object]:
    training = summary.get("training", {})
    evaluation = summary.get("evaluation", {})
    training_points: list[dict[str, object]] = []
    evaluation_points: list[dict[str, object]] = []
    limitations: list[str] = []
    source = "none"

    if isinstance(training, Mapping):
        episodes_detail = training.get("episodes_detail")
        if isinstance(episodes_detail, list) and episodes_detail:
            source = "summary.training.episodes_detail"
            for index, detail in enumerate(episodes_detail, start=1):
                if not isinstance(detail, Mapping):
                    continue
                training_points.append(
                    {
                        "index": index,
                        "episode": int(_coerce_float(detail.get("episode"), index)),
                        "reward": _extract_reward_from_episode(detail),
                    }
                )
        else:
            history_tail = training.get("history_tail")
            if isinstance(history_tail, list) and history_tail:
                source = "summary.training.history_tail"
                for index, detail in enumerate(history_tail, start=1):
                    if not isinstance(detail, Mapping):
                        continue
                    training_points.append(
                        {
                            "index": index,
                            "episode": int(_coerce_float(detail.get("episode"), index)),
                            "reward": _extract_reward_from_episode(detail),
                        }
                    )
            else:
                aggregate_reward = None
                if "mean_reward" in training:
                    aggregate_reward = _coerce_float(training.get("mean_reward"))
                elif "mean_reward_all" in training:
                    aggregate_reward = _coerce_float(training.get("mean_reward_all"))
                if aggregate_reward is not None:
                    source = "summary.training.aggregate"
                    training_points.append(
                        {"index": 1, "episode": 1, "reward": aggregate_reward}
                    )
                limitations.append(
                    "Training detail is partial; the chart may reflect a tail or aggregate instead of the full run."
                )

    if isinstance(evaluation, Mapping):
        episodes_detail = evaluation.get("episodes_detail")
        if isinstance(episodes_detail, list) and episodes_detail:
            for index, detail in enumerate(episodes_detail, start=1):
                if not isinstance(detail, Mapping):
                    continue
                evaluation_points.append(
                    {
                        "index": index,
                        "episode": int(_coerce_float(detail.get("episode"), index)),
                        "reward": _extract_reward_from_episode(detail),
                    }
                )
        elif "mean_reward" in evaluation:
            evaluation_points.append(
                {
                    "index": 1,
                    "episode": 1,
                    "reward": _coerce_float(evaluation.get("mean_reward")),
                }
            )
            limitations.append(
                "Evaluation detail is partial; the chart includes aggregate reward only."
            )

    available = bool(training_points or evaluation_points)
    if not available:
        limitations.append("No training or evaluation reward series was available.")
    return {
        "available": available,
        "source": source,
        "training_points": training_points,
        "evaluation_points": evaluation_points,
        "limitations": limitations,
    }


def _suite_from_summary(summary: Mapping[str, object]) -> dict[str, object]:
    behavior_evaluation = summary.get("behavior_evaluation", {})
    if not isinstance(behavior_evaluation, Mapping):
        return {}
    suite = behavior_evaluation.get("suite", {})
    if not isinstance(suite, Mapping):
        return {}
    return dict(suite)


def _scenario_suite_from_rows(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    by_scenario: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        scenario = str(row.get("scenario") or "")
        if scenario:
            by_scenario[scenario].append(row)
    suite: dict[str, object] = {}
    for scenario, items in sorted(by_scenario.items()):
        checks: dict[str, object] = {}
        check_names = sorted(
            {
                key[len("check_") : -len("_passed")]
                for row in items
                for key in row
                if key.startswith("check_") and key.endswith("_passed")
            }
        )
        for check_name in check_names:
            passed_values = [
                1.0 if _coerce_bool(row.get(f"check_{check_name}_passed")) else 0.0
                for row in items
            ]
            value_samples = [
                _coerce_float(row.get(f"check_{check_name}_value"))
                for row in items
                if row.get(f"check_{check_name}_value") not in {"", None}
            ]
            expected = next(
                (
                    str(row.get(f"check_{check_name}_expected") or "")
                    for row in items
                    if row.get(f"check_{check_name}_expected")
                ),
                "",
            )
            checks[check_name] = {
                "description": "Recovered from behavior_csv.",
                "expected": expected,
                "pass_rate": _mean(passed_values),
                "mean_value": _mean(value_samples) if value_samples else _mean(passed_values),
            }
        successes = [1.0 if _coerce_bool(row.get("success")) else 0.0 for row in items]
        failures = sorted(
            {
                failure.strip()
                for row in items
                for failure in str(row.get("failures") or "").split(";")
                if failure.strip()
            }
        )
        suite[scenario] = {
            "scenario": scenario,
            "description": str(items[0].get("scenario_description") or ""),
            "objective": str(items[0].get("scenario_objective") or ""),
            "episodes": len(items),
            "success_rate": _mean(successes),
            "checks": checks,
            "behavior_metrics": {},
            "failures": failures,
            "legacy_metrics": {},
        }
    return suite


def extract_scenario_success(
    summary: Mapping[str, object],
    behavior_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    suite = _suite_from_summary(summary)
    source = "summary.behavior_evaluation.suite"
    limitations: list[str] = []
    if not suite:
        suite = _scenario_suite_from_rows(behavior_rows)
        source = "behavior_csv"
        limitations.append(
            "Scenario checks were reconstructed from behavior_csv because summary.behavior_evaluation.suite was absent."
        )
    scenarios: list[dict[str, object]] = []
    for scenario, payload in sorted(suite.items()):
        if not isinstance(payload, Mapping):
            continue
        scenarios.append(
            {
                "scenario": scenario,
                "description": str(payload.get("description") or ""),
                "objective": str(payload.get("objective") or ""),
                "success_rate": _coerce_float(payload.get("success_rate")),
                "episodes": int(_coerce_float(payload.get("episodes"), 0.0)),
                "failures": list(payload.get("failures", []))
                if isinstance(payload.get("failures"), list)
                else [],
                "checks": dict(payload.get("checks", {}))
                if isinstance(payload.get("checks"), Mapping)
                else {},
                "legacy_metrics": dict(payload.get("legacy_metrics", {}))
                if isinstance(payload.get("legacy_metrics"), Mapping)
                else {},
            }
        )
    available = bool(scenarios)
    if not available:
        limitations.append("No scenario-level success data was available.")
    return {
        "available": available,
        "source": source,
        "scenarios": scenarios,
        "limitations": limitations,
    }


def _rows_with_default_reflex_support(
    rows: Sequence[Mapping[str, object]],
) -> list[Mapping[str, object]]:
    by_variant: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        by_variant[str(row.get("ablation_variant") or "")].append(row)
    selected: list[Mapping[str, object]] = []
    for items in by_variant.values():
        if not items:
            continue
        max_scale = max(_coerce_float(item.get("eval_reflex_scale"), 0.0) for item in items)
        selected.extend(
            item
            for item in items
            if math.isclose(
                _coerce_float(item.get("eval_reflex_scale"), 0.0),
                max_scale,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        )
    return selected or list(rows)


def _fallback_group_summary(
    rows: Sequence[Mapping[str, object]],
    *,
    key_name: str,
) -> dict[str, object]:
    by_key: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        key = str(row.get(key_name) or "")
        if key:
            by_key[key].append(row)
    summary: dict[str, object] = {}
    for key, items in sorted(by_key.items()):
        success_values = [1.0 if _coerce_bool(row.get("success")) else 0.0 for row in items]
        scenario_names = sorted({str(row.get("scenario") or "") for row in items if row.get("scenario")})
        summary[key] = {
            "summary": {
                "scenario_success_rate": _mean(success_values),
                "episode_success_rate": _mean(success_values),
                "scenario_count": len(scenario_names),
                "episode_count": len(items),
            }
        }
    return summary


def extract_comparisons(
    summary: Mapping[str, object],
    behavior_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    behavior_evaluation = summary.get("behavior_evaluation", {})
    comparisons = (
        behavior_evaluation.get("comparisons", {})
        if isinstance(behavior_evaluation, Mapping)
        else {}
    )
    limitations: list[str] = []
    if isinstance(comparisons, Mapping) and comparisons:
        reward_profiles = comparisons.get("reward_profiles", {})
        map_templates = comparisons.get("map_templates", {})
        return {
            "available": bool(reward_profiles or map_templates),
            "source": "summary.behavior_evaluation.comparisons",
            "reward_profiles": dict(reward_profiles) if isinstance(reward_profiles, Mapping) else {},
            "map_templates": dict(map_templates) if isinstance(map_templates, Mapping) else {},
            "limitations": [],
        }

    reward_profiles = _fallback_group_summary(behavior_rows, key_name="reward_profile")
    map_templates = _fallback_group_summary(behavior_rows, key_name="evaluation_map")
    if reward_profiles or map_templates:
        limitations.append(
            "Profile/map comparisons were reconstructed from behavior_csv and may not match the compact comparison payload exactly."
        )
    else:
        limitations.append("No profile or map comparison payload was available.")
    return {
        "available": bool(reward_profiles or map_templates),
        "source": "behavior_csv",
        "reward_profiles": reward_profiles,
        "map_templates": map_templates,
        "limitations": limitations,
    }


def extract_ablations(
    summary: Mapping[str, object],
    behavior_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    behavior_evaluation = summary.get("behavior_evaluation", {})
    ablations = (
        behavior_evaluation.get("ablations", {})
        if isinstance(behavior_evaluation, Mapping)
        else {}
    )
    limitations: list[str] = []
    if isinstance(ablations, Mapping) and ablations:
        variants = ablations.get("variants", {})
        return {
            "available": bool(variants),
            "source": "summary.behavior_evaluation.ablations",
            "reference_variant": str(ablations.get("reference_variant") or ""),
            "variants": dict(variants) if isinstance(variants, Mapping) else {},
            "deltas_vs_reference": dict(ablations.get("deltas_vs_reference", {}))
            if isinstance(ablations.get("deltas_vs_reference"), Mapping)
            else {},
            "limitations": [],
        }

    filtered_rows = _rows_with_default_reflex_support(behavior_rows)
    by_variant: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in filtered_rows:
        variant = str(row.get("ablation_variant") or "")
        if variant:
            by_variant[variant].append(row)
    variants: dict[str, object] = {}
    for variant, items in sorted(by_variant.items()):
        success_values = [1.0 if _coerce_bool(row.get("success")) else 0.0 for row in items]
        variants[variant] = {
            "config": {
                "architecture": str(items[0].get("ablation_architecture") or ""),
            },
            "summary": {
                "scenario_success_rate": _mean(success_values),
                "episode_success_rate": _mean(success_values),
            },
        }
    if variants:
        limitations.append(
            "Ablation comparisons were reconstructed from behavior_csv and omit deltas_vs_reference because the compact ablation payload was absent."
        )
    else:
        limitations.append("No ablation comparison data was available.")
    return {
        "available": bool(variants),
        "source": "behavior_csv",
        "reference_variant": "",
        "variants": variants,
        "deltas_vs_reference": {},
        "limitations": limitations,
    }


def extract_reflex_frequency(trace: Sequence[Mapping[str, object]]) -> dict[str, object]:
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
                reflex = payload.get("reflex")
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


def build_scenario_checks_rows(
    scenario_success: Mapping[str, object],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for scenario in scenario_success.get("scenarios", []) if isinstance(scenario_success.get("scenarios"), list) else []:
        if not isinstance(scenario, Mapping):
            continue
        checks = scenario.get("checks", {})
        if not isinstance(checks, Mapping):
            continue
        for check_name, payload in sorted(checks.items()):
            if not isinstance(payload, Mapping):
                continue
            rows.append(
                {
                    "scenario": str(scenario.get("scenario") or ""),
                    "check_name": str(check_name),
                    "pass_rate": round(_coerce_float(payload.get("pass_rate")), 6),
                    "mean_value": round(_coerce_float(payload.get("mean_value")), 6),
                    "expected": str(payload.get("expected") or ""),
                    "description": str(payload.get("description") or ""),
                    "source": str(scenario_success.get("source") or ""),
                }
            )
    return rows


def build_reward_component_rows(
    summary: Mapping[str, object],
    scenario_success: Mapping[str, object],
    trace: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    def add_components(source: str, scope: str, components: object) -> None:
        if not isinstance(components, Mapping):
            return
        for component, value in sorted(components.items()):
            rows.append(
                {
                    "source": source,
                    "scope": scope,
                    "component": str(component),
                    "value": round(_coerce_float(value), 6),
                }
            )

    for aggregate_name in ("training", "training_last_window", "evaluation"):
        aggregate = summary.get(aggregate_name, {})
        if isinstance(aggregate, Mapping):
            add_components(
                "summary",
                aggregate_name,
                aggregate.get("mean_reward_components"),
            )

    for scenario in scenario_success.get("scenarios", []) if isinstance(scenario_success.get("scenarios"), list) else []:
        if not isinstance(scenario, Mapping):
            continue
        legacy_metrics = scenario.get("legacy_metrics")
        if isinstance(legacy_metrics, Mapping):
            add_components(
                "behavior_evaluation.suite",
                f"scenario:{scenario.get('scenario')}",
                legacy_metrics.get("mean_reward_components"),
            )

    trace_totals: defaultdict[str, float] = defaultdict(float)
    for item in trace:
        components = item.get("reward_components")
        if not isinstance(components, Mapping):
            continue
        for component, value in components.items():
            trace_totals[str(component)] += _coerce_float(value)
    if trace_totals:
        add_components("trace", "trace_total", trace_totals)
    return rows


def build_diagnostics(
    summary: Mapping[str, object],
    scenario_success: Mapping[str, object],
    ablations: Mapping[str, object],
    reflex_frequency: Mapping[str, object],
) -> list[dict[str, object]]:
    diagnostics: list[dict[str, object]] = []
    evaluation = summary.get("evaluation", {})
    if isinstance(evaluation, Mapping):
        diagnostics.extend(
            [
                {
                    "label": "Evaluation mean reward",
                    "value": round(_coerce_float(evaluation.get("mean_reward")), 6),
                },
                {
                    "label": "Evaluation mean food distance delta",
                    "value": round(
                        _coerce_float(evaluation.get("mean_food_distance_delta")),
                        6,
                    ),
                },
                {
                    "label": "Evaluation mean shelter distance delta",
                    "value": round(
                        _coerce_float(evaluation.get("mean_shelter_distance_delta")),
                        6,
                    ),
                },
                {
                    "label": "Evaluation predator mode transitions",
                    "value": round(
                        _coerce_float(evaluation.get("mean_predator_mode_transitions")),
                        6,
                    ),
                },
                {
                    "label": "Evaluation dominant predator state",
                    "value": str(evaluation.get("dominant_predator_state") or ""),
                },
            ]
        )

    scenarios = scenario_success.get("scenarios", [])
    if isinstance(scenarios, list) and scenarios:
        weakest = min(
            (
                scenario
                for scenario in scenarios
                if isinstance(scenario, Mapping)
            ),
            key=lambda item: _coerce_float(item.get("success_rate"), 0.0),
        )
        diagnostics.append(
            {
                "label": "Weakest scenario",
                "value": f"{weakest.get('scenario')} ({_coerce_float(weakest.get('success_rate')):.2f})",
            }
        )

    variants = ablations.get("variants", {})
    if isinstance(variants, Mapping) and variants:
        sortable = []
        for variant_name, payload in variants.items():
            if not isinstance(payload, Mapping):
                continue
            summary_payload = payload.get("summary", {})
            sortable.append(
                (
                    _coerce_float(
                        summary_payload.get("scenario_success_rate"),
                        0.0,
                    ),
                    str(variant_name),
                )
            )
        if sortable:
            best_score, best_variant = max(sortable)
            diagnostics.append(
                {
                    "label": "Best ablation variant",
                    "value": f"{best_variant} ({best_score:.2f})",
                }
            )

    modules = reflex_frequency.get("modules", [])
    if isinstance(modules, list) and modules:
        highest = max(
            (
                module for module in modules
                if isinstance(module, Mapping)
            ),
            key=lambda item: _coerce_float(item.get("reflex_events"), 0.0),
        )
        diagnostics.append(
            {
                "label": "Most frequent reflex source",
                "value": f"{highest.get('module')} ({int(_coerce_float(highest.get('reflex_events'), 0.0))} events)",
            }
        )
    return diagnostics


def _chart_bounds(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 1.0
    minimum = min(values)
    maximum = max(values)
    if math.isclose(minimum, maximum):
        pad = 1.0 if minimum == 0.0 else abs(minimum) * 0.15
        return minimum - pad, maximum + pad
    pad = (maximum - minimum) * 0.1
    return minimum - pad, maximum + pad


def render_placeholder_svg(title: str, message: str) -> str:
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="900" height="260" viewBox="0 0 900 260">'
        '<rect x="0" y="0" width="900" height="260" fill="#f8fafc" />'
        f'<text x="40" y="54" font-size="24" font-family="monospace" fill="#0f172a">{escape(title)}</text>'
        f'<text x="40" y="120" font-size="16" font-family="monospace" fill="#475569">{escape(message)}</text>'
        "</svg>"
    )


def render_line_chart(
    title: str,
    series: Sequence[dict[str, object]],
    *,
    x_key: str = "index",
    y_key: str = "reward",
) -> str:
    if not series:
        return render_placeholder_svg(title, "No data available.")
    width = 900
    height = 320
    left = 70
    right = 30
    top = 55
    bottom = 45
    plot_width = width - left - right
    plot_height = height - top - bottom
    all_y = [_coerce_float(point.get(y_key)) for point in series]
    min_y, max_y = _chart_bounds(all_y)
    max_x = max(_coerce_float(point.get(x_key), 1.0) for point in series)
    max_x = max(max_x, 1.0)

    def x_coord(value: float) -> float:
        return left + (value / max_x) * plot_width

    def y_coord(value: float) -> float:
        if math.isclose(max_y, min_y):
            return top + plot_height / 2.0
        normalized = (value - min_y) / (max_y - min_y)
        return top + (1.0 - normalized) * plot_height

    points_attr = " ".join(
        f"{x_coord(_coerce_float(point.get(x_key), 0.0)):.2f},{y_coord(_coerce_float(point.get(y_key), 0.0)):.2f}"
        for point in series
    )
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect x="0" y="0" width="100%" height="100%" fill="#ffffff" />',
        f'<text x="{left}" y="32" font-size="22" font-family="monospace" fill="#0f172a">{escape(title)}</text>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#94a3b8" stroke-width="1"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#94a3b8" stroke-width="1"/>',
        f'<polyline fill="none" stroke="#2563eb" stroke-width="2.5" points="{points_attr}" />',
    ]
    for point in series:
        px = x_coord(_coerce_float(point.get(x_key), 0.0))
        py = y_coord(_coerce_float(point.get(y_key), 0.0))
        lines.append(
            f'<circle cx="{px:.2f}" cy="{py:.2f}" r="3.5" fill="#1d4ed8" />'
        )
    for idx, label in enumerate((min_y, (min_y + max_y) / 2.0, max_y)):
        y = y_coord(label)
        lines.append(
            f'<text x="12" y="{y + 5:.2f}" font-size="12" font-family="monospace" fill="#475569">{label:.2f}</text>'
        )
        lines.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}" stroke="#e2e8f0" stroke-width="1"/>'
        )
    lines.append("</svg>")
    return "".join(lines)


def render_bar_chart(
    title: str,
    items: Sequence[Mapping[str, object]],
    *,
    label_key: str,
    value_key: str,
) -> str:
    if not items:
        return render_placeholder_svg(title, "No data available.")
    width = 960
    height = max(260, 70 + 34 * len(items))
    left = 250
    right = 30
    top = 55
    bar_height = 22
    gap = 10
    plot_width = width - left - right
    max_value = max(_coerce_float(item.get(value_key), 0.0) for item in items)
    max_value = max(max_value, 1.0)
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect x="0" y="0" width="100%" height="100%" fill="#ffffff" />',
        f'<text x="32" y="32" font-size="22" font-family="monospace" fill="#0f172a">{escape(title)}</text>',
    ]
    for index, item in enumerate(items):
        value = _coerce_float(item.get(value_key), 0.0)
        label = str(item.get(label_key) or "")
        y = top + index * (bar_height + gap)
        bar_width = plot_width * _safe_divide(value, max_value)
        lines.append(
            f'<text x="32" y="{y + 16}" font-size="13" font-family="monospace" fill="#334155">{escape(label)}</text>'
        )
        lines.append(
            f'<rect x="{left}" y="{y}" width="{plot_width}" height="{bar_height}" fill="#e2e8f0" rx="4" />'
        )
        lines.append(
            f'<rect x="{left}" y="{y}" width="{bar_width:.2f}" height="{bar_height}" fill="#2563eb" rx="4" />'
        )
        lines.append(
            f'<text x="{left + plot_width + 8}" y="{y + 16}" font-size="12" font-family="monospace" fill="#475569">{value:.2f}</text>'
        )
    lines.append("</svg>")
    return "".join(lines)


def _write_svg(path: Path, svg: str) -> None:
    path.write_text(svg, encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _markdown_table(rows: Sequence[Sequence[object]], headers: Sequence[str]) -> str:
    if not rows:
        return "_No data available._"
    header_line = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    body = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in rows
    ]
    return "\n".join([header_line, separator, *body])


def build_report_data(
    *,
    summary: Mapping[str, object],
    trace: Sequence[Mapping[str, object]],
    behavior_rows: Sequence[Mapping[str, object]],
    summary_path: str | Path | None = None,
    trace_path: str | Path | None = None,
    behavior_csv_path: str | Path | None = None,
) -> dict[str, object]:
    normalized_rows = normalize_behavior_rows(behavior_rows)
    training_eval = extract_training_eval_series(summary)
    scenario_success = extract_scenario_success(summary, normalized_rows)
    comparisons = extract_comparisons(summary, normalized_rows)
    ablations = extract_ablations(summary, normalized_rows)
    reflex_frequency = extract_reflex_frequency(trace)
    scenario_check_rows = build_scenario_checks_rows(scenario_success)
    reward_component_rows = build_reward_component_rows(
        summary,
        scenario_success,
        trace,
    )
    diagnostics = build_diagnostics(
        summary,
        scenario_success,
        ablations,
        reflex_frequency,
    )

    limitations: list[str] = []
    for section in (
        training_eval,
        scenario_success,
        comparisons,
        ablations,
        reflex_frequency,
    ):
        if isinstance(section, Mapping):
            limitations.extend(
                str(item)
                for item in section.get("limitations", [])
                if item
            )

    inputs = {
        "summary_path": str(summary_path) if summary_path is not None else None,
        "trace_path": str(trace_path) if trace_path is not None else None,
        "behavior_csv_path": str(behavior_csv_path) if behavior_csv_path is not None else None,
        "summary_loaded": bool(summary),
        "trace_loaded": bool(trace),
        "behavior_csv_loaded": bool(normalized_rows),
    }
    return {
        "inputs": inputs,
        "training_eval": training_eval,
        "scenario_success": scenario_success,
        "comparisons": comparisons,
        "ablations": ablations,
        "reflex_frequency": reflex_frequency,
        "scenario_checks": scenario_check_rows,
        "reward_components": reward_component_rows,
        "diagnostics": diagnostics,
        "limitations": limitations,
    }


def write_report(output_dir: str | Path, report: Mapping[str, object]) -> dict[str, str]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    training_eval = report.get("training_eval", {})
    scenario_success = report.get("scenario_success", {})
    ablations = report.get("ablations", {})
    reflex_frequency = report.get("reflex_frequency", {})

    training_series = []
    if isinstance(training_eval, Mapping):
        training_series.extend(
            training_eval.get("training_points", [])
            if isinstance(training_eval.get("training_points"), list)
            else []
        )
        evaluation_points = (
            training_eval.get("evaluation_points", [])
            if isinstance(training_eval.get("evaluation_points"), list)
            else []
        )
    else:
        evaluation_points = []

    training_svg_path = output_path / "training_eval.svg"
    if training_series:
        _write_svg(
            training_svg_path,
            render_line_chart("Training reward trajectory", training_series),
        )
    elif evaluation_points:
        _write_svg(
            training_svg_path,
            render_line_chart("Evaluation reward trajectory", evaluation_points),
        )
    else:
        _write_svg(
            training_svg_path,
            render_placeholder_svg("Training / evaluation", "No summary reward series was available."),
        )

    scenario_svg_path = output_path / "scenario_success.svg"
    scenario_items = (
        scenario_success.get("scenarios", [])
        if isinstance(scenario_success, Mapping)
        else []
    )
    scenario_chart_items = [
        {"scenario": item.get("scenario"), "success_rate": item.get("success_rate")}
        for item in scenario_items
        if isinstance(item, Mapping)
    ]
    _write_svg(
        scenario_svg_path,
        render_bar_chart(
            "Scenario success rate",
            scenario_chart_items,
            label_key="scenario",
            value_key="success_rate",
        ),
    )

    ablation_svg_path = ""
    if isinstance(ablations, Mapping) and ablations.get("available"):
        variants = []
        variants_payload = ablations.get("variants", {})
        if isinstance(variants_payload, Mapping):
            for variant_name, payload in sorted(variants_payload.items()):
                if not isinstance(payload, Mapping):
                    continue
                summary_payload = payload.get("summary", {})
                variants.append(
                    {
                        "variant": variant_name,
                        "score": _coerce_float(
                            summary_payload.get("scenario_success_rate"),
                            0.0,
                        ),
                    }
                )
        if variants:
            ablation_svg_path = str(output_path / "ablation_comparison.svg")
            _write_svg(
                Path(ablation_svg_path),
                render_bar_chart(
                    "Ablation scenario success rate",
                    variants,
                    label_key="variant",
                    value_key="score",
                ),
            )

    reflex_svg_path = ""
    if isinstance(reflex_frequency, Mapping) and reflex_frequency.get("available"):
        modules = reflex_frequency.get("modules", [])
        if isinstance(modules, list) and modules:
            reflex_svg_path = str(output_path / "reflex_frequency.svg")
            _write_svg(
                Path(reflex_svg_path),
                render_bar_chart(
                    "Reflex events per module",
                    modules,
                    label_key="module",
                    value_key="reflex_events",
                ),
            )

    scenario_checks_path = output_path / "scenario_checks.csv"
    _write_csv(
        scenario_checks_path,
        report.get("scenario_checks", []) if isinstance(report.get("scenario_checks"), list) else [],
        ("scenario", "check_name", "pass_rate", "mean_value", "expected", "description", "source"),
    )

    reward_components_path = output_path / "reward_components.csv"
    _write_csv(
        reward_components_path,
        report.get("reward_components", []) if isinstance(report.get("reward_components"), list) else [],
        ("source", "scope", "component", "value"),
    )

    report_json_path = output_path / "report.json"
    report_json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    diagnostics = report.get("diagnostics", [])
    diagnostic_rows = [
        (
            str(item.get("label") or ""),
            str(item.get("value") or ""),
        )
        for item in diagnostics
        if isinstance(item, Mapping)
    ]
    scenario_rows = [
        (
            str(item.get("scenario") or ""),
            f"{_coerce_float(item.get('success_rate')):.2f}",
            str(len(item.get("checks", {})) if isinstance(item.get("checks"), Mapping) else 0),
        )
        for item in scenario_items
        if isinstance(item, Mapping)
    ]
    ablation_rows = []
    if isinstance(ablations, Mapping):
        variants_payload = ablations.get("variants", {})
        if isinstance(variants_payload, Mapping):
            for variant_name, payload in sorted(variants_payload.items()):
                if not isinstance(payload, Mapping):
                    continue
                summary_payload = payload.get("summary", {})
                config_payload = payload.get("config", {})
                ablation_rows.append(
                    (
                        str(variant_name),
                        str(config_payload.get("architecture") or ""),
                        f"{_coerce_float(summary_payload.get('scenario_success_rate')):.2f}",
                    )
                )

    markdown_lines = [
        "# Offline Analysis Report",
        "",
        "## Inputs",
        "",
        f"- `summary`: {report['inputs']['summary_path']}",
        f"- `trace`: {report['inputs']['trace_path']}",
        f"- `behavior_csv`: {report['inputs']['behavior_csv_path']}",
        "",
        "## Diagnostics",
        "",
        _markdown_table(diagnostic_rows, ("metric", "value")),
        "",
        "## Scenario Success",
        "",
        _markdown_table(scenario_rows, ("scenario", "success_rate", "check_count")),
        "",
        "## Ablations",
        "",
        _markdown_table(ablation_rows, ("variant", "architecture", "scenario_success_rate")),
        "",
        "## Limitations",
        "",
    ]
    limitations = report.get("limitations", [])
    if isinstance(limitations, list) and limitations:
        markdown_lines.extend(f"- {item}" for item in limitations)
    else:
        markdown_lines.append("- None.")
    markdown_lines.extend(
        [
            "",
            "## Generated Files",
            "",
            f"- `training_eval.svg`",
            f"- `scenario_success.svg`",
            f"- `scenario_checks.csv`",
            f"- `reward_components.csv`",
            f"- `report.json`",
        ]
    )
    if ablation_svg_path:
        markdown_lines.append("- `ablation_comparison.svg`")
    if reflex_svg_path:
        markdown_lines.append("- `reflex_frequency.svg`")

    report_md_path = output_path / "report.md"
    report_md_path.write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")

    return {
        "report_md": str(report_md_path),
        "report_json": str(report_json_path),
        "training_eval_svg": str(training_svg_path),
        "scenario_success_svg": str(scenario_svg_path),
        "scenario_checks_csv": str(scenario_checks_path),
        "reward_components_csv": str(reward_components_path),
        "ablation_comparison_svg": ablation_svg_path,
        "reflex_frequency_svg": reflex_svg_path,
    }


def run_offline_analysis(
    *,
    summary_path: str | Path | None = None,
    trace_path: str | Path | None = None,
    behavior_csv_path: str | Path | None = None,
    output_dir: str | Path,
) -> dict[str, object]:
    summary = load_summary(summary_path)
    trace = load_trace(trace_path)
    behavior_rows = load_behavior_csv(behavior_csv_path)
    report = build_report_data(
        summary=summary,
        trace=trace,
        behavior_rows=behavior_rows,
        summary_path=summary_path,
        trace_path=trace_path,
        behavior_csv_path=behavior_csv_path,
    )
    report["generated_files"] = write_report(output_dir, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Gera análise offline reproduzível a partir de summary, trace e behavior_csv.",
    )
    parser.add_argument("--summary", type=Path, default=None, help="Arquivo summary.json.")
    parser.add_argument("--trace", type=Path, default=None, help="Arquivo trace.jsonl.")
    parser.add_argument(
        "--behavior-csv",
        type=Path,
        default=None,
        help="Arquivo behavior_csv exportado pela CLI principal.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Diretório onde o relatório e os artefatos serão gerados.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.summary is None and args.trace is None and args.behavior_csv is None:
        parser.error(
            "Ao menos um entre --summary, --trace e --behavior-csv é obrigatório."
        )
    report = run_offline_analysis(
        summary_path=args.summary,
        trace_path=args.trace,
        behavior_csv_path=args.behavior_csv,
        output_dir=args.output_dir,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
