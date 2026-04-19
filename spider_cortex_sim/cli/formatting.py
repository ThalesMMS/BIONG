"""CLI run-summary formatting helpers.

Formatting-only helpers live here so command orchestration can stay separate.
The old private short-summary names are compatibility aliases for the public
domain helpers in ``claim_evaluation`` and ``comparison``.

These formatters expect well-formed summary payloads from
``SpiderSimulation.build_summary``. In particular, ``format_run_summary`` treats
``training_last_window.mean_reward``,
``training_last_window.mean_night_shelter_occupancy_rate``,
``training_last_window.mean_night_stillness_rate``,
``evaluation.mean_reward``, ``evaluation.mean_food``,
``evaluation.mean_sleep``, ``evaluation.mean_sleep_debt``,
``evaluation.mean_predator_contacts``, ``evaluation.mean_predator_escapes``,
``evaluation.mean_night_shelter_occupancy_rate``,
``evaluation.mean_night_stillness_rate``,
``evaluation.mean_predator_response_events``,
``evaluation.mean_predator_response_latency``,
``evaluation.mean_predator_mode_transitions``,
``evaluation.dominant_predator_state``,
``evaluation.mean_food_distance_delta``,
``evaluation.mean_shelter_distance_delta``, ``evaluation.survival_rate``, and
``evaluation.mean_night_role_distribution`` as required when those sections are
present.
"""

from __future__ import annotations

from typing import Mapping

from ..claim_evaluation import condense_claim_test_summary
from ..comparison import condense_robustness_summary
from ..simulation import default_behavior_evaluation


_short_robustness_matrix_summary = condense_robustness_summary
_short_claim_test_suite_summary = condense_claim_test_summary


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def format_scenarios(summary: Mapping[str, object]) -> dict[str, object]:
    scenarios = _mapping(summary.get("scenarios"))
    return {
        name: {
            "mean_reward": data.get("mean_reward"),
            "survival_rate": data.get("survival_rate"),
            "dominant_predator_state": data.get("dominant_predator_state"),
            "mean_predator_mode_transitions": data.get(
                "mean_predator_mode_transitions"
            ),
            "mean_food_distance_delta": data.get("mean_food_distance_delta"),
            "mean_shelter_distance_delta": data.get("mean_shelter_distance_delta"),
        }
        for name, data in scenarios.items()
        if isinstance(data, Mapping)
    }


def format_comparisons(summary: Mapping[str, object]) -> dict[str, object]:
    comparisons = _mapping(summary.get("comparisons"))
    return {
        "seeds": comparisons.get("seeds", []),
        "reward_profiles": {
            name: {
                "mean_reward": data.get("mean_reward"),
                "mean_food": data.get("mean_food"),
                "survival_rate": data.get("survival_rate"),
            }
            for name, data in _mapping(comparisons.get("reward_profiles")).items()
            if isinstance(data, Mapping)
        },
        "map_templates": {
            name: {
                "mean_reward": data.get("mean_reward"),
                "mean_food": data.get("mean_food"),
                "survival_rate": data.get("survival_rate"),
            }
            for name, data in _mapping(comparisons.get("map_templates")).items()
            if isinstance(data, Mapping)
        },
    }


def format_reward_audit(summary: Mapping[str, object]) -> dict[str, object]:
    reward_audit = _mapping(summary.get("reward_audit"))
    observation_signals = _mapping(reward_audit.get("observation_signals"))
    memory_signals = _mapping(reward_audit.get("memory_signals"))
    return {
        "current_profile": reward_audit.get("current_profile"),
        "minimal_profile": reward_audit.get("minimal_profile"),
        "high_risk_observation_signals": sorted(
            name
            for name, data in observation_signals.items()
            if isinstance(data, Mapping) and data.get("risk") == "high"
        ),
        "high_risk_memory_signals": sorted(
            name
            for name, data in memory_signals.items()
            if isinstance(data, Mapping) and data.get("risk") == "high"
        ),
    }


def format_behavior_comparisons(
    behavior_evaluation: Mapping[str, object],
) -> dict[str, object]:
    comparisons = _mapping(behavior_evaluation.get("comparisons"))
    return {
        "seeds": comparisons.get("seeds", []),
        "reward_profiles": {
            name: {
                "scenario_success_rate": data["summary"][
                    "scenario_success_rate"
                ],
                "episode_success_rate": data["summary"]["episode_success_rate"],
            }
            for name, data in _mapping(comparisons.get("reward_profiles")).items()
            if isinstance(data, Mapping)
        },
        "map_templates": {
            name: {
                "scenario_success_rate": data["summary"][
                    "scenario_success_rate"
                ],
                "episode_success_rate": data["summary"]["episode_success_rate"],
            }
            for name, data in _mapping(comparisons.get("map_templates")).items()
            if isinstance(data, Mapping)
        },
    }


def format_behavior_ablations(
    behavior_evaluation: Mapping[str, object],
) -> dict[str, object]:
    ablations = _mapping(behavior_evaluation.get("ablations"))
    return {
        "reference_variant": ablations.get("reference_variant"),
        "seeds": ablations.get("seeds", []),
        "variants": {
            name: {
                "architecture": data["config"]["architecture"],
                "scenario_success_rate": data["summary"][
                    "scenario_success_rate"
                ],
                "episode_success_rate": data["summary"]["episode_success_rate"],
            }
            for name, data in _mapping(ablations.get("variants", {})).items()
            if isinstance(data, Mapping)
        },
        "deltas_vs_reference": ablations.get("deltas_vs_reference", {}),
    }


def format_curriculum_comparison(
    behavior_evaluation: Mapping[str, object],
) -> dict[str, object]:
    curriculum = _mapping(behavior_evaluation.get("curriculum_comparison"))
    return {
        "curriculum_profile": curriculum.get("curriculum_profile"),
        "reference_regime": curriculum.get("reference_regime"),
        "focus_scenarios": curriculum.get("focus_scenarios", []),
        "regimes": {
            name: {
                "scenario_success_rate": data["summary"][
                    "scenario_success_rate"
                ],
                "episode_success_rate": data["summary"]["episode_success_rate"],
            }
            for name, data in _mapping(curriculum.get("regimes")).items()
            if isinstance(data, Mapping)
        },
    }


def format_learning_evidence(
    behavior_evaluation: Mapping[str, object],
) -> dict[str, object]:
    learning = _mapping(behavior_evaluation.get("learning_evidence"))
    return {
        "reference_condition": learning.get("reference_condition"),
        "seeds": learning.get("seeds", []),
        "evidence_summary": learning.get("evidence_summary", {}),
        "conditions": {
            name: (
                {
                    "skipped": True,
                    "reason": data.get("reason", ""),
                }
                if data.get("skipped")
                else {
                    "policy_mode": data["policy_mode"],
                    "train_episodes": data["train_episodes"],
                    "checkpoint_source": data["checkpoint_source"],
                    "scenario_success_rate": data["summary"][
                        "scenario_success_rate"
                    ],
                    "episode_success_rate": data["summary"][
                        "episode_success_rate"
                    ],
                }
            )
            for name, data in _mapping(learning.get("conditions")).items()
            if isinstance(data, Mapping)
        },
    }


def format_behavior_evaluation(summary: Mapping[str, object]) -> dict[str, object]:
    behavior = _mapping(summary.get("behavior_evaluation"))
    if (
        "robustness_matrix" in behavior
        and not _mapping(behavior.get("suite"))
        and "comparisons" not in behavior
        and "ablations" not in behavior
        and "learning_evidence" not in behavior
        and "claim_tests" not in behavior
    ):
        printable_behavior = default_behavior_evaluation()
        printable_behavior["robustness_matrix"] = condense_robustness_summary(
            _mapping(behavior.get("robustness_matrix"))
        )
        return printable_behavior

    behavior_summary = _mapping(behavior.get("summary"))
    printable = {
        "scenario_success_rate": behavior_summary.get("scenario_success_rate", 0.0),
        "episode_success_rate": behavior_summary.get("episode_success_rate", 0.0),
        "regressions": behavior_summary.get("regressions", []),
    }
    suite = _mapping(behavior.get("suite"))
    if suite:
        printable["suite"] = {
            name: {
                "success_rate": data["success_rate"],
                "failures": data["failures"],
            }
            for name, data in suite.items()
            if isinstance(data, Mapping)
        }
    if "comparisons" in behavior:
        printable["comparisons"] = format_behavior_comparisons(behavior)
    if "shaping_audit" in behavior:
        shaping = _mapping(behavior.get("shaping_audit"))
        printable["shaping_audit"] = {
            "minimal_profile": shaping.get("minimal_profile"),
            "deltas_vs_minimal": _mapping(
                shaping.get("comparison")
            ).get("deltas_vs_minimal", {}),
        }
    if "ablations" in behavior:
        printable["ablations"] = format_behavior_ablations(behavior)
    if "curriculum_comparison" in behavior:
        printable["curriculum_comparison"] = format_curriculum_comparison(
            behavior
        )
    if "learning_evidence" in behavior:
        printable["learning_evidence"] = format_learning_evidence(behavior)
    if "claim_tests" in behavior:
        printable["claim_tests"] = condense_claim_test_summary(
            _mapping(behavior.get("claim_tests"))
        )
    if "robustness_matrix" in behavior:
        printable["robustness_matrix"] = condense_robustness_summary(
            _mapping(behavior.get("robustness_matrix"))
        )
    return printable


def format_checkpointing(summary: Mapping[str, object]) -> dict[str, object]:
    checkpointing = _mapping(summary.get("checkpointing"))
    return {
        "selection": checkpointing.get("selection", "none"),
        "metric": checkpointing.get("metric"),
        "checkpoint_interval": checkpointing.get("checkpoint_interval"),
        "selected_checkpoint": checkpointing.get("selected_checkpoint", {}),
    }


def format_run_summary(
    summary: Mapping[str, object],
    *,
    full: bool = False,
) -> Mapping[str, object]:
    if full:
        return summary

    config = _mapping(summary.get("config"))
    world = _mapping(config.get("world"))
    budget = _mapping(config.get("budget"))
    training_regime = _mapping(config.get("training_regime"))
    operational_profile = _mapping(config.get("operational_profile"))
    noise_profile = _mapping(config.get("noise_profile"))

    printable: dict[str, object] = {
        "reward_profile": world.get("reward_profile"),
        "map_template": world.get("map_template"),
        "budget_profile": budget.get("profile"),
        "benchmark_strength": budget.get("benchmark_strength"),
        "training_regime": training_regime,
        "operational_profile": operational_profile.get("name"),
    }
    if noise_profile:
        printable["noise_profile"] = noise_profile.get("name")

    training = _mapping(summary.get("training_last_window"))
    if training:
        printable.update(
            {
                "training_mean_reward_last_window": training["mean_reward"],
                "training_mean_night_shelter_occupancy_rate_last_window": training[
                    "mean_night_shelter_occupancy_rate"
                ],
                "training_mean_night_stillness_rate_last_window": training[
                    "mean_night_stillness_rate"
                ],
            }
        )

    evaluation = _mapping(summary.get("evaluation"))
    if evaluation:
        printable.update(
            {
                "evaluation_mean_reward": evaluation["mean_reward"],
                "evaluation_mean_food": evaluation["mean_food"],
                "evaluation_mean_sleep": evaluation["mean_sleep"],
                "evaluation_mean_sleep_debt": evaluation["mean_sleep_debt"],
                "evaluation_mean_predator_contacts": evaluation[
                    "mean_predator_contacts"
                ],
                "evaluation_mean_predator_escapes": evaluation[
                    "mean_predator_escapes"
                ],
                "evaluation_mean_night_shelter_occupancy_rate": evaluation[
                    "mean_night_shelter_occupancy_rate"
                ],
                "evaluation_mean_night_stillness_rate": evaluation[
                    "mean_night_stillness_rate"
                ],
                "evaluation_mean_predator_response_events": evaluation[
                    "mean_predator_response_events"
                ],
                "evaluation_mean_predator_response_latency": evaluation[
                    "mean_predator_response_latency"
                ],
                "evaluation_mean_predator_mode_transitions": evaluation[
                    "mean_predator_mode_transitions"
                ],
                "evaluation_dominant_predator_state": evaluation[
                    "dominant_predator_state"
                ],
                "evaluation_mean_food_distance_delta": evaluation[
                    "mean_food_distance_delta"
                ],
                "evaluation_mean_shelter_distance_delta": evaluation[
                    "mean_shelter_distance_delta"
                ],
                "evaluation_survival_rate": evaluation["survival_rate"],
                "evaluation_mean_night_role_distribution": evaluation[
                    "mean_night_role_distribution"
                ],
            }
        )

    if "scenarios" in summary:
        printable["scenarios"] = format_scenarios(summary)
    if "comparisons" in summary:
        printable["comparisons"] = format_comparisons(summary)
    if "reward_audit" in summary:
        printable["reward_audit"] = format_reward_audit(summary)
    if "behavior_evaluation" in summary:
        printable["behavior_evaluation"] = format_behavior_evaluation(summary)
    if "checkpointing" in summary:
        printable["checkpointing"] = format_checkpointing(summary)
    if "benchmark_package" in summary:
        printable["benchmark_package"] = summary["benchmark_package"]
    return printable


__all__ = [
    "format_behavior_evaluation",
    "format_checkpointing",
    "format_comparisons",
    "format_run_summary",
    "format_scenarios",
]
