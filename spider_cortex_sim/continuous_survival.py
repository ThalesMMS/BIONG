from __future__ import annotations

from typing import Mapping, Sequence

from .metrics import EpisodeStats
from .scenarios.trace_food import _resolve_initial_food_distance

_SHELTER_ROLES = frozenset({"entrance", "inside", "deep"})


def _state_from_trace_item(item: Mapping[str, object]) -> Mapping[str, object]:
    state = item.get("state", {})
    return state if isinstance(state, Mapping) else {}


def _first_trace_state(trace: Sequence[Mapping[str, object]]) -> Mapping[str, object]:
    if not trace:
        return {}
    first = trace[0]
    return _state_from_trace_item(first) if isinstance(first, Mapping) else {}


def _is_shelter_role(role: object) -> bool:
    return isinstance(role, str) and role in _SHELTER_ROLES


def _shelter_transition_counts(
    trace: Sequence[Mapping[str, object]],
) -> tuple[bool, int, int]:
    started_in_shelter = False
    previous_in_shelter: bool | None = None
    shelter_exits = 0
    shelter_returns = 0
    for item in trace:
        if not isinstance(item, Mapping):
            continue
        role = _state_from_trace_item(item).get("shelter_role")
        current_in_shelter = _is_shelter_role(role)
        if previous_in_shelter is None:
            started_in_shelter = current_in_shelter
            previous_in_shelter = current_in_shelter
            continue
        if previous_in_shelter and not current_in_shelter:
            shelter_exits += 1
        elif not previous_in_shelter and current_in_shelter:
            shelter_returns += 1
        previous_in_shelter = current_in_shelter
    return started_in_shelter, shelter_exits, shelter_returns


def _predator_threat_exposure(trace: Sequence[Mapping[str, object]]) -> int:
    exposure_ticks = 0
    for item in trace:
        if not isinstance(item, Mapping):
            continue
        state = _state_from_trace_item(item)
        predator_trace = state.get("predator_trace")
        predator_memory = state.get("predator_memory")
        predator_trace_strength = 0.0
        if isinstance(predator_trace, Mapping):
            predator_trace_strength = float(predator_trace.get("strength", 0.0) or 0.0)
        predator_memory_present = bool(
            isinstance(predator_memory, Mapping) and predator_memory.get("target") is not None
        )
        predator_smell_strength = float(state.get("predator_smell_strength", 0.0) or 0.0)
        if (
            float(state.get("predator_motion_salience", 0.0) or 0.0) > 0.0
            or float(state.get("recent_contact", 0.0) or 0.0) > 0.0
            or predator_trace_strength > 0.0
            or predator_memory_present
            or predator_smell_strength > 0.0
        ):
            exposure_ticks += 1
    return exposure_ticks


def _continuous_survival_tier(
    *,
    scenario_name: str,
    reward_profile: str,
    initial_food_distance: float | None,
    started_in_shelter: bool,
    predator_threat_exposure: int,
) -> int:
    scenario_label = scenario_name.lower()
    reward_label = reward_profile.lower()
    if "bootstrap" in scenario_label or reward_label == "survival_balance":
        return 0
    if started_in_shelter and initial_food_distance is not None and initial_food_distance <= 1.0:
        return 0
    if predator_threat_exposure <= 0:
        return 1
    return 2


def build_continuous_survival_evaluation(
    *,
    stats: EpisodeStats,
    trace: Sequence[Mapping[str, object]],
    day_length: int,
    night_length: int,
    target_days: int = 10,
    scenario_name: str = "",
    reward_profile: str = "",
) -> dict[str, object]:
    """
    Build a single-episode continuous-survival summary.

    The summary is intentionally explicit about the day/night cycle so the
    10-day gate can be audited from JSON without reconstructing the runner.
    """
    ticks_per_day = max(1, int(day_length) + int(night_length))
    steps = max(0, int(stats.steps))
    simulated_days_survived = float(steps / ticks_per_day)
    completed_day_night_cycles = int(steps // ticks_per_day)
    first_state = _first_trace_state(trace)
    started_in_shelter, shelter_exits, shelter_returns = _shelter_transition_counts(trace)
    predator_threat_exposure = _predator_threat_exposure(trace)
    initial_food_distance = _resolve_initial_food_distance(trace)

    initial_hunger = float(first_state.get("hunger", stats.final_hunger))
    initial_fatigue = float(first_state.get("fatigue", stats.final_fatigue))
    initial_sleep_debt = float(first_state.get("sleep_debt", stats.final_sleep_debt))
    final_hunger = float(stats.final_hunger)
    final_fatigue = float(stats.final_fatigue)
    final_sleep_debt = float(stats.final_sleep_debt)

    food_cycle_detected = bool(
        stats.food_eaten > 0 or final_hunger < initial_hunger - 1e-6
    )
    rest_cycle_detected = bool(
        stats.sleep_events > 0 or final_sleep_debt < initial_sleep_debt - 1e-6
    )
    predator_contacts = int(stats.predator_contacts)
    predator_contact_rate = float(predator_contacts / steps) if steps else 0.0
    predator_contacts_low = bool(
        predator_contacts <= max(3, int(target_days) // 2)
        and predator_contact_rate <= 0.05
    )
    health_ok = bool(float(stats.final_health) > 0.0)
    hunger_ok = bool(final_hunger < 0.95)
    fatigue_ok = bool(final_fatigue < 0.95)
    ecology_ok = bool(food_cycle_detected and rest_cycle_detected and predator_contacts_low)
    continuous_survival_passed = bool(
        health_ok
        and hunger_ok
        and fatigue_ok
        and ecology_ok
        and completed_day_night_cycles >= int(target_days)
    )

    deep_shelter_occupancy_rate = float(stats.night_role_distribution.get("deep", 0.0))
    shelter_occupancy_rate = float(stats.night_shelter_occupancy_rate)
    hunger_delta = float(initial_hunger - final_hunger)
    fatigue_delta = float(initial_fatigue - final_fatigue)
    sleep_debt_delta = float(initial_sleep_debt - final_sleep_debt)
    continuous_survival_tier = _continuous_survival_tier(
        scenario_name=scenario_name,
        reward_profile=reward_profile,
        initial_food_distance=initial_food_distance,
        started_in_shelter=started_in_shelter,
        predator_threat_exposure=predator_threat_exposure,
    )

    return {
        "target_days": int(target_days),
        "ticks_per_day": ticks_per_day,
        "simulated_day": float(min(float(target_days), simulated_days_survived)),
        "simulated_days_survived": round(simulated_days_survived, 6),
        "completed_day_night_cycles": completed_day_night_cycles,
        "continuous_survival_passed": continuous_survival_passed,
        "health_ok": health_ok,
        "hunger_ok": hunger_ok,
        "fatigue_ok": fatigue_ok,
        "ecology_ok": ecology_ok,
        "food_cycle_detected": food_cycle_detected,
        "rest_cycle_detected": rest_cycle_detected,
        "predator_contacts_low": predator_contacts_low,
        "predator_contact_rate": round(predator_contact_rate, 6),
        "predator_contacts": predator_contacts,
        "predator_escapes": int(stats.predator_escapes),
        "final_health": float(stats.final_health),
        "final_hunger": final_hunger,
        "final_fatigue": final_fatigue,
        "final_sleep_debt": final_sleep_debt,
        "initial_hunger": round(initial_hunger, 6),
        "initial_fatigue": round(initial_fatigue, 6),
        "initial_sleep_debt": round(initial_sleep_debt, 6),
        "hunger_delta": round(hunger_delta, 6),
        "fatigue_delta": round(fatigue_delta, 6),
        "sleep_debt_delta": round(sleep_debt_delta, 6),
        "food_eaten": int(stats.food_eaten),
        "sleep_events": int(stats.sleep_events),
        "rest_events": int(stats.sleep_events),
        "shelter_entries": int(stats.shelter_entries),
        "started_in_shelter": started_in_shelter,
        "shelter_exits": shelter_exits,
        "shelter_returns": shelter_returns,
        "predator_threat_exposure": predator_threat_exposure,
        "initial_food_distance": initial_food_distance,
        "night_ticks": int(stats.night_ticks),
        "night_shelter_occupancy_rate": shelter_occupancy_rate,
        "night_stillness_rate": float(stats.night_stillness_rate),
        "deep_shelter_occupancy_rate": deep_shelter_occupancy_rate,
        "continuous_survival_tier": continuous_survival_tier,
    }
