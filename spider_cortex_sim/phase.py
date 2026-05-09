from __future__ import annotations

from typing import Mapping


PHASE_LABELS: tuple[str, ...] = (
    "INITIAL_FORAGE",
    "RETURNING_TO_SHELTER",
    "RESTING",
    "RECOVERED_IN_SHELTER",
    "POST_REST_REACTIVATE",
    "LATE_FORAGE",
    "RETURN_AFTER_LATE_FORAGE",
    "ESCAPE_OR_ACUTE_THREAT",
)

PHASE_TO_INDEX: dict[str, int] = {
    label: index for index, label in enumerate(PHASE_LABELS)
}


def derive_phase_target(
    *,
    state: Mapping[str, object],
    observation_meta: Mapping[str, object],
) -> str:
    sleep_phase = str(
        state.get("sleep_phase")
        or observation_meta.get("sleep_phase")
        or "AWAKE"
    )
    on_shelter = bool(observation_meta.get("on_shelter", False))
    is_day = bool(observation_meta.get("day", False))
    is_night = bool(observation_meta.get("night", False))
    predator_visible = bool(observation_meta.get("predator_visible", False))
    recent_contact = float(state.get("recent_contact", 0.0) or 0.0)
    recent_pain = float(state.get("recent_pain", 0.0) or 0.0)
    visual_threat = float(observation_meta.get("visual_predator_threat", 0.0) or 0.0)
    olfactory_threat = float(observation_meta.get("olfactory_predator_threat", 0.0) or 0.0)
    food_eaten = int(state.get("food_eaten", 0) or 0)
    sleep_events = int(state.get("sleep_events", 0) or 0)
    hunger = float(state.get("hunger", 0.0) or 0.0)
    fatigue = float(state.get("fatigue", 0.0) or 0.0)
    sleep_debt = float(state.get("sleep_debt", 0.0) or 0.0)
    rest_streak = int(state.get("rest_streak", 0) or 0)

    if (
        recent_contact > 0.0
        or recent_pain > 0.0
        or predator_visible
        or visual_threat >= 0.5
        or (not on_shelter and olfactory_threat >= 0.7)
    ):
        return "ESCAPE_OR_ACUTE_THREAT"

    if (
        on_shelter
        and is_day
        and sleep_events > 0
        and sleep_phase == "AWAKE"
    ):
        return "POST_REST_REACTIVATE"

    recovered_in_shelter = (
        on_shelter
        and is_day
        and sleep_events > 0
        and fatigue <= 0.20
        and sleep_debt <= 0.30
    )
    if recovered_in_shelter:
        return "RECOVERED_IN_SHELTER"

    if on_shelter and (
        sleep_phase in {"RESTING", "DEEP_SLEEP"} or rest_streak > 0
    ):
        return "RESTING"

    if sleep_events > 0:
        if not on_shelter:
            if is_night or hunger < 0.45:
                return "RETURN_AFTER_LATE_FORAGE"
            return "LATE_FORAGE"
        return "RETURN_AFTER_LATE_FORAGE"

    if not on_shelter and food_eaten > 0 and (is_night or hunger < 0.35):
        return "RETURNING_TO_SHELTER"

    if on_shelter and food_eaten > 0:
        return "RETURNING_TO_SHELTER"

    return "INITIAL_FORAGE"
