from __future__ import annotations


EVENT_TYPE_NAMES: tuple[str, ...] = (
    "FOOD_EATEN",
    "SHELTER_EXIT",
    "SHELTER_RETURN",
    "REST_STARTED",
    "DEEP_SLEEP_REACHED",
    "RECOVERY_COMPLETED",
    "POST_REST_RELEASE_ATTEMPT",
    "BLOCKED_MOVE",
    "ACUTE_PREDATOR_THREAT",
    "RESIDUAL_PREDATOR_MEMORY",
)

EVENT_TYPE_TO_INDEX: dict[str, int] = {
    name: index for index, name in enumerate(EVENT_TYPE_NAMES)
}
