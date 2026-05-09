from __future__ import annotations

OPTION_NAMES: tuple[str, ...] = (
    "FORAGE",
    "RETURN_TO_SHELTER",
    "DEEPEN_IN_SHELTER",
    "REST",
    "POST_REST_REACTIVATE",
    "ESCAPE",
)

OPTION_TO_INDEX: dict[str, int] = {
    name: index for index, name in enumerate(OPTION_NAMES)
}
