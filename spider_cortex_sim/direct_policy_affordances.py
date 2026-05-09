from __future__ import annotations

AFFORDANCE_SHELTER_ROLE_NAMES: tuple[str, ...] = (
    "outside",
    "entrance",
    "inside",
    "deep",
)

AFFORDANCE_SHELTER_ROLE_TO_INDEX: dict[str, int] = {
    name: index for index, name in enumerate(AFFORDANCE_SHELTER_ROLE_NAMES)
}

AFFORDANCE_GEOMETRY_TARGET_NAMES: tuple[str, ...] = (
    "deepen_shelter",
    "toward_entrance",
    "toward_outside",
)

AFFORDANCE_SHELTER_COLUMN_NAMES: tuple[str, ...] = (
    "outside",
    "left",
    "center",
    "right",
)

AFFORDANCE_SHELTER_COLUMN_TO_INDEX: dict[str, int] = {
    name: index for index, name in enumerate(AFFORDANCE_SHELTER_COLUMN_NAMES)
}

AFFORDANCE_SHELTER_POSITION_NAMES: tuple[str, ...] = (
    "outside",
    "entrance_left",
    "entrance_center",
    "entrance_right",
    "inside_left",
    "inside_center",
    "inside_right",
    "deep_left",
    "deep_center",
    "deep_right",
)

AFFORDANCE_SHELTER_POSITION_TO_INDEX: dict[str, int] = {
    name: index for index, name in enumerate(AFFORDANCE_SHELTER_POSITION_NAMES)
}

DIRECT_POLICY_LOCAL_AFFORDANCE_ACTION_NAMES: tuple[str, ...] = (
    "STAY",
    "MOVE_UP",
    "MOVE_DOWN",
    "MOVE_LEFT",
    "MOVE_RIGHT",
)

DIRECT_POLICY_LOCAL_AFFORDANCE_INPUT_DIM: int = (
    len(AFFORDANCE_SHELTER_ROLE_NAMES)
    + len(DIRECT_POLICY_LOCAL_AFFORDANCE_ACTION_NAMES) * 4
)

DIRECT_POLICY_LOCAL_SPATIAL_PATCH_RADIUS: int = 1
DIRECT_POLICY_LOCAL_SPATIAL_PATCH_WIDTH: int = (
    DIRECT_POLICY_LOCAL_SPATIAL_PATCH_RADIUS * 2 + 1
)
DIRECT_POLICY_LOCAL_SPATIAL_INPUT_DIM: int = (
    DIRECT_POLICY_LOCAL_SPATIAL_PATCH_WIDTH
    * DIRECT_POLICY_LOCAL_SPATIAL_PATCH_WIDTH
    * 3
)

DIRECT_POLICY_LOCAL_TRANSITION_INPUT_DIM: int = (
    len(DIRECT_POLICY_LOCAL_AFFORDANCE_ACTION_NAMES) * 4
)

DIRECT_POLICY_LOCAL_TRANSITION_ROLLOUT_INPUT_DIM: int = (
    len(DIRECT_POLICY_LOCAL_AFFORDANCE_ACTION_NAMES) * 4
)

DIRECT_POLICY_LOCAL_GEODESIC_INPUT_DIM: int = (
    len(DIRECT_POLICY_LOCAL_AFFORDANCE_ACTION_NAMES) * 4
)

DIRECT_POLICY_TRANSITION_PREDICTION_FEATURE_DIM: int = (
    len(DIRECT_POLICY_LOCAL_AFFORDANCE_ACTION_NAMES) * 4
)

DIRECT_POLICY_TRANSITION_ROLLOUT_PREDICTION_FEATURE_DIM: int = (
    len(DIRECT_POLICY_LOCAL_AFFORDANCE_ACTION_NAMES) * 4
)
