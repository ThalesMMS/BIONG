from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _float_value(value: object, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(numeric):
        return float(default)
    return float(numeric)


@dataclass(frozen=True)
class LocalActionAffordance:
    blocked: bool
    next_role: str
    next_role_level: float


@dataclass(frozen=True)
class LocalTransitionConsequence:
    food_dist_delta: float = 0.0
    shelter_dist_delta: float = 0.0
    predator_dist_delta: float = 0.0
    next_cell_has_food: bool = False


@dataclass(frozen=True)
class LocalTransitionRollout:
    best_food_dist_delta: float = 0.0
    best_shelter_dist_delta: float = 0.0
    best_predator_dist_delta: float = 0.0
    food_reachable_within_two_steps: bool = False


@dataclass(frozen=True)
class LocalGeodesicConsequence:
    exit_geodesic_delta: float = 0.0
    deep_geodesic_delta: float = 0.0
    next_on_exit_target: bool = False
    next_on_deep_target: bool = False


@dataclass(frozen=True)
class LocalEcologyObservationAdapter:
    meta: Mapping[str, object]

    @classmethod
    def from_observation(
        cls,
        observation: Mapping[str, object],
    ) -> "LocalEcologyObservationAdapter":
        return cls.from_meta(observation.get("meta"))

    @classmethod
    def from_meta(cls, meta: object) -> "LocalEcologyObservationAdapter":
        return cls(_mapping(meta))

    @property
    def shelter_role(self) -> str:
        return str(self.meta.get("shelter_role", "outside"))

    @property
    def shelter_role_level(self) -> float:
        return _float_value(self.meta.get("shelter_role_level"), 0.0)

    @property
    def map_template(self) -> str:
        return str(self.meta.get("map_template", ""))

    @property
    def on_food(self) -> bool:
        return bool(self.meta.get("on_food", False))

    @property
    def on_shelter(self) -> bool:
        return bool(self.meta.get("on_shelter", False))

    def affordance_for(self, action_name: str) -> LocalActionAffordance:
        affordances = _mapping(self.meta.get("local_affordances"))
        payload = _mapping(affordances.get(action_name))
        return LocalActionAffordance(
            blocked=bool(payload.get("blocked", False)),
            next_role=str(payload.get("next_role", self.shelter_role)),
            next_role_level=_float_value(
                payload.get("next_role_level"),
                self.shelter_role_level,
            ),
        )

    def transition_for(self, action_name: str) -> LocalTransitionConsequence:
        transitions = _mapping(self.meta.get("local_transition_consequences"))
        payload = _mapping(transitions.get(action_name))
        return LocalTransitionConsequence(
            food_dist_delta=_float_value(payload.get("food_dist_delta"), 0.0),
            shelter_dist_delta=_float_value(payload.get("shelter_dist_delta"), 0.0),
            predator_dist_delta=_float_value(payload.get("predator_dist_delta"), 0.0),
            next_cell_has_food=bool(payload.get("next_cell_has_food", False)),
        )

    def transition_rollout_for(self, action_name: str) -> LocalTransitionRollout:
        rollouts = _mapping(self.meta.get("local_transition_rollouts"))
        payload = _mapping(rollouts.get(action_name))
        return LocalTransitionRollout(
            best_food_dist_delta=_float_value(
                payload.get("best_food_dist_delta"),
                0.0,
            ),
            best_shelter_dist_delta=_float_value(
                payload.get("best_shelter_dist_delta"),
                0.0,
            ),
            best_predator_dist_delta=_float_value(
                payload.get("best_predator_dist_delta"),
                0.0,
            ),
            food_reachable_within_two_steps=bool(
                payload.get("food_reachable_within_two_steps", False)
            ),
        )

    def geodesic_for(self, action_name: str) -> LocalGeodesicConsequence:
        geodesics = _mapping(self.meta.get("local_geodesic_consequences"))
        payload = _mapping(geodesics.get(action_name))
        return LocalGeodesicConsequence(
            exit_geodesic_delta=_float_value(payload.get("exit_geodesic_delta"), 0.0),
            deep_geodesic_delta=_float_value(payload.get("deep_geodesic_delta"), 0.0),
            next_on_exit_target=bool(payload.get("next_on_exit_target", False)),
            next_on_deep_target=bool(payload.get("next_on_deep_target", False)),
        )

    def spatial_patch_values(
        self,
        key: str,
        *,
        max_count: int | None = None,
    ) -> tuple[object, ...]:
        local_patch = _mapping(self.meta.get("local_spatial_patch"))
        values = local_patch.get(key, ())
        if not isinstance(values, (list, tuple)):
            return ()
        if max_count is None:
            return tuple(values)
        return tuple(values[:max_count])


__all__ = [
    "LocalActionAffordance",
    "LocalEcologyObservationAdapter",
    "LocalGeodesicConsequence",
    "LocalTransitionConsequence",
    "LocalTransitionRollout",
]
