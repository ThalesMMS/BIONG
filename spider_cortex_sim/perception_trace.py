from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from math import atan2, degrees
from typing import TYPE_CHECKING, Final, Iterable, Literal, Mapping, Protocol

import numpy as np

from .interfaces import (
    OBSERVATION_DIMS,
    OBSERVATION_INTERFACE_BY_KEY,
    AlertObservation,
    ActionContextObservation,
    HungerObservation,
    MotorContextObservation,
    ObservationView,
    SensoryObservation,
    SleepObservation,
    VisualObservation,
)
from .maps import BLOCKED, CLUTTER, NARROW, OPEN
from .world_types import PerceptTrace

from .perception_geometry import _compute_target_visibility_zone, _percept_trace_decay, _percept_trace_ttl, has_line_of_sight, trace_strength
from .perception_targets import PerceivedTarget

def empty_percept_trace() -> PerceptTrace:
    """
    Create a new empty short-lived percept trace slot.
    """
    return PerceptTrace(target=None, age=0, certainty=0.0, heading_dx=0, heading_dy=0)

def trace_view(world: "SpiderWorld", trace: PerceptTrace) -> dict[str, object]:
    """
    Serialize a percept trace with derived direction and strength metadata.
    """
    strength = trace_strength(world, trace)
    if trace.target is None or strength <= 0.0:
        dx = 0.0
        dy = 0.0
        heading_dx = 0
        heading_dy = 0
    else:
        dx, dy, _ = world._relative(trace.target)
        heading_dx = int(trace.heading_dx)
        heading_dy = int(trace.heading_dy)
    return {
        "target": (
            [int(trace.target[0]), int(trace.target[1])]
            if trace.target is not None and strength > 0.0
            else None
        ),
        "age": int(trace.age),
        "certainty": float(trace.certainty),
        "strength": float(strength),
        "dx": float(dx),
        "dy": float(dy),
        "heading_dx": int(heading_dx),
        "heading_dy": int(heading_dy),
        "ttl": _percept_trace_ttl(world),
        "decay": _percept_trace_decay(world),
    }

def advance_percept_trace(
    world: "SpiderWorld",
    trace: PerceptTrace,
    percept: PerceivedTarget,
    positions: Iterable[tuple[int, int]],
) -> PerceptTrace:
    """
    Refresh or age a short-lived percept trace using the latest raw percept.

    Visible, non-occluded percepts reset the trace only when their position is
    one of the candidate positions, still matches the reported distance from
    the spider, lies inside the current visual field, and has line of sight.
    Otherwise the existing trace ages and expires through configured trace TTL
    and decay.
    """
    if percept.visible > 0.0 and percept.occluded <= 0.0 and percept.position is not None:
        source = world.spider_pos()
        candidate_set = {tuple(pos) for pos in positions}
        percept_dist = int(percept.dist)
        target = tuple(percept.position)
        if target in candidate_set:
            visibility_zone = _compute_target_visibility_zone(world, source, target)
            if (
                world.manhattan(source, target) == percept_dist
                and visibility_zone != "outside"
                and has_line_of_sight(world, source, target)
            ):
                return PerceptTrace(
                    target=(int(target[0]), int(target[1])),
                    age=0,
                    certainty=float(np.clip(percept.certainty, 0.0, 1.0)),
                    heading_dx=int(world.state.heading_dx),
                    heading_dy=int(world.state.heading_dy),
                )

    if trace.target is None:
        return empty_percept_trace()
    aged = PerceptTrace(
        target=trace.target,
        age=int(trace.age) + 1,
        certainty=float(np.clip(trace.certainty, 0.0, 1.0)),
        heading_dx=int(trace.heading_dx),
        heading_dy=int(trace.heading_dy),
    )
    if trace_strength(world, aged) <= 0.0:
        return empty_percept_trace()
    return aged
