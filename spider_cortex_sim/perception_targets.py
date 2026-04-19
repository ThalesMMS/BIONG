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

VisibilityZone = Literal["foveal", "peripheral", "outside"]

class HasPosition(Protocol):
    x: float
    y: float

@dataclass
class PerceivedTarget:
    """Perception summary for a single candidate target.

    Visible percepts must carry a concrete grid `position`. Non-visible or
    no-target percepts may keep `position=None`.
    """

    visible: float
    certainty: float
    occluded: float
    dx: float
    dy: float
    dist: int
    position: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        if float(self.visible) > 0.0 and self.position is None:
            raise ValueError("Visible PerceivedTarget requires position.")

NO_TARGET_DISTANCE = 10**9

DOMINANT_PREDATOR_TYPE_NONE = 0.0

DOMINANT_PREDATOR_TYPE_VISUAL = 0.5

DOMINANT_PREDATOR_TYPE_OLFACTORY = 1.0

TERRAIN_DIFFICULTY: Final[Mapping[str, float]] = {
    OPEN: 0.0,
    CLUTTER: 0.4,
    NARROW: 0.7,
}
