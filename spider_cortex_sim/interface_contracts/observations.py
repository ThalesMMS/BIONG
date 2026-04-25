from __future__ import annotations

from dataclasses import dataclass, fields
from typing import ClassVar, Mapping, Sequence, TypeVar

from .actions import *

@dataclass(frozen=True)
class SignalSpec:
    name: str
    description: str
    minimum: float = -1.0
    maximum: float = 1.0

    def to_summary(self) -> dict[str, object]:
        """
        Produce a JSON-serializable dictionary describing this signal's metadata.
        
        Returns:
            dict[str, object]: Mapping with keys "name" (str), "description" (str), "minimum" (float), and "maximum" (float).
        """
        return {
            "name": self.name,
            "description": self.description,
            "minimum": float(self.minimum),
            "maximum": float(self.maximum),
        }


def _validate_exact_keys(label: str, expected_names: Sequence[str], values: Mapping[str, float]) -> None:
    """
    Validate that `values` contains exactly the keys listed in `expected_names`.
    
    Parameters:
        label (str): Human-readable label included in the error message on mismatch.
        expected_names (Sequence[str]): Ordered sequence of required key names.
        values (Mapping[str, float]): Mapping whose keys are checked.
    
    Raises:
        ValueError: If any expected names are missing or any extra keys are present. The message includes `label` and lists missing and/or unexpected keys.
    """
    expected = tuple(expected_names)
    received = set(values.keys())
    expected_set = set(expected)

    missing = [name for name in expected if name not in received]
    extra = sorted(name for name in values.keys() if name not in expected_set)
    if not missing and not extra:
        return

    details = []
    if missing:
        details.append(f"missing {missing}")
    if extra:
        details.append(f"unexpected {extra}")
    raise ValueError(f"{label} received incompatible signals: {'; '.join(details)}.")


_ObservationViewT = TypeVar("_ObservationViewT", bound="ObservationView")


class ObservationView:
    observation_key: ClassVar[str]

    def as_mapping(self) -> dict[str, float]:
        """
        Map this observation's dataclass field names to their values cast to float.
        
        Returns:
            mapping (dict[str, float]): A dict mapping each dataclass field name to its value converted to float.
        """
        return {
            field.name: float(getattr(self, field.name))
            for field in fields(self)
        }

    @classmethod
    def field_names(cls) -> tuple[str, ...]:
        """
        Get the dataclass field names for the class in declaration order.
        
        Returns:
            tuple[str, ...]: Tuple of field name strings in declaration order.
        """
        return tuple(field.name for field in fields(cls))

    @classmethod
    def from_mapping(cls: type[_ObservationViewT], values: Mapping[str, float]) -> _ObservationViewT:
        """
        Construct an instance of the observation view from a mapping of signal names to numeric values.
        
        Validates that `values` contains exactly the class's field names, converts each value to `float`, and returns a new instance of `cls`.
        
        Parameters:
            values (Mapping[str, float]): Mapping from each field name to its numeric value.
        
        Returns:
            _ObservationViewT: An instance of `cls` populated with the provided values converted to floats.
        
        Raises:
            ValueError: If `values` is missing expected field names or contains unexpected keys.
        """
        names = cls.field_names()
        _validate_exact_keys(cls.__name__, names, values)
        return cls(**{name: float(values[name]) for name in names})


@dataclass(frozen=True)
class VisualObservation(ObservationView):
    observation_key: ClassVar[str] = "visual"

    food_visible: float
    food_certainty: float
    food_occluded: float
    food_dx: float
    food_dy: float
    shelter_visible: float
    shelter_certainty: float
    shelter_occluded: float
    shelter_dx: float
    shelter_dy: float
    predator_visible: float
    predator_certainty: float
    predator_occluded: float
    predator_dx: float
    predator_dy: float
    heading_dx: float
    heading_dy: float
    foveal_scan_age: float
    food_trace_strength: float
    food_trace_heading_dx: float
    food_trace_heading_dy: float
    shelter_trace_strength: float
    shelter_trace_heading_dx: float
    shelter_trace_heading_dy: float
    predator_trace_strength: float
    predator_trace_heading_dx: float
    predator_trace_heading_dy: float
    predator_motion_salience: float
    visual_predator_threat: float
    olfactory_predator_threat: float
    day: float
    night: float


@dataclass(frozen=True)
class SensoryObservation(ObservationView):
    observation_key: ClassVar[str] = "sensory"

    recent_pain: float
    recent_contact: float
    health: float
    hunger: float
    fatigue: float
    food_smell_strength: float
    food_smell_dx: float
    food_smell_dy: float
    predator_smell_strength: float
    predator_smell_dx: float
    predator_smell_dy: float
    light: float


@dataclass(frozen=True)
class HungerObservation(ObservationView):
    observation_key: ClassVar[str] = "hunger"

    hunger: float
    on_food: float
    food_visible: float
    food_certainty: float
    food_occluded: float
    food_dx: float
    food_dy: float
    food_smell_strength: float
    food_smell_dx: float
    food_smell_dy: float
    food_trace_dx: float
    food_trace_dy: float
    food_trace_strength: float
    food_trace_heading_dx: float
    food_trace_heading_dy: float
    food_memory_dx: float
    food_memory_dy: float
    food_memory_age: float


@dataclass(frozen=True)
class SleepObservation(ObservationView):
    observation_key: ClassVar[str] = "sleep"

    fatigue: float
    hunger: float
    on_shelter: float
    night: float
    health: float
    recent_pain: float
    sleep_phase_level: float
    rest_streak_norm: float
    sleep_debt: float
    shelter_role_level: float
    shelter_trace_dx: float
    shelter_trace_dy: float
    shelter_trace_strength: float
    shelter_trace_heading_dx: float
    shelter_trace_heading_dy: float
    shelter_memory_dx: float
    shelter_memory_dy: float
    shelter_memory_age: float


@dataclass(frozen=True)
class AlertObservation(ObservationView):
    observation_key: ClassVar[str] = "alert"

    predator_visible: float
    predator_certainty: float
    predator_occluded: float
    predator_dx: float
    predator_dy: float
    predator_smell_strength: float
    predator_motion_salience: float
    visual_predator_threat: float
    olfactory_predator_threat: float
    dominant_predator_none: float
    dominant_predator_visual: float
    dominant_predator_olfactory: float
    recent_pain: float
    recent_contact: float
    on_shelter: float
    night: float
    predator_trace_dx: float
    predator_trace_dy: float
    predator_trace_strength: float
    predator_trace_heading_dx: float
    predator_trace_heading_dy: float
    predator_memory_dx: float
    predator_memory_dy: float
    predator_memory_age: float
    escape_memory_dx: float
    escape_memory_dy: float
    escape_memory_age: float


@dataclass(frozen=True)
class PerceptionObservation(ObservationView):
    observation_key: ClassVar[str] = "perception"

    food_visible: float
    food_certainty: float
    food_dx: float
    food_dy: float
    shelter_visible: float
    shelter_certainty: float
    shelter_dx: float
    shelter_dy: float
    predator_visible: float
    predator_certainty: float
    predator_dx: float
    predator_dy: float
    heading_dx: float
    heading_dy: float
    foveal_scan_age: float
    food_smell_strength: float
    food_smell_dx: float
    food_smell_dy: float
    predator_smell_strength: float
    predator_smell_dx: float
    predator_smell_dy: float
    light: float
    day: float
    night: float
    food_trace_strength: float
    food_trace_heading_dx: float
    food_trace_heading_dy: float
    shelter_trace_strength: float
    shelter_trace_heading_dx: float
    shelter_trace_heading_dy: float
    predator_trace_strength: float
    predator_trace_heading_dx: float
    predator_trace_heading_dy: float
    food_memory_dx: float
    food_memory_dy: float
    food_memory_age: float
    shelter_memory_dx: float
    shelter_memory_dy: float
    shelter_memory_age: float
    predator_memory_dx: float
    predator_memory_dy: float
    predator_memory_age: float


@dataclass(frozen=True)
class HomeostasisObservation(ObservationView):
    observation_key: ClassVar[str] = "homeostasis"

    hunger: float
    fatigue: float
    health: float
    on_food: float
    on_shelter: float
    day: float
    night: float
    sleep_phase_level: float
    rest_streak_norm: float
    sleep_debt: float
    shelter_role_level: float
    food_visible: float
    food_certainty: float
    food_smell_strength: float
    food_smell_dx: float
    food_smell_dy: float
    food_trace_dx: float
    food_trace_dy: float
    food_trace_strength: float
    food_memory_dx: float
    food_memory_dy: float
    food_memory_age: float
    shelter_trace_dx: float
    shelter_trace_dy: float
    shelter_trace_strength: float
    shelter_memory_dx: float
    shelter_memory_dy: float
    shelter_memory_age: float


@dataclass(frozen=True)
class ThreatObservation(ObservationView):
    observation_key: ClassVar[str] = "threat"

    predator_visible: float
    predator_certainty: float
    predator_dx: float
    predator_dy: float
    predator_smell_strength: float
    predator_smell_dx: float
    predator_smell_dy: float
    predator_motion_salience: float
    visual_predator_threat: float
    olfactory_predator_threat: float
    dominant_predator_none: float
    dominant_predator_visual: float
    dominant_predator_olfactory: float
    recent_pain: float
    recent_contact: float
    health: float
    on_shelter: float
    night: float
    predator_trace_dx: float
    predator_trace_dy: float
    predator_trace_strength: float
    predator_memory_dx: float
    predator_memory_dy: float
    predator_memory_age: float
    escape_memory_dx: float
    escape_memory_dy: float
    escape_memory_age: float


@dataclass(frozen=True)
class ActionContextObservation(ObservationView):
    observation_key: ClassVar[str] = "action_context"

    hunger: float
    fatigue: float
    health: float
    recent_pain: float
    recent_contact: float
    on_food: float
    on_shelter: float
    predator_visible: float
    predator_certainty: float
    day: float
    night: float
    last_move_dx: float
    last_move_dy: float
    sleep_debt: float
    shelter_role_level: float


@dataclass(frozen=True)
class MotorContextObservation(ObservationView):
    """
    Execution-relevant context for motor correction.

    `momentum` is field #14 and is a bounded body-state scalar in [0, 1].
    It belongs to motor execution context, not action-center priority context.
    Runtime momentum dynamics are defined by the embodiment design and are
    wired by the world execution implementation.
    """

    observation_key: ClassVar[str] = "motor_context"

    on_food: float
    on_shelter: float
    predator_visible: float
    predator_certainty: float
    day: float
    night: float
    last_move_dx: float
    last_move_dy: float
    shelter_role_level: float
    heading_dx: float
    heading_dy: float
    terrain_difficulty: float
    fatigue: float
    momentum: float = 0.0

__all__ = [
    "ActionContextObservation",
    "AlertObservation",
    "HomeostasisObservation",
    "HungerObservation",
    "MotorContextObservation",
    "ObservationView",
    "PerceptionObservation",
    "SensoryObservation",
    "SignalSpec",
    "SleepObservation",
    "ThreatObservation",
    "VisualObservation",
    "_validate_exact_keys",
]
