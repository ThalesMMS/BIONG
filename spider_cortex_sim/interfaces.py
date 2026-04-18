from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, fields
from typing import ClassVar, Dict, Mapping, Sequence, Tuple, TypeVar

import numpy as np

LOCOMOTION_ACTIONS: Sequence[str] = (
    "MOVE_UP",
    "MOVE_DOWN",
    "MOVE_LEFT",
    "MOVE_RIGHT",
    "STAY",
    "ORIENT_UP",
    "ORIENT_DOWN",
    "ORIENT_LEFT",
    "ORIENT_RIGHT",
)
ACTION_TO_INDEX = {name: idx for idx, name in enumerate(LOCOMOTION_ACTIONS)}
ACTION_DELTAS: Dict[str, Tuple[int, int]] = {
    "MOVE_UP": (0, -1),
    "MOVE_DOWN": (0, 1),
    "MOVE_LEFT": (-1, 0),
    "MOVE_RIGHT": (1, 0),
    "STAY": (0, 0),
}
ORIENT_HEADINGS: Dict[str, Tuple[int, int]] = {
    "ORIENT_UP": (0, -1),
    "ORIENT_DOWN": (0, 1),
    "ORIENT_LEFT": (-1, 0),
    "ORIENT_RIGHT": (1, 0),
}


@dataclass(frozen=True)
class SignalSpec:
    name: str
    description: str
    minimum: float = -1.0
    maximum: float = 1.0

    def to_summary(self) -> dict[str, object]:
        """
        Return a JSON-serializable schema entry for this signal.
        """
        return {
            "name": self.name,
            "description": self.description,
            "minimum": float(self.minimum),
            "maximum": float(self.maximum),
        }


def _validate_exact_keys(label: str, expected_names: Sequence[str], values: Mapping[str, float]) -> None:
    """
    Validate that the keys of `values` exactly match `expected_names`.
    
    Parameters:
    	label (str): Human-readable label inserted into the error message when keys do not match.
    	expected_names (Sequence[str]): Ordered sequence of required key names.
    	values (Mapping[str, float]): Mapping whose keys are validated against `expected_names`.
    
    Raises:
    	ValueError: If any expected names are missing or any extra keys are present; the error message lists missing and/or extra keys and includes `label`.
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
        Return a mapping of this observation's dataclass field names to their float values.
        
        Returns:
            dict[str, float]: Mapping from each field name to its value converted to float.
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


@dataclass(frozen=True)
class ModuleInterface:
    """Named contract between the world and a cortical subsystem."""

    name: str
    observation_key: str
    role: str
    inputs: tuple[SignalSpec, ...]
    outputs: tuple[str, ...] = tuple(LOCOMOTION_ACTIONS)
    version: int = 2
    description: str = ""
    save_compatibility: str = "exact_match_required"
    compatibility_notes: tuple[str, ...] = (
        "Changes to the name, observation_key, signal order, version, or outputs require explicit incompatibility handling.",
        "Older checkpoints are rejected; there is no automatic migration.",
    )

    @property
    def input_dim(self) -> int:
        return len(self.inputs)

    @property
    def output_dim(self) -> int:
        return len(self.outputs)

    @property
    def signal_names(self) -> tuple[str, ...]:
        """
        Get the ordered names of the module's input signals.
        
        Returns:
            signal_names (tuple[str, ...]): Tuple of input signal `name` values in the interface's defined order.
        """
        return tuple(signal.name for signal in self.inputs)

    def validate_signal_mapping(self, values: Mapping[str, float]) -> None:
        """
        Validate that the provided mapping contains exactly the interface's expected input signal names.
        
        Parameters:
            values (Mapping[str, float]): Mapping from input signal names to scalar values.
        
        Raises:
            ValueError: If any expected signal names are missing or any unexpected names are present.
        """
        _validate_exact_keys(f"Interface '{self.name}'", self.signal_names, values)

    def bind_values(self, values: Sequence[float]) -> dict[str, float]:
        """
        Bind an ordered sequence of input scalars to this interface's input signal names.
        
        Parameters:
            values (Sequence[float]): Numeric inputs in the order of the interface's `inputs`; length must equal `input_dim`.
        
        Returns:
            dict[str, float]: Mapping from each input signal name to its corresponding scalar value.
        
        Raises:
            ValueError: If `values` does not have length equal to `input_dim`.
        """
        array = np.asarray(values, dtype=float)
        if array.shape != (self.input_dim,):
            raise ValueError(
                f"Interface '{self.name}' expected shape {(self.input_dim,)}, received {array.shape}."
            )
        return {
            signal.name: float(array[idx])
            for idx, signal in enumerate(self.inputs)
        }

    def vector_from_mapping(self, values: Mapping[str, float]) -> np.ndarray:
        """
        Build a numeric vector of input signals ordered by this interface's signal names.
        
        Parameters:
            values (Mapping[str, float]): Mapping from input signal name to its scalar value.
        
        Returns:
            np.ndarray: 1-D float array containing the signal values in the order of this interface's `signal_names`.
        """
        self.validate_signal_mapping(values)
        return np.array(
            [float(values[name]) for name in self.signal_names],
            dtype=float,
        )

    def to_summary(self) -> dict[str, object]:
        """
        Return a JSON-serializable schema entry for this interface.
        """
        return {
            "name": self.name,
            "observation_key": self.observation_key,
            "role": self.role,
            "version": int(self.version),
            "description": self.description,
            "signals": [signal.to_summary() for signal in self.inputs],
            "outputs": list(self.outputs),
            "compatibility": {
                "save_policy": self.save_compatibility,
                "exact_signal_order_required": True,
                "exact_output_order_required": True,
                "exact_observation_key_required": True,
                "notes": list(self.compatibility_notes),
            },
        }


MODULE_INTERFACES: tuple[ModuleInterface, ...] = (
    ModuleInterface(
        name="visual_cortex",
        observation_key="visual",
        role="proposal",
        version=7,
        description="Visual proposer driven by food, shelter, and predator cues inside the local visual field.",
        inputs=(
            SignalSpec("food_visible", "1 when food is detected with sufficient confidence in the visual field."),
            SignalSpec("food_certainty", "Visual confidence in the perceived food target.", 0.0, 1.0),
            SignalSpec("food_occluded", "1 when the nearest food is within range but occluded.", 0.0, 1.0),
            SignalSpec("food_dx", "Relative horizontal offset of the detected food."),
            SignalSpec("food_dy", "Relative vertical offset of the detected food."),
            SignalSpec("shelter_visible", "1 when the shelter is visible with sufficient confidence."),
            SignalSpec("shelter_certainty", "Visual confidence in the perceived shelter.", 0.0, 1.0),
            SignalSpec("shelter_occluded", "1 when the shelter is within visual range but occluded.", 0.0, 1.0),
            SignalSpec("shelter_dx", "Relative horizontal offset of the detected shelter."),
            SignalSpec("shelter_dy", "Relative vertical offset of the detected shelter."),
            SignalSpec("predator_visible", "1 when the predator is visible with sufficient confidence."),
            SignalSpec("predator_certainty", "Visual confidence in the perceived predator.", 0.0, 1.0),
            SignalSpec("predator_occluded", "1 when the predator is within visual range but occluded.", 0.0, 1.0),
            SignalSpec("predator_dx", "Relative horizontal offset of the detected predator."),
            SignalSpec("predator_dy", "Relative vertical offset of the detected predator."),
            SignalSpec("heading_dx", "Current horizontal body or gaze orientation."),
            SignalSpec("heading_dy", "Current vertical body or gaze orientation."),
            SignalSpec("foveal_scan_age", "Normalized ticks since the current foveal heading was actively scanned; 0 is fresh and 1 is stale or never scanned.", 0.0, 1.0),
            SignalSpec("food_trace_strength", "Decayed strength of the short food trace.", 0.0, 1.0),
            SignalSpec("food_trace_heading_dx", "Horizontal heading active when the short food trace was refreshed."),
            SignalSpec("food_trace_heading_dy", "Vertical heading active when the short food trace was refreshed."),
            SignalSpec("shelter_trace_strength", "Decayed strength of the short shelter trace.", 0.0, 1.0),
            SignalSpec("shelter_trace_heading_dx", "Horizontal heading active when the short shelter trace was refreshed."),
            SignalSpec("shelter_trace_heading_dy", "Vertical heading active when the short shelter trace was refreshed."),
            SignalSpec("predator_trace_strength", "Decayed strength of the short predator trace.", 0.0, 1.0),
            SignalSpec("predator_trace_heading_dx", "Horizontal heading active when the short predator trace was refreshed."),
            SignalSpec("predator_trace_heading_dy", "Vertical heading active when the short predator trace was refreshed."),
            SignalSpec("predator_motion_salience", "Explicit motion salience of the predator.", 0.0, 1.0),
            SignalSpec("visual_predator_threat", "Aggregated threat from visually oriented predators.", 0.0, 1.0),
            SignalSpec("olfactory_predator_threat", "Aggregated threat from olfactory predators.", 0.0, 1.0),
            SignalSpec("day", "1 during daytime."),
            SignalSpec("night", "1 during nighttime."),
        ),
    ),
    ModuleInterface(
        name="sensory_cortex",
        observation_key="sensory",
        role="proposal",
        description="Sensory proposer that integrates pain, body state, smell, and light.",
        inputs=(
            SignalSpec("recent_pain", "Recent pain caused by a predator attack.", 0.0, 1.0),
            SignalSpec("recent_contact", "Recent physical contact with the predator.", 0.0, 1.0),
            SignalSpec("health", "Body health.", 0.0, 1.0),
            SignalSpec("hunger", "Body hunger.", 0.0, 1.0),
            SignalSpec("fatigue", "Body fatigue.", 0.0, 1.0),
            SignalSpec("food_smell_strength", "Intensity of the food scent.", 0.0, 1.0),
            SignalSpec("food_smell_dx", "Horizontal gradient of the food scent."),
            SignalSpec("food_smell_dy", "Vertical gradient of the food scent."),
            SignalSpec("predator_smell_strength", "Intensity of the predator scent.", 0.0, 1.0),
            SignalSpec("predator_smell_dx", "Horizontal gradient of the predator scent."),
            SignalSpec("predator_smell_dy", "Vertical gradient of the predator scent."),
            SignalSpec("light", "Simple light level.", 0.0, 1.0),
        ),
    ),
    ModuleInterface(
        name="hunger_center",
        observation_key="hunger",
        role="proposal",
        version=4,
        description="Homeostatic proposer aimed at foraging and returning to the last perceived food.",
        inputs=(
            SignalSpec("hunger", "Internal hunger state.", 0.0, 1.0),
            SignalSpec("on_food", "1 when the spider is standing on food.", 0.0, 1.0),
            SignalSpec("food_visible", "1 when food is visible.", 0.0, 1.0),
            SignalSpec("food_certainty", "Visual confidence in the perceived food.", 0.0, 1.0),
            SignalSpec("food_occluded", "1 when nearby food exists but is occluded.", 0.0, 1.0),
            SignalSpec("food_dx", "Horizontal direction of the detected food."),
            SignalSpec("food_dy", "Vertical direction of the detected food."),
            SignalSpec("food_smell_strength", "Intensity of the food scent.", 0.0, 1.0),
            SignalSpec("food_smell_dx", "Horizontal gradient of the food scent."),
            SignalSpec("food_smell_dy", "Vertical gradient of the food scent."),
            SignalSpec("food_trace_dx", "Horizontal direction of the short food trace."),
            SignalSpec("food_trace_dy", "Vertical direction of the short food trace."),
            SignalSpec("food_trace_strength", "Decayed strength of the short food trace.", 0.0, 1.0),
            SignalSpec("food_trace_heading_dx", "Horizontal heading active when the short food trace was refreshed."),
            SignalSpec("food_trace_heading_dy", "Vertical heading active when the short food trace was refreshed."),
            SignalSpec("food_memory_dx", "Horizontal direction of the last seen food."),
            SignalSpec("food_memory_dy", "Vertical direction of the last seen food."),
            SignalSpec("food_memory_age", "Normalized age of the food memory.", 0.0, 1.0),
        ),
    ),
    ModuleInterface(
        name="sleep_center",
        observation_key="sleep",
        role="proposal",
        version=5,
        description="Homeostatic proposer aimed at returning to shelter, resting, and seeking safe depth.",
        inputs=(
            SignalSpec("fatigue", "Internal fatigue state.", 0.0, 1.0),
            SignalSpec("hunger", "Internal hunger state.", 0.0, 1.0),
            SignalSpec("on_shelter", "1 when the spider is on the shelter.", 0.0, 1.0),
            SignalSpec("night", "1 during nighttime.", 0.0, 1.0),
            SignalSpec("health", "Body health.", 0.0, 1.0),
            SignalSpec("recent_pain", "Recent pain.", 0.0, 1.0),
            SignalSpec("sleep_phase_level", "Current sleep phase level.", 0.0, 1.0),
            SignalSpec("rest_streak_norm", "Recent rest continuity.", 0.0, 1.0),
            SignalSpec("sleep_debt", "Accumulated sleep debt.", 0.0, 1.0),
            SignalSpec("shelter_role_level", "Current depth inside the shelter.", 0.0, 1.0),
            SignalSpec("shelter_trace_dx", "Horizontal direction of the short shelter trace."),
            SignalSpec("shelter_trace_dy", "Vertical direction of the short shelter trace."),
            SignalSpec("shelter_trace_strength", "Decayed strength of the short shelter trace.", 0.0, 1.0),
            SignalSpec("shelter_trace_heading_dx", "Horizontal heading active when the short shelter trace was refreshed."),
            SignalSpec("shelter_trace_heading_dy", "Vertical heading active when the short shelter trace was refreshed."),
            SignalSpec("shelter_memory_dx", "Horizontal direction of the nearest visible shelter cell."),
            SignalSpec("shelter_memory_dy", "Vertical direction of the nearest visible shelter cell."),
            SignalSpec("shelter_memory_age", "Normalized age of the nearest visible shelter cell memory.", 0.0, 1.0),
        ),
    ),
    ModuleInterface(
        name="alert_center",
        observation_key="alert",
        role="proposal",
        version=8,
        description="Defensive proposer aimed at threat response, escape, and shelter prioritization under risk.",
        inputs=(
            SignalSpec("predator_visible", "1 when the predator is visible.", 0.0, 1.0),
            SignalSpec("predator_certainty", "Visual confidence in the perceived predator.", 0.0, 1.0),
            SignalSpec("predator_occluded", "1 when the predator is within range but occluded.", 0.0, 1.0),
            SignalSpec("predator_dx", "Horizontal direction of the detected predator."),
            SignalSpec("predator_dy", "Vertical direction of the detected predator."),
            SignalSpec("predator_smell_strength", "Intensity of the predator scent.", 0.0, 1.0),
            SignalSpec("predator_motion_salience", "Explicit motion salience of the predator.", 0.0, 1.0),
            SignalSpec("visual_predator_threat", "Aggregated threat from visually oriented predators.", 0.0, 1.0),
            SignalSpec("olfactory_predator_threat", "Aggregated threat from olfactory predators.", 0.0, 1.0),
            SignalSpec("dominant_predator_none", "1 when no predator type is currently dominant.", 0.0, 1.0),
            SignalSpec("dominant_predator_visual", "1 when visual predators are the dominant threat type.", 0.0, 1.0),
            SignalSpec("dominant_predator_olfactory", "1 when olfactory predators are the dominant threat type.", 0.0, 1.0),
            SignalSpec("recent_pain", "Recent pain.", 0.0, 1.0),
            SignalSpec("recent_contact", "Recent physical contact.", 0.0, 1.0),
            SignalSpec("on_shelter", "1 when the spider is on the shelter.", 0.0, 1.0),
            SignalSpec("night", "1 during nighttime.", 0.0, 1.0),
            SignalSpec("predator_trace_dx", "Horizontal direction of the short predator trace."),
            SignalSpec("predator_trace_dy", "Vertical direction of the short predator trace."),
            SignalSpec("predator_trace_strength", "Decayed strength of the short predator trace.", 0.0, 1.0),
            SignalSpec("predator_trace_heading_dx", "Horizontal heading active when the short predator trace was refreshed."),
            SignalSpec("predator_trace_heading_dy", "Vertical heading active when the short predator trace was refreshed."),
            SignalSpec("predator_memory_dx", "Horizontal direction of the predator position perceived during visual detection or inferred from contact event."),
            SignalSpec("predator_memory_dy", "Vertical direction of the predator position perceived during visual detection or inferred from contact event."),
            SignalSpec("predator_memory_age", "Normalized age of the predator position perceived during visual detection or inferred from contact event.", 0.0, 1.0),
            SignalSpec("escape_memory_dx", "Horizontal direction of the recent escape target derived from movement history without walkability assumptions."),
            SignalSpec("escape_memory_dy", "Vertical direction of the recent escape target derived from movement history without walkability assumptions."),
            SignalSpec("escape_memory_age", "Normalized age of the escape memory derived from movement history without walkability assumptions.", 0.0, 1.0),
        ),
    ),
)


ACTION_CONTEXT_INTERFACE = ModuleInterface(
    name="action_center_context",
    observation_key="action_context",
    role="context",
    version=3,
    description="Raw context used by the action_center to arbitrate competing locomotion proposals.",
    inputs=(
        SignalSpec("hunger", "Body hunger.", 0.0, 1.0),
        SignalSpec("fatigue", "Body fatigue.", 0.0, 1.0),
        SignalSpec("health", "Body health.", 0.0, 1.0),
        SignalSpec("recent_pain", "Recent pain.", 0.0, 1.0),
        SignalSpec("recent_contact", "Recent contact with the predator.", 0.0, 1.0),
        SignalSpec("on_food", "1 when standing on food.", 0.0, 1.0),
        SignalSpec("on_shelter", "1 when standing on the shelter.", 0.0, 1.0),
        SignalSpec("predator_visible", "1 when the predator is visible.", 0.0, 1.0),
        SignalSpec("predator_certainty", "Visual confidence in the predator.", 0.0, 1.0),
        SignalSpec("day", "1 during daytime.", 0.0, 1.0),
        SignalSpec("night", "1 during nighttime.", 0.0, 1.0),
        SignalSpec("last_move_dx", "Last executed horizontal displacement."),
        SignalSpec("last_move_dy", "Last executed vertical displacement."),
        SignalSpec("sleep_debt", "Accumulated sleep debt.", 0.0, 1.0),
        SignalSpec("shelter_role_level", "Current depth inside the shelter.", 0.0, 1.0),
    ),
)


MOTOR_CONTEXT_INTERFACE = ModuleInterface(
    name="motor_cortex_context",
    observation_key="motor_context",
    role="context",
    version=5,
    description="Raw context used by the motor_cortex to execute locomotion intent within local embodiment constraints.",
    inputs=(
        SignalSpec("on_food", "1 when standing on food.", 0.0, 1.0),
        SignalSpec("on_shelter", "1 when standing on the shelter.", 0.0, 1.0),
        SignalSpec("predator_visible", "1 when the predator is visible.", 0.0, 1.0),
        SignalSpec("predator_certainty", "Visual confidence in the predator.", 0.0, 1.0),
        SignalSpec("day", "1 during daytime.", 0.0, 1.0),
        SignalSpec("night", "1 during nighttime.", 0.0, 1.0),
        SignalSpec("last_move_dx", "Last executed horizontal displacement."),
        SignalSpec("last_move_dy", "Last executed vertical displacement."),
        SignalSpec("shelter_role_level", "Current depth inside the shelter.", 0.0, 1.0),
        SignalSpec("heading_dx", "Current horizontal body orientation."),
        SignalSpec("heading_dy", "Current vertical body orientation."),
        SignalSpec("terrain_difficulty", "Local movement cost from the terrain under the spider.", 0.0, 1.0),
        SignalSpec("fatigue", "Body fatigue affecting execution reliability.", 0.0, 1.0),
        SignalSpec("momentum", "Bounded execution momentum state; motor-only body state, not decision-priority context.", 0.0, 1.0),
    ),
)


ALL_INTERFACES: tuple[ModuleInterface, ...] = MODULE_INTERFACES + (
    ACTION_CONTEXT_INTERFACE,
    MOTOR_CONTEXT_INTERFACE,
)

MODULE_INTERFACE_BY_NAME = {spec.name: spec for spec in MODULE_INTERFACES}
OBSERVATION_INTERFACE_BY_KEY: Dict[str, ModuleInterface] = {
    spec.observation_key: spec for spec in ALL_INTERFACES
}


def _build_observation_view_registry() -> Dict[str, type[ObservationView]]:
    """
    Collects all ObservationView subclasses and indexes them by their declared observation_key.
    
    Scans direct subclasses of ObservationView and builds a mapping from each subclass's non-empty
    observation_key to the subclass type.
    
    Returns:
        Dict[str, type[ObservationView]]: Mapping from observation_key to the corresponding ObservationView subclass.
    
    Raises:
        ValueError: If any subclass does not declare a non-empty `observation_key`, or if multiple subclasses
            declare the same `observation_key`.
    """
    registry: Dict[str, type[ObservationView]] = {}
    for view_cls in ObservationView.__subclasses__():
        key = view_cls.observation_key
        if not key:
            raise ValueError(f"ObservationView '{view_cls.__name__}' must declare observation_key.")
        if key in registry:
            raise ValueError(f"Duplicate ObservationView for observation_key '{key}'.")
        registry[key] = view_cls
    return registry


OBSERVATION_VIEW_BY_KEY: Dict[str, type[ObservationView]] = _build_observation_view_registry()
OBSERVATION_DIMS: Dict[str, int] = {
    spec.observation_key: spec.input_dim for spec in ALL_INTERFACES
}


INTERFACE_REGISTRY_SCHEMA_VERSION = 1
ARBITRATION_EVIDENCE_INPUT_DIM = 24
ARBITRATION_HIDDEN_DIM = 32
ARBITRATION_VALENCE_DIM = 4
ARBITRATION_GATE_DIM = 6


def _stable_json(payload: object) -> str:
    """
    Produce a deterministic, compact JSON representation of `payload` suitable for stable hashing.
    
    The serialization sorts object keys, uses compact separators without extra whitespace, and preserves non-ASCII characters.
    
    Returns:
        json_str (str): JSON string representation of `payload`.
    """
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _fingerprint_payload(payload: object) -> str:
    """
    Compute a stable SHA-256 fingerprint for a JSON-serializable payload.
    """
    return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def interface_registry() -> dict[str, object]:
    """
    Return the machine-readable registry that governs all observation interfaces.
    """
    return {
        "schema_version": INTERFACE_REGISTRY_SCHEMA_VERSION,
        "actions": list(LOCOMOTION_ACTIONS),
        "proposal_interfaces": [spec.name for spec in MODULE_INTERFACES],
        "context_interfaces": [
            ACTION_CONTEXT_INTERFACE.name,
            MOTOR_CONTEXT_INTERFACE.name,
        ],
        "interfaces": {
            spec.name: spec.to_summary()
            for spec in ALL_INTERFACES
        },
        "observation_views": {
            key: {
                "field_names": list(view_cls.field_names()),
            }
            for key, view_cls in sorted(OBSERVATION_VIEW_BY_KEY.items())
        },
    }


def interface_registry_fingerprint() -> str:
    """
    Compute a stable fingerprint of the current interface registry.
    
    The fingerprint is the SHA-256 hex digest of a deterministic JSON serialization of the registry produced by interface_registry().
    
    Returns:
        fingerprint (str): Hexadecimal SHA-256 digest representing the current interface registry.
    """
    return _fingerprint_payload(interface_registry())


def architecture_signature(
    *,
    proposal_backend: str = "modular",
    proposal_order: Sequence[str] | None = None,
    learned_arbitration: bool = True,
    arbitration_input_dim: int = ARBITRATION_EVIDENCE_INPUT_DIM,
    arbitration_hidden_dim: int = ARBITRATION_HIDDEN_DIM,
    arbitration_regularization_weight: float = 0.1,
) -> dict[str, object]:
    """
    Generate a JSON-serializable signature describing the agent's observation/action architecture.
    
    Parameters:
        proposal_backend (str): Proposal-stage topology; must be "modular" or "monolithic".
        proposal_order (Sequence[str] | None): Ordered proposal sources for the action-center input. If omitted,
            "modular" uses all MODULE_INTERFACES names in declaration order and "monolithic" uses ["monolithic_policy"].
        learned_arbitration (bool): Whether arbitration uses the learned network or the fixed-formula baseline.
        arbitration_input_dim (int): Size of the concatenated arbitration evidence vector.
        arbitration_hidden_dim (int): Size of the learned arbitration shared hidden layer.
        arbitration_regularization_weight (float): Gate regularization strength applied by the learned arbitration update.
    
    Returns:
        dict[str, object]: Top-level architecture description containing at least:
            - "schema_version": registry schema version.
            - "proposal_backend": the chosen proposal backend.
            - "registry_fingerprint": fingerprint of the current interface registry.
            - "actions": list of locomotion action names.
            - "modules": mapping from proposal interface name to its summary (observation key, inputs, outputs, etc.).
            - "contexts": summaries for action and motor context interfaces.
            - "interface_versions": mapping of interface name to its numeric version.
            - "proposal_order": resolved ordered list of proposal sources.
            - "arbitration_network": learned-arbitration topology and parameter-layout fingerprint.
            - "action_center": action-center specification including context interface, raw input names,
              an "arbitration" object (strategy, valences, and module_roles), inter-stage proposal inputs,
              proposal slot metadata, outputs, and "value_head": True.
            - "motor_cortex": motor-cortex specification including context interface, raw input names,
              inter-stage intent input metadata, outputs, and "value_head": False.
            - "fingerprint": SHA-256 fingerprint of the returned payload.
    """
    if proposal_backend not in {"modular", "monolithic"}:
        raise ValueError("proposal_backend must be 'modular' or 'monolithic'.")
    if proposal_order is None:
        if proposal_backend == "modular":
            proposal_order = [spec.name for spec in MODULE_INTERFACES]
        else:
            proposal_order = ["monolithic_policy"]
    proposal_order = [str(name) for name in proposal_order]
    arbitration_module_roles = {
        "alert_center": "threat",
        "hunger_center": "hunger",
        "sleep_center": "sleep",
        "visual_cortex": "support",
        "sensory_cortex": "support",
        "monolithic_policy": "integrated_policy",
    }
    if proposal_backend == "monolithic":
        filtered_module_roles = (
            {"monolithic_policy": arbitration_module_roles["monolithic_policy"]}
            if "monolithic_policy" in proposal_order
            else {}
        )
    else:
        filtered_module_roles = {
            name: arbitration_module_roles[name]
            for name in proposal_order
            if name in arbitration_module_roles
        }
    arbitration_input_dim = int(arbitration_input_dim)
    arbitration_hidden_dim = int(arbitration_hidden_dim)
    arbitration_parameter_shapes = {
        "W1": [arbitration_hidden_dim, arbitration_input_dim],
        "b1": [arbitration_hidden_dim],
        "W2_valence": [ARBITRATION_VALENCE_DIM, arbitration_hidden_dim],
        "b2_valence": [ARBITRATION_VALENCE_DIM],
        "W2_gate": [ARBITRATION_GATE_DIM, arbitration_hidden_dim],
        "b2_gate": [ARBITRATION_GATE_DIM],
        "W2_value": [1, arbitration_hidden_dim],
        "b2_value": [1],
    }
    arbitration_parameter_fingerprint = _fingerprint_payload(
        {
            "input_dim": arbitration_input_dim,
            "hidden_dim": arbitration_hidden_dim,
            "valence_dim": ARBITRATION_VALENCE_DIM,
            "gate_dim": ARBITRATION_GATE_DIM,
            "parameter_shapes": arbitration_parameter_shapes,
        }
    )
    registry = interface_registry()
    proposal_interfaces = registry["interfaces"]
    payload = {
        "schema_version": INTERFACE_REGISTRY_SCHEMA_VERSION,
        "proposal_backend": proposal_backend,
        "registry_fingerprint": interface_registry_fingerprint(),
        "actions": list(LOCOMOTION_ACTIONS),
        "modules": {
            spec.name: proposal_interfaces[spec.name]
            for spec in MODULE_INTERFACES
        },
        "contexts": {
            ACTION_CONTEXT_INTERFACE.name: proposal_interfaces[ACTION_CONTEXT_INTERFACE.name],
            MOTOR_CONTEXT_INTERFACE.name: proposal_interfaces[MOTOR_CONTEXT_INTERFACE.name],
        },
        "interface_versions": {
            spec.name: int(spec.version)
            for spec in ALL_INTERFACES
        },
        "proposal_order": proposal_order,
        "arbitration_network": {
            "learned": bool(learned_arbitration),
            "input_dim": arbitration_input_dim,
            "hidden_dim": arbitration_hidden_dim,
            "regularization_weight": float(arbitration_regularization_weight),
            "parameter_shapes": arbitration_parameter_shapes,
            "parameter_fingerprint": arbitration_parameter_fingerprint,
        },
        "action_center": {
            "context_interface": ACTION_CONTEXT_INTERFACE.name,
            "observation_key": ACTION_CONTEXT_INTERFACE.observation_key,
            "version": ACTION_CONTEXT_INTERFACE.version,
            "inputs": [signal.name for signal in ACTION_CONTEXT_INTERFACE.inputs],
            "arbitration": {
                "strategy": "priority_gating",
                "valences": ["threat", "hunger", "sleep", "exploration"],
                "module_roles": filtered_module_roles,
            },
            "inter_stage_inputs": [
                {
                    "name": "proposal_logits",
                    "source": name,
                    "encoding": "logits",
                    "size": len(LOCOMOTION_ACTIONS),
                    "signals": list(LOCOMOTION_ACTIONS),
                }
                for name in proposal_order
            ],
            "outputs": list(LOCOMOTION_ACTIONS),
            "proposal_slots": [
                {
                    "module": name,
                    "actions": list(LOCOMOTION_ACTIONS),
                }
                for name in proposal_order
            ],
            "value_head": True,
        },
        "motor_cortex": {
            "context_interface": MOTOR_CONTEXT_INTERFACE.name,
            "observation_key": MOTOR_CONTEXT_INTERFACE.observation_key,
            "version": MOTOR_CONTEXT_INTERFACE.version,
            "inputs": [signal.name for signal in MOTOR_CONTEXT_INTERFACE.inputs],
            "inter_stage_inputs": [
                {
                    "name": "intent",
                    "source": "action_center",
                    "encoding": "one_hot",
                    "size": len(LOCOMOTION_ACTIONS),
                    "signals": list(LOCOMOTION_ACTIONS),
                }
            ],
            "outputs": list(LOCOMOTION_ACTIONS),
            "value_head": False,
        },
    }
    payload["fingerprint"] = _fingerprint_payload(payload)
    return payload


from .interface_docs import render_interfaces_markdown
