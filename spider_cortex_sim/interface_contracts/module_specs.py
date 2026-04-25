from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .actions import *
from .observations import *

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
        """
        Number of input signals defined by the interface.
        
        Returns:
            int: The count of input SignalSpec entries (length of the interface's inputs).
        """
        return len(self.inputs)

    @property
    def output_dim(self) -> int:
        """
        Return the number of output action signals defined by the interface.
        
        Returns:
            output_count (int): The number of output signals in this interface.
        """
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
        Produce a 1-D NumPy float array of this interface's input signals ordered by the interface's `signal_names`.
        
        Parameters:
            values (Mapping[str, float]): Mapping from input signal name to its scalar value.
        
        Returns:
            np.ndarray: 1-D float array with values ordered to match `self.signal_names`.
        """
        self.validate_signal_mapping(values)
        return np.array(
            [float(values[name]) for name in self.signal_names],
            dtype=float,
        )

    def to_summary(self) -> dict[str, object]:
        """
        Produce a JSON-serializable dictionary describing this interface.
        
        The returned mapping contains identity and version metadata, an ordered list of input signal summaries, the ordered outputs list, and a compatibility sub-dictionary (including the save policy, exact-order/observation-key booleans, and notes).
        
        Returns:
            summary (dict): A dict with keys "name", "observation_key", "role", "version", "description",
                "signals" (list of signal summary dicts), "outputs" (list of output names), and
                "compatibility" (dict with "save_policy", "exact_signal_order_required",
                "exact_output_order_required", "exact_observation_key_required", and "notes").
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
    ModuleInterface(
        name="perception_center",
        observation_key="perception",
        role="proposal",
        version=1,
        description="Coarse perception proposer that fuses visual, olfactory, trace, and perceptual-memory cues.",
        inputs=(
            SignalSpec("food_visible", "1 when food is visible.", 0.0, 1.0),
            SignalSpec("food_certainty", "Visual confidence in the perceived food.", 0.0, 1.0),
            SignalSpec("food_dx", "Horizontal direction of the detected food."),
            SignalSpec("food_dy", "Vertical direction of the detected food."),
            SignalSpec("shelter_visible", "1 when shelter is visible.", 0.0, 1.0),
            SignalSpec("shelter_certainty", "Visual confidence in the perceived shelter.", 0.0, 1.0),
            SignalSpec("shelter_dx", "Horizontal direction of the detected shelter."),
            SignalSpec("shelter_dy", "Vertical direction of the detected shelter."),
            SignalSpec("predator_visible", "1 when a predator is visible.", 0.0, 1.0),
            SignalSpec("predator_certainty", "Visual confidence in the perceived predator.", 0.0, 1.0),
            SignalSpec("predator_dx", "Horizontal direction of the detected predator."),
            SignalSpec("predator_dy", "Vertical direction of the detected predator."),
            SignalSpec("heading_dx", "Current horizontal body or gaze orientation."),
            SignalSpec("heading_dy", "Current vertical body or gaze orientation."),
            SignalSpec("foveal_scan_age", "Normalized ticks since the current foveal heading was actively scanned.", 0.0, 1.0),
            SignalSpec("food_smell_strength", "Intensity of the food scent.", 0.0, 1.0),
            SignalSpec("food_smell_dx", "Horizontal gradient of the food scent."),
            SignalSpec("food_smell_dy", "Vertical gradient of the food scent."),
            SignalSpec("predator_smell_strength", "Intensity of the predator scent.", 0.0, 1.0),
            SignalSpec("predator_smell_dx", "Horizontal gradient of the predator scent."),
            SignalSpec("predator_smell_dy", "Vertical gradient of the predator scent."),
            SignalSpec("light", "Simple light level.", 0.0, 1.0),
            SignalSpec("day", "1 during daytime.", 0.0, 1.0),
            SignalSpec("night", "1 during nighttime.", 0.0, 1.0),
            SignalSpec("food_trace_strength", "Decayed strength of the short food trace.", 0.0, 1.0),
            SignalSpec("food_trace_heading_dx", "Horizontal heading active when the short food trace was refreshed."),
            SignalSpec("food_trace_heading_dy", "Vertical heading active when the short food trace was refreshed."),
            SignalSpec("shelter_trace_strength", "Decayed strength of the short shelter trace.", 0.0, 1.0),
            SignalSpec("shelter_trace_heading_dx", "Horizontal heading active when the short shelter trace was refreshed."),
            SignalSpec("shelter_trace_heading_dy", "Vertical heading active when the short shelter trace was refreshed."),
            SignalSpec("predator_trace_strength", "Decayed strength of the short predator trace.", 0.0, 1.0),
            SignalSpec("predator_trace_heading_dx", "Horizontal heading active when the short predator trace was refreshed."),
            SignalSpec("predator_trace_heading_dy", "Vertical heading active when the short predator trace was refreshed."),
            SignalSpec("food_memory_dx", "Horizontal direction of the last seen food."),
            SignalSpec("food_memory_dy", "Vertical direction of the last seen food."),
            SignalSpec("food_memory_age", "Normalized age of the food memory.", 0.0, 1.0),
            SignalSpec("shelter_memory_dx", "Horizontal direction of the remembered shelter."),
            SignalSpec("shelter_memory_dy", "Vertical direction of the remembered shelter."),
            SignalSpec("shelter_memory_age", "Normalized age of the shelter memory.", 0.0, 1.0),
            SignalSpec("predator_memory_dx", "Horizontal direction of the remembered predator."),
            SignalSpec("predator_memory_dy", "Vertical direction of the remembered predator."),
            SignalSpec("predator_memory_age", "Normalized age of the predator memory.", 0.0, 1.0),
        ),
    ),
    ModuleInterface(
        name="homeostasis_center",
        observation_key="homeostasis",
        role="proposal",
        version=1,
        description="Coarse homeostatic proposer that blends hunger, fatigue, sheltering, and recovery drives.",
        inputs=(
            SignalSpec("hunger", "Internal hunger state.", 0.0, 1.0),
            SignalSpec("fatigue", "Internal fatigue state.", 0.0, 1.0),
            SignalSpec("health", "Body health.", 0.0, 1.0),
            SignalSpec("on_food", "1 when the spider is standing on food.", 0.0, 1.0),
            SignalSpec("on_shelter", "1 when the spider is on the shelter.", 0.0, 1.0),
            SignalSpec("day", "1 during daytime.", 0.0, 1.0),
            SignalSpec("night", "1 during nighttime.", 0.0, 1.0),
            SignalSpec("sleep_phase_level", "Current sleep phase level.", 0.0, 1.0),
            SignalSpec("rest_streak_norm", "Recent rest continuity.", 0.0, 1.0),
            SignalSpec("sleep_debt", "Accumulated sleep debt.", 0.0, 1.0),
            SignalSpec("shelter_role_level", "Current depth inside the shelter.", 0.0, 1.0),
            SignalSpec("food_visible", "1 when food is visible.", 0.0, 1.0),
            SignalSpec("food_certainty", "Visual confidence in the perceived food.", 0.0, 1.0),
            SignalSpec("food_smell_strength", "Intensity of the food scent.", 0.0, 1.0),
            SignalSpec("food_smell_dx", "Horizontal gradient of the food scent."),
            SignalSpec("food_smell_dy", "Vertical gradient of the food scent."),
            SignalSpec("food_trace_dx", "Horizontal direction of the short food trace."),
            SignalSpec("food_trace_dy", "Vertical direction of the short food trace."),
            SignalSpec("food_trace_strength", "Decayed strength of the short food trace.", 0.0, 1.0),
            SignalSpec("food_memory_dx", "Horizontal direction of the last seen food."),
            SignalSpec("food_memory_dy", "Vertical direction of the last seen food."),
            SignalSpec("food_memory_age", "Normalized age of the food memory.", 0.0, 1.0),
            SignalSpec("shelter_trace_dx", "Horizontal direction of the short shelter trace."),
            SignalSpec("shelter_trace_dy", "Vertical direction of the short shelter trace."),
            SignalSpec("shelter_trace_strength", "Decayed strength of the short shelter trace.", 0.0, 1.0),
            SignalSpec("shelter_memory_dx", "Horizontal direction of the remembered shelter."),
            SignalSpec("shelter_memory_dy", "Vertical direction of the remembered shelter."),
            SignalSpec("shelter_memory_age", "Normalized age of the shelter memory.", 0.0, 1.0),
        ),
    ),
    ModuleInterface(
        name="threat_center",
        observation_key="threat",
        role="proposal",
        version=1,
        description="Coarse threat proposer that blends threat sensing, pain/contact, and escape memory.",
        inputs=(
            SignalSpec("predator_visible", "1 when a predator is visible.", 0.0, 1.0),
            SignalSpec("predator_certainty", "Visual confidence in the perceived predator.", 0.0, 1.0),
            SignalSpec("predator_dx", "Horizontal direction of the detected predator."),
            SignalSpec("predator_dy", "Vertical direction of the detected predator."),
            SignalSpec("predator_smell_strength", "Intensity of the predator scent.", 0.0, 1.0),
            SignalSpec("predator_smell_dx", "Horizontal gradient of the predator scent."),
            SignalSpec("predator_smell_dy", "Vertical gradient of the predator scent."),
            SignalSpec("predator_motion_salience", "Explicit motion salience of the predator.", 0.0, 1.0),
            SignalSpec("visual_predator_threat", "Aggregated threat from visually oriented predators.", 0.0, 1.0),
            SignalSpec("olfactory_predator_threat", "Aggregated threat from olfactory predators.", 0.0, 1.0),
            SignalSpec("dominant_predator_none", "1 when no predator type is currently dominant.", 0.0, 1.0),
            SignalSpec("dominant_predator_visual", "1 when visual predators are the dominant threat type.", 0.0, 1.0),
            SignalSpec("dominant_predator_olfactory", "1 when olfactory predators are the dominant threat type.", 0.0, 1.0),
            SignalSpec("recent_pain", "Recent pain.", 0.0, 1.0),
            SignalSpec("recent_contact", "Recent physical contact.", 0.0, 1.0),
            SignalSpec("health", "Body health.", 0.0, 1.0),
            SignalSpec("on_shelter", "1 when the spider is on the shelter.", 0.0, 1.0),
            SignalSpec("night", "1 during nighttime.", 0.0, 1.0),
            SignalSpec("predator_trace_dx", "Horizontal direction of the short predator trace."),
            SignalSpec("predator_trace_dy", "Vertical direction of the short predator trace."),
            SignalSpec("predator_trace_strength", "Decayed strength of the short predator trace.", 0.0, 1.0),
            SignalSpec("predator_memory_dx", "Horizontal direction of the remembered predator."),
            SignalSpec("predator_memory_dy", "Vertical direction of the remembered predator."),
            SignalSpec("predator_memory_age", "Normalized age of the predator memory.", 0.0, 1.0),
            SignalSpec("escape_memory_dx", "Horizontal direction of the remembered escape target."),
            SignalSpec("escape_memory_dy", "Vertical direction of the remembered escape target."),
            SignalSpec("escape_memory_age", "Normalized age of the escape memory.", 0.0, 1.0),
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
    outputs=(),
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
    outputs=(),
)


def _subset_interface(
    parent: ModuleInterface,
    *,
    name: str,
    signal_names: Sequence[str],
    description: str,
) -> ModuleInterface:
    """
    Create a new ModuleInterface containing an ordered subset of signals from a parent interface.
    
    Parameters:
        parent (ModuleInterface): The source interface to derive signals from.
        name (str): Name for the new variant interface.
        signal_names (Sequence[str]): Ordered sequence of unique signal names to include; each name must exist in the parent.
        description (str): Human-readable description for the new interface.
    
    Returns:
        ModuleInterface: A new interface whose inputs are the selected signals (ordered as in the parent) and whose metadata
        (observation_key, role, version, outputs, and compatibility settings) is copied from the parent.
    
    Raises:
        ValueError: If `signal_names` contains duplicates, references names not present in the parent, or if the selection
        does not yield exactly the requested unique signals.
    """
    duplicate_names = sorted(
        {
            signal_name
            for signal_name in signal_names
            if signal_names.count(signal_name) > 1
        }
    )
    if duplicate_names:
        raise ValueError(
            f"Variant interface '{name}' declares duplicate signals: {duplicate_names}."
        )

    parent_by_name = {signal.name: signal for signal in parent.inputs}
    missing = [signal_name for signal_name in signal_names if signal_name not in parent_by_name]
    if missing:
        raise ValueError(
            f"Variant interface '{name}' references signals not present in parent "
            f"'{parent.name}': {missing}."
        )

    requested = set(signal_names)
    inputs = tuple(signal for signal in parent.inputs if signal.name in requested)
    if len(inputs) != len(signal_names):
        raise ValueError(
            f"Variant interface '{name}' expected {len(signal_names)} unique signals, "
            f"selected {len(inputs)} from parent '{parent.name}'."
        )

    return ModuleInterface(
        name=name,
        observation_key=parent.observation_key,
        role=parent.role,
        version=parent.version,
        description=description,
        inputs=inputs,
        outputs=parent.outputs,
        save_compatibility=parent.save_compatibility,
        compatibility_notes=parent.compatibility_notes,
    )


ALL_INTERFACES: tuple[ModuleInterface, ...] = (
    *MODULE_INTERFACES,
    ACTION_CONTEXT_INTERFACE,
    MOTOR_CONTEXT_INTERFACE,
)

MODULE_INTERFACE_BY_NAME = {spec.name: spec for spec in MODULE_INTERFACES}
VARIANT_MODULES: tuple[str, ...] = (
    "visual_cortex",
    "sensory_cortex",
    "hunger_center",
    "sleep_center",
    "alert_center",
)


__all__ = [
    "ACTION_CONTEXT_INTERFACE",
    "ALL_INTERFACES",
    "MODULE_INTERFACES",
    "MODULE_INTERFACE_BY_NAME",
    "MOTOR_CONTEXT_INTERFACE",
    "ModuleInterface",
    "VARIANT_MODULES",
    "_subset_interface",
]
