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

from .perception_geometry import _max_scan_age, _normalized_scan_age, smell_gradient, visible_object, visible_range
from .perception_predators import _predator_candidates, _predator_view_from_views, _predator_views_by_type_from_sampled_views, _sample_predator_views, compute_per_type_threats, predator_motion_salience
from .perception_targets import DOMINANT_PREDATOR_TYPE_NONE, DOMINANT_PREDATOR_TYPE_OLFACTORY, DOMINANT_PREDATOR_TYPE_VISUAL, NO_TARGET_DISTANCE, PerceivedTarget, TERRAIN_DIFFICULTY
from .perception_trace import trace_view

def _unpack_trace_view(trace_view: Mapping[str, object]) -> tuple[float, float, float, float, float]:
    """
    Return the direction, strength, and scan-heading components from a trace-view mapping.
    
    Parameters:
        trace_view (Mapping[str, object]): Mapping containing keys "dx", "dy", "strength",
            "heading_dx", and "heading_dy".
    
    Returns:
        tuple[float, float, float, float, float]: A tuple
        (dx, dy, strength, heading_dx, heading_dy) converted to floats.
    """
    return (
        float(trace_view["dx"]),
        float(trace_view["dy"]),
        float(trace_view["strength"]),
        float(trace_view["heading_dx"]),
        float(trace_view["heading_dy"]),
    )

def serialize_observation_view(observation_key: str, view: ObservationView) -> np.ndarray:
    """
    Serialize an ObservationView into a fixed-size numpy vector using the interface registered for the given observation key.
    
    Parameters:
        observation_key (str): Key identifying which observation interface to use.
        view (ObservationView): The observation view to serialize; its mapping form will be converted to a vector.
    
    Returns:
        np.ndarray: One-dimensional numpy array containing the serialized observation vector.
    """
    interface = OBSERVATION_INTERFACE_BY_KEY[observation_key]
    return interface.vector_from_mapping(view.as_mapping())

def build_visual_observation(
    *,
    food_view: PerceivedTarget,
    shelter_view: PerceivedTarget,
    predator_view: PerceivedTarget,
    world: "SpiderWorld" | None = None,
    heading_dx: float,
    heading_dy: float,
    foveal_scan_age: float | None = None,
    food_trace_strength: float,
    food_trace_heading: tuple[float, float] = (0.0, 0.0),
    shelter_trace_strength: float,
    shelter_trace_heading: tuple[float, float] = (0.0, 0.0),
    predator_trace_strength: float,
    predator_trace_heading: tuple[float, float] = (0.0, 0.0),
    predator_motion_salience_value: float,
    visual_predator_threat: float,
    olfactory_predator_threat: float,
    day: float,
    night: float,
) -> VisualObservation:
    """
    Constructs a VisualObservation from perceived food, shelter, and predator targets plus heading, scan recency, trace strengths, predator salience/threats, and day/night flags.
    
    Parameters:
        food_view (PerceivedTarget): Perception for the nearest food target; populates food_* fields.
        shelter_view (PerceivedTarget): Perception for the nearest shelter target; populates shelter_* fields.
        predator_view (PerceivedTarget): Perception for the predator; populates predator_* fields.
        world (SpiderWorld | None): Optional world used to derive foveal_scan_age when it is not provided.
        heading_dx (float): Heading x component included as heading_dx.
        heading_dy (float): Heading y component included as heading_dy.
        foveal_scan_age (float | None): Normalized scan age in [0, 1], or None to derive it from world.
        food_trace_strength (float): Trace strength value for food_trace_strength.
        food_trace_heading (tuple[float, float]): Heading active when the food trace was refreshed.
        shelter_trace_strength (float): Trace strength value for shelter_trace_strength.
        shelter_trace_heading (tuple[float, float]): Heading active when the shelter trace was refreshed.
        predator_trace_strength (float): Trace strength value for predator_trace_strength.
        predator_trace_heading (tuple[float, float]): Heading active when the predator trace was refreshed.
        predator_motion_salience_value (float): Motion salience value included as predator_motion_salience.
        visual_predator_threat (float): Visual predator threat score included as visual_predator_threat.
        olfactory_predator_threat (float): Olfactory predator threat score included as olfactory_predator_threat.
        day (float): Day indicator value included as day.
        night (float): Night indicator value included as night.
    
    Returns:
        VisualObservation: Observation whose food_*, shelter_*, and predator_* perception fields are taken from the corresponding PerceivedTarget inputs, and which includes heading, scan recency, trace strengths, predator salience/threats, and day/night values.
    """
    if foveal_scan_age is None:
        foveal_scan_age = _normalized_scan_age(world, heading_dx, heading_dy) if world is not None else 1.0
    food_trace_heading_dx, food_trace_heading_dy = food_trace_heading
    shelter_trace_heading_dx, shelter_trace_heading_dy = shelter_trace_heading
    predator_trace_heading_dx, predator_trace_heading_dy = predator_trace_heading
    return VisualObservation(
        food_visible=food_view.visible,
        food_certainty=food_view.certainty,
        food_occluded=food_view.occluded,
        food_dx=food_view.dx,
        food_dy=food_view.dy,
        shelter_visible=shelter_view.visible,
        shelter_certainty=shelter_view.certainty,
        shelter_occluded=shelter_view.occluded,
        shelter_dx=shelter_view.dx,
        shelter_dy=shelter_view.dy,
        predator_visible=predator_view.visible,
        predator_certainty=predator_view.certainty,
        predator_occluded=predator_view.occluded,
        predator_dx=predator_view.dx,
        predator_dy=predator_view.dy,
        heading_dx=heading_dx,
        heading_dy=heading_dy,
        foveal_scan_age=float(np.clip(foveal_scan_age, 0.0, 1.0)),
        food_trace_strength=food_trace_strength,
        food_trace_heading_dx=food_trace_heading_dx,
        food_trace_heading_dy=food_trace_heading_dy,
        shelter_trace_strength=shelter_trace_strength,
        shelter_trace_heading_dx=shelter_trace_heading_dx,
        shelter_trace_heading_dy=shelter_trace_heading_dy,
        predator_trace_strength=predator_trace_strength,
        predator_trace_heading_dx=predator_trace_heading_dx,
        predator_trace_heading_dy=predator_trace_heading_dy,
        predator_motion_salience=predator_motion_salience_value,
        visual_predator_threat=visual_predator_threat,
        olfactory_predator_threat=olfactory_predator_threat,
        day=day,
        night=night,
    )

def build_sensory_observation(
    world: "SpiderWorld",
    *,
    food_smell_strength: float,
    food_smell_dx: float,
    food_smell_dy: float,
    predator_smell_strength: float,
    predator_smell_dx: float,
    predator_smell_dy: float,
    light: float,
) -> SensoryObservation:
    """
    Builds a SensoryObservation combining the spider's current internal state with external smell and light inputs.
    
    Parameters:
        world (SpiderWorld): World object used to read the spider's recent_pain, recent_contact, health, hunger, and fatigue state.
        food_smell_strength (float): Food smell total strength (0.0-1.0) within sensing radius.
        food_smell_dx (float): Food smell horizontal gradient component (signed, normalized).
        food_smell_dy (float): Food smell vertical gradient component (signed, normalized).
        predator_smell_strength (float): Predator smell total strength (0.0-1.0) within sensing radius.
        predator_smell_dx (float): Predator smell horizontal gradient component (signed, normalized).
        predator_smell_dy (float): Predator smell vertical gradient component (signed, normalized).
        light (float): Ambient light level.
    
    Returns:
        SensoryObservation: Observation populated with recent_pain, recent_contact, health, hunger, fatigue,
        the provided food and predator smell gradient components, and the ambient light value.
    """
    return SensoryObservation(
        recent_pain=world.state.recent_pain,
        recent_contact=world.state.recent_contact,
        health=world.state.health,
        hunger=world.state.hunger,
        fatigue=world.state.fatigue,
        food_smell_strength=food_smell_strength,
        food_smell_dx=food_smell_dx,
        food_smell_dy=food_smell_dy,
        predator_smell_strength=predator_smell_strength,
        predator_smell_dx=predator_smell_dx,
        predator_smell_dy=predator_smell_dy,
        light=light,
    )

def build_hunger_observation(
    world: "SpiderWorld",
    *,
    on_food: float,
    food_view: PerceivedTarget,
    food_smell_strength: float,
    food_smell_dx: float,
    food_smell_dy: float,
    food_trace: tuple[float, float, float, float, float],
    food_memory: tuple[float, float, float],
) -> HungerObservation:
    """Build the hunger observation from food perception, smell, trace, and memory signals."""
    (
        food_trace_dx,
        food_trace_dy,
        food_trace_strength,
        food_trace_heading_dx,
        food_trace_heading_dy,
    ) = food_trace
    food_mem_dx, food_mem_dy, food_mem_age = food_memory
    return HungerObservation(
        hunger=world.state.hunger,
        on_food=on_food,
        food_visible=food_view.visible,
        food_certainty=food_view.certainty,
        food_occluded=food_view.occluded,
        food_dx=food_view.dx,
        food_dy=food_view.dy,
        food_smell_strength=food_smell_strength,
        food_smell_dx=food_smell_dx,
        food_smell_dy=food_smell_dy,
        food_trace_dx=food_trace_dx,
        food_trace_dy=food_trace_dy,
        food_trace_strength=food_trace_strength,
        food_trace_heading_dx=food_trace_heading_dx,
        food_trace_heading_dy=food_trace_heading_dy,
        food_memory_dx=food_mem_dx,
        food_memory_dy=food_mem_dy,
        food_memory_age=food_mem_age,
    )

def build_sleep_observation(
    world: "SpiderWorld",
    *,
    on_shelter: float,
    night: float,
    sleep_phase_level: float,
    rest_streak_norm: float,
    shelter_role_level: float,
    shelter_trace: tuple[float, float, float, float, float],
    shelter_memory: tuple[float, float, float],
) -> SleepObservation:
    """Build the sleep observation from shelter state, circadian state, trace, and memory."""
    (
        shelter_trace_dx,
        shelter_trace_dy,
        shelter_trace_strength,
        shelter_trace_heading_dx,
        shelter_trace_heading_dy,
    ) = shelter_trace
    shelter_mem_dx, shelter_mem_dy, shelter_mem_age = shelter_memory
    return SleepObservation(
        fatigue=world.state.fatigue,
        hunger=world.state.hunger,
        on_shelter=on_shelter,
        night=night,
        health=world.state.health,
        recent_pain=world.state.recent_pain,
        sleep_phase_level=sleep_phase_level,
        rest_streak_norm=rest_streak_norm,
        sleep_debt=world.state.sleep_debt,
        shelter_role_level=shelter_role_level,
        shelter_trace_dx=shelter_trace_dx,
        shelter_trace_dy=shelter_trace_dy,
        shelter_trace_strength=shelter_trace_strength,
        shelter_trace_heading_dx=shelter_trace_heading_dx,
        shelter_trace_heading_dy=shelter_trace_heading_dy,
        shelter_memory_dx=shelter_mem_dx,
        shelter_memory_dy=shelter_mem_dy,
        shelter_memory_age=shelter_mem_age,
    )

def build_alert_observation(
    world: "SpiderWorld",
    *,
    predator_view: PerceivedTarget,
    predator_smell_strength: float,
    predator_motion_salience_value: float,
    visual_predator_threat: float,
    olfactory_predator_threat: float,
    dominant_predator_type: float,
    on_shelter: float,
    night: float,
    predator_trace: tuple[float, float, float, float, float],
    predator_memory: tuple[float, float, float],
    escape_memory: tuple[float, float, float],
) -> AlertObservation:
    """
    Constructs an AlertObservation combining current predator perception, threat metrics, internal state, and trace/memory vectors.
    
    Parameters:
        predator_view (PerceivedTarget): Perceptual summary of the most relevant predator (visible/certainty/occluded/dx/dy).
        predator_smell_strength (float): Olfactory signal strength for the predator at the spider (0-1).
        predator_motion_salience_value (float): Motion-based salience bonus contributed by predator movement.
        visual_predator_threat (float): Aggregated visual-threat score (0-1).
        olfactory_predator_threat (float): Aggregated olfactory-threat score (0-1).
        dominant_predator_type (float): Numeric encoding of the dominant predator detection type (none/visual/olfactory).
        on_shelter (float): Indicator that the spider is on shelter (0 or 1).
        night (float): Night indicator (0 or 1).
        predator_trace (tuple[float, float, float, float, float]): Trace vector from predator sensing as
            (dx, dy, strength, heading_dx, heading_dy).
        predator_memory (tuple[float, float, float]): Stored predator memory as (dx, dy, age).
        escape_memory (tuple[float, float, float]): Stored escape memory as (dx, dy, age).
    
    Returns:
        AlertObservation: A populated AlertObservation containing predator perception fields, threat values, internal pain/contact state, shelter/night flags, predator trace and memory, and escape memory.
    """
    (
        predator_trace_dx,
        predator_trace_dy,
        predator_trace_strength,
        predator_trace_heading_dx,
        predator_trace_heading_dy,
    ) = predator_trace
    predator_mem_dx, predator_mem_dy, predator_mem_age = predator_memory
    escape_mem_dx, escape_mem_dy, escape_mem_age = escape_memory
    dominant_predator_none = 1.0 if dominant_predator_type == DOMINANT_PREDATOR_TYPE_NONE else 0.0
    dominant_predator_visual = 1.0 if dominant_predator_type == DOMINANT_PREDATOR_TYPE_VISUAL else 0.0
    dominant_predator_olfactory = 1.0 if dominant_predator_type == DOMINANT_PREDATOR_TYPE_OLFACTORY else 0.0
    return AlertObservation(
        predator_visible=predator_view.visible,
        predator_certainty=predator_view.certainty,
        predator_occluded=predator_view.occluded,
        predator_dx=predator_view.dx,
        predator_dy=predator_view.dy,
        predator_smell_strength=predator_smell_strength,
        predator_motion_salience=predator_motion_salience_value,
        visual_predator_threat=visual_predator_threat,
        olfactory_predator_threat=olfactory_predator_threat,
        dominant_predator_none=dominant_predator_none,
        dominant_predator_visual=dominant_predator_visual,
        dominant_predator_olfactory=dominant_predator_olfactory,
        recent_pain=world.state.recent_pain,
        recent_contact=world.state.recent_contact,
        on_shelter=on_shelter,
        night=night,
        predator_trace_dx=predator_trace_dx,
        predator_trace_dy=predator_trace_dy,
        predator_trace_strength=predator_trace_strength,
        predator_trace_heading_dx=predator_trace_heading_dx,
        predator_trace_heading_dy=predator_trace_heading_dy,
        predator_memory_dx=predator_mem_dx,
        predator_memory_dy=predator_mem_dy,
        predator_memory_age=predator_mem_age,
        escape_memory_dx=escape_mem_dx,
        escape_memory_dy=escape_mem_dy,
        escape_memory_age=escape_mem_age,
    )

def build_action_context_observation(
    world: "SpiderWorld",
    *,
    on_food: float,
    on_shelter: float,
    predator_view: PerceivedTarget,
    day: float,
    night: float,
    shelter_role_level: float,
) -> ActionContextObservation:
    """Build the action-center arbitration context from local state and perceived threat."""
    return ActionContextObservation(
        hunger=world.state.hunger,
        fatigue=world.state.fatigue,
        health=world.state.health,
        recent_pain=world.state.recent_pain,
        recent_contact=world.state.recent_contact,
        on_food=on_food,
        on_shelter=on_shelter,
        predator_visible=predator_view.visible,
        predator_certainty=predator_view.certainty,
        day=day,
        night=night,
        last_move_dx=float(world.state.last_move_dx),
        last_move_dy=float(world.state.last_move_dy),
        sleep_debt=world.state.sleep_debt,
        shelter_role_level=shelter_role_level,
    )

def build_motor_context_observation(
    world: "SpiderWorld",
    *,
    on_food: float,
    on_shelter: float,
    predator_view: PerceivedTarget,
    day: float,
    night: float,
    shelter_role_level: float,
) -> MotorContextObservation:
    """Build the motor execution context used to execute the action-center intent.

    The context includes local embodiment signals so the motor stage can see
    body orientation, terrain movement cost, fatigue, and momentum.
    """
    terrain_difficulty = TERRAIN_DIFFICULTY.get(
        world.terrain_at(world.spider_pos()),
        0.0,
    )
    return MotorContextObservation(
        on_food=on_food,
        on_shelter=on_shelter,
        predator_visible=predator_view.visible,
        predator_certainty=predator_view.certainty,
        day=day,
        night=night,
        last_move_dx=float(world.state.last_move_dx),
        last_move_dy=float(world.state.last_move_dy),
        shelter_role_level=shelter_role_level,
        heading_dx=float(world.state.heading_dx),
        heading_dy=float(world.state.heading_dy),
        terrain_difficulty=terrain_difficulty,
        fatigue=world.state.fatigue,
        momentum=float(np.clip(world.state.momentum, 0.0, 1.0)),
    )

def observe_world(world: "SpiderWorld") -> dict[str, object]:
    """
    Assembles and serializes the spider's observation vectors and diagnostic metadata from the current world state.
    
    Builds all perception, smell, memory and internal-state views, converts each typed observation into a flat numpy vector, validates expected shapes, and returns those vectors together with a comprehensive "meta" diagnostics mapping.
    
    Parameters:
        world (SpiderWorld): Simulation state and environment used to compute perceptions, gradients, memory vectors, traces, and internal-state features.
    
    Returns:
        dict[str, object]: A dictionary containing:
            - Serialized observation vectors keyed by view name: "visual", "sensory", "hunger", "sleep", "alert", "action_context", "motor_context". "motor_extra" is provided as a copy of "motor_context" for compatibility.
            - "meta": a dict of diagnostic fields including nearest distances, diagnostic-only privileged values under "diagnostic", day/night flags, on_shelter/on_food booleans, predator visibility flag, lizard position and mode, shelter role and terrain at the spider, phase and sleep metrics, heading, configured profiles, predator motion salience, per-vision PerceivedTarget entries under "vision", perceptual trace views under "percept_traces", memory vectors with TTLs under "memory_vectors", and individual memory ages (or None when absent).
    """
    from .memory import MEMORY_TTLS, memory_vector

    _, food_dist = world.nearest(world.food_positions)
    _, shelter_dist = world.nearest(world.shelter_cells)
    phase_sin, phase_cos = world.phase_features()
    day = 0.0 if world.is_night() else 1.0
    night = 1.0 - day
    light = world.light_level()
    on_shelter = float(world.on_shelter())
    on_food = float(world.on_food())
    open_exposed = float(world.shelter_role_at(world.spider_pos()) == "outside")
    shelter_role_level = world.shelter_role_level()
    sleep_phase = world.sleep_phase_level()
    rest_norm = world.rest_streak_norm()
    heading_dx = float(world.state.heading_dx)
    heading_dy = float(world.state.heading_dy)
    food_trace_view = trace_view(world, world.state.food_trace)
    shelter_trace_view = trace_view(world, world.state.shelter_trace)
    predator_trace_view = trace_view(world, world.state.predator_trace)
    (
        food_trace_dx,
        food_trace_dy,
        food_trace_strength,
        food_trace_heading_dx,
        food_trace_heading_dy,
    ) = _unpack_trace_view(food_trace_view)
    (
        shelter_trace_dx,
        shelter_trace_dy,
        shelter_trace_strength,
        shelter_trace_heading_dx,
        shelter_trace_heading_dy,
    ) = _unpack_trace_view(shelter_trace_view)
    (
        predator_trace_dx,
        predator_trace_dy,
        predator_trace_strength,
        predator_trace_heading_dx,
        predator_trace_heading_dy,
    ) = _unpack_trace_view(predator_trace_view)

    vision_radius = visible_range(world)
    food_view = visible_object(world, world.food_positions, radius=vision_radius)
    shelter_view = visible_object(world, world.shelter_cells, radius=vision_radius)
    sampled_predator_views = _sample_predator_views(world, apply_noise=True)
    predator_views_by_type = _predator_views_by_type_from_sampled_views(
        world,
        sampled_predator_views,
    )
    predator_view = _predator_view_from_views(world, predator_views_by_type)
    per_type_threats = compute_per_type_threats(
        world,
        sampled_predator_views=sampled_predator_views,
        predator_views_by_type=predator_views_by_type,
    )
    motion_salience = predator_motion_salience(world, predator_view=predator_view)

    food_smell_strength, food_smell_dx, food_smell_dy, _ = smell_gradient(
        world,
        world.food_positions,
        radius=world.food_smell_range,
    )
    predator_smell_strength, predator_smell_dx, predator_smell_dy, _ = smell_gradient(
        world,
        world.predator_positions(),
        radius=world.predator_smell_range,
    )

    food_mem_dx, food_mem_dy, food_mem_age = memory_vector(world, world.state.food_memory, ttl_name="food")
    shelter_mem_dx, shelter_mem_dy, shelter_mem_age = memory_vector(world, world.state.shelter_memory, ttl_name="shelter")
    predator_mem_dx, predator_mem_dy, predator_mem_age = memory_vector(world, world.state.predator_memory, ttl_name="predator")
    escape_mem_dx, escape_mem_dy, escape_mem_age = memory_vector(world, world.state.escape_memory, ttl_name="escape")

    visual_observation = build_visual_observation(
        food_view=food_view,
        shelter_view=shelter_view,
        predator_view=predator_view,
        world=world,
        heading_dx=heading_dx,
        heading_dy=heading_dy,
        food_trace_strength=food_trace_strength,
        food_trace_heading=(food_trace_heading_dx, food_trace_heading_dy),
        shelter_trace_strength=shelter_trace_strength,
        shelter_trace_heading=(shelter_trace_heading_dx, shelter_trace_heading_dy),
        predator_trace_strength=predator_trace_strength,
        predator_trace_heading=(predator_trace_heading_dx, predator_trace_heading_dy),
        predator_motion_salience_value=motion_salience,
        visual_predator_threat=per_type_threats["visual_predator_threat"],
        olfactory_predator_threat=per_type_threats["olfactory_predator_threat"],
        day=day,
        night=night,
    )
    sensory_observation = build_sensory_observation(
        world,
        food_smell_strength=food_smell_strength,
        food_smell_dx=food_smell_dx,
        food_smell_dy=food_smell_dy,
        predator_smell_strength=predator_smell_strength,
        predator_smell_dx=predator_smell_dx,
        predator_smell_dy=predator_smell_dy,
        light=light,
    )
    hunger_observation = build_hunger_observation(
        world,
        on_food=on_food,
        food_view=food_view,
        food_smell_strength=food_smell_strength,
        food_smell_dx=food_smell_dx,
        food_smell_dy=food_smell_dy,
        food_trace=(
            food_trace_dx,
            food_trace_dy,
            food_trace_strength,
            food_trace_heading_dx,
            food_trace_heading_dy,
        ),
        food_memory=(food_mem_dx, food_mem_dy, food_mem_age),
    )
    sleep_observation = build_sleep_observation(
        world,
        on_shelter=on_shelter,
        night=night,
        sleep_phase_level=sleep_phase,
        rest_streak_norm=rest_norm,
        shelter_role_level=shelter_role_level,
        shelter_trace=(
            shelter_trace_dx,
            shelter_trace_dy,
            shelter_trace_strength,
            shelter_trace_heading_dx,
            shelter_trace_heading_dy,
        ),
        shelter_memory=(shelter_mem_dx, shelter_mem_dy, shelter_mem_age),
    )
    alert_observation = build_alert_observation(
        world,
        predator_view=predator_view,
        predator_smell_strength=predator_smell_strength,
        predator_motion_salience_value=motion_salience,
        visual_predator_threat=per_type_threats["visual_predator_threat"],
        olfactory_predator_threat=per_type_threats["olfactory_predator_threat"],
        dominant_predator_type=per_type_threats["dominant_predator_type"],
        on_shelter=on_shelter,
        night=night,
        predator_trace=(
            predator_trace_dx,
            predator_trace_dy,
            predator_trace_strength,
            predator_trace_heading_dx,
            predator_trace_heading_dy,
        ),
        predator_memory=(predator_mem_dx, predator_mem_dy, predator_mem_age),
        escape_memory=(escape_mem_dx, escape_mem_dy, escape_mem_age),
    )
    action_context_observation = build_action_context_observation(
        world,
        on_food=on_food,
        on_shelter=on_shelter,
        predator_view=predator_view,
        day=day,
        night=night,
        shelter_role_level=shelter_role_level,
    )
    motor_context_observation = build_motor_context_observation(
        world,
        on_food=on_food,
        on_shelter=on_shelter,
        predator_view=predator_view,
        day=day,
        night=night,
        shelter_role_level=shelter_role_level,
    )

    obs = {
        "visual": serialize_observation_view("visual", visual_observation),
        "sensory": serialize_observation_view("sensory", sensory_observation),
        "hunger": serialize_observation_view("hunger", hunger_observation),
        "sleep": serialize_observation_view("sleep", sleep_observation),
        "alert": serialize_observation_view("alert", alert_observation),
        "action_context": serialize_observation_view("action_context", action_context_observation),
        "motor_context": serialize_observation_view("motor_context", motor_context_observation),
    }
    obs["motor_extra"] = obs["motor_context"].copy()

    for key, dim in OBSERVATION_DIMS.items():
        if key not in obs:
            continue
        if obs[key].shape != (dim,):
            raise ValueError(f"Observation '{key}' expected shape {(dim,)}, received {obs[key].shape}")

    # Diagnostic-only privileged values are kept out of network-facing observations
    # but remain available here for audits and offline inspection.
    predator_positions = world.predator_positions()
    diagnostic_predator_dist = min(
        (world.manhattan(world.spider_pos(), position) for position in predator_positions),
        default=NO_TARGET_DISTANCE,
    )
    diagnostic_home_dx, diagnostic_home_dy, diagnostic_home_dist = world._relative(
        world.safest_shelter_target()
    )
    dominant_type_value = float(per_type_threats["dominant_predator_type"])
    if dominant_type_value == DOMINANT_PREDATOR_TYPE_VISUAL:
        dominant_type_label = "visual"
    elif dominant_type_value == DOMINANT_PREDATOR_TYPE_OLFACTORY:
        dominant_type_label = "olfactory"
    else:
        dominant_type_label = "none"

    obs["meta"] = {
        "food_dist": food_dist,
        "shelter_dist": shelter_dist,
        "diagnostic": {
            "diagnostic_predator_dist": diagnostic_predator_dist,
            "diagnostic_home_dx": diagnostic_home_dx,
            "diagnostic_home_dy": diagnostic_home_dy,
            "diagnostic_home_dist": diagnostic_home_dist,
        },
        "predator_dist_visible": predator_view.dist,
        "night": bool(night),
        "day": bool(day),
        "on_shelter": bool(on_shelter),
        "on_food": bool(on_food),
        "predator_visible": bool(predator_view.visible > 0.5),
        "lizard_x": world.lizard.x,
        "lizard_y": world.lizard.y,
        "lizard_mode": world.lizard.mode,
        "open_exposed": bool(open_exposed),
        "phase_sin": phase_sin,
        "phase_cos": phase_cos,
        "sleep_phase": world.state.sleep_phase,
        "sleep_phase_level": sleep_phase,
        "rest_streak": world.state.rest_streak,
        "sleep_debt": world.state.sleep_debt,
        "shelter_role": world.shelter_role_at(world.spider_pos()),
        "shelter_role_level": shelter_role_level,
        "terrain": world.terrain_at(world.spider_pos()),
        "map_template": world.map_template_name,
        "reward_profile": world.reward_profile,
        "noise_profile": world.noise_profile.name,
        "heading": {"dx": int(world.state.heading_dx), "dy": int(world.state.heading_dy)},
        "active_sensing": {
            "foveal_scan_age": float(visual_observation.foveal_scan_age),
            "raw_foveal_scan_age": int(
                world._scan_age_for_heading(world.state.heading_dx, world.state.heading_dy)
            ),
            "max_scan_age": _max_scan_age(world),
        },
        "predator_motion_salience": motion_salience,
        "visual_predator_threat": per_type_threats["visual_predator_threat"],
        "olfactory_predator_threat": per_type_threats["olfactory_predator_threat"],
        "dominant_predator_type": dominant_type_value,
        "dominant_predator_type_label": dominant_type_label,
        "vision": {
            "food": asdict(food_view),
            "shelter": asdict(shelter_view),
            "predator": asdict(predator_view),
            "predators_by_type": {
                style: asdict(view)
                for style, view in predator_views_by_type.items()
            },
        },
        "predators": [
            {
                "index": idx,
                **asdict(predator),
            }
            for idx, predator in enumerate(_predator_candidates(world))
        ],
        "percept_traces": {
            "food": food_trace_view,
            "shelter": shelter_trace_view,
            "predator": predator_trace_view,
        },
        "memory_vectors": {
            "food": {"dx": food_mem_dx, "dy": food_mem_dy, "age": food_mem_age, "ttl": MEMORY_TTLS["food"]},
            "predator": {"dx": predator_mem_dx, "dy": predator_mem_dy, "age": predator_mem_age, "ttl": MEMORY_TTLS["predator"]},
            "shelter": {"dx": shelter_mem_dx, "dy": shelter_mem_dy, "age": shelter_mem_age, "ttl": MEMORY_TTLS["shelter"]},
            "escape": {"dx": escape_mem_dx, "dy": escape_mem_dy, "age": escape_mem_age, "ttl": MEMORY_TTLS["escape"]},
        },
        "food_memory_age": world.state.food_memory.age if world.state.food_memory.target is not None else None,
        "predator_memory_age": world.state.predator_memory.age if world.state.predator_memory.target is not None else None,
        "shelter_memory_age": world.state.shelter_memory.age if world.state.shelter_memory.target is not None else None,
        "escape_memory_age": world.state.escape_memory.age if world.state.escape_memory.target is not None else None,
    }
    return obs
