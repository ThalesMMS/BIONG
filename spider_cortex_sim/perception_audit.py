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

OBSERVATION_LEAKAGE_AUDIT = {
    "predator_visible_and_direction": {
        "classification": "direct_perception",
        "risk": "low",
        "modules": ["visual", "alert", "action_context", "motor_context"],
        "source": "spider_cortex_sim.perception.predator_visible_to_spider",
        "notes": "Derived from line of sight, certainty, and relative direction, so it is a direct ecological signal.",
    },
    "food_and_predator_smell_gradients": {
        "classification": "direct_perception",
        "risk": "low",
        "modules": ["sensory", "hunger", "alert"],
        "source": "spider_cortex_sim.perception.smell_gradient",
        "notes": "These are local sensory signals with explicit noise and no direct access to the optimal path.",
    },
    "predator_dist": {
        "classification": "privileged_world_signal",
        "risk": "resolved",
        "status": "removed_from_observations",
        "modules": ["alert", "action_context", "motor_context"],
        "source": "spider_cortex_sim.perception.observe_world",
        "evidence": "obs['meta']['diagnostic']['diagnostic_predator_dist']",
        "notes": "Removed from network-facing observations and retained only as diagnostic metadata for leakage audits and offline analysis.",
    },
    "home_vector": {
        "classification": "world_derived_navigation_hint",
        "risk": "resolved",
        "status": "removed_from_observations",
        "modules": ["sleep", "alert"],
        "source": "spider_cortex_sim.perception.observe_world",
        "evidence": "obs['meta']['diagnostic'] contains diagnostic_home_dx, diagnostic_home_dy, and diagnostic_home_dist",
        "notes": "Removed from network-facing observations and retained only as diagnostic metadata for leakage audits and offline analysis.",
    },
    "predator_memory_vector": {
        "classification": "plausible_memory",
        "risk": "low",
        "modules": ["alert"],
        "source": "spider_cortex_sim.memory.refresh_memory + spider_cortex_sim.memory.memory_vector",
        "notes": "Derived from predator memory that is written only from visual perception or local contact events.",
    },
    "shelter_memory_vector": {
        "classification": "plausible_memory",
        "risk": "low",
        "modules": ["sleep"],
        "source": "spider_cortex_sim.memory.refresh_memory + spider_cortex_sim.memory.memory_vector",
        "notes": "Derived from shelter memory that stores perceived shelter cells rather than a world-selected best shelter.",
    },
    "escape_memory_vector": {
        "classification": "plausible_memory",
        "risk": "low",
        "modules": ["alert"],
        "source": "spider_cortex_sim.memory.refresh_memory + spider_cortex_sim.memory.memory_vector",
        "notes": "Derived from movement history and world-boundary clamping without walkability or shelter-policy queries.",
    },
    "foveal_scan_age": {
        "classification": "self_knowledge",
        "risk": "low",
        "modules": ["visual"],
        "source": "spider_cortex_sim.world._scan_age_for_heading",
        "notes": "Derived from the spider's own heading-change history and does not expose hidden target locations.",
    },
}

def observation_leakage_audit() -> dict[str, dict[str, object]]:
    """
    Return a deep-copied audit catalog describing perception-side observation leakage candidates.
    
    The catalog maps leakage identifiers to metadata describing the leakage class, risk level, implicated modules, source identifiers, and evidence or notes; the returned mapping is a deep copy to prevent callers from mutating the module-level catalog.
    
    Returns:
        audit (dict[str, dict[str, object]]): A mapping from leakage keys to metadata dictionaries. Each metadata dictionary includes fields such as classification, risk, modules, sources, and notes.
    """
    return deepcopy(OBSERVATION_LEAKAGE_AUDIT)
