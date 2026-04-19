"""Comprehensive tests for the action_center addition introduced in this PR.

Covers:
- architecture_signature() new structure (action_center, motor_cortex keys)
- SpiderBrain.ARCHITECTURE_VERSION = 12
- SpiderBrain.action_center / motor_cortex network types
- BrainStep new fields (action_center_logits, action_center_policy,
  action_intent_idx, motor_override, action_center_input)
- SpiderBrain.estimate_value uses action_center, not motor_cortex alone
- _build_motor_input NaN / inf sanitization
- ActionContextObservation and MotorContextObservation field composition
- ACTION_CONTEXT_INTERFACE / MOTOR_CONTEXT_INTERFACE properties
- OBSERVATION_DIMS updated entries
- build_action_context_observation / build_motor_context_observation
"""

import json
import tempfile
import unittest
from pathlib import Path
from typing import ClassVar
from unittest.mock import patch

import numpy as np

from spider_cortex_sim.ablations import (
    BrainAblationConfig,
    MONOLITHIC_POLICY_NAME,
    canonical_ablation_configs,
    default_brain_config,
)
from spider_cortex_sim.agent import BrainStep, SpiderBrain
from spider_cortex_sim.arbitration import ArbitrationDecision
from spider_cortex_sim.bus import MessageBus
from spider_cortex_sim.interfaces import (
    ACTION_CONTEXT_INTERFACE,
    ACTION_DELTAS,
    ACTION_TO_INDEX,
    LOCOMOTION_ACTIONS,
    MODULE_INTERFACES,
    MOTOR_CONTEXT_INTERFACE,
    MODULE_INTERFACE_BY_NAME,
    OBSERVATION_DIMS,
    ActionContextObservation,
    MotorContextObservation,
    architecture_signature,
)
from spider_cortex_sim.maps import CLUTTER, NARROW, OPEN
from spider_cortex_sim.modules import CorticalModuleBank, ModuleResult
from spider_cortex_sim.nn import ArbitrationNetwork, MotorNetwork, ProposalNetwork
from spider_cortex_sim.noise import compute_execution_difficulty
from spider_cortex_sim.operational_profiles import DEFAULT_OPERATIONAL_PROFILE, OperationalProfile
from spider_cortex_sim.perception import (
    PerceivedTarget,
    build_action_context_observation,
    build_motor_context_observation,
)
from spider_cortex_sim.world import SpiderWorld


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _null_percept() -> PerceivedTarget:
    """Return a PerceivedTarget representing no visible predator."""
    return PerceivedTarget(visible=0.0, certainty=0.0, occluded=0.0, dx=0.0, dy=0.0, dist=10**9)

def _blank_obs() -> dict[str, np.ndarray]:
    """
    Create a zero-filled observation mapping for every module and context interface.

    Constructs a dict whose keys are each interface's `observation_key` (for all entries in
    `MODULE_INTERFACES`, plus `ACTION_CONTEXT_INTERFACE` and `MOTOR_CONTEXT_INTERFACE`) and
    whose values are NumPy arrays of zeros sized to the interface's `input_dim`.

    Returns:
        dict[str, np.ndarray]: Mapping from observation key to zeroed observation vector.
    """
    obs: dict[str, np.ndarray] = {}
    for iface in MODULE_INTERFACES:
        obs[iface.observation_key] = np.zeros(iface.input_dim, dtype=float)
    obs[ACTION_CONTEXT_INTERFACE.observation_key] = np.zeros(
        ACTION_CONTEXT_INTERFACE.input_dim, dtype=float
    )
    obs[MOTOR_CONTEXT_INTERFACE.observation_key] = np.zeros(
        MOTOR_CONTEXT_INTERFACE.input_dim, dtype=float
    )
    return obs

def _module_interface(name: str):
    """
    Retrieve the module interface spec with the given name.

    Parameters:
        name (str): The name of the module interface to find.

    Returns:
        The matching module interface spec.

    Raises:
        StopIteration: If no module interface with the given name exists.
    """
    return next(spec for spec in MODULE_INTERFACES if spec.name == name)

def _vector_for(interface, **updates) -> np.ndarray:
    """
    Create a vector for the given interface by filling unspecified signals with 0.0 and applying provided signal-value updates.

    Parameters:
        interface: An interface spec exposing `signal_names` and `vector_from_mapping(mapping)`; `signal_names` defines the ordered signal keys.
        **updates: Signal name → numeric value overrides to apply on top of zeros.

    Returns:
        A numpy ndarray produced by `interface.vector_from_mapping(mapping)` where `mapping` contains all signals from `interface.signal_names` with unspecified signals set to 0.0 and specified signals set to the provided values.
    """
    mapping = {name: 0.0 for name in interface.signal_names}
    mapping.update({key: float(value) for key, value in updates.items()})
    return interface.vector_from_mapping(mapping)

def _valence_vector(**updates: float) -> np.ndarray:
    values = np.zeros(len(SpiderBrain.VALENCE_ORDER), dtype=float)
    index_by_name = {
        name: index
        for index, name in enumerate(SpiderBrain.VALENCE_ORDER)
    }
    for name, value in updates.items():
        values[index_by_name[name]] = float(value)
    return values

def _profile_with_updates(**reward_updates: float) -> OperationalProfile:
    summary = DEFAULT_OPERATIONAL_PROFILE.to_summary()
    summary["name"] = "action_center_test_profile"
    summary["version"] = 21
    summary["reward"].update({name: float(value) for name, value in reward_updates.items()})
    return OperationalProfile.from_summary(summary)
