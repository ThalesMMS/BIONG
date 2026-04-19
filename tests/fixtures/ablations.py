from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from spider_cortex_sim.ablations import (
    BrainAblationConfig,
    MODULE_NAMES,
    MULTI_PREDATOR_SCENARIOS,
    MULTI_PREDATOR_SCENARIO_GROUPS,
    VISUAL_PREDATOR_SCENARIOS,
    OLFACTORY_PREDATOR_SCENARIOS,
    MONOLITHIC_POLICY_NAME,
    canonical_ablation_configs,
    canonical_ablation_scenario_groups,
    compare_predator_type_ablation_performance,
    canonical_ablation_variant_names,
    default_brain_config,
    resolve_ablation_scenario_group,
    resolve_ablation_configs,
    _safe_float,
    _mean,
    _scenario_success_rate,
)
from spider_cortex_sim.agent import SpiderBrain
from spider_cortex_sim.interfaces import (
    ACTION_CONTEXT_INTERFACE,
    LOCOMOTION_ACTIONS,
    MODULE_INTERFACE_BY_NAME,
    MODULE_INTERFACES,
    MOTOR_CONTEXT_INTERFACE,
)
from spider_cortex_sim.modules import CorticalModuleBank, ModuleResult
from spider_cortex_sim.nn import RecurrentProposalNetwork
from spider_cortex_sim.operational_profiles import DEFAULT_OPERATIONAL_PROFILE, OperationalProfile
from spider_cortex_sim.simulation import SpiderSimulation

def _blank_mapping(interface) -> dict[str, float]:
    """
    Map each input signal name to a default numeric value where signals ending with "_age" map to 1.0 and all others map to 0.0.
    
    Parameters:
        interface: An object with an iterable `inputs` attribute whose items have a `name` string.
    
    Returns:
        dict[str, float]: A mapping from input signal name to 1.0 for names ending with "_age", otherwise 0.0.
    """
    return {
        signal.name: (1.0 if signal.name.endswith("_age") else 0.0)
        for signal in interface.inputs
    }

def _build_observation(**overrides: dict[str, float]) -> dict[str, np.ndarray]:
    """
    Builds a complete observation dict for all module, action-context, and motor-context interfaces, applying any per-interface signal overrides.
    
    Parameters:
        overrides (dict[str, dict[str, float]]): Optional per-interface overrides keyed by each interface's
            `observation_key`. Each value is a mapping from signal name to numeric value that will replace
            the default for that signal.
    
    Returns:
        dict[str, np.ndarray]: A mapping from each interface's `observation_key` to its vectorized observation
        (NumPy array).
    """
    observation: dict[str, np.ndarray] = {}
    for interface in MODULE_INTERFACES:
        mapping = _blank_mapping(interface)
        mapping.update({name: float(value) for name, value in overrides.get(interface.observation_key, {}).items()})
        observation[interface.observation_key] = interface.vector_from_mapping(mapping)
    action_mapping = _blank_mapping(ACTION_CONTEXT_INTERFACE)
    action_mapping.update({name: float(value) for name, value in overrides.get(ACTION_CONTEXT_INTERFACE.observation_key, {}).items()})
    observation[ACTION_CONTEXT_INTERFACE.observation_key] = ACTION_CONTEXT_INTERFACE.vector_from_mapping(action_mapping)
    motor_mapping = _blank_mapping(MOTOR_CONTEXT_INTERFACE)
    motor_mapping.update({name: float(value) for name, value in overrides.get(MOTOR_CONTEXT_INTERFACE.observation_key, {}).items()})
    observation[MOTOR_CONTEXT_INTERFACE.observation_key] = MOTOR_CONTEXT_INTERFACE.vector_from_mapping(motor_mapping)
    return observation

def _profile_with_updates(**reward_updates: float) -> OperationalProfile:
    """
    Create an OperationalProfile based on the default profile with specified reward component overrides.
    
    Parameters:
        reward_updates (float): Keyword mapping of reward component names to numeric values; each provided value is coerced to float and merged into the profile's "reward" section.
    
    Returns:
        OperationalProfile: A profile derived from the default operational profile with the supplied reward updates applied. The returned profile is labeled "ablation_test_profile" and has version 21.
    """
    summary = DEFAULT_OPERATIONAL_PROFILE.to_summary()
    summary["name"] = "ablation_test_profile"
    summary["version"] = 21
    summary["reward"].update({name: float(value) for name, value in reward_updates.items()})
    return OperationalProfile.from_summary(summary)
