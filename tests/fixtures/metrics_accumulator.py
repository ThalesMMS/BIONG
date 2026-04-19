"""Focused metrics and behavior-evaluation tests."""

from __future__ import annotations

import unittest
from typing import Dict
from unittest.mock import MagicMock

from spider_cortex_sim.ablations import PROPOSAL_SOURCE_NAMES
from spider_cortex_sim.metrics import (
    ACTION_CENTER_REPRESENTATION_FIELDS,
    EpisodeMetricAccumulator,
    PRIMARY_REPRESENTATION_READOUT_MODULES,
    REFLEX_MODULE_NAMES,
    _contact_predator_types,
    _diagnostic_predator_distance,
    _dominant_predator_type,
    _first_active_predator_type,
    _normalize_distribution,
    _predator_type_threat,
    jensen_shannon_divergence,
)
from spider_cortex_sim.predator import PREDATOR_STATES
from spider_cortex_sim.world import REWARD_COMPONENT_NAMES

def _make_accumulator() -> EpisodeMetricAccumulator:
    """
    Create an EpisodeMetricAccumulator configured with the simulation's reward component names and predator states.
    
    Returns:
        EpisodeMetricAccumulator: An accumulator initialized using the environment's REWARD_COMPONENT_NAMES and PREDATOR_STATES.
    """
    return EpisodeMetricAccumulator(
        reward_component_names=REWARD_COMPONENT_NAMES,
        predator_states=PREDATOR_STATES,
    )

def _make_fake_meta(food_dist: int = 5, shelter_dist: int = 3,
                    night: bool = False, predator_visible: bool = False,
                    shelter_role: str = "outside", predator_dist: int = 10,
                    on_shelter: bool = False) -> Dict[str, object]:
    """Create a fake observation metadata mapping for tests."""
    return {
        "food_dist": food_dist,
        "shelter_dist": shelter_dist,
        "night": night,
        "predator_visible": predator_visible,
        "shelter_role": shelter_role,
        "diagnostic": {
            "diagnostic_predator_dist": predator_dist,
            "diagnostic_home_dx": 0.0,
            "diagnostic_home_dy": 0.0,
            "diagnostic_home_dist": float(shelter_dist),
        },
        "on_shelter": on_shelter,
    }

def _make_fake_info(reward_components: Dict[str, float] | None = None,
                    predator_contact: bool = False) -> Dict[str, object]:
    """Create a test-friendly info dict with reward components and predator contact flag."""
    if reward_components is None:
        reward_components = {name: 0.0 for name in REWARD_COMPONENT_NAMES}
    return {
        "reward_components": reward_components,
        "predator_contact": predator_contact,
    }

def _make_fake_state(sleep_debt: float = 0.2, last_move_dx: int = 0,
                     last_move_dy: int = 0) -> object:
    """Create a lightweight fake state object for testing."""
    state = MagicMock()
    state.sleep_debt = sleep_debt
    state.last_move_dx = last_move_dx
    state.last_move_dy = last_move_dy
    return state
