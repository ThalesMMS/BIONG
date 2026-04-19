from __future__ import annotations

import unittest
from typing import ClassVar
from unittest.mock import patch

import numpy as np

from spider_cortex_sim.ablations import BrainAblationConfig, canonical_ablation_configs, default_brain_config
from spider_cortex_sim.agent import SpiderBrain
from spider_cortex_sim.arbitration import (
    ARBITRATION_EVIDENCE_FIELDS,
    ARBITRATION_GATE_MODULE_ORDER,
    PRIORITY_GATING_WEIGHTS,
    VALENCE_EVIDENCE_WEIGHTS,
    VALENCE_ORDER,
    ArbitrationDecision,
    ValenceScore,
    apply_priority_gating,
    arbitration_evidence_input_dim,
    arbitration_evidence_signal_names,
    arbitration_evidence_vector,
    arbitration_gate_weight_for,
    clamp_unit,
    compute_arbitration,
    deterministic_valence_winner,
    fixed_formula_valence_scores_from_evidence,
    priority_gate_weight_for,
    proposal_contribution_share,
    warm_start_arbitration_network,
)
from spider_cortex_sim.modules import ModuleResult
from spider_cortex_sim.bus import MessageBus
from spider_cortex_sim.interfaces import (
    ACTION_CONTEXT_INTERFACE,
    LOCOMOTION_ACTIONS,
    MODULE_INTERFACES,
    MOTOR_CONTEXT_INTERFACE,
)
from spider_cortex_sim.nn import ArbitrationNetwork


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _blank_obs() -> dict[str, np.ndarray]:
    obs: dict[str, np.ndarray] = {}
    for iface in MODULE_INTERFACES:
        obs[iface.observation_key] = np.zeros(iface.input_dim, dtype=float)
    obs[ACTION_CONTEXT_INTERFACE.observation_key] = np.zeros(
        ACTION_CONTEXT_INTERFACE.input_dim,
        dtype=float,
    )
    obs[MOTOR_CONTEXT_INTERFACE.observation_key] = np.zeros(
        MOTOR_CONTEXT_INTERFACE.input_dim,
        dtype=float,
    )
    return obs

def _module_interface(name: str):
    return next(spec for spec in MODULE_INTERFACES if spec.name == name)

def _vector_for(interface, **updates) -> np.ndarray:
    mapping = {name: 0.0 for name in interface.signal_names}
    mapping.update({key: float(value) for key, value in updates.items()})
    return interface.vector_from_mapping(mapping)

def _valence_vector(**updates: float) -> np.ndarray:
    values = np.zeros(len(VALENCE_ORDER), dtype=float)
    index_by_name = {
        name: index
        for index, name in enumerate(VALENCE_ORDER)
    }
    for name, value in updates.items():
        values[index_by_name[name]] = float(value)
    return values

def _small_arbitration_evidence_vector() -> np.ndarray:
    return np.linspace(0.01, 0.24, ArbitrationNetwork.INPUT_DIM, dtype=float)

def _assert_relative_scaled_logits(
    test_case: unittest.TestCase,
    scaled_logits: np.ndarray,
    reference_logits: np.ndarray,
    scale: float,
    *,
    max_relative_error: float = 1e-4,
) -> None:
    expected = scale * reference_logits
    denominator = float(np.linalg.norm(reference_logits)) + 1e-12
    relative_error = float(
        np.linalg.norm(scaled_logits - expected) / denominator
    )
    test_case.assertLess(relative_error, max_relative_error)
