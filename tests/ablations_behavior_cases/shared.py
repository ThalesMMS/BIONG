from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from spider_cortex_sim.ablations import (
    BrainAblationConfig,
    COARSE_ROLLUP_MODULES,
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
from spider_cortex_sim.direct_policy_affordances import (
    AFFORDANCE_SHELTER_POSITION_NAMES,
    DIRECT_POLICY_LOCAL_AFFORDANCE_INPUT_DIM,
    DIRECT_POLICY_LOCAL_GEODESIC_INPUT_DIM,
    DIRECT_POLICY_TRANSITION_PREDICTION_FEATURE_DIM,
    DIRECT_POLICY_TRANSITION_ROLLOUT_PREDICTION_FEATURE_DIM,
    DIRECT_POLICY_LOCAL_TRANSITION_INPUT_DIM,
    DIRECT_POLICY_LOCAL_TRANSITION_ROLLOUT_INPUT_DIM,
    DIRECT_POLICY_LOCAL_SPATIAL_INPUT_DIM,
)
from spider_cortex_sim.direct_policy_options import OPTION_NAMES
from spider_cortex_sim.interfaces import (
    ACTION_TO_INDEX,
    ACTION_CONTEXT_INTERFACE,
    LOCOMOTION_ACTIONS,
    MODULE_INTERFACE_BY_NAME,
    MODULE_INTERFACES,
    MOTOR_CONTEXT_INTERFACE,
)
from spider_cortex_sim.modules import CorticalModuleBank, ModuleResult
from spider_cortex_sim.nn import (
    RecurrentEventAttentionTrueMonolithicNetwork,
    RecurrentOptionAffordanceFeedbackTrueMonolithicNetwork,
    RecurrentOptionAffordanceGeometryFeedbackTrueMonolithicNetwork,
    RecurrentOptionAffordancePositionFeedbackTrueMonolithicNetwork,
    RecurrentOptionAffordanceTopologyFeedbackTrueMonolithicNetwork,
    RecurrentOptionAffordanceTrueMonolithicNetwork,
    RecurrentOptionTrueMonolithicNetwork,
    RecurrentProposalNetwork,
    RecurrentTrueMonolithicNetwork,
)
from spider_cortex_sim.operational_profiles import DEFAULT_OPERATIONAL_PROFILE, OperationalProfile
from spider_cortex_sim.phase import PHASE_TO_INDEX
from spider_cortex_sim.simulation import SpiderSimulation

from tests.fixtures.ablations import _blank_mapping, _build_observation, _profile_with_updates

__all__ = [name for name in globals() if not name.startswith("__")]
