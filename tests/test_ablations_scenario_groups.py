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

from tests.fixtures.ablations import _blank_mapping, _build_observation, _profile_with_updates

class CanonicalAblationScenarioGroupsTest(unittest.TestCase):
    """Tests for canonical_ablation_scenario_groups() - new in this PR."""

    def test_returns_all_three_groups(self) -> None:
        groups = canonical_ablation_scenario_groups()
        self.assertIn("multi_predator_ecology", groups)
        self.assertIn("visual_predator_scenarios", groups)
        self.assertIn("olfactory_predator_scenarios", groups)

    def test_multi_predator_ecology_matches_constant(self) -> None:
        groups = canonical_ablation_scenario_groups()
        self.assertEqual(groups["multi_predator_ecology"], MULTI_PREDATOR_SCENARIOS)

    def test_visual_predator_scenarios_matches_constant(self) -> None:
        groups = canonical_ablation_scenario_groups()
        self.assertEqual(groups["visual_predator_scenarios"], VISUAL_PREDATOR_SCENARIOS)

    def test_olfactory_predator_scenarios_matches_constant(self) -> None:
        groups = canonical_ablation_scenario_groups()
        self.assertEqual(groups["olfactory_predator_scenarios"], OLFACTORY_PREDATOR_SCENARIOS)

    def test_values_are_tuples(self) -> None:
        groups = canonical_ablation_scenario_groups()
        for name, scenarios in groups.items():
            with self.subTest(group=name):
                self.assertIsInstance(scenarios, tuple)

    def test_visual_pincer_in_all_groups(self) -> None:
        groups = canonical_ablation_scenario_groups()
        for name, scenarios in groups.items():
            with self.subTest(group=name):
                self.assertIn("visual_olfactory_pincer", scenarios)

class ResolveAblationScenarioGroupEdgeCasesTest(unittest.TestCase):
    """Additional edge-case tests for resolve_ablation_scenario_group() - new in this PR."""

    def test_unknown_group_raises_key_error(self) -> None:
        with self.assertRaises(KeyError):
            resolve_ablation_scenario_group("nonexistent_group")

    def test_key_error_message_lists_available_groups(self) -> None:
        try:
            resolve_ablation_scenario_group("bad_group")
        except KeyError as e:
            msg = str(e)
            self.assertIn("multi_predator_ecology", msg)
            self.assertIn("visual_predator_scenarios", msg)

    def test_returns_tuple_type(self) -> None:
        result = resolve_ablation_scenario_group("multi_predator_ecology")
        self.assertIsInstance(result, tuple)

    def test_visual_group_contains_correct_scenarios(self) -> None:
        result = resolve_ablation_scenario_group("visual_predator_scenarios")
        self.assertIn("visual_olfactory_pincer", result)
        self.assertIn("visual_hunter_open_field", result)
        self.assertNotIn("olfactory_ambush", result)

    def test_olfactory_group_contains_correct_scenarios(self) -> None:
        result = resolve_ablation_scenario_group("olfactory_predator_scenarios")
        self.assertIn("visual_olfactory_pincer", result)
        self.assertIn("olfactory_ambush", result)
        self.assertNotIn("visual_hunter_open_field", result)

class MultiPredatorScenarioConstantsTest(unittest.TestCase):
    """Tests for new constants in ablations.py - MULTI_PREDATOR_SCENARIOS etc."""

    def test_multi_predator_scenarios_contains_three_entries(self) -> None:
        self.assertEqual(len(MULTI_PREDATOR_SCENARIOS), 3)

    def test_visual_predator_scenarios_contains_two_entries(self) -> None:
        self.assertEqual(len(VISUAL_PREDATOR_SCENARIOS), 2)

    def test_olfactory_predator_scenarios_contains_two_entries(self) -> None:
        self.assertEqual(len(OLFACTORY_PREDATOR_SCENARIOS), 2)

    def test_visual_olfactory_pincer_in_all_groups(self) -> None:
        self.assertIn("visual_olfactory_pincer", MULTI_PREDATOR_SCENARIOS)
        self.assertIn("visual_olfactory_pincer", VISUAL_PREDATOR_SCENARIOS)
        self.assertIn("visual_olfactory_pincer", OLFACTORY_PREDATOR_SCENARIOS)

    def test_visual_hunter_open_field_in_visual_not_olfactory(self) -> None:
        self.assertIn("visual_hunter_open_field", VISUAL_PREDATOR_SCENARIOS)
        self.assertNotIn("visual_hunter_open_field", OLFACTORY_PREDATOR_SCENARIOS)

    def test_olfactory_ambush_in_olfactory_not_visual(self) -> None:
        self.assertIn("olfactory_ambush", OLFACTORY_PREDATOR_SCENARIOS)
        self.assertNotIn("olfactory_ambush", VISUAL_PREDATOR_SCENARIOS)

    def test_multi_predator_scenario_groups_has_three_keys(self) -> None:
        self.assertEqual(len(MULTI_PREDATOR_SCENARIO_GROUPS), 3)

    def test_multi_predator_scenario_groups_keys(self) -> None:
        keys = set(MULTI_PREDATOR_SCENARIO_GROUPS.keys())
        self.assertIn("multi_predator_ecology", keys)
        self.assertIn("visual_predator_scenarios", keys)
        self.assertIn("olfactory_predator_scenarios", keys)
