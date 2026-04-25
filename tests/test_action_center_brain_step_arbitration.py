"""Comprehensive tests for the action_center addition introduced in this PR.

Covers:
- architecture_signature() new structure (action_center, motor_cortex keys)
- SpiderBrain.ARCHITECTURE_VERSION = 13
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

from tests.fixtures.action_center import (
    _null_percept,
    _blank_obs,
    _module_interface,
    _vector_for,
    _valence_vector,
    _profile_with_updates,
)

class ValenceNormalizationEdgeCaseTest(unittest.TestCase):
    """Tests for learned valence probabilities when all evidence logits are equal."""

    def test_all_zero_evidence_logits_softmax_to_uniform_scores(self) -> None:
        import unittest.mock

        brain = SpiderBrain(seed=99, module_dropout=0.0)
        obs = _blank_obs()
        module_results = brain._proposal_results(obs, store_cache=False, training=False)

        with unittest.mock.patch("spider_cortex_sim.agent.clamp_unit", return_value=0.0):
            arb = brain._compute_arbitration(module_results, obs)

        for valence in SpiderBrain.VALENCE_ORDER:
            self.assertAlmostEqual(arb.valence_scores[valence], 0.25, places=5)
        self.assertEqual(arb.winning_valence, "threat")

        total = sum(arb.valence_scores.values())
        self.assertAlmostEqual(total, 1.0, places=5)

    def test_valence_scores_are_non_negative_for_blank_obs(self) -> None:
        brain = SpiderBrain(seed=99, module_dropout=0.0)
        step = brain.act(_blank_obs(), bus=None, sample=False)
        for valence, score in step.arbitration_decision.valence_scores.items():
            self.assertGreaterEqual(score, 0.0, f"Score for {valence} is negative")

class ReflexOnlyModeArbitrationTest(unittest.TestCase):
    """Tests that reflex_only mode does not expose arbitration from prior act calls."""

    def setUp(self) -> None:
        self.brain = SpiderBrain(seed=22, module_dropout=0.0)
        self.obs = _blank_obs()

    def test_reflex_only_mode_also_has_arbitration_decision(self) -> None:
        # reflex_only still runs arbitration since it's computed before the mode branch
        step = self.brain.act(self.obs, bus=None, sample=False, policy_mode="reflex_only")
        # Test that the act call succeeds and returns a BrainStep
        self.assertIsInstance(step, BrainStep)
        # Test that arbitration_decision is computed for reflex_only path
        self.assertIsNotNone(step.arbitration_decision)
        self.assertIsInstance(step.arbitration_decision, ArbitrationDecision)

    def test_invalid_policy_mode_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.brain.act(self.obs, bus=None, sample=False, policy_mode="invalid_mode")

class ArbitrationDecisionPredatorProximityCompatTest(unittest.TestCase):
    """Tests that the legacy predator_proximity key is injected and sorted correctly."""

    def _make_with_threat(self, extra_threat_keys: dict | None = None) -> "ArbitrationDecision":
        threat_evidence = {"predator_visible": 1.0, "predator_certainty": 0.9}
        if extra_threat_keys:
            threat_evidence.update(extra_threat_keys)
        return ArbitrationDecision(
            strategy="priority_gating",
            winning_valence="threat",
            valence_scores={"threat": 0.8, "hunger": 0.1, "sleep": 0.05, "exploration": 0.05},
            module_gates={"alert_center": 1.0},
            suppressed_modules=[],
            evidence={"threat": threat_evidence},
            intent_before_gating_idx=0,
            intent_after_gating_idx=0,
        )

    def test_predator_proximity_injected_as_zero(self) -> None:
        d = self._make_with_threat()
        payload = d.to_payload()
        self.assertIn("predator_proximity", payload["evidence"]["threat"])
        self.assertEqual(payload["evidence"]["threat"]["predator_proximity"], 0.0)

    def test_injected_predator_proximity_keys_are_sorted(self) -> None:
        d = self._make_with_threat()
        payload = d.to_payload()
        keys = list(payload["evidence"]["threat"].keys())
        self.assertEqual(keys, sorted(keys))

    def test_predator_proximity_not_injected_when_already_present(self) -> None:
        """If the evidence dict already has predator_proximity, it must not be overwritten."""
        d = ArbitrationDecision(
            strategy="priority_gating",
            winning_valence="threat",
            valence_scores={},
            module_gates={},
            suppressed_modules=[],
            evidence={"threat": {"predator_proximity": 0.77}},
            intent_before_gating_idx=0,
            intent_after_gating_idx=0,
        )
        payload = d.to_payload()
        # The already-present value should be preserved
        self.assertAlmostEqual(payload["evidence"]["threat"]["predator_proximity"], 0.77)

    def test_no_threat_evidence_no_injection(self) -> None:
        """If threat is absent from evidence, no predator_proximity is injected."""
        d = ArbitrationDecision(
            strategy="priority_gating",
            winning_valence="hunger",
            valence_scores={},
            module_gates={},
            suppressed_modules=[],
            evidence={"hunger": {"hunger": 0.9}},
            intent_before_gating_idx=0,
            intent_after_gating_idx=0,
        )
        payload = d.to_payload()
        self.assertNotIn("predator_proximity", payload["evidence"].get("hunger", {}))

class CorticalModuleBankResultFieldsTest(unittest.TestCase):
    """Tests that CorticalModuleBank.forward initializes new reflex fields on ModuleResult."""

    def _make_bank(self) -> CorticalModuleBank:
        """
        Create a CorticalModuleBank configured for deterministic test runs.

        Returns:
            CorticalModuleBank: instance with action_dim equal to len(LOCOMOTION_ACTIONS), an RNG seeded with 99, and module_dropout set to 0.0.
        """
        rng = np.random.default_rng(99)
        return CorticalModuleBank(
            action_dim=len(LOCOMOTION_ACTIONS),
            rng=rng,
            module_dropout=0.0,
        )

    def _blank_observation(self, bank: CorticalModuleBank) -> dict:
        """
        Build an observation dictionary containing zeroed input vectors for every spec in the cortical module bank.

        Parameters:
            bank (CorticalModuleBank): Module bank whose specs determine observation keys and input dimensions.

        Returns:
            dict: Mapping from each spec.observation_key to a NumPy float array of zeros with length equal to spec.input_dim.
        """
        obs = {}
        for spec in bank.specs:
            obs[spec.observation_key] = np.zeros(spec.input_dim, dtype=float)
        return obs

    def test_forward_sets_neural_logits_to_copy_of_logits(self) -> None:
        bank = self._make_bank()
        obs = self._blank_observation(bank)
        results = bank.forward(obs, store_cache=False, training=False)
        for result in results:
            if result.active:
                self.assertIsNotNone(result.neural_logits)
                np.testing.assert_allclose(result.neural_logits, result.logits)

    def test_forward_sets_reflex_delta_logits_to_zeros(self) -> None:
        bank = self._make_bank()
        obs = self._blank_observation(bank)
        results = bank.forward(obs, store_cache=False, training=False)
        for result in results:
            if result.active:
                self.assertIsNotNone(result.reflex_delta_logits)
                np.testing.assert_allclose(
                    result.reflex_delta_logits,
                    np.zeros_like(result.logits),
                )

    def test_forward_sets_post_reflex_logits_to_copy_of_logits(self) -> None:
        bank = self._make_bank()
        obs = self._blank_observation(bank)
        results = bank.forward(obs, store_cache=False, training=False)
        for result in results:
            if result.active:
                self.assertIsNotNone(result.post_reflex_logits)
                np.testing.assert_allclose(result.post_reflex_logits, result.logits)
