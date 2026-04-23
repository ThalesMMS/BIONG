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

from tests.fixtures.action_center import (
    _null_percept,
    _blank_obs,
    _module_interface,
    _vector_for,
    _valence_vector,
    _profile_with_updates,
)

class SpiderBrainSaveLoadArbitrationTest(unittest.TestCase):
    """Tests that save() persists arbitration_network.npz and load() reads it back."""

    def setUp(self) -> None:
        import tempfile
        self._tmpdir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _brain(self, seed: int = 7) -> SpiderBrain:
        return SpiderBrain(seed=seed, module_dropout=0.0)

    def test_save_creates_arbitration_npz(self) -> None:
        import os
        brain = self._brain()
        brain.save(self._tmpdir)
        arb_path = os.path.join(self._tmpdir, "arbitration_network.npz")
        self.assertTrue(os.path.exists(arb_path), "arbitration_network.npz should be created")

    def test_save_metadata_contains_arbitration_network_entry(self) -> None:
        import json
        import os
        brain = self._brain()
        brain.save(self._tmpdir)
        meta_path = os.path.join(self._tmpdir, "metadata.json")
        with open(meta_path, encoding="utf-8") as f:
            metadata = json.load(f)
        self.assertIn("arbitration_network", metadata["modules"])
        arb_meta = metadata["modules"]["arbitration_network"]
        self.assertEqual(arb_meta["type"], "arbitration")
        self.assertIn("input_dim", arb_meta)
        self.assertIn("hidden_dim", arb_meta)
        self.assertIn("valence_dim", arb_meta)
        self.assertIn("gate_dim", arb_meta)
        self.assertGreater(arb_meta["parameter_count"], 0)

    def test_save_metadata_contains_parameter_counts_summary(self) -> None:
        brain = self._brain()
        brain.save(self._tmpdir)
        meta_path = Path(self._tmpdir) / "metadata.json"
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        self.assertIn("parameter_counts", metadata)
        self.assertGreater(metadata["total_parameters"], 0)
        self.assertIn("action_center", metadata["parameter_counts"])
        self.assertIn("motor_cortex", metadata["parameter_counts"])

    def test_save_metadata_includes_arbitration_lr(self) -> None:
        """Verify that saving a Brain writes arbitration hyperparameters into metadata.json."""
        import json
        import os
        brain = self._brain()
        brain.save(self._tmpdir)
        meta_path = os.path.join(self._tmpdir, "metadata.json")
        with open(meta_path, encoding="utf-8") as f:
            metadata = json.load(f)
        self.assertIn("arbitration_lr", metadata)
        self.assertIn("arbitration_regularization_weight", metadata)
        self.assertIn("arbitration_valence_regularization_weight", metadata)

    def test_load_restores_arbitration_network_weights(self) -> None:
        brain1 = self._brain(seed=7)
        brain1.save(self._tmpdir)

        brain2 = self._brain(seed=99)
        brain2.load(self._tmpdir)

        np.testing.assert_array_almost_equal(
            brain1.arbitration_network.W1,
            brain2.arbitration_network.W1,
        )
        np.testing.assert_array_almost_equal(
            brain1.arbitration_network.W2_valence,
            brain2.arbitration_network.W2_valence,
        )

    def test_load_with_missing_arbitration_npz_raises(self) -> None:
        import os
        brain = self._brain()
        brain.save(self._tmpdir)
        # Remove the arbitration npz to simulate a missing file
        arb_path = os.path.join(self._tmpdir, "arbitration_network.npz")
        os.remove(arb_path)
        brain2 = self._brain(seed=99)
        with self.assertRaises(FileNotFoundError):
            brain2.load(self._tmpdir)

    def test_load_with_missing_arbitration_metadata_raises(self) -> None:
        import json
        import os
        brain = self._brain()
        brain.save(self._tmpdir)
        # Remove arbitration_network from metadata["modules"]
        meta_path = os.path.join(self._tmpdir, "metadata.json")
        with open(meta_path, encoding="utf-8") as f:
            metadata = json.load(f)
        del metadata["modules"]["arbitration_network"]
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f)
        brain2 = self._brain(seed=99)
        with self.assertRaises(FileNotFoundError):
            brain2.load(self._tmpdir)

    def test_roundtrip_produces_same_act_output(self) -> None:
        brain1 = self._brain(seed=7)
        obs = _blank_obs()
        step1 = brain1.act(obs, bus=None, sample=False)

        brain1.save(self._tmpdir)
        brain2 = self._brain(seed=99)
        brain2.load(self._tmpdir)
        step2 = brain2.act(obs, bus=None, sample=False)

        self.assertEqual(step1.action_intent_idx, step2.action_intent_idx)
        np.testing.assert_array_almost_equal(
            step1.action_center_logits, step2.action_center_logits, decimal=5
        )
