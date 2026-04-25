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
    TRUE_MONOLITHIC_POLICY_NAME,
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
from tests.fixtures.action_center_monolithic import MonolithicArchitectureFixtures


class SpiderBrainMonolithicBasicsTest(MonolithicArchitectureFixtures, unittest.TestCase):
    def test_monolithic_brain_has_no_module_bank(self) -> None:
        brain = SpiderBrain(seed=1, config=self._monolithic_config())
        self.assertIsNone(brain.module_bank)

    def test_monolithic_brain_has_monolithic_policy(self) -> None:
        brain = SpiderBrain(seed=1, config=self._monolithic_config())
        self.assertIsNotNone(brain.monolithic_policy)

    def test_modular_brain_has_module_bank(self) -> None:
        brain = SpiderBrain(seed=1)
        self.assertIsNotNone(brain.module_bank)

    def test_modular_brain_has_no_monolithic_policy(self) -> None:
        brain = SpiderBrain(seed=1)
        self.assertIsNone(brain.monolithic_policy)

    def test_monolithic_act_returns_single_module_result(self) -> None:
        brain = SpiderBrain(seed=5, config=self._monolithic_config())
        obs = self._build_observation()
        step = brain.act(obs, sample=False)
        self.assertEqual(len(step.module_results), 1)
        self.assertEqual(step.module_results[0].name, SpiderBrain.MONOLITHIC_POLICY_NAME)

    def test_true_monolithic_brain_initializes_direct_policy_only(self) -> None:
        brain = SpiderBrain(seed=5, config=self._true_monolithic_config())
        self.assertIsNone(brain.module_bank)
        self.assertIsNone(brain.monolithic_policy)
        self.assertIsNotNone(brain.true_monolithic_policy)

    def test_true_monolithic_act_returns_single_direct_module_result(self) -> None:
        brain = SpiderBrain(seed=5, config=self._true_monolithic_config())
        obs = self._build_observation()
        step = brain.act(obs, sample=False)
        self.assertEqual(len(step.module_results), 1)
        self.assertEqual(step.module_results[0].name, TRUE_MONOLITHIC_POLICY_NAME)
        self.assertGreaterEqual(step.action_idx, 0)
        self.assertLess(step.action_idx, len(LOCOMOTION_ACTIONS))
        self.assertTrue(np.isfinite(step.value))
        self.assertTrue(np.all(np.isfinite(step.policy)))
        self.assertEqual(step.motor_correction_logits.shape, (len(LOCOMOTION_ACTIONS),))
        self.assertTrue(np.allclose(step.motor_correction_logits, 0.0))
        self.assertIsNotNone(step.arbitration_decision)
        self.assertEqual(
            step.arbitration_decision.module_gates[TRUE_MONOLITHIC_POLICY_NAME],
            1.0,
        )

    def test_true_monolithic_estimate_value_returns_finite_float(self) -> None:
        brain = SpiderBrain(seed=7, config=self._true_monolithic_config())
        obs = self._build_observation()
        value = brain.estimate_value(obs)
        self.assertIsInstance(value, float)
        self.assertTrue(np.isfinite(value))

    def test_true_monolithic_learn_updates_weights_and_returns_valid_stats(self) -> None:
        brain = SpiderBrain(seed=17, config=self._true_monolithic_config())
        obs = self._build_observation()
        step = brain.act(obs, sample=True)
        next_obs = self._build_observation()
        before = {
            key: value.copy()
            for key, value in brain.true_monolithic_policy.state_dict().items()
            if isinstance(value, np.ndarray)
        }
        stats = brain.learn(step, reward=1.5, next_observation=next_obs, done=True)
        self.assertIn("reward", stats)
        self.assertEqual(stats["credit_strategy"], "broadcast")
        self.assertEqual(
            stats["module_credit_weights"],
            {TRUE_MONOLITHIC_POLICY_NAME: 1.0},
        )
        self.assertIn(TRUE_MONOLITHIC_POLICY_NAME, stats["module_gradient_norms"])
        self.assertTrue(np.isfinite(stats["td_error"]))
        after = {
            key: value.copy()
            for key, value in brain.true_monolithic_policy.state_dict().items()
            if isinstance(value, np.ndarray)
        }
        self.assertTrue(
            any(not np.allclose(before[key], after[key]) for key in before)
        )

    def test_true_monolithic_frozen_proposer_skips_weight_update_and_reports_zero_norm(self) -> None:
        brain = SpiderBrain(seed=17, config=self._true_monolithic_config())
        obs = self._build_observation()
        step = brain.act(obs, sample=True)
        next_obs = self._build_observation()
        before = self._numeric_state(brain.true_monolithic_policy)

        brain.freeze_proposers([TRUE_MONOLITHIC_POLICY_NAME])
        stats = brain.learn(step, reward=1.5, next_observation=next_obs, done=True)

        self.assertEqual(brain.frozen_module_names(), [TRUE_MONOLITHIC_POLICY_NAME])
        self.assertFalse(
            self._state_changed(before, self._numeric_state(brain.true_monolithic_policy))
        )
        self.assertEqual(
            stats["module_gradient_norms"][TRUE_MONOLITHIC_POLICY_NAME],
            0.0,
        )

    def test_true_monolithic_counterfactual_credit_is_coerced_before_credit_logic(self) -> None:
        brain = SpiderBrain(
            seed=17,
            config=self._true_monolithic_config(credit_strategy="counterfactual"),
        )
        obs = self._build_observation()
        step = brain.act(obs, sample=True)

        def fail_counterfactual(*args, **kwargs):
            raise AssertionError(
                "_compute_counterfactual_credit should not run for true_monolithic"
            )

        brain._compute_counterfactual_credit = fail_counterfactual
        stats = brain.learn(step, reward=1.0, next_observation=obs, done=False)

        self.assertEqual(stats["credit_strategy"], "broadcast")
        self.assertEqual(
            stats["module_credit_weights"],
            {TRUE_MONOLITHIC_POLICY_NAME: 1.0},
        )
        self.assertEqual(stats["counterfactual_credit_weights"], {})

    def test_true_monolithic_save_and_load_round_trip(self) -> None:
        config = self._true_monolithic_config()
        brain = SpiderBrain(seed=11, config=config)
        obs = self._build_observation()
        step_before = brain.act(obs, sample=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "true_mono_brain"
            brain.save(save_path)
            self.assertTrue((save_path / "true_monolithic_policy.npz").exists())

            brain2 = SpiderBrain(seed=999, config=config)
            brain2.load(save_path)
            step_after = brain2.act(obs, sample=False)

        np.testing.assert_allclose(step_before.policy, step_after.policy, atol=1e-5)
        np.testing.assert_allclose(step_before.value, step_after.value, atol=1e-5)

    def test_monolithic_act_result_has_none_interface(self) -> None:
        brain = SpiderBrain(seed=5, config=self._monolithic_config())
        obs = self._build_observation()
        step = brain.act(obs, sample=False)
        self.assertIsNone(step.module_results[0].interface)

    def test_monolithic_act_result_is_active(self) -> None:
        brain = SpiderBrain(seed=5, config=self._monolithic_config())
        obs = self._build_observation()
        step = brain.act(obs, sample=False)
        self.assertTrue(step.module_results[0].active)

    def test_monolithic_module_names_includes_monolithic_policy(self) -> None:
        brain = SpiderBrain(seed=5, config=self._monolithic_config())
        names = brain._module_names()
        self.assertIn(SpiderBrain.MONOLITHIC_POLICY_NAME, names)
        self.assertIn("arbitration_network", names)
        self.assertIn("action_center", names)
        self.assertIn("motor_cortex", names)

    def test_modular_module_names_includes_each_module(self) -> None:
        brain = SpiderBrain(seed=5)
        names = brain._module_names()
        self.assertIn("visual_cortex", names)
        self.assertIn("alert_center", names)
        self.assertIn("action_center", names)
        self.assertIn("motor_cortex", names)
        self.assertNotIn(SpiderBrain.MONOLITHIC_POLICY_NAME, names)

    def test_monolithic_parameter_norms_has_monolithic_policy(self) -> None:
        brain = SpiderBrain(seed=3, config=self._monolithic_config())
        norms = brain.parameter_norms()
        self.assertIn(SpiderBrain.MONOLITHIC_POLICY_NAME, norms)
        self.assertIn("arbitration_network", norms)
        self.assertIn("action_center", norms)
        self.assertIn("motor_cortex", norms)

    def test_monolithic_parameter_counts_has_monolithic_policy(self) -> None:
        brain = SpiderBrain(seed=3, config=self._monolithic_config())
        counts = brain.count_parameters()
        self.assertIn(SpiderBrain.MONOLITHIC_POLICY_NAME, counts)
        self.assertIn("arbitration_network", counts)
        self.assertIn("action_center", counts)
        self.assertIn("motor_cortex", counts)
        self.assertGreater(counts[SpiderBrain.MONOLITHIC_POLICY_NAME], 0)

    def test_modular_parameter_norms_has_module_names(self) -> None:
        brain = SpiderBrain(seed=3)
        norms = brain.parameter_norms()
        self.assertIn("visual_cortex", norms)
        self.assertIn("alert_center", norms)
        self.assertIn("action_center", norms)
        self.assertIn("motor_cortex", norms)
        self.assertNotIn(SpiderBrain.MONOLITHIC_POLICY_NAME, norms)

    def test_modular_parameter_counts_has_module_names(self) -> None:
        brain = SpiderBrain(seed=3)
        counts = brain.count_parameters()
        self.assertIn("visual_cortex", counts)
        self.assertIn("alert_center", counts)
        self.assertIn("action_center", counts)
        self.assertIn("motor_cortex", counts)
        self.assertNotIn(SpiderBrain.MONOLITHIC_POLICY_NAME, counts)

    def test_modular_and_monolithic_parameter_counts_distinguish_architectures(self) -> None:
        modular_brain = SpiderBrain(seed=3)
        monolithic_brain = SpiderBrain(seed=3, config=self._monolithic_config())

        modular_counts = modular_brain.count_parameters()
        monolithic_counts = monolithic_brain.count_parameters()

        self.assertIn("visual_cortex", modular_counts)
        self.assertIn(SpiderBrain.MONOLITHIC_POLICY_NAME, monolithic_counts)
        self.assertGreater(sum(modular_counts.values()), 0)
        self.assertGreater(sum(monolithic_counts.values()), 0)
        shared_keys = set(modular_counts) & set(monolithic_counts)
        self.assertTrue(
            set(modular_counts) != set(monolithic_counts)
            or any(modular_counts[key] != monolithic_counts[key] for key in shared_keys)
        )

    def test_monolithic_estimate_value_returns_finite_float(self) -> None:
        brain = SpiderBrain(seed=7, config=self._monolithic_config())
        obs = self._build_observation()
        value = brain.estimate_value(obs)
        self.assertIsInstance(value, float)
        self.assertTrue(np.isfinite(value))

    def test_monolithic_action_idx_is_valid(self) -> None:
        brain = SpiderBrain(seed=7, config=self._monolithic_config())
        obs = self._build_observation()
        step = brain.act(obs, sample=False)
        self.assertGreaterEqual(step.action_idx, 0)
        self.assertLess(step.action_idx, len(LOCOMOTION_ACTIONS))

    def test_monolithic_step_exposes_arbitration_metadata(self) -> None:
        brain = SpiderBrain(seed=7, config=self._monolithic_config())
        obs = self._build_observation()
        step = brain.act(obs, sample=False)
        self.assertIsNotNone(step.arbitration_decision)
        self.assertEqual(step.arbitration_decision.module_gates["monolithic_policy"], 1.0)
        self.assertEqual(
            step.arbitration_decision.module_contribution_share["monolithic_policy"],
            1.0,
        )

    def test_monolithic_save_and_load_round_trip(self) -> None:
        """
        Verifies that saving then loading a monolithic SpiderBrain reproduces its policy and value outputs.

        Uses a fixed observation and monolithic configuration: saves the brain to disk, loads it into a new SpiderBrain instance with the same configuration, and asserts that the resulting policy and value vectors match within a small numerical tolerance.
        """
        config = self._monolithic_config()
        brain = SpiderBrain(seed=11, config=config)
        obs = self._build_observation()
        step_before = brain.act(obs, sample=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "mono_brain"
            brain.save(save_path)

            brain2 = SpiderBrain(seed=999, config=config)
            brain2.load(save_path)
            step_after = brain2.act(obs, sample=False)

        np.testing.assert_allclose(step_before.policy, step_after.policy, atol=1e-5)
        np.testing.assert_allclose(step_before.value, step_after.value, atol=1e-5)

    def test_loading_mismatched_architecture_version_raises_value_error(self) -> None:
        config = self._monolithic_config()
        brain = SpiderBrain(seed=11, config=config)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "mono_brain"
            brain.save(save_path)

            metadata_path = save_path / SpiderBrain._METADATA_FILE
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["architecture_version"] = SpiderBrain.ARCHITECTURE_VERSION - 1
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

            brain2 = SpiderBrain(seed=999, config=config)
            with self.assertRaises(ValueError):
                brain2.load(save_path)

    def test_save_metadata_contains_registry_and_fingerprint(self) -> None:
        config = self._monolithic_config()
        brain = SpiderBrain(seed=11, config=config)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "mono_brain"
            brain.save(save_path)

            metadata_path = save_path / SpiderBrain._METADATA_FILE
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertIn("interface_registry", metadata)
            self.assertIn("architecture_fingerprint", metadata)
            self.assertEqual(
                metadata["architecture_fingerprint"],
                metadata["architecture"]["fingerprint"],
            )

    def test_save_writes_arbitration_network_weights(self) -> None:
        config = self._monolithic_config()
        brain = SpiderBrain(seed=11, config=config)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "mono_brain"
            brain.save(save_path)

            self.assertTrue((save_path / "arbitration_network.npz").exists())
            with np.load(save_path / "arbitration_network.npz") as arbitration_npz:
                self.assertIn("input_dim", arbitration_npz.files)
                self.assertIn("W2_gate", arbitration_npz.files)
            metadata = json.loads(
                (save_path / SpiderBrain._METADATA_FILE).read_text(encoding="utf-8")
            )
            self.assertIn("arbitration_network", metadata["modules"])
            self.assertEqual(
                metadata["modules"]["arbitration_network"]["type"],
                "arbitration",
            )

    def test_loading_missing_arbitration_network_weights_raises_file_not_found(self) -> None:
        config = self._monolithic_config()
        brain = SpiderBrain(seed=11, config=config)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "mono_brain"
            brain.save(save_path)
            (save_path / "arbitration_network.npz").unlink()

            brain2 = SpiderBrain(seed=999, config=config)
            with self.assertRaisesRegex(FileNotFoundError, "arbitration weights"):
                brain2.load(save_path)

    def test_loading_mismatched_interface_registry_raises_value_error(self) -> None:
        config = self._monolithic_config()
        brain = SpiderBrain(seed=11, config=config)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "mono_brain"
            brain.save(save_path)

            metadata_path = save_path / SpiderBrain._METADATA_FILE
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["interface_registry"]["interfaces"]["motor_cortex_context"]["version"] = 999
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

            brain2 = SpiderBrain(seed=999, config=config)
            with self.assertRaisesRegex(ValueError, "interface registry incompatible"):
                brain2.load(save_path)

    def test_loading_monolithic_into_modular_raises_value_error(self) -> None:
        mono_config = self._monolithic_config()
        brain_mono = SpiderBrain(seed=13, config=mono_config)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "mono_brain"
            brain_mono.save(save_path)

            brain_modular = SpiderBrain(seed=13)
            with self.assertRaises(ValueError):
                brain_modular.load(save_path)

    def test_loading_different_modular_ablation_config_raises_value_error(self) -> None:
        saved_brain = SpiderBrain(seed=19, config=default_brain_config(module_dropout=0.0))
        different_config = BrainAblationConfig(
            name="no_module_reflexes",
            architecture="modular",
            module_dropout=0.0,
            enable_reflexes=False,
            enable_auxiliary_targets=False,
            disabled_modules=(),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "modular_brain"
            saved_brain.save(save_path)

            mismatched_brain = SpiderBrain(seed=19, config=different_config)
            with self.assertRaises(ValueError):
                mismatched_brain.load(save_path)

    def test_loading_different_operational_profile_raises_value_error(self) -> None:
        saved_brain = SpiderBrain(seed=23)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "profiled_brain"
            saved_brain.save(save_path)

            mismatched_brain = SpiderBrain(
                seed=23,
                operational_profile=_profile_with_updates(predator_threat_smell_threshold=0.01),
            )
            with self.assertRaises(ValueError):
                mismatched_brain.load(save_path)

    def test_monolithic_config_stored_on_brain(self) -> None:
        config = self._monolithic_config()
        brain = SpiderBrain(seed=7, config=config)
        self.assertEqual(brain.config.architecture, "monolithic")
        self.assertEqual(brain.config.name, "monolithic_policy")

    def test_monolithic_learn_does_not_raise(self) -> None:
        brain = SpiderBrain(seed=17, config=self._monolithic_config())
        obs = self._build_observation()
        step = brain.act(obs, sample=True)
        next_obs = self._build_observation()
        stats = brain.learn(step, reward=0.5, next_observation=next_obs, done=False)
        self.assertIn("reward", stats)
        self.assertTrue(np.isfinite(stats["td_error"]))
