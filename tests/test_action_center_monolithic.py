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

class SpiderBrainMonolithicArchitectureTest(unittest.TestCase):
    """Tests for the new monolithic architecture branch in SpiderBrain."""

    def _monolithic_config(
        self,
        *,
        credit_strategy: str = "broadcast",
    ) -> BrainAblationConfig:
        return BrainAblationConfig(
            name=MONOLITHIC_POLICY_NAME,
            architecture="monolithic",
            module_dropout=0.0,
            enable_reflexes=False,
            enable_auxiliary_targets=False,
            credit_strategy=credit_strategy,
            disabled_modules=(),
        )

    def _build_observation(self) -> dict:
        """
        Builds a baseline observation dictionary containing zeroed input vectors for every module and context.

        The returned dictionary contains entries for each interface in MODULE_INTERFACES plus ACTION_CONTEXT_INTERFACE and MOTOR_CONTEXT_INTERFACE. Each key is the interface's `observation_key` and each value is a NumPy array of zeros with length equal to that interface's `input_dim`.

        Returns:
            dict: Mapping from observation_key to a zeroed NumPy array sized to the corresponding interface's input_dim.
        """
        obs = {}
        for interface in MODULE_INTERFACES:
            obs[interface.observation_key] = np.zeros(interface.input_dim, dtype=float)
        obs[ACTION_CONTEXT_INTERFACE.observation_key] = np.zeros(
            ACTION_CONTEXT_INTERFACE.input_dim, dtype=float
        )
        obs[MOTOR_CONTEXT_INTERFACE.observation_key] = np.zeros(
            MOTOR_CONTEXT_INTERFACE.input_dim, dtype=float
        )
        return obs

    def _numeric_state(self, network) -> dict[str, np.ndarray]:
        return {
            key: value.copy()
            for key, value in network.state_dict().items()
            if isinstance(value, np.ndarray)
        }

    def _state_changed(
        self,
        before: dict[str, np.ndarray],
        after: dict[str, np.ndarray],
    ) -> bool:
        return any(not np.allclose(before[key], after[key]) for key in before)

    def _true_monolithic_config(
        self,
        *,
        credit_strategy: str = "broadcast",
    ) -> BrainAblationConfig:
        return BrainAblationConfig(
            name=TRUE_MONOLITHIC_POLICY_NAME,
            architecture="true_monolithic",
            module_dropout=0.0,
            enable_reflexes=False,
            enable_auxiliary_targets=False,
            use_learned_arbitration=False,
            credit_strategy=credit_strategy,
            reflex_scale=0.0,
            disabled_modules=(),
        )

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

    def test_modular_learn_routes_action_center_input_grads_to_module_aux_paths(self) -> None:
        config = BrainAblationConfig(
            name="modular_no_reflex_aux",
            architecture="modular",
            module_dropout=0.0,
            enable_reflexes=False,
            enable_auxiliary_targets=False,
        )
        brain = SpiderBrain(seed=7, config=config)
        obs = self._build_observation()
        step = brain.act(obs, sample=False)

        captured: dict[str, object] = {}
        proposal_grads = np.arange(
            len(step.module_results) * len(LOCOMOTION_ACTIONS),
            dtype=float,
        ).reshape(len(step.module_results), len(LOCOMOTION_ACTIONS))
        upstream = np.concatenate(
            [
                proposal_grads.reshape(-1),
                np.zeros(ACTION_CONTEXT_INTERFACE.input_dim, dtype=float),
            ]
        )

        def fake_action_center_backward(*, grad_policy_logits, grad_value, lr):
            captured["shared_grad"] = np.asarray(grad_policy_logits, dtype=float).copy()
            captured["grad_value"] = float(grad_value)
            captured["lr"] = float(lr)
            return upstream

        def fake_module_bank_backward(grad_logits, lr, aux_grads=None):
            captured["bank_grad"] = np.asarray(grad_logits, dtype=float).copy()
            captured["bank_lr"] = float(lr)
            captured["aux_grads"] = {
                name: np.asarray(value, dtype=float).copy()
                for name, value in (aux_grads or {}).items()
            }

        brain.action_center.backward = fake_action_center_backward
        brain.module_bank.backward = fake_module_bank_backward
        brain.motor_cortex.backward = lambda grad_logits, lr: None
        brain.estimate_value = lambda _: 0.0

        expected_reflex_aux = brain._auxiliary_module_gradients(step.module_results)
        stats = brain.learn(step, reward=0.5, next_observation=obs, done=False)

        np.testing.assert_allclose(
            captured["bank_grad"],
            np.zeros_like(captured["shared_grad"]),
        )
        for idx, result in enumerate(step.module_results):
            expected_total_grad = result.gate_weight * (
                captured["shared_grad"] + proposal_grads[idx]
            )
            expected_total_grad += result.gate_weight * expected_reflex_aux.get(
                result.name,
                np.zeros(len(LOCOMOTION_ACTIONS), dtype=float),
            )
            np.testing.assert_allclose(
                captured["aux_grads"][result.name],
                expected_total_grad,
            )
            self.assertAlmostEqual(
                stats["module_credit_weights"][result.name],
                1.0,
            )
            self.assertAlmostEqual(
                stats["module_gradient_norms"][result.name],
                float(np.linalg.norm(expected_total_grad)),
            )
        self.assertEqual(stats["aux_modules"], float(len(expected_reflex_aux)))
        self.assertEqual(stats["credit_strategy"], "broadcast")

    def test_learn_projects_action_center_input_grads_to_arbitration_gates(self) -> None:
        config = BrainAblationConfig(
            name="modular_no_reflex_aux",
            architecture="modular",
            module_dropout=0.0,
            enable_reflexes=False,
            enable_auxiliary_targets=False,
        )
        brain = SpiderBrain(
            seed=7,
            config=config,
            arbitration_regularization_weight=0.25,
            arbitration_valence_regularization_weight=0.4,
        )
        brain.arbitration_network.b2_gate[:] = np.log(0.7 / 0.3)
        obs = self._build_observation()
        step = brain.act(obs, sample=False, training=True)
        arbitration = step.arbitration_decision
        self.assertIsNotNone(arbitration)

        captured: dict[str, object] = {}
        proposal_grads = np.linspace(
            -0.2,
            0.3,
            len(step.module_results) * len(LOCOMOTION_ACTIONS),
            dtype=float,
        ).reshape(len(step.module_results), len(LOCOMOTION_ACTIONS))
        upstream = np.concatenate(
            [
                proposal_grads.reshape(-1),
                np.zeros(ACTION_CONTEXT_INTERFACE.input_dim, dtype=float),
            ]
        )

        def fake_action_center_backward(*, grad_policy_logits, grad_value, lr):
            """
            Capture a copy of the policy logits gradient into the outer `captured` mapping and return the upstream gradient unchanged.

            Parameters:
                grad_policy_logits (array-like): Gradient of the policy logits to capture; converted to a float NumPy array and stored as `captured["policy_grad"]`.
                grad_value: Ignored.
                lr: Ignored.

            Returns:
                The upstream gradient value unchanged.
            """
            captured["policy_grad"] = np.asarray(grad_policy_logits, dtype=float).copy()
            return upstream

        def fake_arbitration_backward(*, grad_valence_logits, grad_gate_adjustments, grad_value, lr):
            """
            Record provided arbitration gradients and learning rate into the `captured` mapping and return a zeroed input-gradient vector matching the arbitration network input dimension.

            Parameters:
                grad_valence_logits (array-like): Gradients for arbitration valence logits; stored as a float NumPy array under key `"arb_valence_grad"`.
                grad_gate_adjustments (array-like): Gradients for arbitration gate adjustments; stored as a float NumPy array under key `"arb_gate_grad"`.
                grad_value (float): Gradient for the arbitration value output; stored as a float under key `"arb_value_grad"`.
                lr (float): Learning rate used for this backward pass; stored as a float under key `"arb_lr"`.

            Returns:
                numpy.ndarray: A zero-filled float array whose length equals the arbitration network input dimension.
            """
            captured["arb_valence_grad"] = np.asarray(grad_valence_logits, dtype=float).copy()
            captured["arb_gate_grad"] = np.asarray(grad_gate_adjustments, dtype=float).copy()
            captured["arb_value_grad"] = float(grad_value)
            captured["arb_lr"] = float(lr)
            return np.zeros(brain.arbitration_network.input_dim, dtype=float)

        brain.action_center.backward = fake_action_center_backward
        brain.arbitration_network.backward = fake_arbitration_backward
        brain.module_bank.backward = lambda grad_logits, lr, aux_grads=None: None
        brain.motor_cortex.backward = lambda grad_logits, lr: None
        brain.estimate_value = lambda _: 0.0

        stats = brain.learn(step, reward=0.5, next_observation=obs, done=False)

        td_target = 0.5
        advantage = float(np.clip(td_target - step.value, -4.0, 4.0))
        valence_probs = np.array(
            [
                float(arbitration.valence_scores[name])
                for name in SpiderBrain.VALENCE_ORDER
            ],
            dtype=float,
        )
        fixed_targets = brain._fixed_formula_valence_scores_from_evidence(arbitration.evidence)
        expected_valence_grad = advantage * (
            valence_probs
            - np.eye(len(SpiderBrain.VALENCE_ORDER))[SpiderBrain.VALENCE_ORDER.index(arbitration.winning_valence)]
        )
        expected_valence_grad += brain.arbitration_valence_regularization_weight * (
            valence_probs - fixed_targets
        )

        gate_indices = {
            name: index
            for index, name in enumerate(SpiderBrain.ARBITRATION_GATE_MODULE_ORDER)
        }
        expected_final_gate_grad = np.zeros(
            len(SpiderBrain.ARBITRATION_GATE_MODULE_ORDER),
            dtype=float,
        )
        expected_gate_reg = np.zeros_like(expected_final_gate_grad)
        expected_base = np.zeros_like(expected_final_gate_grad)
        for idx, result in enumerate(step.module_results):
            gate_index = gate_indices[result.name]
            raw_logits = result.post_reflex_logits
            self.assertIsNotNone(raw_logits)
            expected_base[gate_index] = arbitration.base_gates[result.name]
            expected_final_gate_grad[gate_index] += float(
                np.dot(proposal_grads[idx], raw_logits)
            )
            expected_gate_reg[gate_index] += (
                brain.arbitration_regularization_weight
                * (arbitration.module_gates[result.name] - arbitration.base_gates[result.name])
            )
        expected_gate_grad = expected_base * (
            expected_final_gate_grad + expected_gate_reg
        )

        np.testing.assert_allclose(captured["arb_valence_grad"], expected_valence_grad)
        np.testing.assert_allclose(captured["arb_gate_grad"], expected_gate_grad)
        self.assertAlmostEqual(captured["arb_value_grad"], arbitration.arbitration_value - td_target)
        self.assertEqual(captured["arb_lr"], brain.arbitration_lr)
        self.assertGreater(stats["arbitration_gate_regularization_norm"], 0.0)
        self.assertGreater(stats["arbitration_valence_regularization_norm"], 0.0)
        self.assertIn("arbitration_loss", stats)
        self.assertIn("gate_adjustment_magnitude", stats)
        self.assertIn("regularization_loss", stats)
        self.assertGreater(stats["gate_adjustment_magnitude"], 0.0)
        self.assertGreater(stats["regularization_loss"], 0.0)

    def test_local_credit_only_learn_omits_shared_policy_broadcast(self) -> None:
        """
        Verifies that the "local_credit_only" ablation prevents shared policy gradients from being broadcast to module auxiliary gradient paths.

        Sets up a modular brain configured for the local-credit-only variant, stubs the action-center and module-bank backward methods to capture gradients, invokes learning on a single step, and asserts each module's auxiliary gradient equals the module's gate weight multiplied by its proposal gradients (i.e., only local gate-weighted proposal gradients are applied to module auxiliary paths).
        """
        config = BrainAblationConfig(
            name="local_credit_only",
            architecture="modular",
            module_dropout=0.0,
            enable_reflexes=False,
            enable_auxiliary_targets=False,
            credit_strategy="local_only",
        )
        brain = SpiderBrain(seed=7, config=config)
        obs = self._build_observation()
        step = brain.act(obs, sample=False)

        captured: dict[str, object] = {}
        proposal_grads = np.arange(
            len(step.module_results) * len(LOCOMOTION_ACTIONS),
            dtype=float,
        ).reshape(len(step.module_results), len(LOCOMOTION_ACTIONS))
        upstream = np.concatenate(
            [
                proposal_grads.reshape(-1),
                np.zeros(ACTION_CONTEXT_INTERFACE.input_dim, dtype=float),
            ]
        )

        def fake_action_center_backward(*, grad_policy_logits, grad_value, lr):
            captured["shared_grad"] = np.asarray(grad_policy_logits, dtype=float).copy()
            return upstream

        def fake_module_bank_backward(grad_logits, lr, aux_grads=None):
            captured["aux_grads"] = {
                name: np.asarray(value, dtype=float).copy()
                for name, value in (aux_grads or {}).items()
            }

        brain.action_center.backward = fake_action_center_backward
        brain.module_bank.backward = fake_module_bank_backward
        brain.motor_cortex.backward = lambda grad_logits, lr: None
        brain.estimate_value = lambda _: 0.0

        stats = brain.learn(step, reward=0.5, next_observation=obs, done=False)

        for idx, result in enumerate(step.module_results):
            expected_total_grad = result.gate_weight * proposal_grads[idx]
            np.testing.assert_allclose(
                captured["aux_grads"][result.name],
                expected_total_grad,
            )
            self.assertAlmostEqual(
                stats["module_credit_weights"][result.name],
                0.0,
            )
            self.assertAlmostEqual(
                stats["module_gradient_norms"][result.name],
                float(np.linalg.norm(expected_total_grad)),
            )
        self.assertEqual(stats["credit_strategy"], "local_only")

    def test_counterfactual_credit_normalizes_absolute_interference(self) -> None:
        """
        Verifies that counterfactual credit weights normalize absolute interference between modules.

        Constructs two module results with opposing effects and asserts the counterfactual credit
        computation:
        - returns positive weights for each contributing module,
        - normalizes weights to sum to 1.0,
        - assigns different weights when modules have different absolute contributions.
        """
        brain = SpiderBrain(
            seed=7,
            config=BrainAblationConfig(
                name="counterfactual_credit",
                architecture="modular",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                credit_strategy="counterfactual",
            ),
        )
        action_dim = len(LOCOMOTION_ACTIONS)

        def module_result(module_name: str, logits: np.ndarray) -> ModuleResult:
            """
            Create a ready-to-use active ModuleResult for the given module name using the provided logits.

            Parameters:
                module_name (str): Module identifier; its interface is looked up to set `interface` and `observation_key`.
                logits (np.ndarray): Logits for the module's action outputs; length must match the action dimension used by the brain.

            Returns:
                ModuleResult: A ModuleResult with the resolved interface and observation_key, a zeroed observation vector sized to the interface's input dimension, the supplied `logits`, a uniform probability vector over actions, and `active=True`.
            """
            interface = MODULE_INTERFACE_BY_NAME[module_name]
            return ModuleResult(
                interface=interface,
                name=module_name,
                observation_key=interface.observation_key,
                observation=np.zeros(interface.input_dim, dtype=float),
                logits=logits,
                probs=np.full(action_dim, 1.0 / action_dim, dtype=float),
                active=True,
            )

        visual_logits = np.zeros(action_dim, dtype=float)
        sensory_logits = np.zeros(action_dim, dtype=float)
        visual_logits[0] = 1.0
        sensory_logits[0] = -1.0
        visual = module_result("visual_cortex", visual_logits)
        sensory = module_result("sensory_cortex", sensory_logits)
        def fake_action_center_forward(x, *, store_cache=False):
            """
            Compute a small corrective action logits vector from a concatenated visual+sensory input and return it with a zero baseline value.

            Parameters:
                x (array-like): Flattened input whose first `action_dim` entries are the visual segment and the next `action_dim` entries are the sensory segment; length must be at least 2 * action_dim.

            Returns:
                tuple:
                    correction (numpy.ndarray): 1D float array of length `action_dim` containing the corrective logits.
                    baseline (float): A scalar baseline value (always 0.0).
            """
            action_input = np.asarray(x, dtype=float)
            visual_segment = action_input[:action_dim]
            sensory_segment = action_input[action_dim : 2 * action_dim]
            correction = np.zeros(action_dim, dtype=float)
            correction[0] = (
                0.75 * visual_segment[0]
                - 0.50 * sensory_segment[0]
            )
            correction[1] = (
                -0.25 * visual_segment[0]
                + 0.25 * sensory_segment[0]
            )
            return correction, 0.0

        brain.action_center.forward = fake_action_center_forward

        weights = brain._compute_counterfactual_credit(
            [visual, sensory],
            self._build_observation(),
            action_idx=0,
        )

        self.assertEqual(set(weights), {"visual_cortex", "sensory_cortex"})
        self.assertAlmostEqual(sum(weights.values()), 1.0)
        self.assertGreater(weights["visual_cortex"], 0.0)
        self.assertGreater(weights["sensory_cortex"], 0.0)
        self.assertNotAlmostEqual(
            weights["visual_cortex"],
            weights["sensory_cortex"],
        )

    def test_counterfactual_learn_scales_shared_policy_broadcast(self) -> None:
        config = BrainAblationConfig(
            name="counterfactual_credit",
            architecture="modular",
            module_dropout=0.0,
            enable_reflexes=False,
            enable_auxiliary_targets=False,
            credit_strategy="counterfactual",
        )
        brain = SpiderBrain(seed=7, config=config)
        obs = self._build_observation()
        step = brain.act(obs, sample=False)
        weights = {
            result.name: float(index + 1)
            for index, result in enumerate(step.module_results)
        }
        weight_total = float(sum(weights.values()))
        weights = {
            name: value / weight_total
            for name, value in weights.items()
        }

        captured: dict[str, object] = {}
        proposal_grads = np.arange(
            len(step.module_results) * len(LOCOMOTION_ACTIONS),
            dtype=float,
        ).reshape(len(step.module_results), len(LOCOMOTION_ACTIONS))
        upstream = np.concatenate(
            [
                proposal_grads.reshape(-1),
                np.zeros(ACTION_CONTEXT_INTERFACE.input_dim, dtype=float),
            ]
        )

        def fake_action_center_backward(*, grad_policy_logits, grad_value, lr):
            """
            Stub backward function used in tests to capture the shared policy gradient.

            Stores a copy of `grad_policy_logits` (as a float numpy array) into `captured["shared_grad"]` and returns the incoming `upstream` value unchanged.

            Parameters:
                grad_policy_logits: array-like
                    Gradients for the policy logits to be captured.
                grad_value:
                    Ignored by this stub.
                lr:
                    Ignored by this stub.

            Returns:
                upstream: the unchanged upstream value passed through by the stub.
            """
            captured["shared_grad"] = np.asarray(grad_policy_logits, dtype=float).copy()
            return upstream

        def fake_module_bank_backward(grad_logits, lr, aux_grads=None):
            """
            Capture and store a copy of auxiliary gradients into the outer `captured` mapping.

            Parameters:
                grad_logits: Unused in this helper; present to match the real backward signature.
                lr: Unused learning-rate parameter; present to match the real backward signature.
                aux_grads (dict[str, array_like] | None): Mapping from module name to gradient array. Each value is converted to a NumPy float array and stored as a copy in `captured["aux_grads"]`.
            """
            captured["aux_grads"] = {
                name: np.asarray(value, dtype=float).copy()
                for name, value in (aux_grads or {}).items()
            }

        brain._compute_counterfactual_credit = lambda module_results, observation, action_idx: dict(weights)
        brain.action_center.backward = fake_action_center_backward
        brain.module_bank.backward = fake_module_bank_backward
        brain.motor_cortex.backward = lambda grad_logits, lr: None
        brain.estimate_value = lambda _: 0.0

        stats = brain.learn(step, reward=0.5, next_observation=obs, done=False)

        shared_grad = np.asarray(captured["shared_grad"], dtype=float)
        for idx, result in enumerate(step.module_results):
            expected_total_grad = result.gate_weight * (
                weights[result.name] * shared_grad + proposal_grads[idx]
            )
            np.testing.assert_allclose(
                captured["aux_grads"][result.name],
                expected_total_grad,
            )
            self.assertAlmostEqual(
                stats["module_gradient_norms"][result.name],
                float(np.linalg.norm(expected_total_grad)),
            )
        self.assertEqual(stats["credit_strategy"], "counterfactual")
        self.assertEqual(stats["module_credit_weights"], weights)
        self.assertEqual(stats["counterfactual_credit_weights"], weights)

    def test_monolithic_learn_adds_action_center_input_grads_to_policy_update(self) -> None:
        """
        Test that monolithic learning adds action-center input gradients to the monolithic policy update and records credit diagnostics.

        Sets up a monolithic SpiderBrain with stubbed backward methods to capture gradients, runs a single learn step, and asserts:
        - the monolithic policy backward receives the elementwise sum of the shared policy gradient and action-center input gradients,
        - the motor cortex receives only the shared policy gradient,
        - learning stats include `credit_strategy == "broadcast"`,
        - `module_credit_weights` contains the monolithic policy with weight 1.0,
        - `module_gradient_norms` contains the norm of the monolithic policy gradient.
        """
        brain = SpiderBrain(seed=11, config=self._monolithic_config())
        obs = self._build_observation()
        step = brain.act(obs, sample=False)

        captured: dict[str, object] = {}
        extra_grad = np.linspace(0.1, 0.5, len(LOCOMOTION_ACTIONS), dtype=float)
        upstream = np.concatenate(
            [
                extra_grad,
                np.zeros(ACTION_CONTEXT_INTERFACE.input_dim, dtype=float),
            ]
        )

        def fake_action_center_backward(*, grad_policy_logits, grad_value, lr):
            captured["shared_grad"] = np.asarray(grad_policy_logits, dtype=float).copy()
            return upstream

        def fake_monolithic_backward(grad_logits, lr):
            captured["mono_grad"] = np.asarray(grad_logits, dtype=float).copy()
            captured["mono_lr"] = float(lr)

        def fake_motor_backward(grad_logits, lr):
            captured["motor_grad"] = np.asarray(grad_logits, dtype=float).copy()
            captured["motor_lr"] = float(lr)

        brain.action_center.backward = fake_action_center_backward
        brain.monolithic_policy.backward = fake_monolithic_backward
        brain.motor_cortex.backward = fake_motor_backward
        brain.estimate_value = lambda _: 0.0

        stats = brain.learn(step, reward=0.5, next_observation=obs, done=False)

        np.testing.assert_allclose(
            captured["mono_grad"],
            captured["shared_grad"] + extra_grad,
        )
        self.assertEqual(stats["credit_strategy"], "broadcast")
        self.assertEqual(
            stats["module_credit_weights"],
            {SpiderBrain.MONOLITHIC_POLICY_NAME: 1.0},
        )
        self.assertAlmostEqual(
            stats["module_gradient_norms"][SpiderBrain.MONOLITHIC_POLICY_NAME],
            float(np.linalg.norm(captured["mono_grad"])),
        )
        np.testing.assert_allclose(
            captured["motor_grad"],
            captured["shared_grad"],
        )

    def test_monolithic_frozen_proposer_skips_weight_update_and_reports_zero_norm(self) -> None:
        brain = SpiderBrain(seed=11, config=self._monolithic_config())
        obs = self._build_observation()
        step = brain.act(obs, sample=True)

        brain.freeze_proposers([SpiderBrain.MONOLITHIC_POLICY_NAME])
        monolithic_before = self._numeric_state(brain.monolithic_policy)
        action_center_before = self._numeric_state(brain.action_center)
        motor_cortex_before = self._numeric_state(brain.motor_cortex)

        stats = brain.learn(step, reward=0.5, next_observation=obs, done=False)

        self.assertEqual(
            brain.frozen_module_names(),
            [SpiderBrain.MONOLITHIC_POLICY_NAME],
        )
        self.assertFalse(
            self._state_changed(
                monolithic_before,
                self._numeric_state(brain.monolithic_policy),
            )
        )
        self.assertEqual(
            stats["module_gradient_norms"][SpiderBrain.MONOLITHIC_POLICY_NAME],
            0.0,
        )
        self.assertTrue(
            self._state_changed(
                action_center_before,
                self._numeric_state(brain.action_center),
            )
        )
        self.assertTrue(
            self._state_changed(
                motor_cortex_before,
                self._numeric_state(brain.motor_cortex),
            )
        )

    def _assert_monolithic_non_broadcast_credit_coerces_to_broadcast(
        self,
        credit_strategy: str,
    ) -> None:
        brain = SpiderBrain(
            seed=13,
            config=self._monolithic_config(credit_strategy=credit_strategy),
        )
        obs = self._build_observation()
        step = brain.act(obs, sample=False)

        captured: dict[str, object] = {}
        extra_grad = np.linspace(0.1, 0.5, len(LOCOMOTION_ACTIONS), dtype=float)
        upstream = np.concatenate(
            [
                extra_grad,
                np.zeros(ACTION_CONTEXT_INTERFACE.input_dim, dtype=float),
            ]
        )

        def fake_action_center_backward(*, grad_policy_logits, grad_value, lr):
            captured["shared_grad"] = np.asarray(grad_policy_logits, dtype=float).copy()
            return upstream

        def fake_monolithic_backward(grad_logits, lr):
            captured["mono_grad"] = np.asarray(grad_logits, dtype=float).copy()

        brain.action_center.backward = fake_action_center_backward
        brain.monolithic_policy.backward = fake_monolithic_backward
        brain.motor_cortex.backward = lambda grad_logits, lr: None
        brain.estimate_value = lambda _: 0.0

        stats = brain.learn(step, reward=0.5, next_observation=obs, done=False)

        np.testing.assert_allclose(
            captured["mono_grad"],
            captured["shared_grad"] + extra_grad,
        )
        self.assertEqual(stats["credit_strategy"], "broadcast")
        self.assertEqual(
            stats["module_credit_weights"],
            {SpiderBrain.MONOLITHIC_POLICY_NAME: 1.0},
        )
        self.assertEqual(stats["counterfactual_credit_weights"], {})
        self.assertAlmostEqual(
            stats["module_gradient_norms"][SpiderBrain.MONOLITHIC_POLICY_NAME],
            float(np.linalg.norm(captured["mono_grad"])),
        )

    def test_monolithic_local_only_credit_reports_effective_broadcast(self) -> None:
        self._assert_monolithic_non_broadcast_credit_coerces_to_broadcast("local_only")

    def test_monolithic_counterfactual_credit_reports_effective_broadcast(self) -> None:
        self._assert_monolithic_non_broadcast_credit_coerces_to_broadcast(
            "counterfactual"
        )
