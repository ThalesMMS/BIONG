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

from tests.fixtures.arbitration import (
    _blank_obs,
    _module_interface,
    _vector_for,
    _valence_vector,
    _small_arbitration_evidence_vector,
    _assert_relative_scaled_logits,
)

class ComputeArbitrationTest(unittest.TestCase):
    """Tests for SpiderBrain._compute_arbitration."""

    def setUp(self) -> None:
        self.brain = SpiderBrain(seed=3, module_dropout=0.0)
        self.obs = _blank_obs()

    def _get_arbitration(self, obs=None):
        if obs is None:
            obs = self.obs
        step = self.brain.act(obs, bus=None, sample=False)
        return step.arbitration_decision

    def _make_brain_with_arbitration_winner(
        self,
        *,
        seed: int,
        winner: str,
        config: BrainAblationConfig | None = None,
        config_name: str | None = None,
    ) -> SpiderBrain:
        if config is not None and config_name is not None:
            raise ValueError("Provide either config or config_name, not both.")
        if config_name is not None:
            config = canonical_ablation_configs(module_dropout=0.0)[config_name]
        if config is None:
            brain = SpiderBrain(seed=seed, module_dropout=0.0)
        else:
            brain = SpiderBrain(seed=seed, config=config)
        brain.arbitration_network.W2_valence.fill(0.0)
        brain.arbitration_network.b2_valence[:] = _valence_vector(**{winner: 5.0})
        return brain

    def _build_obs(
        self,
        *,
        action_context_kwargs: dict[str, float],
        module_name: str,
        module_kwargs: dict[str, float],
    ) -> dict[str, np.ndarray]:
        obs = _blank_obs()
        obs["action_context"] = _vector_for(
            ACTION_CONTEXT_INTERFACE,
            **action_context_kwargs,
        )
        iface = _module_interface(module_name)
        obs[iface.observation_key] = _vector_for(iface, **module_kwargs)
        return obs

    def test_returns_arbitration_decision(self) -> None:
        arb = self._get_arbitration()
        self.assertIsInstance(arb, ArbitrationDecision)

    def test_strategy_is_priority_gating(self) -> None:
        arb = self._get_arbitration()
        self.assertEqual(arb.strategy, "priority_gating")

    def test_valence_scores_sum_to_one(self) -> None:
        arb = self._get_arbitration()
        total = sum(arb.valence_scores.values())
        self.assertAlmostEqual(total, 1.0, places=5)

    def test_valence_scores_all_non_negative(self) -> None:
        arb = self._get_arbitration()
        for score in arb.valence_scores.values():
            self.assertGreaterEqual(score, 0.0)

    def test_winning_valence_is_in_valence_order(self) -> None:
        arb = self._get_arbitration()
        self.assertIn(arb.winning_valence, VALENCE_ORDER)

    def test_module_gates_contains_all_active_modules(self) -> None:
        arb = self._get_arbitration()
        for iface in MODULE_INTERFACES:
            self.assertIn(iface.name, arb.module_gates)

    def test_module_gate_values_in_unit_range(self) -> None:
        arb = self._get_arbitration()
        for weight in arb.module_gates.values():
            self.assertGreaterEqual(weight, 0.0)
            self.assertLessEqual(weight, 1.0)

    def test_learned_arbitration_fields_are_present(self) -> None:
        arb = self._get_arbitration()
        self.assertEqual(set(arb.valence_logits.keys()), set(VALENCE_ORDER))
        self.assertEqual(set(arb.base_gates.keys()), set(arb.module_gates.keys()))
        self.assertEqual(set(arb.gate_adjustments.keys()), set(arb.module_gates.keys()))
        self.assertTrue(arb.learned_adjustment)
        self.assertTrue(np.isfinite(arb.arbitration_value))
        self.assertFalse(arb.guards_applied)
        self.assertFalse(arb.food_bias_applied)
        self.assertIsNone(arb.food_bias_action)

    def test_no_active_modules_reports_no_dominant_module(self) -> None:
        module_results = self.brain._proposal_results(
            self.obs,
            store_cache=False,
            training=False,
        )
        for result in module_results:
            result.active = False
            result.logits = np.zeros_like(result.logits)

        arb = self.brain._compute_arbitration(
            module_results,
            self.obs,
            training=False,
            store_cache=False,
        )

        self.assertTrue(
            all(share == 0.0 for share in arb.module_contribution_share.values())
        )
        self.assertEqual(arb.dominant_module, "")
        self.assertEqual(arb.dominant_module_share, 0.0)
        self.assertEqual(arb.effective_module_count, 0.0)

    def test_default_config_trace_payload_marks_scaffolding_inactive(self) -> None:
        bus = MessageBus()

        step = self.brain.act(self.obs, bus=bus, sample=False, training=False)
        selection_messages = bus.topic_messages("action.selection")
        self.assertTrue(selection_messages)
        selection_payload = selection_messages[-1].payload

        self.assertFalse(step.arbitration_decision.guards_applied)
        self.assertFalse(step.arbitration_decision.food_bias_applied)
        self.assertIsNone(step.arbitration_decision.food_bias_action)
        self.assertIn("guards_applied", selection_payload)
        self.assertIn("food_bias_applied", selection_payload)
        self.assertIn("food_bias_action", selection_payload)
        self.assertFalse(selection_payload["guards_applied"])
        self.assertFalse(selection_payload["food_bias_applied"])
        self.assertIsNone(selection_payload["food_bias_action"])

    def test_gate_adjustments_are_constrained_before_final_clamp(self) -> None:
        arb = self._get_arbitration()
        for adjustment in arb.gate_adjustments.values():
            self.assertGreaterEqual(adjustment, ArbitrationNetwork.GATE_ADJUSTMENT_MIN)
            self.assertLessEqual(adjustment, ArbitrationNetwork.GATE_ADJUSTMENT_MAX)

    def test_fixed_arbitration_config_uses_fixed_formula_path(self) -> None:
        """
        Verify that when learned arbitration is disabled the brain follows the fixed-formula arbitration path.

        Asserts that:
        - the arbitration decision marks learned_adjustment as False,
        - the arbitration network cache is None,
        - every module's gate_adjustment equals 1.0 and module_gates equal the reported base_gates,
        - arbitration_value equals 0.0.
        """
        brain = SpiderBrain(
            seed=3,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="fixed_arbitration_baseline",
                use_learned_arbitration=False,
            ),
        )
        step = brain.act(self.obs, bus=None, sample=False)
        arb = step.arbitration_decision
        self.assertIsNotNone(arb)
        self.assertFalse(arb.learned_adjustment)
        self.assertIsNone(brain.arbitration_network.cache)
        for module_name, base_gate in arb.base_gates.items():
            self.assertAlmostEqual(arb.gate_adjustments[module_name], 1.0)
            self.assertAlmostEqual(arb.module_gates[module_name], base_gate)
        self.assertAlmostEqual(arb.arbitration_value, 0.0)

    def test_learned_arbitration_uses_learned_winner_under_food_predator_conflict(self) -> None:
        """
        Assert that the benchmark path does not force a hard threat override when strong predator signals conflict with high hunger.

        The selected valence should be the learned deterministic winner for the
        current scores, with deterministic guards left inactive by default.
        """
        obs = _blank_obs()
        obs["action_context"] = _vector_for(
            ACTION_CONTEXT_INTERFACE,
            hunger=0.96,
            day=1.0,
            predator_visible=1.0,
            predator_certainty=0.9,
        )
        obs["alert"] = _vector_for(
            _module_interface("alert_center"),
            predator_visible=1.0,
            predator_certainty=0.9,
            predator_motion_salience=0.8,
            predator_smell_strength=0.7,
        )
        obs["hunger"] = _vector_for(
            _module_interface("hunger_center"),
            hunger=0.96,
            food_visible=1.0,
            food_certainty=0.9,
            food_smell_strength=0.8,
            food_memory_age=0.0,
        )

        arb = self._get_arbitration(obs)

        self.assertTrue(arb.learned_adjustment)
        self.assertFalse(arb.guards_applied)
        self.assertEqual(
            arb.winning_valence,
            deterministic_valence_winner(arb.valence_scores),
        )

    def test_deterministic_threat_guard_disabled_by_default(self) -> None:
        obs = self._build_obs(
            action_context_kwargs={
                "predator_visible": 1.0,
                "predator_certainty": 0.9,
            },
            module_name="alert_center",
            module_kwargs={
                "predator_visible": 1.0,
                "predator_certainty": 0.9,
            },
        )
        self.brain.arbitration_network.W2_valence.fill(0.0)
        self.brain.arbitration_network.b2_valence[:] = _valence_vector(hunger=5.0)
        module_results = self.brain._proposal_results(
            obs,
            store_cache=False,
            training=False,
        )

        arb = self.brain._compute_arbitration(
            module_results,
            obs,
            training=False,
            store_cache=False,
        )

        self.assertEqual(arb.winning_valence, "hunger")
        self.assertFalse(arb.guards_applied)

    def test_deterministic_threat_guard_can_be_enabled(self) -> None:
        brain = self._make_brain_with_arbitration_winner(
            seed=3,
            winner="hunger",
            config_name="constrained_arbitration",
        )
        obs = self._build_obs(
            action_context_kwargs={
                "predator_visible": 1.0,
                "predator_certainty": 0.9,
            },
            module_name="alert_center",
            module_kwargs={
                "predator_visible": 1.0,
                "predator_certainty": 0.9,
            },
        )
        module_results = brain._proposal_results(
            obs,
            store_cache=False,
            training=False,
        )

        arb = brain._compute_arbitration(
            module_results,
            obs,
            training=False,
            store_cache=False,
        )

        self.assertEqual(arb.winning_valence, "threat")
        self.assertTrue(arb.guards_applied)

    def test_deterministic_hunger_guard_can_be_enabled(self) -> None:
        brain = self._make_brain_with_arbitration_winner(
            seed=3,
            winner="exploration",
            config_name="constrained_arbitration",
        )
        obs = self._build_obs(
            action_context_kwargs={
                "hunger": 0.9,
                "night": 0.0,
            },
            module_name="hunger_center",
            module_kwargs={
                "hunger": 0.9,
            },
        )
        module_results = brain._proposal_results(
            obs,
            store_cache=False,
            training=False,
        )

        arb = brain._compute_arbitration(
            module_results,
            obs,
            training=False,
            store_cache=False,
        )

        self.assertEqual(arb.winning_valence, "hunger")
        self.assertTrue(arb.guards_applied)

    def test_food_direction_bias_disabled_by_default(self) -> None:
        brain = self._make_brain_with_arbitration_winner(
            seed=5,
            winner="hunger",
            config=default_brain_config(module_dropout=0.0),
        )
        obs = self._build_obs(
            action_context_kwargs={
                "hunger": 0.9,
                "day": 1.0,
            },
            module_name="hunger_center",
            module_kwargs={
                "hunger": 0.9,
                "food_visible": 1.0,
                "food_certainty": 1.0,
                "food_dx": 1.0,
                "food_dy": 0.0,
            },
        )

        step = brain.act(obs, bus=None, sample=False, training=False)

        self.assertEqual(step.arbitration_decision.winning_valence, "hunger")
        self.assertFalse(step.arbitration_decision.food_bias_applied)
        self.assertIsNone(step.arbitration_decision.food_bias_action)

    def test_food_direction_bias_can_be_enabled(self) -> None:
        base_brain = self._make_brain_with_arbitration_winner(
            seed=5,
            winner="hunger",
            config=default_brain_config(module_dropout=0.0),
        )
        biased_brain = self._make_brain_with_arbitration_winner(
            seed=5,
            winner="hunger",
            config_name="constrained_arbitration",
        )
        obs = self._build_obs(
            action_context_kwargs={
                "hunger": 0.9,
                "day": 1.0,
            },
            module_name="hunger_center",
            module_kwargs={
                "hunger": 0.9,
                "food_visible": 1.0,
                "food_certainty": 1.0,
                "food_dx": 1.0,
                "food_dy": 0.0,
            },
        )

        base_step = base_brain.act(obs, bus=None, sample=False, training=False)
        biased_step = biased_brain.act(obs, bus=None, sample=False, training=False)

        self.assertFalse(base_step.arbitration_decision.food_bias_applied)
        self.assertTrue(biased_step.arbitration_decision.food_bias_applied)
        self.assertEqual(biased_step.arbitration_decision.food_bias_action, "MOVE_RIGHT")
        diff = biased_step.total_logits - base_step.total_logits
        expected = np.zeros_like(diff)
        expected[LOCOMOTION_ACTIONS.index("MOVE_RIGHT")] = 3.0
        np.testing.assert_allclose(diff, expected)

    def test_food_direction_bias_skips_stay_direction(self) -> None:
        brain = self._make_brain_with_arbitration_winner(
            seed=5,
            winner="hunger",
            config_name="constrained_arbitration",
        )
        obs = self._build_obs(
            action_context_kwargs={
                "hunger": 0.9,
                "day": 1.0,
            },
            module_name="hunger_center",
            module_kwargs={
                "hunger": 0.9,
                "food_visible": 1.0,
                "food_certainty": 1.0,
                "food_dx": 0.04,
                "food_dy": 0.04,
            },
        )

        step = brain.act(obs, bus=None, sample=False, training=False)

        self.assertEqual(step.arbitration_decision.winning_valence, "hunger")
        self.assertFalse(step.arbitration_decision.food_bias_applied)
        self.assertIsNone(step.arbitration_decision.food_bias_action)

    def test_food_bias_flip_does_not_count_as_final_reflex_override(self) -> None:
        config = BrainAblationConfig(
            name="food_bias_telemetry_regression",
            module_dropout=0.0,
            enable_reflexes=False,
            enable_deterministic_guards=True,
            enable_food_direction_bias=True,
        )
        brain = SpiderBrain(seed=5, config=config)
        self.assertIsNotNone(brain.module_bank)
        for network in brain.module_bank.modules.values():
            network.W1.fill(0.0)
            network.b1.fill(0.0)
            network.W2.fill(0.0)
            network.b2.fill(0.0)
        brain.action_center.W1.fill(0.0)
        brain.action_center.b1.fill(0.0)
        brain.action_center.W2_policy.fill(0.0)
        brain.action_center.b2_policy.fill(0.0)
        brain.action_center.W2_value.fill(0.0)
        brain.action_center.b2_value.fill(0.0)
        brain.motor_cortex.W1.fill(0.0)
        brain.motor_cortex.b1.fill(0.0)
        brain.motor_cortex.W2.fill(0.0)
        brain.motor_cortex.b2.fill(0.0)
        brain.arbitration_network.W2_valence.fill(0.0)
        brain.arbitration_network.b2_valence[:] = _valence_vector(hunger=5.0)
        obs = _blank_obs()
        obs["action_context"] = _vector_for(
            ACTION_CONTEXT_INTERFACE,
            hunger=0.9,
            day=1.0,
        )
        obs["hunger"] = _vector_for(
            _module_interface("hunger_center"),
            hunger=0.9,
            food_visible=1.0,
            food_certainty=1.0,
            food_dx=1.0,
            food_dy=0.0,
        )

        step = brain.act(obs, bus=None, sample=False, training=False)

        self.assertTrue(step.arbitration_decision.food_bias_applied)
        self.assertEqual(step.arbitration_decision.food_bias_action, "MOVE_RIGHT")
        self.assertEqual(step.motor_action_idx, LOCOMOTION_ACTIONS.index("MOVE_RIGHT"))
        self.assertFalse(step.final_reflex_override)

    def test_regularization_term_is_reported_when_enabled(self) -> None:
        """
        Verify that arbitration regularization metrics are included and positive when training is enabled and arbitration gate parameters are non-zero.

        This test enables a non-zero gate head bias, performs an action step in training mode, calls the learning update, and asserts that the returned statistics contain a positive "regularization_loss" and a positive "arbitration_gate_regularization_norm".
        """
        self.brain.arbitration_network.b2_gate.fill(1.25)
        decision = self.brain.act(self.obs, bus=None, sample=False, training=True)

        stats = self.brain.learn(
            decision,
            reward=0.25,
            next_observation=self.obs,
            done=True,
        )

        self.assertGreater(stats["regularization_loss"], 0.0)
        self.assertGreater(stats["arbitration_gate_regularization_norm"], 0.0)

    def test_suppressed_modules_is_list(self) -> None:
        arb = self._get_arbitration()
        self.assertIsInstance(arb.suppressed_modules, list)

    def test_evidence_contains_four_valences(self) -> None:
        arb = self._get_arbitration()
        for valence in VALENCE_ORDER:
            self.assertIn(valence, arb.evidence)

    def test_intent_indices_are_valid_action_indices(self) -> None:
        arb = self._get_arbitration()
        self.assertIn(arb.intent_before_gating_idx, range(len(LOCOMOTION_ACTIONS)))
        self.assertIn(arb.intent_after_gating_idx, range(len(LOCOMOTION_ACTIONS)))

    def test_blank_obs_defaults_exploration_wins_or_has_nonzero_score(self) -> None:
        # With blank (all-zero) observation, exploration should have score > 0
        # because residual_drive is high when threat/hunger/sleep are zero
        arb = self._get_arbitration()
        self.assertGreater(arb.valence_scores["exploration"], 0.0)

    def test_suppressed_modules_are_subset_of_module_gate_keys(self) -> None:
        arb = self._get_arbitration()
        for name in arb.suppressed_modules:
            self.assertIn(name, arb.module_gates)

    def test_suppressed_module_gates_are_less_than_one(self) -> None:
        arb = self._get_arbitration()
        for name in arb.suppressed_modules:
            self.assertLess(arb.module_gates[name], 0.999)

    def test_threat_evidence_has_expected_keys(self) -> None:
        arb = self._get_arbitration()
        for key in ("predator_visible", "predator_certainty", "predator_motion_salience",
                    "recent_contact", "recent_pain", "predator_smell_strength"):
            self.assertIn(key, arb.evidence["threat"])

    def test_hunger_evidence_has_expected_keys(self) -> None:
        arb = self._get_arbitration()
        for key in ("hunger", "on_food", "food_visible", "food_certainty",
                    "food_smell_strength", "food_memory_freshness"):
            self.assertIn(key, arb.evidence["hunger"])

    def test_sleep_evidence_has_expected_keys(self) -> None:
        arb = self._get_arbitration()
        for key in ("fatigue", "sleep_debt", "night", "on_shelter",
                    "shelter_role_level", "shelter_path_confidence"):
            self.assertIn(key, arb.evidence["sleep"])

    def test_exploration_evidence_has_expected_keys(self) -> None:
        arb = self._get_arbitration()
        for key in ("safety_margin", "residual_drive", "day", "off_shelter",
                    "visual_openness", "food_smell_directionality"):
            self.assertIn(key, arb.evidence["exploration"])

    def test_threat_evidence_does_not_contain_predator_proximity(self) -> None:
        """Regression: predator_proximity was removed from threat evidence in this PR."""
        arb = self._get_arbitration()
        self.assertNotIn("predator_proximity", arb.evidence["threat"])

    def test_decision_logic_uses_abstract_predator_evidence_only(self) -> None:
        """Runtime arbitration evidence uses abstract predator signals, not diagnostic distances."""
        forbidden = {"diagnostic_predator_dist", "predator_dist"}
        obs = _blank_obs()
        obs["action_context"] = _vector_for(
            ACTION_CONTEXT_INTERFACE,
            predator_visible=1.0,
            predator_certainty=0.75,
            recent_contact=0.25,
            recent_pain=0.5,
        )
        obs["alert"] = _vector_for(
            _module_interface("alert_center"),
            predator_motion_salience=0.6,
            predator_smell_strength=0.4,
        )

        arb = self._get_arbitration(obs)
        threat = arb.evidence["threat"]

        self.assertAlmostEqual(threat["predator_visible"], 1.0)
        self.assertAlmostEqual(threat["predator_certainty"], 0.75)
        self.assertAlmostEqual(threat["predator_motion_salience"], 0.6)
        self.assertAlmostEqual(threat["recent_contact"], 0.25)
        self.assertAlmostEqual(threat["recent_pain"], 0.5)
        self.assertAlmostEqual(threat["predator_smell_strength"], 0.4)
        for valence, evidence in arb.evidence.items():
            with self.subTest(valence=valence):
                self.assertTrue(all(0.0 <= float(value) <= 1.0 for value in evidence.values()))
                for name in forbidden:
                    self.assertNotIn(name, evidence)

    def test_sleep_evidence_does_not_contain_home_pressure(self) -> None:
        """Regression: home_pressure was removed from sleep evidence in this PR."""
        arb = self._get_arbitration()
        self.assertNotIn("home_pressure", arb.evidence["sleep"])

    def test_threat_evidence_predator_motion_salience_in_zero_to_one(self) -> None:
        """predator_motion_salience evidence value must be clamped to [0, 1]."""
        arb = self._get_arbitration()
        val = arb.evidence["threat"]["predator_motion_salience"]
        self.assertGreaterEqual(val, 0.0)
        self.assertLessEqual(val, 1.0)

    def test_sleep_evidence_shelter_path_confidence_in_zero_to_one(self) -> None:
        """shelter_path_confidence evidence value must be clamped to [0, 1]."""
        arb = self._get_arbitration()
        val = arb.evidence["sleep"]["shelter_path_confidence"]
        self.assertGreaterEqual(val, 0.0)
        self.assertLessEqual(val, 1.0)

class ApplyPriorityGatingTest(unittest.TestCase):
    """Tests for SpiderBrain._apply_priority_gating."""

    def setUp(self) -> None:
        self.brain = SpiderBrain(seed=11, module_dropout=0.0)
        self.obs = _blank_obs()

    def test_apply_priority_gating_sets_gate_weight(self) -> None:
        step = self.brain.act(self.obs, bus=None, sample=False)
        for result in step.module_results:
            self.assertIsNotNone(result.gate_weight)
            self.assertIsInstance(result.gate_weight, float)

    def test_apply_priority_gating_sets_gated_logits(self) -> None:
        step = self.brain.act(self.obs, bus=None, sample=False)
        for result in step.module_results:
            self.assertIsNotNone(result.gated_logits)
            self.assertEqual(result.gated_logits.shape, result.probs.shape)

    def test_apply_priority_gating_preserves_raw_logits(self) -> None:
        module_results = self.brain._proposal_results(
            self.obs,
            store_cache=False,
            training=False,
        )
        arbitration = self.brain._compute_arbitration(
            module_results,
            self.obs,
            training=False,
            store_cache=False,
        )
        raw_logits_by_name = {
            result.name: result.logits.copy()
            for result in module_results
        }

        apply_priority_gating(
            module_results,
            arbitration,
            module_valence_roles=self.brain.MODULE_VALENCE_ROLES,
        )

        for result in module_results:
            np.testing.assert_allclose(result.logits, raw_logits_by_name[result.name])
            np.testing.assert_allclose(
                result.gated_logits,
                result.gate_weight * raw_logits_by_name[result.name],
            )

    def test_action_center_input_uses_gated_logits(self) -> None:
        step = self.brain.act(self.obs, bus=None, sample=False)
        proposal_width = len(step.module_results) * len(LOCOMOTION_ACTIONS)
        expected = np.concatenate(
            [result.gated_logits for result in step.module_results],
            axis=0,
        )

        np.testing.assert_allclose(step.action_center_input[:proposal_width], expected)

    def test_apply_priority_gating_sets_valence_role(self) -> None:
        step = self.brain.act(self.obs, bus=None, sample=False)
        for result in step.module_results:
            self.assertIsNotNone(result.valence_role)
            self.assertIsInstance(result.valence_role, str)

    def test_alert_center_has_threat_valence_role(self) -> None:
        step = self.brain.act(self.obs, bus=None, sample=False)
        alert = next(r for r in step.module_results if r.name == "alert_center")
        self.assertEqual(alert.valence_role, "threat")

    def test_sleep_center_has_sleep_valence_role(self) -> None:
        step = self.brain.act(self.obs, bus=None, sample=False)
        sleep = next(r for r in step.module_results if r.name == "sleep_center")
        self.assertEqual(sleep.valence_role, "sleep")

    def test_visual_cortex_has_support_valence_role(self) -> None:
        step = self.brain.act(self.obs, bus=None, sample=False)
        visual = next(r for r in step.module_results if r.name == "visual_cortex")
        self.assertEqual(visual.valence_role, "support")

    def test_probs_sum_to_one_after_gating(self) -> None:
        step = self.brain.act(self.obs, bus=None, sample=False)
        for result in step.module_results:
            self.assertAlmostEqual(float(np.sum(result.probs)), 1.0, places=5)

    def test_gate_weight_one_for_primary_module_under_threat(self) -> None:
        # When threat wins, alert_center should have gate_weight = 1.0
        obs = _blank_obs()
        obs["action_context"] = _vector_for(
            ACTION_CONTEXT_INTERFACE,
            predator_visible=1.0,
            predator_certainty=1.0,
            recent_contact=1.0,
            recent_pain=0.8,
        )
        obs["alert"] = _vector_for(
            _module_interface("alert_center"),
            predator_visible=1.0,
            predator_smell_strength=1.0,
            predator_motion_salience=1.0,
        )
        step = self.brain.act(obs, bus=None, sample=False)
        self.assertEqual(step.arbitration_decision.winning_valence, "threat")
        alert = next(r for r in step.module_results if r.name == "alert_center")
        self.assertAlmostEqual(alert.gate_weight, 1.0, places=5)

class PriorityGatingWeightsTest(unittest.TestCase):
    """Tests for PRIORITY_GATING_WEIGHTS table correctness."""

    def test_all_four_valences_present(self) -> None:
        for valence in VALENCE_ORDER:
            self.assertIn(valence, PRIORITY_GATING_WEIGHTS)

    def test_each_valence_has_all_module_entries(self) -> None:
        expected_modules = {
            "alert_center", "hunger_center", "sleep_center",
            "visual_cortex", "sensory_cortex",
            "perception_center", "homeostasis_center", "threat_center",
            SpiderBrain.MONOLITHIC_POLICY_NAME,
        }
        for valence, weights in PRIORITY_GATING_WEIGHTS.items():
            self.assertEqual(set(weights.keys()), expected_modules,
                             f"Missing modules for valence '{valence}'")

    def test_threat_primary_module_has_weight_one(self) -> None:
        self.assertAlmostEqual(
            PRIORITY_GATING_WEIGHTS["threat"]["alert_center"], 1.0
        )

    def test_threat_suppresses_hunger_center(self) -> None:
        w = PRIORITY_GATING_WEIGHTS["threat"]["hunger_center"]
        self.assertLess(w, 0.5)

    def test_hunger_primary_module_has_weight_one(self) -> None:
        self.assertAlmostEqual(
            PRIORITY_GATING_WEIGHTS["hunger"]["hunger_center"], 1.0
        )

    def test_sleep_primary_module_has_weight_one(self) -> None:
        self.assertAlmostEqual(
            PRIORITY_GATING_WEIGHTS["sleep"]["sleep_center"], 1.0
        )

    def test_exploration_visual_cortex_high_weight(self) -> None:
        w = PRIORITY_GATING_WEIGHTS["exploration"]["visual_cortex"]
        self.assertGreater(w, 0.9)

    def test_all_gate_weights_in_unit_range(self) -> None:
        for valence, weights in PRIORITY_GATING_WEIGHTS.items():
            for module, weight in weights.items():
                self.assertGreaterEqual(weight, 0.0,
                    f"Gate weight for {valence}/{module} is negative")
                self.assertLessEqual(weight, 1.0,
                    f"Gate weight for {valence}/{module} exceeds 1.0")

    def test_monolithic_policy_always_weight_one(self) -> None:
        for valence in VALENCE_ORDER:
            self.assertAlmostEqual(
                PRIORITY_GATING_WEIGHTS[valence][SpiderBrain.MONOLITHIC_POLICY_NAME],
                1.0,
            )

    def test_sleep_suppresses_hunger_center(self) -> None:
        w = PRIORITY_GATING_WEIGHTS["sleep"]["hunger_center"]
        self.assertLess(w, 0.5)

    def test_valence_evidence_weight_tables_are_nested_immutable(self) -> None:
        with self.assertRaises(TypeError):
            VALENCE_EVIDENCE_WEIGHTS["threat"]["predator_visible"] = 0.0

    def test_missing_module_gate_weight_raises_clear_error(self) -> None:
        incomplete_weights = {
            valence: dict(weights)
            for valence, weights in PRIORITY_GATING_WEIGHTS.items()
        }
        del incomplete_weights["threat"]["alert_center"]
        obs = _blank_obs()
        obs["action_context"] = _vector_for(
            ACTION_CONTEXT_INTERFACE,
            predator_visible=1.0,
            predator_certainty=1.0,
            recent_contact=1.0,
            recent_pain=0.8,
        )
        obs["alert"] = _vector_for(
            _module_interface("alert_center"),
            predator_visible=1.0,
            predator_smell_strength=1.0,
            predator_motion_salience=1.0,
        )
        with patch.object(SpiderBrain, "PRIORITY_GATING_WEIGHTS", incomplete_weights):
            brain = SpiderBrain(seed=5, module_dropout=0.0)
            with self.assertRaisesRegex(
                ValueError,
                "Priority gating weights missing module 'alert_center'",
            ):
                brain.act(obs, bus=None, sample=False)

class FixedFormulaValenceScoresTest(unittest.TestCase):
    """Tests for SpiderBrain._fixed_formula_valence_scores_from_evidence."""

    def setUp(self) -> None:
        self.brain = SpiderBrain(seed=42, module_dropout=0.0)

    def _blank_evidence(self) -> dict:
        return {
            "threat": {
                "predator_visible": 0.0, "predator_certainty": 0.0,
                "predator_motion_salience": 0.0, "recent_contact": 0.0,
                "recent_pain": 0.0, "predator_smell_strength": 0.0,
            },
            "hunger": {
                "hunger": 0.0, "on_food": 0.0, "food_visible": 0.0,
                "food_certainty": 0.0, "food_smell_strength": 0.0,
                "food_memory_freshness": 0.0,
            },
            "sleep": {
                "fatigue": 0.0, "sleep_debt": 0.0, "night": 0.0,
                "on_shelter": 0.0, "shelter_role_level": 0.0,
                "shelter_path_confidence": 0.0,
            },
            "exploration": {
                "safety_margin": 0.0, "residual_drive": 0.0, "day": 0.0,
                "off_shelter": 0.0, "visual_openness": 0.0,
                "food_smell_directionality": 0.0,
            },
        }

    def test_returns_array_of_length_4(self) -> None:
        evidence = self._blank_evidence()
        scores = fixed_formula_valence_scores_from_evidence(evidence)
        self.assertEqual(scores.shape, (4,))

    def test_all_zero_evidence_returns_exploration_fallback(self) -> None:
        evidence = self._blank_evidence()
        scores = fixed_formula_valence_scores_from_evidence(evidence)
        # Fallback: [0.0, 0.0, 0.0, 1.0] = exploration
        np.testing.assert_array_almost_equal(scores, [0.0, 0.0, 0.0, 1.0])
        permuted_order = ("exploration", "threat", "hunger", "sleep")
        permuted_scores = fixed_formula_valence_scores_from_evidence(
            evidence,
            valence_order=permuted_order,
        )
        np.testing.assert_array_almost_equal(permuted_scores, [1.0, 0.0, 0.0, 0.0])

    def test_scores_sum_to_one_for_nonzero_input(self) -> None:
        evidence = self._blank_evidence()
        evidence["threat"]["predator_visible"] = 0.8
        evidence["hunger"]["hunger"] = 0.5
        scores = fixed_formula_valence_scores_from_evidence(evidence)
        self.assertAlmostEqual(float(np.sum(scores)), 1.0, places=6)

    def test_scores_are_non_negative(self) -> None:
        evidence = self._blank_evidence()
        evidence["sleep"]["fatigue"] = 0.9
        scores = fixed_formula_valence_scores_from_evidence(evidence)
        self.assertTrue(np.all(scores >= 0.0))

    def test_high_predator_visible_dominates_threat_score(self) -> None:
        evidence = self._blank_evidence()
        evidence["threat"]["predator_visible"] = 1.0
        evidence["threat"]["predator_certainty"] = 1.0
        scores = fixed_formula_valence_scores_from_evidence(evidence)
        # threat should be the largest score
        threat_idx = VALENCE_ORDER.index("threat")
        self.assertEqual(int(np.argmax(scores)), threat_idx)

    def test_high_hunger_dominates_hunger_score(self) -> None:
        evidence = self._blank_evidence()
        evidence["hunger"]["hunger"] = 1.0
        evidence["hunger"]["food_visible"] = 1.0
        scores = fixed_formula_valence_scores_from_evidence(evidence)
        hunger_idx = VALENCE_ORDER.index("hunger")
        self.assertEqual(int(np.argmax(scores)), hunger_idx)

    def test_high_fatigue_and_night_dominates_sleep_score(self) -> None:
        evidence = self._blank_evidence()
        evidence["sleep"]["fatigue"] = 1.0
        evidence["sleep"]["sleep_debt"] = 1.0
        evidence["sleep"]["night"] = 1.0
        evidence["sleep"]["on_shelter"] = 1.0
        scores = fixed_formula_valence_scores_from_evidence(evidence)
        sleep_idx = VALENCE_ORDER.index("sleep")
        self.assertEqual(int(np.argmax(scores)), sleep_idx)

    def test_float_array_output(self) -> None:
        evidence = self._blank_evidence()
        evidence["exploration"]["residual_drive"] = 0.5
        scores = fixed_formula_valence_scores_from_evidence(evidence)
        self.assertEqual(scores.dtype, float)

    def test_scores_ordered_by_valence_order(self) -> None:
        evidence = self._blank_evidence()
        evidence["threat"]["predator_visible"] = 0.9
        scores = fixed_formula_valence_scores_from_evidence(evidence)
        # First element is threat (index 0 in VALENCE_ORDER)
        self.assertEqual(VALENCE_ORDER[0], "threat")
        self.assertGreater(float(scores[0]), 0.0)
