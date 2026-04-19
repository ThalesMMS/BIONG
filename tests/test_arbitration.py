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


class ArbitrationModuleFunctionSignatureTest(unittest.TestCase):
    def test_evidence_helpers_accept_explicit_tables(self) -> None:
        evidence = {
            valence: {
                field: 0.0
                for field in ARBITRATION_EVIDENCE_FIELDS[valence]
            }
            for valence in VALENCE_ORDER
        }
        evidence["hunger"]["hunger"] = 1.0

        self.assertEqual(
            arbitration_evidence_input_dim(
                arbitration_evidence_fields=ARBITRATION_EVIDENCE_FIELDS,
            ),
            ArbitrationNetwork.INPUT_DIM,
        )
        self.assertEqual(
            arbitration_evidence_signal_names(
                valence_order=VALENCE_ORDER,
                arbitration_evidence_fields=ARBITRATION_EVIDENCE_FIELDS,
            ),
            ArbitrationNetwork.EVIDENCE_SIGNAL_NAMES,
        )
        vector = arbitration_evidence_vector(
            evidence,
            valence_order=VALENCE_ORDER,
            arbitration_evidence_fields=ARBITRATION_EVIDENCE_FIELDS,
        )
        self.assertEqual(vector.shape, (ArbitrationNetwork.INPUT_DIM,))
        scores = fixed_formula_valence_scores_from_evidence(
            evidence,
            valence_order=VALENCE_ORDER,
        )
        self.assertEqual(len(VALENCE_ORDER), len(scores))
        self.assertEqual(deterministic_valence_winner(dict(zip(VALENCE_ORDER, scores))), "hunger")

    def test_warm_start_accepts_network_and_config(self) -> None:
        brain = SpiderBrain(seed=17, module_dropout=0.0)
        brain.arbitration_network.forward(
            np.zeros(ArbitrationNetwork.INPUT_DIM),
            store_cache=True,
        )

        warm_start_arbitration_network(
            brain.arbitration_network,
            ablation_config=brain.config,
            warm_start_scale=1.0,
            valence_order=VALENCE_ORDER,
            arbitration_evidence_fields=ARBITRATION_EVIDENCE_FIELDS,
        )

        self.assertIsNone(brain.arbitration_network.cache)
        self.assertNotEqual(float(brain.arbitration_network.W1[0, 0]), 0.0)

    def test_compute_and_gating_accept_explicit_context(self) -> None:
        brain = SpiderBrain(seed=19, module_dropout=0.0)
        obs = _blank_obs()
        module_results = brain._proposal_results(
            obs,
            store_cache=False,
            training=False,
        )

        arbitration = compute_arbitration(
            obs,
            module_results,
            arbitration_network=brain.arbitration_network,
            ablation_config=brain.config,
            arbitration_rng=brain.arbitration_rng,
            operational_profile=brain.operational_profile,
            training=False,
            store_cache=False,
        )
        apply_priority_gating(
            module_results,
            arbitration,
            module_valence_roles=brain.MODULE_VALENCE_ROLES,
        )

        self.assertIsInstance(arbitration, ArbitrationDecision)
        self.assertGreaterEqual(
            priority_gate_weight_for(arbitration.winning_valence, module_results[0].name),
            0.0,
        )
        self.assertEqual(
            module_results[0].gate_weight,
            arbitration_gate_weight_for(arbitration, module_results[0].name),
        )


class ValenceScoreSerializationTest(unittest.TestCase):
    """Tests for ValenceScore.to_payload()."""

    def _make(self, name="threat", score=0.75, evidence=None):
        if evidence is None:
            evidence = {"predator_visible": 1.0, "recent_contact": 0.5}
        return ValenceScore(name=name, score=score, evidence=evidence)

    def test_to_payload_has_required_keys(self) -> None:
        vs = self._make()
        payload = vs.to_payload()
        self.assertIn("name", payload)
        self.assertIn("score", payload)
        self.assertIn("evidence", payload)

    def test_to_payload_name_preserved(self) -> None:
        vs = self._make(name="hunger")
        self.assertEqual(vs.to_payload()["name"], "hunger")

    def test_to_payload_score_rounded_to_six_decimals(self) -> None:
        vs = self._make(score=0.123456789)
        payload = vs.to_payload()
        self.assertEqual(payload["score"], round(0.123456789, 6))

    def test_to_payload_evidence_keys_sorted(self) -> None:
        vs = self._make(evidence={"z_key": 0.1, "a_key": 0.9, "m_key": 0.5})
        payload = vs.to_payload()
        keys = list(payload["evidence"].keys())
        self.assertEqual(keys, sorted(keys))

    def test_to_payload_evidence_values_rounded(self) -> None:
        vs = self._make(evidence={"k": 0.999999999})
        payload = vs.to_payload()
        self.assertEqual(payload["evidence"]["k"], round(0.999999999, 6))

    def test_to_payload_empty_evidence(self) -> None:
        vs = self._make(evidence={})
        payload = vs.to_payload()
        self.assertEqual(payload["evidence"], {})

    def test_to_payload_score_zero(self) -> None:
        vs = self._make(score=0.0)
        self.assertEqual(vs.to_payload()["score"], 0.0)

    def test_to_payload_is_json_serializable(self) -> None:
        import json
        vs = self._make()
        # Should not raise
        json.dumps(vs.to_payload())


# ---------------------------------------------------------------------------
# ArbitrationDecision serialization tests
# ---------------------------------------------------------------------------


class ArbitrationDecisionSerializationTest(unittest.TestCase):
    """Tests for ArbitrationDecision.to_payload()."""

    def _make(self, winning_valence="threat", intent_before=0, intent_after=2):
        return ArbitrationDecision(
            strategy="priority_gating",
            winning_valence=winning_valence,
            valence_scores={"threat": 0.6, "hunger": 0.2, "sleep": 0.1, "exploration": 0.1},
            module_gates={"alert_center": 1.0, "hunger_center": 0.18},
            suppressed_modules=["hunger_center"],
            evidence={
                "threat": {"predator_visible": 1.0},
                "hunger": {"hunger": 0.3},
            },
            intent_before_gating_idx=intent_before,
            intent_after_gating_idx=intent_after,
        )

    def test_to_payload_strategy_preserved(self) -> None:
        d = self._make()
        payload = d.to_payload()
        self.assertEqual(payload["strategy"], "priority_gating")

    def test_to_payload_winning_valence_preserved(self) -> None:
        d = self._make(winning_valence="hunger")
        self.assertEqual(d.to_payload()["winning_valence"], "hunger")

    def test_to_payload_has_all_required_keys(self) -> None:
        payload = self._make().to_payload()
        for key in ("strategy", "winning_valence", "valence_scores", "module_gates",
                    "valence_logits", "base_gates", "gate_adjustments",
                    "arbitration_value", "learned_adjustment",
                    "guards_applied", "food_bias_applied", "food_bias_action",
                    "module_contribution_share", "dominant_module", "dominant_module_share",
                    "effective_module_count", "module_agreement_rate", "module_disagreement_rate",
                    "suppressed_modules", "evidence", "intent_before_gating", "intent_after_gating"):
            self.assertIn(key, payload)

    def test_legacy_constructor_keeps_new_payload_fields_with_defaults(self) -> None:
        payload = self._make().to_payload()
        self.assertEqual(payload["valence_logits"], {})
        self.assertEqual(payload["base_gates"], {})
        self.assertEqual(payload["gate_adjustments"], {})
        self.assertEqual(payload["arbitration_value"], 0.0)
        self.assertFalse(payload["learned_adjustment"])
        self.assertFalse(payload["guards_applied"])
        self.assertFalse(payload["food_bias_applied"])
        self.assertIsNone(payload["food_bias_action"])
        self.assertEqual(payload["module_contribution_share"], {})
        self.assertEqual(payload["dominant_module"], "")
        self.assertEqual(payload["dominant_module_share"], 0.0)
        self.assertEqual(payload["effective_module_count"], 0.0)
        self.assertEqual(payload["module_agreement_rate"], 0.0)
        self.assertEqual(payload["module_disagreement_rate"], 0.0)

    def test_to_payload_module_contribution_share_sorted(self) -> None:
        d = ArbitrationDecision(
            strategy="priority_gating",
            winning_valence="threat",
            valence_scores={},
            module_gates={},
            suppressed_modules=[],
            evidence={},
            intent_before_gating_idx=0,
            intent_after_gating_idx=0,
            module_contribution_share={"visual_cortex": 0.25, "alert_center": 0.75},
        )
        keys = list(d.to_payload()["module_contribution_share"].keys())
        self.assertEqual(keys, sorted(keys))

    def test_to_payload_new_metrics_rounded(self) -> None:
        d = ArbitrationDecision(
            strategy="priority_gating",
            winning_valence="threat",
            valence_scores={},
            module_gates={},
            suppressed_modules=[],
            evidence={},
            intent_before_gating_idx=0,
            intent_after_gating_idx=0,
            module_contribution_share={"alert_center": 0.3333333333},
            dominant_module="alert_center",
            dominant_module_share=0.7777777777,
            effective_module_count=1.6666666666,
            module_agreement_rate=0.8888888888,
            module_disagreement_rate=0.1111111111,
        )
        payload = d.to_payload()
        self.assertEqual(payload["module_contribution_share"]["alert_center"], round(0.3333333333, 6))
        self.assertEqual(payload["dominant_module"], "alert_center")
        self.assertEqual(payload["dominant_module_share"], round(0.7777777777, 6))
        self.assertEqual(payload["effective_module_count"], round(1.6666666666, 6))
        self.assertEqual(payload["module_agreement_rate"], round(0.8888888888, 6))
        self.assertEqual(payload["module_disagreement_rate"], round(0.1111111111, 6))

    def test_to_payload_intent_before_gating_is_action_name(self) -> None:
        d = self._make(intent_before=0)
        payload = d.to_payload()
        self.assertIn(payload["intent_before_gating"], LOCOMOTION_ACTIONS)

    def test_to_payload_intent_after_gating_is_action_name(self) -> None:
        d = self._make(intent_after=2)
        payload = d.to_payload()
        self.assertIn(payload["intent_after_gating"], LOCOMOTION_ACTIONS)

    def test_to_payload_intent_names_differ_when_indices_differ(self) -> None:
        d = self._make(intent_before=0, intent_after=1)
        payload = d.to_payload()
        self.assertNotEqual(payload["intent_before_gating"], payload["intent_after_gating"])

    def test_to_payload_intent_names_same_when_indices_same(self) -> None:
        d = self._make(intent_before=3, intent_after=3)
        payload = d.to_payload()
        self.assertEqual(payload["intent_before_gating"], payload["intent_after_gating"])

    def test_to_payload_valence_scores_sorted(self) -> None:
        d = self._make()
        payload = d.to_payload()
        keys = list(payload["valence_scores"].keys())
        self.assertEqual(keys, sorted(keys))

    def test_to_payload_module_gates_sorted(self) -> None:
        d = self._make()
        payload = d.to_payload()
        keys = list(payload["module_gates"].keys())
        self.assertEqual(keys, sorted(keys))

    def test_to_payload_suppressed_modules_is_list(self) -> None:
        d = self._make()
        self.assertIsInstance(d.to_payload()["suppressed_modules"], list)

    def test_to_payload_evidence_inner_keys_sorted(self) -> None:
        d = ArbitrationDecision(
            strategy="priority_gating",
            winning_valence="threat",
            valence_scores={},
            module_gates={},
            suppressed_modules=[],
            evidence={"threat": {"z": 1.0, "a": 0.5}},
            intent_before_gating_idx=0,
            intent_after_gating_idx=0,
        )
        evidence_threat = d.to_payload()["evidence"]["threat"]
        keys = list(evidence_threat.keys())
        self.assertEqual(keys, sorted(keys))

    def test_to_payload_is_json_serializable(self) -> None:
        import json
        json.dumps(self._make().to_payload())

    def test_to_payload_valence_scores_rounded(self) -> None:
        d = ArbitrationDecision(
            strategy="priority_gating",
            winning_valence="threat",
            valence_scores={"threat": 0.1111111111},
            module_gates={},
            suppressed_modules=[],
            evidence={},
            intent_before_gating_idx=0,
            intent_after_gating_idx=0,
        )
        payload = d.to_payload()
        self.assertEqual(payload["valence_scores"]["threat"], round(0.1111111111, 6))

    def test_to_payload_exports_legacy_predator_proximity_for_scorers(self) -> None:
        d = self._make()
        self.assertNotIn("predator_proximity", d.evidence["threat"])
        payload = d.to_payload()
        self.assertEqual(payload["evidence"]["threat"]["predator_proximity"], 0.0)


# ---------------------------------------------------------------------------
# clamp_unit tests
# ---------------------------------------------------------------------------


class ClampUnitTest(unittest.TestCase):
    """Tests for clamp_unit."""

    def test_value_below_zero_clamped_to_zero(self) -> None:
        self.assertEqual(clamp_unit(-0.5), 0.0)

    def test_value_above_one_clamped_to_one(self) -> None:
        self.assertEqual(clamp_unit(1.5), 1.0)

    def test_exact_zero_unchanged(self) -> None:
        self.assertEqual(clamp_unit(0.0), 0.0)

    def test_exact_one_unchanged(self) -> None:
        self.assertEqual(clamp_unit(1.0), 1.0)

    def test_midrange_value_unchanged(self) -> None:
        self.assertAlmostEqual(clamp_unit(0.5), 0.5)

    def test_returns_float(self) -> None:
        result = clamp_unit(0.3)
        self.assertIsInstance(result, float)

    def test_large_negative_clamped_to_zero(self) -> None:
        self.assertEqual(clamp_unit(-100.0), 0.0)

    def test_large_positive_clamped_to_one(self) -> None:
        self.assertEqual(clamp_unit(100.0), 1.0)

    def test_very_small_positive_preserved(self) -> None:
        result = clamp_unit(1e-10)
        self.assertGreater(result, 0.0)
        self.assertLessEqual(result, 1.0)


# ---------------------------------------------------------------------------
# SpiderBrain._bound_observation tests
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# SpiderBrain._apply_priority_gating tests
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# PRIORITY_GATING_WEIGHTS constant tests
# ---------------------------------------------------------------------------


class PriorityGatingWeightsTest(unittest.TestCase):
    """Tests for PRIORITY_GATING_WEIGHTS table correctness."""

    def test_all_four_valences_present(self) -> None:
        for valence in VALENCE_ORDER:
            self.assertIn(valence, PRIORITY_GATING_WEIGHTS)

    def test_each_valence_has_all_module_entries(self) -> None:
        expected_modules = {
            "alert_center", "hunger_center", "sleep_center",
            "visual_cortex", "sensory_cortex", SpiderBrain.MONOLITHIC_POLICY_NAME,
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


# ---------------------------------------------------------------------------
# MODULE_VALENCE_ROLES constant tests
# ---------------------------------------------------------------------------


class ArbitrationEvidenceInputDimTest(unittest.TestCase):
    """Tests for arbitration_evidence_input_dim classmethod."""

    def test_returns_24(self) -> None:
        self.assertEqual(arbitration_evidence_input_dim(), 24)

    def test_is_classmethod_accessible(self) -> None:
        # Should be callable on the class, not just an instance
        result = arbitration_evidence_input_dim()
        self.assertIsInstance(result, int)

    def test_matches_arbitration_network_input_dim(self) -> None:
        self.assertEqual(
            arbitration_evidence_input_dim(),
            ArbitrationNetwork.INPUT_DIM,
        )

    def test_equals_sum_of_evidence_fields(self) -> None:
        total = sum(len(fields) for fields in ARBITRATION_EVIDENCE_FIELDS.values())
        self.assertEqual(arbitration_evidence_input_dim(), total)


# ---------------------------------------------------------------------------
# arbitration_evidence_signal_names tests
# ---------------------------------------------------------------------------


class ArbitrationEvidenceSignalNamesTest(unittest.TestCase):
    """Tests for arbitration_evidence_signal_names classmethod."""

    def test_returns_tuple(self) -> None:
        self.assertIsInstance(arbitration_evidence_signal_names(), tuple)

    def test_length_is_24(self) -> None:
        self.assertEqual(len(arbitration_evidence_signal_names()), 24)

    def test_first_signal_is_threat_predator_visible(self) -> None:
        names = arbitration_evidence_signal_names()
        self.assertEqual(names[0], "threat.predator_visible")

    def test_last_signal_is_exploration_food_smell_directionality(self) -> None:
        names = arbitration_evidence_signal_names()
        self.assertEqual(names[-1], "exploration.food_smell_directionality")

    def test_all_names_have_valence_prefix(self) -> None:
        names = arbitration_evidence_signal_names()
        valences = set(VALENCE_ORDER)
        for name in names:
            prefix = name.split(".")[0]
            self.assertIn(prefix, valences, f"Signal {name!r} has unknown valence prefix")

    def test_matches_arbitration_network_evidence_signal_names(self) -> None:
        self.assertEqual(
            arbitration_evidence_signal_names(),
            ArbitrationNetwork.EVIDENCE_SIGNAL_NAMES,
        )

    def test_threat_signals_come_before_hunger(self) -> None:
        names = arbitration_evidence_signal_names()
        threat_indices = [i for i, n in enumerate(names) if n.startswith("threat.")]
        hunger_indices = [i for i, n in enumerate(names) if n.startswith("hunger.")]
        self.assertLess(max(threat_indices), min(hunger_indices))

    def test_hunger_signals_come_before_sleep(self) -> None:
        names = arbitration_evidence_signal_names()
        hunger_indices = [i for i, n in enumerate(names) if n.startswith("hunger.")]
        sleep_indices = [i for i, n in enumerate(names) if n.startswith("sleep.")]
        self.assertLess(max(hunger_indices), min(sleep_indices))

    def test_sleep_signals_come_before_exploration(self) -> None:
        names = arbitration_evidence_signal_names()
        sleep_indices = [i for i, n in enumerate(names) if n.startswith("sleep.")]
        exploration_indices = [i for i, n in enumerate(names) if n.startswith("exploration.")]
        self.assertLess(max(sleep_indices), min(exploration_indices))

    def test_each_valence_has_six_signals(self) -> None:
        names = arbitration_evidence_signal_names()
        for valence in VALENCE_ORDER:
            count = sum(1 for n in names if n.startswith(f"{valence}."))
            self.assertEqual(count, 6, f"Expected 6 signals for {valence}, got {count}")


# ---------------------------------------------------------------------------
# SpiderBrain._arbitration_evidence_vector tests
# ---------------------------------------------------------------------------


class ArbitrationEvidenceVectorTest(unittest.TestCase):
    """Tests for SpiderBrain._arbitration_evidence_vector."""

    def setUp(self) -> None:
        self.brain = SpiderBrain(seed=42, module_dropout=0.0)

    def _blank_evidence(self) -> dict:
        return {
            valence: {field: 0.0 for field in fields}
            for valence, fields in ARBITRATION_EVIDENCE_FIELDS.items()
        }

    def test_returns_array_of_length_24(self) -> None:
        evidence = self._blank_evidence()
        vec = arbitration_evidence_vector(evidence)
        self.assertEqual(vec.shape, (24,))

    def test_returns_float_array(self) -> None:
        evidence = self._blank_evidence()
        vec = arbitration_evidence_vector(evidence)
        self.assertEqual(vec.dtype, float)

    def test_all_zeros_for_blank_evidence(self) -> None:
        evidence = self._blank_evidence()
        vec = arbitration_evidence_vector(evidence)
        np.testing.assert_array_equal(vec, np.zeros(24))

    def test_first_element_is_threat_predator_visible(self) -> None:
        evidence = self._blank_evidence()
        evidence["threat"]["predator_visible"] = 0.75
        vec = arbitration_evidence_vector(evidence)
        self.assertAlmostEqual(float(vec[0]), 0.75)

    def test_seventh_element_is_hunger_hunger(self) -> None:
        # threat has 6 fields, so hunger starts at index 6
        evidence = self._blank_evidence()
        evidence["hunger"]["hunger"] = 0.55
        vec = arbitration_evidence_vector(evidence)
        self.assertAlmostEqual(float(vec[6]), 0.55)

    def test_last_element_is_exploration_food_smell_directionality(self) -> None:
        evidence = self._blank_evidence()
        evidence["exploration"]["food_smell_directionality"] = 0.33
        vec = arbitration_evidence_vector(evidence)
        self.assertAlmostEqual(float(vec[-1]), 0.33)

    def test_output_matches_signal_names_ordering(self) -> None:
        evidence = self._blank_evidence()
        # Set each signal to a unique value based on its index
        signal_names = arbitration_evidence_signal_names()
        for idx, name in enumerate(signal_names):
            valence, field = name.split(".", 1)
            evidence[valence][field] = float(idx + 1) * 0.01
        vec = arbitration_evidence_vector(evidence)
        for idx in range(24):
            self.assertAlmostEqual(float(vec[idx]), float(idx + 1) * 0.01, places=8)


# ---------------------------------------------------------------------------
# SpiderBrain._fixed_formula_valence_scores_from_evidence tests
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# SpiderBrain._warm_start_arbitration_network tests
# ---------------------------------------------------------------------------


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


class WarmStartArbitrationNetworkTest(unittest.TestCase):
    """Tests for SpiderBrain._warm_start_arbitration_network."""

    def setUp(self) -> None:
        self.brain = SpiderBrain(seed=42, module_dropout=0.0)

    def test_warm_start_clears_network_cache(self) -> None:
        # Force a forward pass to set cache
        evidence_vector = np.zeros(ArbitrationNetwork.INPUT_DIM)
        self.brain.arbitration_network.forward(evidence_vector, store_cache=True)
        self.assertIsNotNone(self.brain.arbitration_network.cache)
        self.brain._warm_start_arbitration_network()
        self.assertIsNone(self.brain.arbitration_network.cache)

    def test_warm_start_sets_w1_diagonal(self) -> None:
        self.brain._warm_start_arbitration_network()
        net = self.brain.arbitration_network
        # W1 should have non-zero values on the diagonal (first input_dim rows)
        for i in range(net.input_dim):
            self.assertNotEqual(net.W1[i, i], 0.0,
                                f"W1[{i},{i}] should be non-zero after warm start")

    def test_warm_start_zeros_b1(self) -> None:
        self.brain._warm_start_arbitration_network()
        np.testing.assert_array_equal(self.brain.arbitration_network.b1, 0.0)

    def test_warm_start_zeros_b2_valence(self) -> None:
        self.brain._warm_start_arbitration_network()
        np.testing.assert_array_equal(self.brain.arbitration_network.b2_valence, 0.0)

    def test_warm_start_zeros_gate_head(self) -> None:
        self.brain._warm_start_arbitration_network()
        np.testing.assert_array_equal(self.brain.arbitration_network.W2_gate, 0.0)
        np.testing.assert_array_equal(self.brain.arbitration_network.b2_gate, 0.0)

    def test_warm_start_zeros_value_head(self) -> None:
        self.brain._warm_start_arbitration_network()
        np.testing.assert_array_equal(self.brain.arbitration_network.W2_value, 0.0)
        np.testing.assert_array_equal(self.brain.arbitration_network.b2_value, 0.0)

    def test_warm_started_network_produces_finite_outputs(self) -> None:
        self.brain._warm_start_arbitration_network()
        evidence_vector = np.linspace(0.0, 1.0, ArbitrationNetwork.INPUT_DIM)
        valence_logits, gate_adjustments, value = self.brain.arbitration_network.forward(
            evidence_vector, store_cache=False
        )
        self.assertTrue(np.all(np.isfinite(valence_logits)))
        self.assertTrue(np.all(np.isfinite(gate_adjustments)))
        self.assertTrue(np.isfinite(value))

    def test_warm_start_valence_head_produces_nonzero_threat_logit_for_threat_input(self) -> None:
        self.brain._warm_start_arbitration_network()
        # Create a vector with only threat signals active
        evidence_vector = np.zeros(ArbitrationNetwork.INPUT_DIM)
        # First 6 slots are threat signals; set predator_visible (slot 0) to 1.0
        evidence_vector[0] = 1.0
        valence_logits, _, _ = self.brain.arbitration_network.forward(
            evidence_vector, store_cache=False
        )
        # Threat logit (index 0) should be dominant
        threat_idx = VALENCE_ORDER.index("threat")
        self.assertGreater(float(valence_logits[threat_idx]), float(valence_logits[1]))

    def test_warm_start_scale_reduces_forward_valence_logits(self) -> None:
        full = SpiderBrain(seed=42, module_dropout=0.0)
        scaled = SpiderBrain(
            seed=42,
            config=BrainAblationConfig(
                name="scaled_warm_start",
                module_dropout=0.0,
                warm_start_scale=0.5,
            ),
        )
        evidence_vector = _small_arbitration_evidence_vector()

        full_logits, _, _ = full.arbitration_network.forward(
            evidence_vector, store_cache=False
        )
        scaled_logits, _, _ = scaled.arbitration_network.forward(
            evidence_vector, store_cache=False
        )
        _assert_relative_scaled_logits(
            self,
            scaled_logits,
            full_logits,
            0.5,
        )

    def test_warm_start_scale_zero_skips_warm_start(self) -> None:
        no_start = SpiderBrain(
            seed=42,
            config=BrainAblationConfig(
                name="no_warm_start",
                module_dropout=0.0,
                warm_start_scale=0.0,
            ),
        )
        warmed = SpiderBrain(seed=42, module_dropout=0.0)

        self.assertNotEqual(no_start.arbitration_network.W1[0, 0], 0.0)
        self.assertNotAlmostEqual(
            no_start.arbitration_network.W1[0, 0],
            warmed.arbitration_network.W1[0, 0],
        )

    def test_minimal_arbitration_random_weights_follow_brain_seed(self) -> None:
        config = BrainAblationConfig(
            name="minimal_seed_test",
            module_dropout=0.0,
            warm_start_scale=0.0,
        )
        first = SpiderBrain(seed=42, config=config)
        same_seed = SpiderBrain(seed=42, config=config)
        different_seed = SpiderBrain(seed=43, config=config)

        np.testing.assert_allclose(
            first.arbitration_network.W1,
            same_seed.arbitration_network.W1,
        )
        np.testing.assert_allclose(
            first.arbitration_network.W2_gate,
            same_seed.arbitration_network.W2_gate,
        )
        self.assertFalse(
            np.allclose(
                first.arbitration_network.W1,
                different_seed.arbitration_network.W1,
            )
        )
        self.assertFalse(
            np.allclose(
                first.arbitration_network.W2_gate,
                different_seed.arbitration_network.W2_gate,
            )
        )

    def test_arbitration_network_uses_configured_gate_adjustment_bounds(self) -> None:
        brain = SpiderBrain(
            seed=42,
            config=BrainAblationConfig(
                name="wide_gate_bounds",
                module_dropout=0.0,
                gate_adjustment_bounds=(0.25, 1.75),
            ),
        )

        self.assertAlmostEqual(brain.arbitration_network.gate_adjustment_min, 0.25)
        self.assertAlmostEqual(brain.arbitration_network.gate_adjustment_max, 1.75)


class CanonicalArbitrationVariantIntegrationTest(unittest.TestCase):
    """Integration tests for named arbitration-prior ablation variants."""

    EXPECTED: ClassVar[dict[str, dict[str, object]]] = {
        "constrained_arbitration": {
            "enable_deterministic_guards": True,
            "enable_food_direction_bias": True,
            "warm_start_scale": 1.0,
            "gate_adjustment_bounds": (0.5, 1.5),
        },
        "weaker_prior_arbitration": {
            "enable_deterministic_guards": False,
            "enable_food_direction_bias": False,
            "warm_start_scale": 0.5,
            "gate_adjustment_bounds": (0.5, 1.5),
        },
        "minimal_arbitration": {
            "enable_deterministic_guards": False,
            "enable_food_direction_bias": False,
            "warm_start_scale": 0.0,
            "gate_adjustment_bounds": (0.1, 2.0),
        },
    }

    def _configs(self) -> dict[str, BrainAblationConfig]:
        """
        Return the canonical set of brain ablation configurations with module dropout disabled.

        Returns:
            configs (dict[str, BrainAblationConfig]): Mapping from canonical config names to their
            corresponding BrainAblationConfig instances (with module_dropout == 0.0).
        """
        return canonical_ablation_configs(module_dropout=0.0)

    def test_named_arbitration_configs_have_expected_fields(self) -> None:
        configs = self._configs()
        for name, expected in self.EXPECTED.items():
            with self.subTest(name=name):
                config = configs[name]
                self.assertTrue(config.use_learned_arbitration)
                self.assertEqual(config.module_dropout, 0.0)
                self.assertEqual(
                    config.enable_deterministic_guards,
                    expected["enable_deterministic_guards"],
                )
                self.assertEqual(
                    config.enable_food_direction_bias,
                    expected["enable_food_direction_bias"],
                )
                self.assertAlmostEqual(
                    config.warm_start_scale,
                    expected["warm_start_scale"],
                )
                self.assertEqual(
                    config.gate_adjustment_bounds,
                    expected["gate_adjustment_bounds"],
                )

    def test_named_arbitration_configs_pass_gate_bounds_to_network(self) -> None:
        """
        Verify that each canonical arbitration ablation configuration sets the arbitration network's gate adjustment bounds.

        For every named config, instantiate a SpiderBrain and assert that its arbitration_network.gate_adjustment_min and .gate_adjustment_max match the expected lower and upper bounds from the config.
        """
        configs = self._configs()
        for name, expected in self.EXPECTED.items():
            with self.subTest(name=name):
                brain = SpiderBrain(seed=42, config=configs[name])
                lower, upper = expected["gate_adjustment_bounds"]
                self.assertAlmostEqual(
                    brain.arbitration_network.gate_adjustment_min,
                    lower,
                )
                self.assertAlmostEqual(
                    brain.arbitration_network.gate_adjustment_max,
                    upper,
                )

    def test_named_arbitration_warm_start_scales_forward_valence_logits(self) -> None:
        configs = self._configs()
        constrained = SpiderBrain(
            seed=42,
            config=configs["constrained_arbitration"],
        )
        weaker = SpiderBrain(
            seed=42,
            config=configs["weaker_prior_arbitration"],
        )
        minimal = SpiderBrain(
            seed=42,
            config=configs["minimal_arbitration"],
        )
        evidence_vector = _small_arbitration_evidence_vector()

        constrained_logits, _, _ = constrained.arbitration_network.forward(
            evidence_vector, store_cache=False
        )
        weaker_logits, _, _ = weaker.arbitration_network.forward(
            evidence_vector, store_cache=False
        )
        _assert_relative_scaled_logits(
            self,
            weaker_logits,
            constrained_logits,
            0.5,
        )
        self.assertAlmostEqual(
            constrained.arbitration_network.W2_gate[0, 0],
            0.0,
        )
        self.assertAlmostEqual(
            weaker.arbitration_network.W2_gate[0, 0],
            0.0,
        )
        self.assertGreater(np.linalg.norm(minimal.arbitration_network.W2_gate), 0.0)
        self.assertNotAlmostEqual(
            minimal.arbitration_network.W1[0, 0],
            constrained.arbitration_network.W1[0, 0],
        )


# ---------------------------------------------------------------------------
# SpiderBrain arbitration_lr default tests
# ---------------------------------------------------------------------------


class ComputeArbitrationTrainingModeTest(unittest.TestCase):
    """Tests for the training parameter of SpiderBrain._compute_arbitration."""

    def setUp(self) -> None:
        self.brain = SpiderBrain(seed=42, module_dropout=0.0)
        self.obs = _blank_obs()

    def test_training_false_clears_arbitration_cache_for_fixed_config(self) -> None:
        config = BrainAblationConfig(name="fixed_arbitration_baseline", use_learned_arbitration=False)
        brain = SpiderBrain(seed=42, module_dropout=0.0, config=config)
        obs = _blank_obs()
        module_results = brain._proposal_results(obs, store_cache=False, training=False)
        brain._compute_arbitration(module_results, obs, training=False, store_cache=False)
        self.assertIsNone(brain.arbitration_network.cache)

    def test_learned_arbitration_training_true_stores_cache(self) -> None:
        module_results = self.brain._proposal_results(self.obs, store_cache=False, training=False)
        self.brain._compute_arbitration(module_results, self.obs, training=True, store_cache=True)
        self.assertIsNotNone(self.brain.arbitration_network.cache)

    def test_learned_arbitration_training_false_returns_deterministic_result(self) -> None:
        """Calling _compute_arbitration with training=False gives consistent winning_valence."""
        module_results = self.brain._proposal_results(self.obs, store_cache=False, training=False)
        arb1 = self.brain._compute_arbitration(module_results, self.obs, training=False, store_cache=False)
        arb2 = self.brain._compute_arbitration(module_results, self.obs, training=False, store_cache=False)
        self.assertEqual(arb1.winning_valence, arb2.winning_valence)

    def test_learned_adjustment_is_true_for_learned_brain(self) -> None:
        module_results = self.brain._proposal_results(self.obs, store_cache=False, training=False)
        arb = self.brain._compute_arbitration(module_results, self.obs, training=False, store_cache=False)
        self.assertTrue(arb.learned_adjustment)

    def test_learned_adjustment_is_false_for_fixed_brain(self) -> None:
        config = BrainAblationConfig(name="fixed_arbitration_baseline", use_learned_arbitration=False)
        brain = SpiderBrain(seed=42, module_dropout=0.0, config=config)
        obs = _blank_obs()
        module_results = brain._proposal_results(obs, store_cache=False, training=False)
        arb = brain._compute_arbitration(module_results, obs, training=False, store_cache=False)
        self.assertFalse(arb.learned_adjustment)


class ArbitrationNetworkInArbitrationNetworkTest(unittest.TestCase):
    """Additional regression tests for ArbitrationNetwork integration in SpiderBrain."""

    def _build_brain(self, seed: int = 5) -> SpiderBrain:
        return SpiderBrain(seed=seed, module_dropout=0.0)

    def _blank_observation(self) -> dict[str, np.ndarray]:
        return _blank_obs()

    def test_arbitration_network_name_constant(self) -> None:
        self.assertEqual(SpiderBrain.ARBITRATION_NETWORK_NAME, "arbitration_network")

    def test_arbitration_gate_module_order_has_six_entries(self) -> None:
        self.assertEqual(len(SpiderBrain.ARBITRATION_GATE_MODULE_ORDER), 6)

    def test_arbitration_gate_module_order_contains_expected_modules(self) -> None:
        expected = {
            "alert_center", "hunger_center", "sleep_center",
            "visual_cortex", "sensory_cortex", SpiderBrain.MONOLITHIC_POLICY_NAME,
        }
        self.assertEqual(set(SpiderBrain.ARBITRATION_GATE_MODULE_ORDER), expected)

    def test_arbitration_evidence_fields_has_four_valences(self) -> None:
        self.assertEqual(
            set(ARBITRATION_EVIDENCE_FIELDS.keys()),
            set(VALENCE_ORDER),
        )

    def test_each_arbitration_evidence_valence_has_six_fields(self) -> None:
        for valence, fields in ARBITRATION_EVIDENCE_FIELDS.items():
            self.assertEqual(len(fields), 6, f"Expected 6 fields for {valence}")

    def test_arbitration_network_rng_is_independent(self) -> None:
        """Arbitration RNG should be distinct from main brain RNG (seeded offset)."""
        brain = self._build_brain(seed=42)
        self.assertIsNotNone(brain.arbitration_rng)
        self.assertIsNot(brain.arbitration_rng, brain.rng)


class ProposalContributionShareTest(unittest.TestCase):
    """Unit tests for proposal_contribution_share extracted from agent.py in this PR."""

    def _make_result(self, name: str, *, active: bool = True) -> ModuleResult:
        """Create a minimal ModuleResult with the given name and active state."""
        interface = MODULE_INTERFACES[0]  # any interface works for testing
        action_dim = len(LOCOMOTION_ACTIONS)
        return ModuleResult(
            interface=interface,
            name=name,
            observation_key=interface.observation_key,
            observation=np.zeros(interface.input_dim, dtype=float),
            logits=np.zeros(action_dim, dtype=float),
            probs=np.full(action_dim, 1.0 / action_dim, dtype=float),
            active=active,
        )

    def test_two_active_modules_proportional_to_l1_magnitude(self) -> None:
        """Shares are proportional to L1 norm of gated logits for active modules."""
        r1 = self._make_result("module_a")
        r2 = self._make_result("module_b")
        # r1 has total L1=1.0, r2 has total L1=3.0 → shares 0.25 and 0.75
        g1 = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        g2 = np.array([3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        shares = proposal_contribution_share([r1, r2], [g1, g2])
        self.assertAlmostEqual(shares["module_a"], 0.25, places=8)
        self.assertAlmostEqual(shares["module_b"], 0.75, places=8)

    def test_shares_sum_to_one_for_active_modules(self) -> None:
        """All active module shares must sum to 1.0."""
        results = [self._make_result(f"m{i}") for i in range(4)]
        gated = [np.array([float(i + 1)] + [0.0] * 8) for i in range(4)]
        shares = proposal_contribution_share(results, gated)
        total = sum(shares.values())
        self.assertAlmostEqual(total, 1.0, places=8)

    def test_no_active_modules_returns_all_zeros(self) -> None:
        """When no modules are active, all contribution shares are 0.0."""
        r1 = self._make_result("mod_a", active=False)
        r2 = self._make_result("mod_b", active=False)
        g1 = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        g2 = np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        shares = proposal_contribution_share([r1, r2], [g1, g2])
        self.assertEqual(shares["mod_a"], 0.0)
        self.assertEqual(shares["mod_b"], 0.0)

    def test_inactive_modules_still_validate_gated_logit_count(self) -> None:
        """Length mismatches should be reported before the inactive early return."""
        r1 = self._make_result("mod_a", active=False)
        r2 = self._make_result("mod_b", active=False)
        g1 = np.zeros(9, dtype=float)
        with self.assertRaises(ValueError):
            proposal_contribution_share([r1, r2], [g1])

    def test_all_zero_gated_logits_returns_uniform_shares(self) -> None:
        """When all active modules have zero logits, shares are uniform."""
        r1 = self._make_result("mod_x")
        r2 = self._make_result("mod_y")
        g1 = np.zeros(9, dtype=float)
        g2 = np.zeros(9, dtype=float)
        shares = proposal_contribution_share([r1, r2], [g1, g2])
        self.assertAlmostEqual(shares["mod_x"], 0.5, places=8)
        self.assertAlmostEqual(shares["mod_y"], 0.5, places=8)

    def test_single_active_module_gets_full_share(self) -> None:
        """A single active module should receive contribution share of 1.0."""
        r1 = self._make_result("solo")
        g1 = np.array([2.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        shares = proposal_contribution_share([r1], [g1])
        self.assertAlmostEqual(shares["solo"], 1.0, places=8)

    def test_inactive_module_gets_zero_share_even_with_nonzero_logits(self) -> None:
        """Inactive modules receive 0.0 regardless of their gated logit magnitude."""
        active = self._make_result("active_mod")
        inactive = self._make_result("inactive_mod", active=False)
        g_active = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        g_inactive = np.array([9.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        shares = proposal_contribution_share([active, inactive], [g_active, g_inactive])
        self.assertAlmostEqual(shares["active_mod"], 1.0, places=8)
        self.assertEqual(shares["inactive_mod"], 0.0)

    def test_mixed_active_inactive_shares_sum_to_one(self) -> None:
        """With mixed active/inactive modules, shares across all entries still sum to 1.0."""
        active1 = self._make_result("act1")
        active2 = self._make_result("act2")
        inactive = self._make_result("inact", active=False)
        g1 = np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        g2 = np.array([3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        g_inact = np.zeros(9, dtype=float)
        shares = proposal_contribution_share([active1, active2, inactive], [g1, g2, g_inact])
        total = sum(shares.values())
        self.assertAlmostEqual(total, 1.0, places=8)
        self.assertEqual(shares["inact"], 0.0)

    def test_result_keys_match_module_names(self) -> None:
        """Return dict keys match the names of all module results, active or not."""
        results = [self._make_result(f"m{i}", active=(i % 2 == 0)) for i in range(4)]
        gated = [np.ones(9, dtype=float) * float(i) for i in range(4)]
        shares = proposal_contribution_share(results, gated)
        self.assertEqual(set(shares.keys()), {f"m{i}" for i in range(4)})

    def test_single_active_module_with_zero_logits_gets_uniform_share(self) -> None:
        """Uniform fallback: a single active module with zero logits gets share 1.0."""
        r1 = self._make_result("only_active")
        g1 = np.zeros(9, dtype=float)
        shares = proposal_contribution_share([r1], [g1])
        self.assertAlmostEqual(shares["only_active"], 1.0, places=8)

    def test_negative_logits_use_absolute_value_for_magnitude(self) -> None:
        """L1 magnitude uses absolute values, so negative logits contribute positively."""
        r1 = self._make_result("neg")
        r2 = self._make_result("pos")
        # r1 has logits [-2.0, ...], r2 has logits [2.0, ...]; same |magnitude|
        g1 = np.array([-2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        g2 = np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        shares = proposal_contribution_share([r1, r2], [g1, g2])
        self.assertAlmostEqual(shares["neg"], 0.5, places=8)
        self.assertAlmostEqual(shares["pos"], 0.5, places=8)

    def test_empty_module_list_returns_empty_dict(self) -> None:
        """An empty module list should return an empty dict without error."""
        shares = proposal_contribution_share([], [])
        self.assertEqual(shares, {})

    def test_boundary_total_just_above_threshold_uses_proportional_shares(self) -> None:
        """When total L1 is just above 1e-8, proportional shares are used (not uniform)."""
        r1 = self._make_result("small_a")
        r2 = self._make_result("small_b")
        # Give r1 a tiny nonzero logit that keeps total above threshold
        g1 = np.zeros(9, dtype=float)
        g2 = np.zeros(9, dtype=float)
        # Put all weight at index 0
        g1[0] = 1e-7  # just above 1e-8 per dimension; total L1 = 1e-7 > 1e-8
        shares = proposal_contribution_share([r1, r2], [g1, g2])
        # r1 has all the magnitude, r2 has none
        self.assertAlmostEqual(shares["small_a"], 1.0, places=6)
        self.assertAlmostEqual(shares["small_b"], 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
