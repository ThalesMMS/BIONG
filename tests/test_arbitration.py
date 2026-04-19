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
