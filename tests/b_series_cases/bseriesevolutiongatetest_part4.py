from __future__ import annotations

from .shared import *
from .bseriesevolutiongatetest_helpers import BSeriesEvolutionGateTestHelpers



class BSeriesEvolutionGateTestPart4(BSeriesEvolutionGateTestHelpers, unittest.TestCase):
    def test_b53_corridor_gate_rejects_b52_clone_without_arousal(self) -> None:
        results = [
            self._b53_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
                decision=None,
                norepinephrine_level=None,
                arousal_gain=None,
                surprise_signal=None,
                gain_lock=None,
            )
            for episode in range(3)
        ]

        gate = b53_noradrenergic_arousal_corridor_gate_result(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "corridor_b53_aggregate:explicit_b53_decision_episodes",
            gate["failures"],
        )

    def test_b53_corridor_gate_keeps_b52_base_as_diagnostic(self) -> None:
        results = [
            self._b53_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b53_noradrenergic_arousal_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["corridor_safety_episodes"], 3)
        self.assertIn("base_b52_corridor_diagnostic", gate["aggregate"])

    def test_b54_corridor_gate_accepts_serotonergic_patience_progress(self) -> None:
        results = [
            self._b54_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b54_serotonergic_patience_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["explicit_decision_episodes"], 3)
        self.assertEqual(gate["aggregate"]["serotonin_level_episodes"], 3)
        self.assertEqual(gate["aggregate"]["patience_signal_episodes"], 3)
        self.assertEqual(gate["aggregate"]["impulse_suppression_episodes"], 3)
        self.assertEqual(gate["aggregate"]["patience_lock_episodes"], 3)

    def test_b54_corridor_gate_rejects_b53_clone_without_patience(self) -> None:
        results = [
            self._b54_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
                decision=None,
                serotonin_level=None,
                patience_signal=None,
                impulse_suppression=None,
                patience_lock=None,
            )
            for episode in range(3)
        ]

        gate = b54_serotonergic_patience_corridor_gate_result(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "corridor_b54_aggregate:explicit_b54_decision_episodes",
            gate["failures"],
        )

    def test_b54_corridor_gate_keeps_b53_base_as_diagnostic(self) -> None:
        results = [
            self._b54_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b54_serotonergic_patience_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["corridor_safety_episodes"], 3)
        self.assertIn("base_b53_corridor_diagnostic", gate["aggregate"])

    def test_b55_corridor_gate_accepts_hypothalamic_drive_progress(self) -> None:
        results = [
            self._b55_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b55_hypothalamic_drive_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["explicit_decision_episodes"], 3)
        self.assertEqual(gate["aggregate"]["hypothalamic_drive_episodes"], 3)
        self.assertEqual(gate["aggregate"]["satiety_signal_episodes"], 3)
        self.assertEqual(gate["aggregate"]["recovery_bias_episodes"], 3)
        self.assertEqual(gate["aggregate"]["drive_balance_episodes"], 3)
        self.assertEqual(gate["aggregate"]["drive_lock_episodes"], 3)

    def test_b55_corridor_gate_rejects_b54_clone_without_drive(self) -> None:
        results = [
            self._b55_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
                decision=None,
                hypothalamic_drive=None,
                satiety_signal=None,
                recovery_bias=None,
                drive_balance=None,
                drive_lock=None,
            )
            for episode in range(3)
        ]

        gate = b55_hypothalamic_drive_corridor_gate_result(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "corridor_b55_aggregate:explicit_b55_decision_episodes",
            gate["failures"],
        )

    def test_b55_corridor_gate_keeps_b54_base_as_diagnostic(self) -> None:
        results = [
            self._b55_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b55_hypothalamic_drive_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["corridor_safety_episodes"], 3)
        self.assertIn("base_b54_corridor_diagnostic", gate["aggregate"])

    def test_b56_corridor_gate_accepts_hpa_stress_progress(self) -> None:
        results = [
            self._b56_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b56_hpa_stress_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["explicit_decision_episodes"], 3)
        self.assertEqual(gate["aggregate"]["cortisol_level_episodes"], 3)
        self.assertEqual(gate["aggregate"]["stress_load_episodes"], 3)
        self.assertEqual(gate["aggregate"]["recovery_signal_episodes"], 3)
        self.assertEqual(gate["aggregate"]["endocrine_balance_episodes"], 3)
        self.assertEqual(gate["aggregate"]["stress_lock_episodes"], 3)

    def test_b56_corridor_gate_rejects_b55_clone_without_hpa_stress(self) -> None:
        results = [
            self._b56_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
                decision=None,
                cortisol_level=None,
                stress_load=None,
                recovery_signal=None,
                endocrine_balance=None,
                stress_lock=None,
            )
            for episode in range(3)
        ]

        gate = b56_hpa_stress_corridor_gate_result(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "corridor_b56_aggregate:explicit_b56_decision_episodes",
            gate["failures"],
        )

    def test_b56_corridor_gate_keeps_b55_base_as_diagnostic(self) -> None:
        results = [
            self._b56_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b56_hpa_stress_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["corridor_safety_episodes"], 3)
        self.assertIn("base_b55_corridor_diagnostic", gate["aggregate"])

    def test_b57_corridor_gate_accepts_interoceptive_progress(self) -> None:
        results = [
            self._b57_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b57_insular_interoceptive_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["explicit_decision_episodes"], 3)
        self.assertEqual(gate["aggregate"]["interoceptive_awareness_episodes"], 3)
        self.assertEqual(gate["aggregate"]["visceral_salience_episodes"], 3)
        self.assertEqual(gate["aggregate"]["body_state_confidence_episodes"], 3)
        self.assertEqual(gate["aggregate"]["awareness_balance_episodes"], 3)
        self.assertEqual(gate["aggregate"]["awareness_lock_episodes"], 3)

    def test_b57_corridor_gate_rejects_b56_clone_without_interoception(self) -> None:
        results = [
            self._b57_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
                decision=None,
                interoceptive_awareness=None,
                visceral_salience=None,
                body_state_confidence=None,
                awareness_balance=None,
                awareness_lock=None,
            )
            for episode in range(3)
        ]

        gate = b57_insular_interoceptive_corridor_gate_result(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "corridor_b57_aggregate:explicit_b57_decision_episodes",
            gate["failures"],
        )

    def test_b57_corridor_gate_keeps_b56_base_as_diagnostic(self) -> None:
        results = [
            self._b57_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b57_insular_interoceptive_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["corridor_safety_episodes"], 3)
        self.assertIn("base_b56_corridor_diagnostic", gate["aggregate"])

    def test_b58_corridor_gate_accepts_acc_conflict_progress(self) -> None:
        results = [
            self._b58_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b58_acc_conflict_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["explicit_decision_episodes"], 3)
        self.assertEqual(gate["aggregate"]["conflict_signal_episodes"], 3)
        self.assertEqual(gate["aggregate"]["error_likelihood_episodes"], 3)
        self.assertEqual(gate["aggregate"]["control_allocation_episodes"], 3)
        self.assertEqual(gate["aggregate"]["resolution_balance_episodes"], 3)
        self.assertEqual(gate["aggregate"]["conflict_lock_episodes"], 3)

    def test_b58_corridor_gate_rejects_b57_clone_without_conflict(self) -> None:
        results = [
            self._b58_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
                decision=None,
                conflict_signal=None,
                error_likelihood=None,
                control_allocation=None,
                resolution_balance=None,
                conflict_lock=None,
            )
            for episode in range(3)
        ]

        gate = b58_acc_conflict_corridor_gate_result(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "corridor_b58_aggregate:explicit_b58_decision_episodes",
            gate["failures"],
        )

    def test_b58_corridor_gate_keeps_b57_base_as_diagnostic(self) -> None:
        results = [
            self._b58_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b58_acc_conflict_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["corridor_safety_episodes"], 3)
        self.assertIn("base_b57_corridor_diagnostic", gate["aggregate"])

    def test_b59_corridor_gate_accepts_prefrontal_goal_progress(self) -> None:
        results = [
            self._b59_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b59_prefrontal_goal_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["explicit_decision_episodes"], 3)
        self.assertEqual(gate["aggregate"]["goal_context_episodes"], 3)
        self.assertEqual(gate["aggregate"]["working_set_stability_episodes"], 3)
        self.assertEqual(gate["aggregate"]["task_set_confidence_episodes"], 3)
        self.assertEqual(gate["aggregate"]["executive_balance_episodes"], 3)
        self.assertEqual(gate["aggregate"]["executive_lock_episodes"], 3)

    def test_b59_corridor_gate_rejects_b58_clone_without_prefrontal_context(self) -> None:
        results = [
            self._b59_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
                decision=None,
                goal_context=None,
                working_set_stability=None,
                task_set_confidence=None,
                executive_balance=None,
                executive_lock=None,
            )
            for episode in range(3)
        ]

        gate = b59_prefrontal_goal_corridor_gate_result(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "corridor_b59_aggregate:explicit_b59_decision_episodes",
            gate["failures"],
        )

    def test_b59_corridor_gate_keeps_b58_base_as_diagnostic(self) -> None:
        results = [
            self._b59_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b59_prefrontal_goal_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["corridor_safety_episodes"], 3)
        self.assertIn("base_b58_corridor_diagnostic", gate["aggregate"])

    def test_b60_corridor_gate_accepts_orbitofrontal_value_progress(self) -> None:
        results = [
            self._b60_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b60_orbitofrontal_value_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["explicit_decision_episodes"], 3)
        self.assertEqual(gate["aggregate"]["outcome_value_episodes"], 3)
        self.assertEqual(gate["aggregate"]["reversal_signal_episodes"], 3)
        self.assertEqual(gate["aggregate"]["goal_value_confidence_episodes"], 3)
        self.assertEqual(gate["aggregate"]["value_balance_episodes"], 3)
        self.assertEqual(gate["aggregate"]["value_lock_episodes"], 3)

    def test_b60_corridor_gate_rejects_b59_clone_without_value(self) -> None:
        results = [
            self._b60_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
                decision=None,
                outcome_value=None,
                reversal_signal=None,
                goal_value_confidence=None,
                value_balance=None,
                value_lock=None,
            )
            for episode in range(3)
        ]

        gate = b60_orbitofrontal_value_corridor_gate_result(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "corridor_b60_aggregate:explicit_b60_decision_episodes",
            gate["failures"],
        )

    def test_b60_corridor_gate_keeps_b59_base_as_diagnostic(self) -> None:
        results = [
            self._b60_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b60_orbitofrontal_value_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["corridor_safety_episodes"], 3)
        self.assertIn("base_b59_corridor_diagnostic", gate["aggregate"])

    def test_b61_corridor_gate_accepts_amygdala_safety_progress(self) -> None:
        results = [
            self._b61_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b61_amygdala_safety_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["explicit_decision_episodes"], 3)
        self.assertEqual(gate["aggregate"]["safety_value_episodes"], 3)
        self.assertEqual(gate["aggregate"]["threat_channel_episodes"], 3)
        self.assertEqual(gate["aggregate"]["safety_confidence_episodes"], 3)
        self.assertEqual(gate["aggregate"]["affective_balance_episodes"], 3)
        self.assertEqual(gate["aggregate"]["safety_lock_episodes"], 3)

    def test_b61_corridor_gate_rejects_b60_clone_without_safety(self) -> None:
        results = [
            self._b61_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
                decision=None,
                safety_value=None,
                threat_value=None,
                safety_confidence=None,
                affective_balance=None,
                safety_lock=None,
            )
            for episode in range(3)
        ]

        gate = b61_amygdala_safety_corridor_gate_result(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "corridor_b61_aggregate:explicit_b61_decision_episodes",
            gate["failures"],
        )

    def test_b61_corridor_gate_keeps_b60_base_as_diagnostic(self) -> None:
        results = [
            self._b61_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b61_amygdala_safety_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["corridor_safety_episodes"], 3)
        self.assertIn("base_b60_corridor_diagnostic", gate["aggregate"])

    def test_b62_corridor_gate_accepts_defensive_mode_progress(self) -> None:
        results = [
            self._b62_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b62_defensive_mode_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["explicit_decision_episodes"], 3)
        self.assertEqual(gate["aggregate"]["defensive_mode_episodes"], 3)
        self.assertEqual(gate["aggregate"]["defense_pressure_episodes"], 3)
        self.assertEqual(gate["aggregate"]["shelter_bias_episodes"], 3)
        self.assertEqual(gate["aggregate"]["defense_balance_episodes"], 3)
        self.assertEqual(gate["aggregate"]["lock_or_safe_episodes"], 3)

    def test_b62_corridor_gate_rejects_b61_clone_without_defense(self) -> None:
        results = [
            self._b62_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
                decision=None,
                defensive_mode=None,
                freeze_pressure=None,
                flee_pressure=None,
                shelter_bias=None,
                defense_balance=None,
                defense_lock=None,
            )
            for episode in range(3)
        ]

        gate = b62_defensive_mode_corridor_gate_result(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "corridor_b62_aggregate:explicit_b62_decision_episodes",
            gate["failures"],
        )

    def test_b62_corridor_gate_keeps_b61_base_as_diagnostic(self) -> None:
        results = [
            self._b62_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b62_defensive_mode_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["corridor_safety_episodes"], 3)
        self.assertIn("base_b61_corridor_diagnostic", gate["aggregate"])

    def test_b6_promotion_prefers_passing_fusion_then_best_individual(self) -> None:
        risk = {
            "variant": B6_RISK_FORAGE_ARBITER_H48_POLICY_NAME,
            "controller_family": "risk_corridor",
            "status": "accepted",
            "canonical_gate": {"aggregate": {"completed_horizons": 4, "min_steps": 50, "total_predator_contacts": 20}},
            "food_predator_gate": {"aggregate": {"threat_exposure_episodes": 2, "threat_priority_or_suppression_episodes": 2}},
            "corridor_gate": {"aggregate": {"progress_episodes": 2, "survival_progress_episodes": 2}},
            "easy_gate": {"passed": True},
        }
        recurrent = {
            **risk,
            "variant": B6_RECURRENT_CONTEXT_H48_POLICY_NAME,
            "controller_family": "recurrent_memory",
            "canonical_gate": {"aggregate": {"completed_horizons": 5, "min_steps": 60, "total_predator_contacts": 18}},
        }
        fusion = {
            **risk,
            "variant": B6_FUSED_RISK_RECURRENT_H48_POLICY_NAME,
            "controller_family": "fused_risk_recurrent",
            "status": "discarded",
        }

        selected = select_b6_promotion([risk], [recurrent], fusion)
        self.assertEqual(selected["variant"], B6_RECURRENT_CONTEXT_H48_POLICY_NAME)

        fusion["status"] = "accepted"
        selected = select_b6_promotion([risk], [recurrent], fusion)
        self.assertEqual(selected["variant"], B6_FUSED_RISK_RECURRENT_H48_POLICY_NAME)
