from __future__ import annotations

from .shared import *
from .bseriesevolutiongatetest_helpers import BSeriesEvolutionGateTestHelpers



class BSeriesEvolutionGateTestPart2(BSeriesEvolutionGateTestHelpers, unittest.TestCase):
    def test_b5_food_deprivation_gate_accepts_raw_probe_progress(self) -> None:
        results = [
            self._b5_probe_result(
                0,
                scenario="food_deprivation",
                steps=22,
                alive=True,
                food=1,
                final_hunger=0.88,
                food_distance_delta=0.0,
            ),
            self._b5_probe_result(
                1,
                scenario="food_deprivation",
                steps=22,
                alive=True,
                food=2,
                final_hunger=0.56,
                food_distance_delta=0.0,
            ),
            self._b5_probe_result(
                2,
                scenario="food_deprivation",
                steps=22,
                alive=True,
                food=1,
                final_hunger=0.89,
                food_distance_delta=3.0,
            ),
        ]

        gate = b5_food_deprivation_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["progress_episodes"], 2)

    def test_b5_sleep_conflict_gate_rejects_missing_recovery_movement(self) -> None:
        results = [
            self._b5_probe_result(
                episode,
                scenario="sleep_vs_exploration_conflict",
                steps=18,
                alive=True,
                sleep=8,
                final_sleep_debt=0.04,
            )
            for episode in range(3)
        ]

        gate = b5_sleep_conflict_gate_result(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "sleep_conflict_aggregate:post_recovery_movement_episodes",
            gate["failures"],
        )

    def test_b6_food_predator_gate_accepts_partial_threat_progress(self) -> None:
        payload = {
            "winning_valence": "threat",
            "evidence": {
                "threat": {
                    "predator_visible": 1.0,
                    "predator_proximity": 0.5,
                    "predator_certainty": 0.5,
                }
            },
            "module_gates": {"hunger_center": 0.2},
        }
        results = [
            self._b6_probe_result(
                episode,
                scenario="food_vs_predator_conflict",
                steps=16,
                alive=True,
                action_selection_payload=payload if episode < 2 else None,
            )
            for episode in range(3)
        ]

        gate = b6_food_predator_conflict_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["threat_exposure_episodes"], 2)
        self.assertEqual(
            gate["aggregate"]["threat_priority_or_suppression_episodes"],
            2,
        )

    def test_b6_food_predator_gate_rejects_absent_threat_progress(self) -> None:
        results = [
            self._b6_probe_result(
                episode,
                scenario="food_vs_predator_conflict",
                steps=16,
                alive=True,
            )
            for episode in range(3)
        ]

        gate = b6_food_predator_conflict_gate_result(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "food_predator_aggregate:threat_exposure_episodes",
            gate["failures"],
        )

    def test_b6_corridor_gate_accepts_partial_survival_progress(self) -> None:
        results = [
            self._b6_probe_result(
                0,
                scenario="corridor_gauntlet",
                steps=16,
                alive=False,
                food_distance_delta=12.0,
            ),
            self._b6_probe_result(
                1,
                scenario="corridor_gauntlet",
                steps=17,
                alive=False,
                food_distance_delta=10.0,
            ),
            self._b6_probe_result(
                2,
                scenario="corridor_gauntlet",
                steps=14,
                alive=False,
                food_distance_delta=0.0,
            ),
        ]

        gate = b6_corridor_progress_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["progress_episodes"], 2)
        self.assertEqual(gate["aggregate"]["survival_progress_episodes"], 2)

    def test_b6_corridor_gate_rejects_b5_baseline_steps(self) -> None:
        results = [
            self._b6_probe_result(
                episode,
                scenario="corridor_gauntlet",
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b6_corridor_progress_gate_result(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "corridor_aggregate:survival_progress_episodes",
            gate["failures"],
        )

    def test_b6_corridor_gate_accepts_food_progress_over_b5(self) -> None:
        results = [
            self._b6_probe_result(
                episode,
                scenario="corridor_gauntlet",
                steps=14,
                alive=False,
                food_distance_delta=14.0,
            )
            for episode in range(3)
        ]

        gate = b6_corridor_progress_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["survival_progress_episodes"], 3)
        checks = gate["episode_results"][0]["gate"]["checks"]
        self.assertTrue(checks["food_progress_over_b5"])

    def test_b7_corridor_gate_accepts_explicit_viability_progress(self) -> None:
        results = [
            self._b7_corridor_result(
                0,
                steps=14,
                alive=False,
                food_distance_delta=13.0,
                decision="abort_return_unviable",
            ),
            self._b7_corridor_result(
                1,
                steps=15,
                alive=False,
                food_distance_delta=13.0,
                decision="continue_viable",
            ),
            self._b7_corridor_result(
                2,
                steps=14,
                alive=False,
                food_distance_delta=0.0,
            ),
        ]

        gate = b7_corridor_viability_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["food_progress_episodes"], 2)
        self.assertEqual(gate["aggregate"]["explicit_decision_episodes"], 2)
        self.assertEqual(gate["aggregate"]["improvement_episodes"], 2)

    def test_b7_corridor_gate_rejects_clone_without_explicit_decision(self) -> None:
        results = [
            self._b7_corridor_result(
                episode,
                steps=15,
                alive=False,
                food_distance_delta=13.0,
            )
            for episode in range(3)
        ]

        gate = b7_corridor_viability_gate_result(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "corridor_aggregate:explicit_decision_episodes",
            gate["failures"],
        )

    def test_b7_corridor_gate_rejects_contacts_or_absent_improvement(self) -> None:
        contact_gate = b7_corridor_viability_gate_result(
            [
                self._b7_corridor_result(
                    0,
                    steps=15,
                    alive=False,
                    contacts=1,
                    food_distance_delta=13.0,
                    decision="continue_viable",
                ),
                self._b7_corridor_result(
                    1,
                    steps=15,
                    alive=False,
                    food_distance_delta=13.0,
                    decision="continue_viable",
                ),
                self._b7_corridor_result(
                    2,
                    steps=15,
                    alive=False,
                    food_distance_delta=13.0,
                    decision="continue_viable",
                ),
            ]
        )
        self.assertFalse(contact_gate["passed"])
        self.assertIn("corridor_ep0:predator_contacts", contact_gate["failures"])

        no_improvement_gate = b7_corridor_viability_gate_result(
            [
                self._b7_corridor_result(
                    episode,
                    steps=14,
                    alive=False,
                    food_distance_delta=13.0,
                    decision="continue_viable",
                )
                for episode in range(3)
            ]
        )
        self.assertFalse(no_improvement_gate["passed"])
        self.assertIn(
            "corridor_aggregate:improvement_episodes",
            no_improvement_gate["failures"],
        )

    def test_b8_corridor_gate_accepts_spatial_map_progress(self) -> None:
        results = [
            self._b8_corridor_result(
                episode,
                steps=15,
                alive=False,
                food_distance_delta=13.0,
            )
            for episode in range(3)
        ]

        gate = b8_spatial_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["explicit_decision_episodes"], 3)
        self.assertEqual(gate["aggregate"]["spatial_map_episodes"], 3)
        self.assertEqual(gate["aggregate"]["mapped_progress_episodes"], 3)

    def test_b8_corridor_gate_rejects_b7_clone_without_map(self) -> None:
        results = [
            self._b8_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=13.0,
                decision=None,
                map_state=None,
            )
            for episode in range(3)
        ]

        gate = b8_spatial_corridor_gate_result(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "corridor_b8_aggregate:explicit_b8_decision_episodes",
            gate["failures"],
        )

    def test_b9_corridor_gate_accepts_waypoint_route_progress(self) -> None:
        results = [
            self._b9_corridor_result(
                episode,
                steps=15,
                alive=False,
                food_distance_delta=13.0,
            )
            for episode in range(3)
        ]

        gate = b9_waypoint_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["explicit_decision_episodes"], 3)
        self.assertEqual(gate["aggregate"]["route_state_episodes"], 3)
        self.assertEqual(gate["aggregate"]["locked_waypoint_episodes"], 3)

    def test_b9_corridor_gate_rejects_b8_clone_without_waypoint(self) -> None:
        results = [
            self._b9_corridor_result(
                episode,
                steps=15,
                alive=False,
                food_distance_delta=13.0,
                decision=None,
                route_state=None,
                waypoint_lock=None,
            )
            for episode in range(3)
        ]

        gate = b9_waypoint_corridor_gate_result(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "corridor_b9_aggregate:explicit_b9_decision_episodes",
            gate["failures"],
        )

    def test_b10_corridor_gate_accepts_prospective_replay_progress(self) -> None:
        results = [
            self._b10_corridor_result(
                episode,
                steps=15,
                alive=False,
                food_distance_delta=13.0,
            )
            for episode in range(3)
        ]

        gate = b10_prospective_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["explicit_decision_episodes"], 3)
        self.assertEqual(gate["aggregate"]["replay_state_episodes"], 3)
        self.assertEqual(gate["aggregate"]["committed_plan_episodes"], 3)
        self.assertEqual(gate["aggregate"]["value_signal_episodes"], 3)

    def test_b10_corridor_gate_rejects_b9_clone_without_replay(self) -> None:
        results = [
            self._b10_corridor_result(
                episode,
                steps=15,
                alive=False,
                food_distance_delta=13.0,
                decision=None,
                replay_state=None,
                plan_commitment=None,
                prospective_value=None,
            )
            for episode in range(3)
        ]

        gate = b10_prospective_corridor_gate_result(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "corridor_b10_aggregate:explicit_b10_decision_episodes",
            gate["failures"],
        )

    def test_b11_corridor_gate_accepts_confidence_progress(self) -> None:
        results = [
            self._b11_corridor_result(
                episode,
                steps=15,
                alive=False,
                food_distance_delta=13.0,
            )
            for episode in range(3)
        ]

        gate = b11_confidence_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["explicit_decision_episodes"], 3)
        self.assertEqual(gate["aggregate"]["confidence_state_episodes"], 3)
        self.assertEqual(gate["aggregate"]["confidence_lock_episodes"], 3)
        self.assertEqual(gate["aggregate"]["neuromod_signal_episodes"], 3)

    def test_b11_corridor_gate_rejects_b10_clone_without_confidence(self) -> None:
        results = [
            self._b11_corridor_result(
                episode,
                steps=15,
                alive=False,
                food_distance_delta=13.0,
                decision=None,
                confidence_state=None,
                confidence_lock=None,
                neuromod_signal=None,
            )
            for episode in range(3)
        ]

        gate = b11_confidence_corridor_gate_result(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "corridor_b11_aggregate:explicit_b11_decision_episodes",
            gate["failures"],
        )

    def test_b12_corridor_gate_accepts_attention_progress(self) -> None:
        results = [
            self._b12_corridor_result(
                episode,
                steps=15,
                alive=False,
                food_distance_delta=13.0,
            )
            for episode in range(3)
        ]

        gate = b12_attention_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["explicit_decision_episodes"], 3)
        self.assertEqual(gate["aggregate"]["attention_state_episodes"], 3)
        self.assertEqual(gate["aggregate"]["attention_lock_episodes"], 3)
        self.assertEqual(gate["aggregate"]["prediction_signal_episodes"], 3)

    def test_b12_corridor_gate_rejects_b11_clone_without_attention(self) -> None:
        results = [
            self._b12_corridor_result(
                episode,
                steps=15,
                alive=False,
                food_distance_delta=13.0,
                decision=None,
                attention_state=None,
                search_lock=None,
                attention_gain=None,
                prediction_error=None,
            )
            for episode in range(3)
        ]

        gate = b12_attention_corridor_gate_result(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "corridor_b12_aggregate:explicit_b12_decision_episodes",
            gate["failures"],
        )

    def test_b13_corridor_gate_accepts_local_search_progress(self) -> None:
        results = [
            self._b13_corridor_result(
                episode,
                steps=15,
                alive=False,
                food_distance_delta=13.0,
            )
            for episode in range(3)
        ]

        gate = b13_local_search_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["explicit_decision_episodes"], 3)
        self.assertEqual(gate["aggregate"]["search_state_episodes"], 3)
        self.assertEqual(gate["aggregate"]["search_lock_episodes"], 3)
        self.assertEqual(gate["aggregate"]["local_search_signal_episodes"], 3)

    def test_b13_corridor_gate_rejects_b12_clone_without_local_search(self) -> None:
        results = [
            self._b13_corridor_result(
                episode,
                steps=15,
                alive=False,
                food_distance_delta=13.0,
                decision=None,
                search_state=None,
                search_lock=None,
                route_score=None,
                affordance_samples=None,
                dead_end_score=None,
            )
            for episode in range(3)
        ]

        gate = b13_local_search_corridor_gate_result(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "corridor_b13_aggregate:explicit_b13_decision_episodes",
            gate["failures"],
        )

    def test_b14_corridor_gate_accepts_uncertainty_progress(self) -> None:
        results = [
            self._b14_corridor_result(
                episode,
                steps=15,
                alive=False,
                food_distance_delta=13.0,
            )
            for episode in range(3)
        ]

        gate = b14_uncertainty_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["explicit_decision_episodes"], 3)
        self.assertEqual(gate["aggregate"]["uncertainty_state_episodes"], 3)
        self.assertEqual(gate["aggregate"]["confidence_signal_episodes"], 3)
        self.assertEqual(gate["aggregate"]["commitment_lock_episodes"], 3)

    def test_b14_corridor_gate_rejects_b13_clone_without_uncertainty(self) -> None:
        results = [
            self._b14_corridor_result(
                episode,
                steps=15,
                alive=False,
                food_distance_delta=13.0,
                decision=None,
                uncertainty_state=None,
                commitment_lock=None,
                confidence=None,
                uncertainty=None,
                risk_adjusted_score=None,
            )
            for episode in range(3)
        ]

        gate = b14_uncertainty_corridor_gate_result(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "corridor_b14_aggregate:explicit_b14_decision_episodes",
            gate["failures"],
        )

    def test_b15_corridor_gate_accepts_option_progress(self) -> None:
        results = [
            self._b15_corridor_result(
                episode,
                steps=15,
                alive=False,
                food_distance_delta=13.0,
            )
            for episode in range(3)
        ]

        gate = b15_option_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["explicit_decision_episodes"], 3)
        self.assertEqual(gate["aggregate"]["option_state_episodes"], 3)
        self.assertEqual(gate["aggregate"]["option_lock_episodes"], 3)
        self.assertEqual(gate["aggregate"]["option_value_signal_episodes"], 3)

    def test_b15_corridor_gate_rejects_b14_clone_without_option(self) -> None:
        results = [
            self._b15_corridor_result(
                episode,
                steps=15,
                alive=False,
                food_distance_delta=13.0,
                decision=None,
                option_state=None,
                option_lock=None,
                option_value=None,
                termination_pressure=None,
                persistence_score=None,
            )
            for episode in range(3)
        ]

        gate = b15_option_corridor_gate_result(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "corridor_b15_aggregate:explicit_b15_decision_episodes",
            gate["failures"],
        )

    def test_b16_corridor_gate_accepts_ensemble_progress(self) -> None:
        results = [
            self._b16_corridor_result(
                episode,
                steps=15,
                alive=False,
                food_distance_delta=13.0,
            )
            for episode in range(3)
        ]

        gate = b16_option_ensemble_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["explicit_decision_episodes"], 3)
        self.assertEqual(gate["aggregate"]["ensemble_state_episodes"], 3)
        self.assertEqual(gate["aggregate"]["ensemble_lock_episodes"], 3)
        self.assertEqual(gate["aggregate"]["ensemble_signal_episodes"], 3)

    def test_b16_corridor_gate_rejects_b15_clone_without_ensemble(self) -> None:
        results = [
            self._b16_corridor_result(
                episode,
                steps=15,
                alive=False,
                food_distance_delta=13.0,
                decision=None,
                ensemble_state=None,
                ensemble_lock=None,
                continue_vote=None,
                return_vote=None,
                consensus_score=None,
                conflict_score=None,
            )
            for episode in range(3)
        ]

        gate = b16_option_ensemble_corridor_gate_result(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "corridor_b16_aggregate:explicit_b16_decision_episodes",
            gate["failures"],
        )

    def test_b17_corridor_gate_accepts_neuromodulated_progress(self) -> None:
        results = [
            self._b17_corridor_result(
                episode,
                steps=15,
                alive=False,
                food_distance_delta=13.0,
            )
            for episode in range(3)
        ]

        gate = b17_neuromodulated_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["explicit_decision_episodes"], 3)
        self.assertEqual(gate["aggregate"]["modulator_state_episodes"], 3)
        self.assertEqual(gate["aggregate"]["modulation_lock_episodes"], 3)
        self.assertEqual(gate["aggregate"]["modulation_signal_episodes"], 3)

    def test_b17_corridor_gate_rejects_b16_clone_without_modulator(self) -> None:
        results = [
            self._b17_corridor_result(
                episode,
                steps=15,
                alive=False,
                food_distance_delta=13.0,
                decision=None,
                modulator_state=None,
                modulation_lock=None,
                arousal_signal=None,
                homeostatic_gain=None,
                option_gain=None,
                conflict_release=None,
            )
            for episode in range(3)
        ]

        gate = b17_neuromodulated_corridor_gate_result(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "corridor_b17_aggregate:explicit_b17_decision_episodes",
            gate["failures"],
        )

    def test_b18_corridor_gate_accepts_eligibility_trace_progress(self) -> None:
        results = [
            self._b18_corridor_result(
                episode,
                steps=15,
                alive=False,
                food_distance_delta=13.0,
            )
            for episode in range(3)
        ]

        gate = b18_eligibility_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["explicit_decision_episodes"], 3)
        self.assertEqual(gate["aggregate"]["trace_state_episodes"], 3)
        self.assertEqual(gate["aggregate"]["trace_lock_episodes"], 3)
        self.assertEqual(gate["aggregate"]["trace_signal_episodes"], 3)

    def test_b18_corridor_gate_rejects_b17_clone_without_trace(self) -> None:
        results = [
            self._b18_corridor_result(
                episode,
                steps=15,
                alive=False,
                food_distance_delta=13.0,
                decision=None,
                trace_state=None,
                trace_lock=None,
                eligibility_trace=None,
                prediction_proxy=None,
                stability_bias=None,
                switch_pressure=None,
            )
            for episode in range(3)
        ]

        gate = b18_eligibility_corridor_gate_result(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "corridor_b18_aggregate:explicit_b18_decision_episodes",
            gate["failures"],
        )

    def test_b19_corridor_gate_accepts_episodic_memory_progress(self) -> None:
        results = [
            self._b19_corridor_result(
                episode,
                steps=15,
                alive=False,
                food_distance_delta=13.0,
            )
            for episode in range(3)
        ]

        gate = b19_episodic_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["explicit_decision_episodes"], 3)
        self.assertEqual(gate["aggregate"]["memory_state_episodes"], 3)
        self.assertEqual(gate["aggregate"]["memory_lock_episodes"], 3)
        self.assertEqual(gate["aggregate"]["memory_signal_episodes"], 3)

    def test_b19_corridor_gate_rejects_b18_clone_without_memory(self) -> None:
        results = [
            self._b19_corridor_result(
                episode,
                steps=15,
                alive=False,
                food_distance_delta=13.0,
                decision=None,
                memory_state=None,
                memory_lock=None,
                episode_memory=None,
                consolidation_score=None,
                stability_vote=None,
                switch_suppression=None,
            )
            for episode in range(3)
        ]

        gate = b19_episodic_corridor_gate_result(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "corridor_b19_aggregate:explicit_b19_decision_episodes",
            gate["failures"],
        )

    def test_b19_corridor_gate_keeps_b18_base_as_diagnostic(self) -> None:
        results = [
            self._b19_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b19_episodic_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertFalse(gate["aggregate"]["base_b18_corridor_diagnostic"])
        self.assertEqual(gate["aggregate"]["corridor_safety_episodes"], 3)

    def test_b20_corridor_gate_accepts_working_memory_progress(self) -> None:
        results = [
            self._b20_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b20_working_memory_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["explicit_decision_episodes"], 3)
        self.assertEqual(gate["aggregate"]["buffer_state_episodes"], 3)
        self.assertEqual(gate["aggregate"]["buffer_lock_episodes"], 3)
        self.assertEqual(gate["aggregate"]["buffer_signal_episodes"], 3)

    def test_b20_corridor_gate_rejects_b19_clone_without_buffer(self) -> None:
        results = [
            self._b20_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
                decision=None,
                buffer_state=None,
                buffer_lock=None,
                working_buffer=None,
                context_binding=None,
                gate_vote=None,
                release_vote=None,
            )
            for episode in range(3)
        ]

        gate = b20_working_memory_corridor_gate_result(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "corridor_b20_aggregate:explicit_b20_decision_episodes",
            gate["failures"],
        )

    def test_b20_corridor_gate_keeps_b19_base_as_diagnostic(self) -> None:
        results = [
            self._b20_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b20_working_memory_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["corridor_safety_episodes"], 3)
        self.assertIn("base_b19_corridor_diagnostic", gate["aggregate"])

    def test_b21_corridor_gate_accepts_hippocampal_replay_progress(self) -> None:
        results = [
            self._b21_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b21_hippocampal_replay_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["explicit_decision_episodes"], 3)
        self.assertEqual(gate["aggregate"]["replay_state_episodes"], 3)
        self.assertEqual(gate["aggregate"]["replay_lock_episodes"], 3)
        self.assertEqual(gate["aggregate"]["replay_signal_episodes"], 3)

    def test_b21_corridor_gate_rejects_b20_clone_without_replay(self) -> None:
        results = [
            self._b21_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
                decision=None,
                replay_state=None,
                replay_lock=None,
                sequence_memory=None,
                replay_score=None,
                route_commitment=None,
                abort_prediction=None,
            )
            for episode in range(3)
        ]

        gate = b21_hippocampal_replay_corridor_gate_result(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "corridor_b21_aggregate:explicit_b21_decision_episodes",
            gate["failures"],
        )

    def test_b21_corridor_gate_keeps_b20_base_as_diagnostic(self) -> None:
        results = [
            self._b21_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b21_hippocampal_replay_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["corridor_safety_episodes"], 3)
        self.assertIn("base_b20_corridor_diagnostic", gate["aggregate"])

    def test_b22_corridor_gate_accepts_prospective_replay_progress(self) -> None:
        results = [
            self._b22_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b22_prospective_replay_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["explicit_decision_episodes"], 3)
        self.assertEqual(gate["aggregate"]["sim_state_episodes"], 3)
        self.assertEqual(gate["aggregate"]["sim_lock_episodes"], 3)
        self.assertEqual(gate["aggregate"]["sim_signal_episodes"], 3)

    def test_b22_corridor_gate_rejects_b21_clone_without_simulation(self) -> None:
        results = [
            self._b22_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
                decision=None,
                sim_state=None,
                sim_lock=None,
                prospective_sim=None,
                forward_model_score=None,
                viability_projection=None,
                abort_projection=None,
            )
            for episode in range(3)
        ]

        gate = b22_prospective_replay_corridor_gate_result(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "corridor_b22_aggregate:explicit_b22_decision_episodes",
            gate["failures"],
        )

    def test_b22_corridor_gate_keeps_b21_base_as_diagnostic(self) -> None:
        results = [
            self._b22_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b22_prospective_replay_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["corridor_safety_episodes"], 3)
        self.assertIn("base_b21_corridor_diagnostic", gate["aggregate"])

    def test_b23_corridor_gate_accepts_conflict_monitor_progress(self) -> None:
        results = [
            self._b23_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b23_conflict_monitor_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["explicit_decision_episodes"], 3)
        self.assertEqual(gate["aggregate"]["conflict_state_episodes"], 3)
        self.assertEqual(gate["aggregate"]["monitor_lock_episodes"], 3)
        self.assertEqual(gate["aggregate"]["conflict_signal_episodes"], 3)

    def test_b23_corridor_gate_rejects_b22_clone_without_monitor(self) -> None:
        results = [
            self._b23_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
                decision=None,
                conflict_state=None,
                monitor_lock=None,
                prediction_error=None,
                conflict_memory=None,
                stability_vote=None,
                abort_bias=None,
            )
            for episode in range(3)
        ]

        gate = b23_conflict_monitor_corridor_gate_result(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "corridor_b23_aggregate:explicit_b23_decision_episodes",
            gate["failures"],
        )

    def test_b23_corridor_gate_keeps_b22_base_as_diagnostic(self) -> None:
        results = [
            self._b23_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b23_conflict_monitor_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["corridor_safety_episodes"], 3)
        self.assertIn("base_b22_corridor_diagnostic", gate["aggregate"])

    def test_b24_corridor_gate_accepts_precision_conflict_progress(self) -> None:
        results = [
            self._b24_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b24_precision_conflict_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["explicit_decision_episodes"], 3)
        self.assertEqual(gate["aggregate"]["precision_state_episodes"], 3)
        self.assertEqual(gate["aggregate"]["precision_lock_episodes"], 3)
        self.assertEqual(gate["aggregate"]["precision_signal_episodes"], 3)

    def test_b24_corridor_gate_rejects_b23_clone_without_precision(self) -> None:
        results = [
            self._b24_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
                decision=None,
                precision_state=None,
                precision_lock=None,
                precision_memory=None,
                precision_vote=None,
                uncertainty_pressure=None,
                abort_precision=None,
            )
            for episode in range(3)
        ]

        gate = b24_precision_conflict_corridor_gate_result(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "corridor_b24_aggregate:explicit_b24_decision_episodes",
            gate["failures"],
        )

    def test_b24_corridor_gate_keeps_b23_base_as_diagnostic(self) -> None:
        results = [
            self._b24_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b24_precision_conflict_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["corridor_safety_episodes"], 3)
        self.assertIn("base_b23_corridor_diagnostic", gate["aggregate"])

    def test_b25_corridor_gate_accepts_metacognitive_progress(self) -> None:
        results = [
            self._b25_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b25_metacognitive_confidence_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["explicit_decision_episodes"], 3)
        self.assertEqual(gate["aggregate"]["metacognitive_state_episodes"], 3)
        self.assertEqual(gate["aggregate"]["meta_lock_episodes"], 3)
        self.assertEqual(gate["aggregate"]["metacognitive_signal_episodes"], 3)

    def test_b25_corridor_gate_rejects_b24_clone_without_metacognition(self) -> None:
        results = [
            self._b25_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
                decision=None,
                metacognitive_state=None,
                meta_lock=None,
                confidence_memory=None,
                confidence_vote=None,
                doubt_pressure=None,
                control_gain=None,
            )
            for episode in range(3)
        ]

        gate = b25_metacognitive_confidence_corridor_gate_result(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "corridor_b25_aggregate:explicit_b25_decision_episodes",
            gate["failures"],
        )

    def test_b25_corridor_gate_keeps_b24_base_as_diagnostic(self) -> None:
        results = [
            self._b25_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b25_metacognitive_confidence_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["corridor_safety_episodes"], 3)
        self.assertIn("base_b24_corridor_diagnostic", gate["aggregate"])

    def test_b26_corridor_gate_accepts_allostatic_progress(self) -> None:
        results = [
            self._b26_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b26_allostatic_prediction_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["explicit_decision_episodes"], 3)
        self.assertEqual(gate["aggregate"]["allostatic_state_episodes"], 3)
        self.assertEqual(gate["aggregate"]["stability_lock_episodes"], 3)
        self.assertEqual(gate["aggregate"]["allostatic_signal_episodes"], 3)

    def test_b26_corridor_gate_rejects_b25_clone_without_allostasis(self) -> None:
        results = [
            self._b26_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
                decision=None,
                allostatic_state=None,
                stability_lock=None,
                prediction_error=None,
                setpoint_pressure=None,
                control_vote=None,
            )
            for episode in range(3)
        ]

        gate = b26_allostatic_prediction_corridor_gate_result(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "corridor_b26_aggregate:explicit_b26_decision_episodes",
            gate["failures"],
        )

    def test_b26_corridor_gate_keeps_b25_base_as_diagnostic(self) -> None:
        results = [
            self._b26_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b26_allostatic_prediction_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["corridor_safety_episodes"], 3)
        self.assertIn("base_b25_corridor_diagnostic", gate["aggregate"])

    def test_b27_corridor_gate_accepts_arousal_progress(self) -> None:
        results = [
            self._b27_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b27_arousal_gain_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["explicit_decision_episodes"], 3)
        self.assertEqual(gate["aggregate"]["arousal_state_episodes"], 3)
        self.assertEqual(gate["aggregate"]["arousal_lock_episodes"], 3)
        self.assertEqual(gate["aggregate"]["arousal_signal_episodes"], 3)

    def test_b27_corridor_gate_rejects_b26_clone_without_arousal(self) -> None:
        results = [
            self._b27_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
                decision=None,
                arousal_state=None,
                arousal_lock=None,
                arousal_level=None,
                gain_modulation=None,
                stress_pressure=None,
            )
            for episode in range(3)
        ]

        gate = b27_arousal_gain_corridor_gate_result(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "corridor_b27_aggregate:explicit_b27_decision_episodes",
            gate["failures"],
        )

    def test_b27_corridor_gate_keeps_b26_base_as_diagnostic(self) -> None:
        results = [
            self._b27_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b27_arousal_gain_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["corridor_safety_episodes"], 3)
        self.assertIn("base_b26_corridor_diagnostic", gate["aggregate"])

    def test_b28_corridor_gate_accepts_attention_progress(self) -> None:
        results = [
            self._b28_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b28_interoceptive_attention_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["explicit_decision_episodes"], 3)
        self.assertEqual(gate["aggregate"]["attention_state_episodes"], 3)
        self.assertEqual(gate["aggregate"]["attention_lock_episodes"], 3)
        self.assertEqual(gate["aggregate"]["attention_signal_episodes"], 3)

    def test_b28_corridor_gate_rejects_b27_clone_without_attention(self) -> None:
        results = [
            self._b28_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
                decision=None,
                attention_state=None,
                attention_lock=None,
                interoceptive_focus=None,
                attention_gain=None,
                distractor_pressure=None,
            )
            for episode in range(3)
        ]

        gate = b28_interoceptive_attention_corridor_gate_result(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "corridor_b28_aggregate:explicit_b28_decision_episodes",
            gate["failures"],
        )

    def test_b28_corridor_gate_keeps_b27_base_as_diagnostic(self) -> None:
        results = [
            self._b28_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b28_interoceptive_attention_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["corridor_safety_episodes"], 3)
        self.assertIn("base_b27_corridor_diagnostic", gate["aggregate"])

    def test_b29_corridor_gate_accepts_salience_progress(self) -> None:
        results = [
            self._b29_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b29_salience_competition_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["explicit_decision_episodes"], 3)
        self.assertEqual(gate["aggregate"]["salience_state_episodes"], 3)
        self.assertEqual(gate["aggregate"]["salience_lock_episodes"], 3)
        self.assertEqual(gate["aggregate"]["salience_signal_episodes"], 3)
        self.assertEqual(gate["aggregate"]["winner_channel_episodes"], 3)

    def test_b29_corridor_gate_rejects_b28_clone_without_salience(self) -> None:
        results = [
            self._b29_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
                decision=None,
                salience_state=None,
                salience_lock=None,
                threat_salience=None,
                homeostatic_salience=None,
                corridor_salience=None,
                winner_channel=None,
            )
            for episode in range(3)
        ]

        gate = b29_salience_competition_corridor_gate_result(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "corridor_b29_aggregate:explicit_b29_decision_episodes",
            gate["failures"],
        )

    def test_b29_corridor_gate_keeps_b28_base_as_diagnostic(self) -> None:
        results = [
            self._b29_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b29_salience_competition_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["corridor_safety_episodes"], 3)
        self.assertIn("base_b28_corridor_diagnostic", gate["aggregate"])

    def test_b30_corridor_gate_accepts_basal_gate_progress(self) -> None:
        results = [
            self._b30_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b30_basal_ganglia_gate_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["explicit_decision_episodes"], 3)
        self.assertEqual(gate["aggregate"]["gate_state_episodes"], 3)
        self.assertEqual(gate["aggregate"]["gate_lock_episodes"], 3)
        self.assertEqual(gate["aggregate"]["gate_signal_episodes"], 3)
        self.assertEqual(gate["aggregate"]["action_gate_episodes"], 3)
