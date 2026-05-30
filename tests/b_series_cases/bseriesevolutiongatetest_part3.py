from __future__ import annotations

from .shared import *
from .bseriesevolutiongatetest_helpers import BSeriesEvolutionGateTestHelpers



class BSeriesEvolutionGateTestPart3(BSeriesEvolutionGateTestHelpers, unittest.TestCase):
    def test_b30_corridor_gate_rejects_b29_clone_without_gate(self) -> None:
        results = [
            self._b30_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
                decision=None,
                gate_state=None,
                gate_lock=None,
                go_signal=None,
                no_go_signal=None,
                action_gate=None,
            )
            for episode in range(3)
        ]

        gate = b30_basal_ganglia_gate_corridor_gate_result(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "corridor_b30_aggregate:explicit_b30_decision_episodes",
            gate["failures"],
        )

    def test_b30_corridor_gate_keeps_b29_base_as_diagnostic(self) -> None:
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
        self.assertEqual(gate["aggregate"]["corridor_safety_episodes"], 3)
        self.assertIn("base_b29_corridor_diagnostic", gate["aggregate"])

    def test_b31_corridor_gate_accepts_dopamine_progress(self) -> None:
        results = [
            self._b31_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b31_dopamine_prediction_error_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["explicit_decision_episodes"], 3)
        self.assertEqual(gate["aggregate"]["dopamine_state_episodes"], 3)
        self.assertEqual(gate["aggregate"]["dopamine_lock_episodes"], 3)
        self.assertEqual(gate["aggregate"]["dopamine_signal_episodes"], 3)
        self.assertEqual(gate["aggregate"]["gate_bias_episodes"], 3)

    def test_b31_corridor_gate_rejects_b30_clone_without_dopamine(self) -> None:
        results = [
            self._b31_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
                decision=None,
                dopamine_state=None,
                dopamine_lock=None,
                reward_prediction_error=None,
                tonic_dopamine=None,
                phasic_dopamine=None,
                gate_bias=None,
            )
            for episode in range(3)
        ]

        gate = b31_dopamine_prediction_error_corridor_gate_result(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "corridor_b31_aggregate:explicit_b31_decision_episodes",
            gate["failures"],
        )

    def test_b31_corridor_gate_keeps_b30_base_as_diagnostic(self) -> None:
        results = [
            self._b31_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b31_dopamine_prediction_error_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["corridor_safety_episodes"], 3)
        self.assertIn("base_b30_corridor_diagnostic", gate["aggregate"])

    def test_b32_corridor_gate_accepts_actor_critic_progress(self) -> None:
        results = [
            self._b32_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b32_actor_critic_value_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["explicit_decision_episodes"], 3)
        self.assertEqual(gate["aggregate"]["critic_value_episodes"], 3)
        self.assertEqual(gate["aggregate"]["actor_advantage_episodes"], 3)
        self.assertEqual(gate["aggregate"]["value_lock_episodes"], 3)
        self.assertEqual(gate["aggregate"]["policy_bias_episodes"], 3)

    def test_b32_corridor_gate_rejects_b31_clone_without_value(self) -> None:
        results = [
            self._b32_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
                decision=None,
                critic_value=None,
                actor_advantage=None,
                value_error=None,
                policy_bias=None,
                value_lock=None,
            )
            for episode in range(3)
        ]

        gate = b32_actor_critic_value_corridor_gate_result(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "corridor_b32_aggregate:explicit_b32_decision_episodes",
            gate["failures"],
        )

    def test_b32_corridor_gate_keeps_b31_base_as_diagnostic(self) -> None:
        results = [
            self._b32_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b32_actor_critic_value_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["corridor_safety_episodes"], 3)
        self.assertIn("base_b31_corridor_diagnostic", gate["aggregate"])

    def test_b33_corridor_gate_accepts_td_error_progress(self) -> None:
        results = [
            self._b33_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b33_td_error_decomposition_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["explicit_decision_episodes"], 3)
        self.assertEqual(gate["aggregate"]["td_error_episodes"], 3)
        self.assertEqual(gate["aggregate"]["bootstrap_value_episodes"], 3)
        self.assertEqual(gate["aggregate"]["reward_trace_episodes"], 3)
        self.assertEqual(gate["aggregate"]["td_lock_episodes"], 3)

    def test_b33_corridor_gate_rejects_b32_clone_without_td_error(self) -> None:
        results = [
            self._b33_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
                decision=None,
                td_error=None,
                bootstrap_value=None,
                reward_trace=None,
                actor_update=None,
                td_lock=None,
            )
            for episode in range(3)
        ]

        gate = b33_td_error_decomposition_corridor_gate_result(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "corridor_b33_aggregate:explicit_b33_decision_episodes",
            gate["failures"],
        )

    def test_b33_corridor_gate_keeps_b32_base_as_diagnostic(self) -> None:
        results = [
            self._b33_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b33_td_error_decomposition_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["corridor_safety_episodes"], 3)
        self.assertIn("base_b32_corridor_diagnostic", gate["aggregate"])

    def test_b34_corridor_gate_accepts_eligibility_credit_progress(self) -> None:
        results = [
            self._b34_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b34_eligibility_credit_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["explicit_decision_episodes"], 3)
        self.assertEqual(gate["aggregate"]["eligibility_trace_episodes"], 3)
        self.assertEqual(gate["aggregate"]["credit_assignment_episodes"], 3)
        self.assertEqual(gate["aggregate"]["synaptic_tag_episodes"], 3)
        self.assertEqual(gate["aggregate"]["credit_lock_episodes"], 3)

    def test_b34_corridor_gate_rejects_b33_clone_without_eligibility(self) -> None:
        results = [
            self._b34_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
                decision=None,
                eligibility_trace=None,
                credit_assignment=None,
                synaptic_tag=None,
                decay_memory=None,
                credit_lock=None,
            )
            for episode in range(3)
        ]

        gate = b34_eligibility_credit_corridor_gate_result(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "corridor_b34_aggregate:explicit_b34_decision_episodes",
            gate["failures"],
        )

    def test_b34_corridor_gate_keeps_b33_base_as_diagnostic(self) -> None:
        results = [
            self._b34_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b34_eligibility_credit_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["corridor_safety_episodes"], 3)
        self.assertIn("base_b33_corridor_diagnostic", gate["aggregate"])

    def test_b35_corridor_gate_accepts_forward_model_progress(self) -> None:
        results = [
            self._b35_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b35_forward_model_value_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["explicit_decision_episodes"], 3)
        self.assertEqual(gate["aggregate"]["forward_value_episodes"], 3)
        self.assertEqual(gate["aggregate"]["transition_error_episodes"], 3)
        self.assertEqual(gate["aggregate"]["model_confidence_episodes"], 3)
        self.assertEqual(gate["aggregate"]["prediction_memory_episodes"], 3)
        self.assertEqual(gate["aggregate"]["model_lock_episodes"], 3)

    def test_b35_corridor_gate_rejects_b34_clone_without_model(self) -> None:
        results = [
            self._b35_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
                decision=None,
                forward_value=None,
                transition_error=None,
                model_confidence=None,
                prediction_memory=None,
                model_lock=None,
            )
            for episode in range(3)
        ]

        gate = b35_forward_model_value_corridor_gate_result(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "corridor_b35_aggregate:explicit_b35_decision_episodes",
            gate["failures"],
        )

    def test_b35_corridor_gate_keeps_b34_base_as_diagnostic(self) -> None:
        results = [
            self._b35_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b35_forward_model_value_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["corridor_safety_episodes"], 3)
        self.assertIn("base_b34_corridor_diagnostic", gate["aggregate"])

    def test_b36_corridor_gate_accepts_latent_belief_progress(self) -> None:
        results = [
            self._b36_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b36_latent_belief_state_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["explicit_decision_episodes"], 3)
        self.assertEqual(gate["aggregate"]["latent_state_episodes"], 3)
        self.assertEqual(gate["aggregate"]["belief_error_episodes"], 3)
        self.assertEqual(gate["aggregate"]["state_confidence_episodes"], 3)
        self.assertEqual(gate["aggregate"]["context_memory_episodes"], 3)
        self.assertEqual(gate["aggregate"]["belief_lock_episodes"], 3)

    def test_b36_corridor_gate_rejects_b35_clone_without_belief(self) -> None:
        results = [
            self._b36_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
                decision=None,
                latent_state=None,
                belief_error=None,
                state_confidence=None,
                context_memory=None,
                belief_lock=None,
            )
            for episode in range(3)
        ]

        gate = b36_latent_belief_state_corridor_gate_result(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "corridor_b36_aggregate:explicit_b36_decision_episodes",
            gate["failures"],
        )

    def test_b36_corridor_gate_keeps_b35_base_as_diagnostic(self) -> None:
        results = [
            self._b36_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b36_latent_belief_state_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["corridor_safety_episodes"], 3)
        self.assertIn("base_b35_corridor_diagnostic", gate["aggregate"])

    def test_b37_corridor_gate_accepts_state_factor_progress(self) -> None:
        results = [
            self._b37_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b37_state_factor_gate_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["explicit_decision_episodes"], 3)
        self.assertEqual(gate["aggregate"]["external_factor_episodes"], 3)
        self.assertEqual(gate["aggregate"]["internal_factor_episodes"], 3)
        self.assertEqual(gate["aggregate"]["factor_alignment_episodes"], 3)
        self.assertEqual(gate["aggregate"]["factor_confidence_episodes"], 3)
        self.assertEqual(gate["aggregate"]["factor_lock_episodes"], 3)

    def test_b37_corridor_gate_rejects_b36_clone_without_factorization(self) -> None:
        results = [
            self._b37_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
                decision=None,
                external_factor=None,
                internal_factor=None,
                factor_alignment=None,
                factor_confidence=None,
                factor_lock=None,
            )
            for episode in range(3)
        ]

        gate = b37_state_factor_gate_corridor_gate_result(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "corridor_b37_aggregate:explicit_b37_decision_episodes",
            gate["failures"],
        )

    def test_b37_corridor_gate_keeps_b36_base_as_diagnostic(self) -> None:
        results = [
            self._b37_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b37_state_factor_gate_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["corridor_safety_episodes"], 3)
        self.assertIn("base_b36_corridor_diagnostic", gate["aggregate"])

    def test_b38_corridor_gate_accepts_factor_attention_progress(self) -> None:
        results = [
            self._b38_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b38_factor_attention_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["explicit_decision_episodes"], 3)
        self.assertEqual(gate["aggregate"]["external_attention_episodes"], 3)
        self.assertEqual(gate["aggregate"]["internal_attention_episodes"], 3)
        self.assertEqual(gate["aggregate"]["attention_balance_episodes"], 3)
        self.assertEqual(gate["aggregate"]["attention_gain_episodes"], 3)
        self.assertEqual(gate["aggregate"]["attention_lock_episodes"], 3)

    def test_b38_corridor_gate_rejects_b37_clone_without_attention(self) -> None:
        results = [
            self._b38_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
                decision=None,
                external_attention=None,
                internal_attention=None,
                attention_balance=None,
                attention_gain=None,
                attention_lock=None,
            )
            for episode in range(3)
        ]

        gate = b38_factor_attention_corridor_gate_result(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "corridor_b38_aggregate:explicit_b38_decision_episodes",
            gate["failures"],
        )

    def test_b38_corridor_gate_keeps_b37_base_as_diagnostic(self) -> None:
        results = [
            self._b38_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b38_factor_attention_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["corridor_safety_episodes"], 3)
        self.assertIn("base_b37_corridor_diagnostic", gate["aggregate"])

    def test_b39_corridor_gate_accepts_attention_binding_progress(self) -> None:
        results = [
            self._b39_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b39_attention_binding_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["explicit_decision_episodes"], 3)
        self.assertEqual(gate["aggregate"]["binding_strength_episodes"], 3)
        self.assertEqual(gate["aggregate"]["cross_factor_coherence_episodes"], 3)
        self.assertEqual(gate["aggregate"]["bound_context_episodes"], 3)
        self.assertEqual(gate["aggregate"]["binding_gain_episodes"], 3)
        self.assertEqual(gate["aggregate"]["binding_lock_episodes"], 3)

    def test_b39_corridor_gate_rejects_b38_clone_without_binding(self) -> None:
        results = [
            self._b39_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
                decision=None,
                binding_strength=None,
                cross_factor_coherence=None,
                bound_context=None,
                binding_gain=None,
                binding_lock=None,
            )
            for episode in range(3)
        ]

        gate = b39_attention_binding_corridor_gate_result(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "corridor_b39_aggregate:explicit_b39_decision_episodes",
            gate["failures"],
        )

    def test_b39_corridor_gate_keeps_b38_base_as_diagnostic(self) -> None:
        results = [
            self._b39_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b39_attention_binding_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["corridor_safety_episodes"], 3)
        self.assertIn("base_b38_corridor_diagnostic", gate["aggregate"])

    def test_b40_corridor_gate_accepts_global_workspace_progress(self) -> None:
        results = [
            self._b40_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b40_global_workspace_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["explicit_decision_episodes"], 3)
        self.assertEqual(gate["aggregate"]["workspace_activation_episodes"], 3)
        self.assertEqual(gate["aggregate"]["broadcast_gain_episodes"], 3)
        self.assertEqual(gate["aggregate"]["context_availability_episodes"], 3)
        self.assertEqual(gate["aggregate"]["workspace_stability_episodes"], 3)
        self.assertEqual(gate["aggregate"]["workspace_lock_episodes"], 3)

    def test_b40_corridor_gate_rejects_b39_clone_without_workspace(self) -> None:
        results = [
            self._b40_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
                decision=None,
                workspace_activation=None,
                broadcast_gain=None,
                context_availability=None,
                workspace_stability=None,
                workspace_lock=None,
            )
            for episode in range(3)
        ]

        gate = b40_global_workspace_corridor_gate_result(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "corridor_b40_aggregate:explicit_b40_decision_episodes",
            gate["failures"],
        )

    def test_b40_corridor_gate_keeps_b39_base_as_diagnostic(self) -> None:
        results = [
            self._b40_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b40_global_workspace_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["corridor_safety_episodes"], 3)
        self.assertIn("base_b39_corridor_diagnostic", gate["aggregate"])

    def test_b41_corridor_gate_accepts_executive_workspace_progress(self) -> None:
        results = [
            self._b41_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b41_executive_workspace_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["explicit_decision_episodes"], 3)
        self.assertEqual(gate["aggregate"]["executive_selection_episodes"], 3)
        self.assertEqual(gate["aggregate"]["inhibitory_pressure_episodes"], 3)
        self.assertEqual(gate["aggregate"]["goal_context_episodes"], 3)
        self.assertEqual(gate["aggregate"]["executive_stability_episodes"], 3)
        self.assertEqual(gate["aggregate"]["executive_lock_episodes"], 3)

    def test_b41_corridor_gate_rejects_b40_clone_without_executive_control(self) -> None:
        results = [
            self._b41_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
                decision=None,
                executive_selection=None,
                inhibitory_pressure=None,
                goal_context=None,
                executive_stability=None,
                executive_lock=None,
            )
            for episode in range(3)
        ]

        gate = b41_executive_workspace_corridor_gate_result(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "corridor_b41_aggregate:explicit_b41_decision_episodes",
            gate["failures"],
        )

    def test_b41_corridor_gate_keeps_b40_base_as_diagnostic(self) -> None:
        results = [
            self._b41_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b41_executive_workspace_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["corridor_safety_episodes"], 3)
        self.assertIn("base_b40_corridor_diagnostic", gate["aggregate"])

    def test_b42_corridor_gate_accepts_error_monitor_progress(self) -> None:
        results = [
            self._b42_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b42_error_monitor_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["explicit_decision_episodes"], 3)
        self.assertEqual(gate["aggregate"]["error_signal_episodes"], 3)
        self.assertEqual(gate["aggregate"]["conflict_signal_episodes"], 3)
        self.assertEqual(gate["aggregate"]["performance_context_episodes"], 3)
        self.assertEqual(gate["aggregate"]["monitor_stability_episodes"], 3)
        self.assertEqual(gate["aggregate"]["monitor_lock_episodes"], 3)

    def test_b42_corridor_gate_rejects_b41_clone_without_error_monitor(self) -> None:
        results = [
            self._b42_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
                decision=None,
                error_signal=None,
                conflict_signal=None,
                performance_context=None,
                monitor_stability=None,
                monitor_lock=None,
            )
            for episode in range(3)
        ]

        gate = b42_error_monitor_corridor_gate_result(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "corridor_b42_aggregate:explicit_b42_decision_episodes",
            gate["failures"],
        )

    def test_b42_corridor_gate_keeps_b41_base_as_diagnostic(self) -> None:
        results = [
            self._b42_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b42_error_monitor_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["corridor_safety_episodes"], 3)
        self.assertIn("base_b41_corridor_diagnostic", gate["aggregate"])

    def test_b43_corridor_gate_accepts_adaptive_precision_progress(self) -> None:
        results = [
            self._b43_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b43_adaptive_precision_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["explicit_decision_episodes"], 3)
        self.assertEqual(gate["aggregate"]["precision_signal_episodes"], 3)
        self.assertEqual(gate["aggregate"]["adaptive_threshold_episodes"], 3)
        self.assertEqual(gate["aggregate"]["arousal_context_episodes"], 3)
        self.assertEqual(gate["aggregate"]["control_stability_episodes"], 3)
        self.assertEqual(gate["aggregate"]["precision_lock_episodes"], 3)

    def test_b43_corridor_gate_rejects_b42_clone_without_precision(self) -> None:
        results = [
            self._b43_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
                decision=None,
                precision_signal=None,
                adaptive_threshold=None,
                arousal_context=None,
                control_stability=None,
                precision_lock=None,
            )
            for episode in range(3)
        ]

        gate = b43_adaptive_precision_corridor_gate_result(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "corridor_b43_aggregate:explicit_b43_decision_episodes",
            gate["failures"],
        )

    def test_b43_corridor_gate_keeps_b42_base_as_diagnostic(self) -> None:
        results = [
            self._b43_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b43_adaptive_precision_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["corridor_safety_episodes"], 3)
        self.assertIn("base_b42_corridor_diagnostic", gate["aggregate"])

    def test_b44_corridor_gate_accepts_thalamic_relay_progress(self) -> None:
        results = [
            self._b44_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b44_thalamic_relay_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["explicit_decision_episodes"], 3)
        self.assertEqual(gate["aggregate"]["relay_gate_episodes"], 3)
        self.assertEqual(gate["aggregate"]["sensory_precision_episodes"], 3)
        self.assertEqual(gate["aggregate"]["context_relay_episodes"], 3)
        self.assertEqual(gate["aggregate"]["gate_stability_episodes"], 3)
        self.assertEqual(gate["aggregate"]["relay_lock_episodes"], 3)

    def test_b44_corridor_gate_rejects_b43_clone_without_relay(self) -> None:
        results = [
            self._b44_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
                decision=None,
                relay_gate=None,
                sensory_precision=None,
                context_relay=None,
                gate_stability=None,
                relay_lock=None,
            )
            for episode in range(3)
        ]

        gate = b44_thalamic_relay_corridor_gate_result(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "corridor_b44_aggregate:explicit_b44_decision_episodes",
            gate["failures"],
        )

    def test_b44_corridor_gate_keeps_b43_base_as_diagnostic(self) -> None:
        results = [
            self._b44_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b44_thalamic_relay_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["corridor_safety_episodes"], 3)
        self.assertIn("base_b43_corridor_diagnostic", gate["aggregate"])

    def test_b45_corridor_gate_accepts_reticular_inhibition_progress(self) -> None:
        results = [
            self._b45_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b45_reticular_inhibition_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["explicit_decision_episodes"], 3)
        self.assertEqual(gate["aggregate"]["inhibitory_gate_episodes"], 3)
        self.assertEqual(gate["aggregate"]["sensory_filter_episodes"], 3)
        self.assertEqual(gate["aggregate"]["context_suppression_episodes"], 3)
        self.assertEqual(gate["aggregate"]["loop_stability_episodes"], 3)
        self.assertEqual(gate["aggregate"]["inhibition_lock_episodes"], 3)

    def test_b45_corridor_gate_rejects_b44_clone_without_inhibition(self) -> None:
        results = [
            self._b45_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
                decision=None,
                inhibitory_gate=None,
                sensory_filter=None,
                context_suppression=None,
                loop_stability=None,
                inhibition_lock=None,
            )
            for episode in range(3)
        ]

        gate = b45_reticular_inhibition_corridor_gate_result(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "corridor_b45_aggregate:explicit_b45_decision_episodes",
            gate["failures"],
        )

    def test_b45_corridor_gate_keeps_b44_base_as_diagnostic(self) -> None:
        results = [
            self._b45_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b45_reticular_inhibition_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["corridor_safety_episodes"], 3)
        self.assertIn("base_b44_corridor_diagnostic", gate["aggregate"])

    def test_b46_corridor_gate_accepts_feedback_progress(self) -> None:
        results = [
            self._b46_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b46_corticothalamic_feedback_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["explicit_decision_episodes"], 3)
        self.assertEqual(gate["aggregate"]["feedback_gain_episodes"], 3)
        self.assertEqual(gate["aggregate"]["topdown_context_episodes"], 3)
        self.assertEqual(gate["aggregate"]["prediction_match_episodes"], 3)
        self.assertEqual(gate["aggregate"]["feedback_stability_episodes"], 3)
        self.assertEqual(gate["aggregate"]["feedback_lock_episodes"], 3)

    def test_b46_corridor_gate_rejects_b45_clone_without_feedback(self) -> None:
        results = [
            self._b46_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
                decision=None,
                feedback_gain=None,
                topdown_context=None,
                prediction_match=None,
                feedback_stability=None,
                feedback_lock=None,
            )
            for episode in range(3)
        ]

        gate = b46_corticothalamic_feedback_corridor_gate_result(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "corridor_b46_aggregate:explicit_b46_decision_episodes",
            gate["failures"],
        )

    def test_b46_corridor_gate_keeps_b45_base_as_diagnostic(self) -> None:
        results = [
            self._b46_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b46_corticothalamic_feedback_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["corridor_safety_episodes"], 3)
        self.assertIn("base_b45_corridor_diagnostic", gate["aggregate"])

    def test_b47_corridor_gate_accepts_oscillatory_synchrony_progress(self) -> None:
        results = [
            self._b47_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b47_oscillatory_synchrony_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["explicit_decision_episodes"], 3)
        self.assertEqual(gate["aggregate"]["phase_alignment_episodes"], 3)
        self.assertEqual(gate["aggregate"]["synchrony_gain_episodes"], 3)
        self.assertEqual(gate["aggregate"]["cross_loop_coherence_episodes"], 3)
        self.assertEqual(gate["aggregate"]["phase_lock_episodes"], 3)

    def test_b47_corridor_gate_rejects_b46_clone_without_synchrony(self) -> None:
        results = [
            self._b47_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
                decision=None,
                phase_alignment=None,
                synchrony_gain=None,
                cross_loop_coherence=None,
                phase_lock=None,
            )
            for episode in range(3)
        ]

        gate = b47_oscillatory_synchrony_corridor_gate_result(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "corridor_b47_aggregate:explicit_b47_decision_episodes",
            gate["failures"],
        )

    def test_b47_corridor_gate_keeps_b46_base_as_diagnostic(self) -> None:
        results = [
            self._b47_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b47_oscillatory_synchrony_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["corridor_safety_episodes"], 3)
        self.assertIn("base_b46_corridor_diagnostic", gate["aggregate"])

    def test_b48_corridor_gate_accepts_cerebellar_timing_progress(self) -> None:
        results = [
            self._b48_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b48_cerebellar_timing_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["explicit_decision_episodes"], 3)
        self.assertEqual(gate["aggregate"]["timing_error_episodes"], 3)
        self.assertEqual(gate["aggregate"]["predictive_timing_episodes"], 3)
        self.assertEqual(gate["aggregate"]["corrective_gain_episodes"], 3)
        self.assertEqual(gate["aggregate"]["calibration_lock_episodes"], 3)

    def test_b48_corridor_gate_rejects_b47_clone_without_timing(self) -> None:
        results = [
            self._b48_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
                decision=None,
                timing_error=None,
                predictive_timing=None,
                corrective_gain=None,
                calibration_lock=None,
            )
            for episode in range(3)
        ]

        gate = b48_cerebellar_timing_corridor_gate_result(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "corridor_b48_aggregate:explicit_b48_decision_episodes",
            gate["failures"],
        )

    def test_b48_corridor_gate_keeps_b47_base_as_diagnostic(self) -> None:
        results = [
            self._b48_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b48_cerebellar_timing_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["corridor_safety_episodes"], 3)
        self.assertIn("base_b47_corridor_diagnostic", gate["aggregate"])

    def test_b49_corridor_gate_accepts_striatal_gate_progress(self) -> None:
        results = [
            self._b49_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b49_striatal_action_gate_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["explicit_decision_episodes"], 3)
        self.assertEqual(gate["aggregate"]["go_signal_episodes"], 3)
        self.assertEqual(gate["aggregate"]["no_go_signal_episodes"], 3)
        self.assertEqual(gate["aggregate"]["action_gate_balance_episodes"], 3)
        self.assertEqual(gate["aggregate"]["selection_lock_episodes"], 3)

    def test_b49_corridor_gate_rejects_b48_clone_without_striatal_gate(self) -> None:
        results = [
            self._b49_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
                decision=None,
                go_signal=None,
                no_go_signal=None,
                action_gate_balance=None,
                selection_lock=None,
            )
            for episode in range(3)
        ]

        gate = b49_striatal_action_gate_corridor_gate_result(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "corridor_b49_aggregate:explicit_b49_decision_episodes",
            gate["failures"],
        )

    def test_b49_corridor_gate_keeps_b48_base_as_diagnostic(self) -> None:
        results = [
            self._b49_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b49_striatal_action_gate_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["corridor_safety_episodes"], 3)
        self.assertIn("base_b48_corridor_diagnostic", gate["aggregate"])

    def test_b50_corridor_gate_accepts_habit_chunking_progress(self) -> None:
        results = [
            self._b50_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b50_habit_chunking_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["explicit_decision_episodes"], 3)
        self.assertEqual(gate["aggregate"]["habit_strength_episodes"], 3)
        self.assertEqual(gate["aggregate"]["chunk_value_episodes"], 3)
        self.assertEqual(gate["aggregate"]["habit_stability_episodes"], 3)
        self.assertEqual(gate["aggregate"]["chunk_lock_episodes"], 3)

    def test_b50_corridor_gate_rejects_b49_clone_without_habit_chunking(self) -> None:
        results = [
            self._b50_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
                decision=None,
                habit_strength=None,
                chunk_value=None,
                habit_stability=None,
                chunk_lock=None,
            )
            for episode in range(3)
        ]

        gate = b50_habit_chunking_corridor_gate_result(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "corridor_b50_aggregate:explicit_b50_decision_episodes",
            gate["failures"],
        )

    def test_b50_corridor_gate_keeps_b49_base_as_diagnostic(self) -> None:
        results = [
            self._b50_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b50_habit_chunking_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["corridor_safety_episodes"], 3)
        self.assertIn("base_b49_corridor_diagnostic", gate["aggregate"])

    def test_b51_corridor_gate_accepts_dopaminergic_habit_progress(self) -> None:
        results = [
            self._b51_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b51_dopaminergic_habit_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["explicit_decision_episodes"], 3)
        self.assertEqual(gate["aggregate"]["prediction_error_episodes"], 3)
        self.assertEqual(gate["aggregate"]["dopamine_gain_episodes"], 3)
        self.assertEqual(gate["aggregate"]["habit_modulation_episodes"], 3)
        self.assertEqual(gate["aggregate"]["modulation_lock_episodes"], 3)

    def test_b51_corridor_gate_rejects_b50_clone_without_dopamine(self) -> None:
        results = [
            self._b51_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
                decision=None,
                prediction_error=None,
                dopamine_gain=None,
                habit_modulation=None,
                modulation_lock=None,
            )
            for episode in range(3)
        ]

        gate = b51_dopaminergic_habit_corridor_gate_result(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "corridor_b51_aggregate:explicit_b51_decision_episodes",
            gate["failures"],
        )

    def test_b51_corridor_gate_keeps_b50_base_as_diagnostic(self) -> None:
        results = [
            self._b51_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b51_dopaminergic_habit_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["corridor_safety_episodes"], 3)
        self.assertIn("base_b50_corridor_diagnostic", gate["aggregate"])

    def test_b52_corridor_gate_accepts_cholinergic_precision_progress(self) -> None:
        results = [
            self._b52_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b52_cholinergic_precision_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["explicit_decision_episodes"], 3)
        self.assertEqual(gate["aggregate"]["acetylcholine_level_episodes"], 3)
        self.assertEqual(gate["aggregate"]["precision_gain_episodes"], 3)
        self.assertEqual(gate["aggregate"]["uncertainty_signal_episodes"], 3)
        self.assertEqual(gate["aggregate"]["attention_lock_episodes"], 3)

    def test_b52_corridor_gate_rejects_b51_clone_without_precision(self) -> None:
        results = [
            self._b52_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
                decision=None,
                acetylcholine_level=None,
                precision_gain=None,
                uncertainty_signal=None,
                attention_lock=None,
            )
            for episode in range(3)
        ]

        gate = b52_cholinergic_precision_corridor_gate_result(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "corridor_b52_aggregate:explicit_b52_decision_episodes",
            gate["failures"],
        )

    def test_b52_corridor_gate_keeps_b51_base_as_diagnostic(self) -> None:
        results = [
            self._b52_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b52_cholinergic_precision_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["corridor_safety_episodes"], 3)
        self.assertIn("base_b51_corridor_diagnostic", gate["aggregate"])

    def test_b53_corridor_gate_accepts_noradrenergic_arousal_progress(self) -> None:
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
        self.assertEqual(gate["aggregate"]["explicit_decision_episodes"], 3)
        self.assertEqual(gate["aggregate"]["norepinephrine_level_episodes"], 3)
        self.assertEqual(gate["aggregate"]["arousal_gain_episodes"], 3)
        self.assertEqual(gate["aggregate"]["surprise_signal_episodes"], 3)
        self.assertEqual(gate["aggregate"]["gain_lock_episodes"], 3)
