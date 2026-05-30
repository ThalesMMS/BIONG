from __future__ import annotations

from .shared import *


class BSeriesActionSpaceTest(unittest.TestCase):
    def test_current_world_still_exposes_only_nine_primitive_actions(self) -> None:
        self.assertEqual(len(ACTIONS), 9)
        self.assertEqual(tuple(ACTIONS), tuple(ACTION_TO_INDEX.keys()))
        for semantic_action in ("MOVE_TO_FOOD", "MOVE_TO_SHELTER", "EXPLORE", "EAT", "SLEEP"):
            self.assertNotIn(semantic_action, ACTIONS)

    def test_legacy_harness_exposes_exact_six_semantic_actions(self) -> None:
        self.assertEqual(tuple(LEGACY_B0_ACTIONS), tuple(B_SEMANTIC_ACTIONS))
        self.assertEqual(len(LEGACY_B0_ACTIONS), 6)

    def test_diagnostic_catalog_registers_b0_variants(self) -> None:
        configs = resolve_ablation_configs(
            ["b0_legacy_semantic_policy", "b0_current_bridge_policy"]
        )
        self.assertEqual(configs[0].architecture, "b_series")
        self.assertEqual(configs[0].b_mode, "legacy_semantic")
        self.assertEqual(configs[1].architecture, "b_series")
        self.assertEqual(configs[1].b_mode, "current_bridge")

    def test_diagnostic_catalog_registers_b1_evolution_variants(self) -> None:
        h48, h64, threat_guard = resolve_ablation_configs(
            [
                B1_CAPACITY_H48_POLICY_NAME,
                B1_CAPACITY_H64_POLICY_NAME,
                B1_THREAT_GUARD_POLICY_NAME,
            ]
        )

        self.assertEqual(h48.architecture, "b_series")
        self.assertEqual(h48.b_level, 1)
        self.assertEqual(h48.b_parent_level, 0)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(
            h48.b_transfer_source_checkpoint,
            B0_CURRENT_BRIDGE_DEFAULT_CHECKPOINT,
        )
        self.assertFalse(h48.b_transfer_allow_low_coverage)
        self.assertEqual(h64.b_hidden_dim, 64)
        self.assertEqual(threat_guard.b_hidden_dim, 48)
        self.assertEqual(
            threat_guard.b_transfer_source_checkpoint,
            h48.b_transfer_source_checkpoint,
        )
        self.assertFalse(threat_guard.enable_reflexes)

    def test_diagnostic_catalog_registers_b2_temporal_threat_variants(self) -> None:
        h48, h56, h64 = resolve_ablation_configs(
            [
                B2_TEMPORAL_THREAT_H48_POLICY_NAME,
                B2_TEMPORAL_THREAT_H56_POLICY_NAME,
                B2_TEMPORAL_THREAT_H64_POLICY_NAME,
            ]
        )

        self.assertEqual(h48.architecture, "b_series")
        self.assertEqual(h48.b_level, 2)
        self.assertEqual(h48.b_parent_level, 1)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(h64.b_hidden_dim, 64)
        self.assertEqual(
            h48.b_transfer_source_checkpoint,
            B1_THREAT_GUARD_DEFAULT_CHECKPOINT,
        )
        self.assertFalse(h48.b_transfer_allow_low_coverage)

    def test_diagnostic_catalog_registers_b3_contact_memory_variants(self) -> None:
        h48, strict_h48, h56, recurrent = resolve_ablation_configs(
            [
                B3_CONTACT_MEMORY_H48_POLICY_NAME,
                B3_CONTACT_MEMORY_STRICT_H48_POLICY_NAME,
                B3_CONTACT_MEMORY_H56_POLICY_NAME,
                B3_RECURRENT_GUARD_H48_POLICY_NAME,
            ]
        )

        self.assertEqual(h48.architecture, "b_series")
        self.assertEqual(h48.b_level, 3)
        self.assertEqual(h48.b_parent_level, 2)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(strict_h48.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(recurrent.b_hidden_dim, 48)
        self.assertEqual(recurrent.b_parent_level, 2)
        self.assertEqual(
            h48.b_transfer_source_checkpoint,
            B2_TEMPORAL_THREAT_DEFAULT_CHECKPOINT,
        )

    def test_diagnostic_catalog_registers_b4_recovery_variants(self) -> None:
        h48, exit_h48, h56, genetic = resolve_ablation_configs(
            [
                B4_RECOVERY_BALANCE_H48_POLICY_NAME,
                B4_PREDATOR_EXIT_MEMORY_H48_POLICY_NAME,
                B4_RECOVERY_BALANCE_H56_POLICY_NAME,
                B4_GENETIC_RECOVERY_H48_POLICY_NAME,
            ]
        )

        self.assertEqual(h48.architecture, "b_series")
        self.assertEqual(h48.b_level, 4)
        self.assertEqual(h48.b_parent_level, 3)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(exit_h48.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(genetic.b_hidden_dim, 48)
        self.assertEqual(h48.b_controller_profile, "recovery_balance")
        self.assertEqual(exit_h48.b_controller_profile, "predator_exit_memory")
        self.assertEqual(genetic.b_controller_profile, "genetic_recovery")
        tuned = replace(
            h48,
            b_controller_params={"sleep_hunger_max": 0.70},
        )
        roundtrip = BrainAblationConfig.from_summary(tuned.to_summary())
        self.assertEqual(roundtrip.b_controller_profile, "recovery_balance")
        self.assertEqual(roundtrip.b_controller_params["sleep_hunger_max"], 0.70)
        self.assertFalse(h48.b_transfer_allow_low_coverage)

    def test_diagnostic_catalog_registers_b5_homeostatic_variants(self) -> None:
        h48, circadian, h56, genetic = resolve_ablation_configs(
            [
                B5_HOMEOSTATIC_ARBITER_H48_POLICY_NAME,
                B5_CIRCADIAN_RECOVERY_H48_POLICY_NAME,
                B5_HOMEOSTATIC_ARBITER_H56_POLICY_NAME,
                B5_GENETIC_HOMEOSTASIS_H48_POLICY_NAME,
            ]
        )

        self.assertEqual(h48.architecture, "b_series")
        self.assertEqual(h48.b_level, 5)
        self.assertEqual(h48.b_parent_level, 4)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(circadian.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(genetic.b_hidden_dim, 48)
        self.assertEqual(h48.b_controller_profile, "homeostatic_arbiter")
        self.assertEqual(circadian.b_controller_profile, "circadian_recovery")
        self.assertEqual(genetic.b_controller_profile, "genetic_homeostasis")
        self.assertFalse(h48.b_transfer_allow_low_coverage)

    def test_diagnostic_catalog_registers_b6_risk_and_recurrent_variants(self) -> None:
        configs = resolve_ablation_configs(
            [
                B6_RISK_FORAGE_ARBITER_H48_POLICY_NAME,
                B6_CORRIDOR_SURVIVAL_GUARD_H48_POLICY_NAME,
                B6_THREAT_PRIORITY_MEMORY_H48_POLICY_NAME,
                B6_RISK_CORRIDOR_H56_POLICY_NAME,
                B6_GENETIC_RISK_CORRIDOR_H48_POLICY_NAME,
                B6_RECURRENT_CONTEXT_H48_POLICY_NAME,
                B6_RECURRENT_THREAT_HOMEOSTASIS_H48_POLICY_NAME,
                B6_RECURRENT_CORRIDOR_GUARD_H48_POLICY_NAME,
                B6_RECURRENT_CONTEXT_H56_POLICY_NAME,
                B6_GENETIC_RECURRENT_MEMORY_H48_POLICY_NAME,
                B6_FUSED_RISK_RECURRENT_H48_POLICY_NAME,
            ]
        )

        for config in configs:
            self.assertEqual(config.architecture, "b_series")
            self.assertEqual(config.b_level, 6)
            self.assertEqual(config.b_parent_level, 5)
            self.assertEqual(
                config.b_transfer_source_checkpoint,
                B5_GENETIC_HOMEOSTASIS_DEFAULT_CHECKPOINT,
            )
            self.assertFalse(config.b_transfer_allow_low_coverage)
            self.assertIn("b6_family", config.b_controller_params)
        self.assertEqual(configs[0].b_hidden_dim, 48)
        self.assertEqual(configs[3].b_hidden_dim, 56)
        self.assertEqual(configs[8].b_hidden_dim, 56)
        self.assertEqual(configs[-1].b_controller_profile, "fused_risk_recurrent")

    def test_diagnostic_catalog_registers_b7_affordance_budget_variants(self) -> None:
        h48, energy, recurrent, h56, genetic = resolve_ablation_configs(
            [
                B7_AFFORDANCE_BUDGET_H48_POLICY_NAME,
                B7_ENERGY_BUDGET_CORRIDOR_H48_POLICY_NAME,
                B7_RECURRENT_AFFORDANCE_H48_POLICY_NAME,
                B7_AFFORDANCE_BUDGET_H56_POLICY_NAME,
                B7_GENETIC_AFFORDANCE_BUDGET_H48_POLICY_NAME,
            ]
        )

        for config in (h48, energy, recurrent, h56, genetic):
            self.assertEqual(config.architecture, "b_series")
            self.assertEqual(config.b_level, 7)
            self.assertEqual(config.b_parent_level, 6)
            self.assertEqual(
                config.b_transfer_source_checkpoint,
                B6_FUSED_RISK_RECURRENT_DEFAULT_CHECKPOINT,
            )
            self.assertFalse(config.b_transfer_allow_low_coverage)
            self.assertIn("b7_budget_step_cost", config.b_controller_params)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(h48.b_controller_profile, "affordance_budget")
        self.assertEqual(energy.b_controller_profile, "energy_budget_corridor")
        self.assertEqual(recurrent.b_controller_profile, "recurrent_affordance")
        self.assertEqual(genetic.b_controller_profile, "genetic_affordance_budget")

    def test_diagnostic_catalog_registers_b8_spatial_affordance_variants(self) -> None:
        h48, return_vector, place_memory, h56, genetic = resolve_ablation_configs(
            [
                B8_SPATIAL_AFFORDANCE_MAP_H48_POLICY_NAME,
                B8_RETURN_VECTOR_H48_POLICY_NAME,
                B8_CORRIDOR_PLACE_MEMORY_H48_POLICY_NAME,
                B8_SPATIAL_AFFORDANCE_MAP_H56_POLICY_NAME,
                B8_GENETIC_SPATIAL_AFFORDANCE_H48_POLICY_NAME,
            ]
        )

        for config in (h48, return_vector, place_memory, h56, genetic):
            self.assertEqual(config.architecture, "b_series")
            self.assertEqual(config.b_level, 8)
            self.assertEqual(config.b_parent_level, 7)
            self.assertFalse(config.b_transfer_allow_low_coverage)
            self.assertIn("b8_place_memory_decay", config.b_controller_params)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(h48.b_controller_profile, "spatial_affordance_map")
        self.assertEqual(return_vector.b_controller_profile, "return_vector")
        self.assertEqual(place_memory.b_controller_profile, "corridor_place_memory")
        self.assertEqual(genetic.b_controller_profile, "genetic_spatial_affordance")

    def test_diagnostic_catalog_registers_b9_waypoint_variants(self) -> None:
        h48, path, route, h56, genetic = resolve_ablation_configs(
            [
                B9_WAYPOINT_PLANNER_H48_POLICY_NAME,
                B9_PATH_INTEGRATION_H48_POLICY_NAME,
                B9_ROUTE_MEMORY_H48_POLICY_NAME,
                B9_WAYPOINT_PLANNER_H56_POLICY_NAME,
                B9_GENETIC_WAYPOINT_PLANNER_H48_POLICY_NAME,
            ]
        )

        for config in (h48, path, route, h56, genetic):
            self.assertEqual(config.architecture, "b_series")
            self.assertEqual(config.b_level, 9)
            self.assertEqual(config.b_parent_level, 8)
            self.assertFalse(config.b_transfer_allow_low_coverage)
            self.assertIn("b9_route_memory_decay", config.b_controller_params)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(h48.b_controller_profile, "waypoint_planner")
        self.assertEqual(path.b_controller_profile, "path_integration")
        self.assertEqual(route.b_controller_profile, "route_memory")
        self.assertEqual(genetic.b_controller_profile, "genetic_waypoint_planner")

    def test_diagnostic_catalog_registers_b10_replay_variants(self) -> None:
        h48, value_route, replay, h56, genetic = resolve_ablation_configs(
            [
                B10_PROSPECTIVE_REPLAY_H48_POLICY_NAME,
                B10_VALUE_ROUTE_EVALUATOR_H48_POLICY_NAME,
                B10_REPLAY_PLANNER_H48_POLICY_NAME,
                B10_PROSPECTIVE_REPLAY_H56_POLICY_NAME,
                B10_GENETIC_REPLAY_PLANNER_H48_POLICY_NAME,
            ]
        )

        for config in (h48, value_route, replay, h56, genetic):
            self.assertEqual(config.architecture, "b_series")
            self.assertEqual(config.b_level, 10)
            self.assertEqual(config.b_parent_level, 9)
            self.assertFalse(config.b_transfer_allow_low_coverage)
            self.assertIn("b10_replay_memory_decay", config.b_controller_params)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(h48.b_controller_profile, "prospective_replay")
        self.assertEqual(value_route.b_controller_profile, "value_route_evaluator")
        self.assertEqual(replay.b_controller_profile, "replay_planner")
        self.assertEqual(genetic.b_controller_profile, "genetic_replay_planner")

    def test_diagnostic_catalog_registers_b11_confidence_variants(self) -> None:
        h48, uncertainty, neuromod, h56, genetic = resolve_ablation_configs(
            [
                B11_CONFIDENCE_ARBITER_H48_POLICY_NAME,
                B11_UNCERTAINTY_GATE_H48_POLICY_NAME,
                B11_NEUROMODULATED_REPLAY_H48_POLICY_NAME,
                B11_CONFIDENCE_ARBITER_H56_POLICY_NAME,
                B11_GENETIC_CONFIDENCE_GATE_H48_POLICY_NAME,
            ]
        )

        for config in (h48, uncertainty, neuromod, h56, genetic):
            self.assertEqual(config.architecture, "b_series")
            self.assertEqual(config.b_level, 11)
            self.assertEqual(config.b_parent_level, 10)
            self.assertFalse(config.b_transfer_allow_low_coverage)
            self.assertIn("b11_confidence_decay", config.b_controller_params)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(h48.b_controller_profile, "confidence_arbiter")
        self.assertEqual(uncertainty.b_controller_profile, "uncertainty_gate")
        self.assertEqual(neuromod.b_controller_profile, "neuromodulated_replay")
        self.assertEqual(genetic.b_controller_profile, "genetic_confidence_gate")

    def test_diagnostic_catalog_registers_b12_attention_variants(self) -> None:
        h48, active, affordance, h56, genetic = resolve_ablation_configs(
            [
                B12_PREDICTIVE_ATTENTION_H48_POLICY_NAME,
                B12_ACTIVE_INFERENCE_GATE_H48_POLICY_NAME,
                B12_AFFORDANCE_ATTENTION_H48_POLICY_NAME,
                B12_PREDICTIVE_ATTENTION_H56_POLICY_NAME,
                B12_GENETIC_ATTENTION_GATE_H48_POLICY_NAME,
            ]
        )

        for config in (h48, active, affordance, h56, genetic):
            self.assertEqual(config.architecture, "b_series")
            self.assertEqual(config.b_level, 12)
            self.assertEqual(config.b_parent_level, 11)
            self.assertFalse(config.b_transfer_allow_low_coverage)
            self.assertIn("b12_attention_decay", config.b_controller_params)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(h48.b_controller_profile, "predictive_attention")
        self.assertEqual(active.b_controller_profile, "active_inference_gate")
        self.assertEqual(affordance.b_controller_profile, "affordance_attention")
        self.assertEqual(genetic.b_controller_profile, "genetic_attention_gate")

    def test_diagnostic_catalog_registers_b13_local_search_variants(self) -> None:
        h48, counterfactual, sampler, h56, genetic = resolve_ablation_configs(
            [
                B13_LOCAL_AFFORDANCE_SEARCH_H48_POLICY_NAME,
                B13_COUNTERFACTUAL_ROUTE_H48_POLICY_NAME,
                B13_AFFORDANCE_SAMPLER_H48_POLICY_NAME,
                B13_LOCAL_AFFORDANCE_SEARCH_H56_POLICY_NAME,
                B13_GENETIC_LOCAL_SEARCH_H48_POLICY_NAME,
            ]
        )

        for config in (h48, counterfactual, sampler, h56, genetic):
            self.assertEqual(config.architecture, "b_series")
            self.assertEqual(config.b_level, 13)
            self.assertEqual(config.b_parent_level, 12)
            self.assertFalse(config.b_transfer_allow_low_coverage)
            self.assertIn("b13_search_memory_decay", config.b_controller_params)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(h48.b_controller_profile, "local_affordance_search")
        self.assertEqual(counterfactual.b_controller_profile, "counterfactual_route")
        self.assertEqual(sampler.b_controller_profile, "affordance_sampler")
        self.assertEqual(genetic.b_controller_profile, "genetic_local_search")

    def test_diagnostic_catalog_registers_b14_uncertainty_variants(self) -> None:
        h48, risk, confidence, h56, genetic = resolve_ablation_configs(
            [
                B14_AFFORDANCE_UNCERTAINTY_H48_POLICY_NAME,
                B14_RISK_CALIBRATED_SEARCH_H48_POLICY_NAME,
                B14_CONFIDENCE_WEIGHTED_ROUTE_H48_POLICY_NAME,
                B14_AFFORDANCE_UNCERTAINTY_H56_POLICY_NAME,
                B14_GENETIC_UNCERTAINTY_SEARCH_H48_POLICY_NAME,
            ]
        )

        for config in (h48, risk, confidence, h56, genetic):
            self.assertEqual(config.architecture, "b_series")
            self.assertEqual(config.b_level, 14)
            self.assertEqual(config.b_parent_level, 13)
            self.assertFalse(config.b_transfer_allow_low_coverage)
            self.assertIn("b14_uncertainty_decay", config.b_controller_params)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(h48.b_controller_profile, "affordance_uncertainty")
        self.assertEqual(risk.b_controller_profile, "risk_calibrated_search")
        self.assertEqual(confidence.b_controller_profile, "confidence_weighted_route")
        self.assertEqual(genetic.b_controller_profile, "genetic_uncertainty_search")

    def test_diagnostic_catalog_registers_b15_option_variants(self) -> None:
        h48, persistence, value_gated, h56, genetic = resolve_ablation_configs(
            [
                B15_OPTION_CRITIC_H48_POLICY_NAME,
                B15_PERSISTENCE_GATE_H48_POLICY_NAME,
                B15_VALUE_GATED_OPTION_H48_POLICY_NAME,
                B15_OPTION_CRITIC_H56_POLICY_NAME,
                B15_GENETIC_OPTION_CRITIC_H48_POLICY_NAME,
            ]
        )

        for config in (h48, persistence, value_gated, h56, genetic):
            self.assertEqual(config.architecture, "b_series")
            self.assertEqual(config.b_level, 15)
            self.assertEqual(config.b_parent_level, 14)
            self.assertFalse(config.b_transfer_allow_low_coverage)
            self.assertIn("b15_option_memory_decay", config.b_controller_params)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(h48.b_controller_profile, "option_critic")
        self.assertEqual(persistence.b_controller_profile, "persistence_gate")
        self.assertEqual(value_gated.b_controller_profile, "value_gated_option")
        self.assertEqual(genetic.b_controller_profile, "genetic_option_critic")

    def test_diagnostic_catalog_registers_b16_ensemble_variants(self) -> None:
        h48, competing, voter, h56, genetic = resolve_ablation_configs(
            [
                B16_OPTION_ENSEMBLE_H48_POLICY_NAME,
                B16_COMPETING_OPTIONS_H48_POLICY_NAME,
                B16_ACTION_SET_VOTER_H48_POLICY_NAME,
                B16_OPTION_ENSEMBLE_H56_POLICY_NAME,
                B16_GENETIC_OPTION_ENSEMBLE_H48_POLICY_NAME,
            ]
        )

        for config in (h48, competing, voter, h56, genetic):
            self.assertEqual(config.architecture, "b_series")
            self.assertEqual(config.b_level, 16)
            self.assertEqual(config.b_parent_level, 15)
            self.assertFalse(config.b_transfer_allow_low_coverage)
            self.assertIn("b16_ensemble_decay", config.b_controller_params)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(h48.b_controller_profile, "option_ensemble")
        self.assertEqual(competing.b_controller_profile, "competing_options")
        self.assertEqual(voter.b_controller_profile, "action_set_voter")
        self.assertEqual(genetic.b_controller_profile, "genetic_option_ensemble")

    def test_diagnostic_catalog_registers_b17_neuromodulated_variants(self) -> None:
        h48, arousal, homeostasis, h56, genetic = resolve_ablation_configs(
            [
                B17_NEUROMODULATED_ENSEMBLE_H48_POLICY_NAME,
                B17_AROUSAL_GATED_OPTIONS_H48_POLICY_NAME,
                B17_HOMEOSTATIC_MODULATOR_H48_POLICY_NAME,
                B17_NEUROMODULATED_ENSEMBLE_H56_POLICY_NAME,
                B17_GENETIC_NEUROMODULATED_ENSEMBLE_H48_POLICY_NAME,
            ]
        )

        for config in (h48, arousal, homeostasis, h56, genetic):
            self.assertEqual(config.architecture, "b_series")
            self.assertEqual(config.b_level, 17)
            self.assertEqual(config.b_parent_level, 16)
            self.assertFalse(config.b_transfer_allow_low_coverage)
            self.assertIn("b17_arousal_decay", config.b_controller_params)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(h48.b_controller_profile, "neuromodulated_ensemble")
        self.assertEqual(arousal.b_controller_profile, "arousal_gated_options")
        self.assertEqual(homeostasis.b_controller_profile, "homeostatic_modulator")
        self.assertEqual(genetic.b_controller_profile, "genetic_neuromodulated_ensemble")

    def test_diagnostic_catalog_registers_b18_eligibility_variants(self) -> None:
        h48, metastable, synaptic, h56, genetic = resolve_ablation_configs(
            [
                B18_ELIGIBILITY_TRACE_H48_POLICY_NAME,
                B18_METASTABLE_AROUSAL_H48_POLICY_NAME,
                B18_SYNAPTIC_TRACE_MODULATOR_H48_POLICY_NAME,
                B18_ELIGIBILITY_TRACE_H56_POLICY_NAME,
                B18_GENETIC_ELIGIBILITY_TRACE_H48_POLICY_NAME,
            ]
        )

        for config in (h48, metastable, synaptic, h56, genetic):
            self.assertEqual(config.architecture, "b_series")
            self.assertEqual(config.b_level, 18)
            self.assertEqual(config.b_parent_level, 17)
            self.assertFalse(config.b_transfer_allow_low_coverage)
            self.assertIn("b18_trace_decay", config.b_controller_params)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(h48.b_controller_profile, "eligibility_trace")
        self.assertEqual(metastable.b_controller_profile, "metastable_arousal")
        self.assertEqual(synaptic.b_controller_profile, "synaptic_trace_modulator")
        self.assertEqual(genetic.b_controller_profile, "genetic_eligibility_trace")

    def test_diagnostic_catalog_registers_b19_meta_memory_variants(self) -> None:
        h48, stability, suppression, h56, genetic = resolve_ablation_configs(
            [
                B19_EPISODIC_META_MEMORY_H48_POLICY_NAME,
                B19_STABILITY_MEMORY_H48_POLICY_NAME,
                B19_SWITCH_SUPPRESSION_H48_POLICY_NAME,
                B19_EPISODIC_META_MEMORY_H56_POLICY_NAME,
                B19_GENETIC_META_MEMORY_H48_POLICY_NAME,
            ]
        )

        for config in (h48, stability, suppression, h56, genetic):
            self.assertEqual(config.architecture, "b_series")
            self.assertEqual(config.b_level, 19)
            self.assertEqual(config.b_parent_level, 18)
            self.assertFalse(config.b_transfer_allow_low_coverage)
            self.assertIn("b19_memory_decay", config.b_controller_params)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(h48.b_controller_profile, "episodic_meta_memory")
        self.assertEqual(stability.b_controller_profile, "stability_memory")
        self.assertEqual(suppression.b_controller_profile, "switch_suppression")
        self.assertEqual(genetic.b_controller_profile, "genetic_meta_memory")

    def test_diagnostic_catalog_registers_b20_working_memory_variants(self) -> None:
        h48, context, stability, h56, genetic = resolve_ablation_configs(
            [
                B20_WORKING_MEMORY_GATE_H48_POLICY_NAME,
                B20_CONTEXT_BINDING_H48_POLICY_NAME,
                B20_STABILITY_BUFFER_H48_POLICY_NAME,
                B20_WORKING_MEMORY_GATE_H56_POLICY_NAME,
                B20_GENETIC_WORKING_MEMORY_H48_POLICY_NAME,
            ]
        )

        for config in (h48, context, stability, h56, genetic):
            self.assertEqual(config.architecture, "b_series")
            self.assertEqual(config.b_level, 20)
            self.assertEqual(config.b_parent_level, 19)
            self.assertFalse(config.b_transfer_allow_low_coverage)
            self.assertIn("b20_buffer_decay", config.b_controller_params)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(h48.b_controller_profile, "working_memory_gate")
        self.assertEqual(context.b_controller_profile, "context_binding")
        self.assertEqual(stability.b_controller_profile, "stability_buffer")
        self.assertEqual(genetic.b_controller_profile, "genetic_working_memory")

    def test_diagnostic_catalog_registers_b21_replay_variants(self) -> None:
        h48, sequence, route, h56, genetic = resolve_ablation_configs(
            [
                B21_HIPPOCAMPAL_REPLAY_H48_POLICY_NAME,
                B21_SEQUENCE_BINDING_H48_POLICY_NAME,
                B21_ROUTE_REHEARSAL_H48_POLICY_NAME,
                B21_HIPPOCAMPAL_REPLAY_H56_POLICY_NAME,
                B21_GENETIC_REPLAY_GATE_H48_POLICY_NAME,
            ]
        )

        for config in (h48, sequence, route, h56, genetic):
            self.assertEqual(config.architecture, "b_series")
            self.assertEqual(config.b_level, 21)
            self.assertEqual(config.b_parent_level, 20)
            self.assertFalse(config.b_transfer_allow_low_coverage)
            self.assertIn("b21_replay_decay", config.b_controller_params)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(h48.b_controller_profile, "hippocampal_replay")
        self.assertEqual(sequence.b_controller_profile, "sequence_binding")
        self.assertEqual(route.b_controller_profile, "route_rehearsal")
        self.assertEqual(genetic.b_controller_profile, "genetic_replay_gate")

    def test_diagnostic_catalog_registers_b22_prospective_variants(self) -> None:
        h48, forward, route, h56, genetic = resolve_ablation_configs(
            [
                B22_PROSPECTIVE_MAP_REPLAY_H48_POLICY_NAME,
                B22_FORWARD_MODEL_GATE_H48_POLICY_NAME,
                B22_ROUTE_VIABILITY_SIM_H48_POLICY_NAME,
                B22_PROSPECTIVE_MAP_REPLAY_H56_POLICY_NAME,
                B22_GENETIC_PROSPECTIVE_REPLAY_H48_POLICY_NAME,
            ]
        )

        for config in (h48, forward, route, h56, genetic):
            self.assertEqual(config.architecture, "b_series")
            self.assertEqual(config.b_level, 22)
            self.assertEqual(config.b_parent_level, 21)
            self.assertFalse(config.b_transfer_allow_low_coverage)
            self.assertIn("b22_sim_decay", config.b_controller_params)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(h48.b_controller_profile, "prospective_map_replay")
        self.assertEqual(forward.b_controller_profile, "forward_model_gate")
        self.assertEqual(route.b_controller_profile, "route_viability_sim")
        self.assertEqual(genetic.b_controller_profile, "genetic_prospective_replay")

    def test_diagnostic_catalog_registers_b23_conflict_variants(self) -> None:
        h48, error, abort, h56, genetic = resolve_ablation_configs(
            [
                B23_CONFLICT_MONITOR_H48_POLICY_NAME,
                B23_ERROR_GATED_REPLAY_H48_POLICY_NAME,
                B23_ABORT_CONFLICT_ARBITER_H48_POLICY_NAME,
                B23_CONFLICT_MONITOR_H56_POLICY_NAME,
                B23_GENETIC_CONFLICT_MONITOR_H48_POLICY_NAME,
            ]
        )

        for config in (h48, error, abort, h56, genetic):
            self.assertEqual(config.architecture, "b_series")
            self.assertEqual(config.b_level, 23)
            self.assertEqual(config.b_parent_level, 22)
            self.assertFalse(config.b_transfer_allow_low_coverage)
            self.assertIn("b23_conflict_decay", config.b_controller_params)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(h48.b_controller_profile, "conflict_monitor")
        self.assertEqual(error.b_controller_profile, "error_gated_replay")
        self.assertEqual(abort.b_controller_profile, "abort_conflict_arbiter")
        self.assertEqual(genetic.b_controller_profile, "genetic_conflict_monitor")

    def test_diagnostic_catalog_registers_b24_precision_variants(self) -> None:
        h48, precision, abort, h56, genetic = resolve_ablation_configs(
            [
                B24_PRECISION_CONFLICT_H48_POLICY_NAME,
                B24_PREDICTION_PRECISION_GATE_H48_POLICY_NAME,
                B24_RELIABILITY_ABORT_H48_POLICY_NAME,
                B24_PRECISION_CONFLICT_H56_POLICY_NAME,
                B24_GENETIC_PRECISION_CONFLICT_H48_POLICY_NAME,
            ]
        )

        for config in (h48, precision, abort, h56, genetic):
            self.assertEqual(config.architecture, "b_series")
            self.assertEqual(config.b_level, 24)
            self.assertEqual(config.b_parent_level, 23)
            self.assertFalse(config.b_transfer_allow_low_coverage)
            self.assertIn("b24_precision_decay", config.b_controller_params)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(h48.b_controller_profile, "precision_conflict")
        self.assertEqual(precision.b_controller_profile, "prediction_precision_gate")
        self.assertEqual(abort.b_controller_profile, "reliability_abort")
        self.assertEqual(genetic.b_controller_profile, "genetic_precision_conflict")

    def test_diagnostic_catalog_registers_b25_metacognitive_variants(self) -> None:
        h48, calibration, integrator, h56, genetic = resolve_ablation_configs(
            [
                B25_METACOGNITIVE_CONFIDENCE_H48_POLICY_NAME,
                B25_CONFIDENCE_CALIBRATION_H48_POLICY_NAME,
                B25_UNCERTAINTY_INTEGRATOR_H48_POLICY_NAME,
                B25_METACOGNITIVE_CONFIDENCE_H56_POLICY_NAME,
                B25_GENETIC_METACOGNITION_H48_POLICY_NAME,
            ]
        )

        for config in (h48, calibration, integrator, h56, genetic):
            self.assertEqual(config.architecture, "b_series")
            self.assertEqual(config.b_level, 25)
            self.assertEqual(config.b_parent_level, 24)
            self.assertFalse(config.b_transfer_allow_low_coverage)
            self.assertIn("b25_confidence_decay", config.b_controller_params)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(h48.b_controller_profile, "metacognitive_confidence")
        self.assertEqual(calibration.b_controller_profile, "confidence_calibration")
        self.assertEqual(integrator.b_controller_profile, "uncertainty_integrator")
        self.assertEqual(genetic.b_controller_profile, "genetic_metacognition")

    def test_diagnostic_catalog_registers_b26_allostatic_variants(self) -> None:
        h48, drift, suppression, h56, genetic = resolve_ablation_configs(
            [
                B26_ALLOSTATIC_PREDICTION_H48_POLICY_NAME,
                B26_SETPOINT_DRIFT_H48_POLICY_NAME,
                B26_ERROR_SUPPRESSION_H48_POLICY_NAME,
                B26_ALLOSTATIC_PREDICTION_H56_POLICY_NAME,
                B26_GENETIC_ALLOSTASIS_H48_POLICY_NAME,
            ]
        )

        for config in (h48, drift, suppression, h56, genetic):
            self.assertEqual(config.architecture, "b_series")
            self.assertEqual(config.b_level, 26)
            self.assertEqual(config.b_parent_level, 25)
            self.assertFalse(config.b_transfer_allow_low_coverage)
            self.assertIn("b26_error_decay", config.b_controller_params)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(h48.b_controller_profile, "allostatic_prediction")
        self.assertEqual(drift.b_controller_profile, "setpoint_drift")
        self.assertEqual(suppression.b_controller_profile, "error_suppression")
        self.assertEqual(genetic.b_controller_profile, "genetic_allostasis")

    def test_diagnostic_catalog_registers_b27_arousal_variants(self) -> None:
        h48, stress, energy, h56, genetic = resolve_ablation_configs(
            [
                B27_AROUSAL_GAIN_H48_POLICY_NAME,
                B27_STRESS_MODULATION_H48_POLICY_NAME,
                B27_ENERGY_AROUSAL_H48_POLICY_NAME,
                B27_AROUSAL_GAIN_H56_POLICY_NAME,
                B27_GENETIC_AROUSAL_H48_POLICY_NAME,
            ]
        )

        for config in (h48, stress, energy, h56, genetic):
            self.assertEqual(config.architecture, "b_series")
            self.assertEqual(config.b_level, 27)
            self.assertEqual(config.b_parent_level, 26)
            self.assertFalse(config.b_transfer_allow_low_coverage)
            self.assertIn("b27_arousal_decay", config.b_controller_params)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(h48.b_controller_profile, "arousal_gain")
        self.assertEqual(stress.b_controller_profile, "stress_modulation")
        self.assertEqual(energy.b_controller_profile, "energy_arousal")
        self.assertEqual(genetic.b_controller_profile, "genetic_arousal")

    def test_diagnostic_catalog_registers_b28_attention_variants(self) -> None:
        h48, threat, homeostatic, h56, genetic = resolve_ablation_configs(
            [
                B28_INTEROCEPTIVE_ATTENTION_H48_POLICY_NAME,
                B28_THREAT_FOCUS_ATTENTION_H48_POLICY_NAME,
                B28_HOMEOSTATIC_ATTENTION_H48_POLICY_NAME,
                B28_INTEROCEPTIVE_ATTENTION_H56_POLICY_NAME,
                B28_GENETIC_ATTENTION_H48_POLICY_NAME,
            ]
        )

        for config in (h48, threat, homeostatic, h56, genetic):
            self.assertEqual(config.architecture, "b_series")
            self.assertEqual(config.b_level, 28)
            self.assertEqual(config.b_parent_level, 27)
            self.assertFalse(config.b_transfer_allow_low_coverage)
            self.assertIn("b28_attention_decay", config.b_controller_params)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(h48.b_controller_profile, "interoceptive_attention")
        self.assertEqual(threat.b_controller_profile, "threat_focus_attention")
        self.assertEqual(homeostatic.b_controller_profile, "homeostatic_attention")
        self.assertEqual(genetic.b_controller_profile, "genetic_attention")

    def test_diagnostic_catalog_registers_b29_salience_variants(self) -> None:
        h48, threat, homeostatic, h56, genetic = resolve_ablation_configs(
            [
                B29_SALIENCE_COMPETITION_H48_POLICY_NAME,
                B29_THREAT_SALIENCE_GATE_H48_POLICY_NAME,
                B29_HOMEOSTATIC_SALIENCE_GATE_H48_POLICY_NAME,
                B29_SALIENCE_COMPETITION_H56_POLICY_NAME,
                B29_GENETIC_SALIENCE_H48_POLICY_NAME,
            ]
        )

        for config in (h48, threat, homeostatic, h56, genetic):
            self.assertEqual(config.architecture, "b_series")
            self.assertEqual(config.b_level, 29)
            self.assertEqual(config.b_parent_level, 28)
            self.assertFalse(config.b_transfer_allow_low_coverage)
            self.assertIn("b29_salience_decay", config.b_controller_params)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(h48.b_controller_profile, "salience_competition")
        self.assertEqual(threat.b_controller_profile, "threat_salience_gate")
        self.assertEqual(homeostatic.b_controller_profile, "homeostatic_salience_gate")
        self.assertEqual(genetic.b_controller_profile, "genetic_salience")

    def test_diagnostic_catalog_registers_b30_gate_variants(self) -> None:
        h48, balance, inhibition, h56, genetic = resolve_ablation_configs(
            [
                B30_BASAL_GANGLIA_GATE_H48_POLICY_NAME,
                B30_GO_NOGO_BALANCE_H48_POLICY_NAME,
                B30_THREAT_INHIBITION_GATE_H48_POLICY_NAME,
                B30_BASAL_GANGLIA_GATE_H56_POLICY_NAME,
                B30_GENETIC_ACTION_GATE_H48_POLICY_NAME,
            ]
        )

        for config in (h48, balance, inhibition, h56, genetic):
            self.assertEqual(config.architecture, "b_series")
            self.assertEqual(config.b_level, 30)
            self.assertEqual(config.b_parent_level, 29)
            self.assertFalse(config.b_transfer_allow_low_coverage)
            self.assertIn("b30_gate_decay", config.b_controller_params)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(h48.b_controller_profile, "basal_ganglia_gate")
        self.assertEqual(balance.b_controller_profile, "go_nogo_balance")
        self.assertEqual(inhibition.b_controller_profile, "threat_inhibition_gate")
        self.assertEqual(genetic.b_controller_profile, "genetic_action_gate")

    def test_diagnostic_catalog_registers_b31_dopamine_variants(self) -> None:
        h48, tonic, phasic, h56, genetic = resolve_ablation_configs(
            [
                B31_DOPAMINE_PREDICTION_ERROR_H48_POLICY_NAME,
                B31_TONIC_DOPAMINE_GATE_H48_POLICY_NAME,
                B31_PHASIC_DOPAMINE_GATE_H48_POLICY_NAME,
                B31_DOPAMINE_PREDICTION_ERROR_H56_POLICY_NAME,
                B31_GENETIC_DOPAMINE_GATE_H48_POLICY_NAME,
            ]
        )

        for config in (h48, tonic, phasic, h56, genetic):
            self.assertEqual(config.architecture, "b_series")
            self.assertEqual(config.b_level, 31)
            self.assertEqual(config.b_parent_level, 30)
            self.assertFalse(config.b_transfer_allow_low_coverage)
            self.assertIn("b31_dopamine_decay", config.b_controller_params)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(h48.b_controller_profile, "dopamine_prediction_error")
        self.assertEqual(tonic.b_controller_profile, "tonic_dopamine_gate")
        self.assertEqual(phasic.b_controller_profile, "phasic_dopamine_gate")
        self.assertEqual(genetic.b_controller_profile, "genetic_dopamine_gate")

    def test_diagnostic_catalog_registers_b32_actor_critic_variants(self) -> None:
        h48, advantage, stability, h56, genetic = resolve_ablation_configs(
            [
                B32_ACTOR_CRITIC_VALUE_H48_POLICY_NAME,
                B32_ADVANTAGE_VALUE_GATE_H48_POLICY_NAME,
                B32_CRITIC_STABILITY_H48_POLICY_NAME,
                B32_ACTOR_CRITIC_VALUE_H56_POLICY_NAME,
                B32_GENETIC_ACTOR_CRITIC_H48_POLICY_NAME,
            ]
        )

        for config in (h48, advantage, stability, h56, genetic):
            self.assertEqual(config.architecture, "b_series")
            self.assertEqual(config.b_level, 32)
            self.assertEqual(config.b_parent_level, 31)
            self.assertFalse(config.b_transfer_allow_low_coverage)
            self.assertIn("b32_value_decay", config.b_controller_params)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(h48.b_controller_profile, "actor_critic_value")
        self.assertEqual(advantage.b_controller_profile, "advantage_value_gate")
        self.assertEqual(stability.b_controller_profile, "critic_stability")
        self.assertEqual(genetic.b_controller_profile, "genetic_actor_critic")

    def test_diagnostic_catalog_registers_b33_td_error_variants(self) -> None:
        h48, bootstrap, reward_trace, h56, genetic = resolve_ablation_configs(
            [
                B33_TD_ERROR_DECOMPOSITION_H48_POLICY_NAME,
                B33_BOOTSTRAPPED_VALUE_GATE_H48_POLICY_NAME,
                B33_REWARD_TRACE_CRITIC_H48_POLICY_NAME,
                B33_TD_ERROR_DECOMPOSITION_H56_POLICY_NAME,
                B33_GENETIC_TD_VALUE_H48_POLICY_NAME,
            ]
        )

        for config in (h48, bootstrap, reward_trace, h56, genetic):
            self.assertEqual(config.architecture, "b_series")
            self.assertEqual(config.b_level, 33)
            self.assertEqual(config.b_parent_level, 32)
            self.assertFalse(config.b_transfer_allow_low_coverage)
            self.assertIn("b33_td_decay", config.b_controller_params)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(h48.b_controller_profile, "td_error_decomposition")
        self.assertEqual(bootstrap.b_controller_profile, "bootstrapped_value_gate")
        self.assertEqual(reward_trace.b_controller_profile, "reward_trace_critic")
        self.assertEqual(genetic.b_controller_profile, "genetic_td_value")

    def test_diagnostic_catalog_registers_b34_eligibility_variants(self) -> None:
        h48, delayed, tagging, h56, genetic = resolve_ablation_configs(
            [
                B34_ELIGIBILITY_CREDIT_H48_POLICY_NAME,
                B34_DELAYED_CREDIT_GATE_H48_POLICY_NAME,
                B34_SYNAPTIC_TAGGING_H48_POLICY_NAME,
                B34_ELIGIBILITY_CREDIT_H56_POLICY_NAME,
                B34_GENETIC_ELIGIBILITY_H48_POLICY_NAME,
            ]
        )

        for config in (h48, delayed, tagging, h56, genetic):
            self.assertEqual(config.architecture, "b_series")
            self.assertEqual(config.b_level, 34)
            self.assertEqual(config.b_parent_level, 33)
            self.assertFalse(config.b_transfer_allow_low_coverage)
            self.assertIn("b34_eligibility_decay", config.b_controller_params)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(h48.b_controller_profile, "eligibility_credit")
        self.assertEqual(delayed.b_controller_profile, "delayed_credit_gate")
        self.assertEqual(tagging.b_controller_profile, "synaptic_tagging")
        self.assertEqual(genetic.b_controller_profile, "genetic_eligibility")

    def test_diagnostic_catalog_registers_b35_forward_model_variants(self) -> None:
        h48, transition, confidence, h56, genetic = resolve_ablation_configs(
            [
                B35_FORWARD_MODEL_VALUE_H48_POLICY_NAME,
                B35_TRANSITION_ERROR_GATE_H48_POLICY_NAME,
                B35_MODEL_CONFIDENCE_H48_POLICY_NAME,
                B35_FORWARD_MODEL_VALUE_H56_POLICY_NAME,
                B35_GENETIC_FORWARD_MODEL_H48_POLICY_NAME,
            ]
        )

        for config in (h48, transition, confidence, h56, genetic):
            self.assertEqual(config.architecture, "b_series")
            self.assertEqual(config.b_level, 35)
            self.assertEqual(config.b_parent_level, 34)
            self.assertFalse(config.b_transfer_allow_low_coverage)
            self.assertIn("b35_model_decay", config.b_controller_params)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(h48.b_controller_profile, "forward_model_value")
        self.assertEqual(transition.b_controller_profile, "transition_error_gate")
        self.assertEqual(confidence.b_controller_profile, "model_confidence")
        self.assertEqual(genetic.b_controller_profile, "genetic_forward_model")

    def test_diagnostic_catalog_registers_b36_belief_state_variants(self) -> None:
        h48, belief_error, context, h56, genetic = resolve_ablation_configs(
            [
                B36_LATENT_BELIEF_STATE_H48_POLICY_NAME,
                B36_BELIEF_ERROR_GATE_H48_POLICY_NAME,
                B36_CONTEXT_INFERENCE_H48_POLICY_NAME,
                B36_LATENT_BELIEF_STATE_H56_POLICY_NAME,
                B36_GENETIC_BELIEF_STATE_H48_POLICY_NAME,
            ]
        )

        for config in (h48, belief_error, context, h56, genetic):
            self.assertEqual(config.architecture, "b_series")
            self.assertEqual(config.b_level, 36)
            self.assertEqual(config.b_parent_level, 35)
            self.assertFalse(config.b_transfer_allow_low_coverage)
            self.assertIn("b36_belief_decay", config.b_controller_params)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(h48.b_controller_profile, "latent_belief_state")
        self.assertEqual(belief_error.b_controller_profile, "belief_error_gate")
        self.assertEqual(context.b_controller_profile, "context_inference")
        self.assertEqual(genetic.b_controller_profile, "genetic_belief_state")

    def test_diagnostic_catalog_registers_b37_state_factor_variants(self) -> None:
        h48, intero_extero, confidence, h56, genetic = resolve_ablation_configs(
            [
                B37_STATE_FACTOR_GATE_H48_POLICY_NAME,
                B37_INTERO_EXTERO_FACTOR_H48_POLICY_NAME,
                B37_FACTOR_CONFIDENCE_H48_POLICY_NAME,
                B37_STATE_FACTOR_GATE_H56_POLICY_NAME,
                B37_GENETIC_STATE_FACTOR_H48_POLICY_NAME,
            ]
        )

        for config in (h48, intero_extero, confidence, h56, genetic):
            self.assertEqual(config.architecture, "b_series")
            self.assertEqual(config.b_level, 37)
            self.assertEqual(config.b_parent_level, 36)
            self.assertFalse(config.b_transfer_allow_low_coverage)
            self.assertIn("b37_factor_decay", config.b_controller_params)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(h48.b_controller_profile, "state_factor_gate")
        self.assertEqual(intero_extero.b_controller_profile, "intero_extero_factor")
        self.assertEqual(confidence.b_controller_profile, "factor_confidence")
        self.assertEqual(genetic.b_controller_profile, "genetic_state_factor")

    def test_diagnostic_catalog_registers_b38_factor_attention_variants(self) -> None:
        h48, interoceptive, confidence, h56, genetic = resolve_ablation_configs(
            [
                B38_FACTOR_ATTENTION_H48_POLICY_NAME,
                B38_INTEROCEPTIVE_ATTENTION_H48_POLICY_NAME,
                B38_CONFIDENCE_ATTENTION_H48_POLICY_NAME,
                B38_FACTOR_ATTENTION_H56_POLICY_NAME,
                B38_GENETIC_FACTOR_ATTENTION_H48_POLICY_NAME,
            ]
        )

        for config in (h48, interoceptive, confidence, h56, genetic):
            self.assertEqual(config.architecture, "b_series")
            self.assertEqual(config.b_level, 38)
            self.assertEqual(config.b_parent_level, 37)
            self.assertFalse(config.b_transfer_allow_low_coverage)
            self.assertIn("b38_attention_decay", config.b_controller_params)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(h48.b_controller_profile, "factor_attention")
        self.assertEqual(interoceptive.b_controller_profile, "interoceptive_attention")
        self.assertEqual(confidence.b_controller_profile, "confidence_attention")
        self.assertEqual(genetic.b_controller_profile, "genetic_factor_attention")

    def test_diagnostic_catalog_registers_b39_attention_binding_variants(self) -> None:
        h48, cross_factor, context, h56, genetic = resolve_ablation_configs(
            [
                B39_ATTENTION_BINDING_H48_POLICY_NAME,
                B39_CROSS_FACTOR_BINDING_H48_POLICY_NAME,
                B39_CONTEXT_BINDING_ATTENTION_H48_POLICY_NAME,
                B39_ATTENTION_BINDING_H56_POLICY_NAME,
                B39_GENETIC_ATTENTION_BINDING_H48_POLICY_NAME,
            ]
        )

        for config in (h48, cross_factor, context, h56, genetic):
            self.assertEqual(config.architecture, "b_series")
            self.assertEqual(config.b_level, 39)
            self.assertEqual(config.b_parent_level, 38)
            self.assertFalse(config.b_transfer_allow_low_coverage)
            self.assertIn("b39_binding_decay", config.b_controller_params)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(h48.b_controller_profile, "attention_binding")
        self.assertEqual(cross_factor.b_controller_profile, "cross_factor_binding")
        self.assertEqual(context.b_controller_profile, "context_binding_attention")
        self.assertEqual(genetic.b_controller_profile, "genetic_attention_binding")

    def test_diagnostic_catalog_registers_b40_global_workspace_variants(self) -> None:
        h48, sensory, context, h56, genetic = resolve_ablation_configs(
            [
                B40_GLOBAL_WORKSPACE_H48_POLICY_NAME,
                B40_SENSORY_WORKSPACE_H48_POLICY_NAME,
                B40_CONTEXT_WORKSPACE_H48_POLICY_NAME,
                B40_GLOBAL_WORKSPACE_H56_POLICY_NAME,
                B40_GENETIC_GLOBAL_WORKSPACE_H48_POLICY_NAME,
            ]
        )

        for config in (h48, sensory, context, h56, genetic):
            self.assertEqual(config.architecture, "b_series")
            self.assertEqual(config.b_level, 40)
            self.assertEqual(config.b_parent_level, 39)
            self.assertFalse(config.b_transfer_allow_low_coverage)
            self.assertIn("b40_workspace_decay", config.b_controller_params)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(h48.b_controller_profile, "global_workspace")
        self.assertEqual(sensory.b_controller_profile, "sensory_workspace")
        self.assertEqual(context.b_controller_profile, "context_workspace")
        self.assertEqual(genetic.b_controller_profile, "genetic_global_workspace")

    def test_diagnostic_catalog_registers_b41_executive_workspace_variants(self) -> None:
        h48, inhibitory, selector, h56, genetic = resolve_ablation_configs(
            [
                B41_EXECUTIVE_WORKSPACE_H48_POLICY_NAME,
                B41_INHIBITORY_CONTROL_H48_POLICY_NAME,
                B41_GOAL_CONTEXT_SELECTOR_H48_POLICY_NAME,
                B41_EXECUTIVE_WORKSPACE_H56_POLICY_NAME,
                B41_GENETIC_EXECUTIVE_WORKSPACE_H48_POLICY_NAME,
            ]
        )

        for config in (h48, inhibitory, selector, h56, genetic):
            self.assertEqual(config.architecture, "b_series")
            self.assertEqual(config.b_level, 41)
            self.assertEqual(config.b_parent_level, 40)
            self.assertFalse(config.b_transfer_allow_low_coverage)
            self.assertIn("b41_executive_decay", config.b_controller_params)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(h48.b_controller_profile, "executive_workspace")
        self.assertEqual(inhibitory.b_controller_profile, "inhibitory_control")
        self.assertEqual(selector.b_controller_profile, "goal_context_selector")
        self.assertEqual(genetic.b_controller_profile, "genetic_executive_workspace")

    def test_diagnostic_catalog_registers_b42_error_monitor_variants(self) -> None:
        h48, conflict, performance, h56, genetic = resolve_ablation_configs(
            [
                B42_ERROR_MONITOR_H48_POLICY_NAME,
                B42_CONFLICT_MONITOR_H48_POLICY_NAME,
                B42_PERFORMANCE_MONITOR_H48_POLICY_NAME,
                B42_ERROR_MONITOR_H56_POLICY_NAME,
                B42_GENETIC_ERROR_MONITOR_H48_POLICY_NAME,
            ]
        )

        for config in (h48, conflict, performance, h56, genetic):
            self.assertEqual(config.architecture, "b_series")
            self.assertEqual(config.b_level, 42)
            self.assertEqual(config.b_parent_level, 41)
            self.assertFalse(config.b_transfer_allow_low_coverage)
            self.assertIn("b42_monitor_decay", config.b_controller_params)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(h48.b_controller_profile, "error_monitor")
        self.assertEqual(conflict.b_controller_profile, "conflict_monitor")
        self.assertEqual(performance.b_controller_profile, "performance_monitor")
        self.assertEqual(genetic.b_controller_profile, "genetic_error_monitor")

    def test_diagnostic_catalog_registers_b43_adaptive_precision_variants(self) -> None:
        h48, arousal, threshold, h56, genetic = resolve_ablation_configs(
            [
                B43_ADAPTIVE_PRECISION_H48_POLICY_NAME,
                B43_AROUSAL_PRECISION_H48_POLICY_NAME,
                B43_THRESHOLD_ADAPTATION_H48_POLICY_NAME,
                B43_ADAPTIVE_PRECISION_H56_POLICY_NAME,
                B43_GENETIC_ADAPTIVE_PRECISION_H48_POLICY_NAME,
            ]
        )

        for config in (h48, arousal, threshold, h56, genetic):
            self.assertEqual(config.architecture, "b_series")
            self.assertEqual(config.b_level, 43)
            self.assertEqual(config.b_parent_level, 42)
            self.assertFalse(config.b_transfer_allow_low_coverage)
            self.assertIn("b43_precision_decay", config.b_controller_params)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(h48.b_controller_profile, "adaptive_precision")
        self.assertEqual(arousal.b_controller_profile, "arousal_precision")
        self.assertEqual(threshold.b_controller_profile, "threshold_adaptation")
        self.assertEqual(genetic.b_controller_profile, "genetic_adaptive_precision")

    def test_diagnostic_catalog_registers_b44_thalamic_relay_variants(self) -> None:
        h48, sensory, context, h56, genetic = resolve_ablation_configs(
            [
                B44_THALAMIC_RELAY_H48_POLICY_NAME,
                B44_SENSORY_RELAY_H48_POLICY_NAME,
                B44_CONTEXT_RELAY_H48_POLICY_NAME,
                B44_THALAMIC_RELAY_H56_POLICY_NAME,
                B44_GENETIC_THALAMIC_RELAY_H48_POLICY_NAME,
            ]
        )

        for config in (h48, sensory, context, h56, genetic):
            self.assertEqual(config.architecture, "b_series")
            self.assertEqual(config.b_level, 44)
            self.assertEqual(config.b_parent_level, 43)
            self.assertFalse(config.b_transfer_allow_low_coverage)
            self.assertIn("b44_relay_decay", config.b_controller_params)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(h48.b_controller_profile, "thalamic_relay")
        self.assertEqual(sensory.b_controller_profile, "sensory_relay")
        self.assertEqual(context.b_controller_profile, "context_relay")
        self.assertEqual(genetic.b_controller_profile, "genetic_thalamic_relay")

    def test_diagnostic_catalog_registers_b45_reticular_inhibition_variants(self) -> None:
        h48, sensory, context, h56, genetic = resolve_ablation_configs(
            [
                B45_RETICULAR_INHIBITION_H48_POLICY_NAME,
                B45_SENSORY_INHIBITION_H48_POLICY_NAME,
                B45_CONTEXT_INHIBITION_H48_POLICY_NAME,
                B45_RETICULAR_INHIBITION_H56_POLICY_NAME,
                B45_GENETIC_RETICULAR_INHIBITION_H48_POLICY_NAME,
            ]
        )

        for config in (h48, sensory, context, h56, genetic):
            self.assertEqual(config.architecture, "b_series")
            self.assertEqual(config.b_level, 45)
            self.assertEqual(config.b_parent_level, 44)
            self.assertFalse(config.b_transfer_allow_low_coverage)
            self.assertIn("b45_inhibition_decay", config.b_controller_params)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(h48.b_controller_profile, "reticular_inhibition")
        self.assertEqual(sensory.b_controller_profile, "sensory_inhibition")
        self.assertEqual(context.b_controller_profile, "context_inhibition")
        self.assertEqual(genetic.b_controller_profile, "genetic_reticular_inhibition")

    def test_diagnostic_catalog_registers_b46_corticothalamic_feedback_variants(self) -> None:
        h48, feedback, context, h56, genetic = resolve_ablation_configs(
            [
                B46_CORTICOTHALAMIC_FEEDBACK_H48_POLICY_NAME,
                B46_FEEDBACK_GAIN_H48_POLICY_NAME,
                B46_CONTEXT_FEEDBACK_H48_POLICY_NAME,
                B46_CORTICOTHALAMIC_FEEDBACK_H56_POLICY_NAME,
                B46_GENETIC_CORTICOTHALAMIC_FEEDBACK_H48_POLICY_NAME,
            ]
        )

        for config in (h48, feedback, context, h56, genetic):
            self.assertEqual(config.architecture, "b_series")
            self.assertEqual(config.b_level, 46)
            self.assertEqual(config.b_parent_level, 45)
            self.assertFalse(config.b_transfer_allow_low_coverage)
            self.assertIn("b46_feedback_decay", config.b_controller_params)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(h48.b_controller_profile, "corticothalamic_feedback")
        self.assertEqual(feedback.b_controller_profile, "feedback_gain")
        self.assertEqual(context.b_controller_profile, "context_feedback")
        self.assertEqual(genetic.b_controller_profile, "genetic_corticothalamic_feedback")

    def test_diagnostic_catalog_registers_b47_oscillatory_synchrony_variants(self) -> None:
        h48, phase, coherence, h56, genetic = resolve_ablation_configs(
            [
                B47_OSCILLATORY_SYNCHRONY_H48_POLICY_NAME,
                B47_PHASE_LOCKING_H48_POLICY_NAME,
                B47_COHERENCE_GATE_H48_POLICY_NAME,
                B47_OSCILLATORY_SYNCHRONY_H56_POLICY_NAME,
                B47_GENETIC_OSCILLATORY_SYNCHRONY_H48_POLICY_NAME,
            ]
        )

        for config in (h48, phase, coherence, h56, genetic):
            self.assertEqual(config.architecture, "b_series")
            self.assertEqual(config.b_level, 47)
            self.assertEqual(config.b_parent_level, 46)
            self.assertFalse(config.b_transfer_allow_low_coverage)
            self.assertIn("b47_phase_decay", config.b_controller_params)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(h48.b_controller_profile, "oscillatory_synchrony")
        self.assertEqual(phase.b_controller_profile, "phase_locking")
        self.assertEqual(coherence.b_controller_profile, "coherence_gate")
        self.assertEqual(genetic.b_controller_profile, "genetic_oscillatory_synchrony")

    def test_diagnostic_catalog_registers_b48_cerebellar_timing_variants(self) -> None:
        h48, correction, predictive, h56, genetic = resolve_ablation_configs(
            [
                B48_CEREBELLAR_TIMING_H48_POLICY_NAME,
                B48_TIMING_ERROR_CORRECTION_H48_POLICY_NAME,
                B48_PREDICTIVE_TIMING_H48_POLICY_NAME,
                B48_CEREBELLAR_TIMING_H56_POLICY_NAME,
                B48_GENETIC_CEREBELLAR_TIMING_H48_POLICY_NAME,
            ]
        )

        for config in (h48, correction, predictive, h56, genetic):
            self.assertEqual(config.architecture, "b_series")
            self.assertEqual(config.b_level, 48)
            self.assertEqual(config.b_parent_level, 47)
            self.assertFalse(config.b_transfer_allow_low_coverage)
            self.assertIn("b48_timing_decay", config.b_controller_params)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(h48.b_controller_profile, "cerebellar_timing")
        self.assertEqual(correction.b_controller_profile, "timing_error_correction")
        self.assertEqual(predictive.b_controller_profile, "predictive_timing")
        self.assertEqual(genetic.b_controller_profile, "genetic_cerebellar_timing")

    def test_diagnostic_catalog_registers_b49_striatal_action_gate_variants(self) -> None:
        h48, direct, indirect, h56, genetic = resolve_ablation_configs(
            [
                B49_STRIATAL_ACTION_GATE_H48_POLICY_NAME,
                B49_DIRECT_PATH_FACILITATION_H48_POLICY_NAME,
                B49_INDIRECT_PATH_SUPPRESSION_H48_POLICY_NAME,
                B49_STRIATAL_ACTION_GATE_H56_POLICY_NAME,
                B49_GENETIC_STRIATAL_GATE_H48_POLICY_NAME,
            ]
        )

        for config in (h48, direct, indirect, h56, genetic):
            self.assertEqual(config.architecture, "b_series")
            self.assertEqual(config.b_level, 49)
            self.assertEqual(config.b_parent_level, 48)
            self.assertFalse(config.b_transfer_allow_low_coverage)
            self.assertIn("b49_gate_decay", config.b_controller_params)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(h48.b_controller_profile, "striatal_action_gate")
        self.assertEqual(direct.b_controller_profile, "direct_path_facilitation")
        self.assertEqual(indirect.b_controller_profile, "indirect_path_suppression")
        self.assertEqual(genetic.b_controller_profile, "genetic_striatal_gate")

    def test_diagnostic_catalog_registers_b50_habit_chunking_variants(self) -> None:
        h48, value, stability, h56, genetic = resolve_ablation_configs(
            [
                B50_HABIT_CHUNKING_H48_POLICY_NAME,
                B50_ACTION_CHUNK_VALUE_H48_POLICY_NAME,
                B50_HABIT_STABILITY_H48_POLICY_NAME,
                B50_HABIT_CHUNKING_H56_POLICY_NAME,
                B50_GENETIC_HABIT_CHUNKING_H48_POLICY_NAME,
            ]
        )

        for config in (h48, value, stability, h56, genetic):
            self.assertEqual(config.architecture, "b_series")
            self.assertEqual(config.b_level, 50)
            self.assertEqual(config.b_parent_level, 49)
            self.assertFalse(config.b_transfer_allow_low_coverage)
            self.assertIn("b50_habit_decay", config.b_controller_params)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(h48.b_controller_profile, "habit_chunking")
        self.assertEqual(value.b_controller_profile, "action_chunk_value")
        self.assertEqual(stability.b_controller_profile, "habit_stability")
        self.assertEqual(genetic.b_controller_profile, "genetic_habit_chunking")

    def test_diagnostic_catalog_registers_b51_dopaminergic_habit_variants(self) -> None:
        h48, reward, novelty, h56, genetic = resolve_ablation_configs(
            [
                B51_DOPAMINERGIC_HABIT_MODULATION_H48_POLICY_NAME,
                B51_REWARD_PREDICTION_GAIN_H48_POLICY_NAME,
                B51_NOVELTY_MODULATED_HABIT_H48_POLICY_NAME,
                B51_DOPAMINERGIC_HABIT_MODULATION_H56_POLICY_NAME,
                B51_GENETIC_DOPAMINE_HABIT_H48_POLICY_NAME,
            ]
        )

        for config in (h48, reward, novelty, h56, genetic):
            self.assertEqual(config.architecture, "b_series")
            self.assertEqual(config.b_level, 51)
            self.assertEqual(config.b_parent_level, 50)
            self.assertFalse(config.b_transfer_allow_low_coverage)
            self.assertIn("b51_dopamine_decay", config.b_controller_params)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(h48.b_controller_profile, "dopaminergic_habit_modulation")
        self.assertEqual(reward.b_controller_profile, "reward_prediction_gain")
        self.assertEqual(novelty.b_controller_profile, "novelty_modulated_habit")
        self.assertEqual(genetic.b_controller_profile, "genetic_dopamine_habit")

    def test_diagnostic_catalog_registers_b52_cholinergic_precision_variants(self) -> None:
        h48, attention, uncertainty, h56, genetic = resolve_ablation_configs(
            [
                B52_CHOLINERGIC_PRECISION_GATE_H48_POLICY_NAME,
                B52_ATTENTION_GAIN_H48_POLICY_NAME,
                B52_UNCERTAINTY_RELEASE_H48_POLICY_NAME,
                B52_CHOLINERGIC_PRECISION_GATE_H56_POLICY_NAME,
                B52_GENETIC_CHOLINERGIC_PRECISION_H48_POLICY_NAME,
            ]
        )

        for config in (h48, attention, uncertainty, h56, genetic):
            self.assertEqual(config.architecture, "b_series")
            self.assertEqual(config.b_level, 52)
            self.assertEqual(config.b_parent_level, 51)
            self.assertFalse(config.b_transfer_allow_low_coverage)
            self.assertIn("b52_acetylcholine_decay", config.b_controller_params)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(h48.b_controller_profile, "cholinergic_precision_gate")
        self.assertEqual(attention.b_controller_profile, "attention_gain")
        self.assertEqual(uncertainty.b_controller_profile, "uncertainty_release")
        self.assertEqual(genetic.b_controller_profile, "genetic_cholinergic_precision")

    def test_diagnostic_catalog_registers_b53_noradrenergic_arousal_variants(self) -> None:
        h48, surprise, stress, h56, genetic = resolve_ablation_configs(
            [
                B53_NORADRENERGIC_AROUSAL_GAIN_H48_POLICY_NAME,
                B53_SURPRISE_GAIN_H48_POLICY_NAME,
                B53_STRESS_PRECISION_H48_POLICY_NAME,
                B53_NORADRENERGIC_AROUSAL_GAIN_H56_POLICY_NAME,
                B53_GENETIC_AROUSAL_PRECISION_H48_POLICY_NAME,
            ]
        )

        for config in (h48, surprise, stress, h56, genetic):
            self.assertEqual(config.architecture, "b_series")
            self.assertEqual(config.b_level, 53)
            self.assertEqual(config.b_parent_level, 52)
            self.assertFalse(config.b_transfer_allow_low_coverage)
            self.assertIn("b53_norepinephrine_decay", config.b_controller_params)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(h48.b_controller_profile, "noradrenergic_arousal_gain")
        self.assertEqual(surprise.b_controller_profile, "surprise_gain")
        self.assertEqual(stress.b_controller_profile, "stress_precision")
        self.assertEqual(genetic.b_controller_profile, "genetic_arousal_precision")

    def test_diagnostic_catalog_registers_b54_serotonergic_patience_variants(self) -> None:
        h48, suppression, balance, h56, genetic = resolve_ablation_configs(
            [
                B54_SEROTONERGIC_PATIENCE_GATE_H48_POLICY_NAME,
                B54_IMPULSE_SUPPRESSION_H48_POLICY_NAME,
                B54_PATIENCE_BALANCE_H48_POLICY_NAME,
                B54_SEROTONERGIC_PATIENCE_GATE_H56_POLICY_NAME,
                B54_GENETIC_SEROTONIN_PATIENCE_H48_POLICY_NAME,
            ]
        )

        for config in (h48, suppression, balance, h56, genetic):
            self.assertEqual(config.architecture, "b_series")
            self.assertEqual(config.b_level, 54)
            self.assertEqual(config.b_parent_level, 53)
            self.assertFalse(config.b_transfer_allow_low_coverage)
            self.assertIn("b54_serotonin_decay", config.b_controller_params)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(h48.b_controller_profile, "serotonergic_patience_gate")
        self.assertEqual(suppression.b_controller_profile, "impulse_suppression")
        self.assertEqual(balance.b_controller_profile, "patience_balance")
        self.assertEqual(genetic.b_controller_profile, "genetic_serotonin_patience")

    def test_diagnostic_catalog_registers_b55_hypothalamic_drive_variants(self) -> None:
        h48, recovery, arbiter, h56, genetic = resolve_ablation_configs(
            [
                B55_HYPOTHALAMIC_DRIVE_COUPLING_H48_POLICY_NAME,
                B55_SATIETY_RECOVERY_BALANCE_H48_POLICY_NAME,
                B55_SLEEP_HUNGER_ARBITER_H48_POLICY_NAME,
                B55_HYPOTHALAMIC_DRIVE_COUPLING_H56_POLICY_NAME,
                B55_GENETIC_HYPOTHALAMIC_DRIVE_H48_POLICY_NAME,
            ]
        )

        for config in (h48, recovery, arbiter, h56, genetic):
            self.assertEqual(config.architecture, "b_series")
            self.assertEqual(config.b_level, 55)
            self.assertEqual(config.b_parent_level, 54)
            self.assertFalse(config.b_transfer_allow_low_coverage)
            self.assertIn("b55_drive_decay", config.b_controller_params)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(h48.b_controller_profile, "hypothalamic_drive_coupling")
        self.assertEqual(recovery.b_controller_profile, "satiety_recovery_balance")
        self.assertEqual(arbiter.b_controller_profile, "sleep_hunger_arbiter")
        self.assertEqual(genetic.b_controller_profile, "genetic_hypothalamic_drive")

    def test_diagnostic_catalog_registers_b56_hpa_stress_variants(self) -> None:
        h48, recovery, stress, h56, genetic = resolve_ablation_configs(
            [
                B56_HPA_STRESS_AXIS_H48_POLICY_NAME,
                B56_CORTISOL_RECOVERY_BALANCE_H48_POLICY_NAME,
                B56_STRESS_LOAD_GATE_H48_POLICY_NAME,
                B56_HPA_STRESS_AXIS_H56_POLICY_NAME,
                B56_GENETIC_HPA_STRESS_H48_POLICY_NAME,
            ]
        )

        for config in (h48, recovery, stress, h56, genetic):
            self.assertEqual(config.architecture, "b_series")
            self.assertEqual(config.b_level, 56)
            self.assertEqual(config.b_parent_level, 55)
            self.assertFalse(config.b_transfer_allow_low_coverage)
            self.assertIn("b56_endocrine_decay", config.b_controller_params)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(h48.b_controller_profile, "hpa_stress_axis")
        self.assertEqual(recovery.b_controller_profile, "cortisol_recovery_balance")
        self.assertEqual(stress.b_controller_profile, "stress_load_gate")
        self.assertEqual(genetic.b_controller_profile, "genetic_hpa_stress")

    def test_diagnostic_catalog_registers_b57_insular_variants(self) -> None:
        h48, salience, awareness, h56, genetic = resolve_ablation_configs(
            [
                B57_INSULAR_INTEROCEPTIVE_AWARENESS_H48_POLICY_NAME,
                B57_VISCERAL_SALIENCE_GATE_H48_POLICY_NAME,
                B57_STRESS_DRIVE_AWARENESS_H48_POLICY_NAME,
                B57_INSULAR_INTEROCEPTIVE_AWARENESS_H56_POLICY_NAME,
                B57_GENETIC_INTEROCEPTIVE_AWARENESS_H48_POLICY_NAME,
            ]
        )

        for config in (h48, salience, awareness, h56, genetic):
            self.assertEqual(config.architecture, "b_series")
            self.assertEqual(config.b_level, 57)
            self.assertEqual(config.b_parent_level, 56)
            self.assertFalse(config.b_transfer_allow_low_coverage)
            self.assertIn("b57_awareness_decay", config.b_controller_params)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(
            h48.b_controller_profile,
            "insular_interoceptive_awareness",
        )
        self.assertEqual(salience.b_controller_profile, "visceral_salience_gate")
        self.assertEqual(awareness.b_controller_profile, "stress_drive_awareness")
        self.assertEqual(genetic.b_controller_profile, "genetic_interoceptive_awareness")

    def test_diagnostic_catalog_registers_b58_acc_conflict_variants(self) -> None:
        h48, error, balance, h56, genetic = resolve_ablation_configs(
            [
                B58_ACC_CONFLICT_MONITOR_H48_POLICY_NAME,
                B58_ERROR_SALIENCE_GATE_H48_POLICY_NAME,
                B58_CONFLICT_RESOLUTION_BALANCE_H48_POLICY_NAME,
                B58_ACC_CONFLICT_MONITOR_H56_POLICY_NAME,
                B58_GENETIC_ACC_CONFLICT_H48_POLICY_NAME,
            ]
        )

        for config in (h48, error, balance, h56, genetic):
            self.assertEqual(config.architecture, "b_series")
            self.assertEqual(config.b_level, 58)
            self.assertEqual(config.b_parent_level, 57)
            self.assertFalse(config.b_transfer_allow_low_coverage)
            self.assertIn("b58_conflict_decay", config.b_controller_params)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(h48.b_controller_profile, "acc_conflict_monitor")
        self.assertEqual(error.b_controller_profile, "error_salience_gate")
        self.assertEqual(balance.b_controller_profile, "conflict_resolution_balance")
        self.assertEqual(genetic.b_controller_profile, "genetic_acc_conflict")

    def test_diagnostic_catalog_registers_b59_prefrontal_variants(self) -> None:
        h48, working, task, h56, genetic = resolve_ablation_configs(
            [
                B59_PREFRONTAL_GOAL_CONTEXT_H48_POLICY_NAME,
                B59_WORKING_SET_STABILITY_H48_POLICY_NAME,
                B59_EXECUTIVE_TASK_SET_H48_POLICY_NAME,
                B59_PREFRONTAL_GOAL_CONTEXT_H56_POLICY_NAME,
                B59_GENETIC_PREFRONTAL_CONTROL_H48_POLICY_NAME,
            ]
        )

        for config in (h48, working, task, h56, genetic):
            self.assertEqual(config.architecture, "b_series")
            self.assertEqual(config.b_level, 59)
            self.assertEqual(config.b_parent_level, 58)
            self.assertFalse(config.b_transfer_allow_low_coverage)
            self.assertIn("b59_context_decay", config.b_controller_params)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(h48.b_controller_profile, "prefrontal_goal_context")
        self.assertEqual(working.b_controller_profile, "working_set_stability")
        self.assertEqual(task.b_controller_profile, "executive_task_set")
        self.assertEqual(genetic.b_controller_profile, "genetic_prefrontal_control")

    def test_diagnostic_catalog_registers_b60_orbitofrontal_variants(self) -> None:
        h48, reversal, prediction, h56, genetic = resolve_ablation_configs(
            [
                B60_ORBITOFRONTAL_OUTCOME_VALUE_H48_POLICY_NAME,
                B60_REVERSAL_VALUE_GATE_H48_POLICY_NAME,
                B60_GOAL_OUTCOME_PREDICTION_H48_POLICY_NAME,
                B60_ORBITOFRONTAL_OUTCOME_VALUE_H56_POLICY_NAME,
                B60_GENETIC_ORBITOFRONTAL_VALUE_H48_POLICY_NAME,
            ]
        )

        for config in (h48, reversal, prediction, h56, genetic):
            self.assertEqual(config.architecture, "b_series")
            self.assertEqual(config.b_level, 60)
            self.assertEqual(config.b_parent_level, 59)
            self.assertFalse(config.b_transfer_allow_low_coverage)
            self.assertIn("b60_value_decay", config.b_controller_params)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(h48.b_controller_profile, "orbitofrontal_outcome_value")
        self.assertEqual(reversal.b_controller_profile, "reversal_value_gate")
        self.assertEqual(prediction.b_controller_profile, "goal_outcome_prediction")
        self.assertEqual(genetic.b_controller_profile, "genetic_orbitofrontal_value")

    def test_diagnostic_catalog_registers_b61_amygdala_variants(self) -> None:
        h48, threat, safety, h56, genetic = resolve_ablation_configs(
            [
                B61_AMYGDALA_SAFETY_VALUE_H48_POLICY_NAME,
                B61_THREAT_VALUE_TAG_H48_POLICY_NAME,
                B61_SAFETY_PREDICTION_GATE_H48_POLICY_NAME,
                B61_AMYGDALA_SAFETY_VALUE_H56_POLICY_NAME,
                B61_GENETIC_AMYGDALA_SAFETY_H48_POLICY_NAME,
            ]
        )

        for config in (h48, threat, safety, h56, genetic):
            self.assertEqual(config.architecture, "b_series")
            self.assertEqual(config.b_level, 61)
            self.assertEqual(config.b_parent_level, 60)
            self.assertFalse(config.b_transfer_allow_low_coverage)
            self.assertIn("b61_affect_decay", config.b_controller_params)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(h48.b_controller_profile, "amygdala_safety_value")
        self.assertEqual(threat.b_controller_profile, "threat_value_tag")
        self.assertEqual(safety.b_controller_profile, "safety_prediction_gate")
        self.assertEqual(genetic.b_controller_profile, "genetic_amygdala_safety")

    def test_diagnostic_catalog_registers_b62_defensive_mode_variants(self) -> None:
        h48, balance, shelter, h56, genetic = resolve_ablation_configs(
            [
                B62_DEFENSIVE_MODE_SELECTOR_H48_POLICY_NAME,
                B62_FREEZE_FLEE_BALANCE_H48_POLICY_NAME,
                B62_SHELTER_DEFENSE_GATE_H48_POLICY_NAME,
                B62_DEFENSIVE_MODE_SELECTOR_H56_POLICY_NAME,
                B62_GENETIC_DEFENSIVE_MODE_H48_POLICY_NAME,
            ]
        )

        for config in (h48, balance, shelter, h56, genetic):
            self.assertEqual(config.architecture, "b_series")
            self.assertEqual(config.b_level, 62)
            self.assertEqual(config.b_parent_level, 61)
            self.assertFalse(config.b_transfer_allow_low_coverage)
            self.assertIn("b62_defense_decay", config.b_controller_params)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(h48.b_controller_profile, "defensive_mode_selector")
        self.assertEqual(balance.b_controller_profile, "freeze_flee_balance")
        self.assertEqual(shelter.b_controller_profile, "shelter_defense_gate")
        self.assertEqual(genetic.b_controller_profile, "genetic_defensive_mode")

    def test_legacy_variant_is_not_routed_through_current_world_brain(self) -> None:
        config = _b0_config("b0_legacy_semantic_policy")
        with self.assertRaisesRegex(ValueError, "LegacyB0Simulation"):
            SpiderBrain(seed=1, module_dropout=0.0, config=config)
