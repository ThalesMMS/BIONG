from __future__ import annotations

from .shared import *
from .corticalmodulebankrecurrentmodulestest_helpers import CorticalModuleBankRecurrentModulesTestHelpers



class CorticalModuleBankRecurrentModulesTestPart5(CorticalModuleBankRecurrentModulesTestHelpers, unittest.TestCase):
    def test_direct_policy_local_transition_rollout_inputs_require_transition_inputs(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "direct_policy_local_transition_rollout_inputs requires direct_policy_local_transition_inputs=True",
        ):
            BrainAblationConfig(
                name="invalid_local_transition_rollout_inputs_branch",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_affordance_head=True,
                direct_policy_affordance_feedback=True,
                direct_policy_geometry_head=True,
                direct_policy_shelter_position_head=True,
                direct_policy_local_affordance_inputs=True,
                direct_policy_local_spatial_inputs=True,
                direct_policy_local_transition_rollout_inputs=True,
            )

    def test_direct_policy_local_transition_rollout_inputs_extend_observation_vector(self) -> None:
        config = BrainAblationConfig(
            name="local_transition_rollout_inputs_branch",
            architecture="true_monolithic",
            module_dropout=0.0,
            enable_reflexes=False,
            enable_auxiliary_targets=False,
            enable_food_direction_bias=True,
            use_learned_arbitration=False,
            warm_start_scale=0.0,
            direct_policy_hidden_dims=(32,),
            direct_policy_recurrent=True,
            direct_policy_event_attention=True,
            direct_policy_event_buffer_size=8,
            direct_policy_option_head=True,
            direct_policy_option_ttl=4,
            direct_policy_affordance_head=True,
            direct_policy_affordance_feedback=True,
            direct_policy_geometry_head=True,
            direct_policy_shelter_position_head=True,
            direct_policy_local_affordance_inputs=True,
            direct_policy_local_spatial_inputs=True,
            direct_policy_local_transition_inputs=True,
            direct_policy_local_transition_rollout_inputs=True,
        )
        brain = SpiderBrain(seed=23, module_dropout=0.0, config=config)
        observation = _build_observation()
        observation["meta"] = {
            "shelter_role": "inside",
            "shelter_role_level": 0.66,
            "local_affordances": {
                "STAY": {"blocked": False, "next_role": "inside", "next_role_level": 0.66},
                "MOVE_UP": {"blocked": False, "next_role": "entrance", "next_role_level": 0.33},
                "MOVE_DOWN": {"blocked": False, "next_role": "deep", "next_role_level": 1.0},
                "MOVE_LEFT": {"blocked": False, "next_role": "inside", "next_role_level": 0.66},
                "MOVE_RIGHT": {"blocked": True, "next_role": "inside", "next_role_level": 0.66},
            },
            "local_spatial_patch": {
                "blocked": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                "shelter_role_level": [0.0, 0.25, 0.5, 0.0, 0.66, 0.75, 1.0, 1.0, 1.0],
                "food": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            },
            "local_transition_consequences": {
                "STAY": {"food_dist_delta": 0.0, "shelter_dist_delta": 0.0, "predator_dist_delta": 0.0, "next_cell_has_food": False},
                "MOVE_UP": {"food_dist_delta": 1.0, "shelter_dist_delta": 1.0, "predator_dist_delta": -1.0, "next_cell_has_food": False},
                "MOVE_DOWN": {"food_dist_delta": -1.0, "shelter_dist_delta": -1.0, "predator_dist_delta": 1.0, "next_cell_has_food": False},
                "MOVE_LEFT": {"food_dist_delta": 0.0, "shelter_dist_delta": 1.0, "predator_dist_delta": 0.0, "next_cell_has_food": True},
                "MOVE_RIGHT": {"food_dist_delta": 0.0, "shelter_dist_delta": 0.0, "predator_dist_delta": 0.0, "next_cell_has_food": False},
            },
            "local_transition_rollouts": {
                "STAY": {"best_food_dist_delta": 0.0, "best_shelter_dist_delta": 0.0, "best_predator_dist_delta": 0.0, "food_reachable_within_two_steps": False},
                "MOVE_UP": {"best_food_dist_delta": 1.0, "best_shelter_dist_delta": 0.0, "best_predator_dist_delta": -1.0, "food_reachable_within_two_steps": True},
                "MOVE_DOWN": {"best_food_dist_delta": 0.0, "best_shelter_dist_delta": -1.0, "best_predator_dist_delta": 1.0, "food_reachable_within_two_steps": False},
                "MOVE_LEFT": {"best_food_dist_delta": 1.0, "best_shelter_dist_delta": 1.0, "best_predator_dist_delta": 0.0, "food_reachable_within_two_steps": True},
                "MOVE_RIGHT": {"best_food_dist_delta": 0.0, "best_shelter_dist_delta": 0.0, "best_predator_dist_delta": 0.0, "food_reachable_within_two_steps": False},
            },
        }
        vector = brain._build_monolithic_observation(observation)

        self.assertEqual(
            vector.shape[0],
            sum(spec.input_dim for spec in MODULE_INTERFACES)
            + DIRECT_POLICY_LOCAL_AFFORDANCE_INPUT_DIM
            + DIRECT_POLICY_LOCAL_SPATIAL_INPUT_DIM
            + DIRECT_POLICY_LOCAL_TRANSITION_INPUT_DIM
            + DIRECT_POLICY_LOCAL_TRANSITION_ROLLOUT_INPUT_DIM,
        )
        rollout_tail = vector[-DIRECT_POLICY_LOCAL_TRANSITION_ROLLOUT_INPUT_DIM:]
        np.testing.assert_allclose(
            rollout_tail[:8],
            np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 1.0]),
        )

    def test_direct_policy_transition_prediction_head_requires_transition_inputs(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "direct_policy_transition_prediction_head requires direct_policy_local_transition_inputs=True",
        ):
            BrainAblationConfig(
                name="invalid_transition_prediction_head_branch",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_affordance_head=True,
                direct_policy_affordance_feedback=True,
                direct_policy_geometry_head=True,
                direct_policy_shelter_position_head=True,
                direct_policy_local_affordance_inputs=True,
                direct_policy_local_spatial_inputs=True,
                direct_policy_transition_prediction_head=True,
            )

    def test_transition_prediction_feedback_requires_head(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "direct_policy_transition_prediction_feedback requires direct_policy_transition_prediction_head=True",
        ):
            BrainAblationConfig(
                name="invalid_transition_prediction_feedback_branch",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_affordance_head=True,
                direct_policy_affordance_feedback=True,
                direct_policy_geometry_head=True,
                direct_policy_shelter_position_head=True,
                direct_policy_local_affordance_inputs=True,
                direct_policy_local_spatial_inputs=True,
                direct_policy_local_transition_inputs=True,
                direct_policy_transition_prediction_feedback=True,
            )

    def test_transition_prediction_branch_exposes_transition_logits(self) -> None:
        config = BrainAblationConfig(
            name="transition_prediction_branch",
            architecture="true_monolithic",
            module_dropout=0.0,
            enable_reflexes=False,
            enable_auxiliary_targets=False,
            enable_food_direction_bias=True,
            use_learned_arbitration=False,
            warm_start_scale=0.0,
            direct_policy_hidden_dims=(32,),
            direct_policy_recurrent=True,
            direct_policy_event_attention=True,
            direct_policy_event_buffer_size=8,
            direct_policy_option_head=True,
            direct_policy_option_ttl=4,
            direct_policy_affordance_head=True,
            direct_policy_affordance_feedback=True,
            direct_policy_geometry_head=True,
            direct_policy_shelter_position_head=True,
            direct_policy_local_affordance_inputs=True,
            direct_policy_local_spatial_inputs=True,
            direct_policy_local_transition_inputs=True,
            direct_policy_local_transition_rollout_inputs=True,
            direct_policy_transition_prediction_head=True,
            direct_policy_transition_prediction_feedback=True,
        )
        brain = SpiderBrain(seed=24, module_dropout=0.0, config=config)
        decision = brain.act_inference(_build_observation(), sample=False)
        recurrent = brain.true_monolithic_policy
        self.assertIsInstance(
            recurrent,
            RecurrentOptionAffordancePositionFeedbackTrueMonolithicNetwork,
        )
        self.assertEqual(
            decision.transition_prediction_logits.shape,
            (DIRECT_POLICY_TRANSITION_PREDICTION_FEATURE_DIM,),
        )
        self.assertEqual(
            len(
                recurrent.last_affordance_summary.get(
                    "transition_prediction_logits",
                    [],
                )
            ),
            DIRECT_POLICY_TRANSITION_PREDICTION_FEATURE_DIM,
        )

    def test_transition_rollout_prediction_head_requires_rollout_inputs(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "direct_policy_transition_rollout_prediction_head requires direct_policy_local_transition_rollout_inputs=True",
        ):
            BrainAblationConfig(
                name="invalid_transition_rollout_prediction_head_branch",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_affordance_head=True,
                direct_policy_affordance_feedback=True,
                direct_policy_geometry_head=True,
                direct_policy_shelter_position_head=True,
                direct_policy_local_affordance_inputs=True,
                direct_policy_local_spatial_inputs=True,
                direct_policy_local_transition_inputs=True,
                direct_policy_transition_rollout_prediction_head=True,
            )

    def test_transition_rollout_prediction_feedback_requires_head(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "direct_policy_transition_rollout_prediction_feedback requires direct_policy_transition_rollout_prediction_head=True",
        ):
            BrainAblationConfig(
                name="invalid_transition_rollout_prediction_feedback_branch",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_affordance_head=True,
                direct_policy_affordance_feedback=True,
                direct_policy_geometry_head=True,
                direct_policy_shelter_position_head=True,
                direct_policy_local_affordance_inputs=True,
                direct_policy_local_spatial_inputs=True,
                direct_policy_local_transition_inputs=True,
                direct_policy_local_transition_rollout_inputs=True,
                direct_policy_transition_rollout_prediction_feedback=True,
            )

    def test_transition_rollout_prediction_branch_exposes_transition_logits(self) -> None:
        config = BrainAblationConfig(
            name="transition_rollout_prediction_branch",
            architecture="true_monolithic",
            module_dropout=0.0,
            enable_reflexes=False,
            enable_auxiliary_targets=False,
            enable_food_direction_bias=True,
            use_learned_arbitration=False,
            warm_start_scale=0.0,
            direct_policy_hidden_dims=(32,),
            direct_policy_recurrent=True,
            direct_policy_event_attention=True,
            direct_policy_event_buffer_size=8,
            direct_policy_option_head=True,
            direct_policy_option_ttl=4,
            direct_policy_affordance_head=True,
            direct_policy_affordance_feedback=True,
            direct_policy_geometry_head=True,
            direct_policy_shelter_position_head=True,
            direct_policy_local_affordance_inputs=True,
            direct_policy_local_spatial_inputs=True,
            direct_policy_local_transition_inputs=True,
            direct_policy_local_transition_rollout_inputs=True,
            direct_policy_transition_rollout_prediction_head=True,
            direct_policy_transition_rollout_prediction_feedback=True,
        )
        brain = SpiderBrain(seed=25, module_dropout=0.0, config=config)
        decision = brain.act_inference(_build_observation(), sample=False)
        recurrent = brain.true_monolithic_policy
        self.assertIsInstance(
            recurrent,
            RecurrentOptionAffordancePositionFeedbackTrueMonolithicNetwork,
        )
        self.assertEqual(
            decision.transition_rollout_prediction_logits.shape,
            (DIRECT_POLICY_TRANSITION_ROLLOUT_PREDICTION_FEATURE_DIM,),
        )
        self.assertEqual(
            len(
                recurrent.last_affordance_summary.get(
                    "transition_rollout_prediction_logits",
                    [],
                )
            ),
            DIRECT_POLICY_TRANSITION_ROLLOUT_PREDICTION_FEATURE_DIM,
        )

    def test_direct_policy_local_geodesic_inputs_require_rollout_inputs(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "direct_policy_local_geodesic_inputs requires direct_policy_local_transition_rollout_inputs=True",
        ):
            BrainAblationConfig(
                name="invalid_local_geodesic_inputs_branch",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_affordance_head=True,
                direct_policy_affordance_feedback=True,
                direct_policy_geometry_head=True,
                direct_policy_shelter_position_head=True,
                direct_policy_local_affordance_inputs=True,
                direct_policy_local_spatial_inputs=True,
                direct_policy_local_transition_inputs=True,
                direct_policy_local_geodesic_inputs=True,
            )

    def test_direct_policy_local_geodesic_inputs_extend_observation_vector(self) -> None:
        config = BrainAblationConfig(
            name="local_geodesic_inputs_branch",
            architecture="true_monolithic",
            module_dropout=0.0,
            enable_reflexes=False,
            enable_auxiliary_targets=False,
            enable_food_direction_bias=True,
            use_learned_arbitration=False,
            warm_start_scale=0.0,
            direct_policy_hidden_dims=(32,),
            direct_policy_recurrent=True,
            direct_policy_event_attention=True,
            direct_policy_event_buffer_size=8,
            direct_policy_option_head=True,
            direct_policy_option_ttl=4,
            direct_policy_affordance_head=True,
            direct_policy_affordance_feedback=True,
            direct_policy_geometry_head=True,
            direct_policy_shelter_position_head=True,
            direct_policy_local_affordance_inputs=True,
            direct_policy_local_spatial_inputs=True,
            direct_policy_local_transition_inputs=True,
            direct_policy_local_transition_rollout_inputs=True,
            direct_policy_local_geodesic_inputs=True,
        )
        brain = SpiderBrain(seed=29, module_dropout=0.0, config=config)
        observation = _build_observation()
        observation["meta"] = {
            "shelter_role": "inside",
            "shelter_role_level": 0.66,
            "local_affordances": {
                "STAY": {"blocked": False, "next_role": "inside", "next_role_level": 0.66},
                "MOVE_UP": {"blocked": False, "next_role": "entrance", "next_role_level": 0.33},
                "MOVE_DOWN": {"blocked": False, "next_role": "deep", "next_role_level": 1.0},
                "MOVE_LEFT": {"blocked": False, "next_role": "inside", "next_role_level": 0.66},
                "MOVE_RIGHT": {"blocked": True, "next_role": "inside", "next_role_level": 0.66},
            },
            "local_spatial_patch": {
                "blocked": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                "shelter_role_level": [0.0, 0.25, 0.5, 0.0, 0.66, 0.75, 1.0, 1.0, 1.0],
                "food": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            },
            "local_transition_consequences": {
                "STAY": {"food_dist_delta": 0.0, "shelter_dist_delta": 0.0, "predator_dist_delta": 0.0, "next_cell_has_food": False},
                "MOVE_UP": {"food_dist_delta": 1.0, "shelter_dist_delta": 1.0, "predator_dist_delta": -1.0, "next_cell_has_food": False},
                "MOVE_DOWN": {"food_dist_delta": -1.0, "shelter_dist_delta": -1.0, "predator_dist_delta": 1.0, "next_cell_has_food": False},
                "MOVE_LEFT": {"food_dist_delta": 0.0, "shelter_dist_delta": 1.0, "predator_dist_delta": 0.0, "next_cell_has_food": True},
                "MOVE_RIGHT": {"food_dist_delta": 0.0, "shelter_dist_delta": 0.0, "predator_dist_delta": 0.0, "next_cell_has_food": False},
            },
            "local_transition_rollouts": {
                "STAY": {"best_food_dist_delta": 0.0, "best_shelter_dist_delta": 0.0, "best_predator_dist_delta": 0.0, "food_reachable_within_two_steps": False},
                "MOVE_UP": {"best_food_dist_delta": 1.0, "best_shelter_dist_delta": 0.0, "best_predator_dist_delta": -1.0, "food_reachable_within_two_steps": True},
                "MOVE_DOWN": {"best_food_dist_delta": 0.0, "best_shelter_dist_delta": -1.0, "best_predator_dist_delta": 1.0, "food_reachable_within_two_steps": False},
                "MOVE_LEFT": {"best_food_dist_delta": 1.0, "best_shelter_dist_delta": 1.0, "best_predator_dist_delta": 0.0, "food_reachable_within_two_steps": True},
                "MOVE_RIGHT": {"best_food_dist_delta": 0.0, "best_shelter_dist_delta": 0.0, "best_predator_dist_delta": 0.0, "food_reachable_within_two_steps": False},
            },
            "local_geodesic_consequences": {
                "STAY": {"exit_geodesic_delta": 0.0, "deep_geodesic_delta": 0.0, "next_on_exit_target": False, "next_on_deep_target": False},
                "MOVE_UP": {"exit_geodesic_delta": 1.0, "deep_geodesic_delta": -1.0, "next_on_exit_target": True, "next_on_deep_target": False},
                "MOVE_DOWN": {"exit_geodesic_delta": -1.0, "deep_geodesic_delta": 1.0, "next_on_exit_target": False, "next_on_deep_target": True},
                "MOVE_LEFT": {"exit_geodesic_delta": 0.0, "deep_geodesic_delta": 0.0, "next_on_exit_target": False, "next_on_deep_target": False},
                "MOVE_RIGHT": {"exit_geodesic_delta": 0.0, "deep_geodesic_delta": 0.0, "next_on_exit_target": False, "next_on_deep_target": False},
            },
        }
        vector = brain._build_monolithic_observation(observation)

        self.assertEqual(
            vector.shape[0],
            sum(spec.input_dim for spec in MODULE_INTERFACES)
            + DIRECT_POLICY_LOCAL_AFFORDANCE_INPUT_DIM
            + DIRECT_POLICY_LOCAL_SPATIAL_INPUT_DIM
            + DIRECT_POLICY_LOCAL_TRANSITION_INPUT_DIM
            + DIRECT_POLICY_LOCAL_TRANSITION_ROLLOUT_INPUT_DIM
            + DIRECT_POLICY_LOCAL_GEODESIC_INPUT_DIM,
        )
        geodesic_tail = vector[-DIRECT_POLICY_LOCAL_GEODESIC_INPUT_DIM:]
        np.testing.assert_allclose(
            geodesic_tail[:8],
            np.array([0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 1.0, 0.0]),
        )

    def test_transition_rollout_prediction_feedback_affects_selected_action(self) -> None:
        config = BrainAblationConfig(
            name="transition_rollout_prediction_feedback_action_branch",
            architecture="true_monolithic",
            module_dropout=0.0,
            enable_reflexes=False,
            enable_auxiliary_targets=False,
            enable_food_direction_bias=False,
            use_learned_arbitration=False,
            warm_start_scale=0.0,
            direct_policy_hidden_dims=(32,),
            direct_policy_recurrent=True,
            direct_policy_event_attention=True,
            direct_policy_event_buffer_size=8,
            direct_policy_option_head=True,
            direct_policy_option_ttl=4,
            direct_policy_affordance_head=True,
            direct_policy_affordance_feedback=True,
            direct_policy_geometry_head=True,
            direct_policy_shelter_position_head=True,
            direct_policy_local_affordance_inputs=True,
            direct_policy_local_spatial_inputs=True,
            direct_policy_local_transition_inputs=True,
            direct_policy_local_transition_rollout_inputs=True,
            direct_policy_transition_rollout_prediction_head=True,
            direct_policy_transition_rollout_prediction_feedback=True,
        )
        brain = SpiderBrain(seed=26, module_dropout=0.0, config=config)
        recurrent = brain.true_monolithic_policy
        self.assertIsInstance(
            recurrent,
            RecurrentOptionAffordancePositionFeedbackTrueMonolithicNetwork,
        )
        recurrent.W2_policy.fill(0.0)
        recurrent.b2_policy.fill(0.0)
        recurrent.W_affordance_feedback.fill(0.0)
        recurrent.b_affordance_feedback.fill(0.0)
        recurrent.W2_policy_feedback.fill(0.0)
        recurrent.b2_policy_feedback.fill(0.0)
        recurrent.W2_transition_rollout_prediction.fill(0.0)
        recurrent.b2_transition_rollout_prediction.fill(0.0)
        recurrent.W_transition_rollout_prediction_feedback.fill(0.0)
        recurrent.b_transition_rollout_prediction_feedback.fill(0.0)
        recurrent.b2_transition_rollout_prediction[0] = 10.0
        recurrent.W_transition_rollout_prediction_feedback[0, 0] = 5.0
        recurrent.W2_policy_feedback[ACTION_TO_INDEX["MOVE_UP"], 0] = 5.0
        observation = _build_observation()
        monolithic_observation = brain._build_monolithic_observation(observation)
        policy_logits = recurrent.forward(monolithic_observation, store_cache=False)[0]
        self.assertEqual(
            int(np.argmax(policy_logits)),
            ACTION_TO_INDEX["MOVE_UP"],
        )

    def test_collect_direct_policy_probe_trace_distillation_rollout_collects_trace_samples(self) -> None:
        config = resolve_ablation_configs(
            [
                "true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_backbone_post_rest_probe_trace_distill_option_replay_policy"
            ],
            module_dropout=0.0,
        )[0]
        sim = SpiderSimulation(seed=11, max_steps=20, brain_config=config)
        dataset = sim.collect_direct_policy_probe_trace_distillation_rollout(
            episodes=1,
            episode_start=13,
        )
        summary = dataset.to_summary()
        self.assertGreater(len(dataset), 40)
        self.assertEqual(summary["episode_ids"], [13, 14, 15, 16])
        self.assertEqual(
            dataset.teacher_metadata["source"],
            "redirected_direct_policy_post_rest_probe_trace",
        )
        action_stages = set(dataset.teacher_metadata.get("action_stages", []))
        self.assertIn("trace_release", action_stages)
        self.assertIn("trace_realign", action_stages)

    def test_collect_direct_policy_probe_rollout_distillation_rollout_collects_post_handoff_samples(self) -> None:
        config = resolve_ablation_configs(
            [
                "true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_backbone_post_rest_probe_rollout_distill_option_replay_policy"
            ],
            module_dropout=0.0,
        )[0]
        sim = SpiderSimulation(seed=11, max_steps=20, brain_config=config)
        dataset = sim.collect_direct_policy_probe_rollout_distillation_rollout(
            episodes=1,
            episode_start=17,
        )
        summary = dataset.to_summary()
        self.assertGreater(len(dataset), 10)
        self.assertEqual(summary["episode_ids"], [17, 18])
        self.assertEqual(
            dataset.teacher_metadata["source"],
            "redirected_direct_policy_post_rest_probe_rollout",
        )
        action_stages = set(dataset.teacher_metadata.get("action_stages", []))
        self.assertIn("rollout_release", action_stages)
        self.assertIn("rollout_live", action_stages)

    def test_direct_policy_handoff_teacher_targets_follow_probe_sequence(self) -> None:
        sim = SpiderSimulation(seed=7, max_steps=2)
        sim._reset_direct_policy_handoff_teacher_state()
        center_x = sim._teacher_center_column()
        release_state = {
            "x": center_x,
            "y": 6,
            "shelter_role": "inside",
            "sleep_phase": "RESTING",
            "rest_streak": 3,
            "hunger": 0.18,
            "food_memory": {"target": [center_x + 4, 7], "age": 1, "ttl": 12},
            "predator_positions": [[0, 0]],
            "recent_contact": 0.0,
            "recent_pain": 0.0,
            "predator_motion_salience": 0.0,
            "predator_trace": {"strength": 0.0},
        }
        entrance_state = {
            **release_state,
            "y": 5,
            "shelter_role": "entrance",
            "sleep_phase": "AWAKE",
        }
        inside_state = {
            **release_state,
            "y": 6,
            "shelter_role": "inside",
            "sleep_phase": "AWAKE",
        }
        deep_state = {
            **release_state,
            "y": 7,
            "shelter_role": "deep",
            "sleep_phase": "AWAKE",
        }
        post_rest_inside_state = {
            **inside_state,
            "sleep_events": 1,
        }
        post_rest_outside_state = {
            **inside_state,
            "y": 4,
            "shelter_role": "outside",
            "sleep_events": 1,
        }

        target_idx, stage = sim._direct_policy_handoff_teacher_target(
            current_state=release_state,
            food_direction_action="MOVE_RIGHT",
            tick=12,
        )
        self.assertEqual(LOCOMOTION_ACTIONS[target_idx], "MOVE_UP")
        self.assertEqual(stage, "handoff_release")

        target_idx, stage = sim._direct_policy_handoff_teacher_target(
            current_state=entrance_state,
            food_direction_action="MOVE_RIGHT",
            tick=13,
        )
        self.assertEqual(LOCOMOTION_ACTIONS[target_idx], "STAY")
        self.assertEqual(stage, "handoff_hold")

        for tick, state in ((14, entrance_state), (15, inside_state), (16, deep_state)):
            target_idx, stage = sim._direct_policy_handoff_teacher_target(
                current_state=state,
                food_direction_action="MOVE_RIGHT",
                tick=tick,
            )
            self.assertEqual(LOCOMOTION_ACTIONS[target_idx], "MOVE_DOWN")
            self.assertEqual(stage, "handoff_continue")

        option_idx, option_stage = sim._direct_policy_handoff_option_teacher_target(
            current_state=post_rest_inside_state,
            food_direction_action="MOVE_RIGHT",
            tick=17,
        )
        self.assertEqual(OPTION_NAMES[option_idx], "FORAGE")
        self.assertEqual(option_stage, "option_forage_window")

        sim.brain = SpiderBrain(
            seed=99,
            config=BrainAblationConfig(
                name="true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_backbone_post_rest_action_teacher_option_replay_policy",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_phase_head=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_affordance_head=True,
                direct_policy_affordance_feedback=True,
                direct_policy_geometry_head=True,
                direct_policy_shelter_position_head=True,
                direct_policy_handoff_teacher=True,
                direct_policy_handoff_option_teacher=True,
                direct_policy_post_rest_action_teacher=True,
                direct_policy_option_decoder_state=True,
                direct_policy_option_recurrent_dynamics=True,
                direct_policy_option_decoder_recurrent_state=True,
                direct_policy_option_action_recurrent_core=True,
                direct_policy_option_action_separate_recurrent_head=True,
                direct_policy_option_action_separate_policy_path=True,
                direct_policy_option_action_separate_backbone=True,
                direct_policy_continuation_replay_passes=2,
                direct_policy_continuation_replay_lr_scale=0.5,
            ),
        )
        sim._reset_direct_policy_handoff_teacher_state()
        sim._direct_policy_handoff_teacher_target(
            current_state=release_state,
            food_direction_action="MOVE_RIGHT",
            tick=12,
        )
        sim._direct_policy_handoff_teacher_target(
            current_state=entrance_state,
            food_direction_action="MOVE_RIGHT",
            tick=13,
        )
        for tick, state in ((14, entrance_state), (15, inside_state), (16, deep_state)):
            sim._direct_policy_handoff_teacher_target(
                current_state=state,
                food_direction_action="MOVE_RIGHT",
                tick=tick,
            )
        target_idx, stage = sim._direct_policy_handoff_teacher_target(
            current_state=post_rest_outside_state,
            food_direction_action="MOVE_RIGHT",
            tick=17,
        )
        self.assertEqual(target_idx, int(ACTION_TO_INDEX["MOVE_RIGHT"]))
        self.assertEqual(stage, "handoff_forage_window")

        sim._reset_direct_policy_handoff_teacher_state()
        option_idx, option_stage = sim._direct_policy_handoff_option_teacher_target(
            current_state=post_rest_inside_state,
            food_direction_action="MOVE_RIGHT",
            tick=18,
        )
        self.assertEqual(OPTION_NAMES[option_idx], "POST_REST_REACTIVATE")
        self.assertEqual(option_stage, "option_post_rest_inside")

        threatened_outside_state = {
            **post_rest_outside_state,
            "predator_positions": [[center_x, 4]],
            "recent_contact": 0.0,
            "recent_pain": 0.0,
            "predator_motion_salience": 1.0,
            "predator_trace": {"strength": 1.0},
        }
        option_idx, option_stage = sim._direct_policy_handoff_option_teacher_target(
            current_state=threatened_outside_state,
            food_direction_action="MOVE_RIGHT",
            tick=19,
        )
        self.assertEqual(OPTION_NAMES[option_idx], "RETURN_TO_SHELTER")
        self.assertEqual(option_stage, "option_threatened_return")

        sim._reset_direct_policy_handoff_teacher_state()
        target_idx, stage = sim._direct_policy_handoff_teacher_target(
            current_state=post_rest_outside_state,
            food_direction_action="MOVE_RIGHT",
            tick=20,
        )
        self.assertEqual(target_idx, int(ACTION_TO_INDEX["MOVE_RIGHT"]))
        self.assertEqual(stage, "handoff_post_rest_forage")

        sim.brain = SpiderBrain(
            seed=101,
            config=BrainAblationConfig(
                name="true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_backbone_post_rest_release_sequence_teacher_option_replay_policy",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_phase_head=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_affordance_head=True,
                direct_policy_affordance_feedback=True,
                direct_policy_geometry_head=True,
                direct_policy_shelter_position_head=True,
                direct_policy_handoff_teacher=True,
                direct_policy_handoff_option_teacher=True,
                direct_policy_post_rest_action_teacher=True,
                direct_policy_post_rest_release_sequence_teacher=True,
                direct_policy_option_decoder_state=True,
                direct_policy_option_recurrent_dynamics=True,
                direct_policy_option_decoder_recurrent_state=True,
                direct_policy_option_action_recurrent_core=True,
                direct_policy_option_action_separate_recurrent_head=True,
                direct_policy_option_action_separate_policy_path=True,
                direct_policy_option_action_separate_backbone=True,
                direct_policy_continuation_replay_passes=2,
                direct_policy_continuation_replay_lr_scale=0.5,
            ),
        )
        sim._reset_direct_policy_handoff_teacher_state()
        target_idx, stage = sim._direct_policy_handoff_teacher_target(
            current_state=post_rest_inside_state,
            food_direction_action="MOVE_RIGHT",
            tick=21,
        )
        self.assertEqual(LOCOMOTION_ACTIONS[target_idx], "MOVE_UP")
        self.assertEqual(stage, "handoff_release")

    def test_collect_direct_policy_probe_distillation_rollout_collects_teacher_samples(self) -> None:
        sim = SpiderSimulation(
            seed=11,
            max_steps=20,
            brain_config=resolve_ablation_configs(
                [
                    "true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_backbone_post_rest_probe_distill_option_replay_policy"
                ],
                module_dropout=0.0,
            )[0],
        )
        dataset = sim.collect_direct_policy_probe_distillation_rollout(
            scenario_name="continuous_survival_post_rest_inside_v1",
            episodes=1,
            episode_start=3,
        )
        self.assertGreater(len(dataset), 0)
        self.assertEqual(dataset.to_summary()["episode_ids"], [3])
        self.assertEqual(
            dataset.teacher_metadata["source"],
            "scripted_direct_policy_post_rest_probe",
        )
        self.assertIn("handoff_release", dataset.teacher_metadata["stages"])

    def test_collect_direct_policy_probe_sequence_distillation_rollout_collects_multistep_samples(self) -> None:
        sim = SpiderSimulation(
            seed=11,
            max_steps=20,
            brain_config=resolve_ablation_configs(
                [
                    "true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_backbone_post_rest_probe_sequence_distill_option_replay_policy"
                ],
                module_dropout=0.0,
            )[0],
        )
        dataset = sim.collect_direct_policy_probe_sequence_distillation_rollout(
            scenario_name="continuous_survival_return_after_late_forage_v1",
            episodes=1,
            episode_start=4,
        )
        self.assertGreater(len(dataset), 3)
        self.assertEqual(dataset.to_summary()["episode_ids"], [4])
        self.assertEqual(
            dataset.teacher_metadata["source"],
            "scripted_direct_policy_post_rest_probe_sequence",
        )
        self.assertIn("MOVE_LEFT", dataset.teacher_metadata["scripted_sequence"])

    def test_collect_direct_policy_probe_family_distillation_rollout_collects_global_teacher_samples(self) -> None:
        sim = SpiderSimulation(
            seed=11,
            max_steps=20,
            brain_config=resolve_ablation_configs(
                [
                    "true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_backbone_post_rest_probe_family_distill_option_replay_policy"
                ],
                module_dropout=0.0,
            )[0],
        )
        dataset = sim.collect_direct_policy_probe_family_distillation_rollout(
            episodes=1,
            episode_start=5,
        )
        self.assertGreater(len(dataset), 20)
        self.assertEqual(dataset.to_summary()["episode_ids"], [5, 6])
        self.assertEqual(
            dataset.teacher_metadata["source"],
            "scripted_direct_policy_post_rest_probe_family",
        )
        self.assertEqual(
            dataset.teacher_metadata["teacher_scenarios"],
            ["continuous_survival_canonical", "continuous_survival_easy_v1"],
        )

    def test_collect_direct_policy_probe_handoff_distillation_rollout_collects_handoff_samples(self) -> None:
        sim = SpiderSimulation(
            seed=11,
            max_steps=20,
            brain_config=resolve_ablation_configs(
                [
                    "true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_backbone_post_rest_probe_handoff_distill_option_replay_policy"
                ],
                module_dropout=0.0,
            )[0],
        )
        dataset = sim.collect_direct_policy_probe_handoff_distillation_rollout(
            episodes=1,
            episode_start=7,
        )
        self.assertGreater(len(dataset), 20)
        self.assertEqual(dataset.to_summary()["episode_ids"], [7, 8])
        self.assertEqual(
            dataset.teacher_metadata["source"],
            "scripted_direct_policy_post_rest_probe_handoff",
        )

    def test_collect_direct_policy_probe_trajectory_distillation_rollout_collects_redirected_samples(self) -> None:
        sim = SpiderSimulation(
            seed=11,
            max_steps=20,
            brain_config=resolve_ablation_configs(
                [
                    "true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_backbone_post_rest_probe_trajectory_distill_option_replay_policy"
                ],
                module_dropout=0.0,
            )[0],
        )
        dataset = sim.collect_direct_policy_probe_trajectory_distillation_rollout(
            episodes=1,
            episode_start=9,
        )
        self.assertGreater(len(dataset), 40)
        self.assertEqual(dataset.to_summary()["episode_ids"], [9, 10])
        self.assertEqual(
            dataset.teacher_metadata["source"],
            "redirected_direct_policy_post_rest_probe_trajectory",
        )
        self.assertIn(
            "trajectory_release",
            set(dataset.teacher_metadata.get("action_stages", [])),
        )

    def test_collect_direct_policy_probe_cycle_distillation_rollout_collects_cycle_samples(self) -> None:
        sim = SpiderSimulation(
            seed=11,
            max_steps=20,
            brain_config=resolve_ablation_configs(
                [
                    "true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_backbone_post_rest_probe_cycle_distill_option_replay_policy"
                ],
                module_dropout=0.0,
            )[0],
        )
        dataset = sim.collect_direct_policy_probe_cycle_distillation_rollout(
            episodes=1,
            episode_start=11,
        )
        self.assertGreater(len(dataset), 40)
        self.assertEqual(dataset.to_summary()["episode_ids"], [11, 12])
        self.assertEqual(
            dataset.teacher_metadata["source"],
            "redirected_direct_policy_post_rest_probe_cycle",
        )
        self.assertIn(
            "cycle_release",
            set(dataset.teacher_metadata.get("action_stages", [])),
        )

    def test_collect_direct_policy_probe_frontier_teacher_distillation_rollout_collects_fixed_teacher_samples(self) -> None:
        sim = SpiderSimulation(
            seed=17,
            max_steps=20,
            brain_config=resolve_ablation_configs(
                [
                    "true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_backbone_post_rest_probe_frontier_teacher_distill_option_replay_policy"
                ],
                module_dropout=0.0,
            )[0],
        )
        dataset = sim.collect_direct_policy_probe_frontier_teacher_distillation_rollout(
            episodes=1,
            episode_start=17,
        )
        self.assertGreater(len(dataset), 20)
        self.assertEqual(dataset.to_summary()["episode_ids"], [17, 18])
        self.assertEqual(
            dataset.teacher_metadata["source"],
            "checkpoint_direct_policy_post_rest_probe_frontier_teacher",
        )
        self.assertIn(
            "architecture version is incompatible",
            str(dataset.teacher_metadata["teacher_checkpoint_error"]),
        )
        self.assertIn(
            "rollout_release",
            set(dataset.teacher_metadata.get("action_stages", [])),
        )
        self.assertIn(
            "rollout_live",
            set(dataset.teacher_metadata.get("action_stages", [])),
        )

    def test_collect_direct_policy_probe_replayable_teacher_distillation_rollout_collects_probe_samples(self) -> None:
        sim = SpiderSimulation(
            seed=19,
            max_steps=20,
            brain_config=resolve_ablation_configs(
                [
                    "true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_backbone_post_rest_probe_replayable_teacher_distill_option_replay_policy"
                ],
                module_dropout=0.0,
            )[0],
        )
        dataset = sim.collect_direct_policy_probe_replayable_teacher_distillation_rollout(
            episodes=1,
            episode_start=19,
        )
        self.assertGreater(len(dataset), 20)
        self.assertEqual(dataset.to_summary()["episode_ids"], [19])
        self.assertEqual(
            dataset.teacher_metadata["source"],
            "stateful_direct_policy_post_rest_probe_replayable_teacher",
        )
        self.assertGreaterEqual(
            int(dataset.teacher_metadata.get("first_up_redirects", 0)),
            1,
        )
        self.assertGreaterEqual(
            int(dataset.teacher_metadata.get("down_redirects", 0)),
            1,
        )
        self.assertIn(
            "probe_release",
            set(dataset.teacher_metadata.get("action_stages", [])),
        )
        self.assertIn(
            "probe_down_continue",
            set(dataset.teacher_metadata.get("action_stages", [])),
        )
