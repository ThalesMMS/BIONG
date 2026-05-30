from __future__ import annotations

from .shared import *
from .corticalmodulebankrecurrentmodulestest_helpers import CorticalModuleBankRecurrentModulesTestHelpers



class CorticalModuleBankRecurrentModulesTestPart2(CorticalModuleBankRecurrentModulesTestHelpers, unittest.TestCase):
    def test_executive_post_exit_food_guidance_prefers_move_right_from_food_memory(self) -> None:
        brain = SpiderBrain(
            seed=115,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="true_monolithic_executive_option_post_exit_food_guidance_policy",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=False,
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
                direct_policy_phase_option_feedback=True,
                direct_policy_option_transition_feedback=True,
                direct_policy_option_termination_cooldown=True,
                direct_policy_option_decoder_state=True,
                direct_policy_option_recurrent_dynamics=True,
                direct_policy_option_decoder_recurrent_state=True,
                direct_policy_option_action_recurrent_core=True,
                direct_policy_option_action_separate_recurrent_head=True,
                direct_policy_option_action_separate_policy_path=True,
                direct_policy_option_action_separate_backbone=True,
                direct_policy_executive_physiology_option_gating=True,
                direct_policy_executive_affordance_action_gating=True,
                direct_policy_executive_option_action_masking=True,
                direct_policy_executive_event_release_latching=True,
                direct_policy_executive_event_release_action_commitment=True,
                direct_policy_executive_release_phase_state=True,
                direct_policy_executive_release_exit_contract=True,
                direct_policy_executive_release_substate_progression=True,
                direct_policy_executive_post_exit_continuation=True,
                direct_policy_executive_post_exit_food_guidance=True,
            ),
        )
        recurrent = brain.true_monolithic_policy
        self.assertIsInstance(
            recurrent,
            RecurrentOptionAffordancePositionFeedbackTrueMonolithicNetwork,
        )
        recurrent.reset_hidden_state()
        recurrent.reset_event_memory()
        recurrent.current_option_idx = OPTION_NAMES.index("POST_REST_REACTIVATE")
        recurrent.current_option_age = 1
        recurrent.current_option_steps_remaining = 2
        recurrent.executive_post_exit_steps_remaining = 2
        recurrent.W_xh.fill(0.0)
        recurrent.W_hh.fill(0.0)
        recurrent.b_h.fill(0.0)
        recurrent.W2_option.fill(0.0)
        recurrent.b2_option.fill(0.0)
        recurrent.W2_option_feedback.fill(0.0)
        recurrent.b2_option_feedback.fill(0.0)
        recurrent.W_affordance_feedback.fill(0.0)
        recurrent.b_affordance_feedback.fill(0.0)
        recurrent.W2_policy.fill(0.0)
        recurrent.b2_policy.fill(0.0)
        recurrent.b2_policy[ACTION_TO_INDEX["MOVE_DOWN"]] = 10.0
        decision = brain.act_inference(
            _build_observation(
                sleep={
                    "fatigue": 0.05,
                    "hunger": 0.18,
                    "on_shelter": 0.0,
                    "night": 0.0,
                    "sleep_phase_level": 0.0,
                    "rest_streak_norm": 0.6,
                    "sleep_debt": 0.04,
                    "shelter_role_level": 0.0,
                    "shelter_memory_age": 0.0,
                },
                hunger={
                    "food_memory_dx": 0.8,
                    "food_memory_dy": 0.1,
                    "food_memory_age": 0.1,
                },
            ),
            sample=False,
        )
        self.assertEqual(LOCOMOTION_ACTIONS[decision.action_idx], "MOVE_RIGHT")

    def test_executive_post_exit_food_commitment_extends_outside_window(self) -> None:
        brain = SpiderBrain(
            seed=118,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="true_monolithic_executive_option_post_exit_food_commitment_policy",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=False,
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
                direct_policy_phase_option_feedback=True,
                direct_policy_option_transition_feedback=True,
                direct_policy_option_termination_cooldown=True,
                direct_policy_option_decoder_state=True,
                direct_policy_option_recurrent_dynamics=True,
                direct_policy_option_decoder_recurrent_state=True,
                direct_policy_option_action_recurrent_core=True,
                direct_policy_option_action_separate_recurrent_head=True,
                direct_policy_option_action_separate_policy_path=True,
                direct_policy_option_action_separate_backbone=True,
                direct_policy_executive_physiology_option_gating=True,
                direct_policy_executive_affordance_action_gating=True,
                direct_policy_executive_option_action_masking=True,
                direct_policy_executive_event_release_latching=True,
                direct_policy_executive_event_release_action_commitment=True,
                direct_policy_executive_release_phase_state=True,
                direct_policy_executive_release_exit_contract=True,
                direct_policy_executive_release_substate_progression=True,
                direct_policy_executive_post_exit_continuation=True,
                direct_policy_executive_post_exit_food_guidance=True,
                direct_policy_executive_post_exit_food_commitment=True,
            ),
        )
        recurrent = brain.true_monolithic_policy
        self.assertIsInstance(
            recurrent,
            RecurrentOptionAffordancePositionFeedbackTrueMonolithicNetwork,
        )
        recurrent.reset_hidden_state()
        recurrent.reset_event_memory()
        recurrent.current_option_idx = OPTION_NAMES.index("POST_REST_REACTIVATE")
        recurrent.current_option_age = 3
        recurrent.current_option_steps_remaining = 0
        recurrent.executive_post_exit_steps_remaining = 1
        recurrent.W_xh.fill(0.0)
        recurrent.W_hh.fill(0.0)
        recurrent.b_h.fill(0.0)
        recurrent.W2_option.fill(0.0)
        recurrent.b2_option.fill(0.0)
        recurrent.W2_option_feedback.fill(0.0)
        recurrent.b2_option_feedback.fill(0.0)
        recurrent.W_affordance_feedback.fill(0.0)
        recurrent.b_affordance_feedback.fill(0.0)
        recurrent.W2_policy.fill(0.0)
        recurrent.b2_policy.fill(0.0)
        recurrent.b2_policy[ACTION_TO_INDEX["MOVE_DOWN"]] = 10.0
        decision = brain.act_inference(
            _build_observation(
                sleep={
                    "fatigue": 0.05,
                    "hunger": 0.18,
                    "on_shelter": 0.0,
                    "night": 0.0,
                    "sleep_phase_level": 0.0,
                    "rest_streak_norm": 0.6,
                    "sleep_debt": 0.04,
                    "shelter_role_level": 0.0,
                    "shelter_memory_age": 0.0,
                },
                hunger={
                    "food_memory_dx": 0.8,
                    "food_memory_dy": 0.1,
                    "food_memory_age": 0.1,
                },
            ),
            sample=False,
        )
        self.assertEqual(decision.selected_option, "POST_REST_REACTIVATE")
        self.assertGreaterEqual(recurrent.executive_post_exit_steps_remaining, 1)

    def test_executive_post_exit_food_progression_prefers_move_up_over_memory_reversal(self) -> None:
        brain = SpiderBrain(
            seed=121,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="true_monolithic_executive_option_post_exit_food_progression_policy",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=False,
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
                direct_policy_phase_option_feedback=True,
                direct_policy_option_transition_feedback=True,
                direct_policy_option_termination_cooldown=True,
                direct_policy_option_decoder_state=True,
                direct_policy_option_recurrent_dynamics=True,
                direct_policy_option_decoder_recurrent_state=True,
                direct_policy_option_action_recurrent_core=True,
                direct_policy_option_action_separate_recurrent_head=True,
                direct_policy_option_action_separate_policy_path=True,
                direct_policy_option_action_separate_backbone=True,
                direct_policy_executive_physiology_option_gating=True,
                direct_policy_executive_affordance_action_gating=True,
                direct_policy_executive_option_action_masking=True,
                direct_policy_executive_event_release_latching=True,
                direct_policy_executive_event_release_action_commitment=True,
                direct_policy_executive_release_phase_state=True,
                direct_policy_executive_release_exit_contract=True,
                direct_policy_executive_release_substate_progression=True,
                direct_policy_executive_post_exit_continuation=True,
                direct_policy_executive_post_exit_food_guidance=True,
                direct_policy_executive_post_exit_food_progression=True,
            ),
        )
        recurrent = brain.true_monolithic_policy
        self.assertIsInstance(
            recurrent,
            RecurrentOptionAffordancePositionFeedbackTrueMonolithicNetwork,
        )
        recurrent.reset_hidden_state()
        recurrent.reset_event_memory()
        recurrent.current_option_idx = OPTION_NAMES.index("POST_REST_REACTIVATE")
        recurrent.current_option_age = 1
        recurrent.current_option_steps_remaining = 2
        recurrent.executive_post_exit_steps_remaining = 2
        recurrent.previous_action_idx = ACTION_TO_INDEX["MOVE_LEFT"]
        recurrent.W_xh.fill(0.0)
        recurrent.W_hh.fill(0.0)
        recurrent.b_h.fill(0.0)
        recurrent.W2_option.fill(0.0)
        recurrent.b2_option.fill(0.0)
        recurrent.W2_option_feedback.fill(0.0)
        recurrent.b2_option_feedback.fill(0.0)
        recurrent.W_affordance_feedback.fill(0.0)
        recurrent.b_affordance_feedback.fill(0.0)
        recurrent.W2_policy.fill(0.0)
        recurrent.b2_policy.fill(0.0)
        recurrent.b2_policy[ACTION_TO_INDEX["MOVE_RIGHT"]] = 10.0
        decision = brain.act_inference(
            _build_observation(
                sleep={
                    "fatigue": 0.05,
                    "hunger": 0.18,
                    "on_shelter": 0.0,
                    "night": 0.0,
                    "sleep_phase_level": 0.0,
                    "rest_streak_norm": 0.6,
                    "sleep_debt": 0.04,
                    "shelter_role_level": 0.0,
                    "shelter_memory_age": 0.0,
                },
                hunger={
                    "food_memory_dx": 0.8,
                    "food_memory_dy": 0.1,
                    "food_memory_age": 0.1,
                },
            ),
            sample=False,
        )
        self.assertEqual(LOCOMOTION_ACTIONS[decision.action_idx], "MOVE_UP")

    def test_executive_post_exit_food_heading_progression_prefers_move_up_after_right_recenter(self) -> None:
        brain = SpiderBrain(
            seed=124,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="true_monolithic_executive_option_post_exit_food_heading_progression_policy",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=False,
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
                direct_policy_phase_option_feedback=True,
                direct_policy_option_transition_feedback=True,
                direct_policy_option_termination_cooldown=True,
                direct_policy_option_decoder_state=True,
                direct_policy_option_recurrent_dynamics=True,
                direct_policy_option_decoder_recurrent_state=True,
                direct_policy_option_action_recurrent_core=True,
                direct_policy_option_action_separate_recurrent_head=True,
                direct_policy_option_action_separate_policy_path=True,
                direct_policy_option_action_separate_backbone=True,
                direct_policy_executive_physiology_option_gating=True,
                direct_policy_executive_affordance_action_gating=True,
                direct_policy_executive_option_action_masking=True,
                direct_policy_executive_event_release_latching=True,
                direct_policy_executive_event_release_action_commitment=True,
                direct_policy_executive_release_phase_state=True,
                direct_policy_executive_release_exit_contract=True,
                direct_policy_executive_release_substate_progression=True,
                direct_policy_executive_post_exit_continuation=True,
                direct_policy_executive_post_exit_food_guidance=True,
                direct_policy_executive_post_exit_food_heading_progression=True,
            ),
        )
        recurrent = brain.true_monolithic_policy
        self.assertIsInstance(
            recurrent,
            RecurrentOptionAffordancePositionFeedbackTrueMonolithicNetwork,
        )
        recurrent.reset_hidden_state()
        recurrent.reset_event_memory()
        recurrent.current_option_idx = OPTION_NAMES.index("POST_REST_REACTIVATE")
        recurrent.current_option_age = 1
        recurrent.current_option_steps_remaining = 2
        recurrent.executive_post_exit_steps_remaining = 2
        recurrent.previous_action_idx = ACTION_TO_INDEX["MOVE_RIGHT"]
        recurrent.W_xh.fill(0.0)
        recurrent.W_hh.fill(0.0)
        recurrent.b_h.fill(0.0)
        recurrent.W2_option.fill(0.0)
        recurrent.b2_option.fill(0.0)
        recurrent.W2_option_feedback.fill(0.0)
        recurrent.b2_option_feedback.fill(0.0)
        recurrent.W_affordance_feedback.fill(0.0)
        recurrent.b_affordance_feedback.fill(0.0)
        recurrent.W2_policy.fill(0.0)
        recurrent.b2_policy.fill(0.0)
        recurrent.b2_policy[ACTION_TO_INDEX["ORIENT_UP"]] = 10.0
        decision = brain.act_inference(
            _build_observation(
                sleep={
                    "fatigue": 0.05,
                    "hunger": 0.18,
                    "on_shelter": 0.0,
                    "night": 0.0,
                    "sleep_phase_level": 0.0,
                    "rest_streak_norm": 0.6,
                    "sleep_debt": 0.04,
                    "shelter_role_level": 0.0,
                    "shelter_memory_age": 0.0,
                },
                hunger={
                    "food_memory_dx": 0.8,
                    "food_memory_dy": 0.1,
                    "food_memory_age": 0.1,
                },
            ),
            sample=False,
        )
        self.assertEqual(LOCOMOTION_ACTIONS[decision.action_idx], "MOVE_UP")

    def test_executive_post_exit_smell_progression_extends_outside_window(self) -> None:
        brain = SpiderBrain(
            seed=127,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="true_monolithic_executive_option_post_exit_smell_progression_policy",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=False,
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
                direct_policy_phase_option_feedback=True,
                direct_policy_option_transition_feedback=True,
                direct_policy_option_termination_cooldown=True,
                direct_policy_option_decoder_state=True,
                direct_policy_option_recurrent_dynamics=True,
                direct_policy_option_decoder_recurrent_state=True,
                direct_policy_option_action_recurrent_core=True,
                direct_policy_option_action_separate_recurrent_head=True,
                direct_policy_option_action_separate_policy_path=True,
                direct_policy_option_action_separate_backbone=True,
                direct_policy_executive_physiology_option_gating=True,
                direct_policy_executive_affordance_action_gating=True,
                direct_policy_executive_option_action_masking=True,
                direct_policy_executive_event_release_latching=True,
                direct_policy_executive_event_release_action_commitment=True,
                direct_policy_executive_release_phase_state=True,
                direct_policy_executive_release_exit_contract=True,
                direct_policy_executive_release_substate_progression=True,
                direct_policy_executive_post_exit_continuation=True,
                direct_policy_executive_post_exit_food_guidance=True,
                direct_policy_executive_post_exit_smell_progression=True,
            ),
        )
        recurrent = brain.true_monolithic_policy
        self.assertIsInstance(
            recurrent,
            RecurrentOptionAffordancePositionFeedbackTrueMonolithicNetwork,
        )
        recurrent.reset_hidden_state()
        recurrent.reset_event_memory()
        recurrent.current_option_idx = OPTION_NAMES.index("POST_REST_REACTIVATE")
        recurrent.current_option_age = 1
        recurrent.current_option_steps_remaining = 2
        observation = _build_observation(
            sleep={
                "fatigue": 0.05,
                "hunger": 0.18,
                "on_shelter": 0.0,
                "night": 0.0,
                "sleep_phase_level": 0.0,
                "rest_streak_norm": 0.6,
                "sleep_debt": 0.04,
                "shelter_role_level": 0.0,
                "shelter_memory_age": 0.0,
            },
            hunger={
                "food_smell_strength": 0.4,
                "food_smell_dx": 0.8,
                "food_smell_dy": 0.1,
                "food_memory_age": 0.95,
            },
        )
        monolithic_observation = brain._build_monolithic_observation(observation)
        termination_reason = recurrent._apply_executive_post_exit_continuation(
            monolithic_observation,
            "shelter_exited",
        )
        self.assertIsNone(termination_reason)
        self.assertEqual(recurrent.executive_post_exit_steps_remaining, 3)

    def test_executive_post_exit_corridor_progression_prefers_blind_rightward_advance(self) -> None:
        brain = SpiderBrain(
            seed=128,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="true_monolithic_executive_option_post_exit_corridor_progression_policy",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=False,
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
                direct_policy_phase_option_feedback=True,
                direct_policy_option_transition_feedback=True,
                direct_policy_option_termination_cooldown=True,
                direct_policy_option_decoder_state=True,
                direct_policy_option_recurrent_dynamics=True,
                direct_policy_option_decoder_recurrent_state=True,
                direct_policy_option_action_recurrent_core=True,
                direct_policy_option_action_separate_recurrent_head=True,
                direct_policy_option_action_separate_policy_path=True,
                direct_policy_option_action_separate_backbone=True,
                direct_policy_executive_physiology_option_gating=True,
                direct_policy_executive_affordance_action_gating=True,
                direct_policy_executive_option_action_masking=True,
                direct_policy_executive_event_release_latching=True,
                direct_policy_executive_event_release_action_commitment=True,
                direct_policy_executive_release_phase_state=True,
                direct_policy_executive_release_exit_contract=True,
                direct_policy_executive_release_substate_progression=True,
                direct_policy_executive_post_exit_continuation=True,
                direct_policy_executive_post_exit_corridor_progression=True,
            ),
        )
        recurrent = brain.true_monolithic_policy
        self.assertIsInstance(
            recurrent,
            RecurrentOptionAffordancePositionFeedbackTrueMonolithicNetwork,
        )
        recurrent.reset_hidden_state()
        recurrent.reset_event_memory()
        recurrent.current_option_idx = OPTION_NAMES.index("POST_REST_REACTIVATE")
        recurrent.current_option_age = 1
        recurrent.current_option_steps_remaining = 3
        recurrent.executive_post_exit_steps_remaining = 7
        recurrent.executive_post_exit_corridor_steps_remaining = 7
        observation = _build_observation(
            sleep={
                "fatigue": 0.05,
                "hunger": 0.18,
                "on_shelter": 0.0,
                "night": 0.0,
                "sleep_phase_level": 0.0,
                "rest_streak_norm": 0.6,
                "sleep_debt": 0.04,
                "shelter_role_level": 0.0,
                "shelter_memory_age": 0.0,
            },
            hunger={
                "food_smell_strength": 0.0,
                "food_memory_age": 1.0,
            },
        )
        decision = brain.act_inference(observation, sample=False)
        self.assertEqual(LOCOMOTION_ACTIONS[decision.action_idx], "MOVE_RIGHT")

    def test_executive_post_exit_corridor_affordance_progression_delays_drop_until_outside(self) -> None:
        brain = SpiderBrain(
            seed=129,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="true_monolithic_executive_option_post_exit_corridor_affordance_progression_policy",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=False,
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
                direct_policy_phase_option_feedback=True,
                direct_policy_option_transition_feedback=True,
                direct_policy_option_termination_cooldown=True,
                direct_policy_option_decoder_state=True,
                direct_policy_option_recurrent_dynamics=True,
                direct_policy_option_decoder_recurrent_state=True,
                direct_policy_option_action_recurrent_core=True,
                direct_policy_option_action_separate_recurrent_head=True,
                direct_policy_option_action_separate_policy_path=True,
                direct_policy_option_action_separate_backbone=True,
                direct_policy_executive_physiology_option_gating=True,
                direct_policy_executive_affordance_action_gating=True,
                direct_policy_executive_option_action_masking=True,
                direct_policy_executive_event_release_latching=True,
                direct_policy_executive_event_release_action_commitment=True,
                direct_policy_executive_release_phase_state=True,
                direct_policy_executive_release_exit_contract=True,
                direct_policy_executive_release_substate_progression=True,
                direct_policy_executive_post_exit_continuation=True,
                direct_policy_executive_post_exit_corridor_progression=True,
                direct_policy_executive_post_exit_corridor_affordance_progression=True,
            ),
        )
        recurrent = brain.true_monolithic_policy
        self.assertIsInstance(
            recurrent,
            RecurrentOptionAffordancePositionFeedbackTrueMonolithicNetwork,
        )
        recurrent.reset_hidden_state()
        recurrent.reset_event_memory()
        recurrent.current_option_idx = OPTION_NAMES.index("POST_REST_REACTIVATE")
        recurrent.current_option_age = 1
        recurrent.current_option_steps_remaining = 3
        recurrent.executive_post_exit_steps_remaining = 7
        recurrent.executive_post_exit_corridor_steps_remaining = 7
        recurrent.W_xh.fill(0.0)
        recurrent.W_hh.fill(0.0)
        recurrent.b_h.fill(0.0)
        recurrent.W2_option.fill(0.0)
        recurrent.b2_option.fill(0.0)
        recurrent.W2_option_feedback.fill(0.0)
        recurrent.b2_option_feedback.fill(0.0)
        recurrent.W_affordance_feedback.fill(0.0)
        recurrent.b_affordance_feedback.fill(0.0)
        recurrent.W2_policy.fill(0.0)
        recurrent.b2_policy.fill(0.0)
        recurrent.b2_policy[ACTION_TO_INDEX["MOVE_DOWN"]] = 10.0
        recurrent.W2_shelter_position.fill(0.0)
        recurrent.b2_shelter_position.fill(0.0)
        pos_dim = len(AFFORDANCE_SHELTER_POSITION_NAMES)
        move_down_idx = ACTION_TO_INDEX["MOVE_DOWN"]
        move_right_idx = ACTION_TO_INDEX["MOVE_RIGHT"]
        entrance_center_idx = AFFORDANCE_SHELTER_POSITION_NAMES.index("entrance_center")
        outside_idx = AFFORDANCE_SHELTER_POSITION_NAMES.index("outside")
        recurrent.b2_shelter_position[move_down_idx * pos_dim + entrance_center_idx] = 6.0
        recurrent.b2_shelter_position[move_right_idx * pos_dim + outside_idx] = 6.0

        shelter_observation = _build_observation(
            sleep={
                "fatigue": 0.05,
                "hunger": 0.18,
                "on_shelter": 0.0,
                "night": 0.0,
                "sleep_phase_level": 0.0,
                "rest_streak_norm": 0.6,
                "sleep_debt": 0.04,
                "shelter_role_level": 0.0,
                "shelter_memory_age": 0.0,
            },
            hunger={
                "food_smell_strength": 0.0,
                "food_memory_age": 1.0,
            },
        )
        decision = brain.act_inference(shelter_observation, sample=False)
        self.assertEqual(LOCOMOTION_ACTIONS[decision.action_idx], "MOVE_RIGHT")

        recurrent.reset_hidden_state()
        recurrent.reset_event_memory()
        recurrent.current_option_idx = OPTION_NAMES.index("POST_REST_REACTIVATE")
        recurrent.current_option_age = 1
        recurrent.current_option_steps_remaining = 3
        recurrent.executive_post_exit_steps_remaining = 4
        recurrent.executive_post_exit_corridor_steps_remaining = 4
        recurrent.W_xh.fill(0.0)
        recurrent.W_hh.fill(0.0)
        recurrent.b_h.fill(0.0)
        recurrent.W2_option.fill(0.0)
        recurrent.b2_option.fill(0.0)
        recurrent.W2_option_feedback.fill(0.0)
        recurrent.b2_option_feedback.fill(0.0)
        recurrent.W_affordance_feedback.fill(0.0)
        recurrent.b_affordance_feedback.fill(0.0)
        recurrent.W2_policy.fill(0.0)
        recurrent.b2_policy.fill(0.0)
        recurrent.b2_policy[ACTION_TO_INDEX["MOVE_RIGHT"]] = 10.0
        recurrent.W2_shelter_position.fill(0.0)
        recurrent.b2_shelter_position.fill(0.0)
        recurrent.b2_shelter_position[move_down_idx * pos_dim + outside_idx] = 6.0
        recurrent.b2_shelter_position[move_right_idx * pos_dim + outside_idx] = 6.0
        outside_observation = _build_observation(
            sleep={
                "fatigue": 0.05,
                "hunger": 0.18,
                "on_shelter": 0.0,
                "night": 0.0,
                "sleep_phase_level": 0.0,
                "rest_streak_norm": 0.6,
                "sleep_debt": 0.04,
                "shelter_role_level": 0.0,
                "shelter_memory_age": 0.0,
            },
            hunger={
                "food_smell_strength": 0.0,
                "food_memory_age": 1.0,
            },
        )
        decision = brain.act_inference(outside_observation, sample=False)
        self.assertEqual(LOCOMOTION_ACTIONS[decision.action_idx], "MOVE_DOWN")

    def test_executive_post_food_return_prefers_shelterward_recovery_and_deep_stay(self) -> None:
        brain = SpiderBrain(
            seed=130,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="true_monolithic_executive_option_post_food_return_policy",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=False,
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
                direct_policy_phase_option_feedback=True,
                direct_policy_option_transition_feedback=True,
                direct_policy_option_termination_cooldown=True,
                direct_policy_option_decoder_state=True,
                direct_policy_option_recurrent_dynamics=True,
                direct_policy_option_decoder_recurrent_state=True,
                direct_policy_option_action_recurrent_core=True,
                direct_policy_option_action_separate_recurrent_head=True,
                direct_policy_option_action_separate_policy_path=True,
                direct_policy_option_action_separate_backbone=True,
                direct_policy_executive_physiology_option_gating=True,
                direct_policy_executive_affordance_action_gating=True,
                direct_policy_executive_option_action_masking=True,
                direct_policy_executive_event_release_latching=True,
                direct_policy_executive_event_release_action_commitment=True,
                direct_policy_executive_release_phase_state=True,
                direct_policy_executive_release_exit_contract=True,
                direct_policy_executive_release_substate_progression=True,
                direct_policy_executive_post_exit_continuation=True,
                direct_policy_executive_post_exit_corridor_progression=True,
                direct_policy_executive_post_exit_corridor_affordance_progression=True,
                direct_policy_executive_post_food_return=True,
            ),
        )
        recurrent = brain.true_monolithic_policy
        self.assertIsInstance(
            recurrent,
            RecurrentOptionAffordancePositionFeedbackTrueMonolithicNetwork,
        )
        recurrent.reset_hidden_state()
        recurrent.reset_event_memory()
        recurrent.current_option_idx = OPTION_NAMES.index("RETURN_TO_SHELTER")
        recurrent.current_option_age = 1
        recurrent.current_option_steps_remaining = 3
        recurrent.executive_post_food_return_steps_remaining = 8
        recurrent.W_xh.fill(0.0)
        recurrent.W_hh.fill(0.0)
        recurrent.b_h.fill(0.0)
        recurrent.W2_option.fill(0.0)
        recurrent.b2_option.fill(0.0)
        recurrent.W2_option_feedback.fill(0.0)
        recurrent.b2_option_feedback.fill(0.0)
        recurrent.W_affordance_feedback.fill(0.0)
        recurrent.b_affordance_feedback.fill(0.0)
        recurrent.W2_policy.fill(0.0)
        recurrent.b2_policy.fill(0.0)
        recurrent.b2_policy[ACTION_TO_INDEX["MOVE_RIGHT"]] = 10.0
        recurrent.W2_shelter_position.fill(0.0)
        recurrent.b2_shelter_position.fill(0.0)
        pos_dim = len(AFFORDANCE_SHELTER_POSITION_NAMES)
        move_left_idx = ACTION_TO_INDEX["MOVE_LEFT"]
        move_right_idx = ACTION_TO_INDEX["MOVE_RIGHT"]
        stay_idx = ACTION_TO_INDEX["STAY"]
        deep_center_idx = AFFORDANCE_SHELTER_POSITION_NAMES.index("deep_center")
        outside_idx = AFFORDANCE_SHELTER_POSITION_NAMES.index("outside")
        recurrent.b2_shelter_position[move_left_idx * pos_dim + deep_center_idx] = 8.0
        recurrent.b2_shelter_position[move_right_idx * pos_dim + outside_idx] = 8.0

        outside_observation = _build_observation(
            sleep={
                "fatigue": 0.05,
                "hunger": 0.12,
                "on_shelter": 0.0,
                "night": 0.0,
                "sleep_phase_level": 0.0,
                "rest_streak_norm": 0.2,
                "sleep_debt": 0.05,
                "shelter_role_level": 0.0,
                "shelter_memory_age": 0.1,
            },
            predator={
                "predator_visible": 0.0,
                "predator_proximity": 0.0,
                "predator_motion": 0.0,
                "predator_heading_alignment": 0.0,
                "predator_recent_contact": 0.0,
                "recent_pain": 0.0,
            },
        )
        decision = brain.act_inference(outside_observation, sample=False)
        self.assertEqual(LOCOMOTION_ACTIONS[decision.action_idx], "MOVE_LEFT")

        recurrent.current_option_idx = OPTION_NAMES.index("DEEPEN_IN_SHELTER")
        recurrent.current_option_steps_remaining = 2
        recurrent.executive_post_food_return_steps_remaining = 4
        recurrent.W2_shelter_position.fill(0.0)
        recurrent.b2_shelter_position.fill(0.0)
        recurrent.b2_shelter_position[stay_idx * pos_dim + deep_center_idx] = 8.0
        deep_observation = _build_observation(
            sleep={
                "fatigue": 0.08,
                "hunger": 0.08,
                "on_shelter": 1.0,
                "night": 1.0,
                "sleep_phase_level": 0.0,
                "rest_streak_norm": 0.4,
                "sleep_debt": 0.08,
                "shelter_role_level": 1.0,
                "shelter_memory_age": 0.0,
            },
            predator={
                "predator_visible": 0.0,
                "predator_proximity": 0.0,
                "predator_motion": 0.0,
                "predator_heading_alignment": 0.0,
                "predator_recent_contact": 0.0,
                "recent_pain": 0.0,
            },
        )
        deep_decision = brain.act_inference(deep_observation, sample=False)
        self.assertEqual(LOCOMOTION_ACTIONS[deep_decision.action_idx], "STAY")

    def test_executive_post_food_vector_return_uses_shelter_memory_direction(self) -> None:
        brain = SpiderBrain(
            seed=131,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="true_monolithic_executive_option_post_food_vector_return_policy",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=False,
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
                direct_policy_phase_option_feedback=True,
                direct_policy_option_transition_feedback=True,
                direct_policy_option_termination_cooldown=True,
                direct_policy_option_decoder_state=True,
                direct_policy_option_recurrent_dynamics=True,
                direct_policy_option_decoder_recurrent_state=True,
                direct_policy_option_action_recurrent_core=True,
                direct_policy_option_action_separate_recurrent_head=True,
                direct_policy_option_action_separate_policy_path=True,
                direct_policy_option_action_separate_backbone=True,
                direct_policy_executive_physiology_option_gating=True,
                direct_policy_executive_affordance_action_gating=True,
                direct_policy_executive_option_action_masking=True,
                direct_policy_executive_event_release_latching=True,
                direct_policy_executive_event_release_action_commitment=True,
                direct_policy_executive_release_phase_state=True,
                direct_policy_executive_release_exit_contract=True,
                direct_policy_executive_release_substate_progression=True,
                direct_policy_executive_post_exit_continuation=True,
                direct_policy_executive_post_exit_corridor_progression=True,
                direct_policy_executive_post_exit_corridor_affordance_progression=True,
                direct_policy_executive_post_food_return=True,
                direct_policy_executive_post_food_vector_return=True,
            ),
        )
        recurrent = brain.true_monolithic_policy
        self.assertIsInstance(
            recurrent,
            RecurrentOptionAffordancePositionFeedbackTrueMonolithicNetwork,
        )
        recurrent.reset_hidden_state()
        recurrent.reset_event_memory()
        recurrent.current_option_idx = OPTION_NAMES.index("RETURN_TO_SHELTER")
        recurrent.current_option_age = 1
        recurrent.current_option_steps_remaining = 3
        recurrent.executive_post_food_return_steps_remaining = 8
        recurrent.W_xh.fill(0.0)
        recurrent.W_hh.fill(0.0)
        recurrent.b_h.fill(0.0)
        recurrent.W2_option.fill(0.0)
        recurrent.b2_option.fill(0.0)
        recurrent.W2_option_feedback.fill(0.0)
        recurrent.b2_option_feedback.fill(0.0)
        recurrent.W_affordance_feedback.fill(0.0)
        recurrent.b_affordance_feedback.fill(0.0)
        recurrent.W2_policy.fill(0.0)
        recurrent.b2_policy.fill(0.0)
        recurrent.b2_policy[ACTION_TO_INDEX["MOVE_RIGHT"]] = 10.0
        recurrent.W2_shelter_position.fill(0.0)
        recurrent.b2_shelter_position.fill(0.0)
        pos_dim = len(AFFORDANCE_SHELTER_POSITION_NAMES)
        move_right_idx = ACTION_TO_INDEX["MOVE_RIGHT"]
        outside_idx = AFFORDANCE_SHELTER_POSITION_NAMES.index("outside")
        recurrent.b2_shelter_position[move_right_idx * pos_dim + outside_idx] = 8.0
        recurrent.set_runtime_observation_meta(
            {
                "memory_vectors": {
                    "shelter": {"dx": -0.8, "dy": 0.0, "age": 0.0, "ttl": 20}
                }
            }
        )

        observation = _build_observation(
            sleep={
                "fatigue": 0.05,
                "hunger": 0.12,
                "on_shelter": 0.0,
                "night": 0.0,
                "sleep_phase_level": 0.0,
                "rest_streak_norm": 0.2,
                "sleep_debt": 0.05,
                "shelter_role_level": 0.0,
                "shelter_memory_age": 0.1,
            },
            predator={
                "predator_visible": 0.0,
                "predator_proximity": 0.0,
                "predator_motion": 0.0,
                "predator_heading_alignment": 0.0,
                "predator_recent_contact": 0.0,
                "recent_pain": 0.0,
            },
        )
        monolithic_observation = brain._build_monolithic_observation(observation)
        policy_logits, *_ = recurrent.forward(monolithic_observation, store_cache=False)
        self.assertEqual(LOCOMOTION_ACTIONS[int(np.argmax(policy_logits))], "MOVE_LEFT")

    def test_executive_post_food_path_return_replays_inverse_outbound_action(self) -> None:
        brain = SpiderBrain(
            seed=132,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="true_monolithic_executive_option_post_food_path_return_policy",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=False,
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
                direct_policy_phase_option_feedback=True,
                direct_policy_option_transition_feedback=True,
                direct_policy_option_termination_cooldown=True,
                direct_policy_option_decoder_state=True,
                direct_policy_option_recurrent_dynamics=True,
                direct_policy_option_decoder_recurrent_state=True,
                direct_policy_option_action_recurrent_core=True,
                direct_policy_option_action_separate_recurrent_head=True,
                direct_policy_option_action_separate_policy_path=True,
                direct_policy_option_action_separate_backbone=True,
                direct_policy_executive_physiology_option_gating=True,
                direct_policy_executive_affordance_action_gating=True,
                direct_policy_executive_option_action_masking=True,
                direct_policy_executive_event_release_latching=True,
                direct_policy_executive_event_release_action_commitment=True,
                direct_policy_executive_release_phase_state=True,
                direct_policy_executive_release_exit_contract=True,
                direct_policy_executive_release_substate_progression=True,
                direct_policy_executive_post_exit_continuation=True,
                direct_policy_executive_post_exit_corridor_progression=True,
                direct_policy_executive_post_exit_corridor_affordance_progression=True,
                direct_policy_executive_post_food_return=True,
                direct_policy_executive_post_food_path_return=True,
            ),
        )
        recurrent = brain.true_monolithic_policy
        self.assertIsInstance(
            recurrent,
            RecurrentOptionAffordancePositionFeedbackTrueMonolithicNetwork,
        )
        recurrent.reset_hidden_state()
        recurrent.reset_event_memory()
        recurrent.current_option_idx = OPTION_NAMES.index("RETURN_TO_SHELTER")
        recurrent.current_option_age = 1
        recurrent.current_option_steps_remaining = 3
        recurrent.executive_post_food_return_steps_remaining = 8
        recurrent.executive_post_food_return_queue = [
            ACTION_TO_INDEX["MOVE_LEFT"]
        ]
        recurrent.W_xh.fill(0.0)
        recurrent.W_hh.fill(0.0)
        recurrent.b_h.fill(0.0)
        recurrent.W2_option.fill(0.0)
        recurrent.b2_option.fill(0.0)
        recurrent.W2_option_feedback.fill(0.0)
        recurrent.b2_option_feedback.fill(0.0)
        recurrent.W_affordance_feedback.fill(0.0)
        recurrent.b_affordance_feedback.fill(0.0)
        recurrent.W2_policy.fill(0.0)
        recurrent.b2_policy.fill(0.0)
        recurrent.b2_policy[ACTION_TO_INDEX["MOVE_DOWN"]] = 10.0

        observation = _build_observation(
            sleep={
                "fatigue": 0.05,
                "hunger": 0.12,
                "on_shelter": 0.0,
                "night": 0.0,
                "sleep_phase_level": 0.0,
                "rest_streak_norm": 0.2,
                "sleep_debt": 0.05,
                "shelter_role_level": 0.0,
                "shelter_memory_age": 0.1,
            },
            predator={
                "predator_visible": 0.0,
                "predator_proximity": 0.0,
                "predator_motion": 0.0,
                "predator_heading_alignment": 0.0,
                "predator_recent_contact": 0.0,
                "recent_pain": 0.0,
            },
        )
        monolithic_observation = brain._build_monolithic_observation(observation)
        policy_logits, *_ = recurrent.forward(monolithic_observation, store_cache=False)
        self.assertEqual(LOCOMOTION_ACTIONS[int(np.argmax(policy_logits))], "MOVE_LEFT")

    def test_true_monolithic_option_affordance_position_teacher_option_replay_policy_uses_position_network(self) -> None:
        brain = SpiderBrain(
            seed=37,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="true_monolithic_option_affordance_position_teacher_option_replay_policy",
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
                direct_policy_handoff_teacher=True,
                direct_policy_handoff_option_teacher=True,
                direct_policy_continuation_replay_passes=2,
                direct_policy_continuation_replay_lr_scale=0.5,
            ),
        )
        self.assertIsInstance(
            brain.true_monolithic_policy,
            RecurrentOptionAffordancePositionFeedbackTrueMonolithicNetwork,
        )

    def test_true_monolithic_option_affordance_position_phase_teacher_option_replay_policy_exposes_phase_and_option_logits(self) -> None:
        brain = SpiderBrain(
            seed=38,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="true_monolithic_option_affordance_position_phase_teacher_option_replay_policy",
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
                direct_policy_continuation_replay_passes=2,
                direct_policy_continuation_replay_lr_scale=0.5,
            ),
        )
        self.assertIsInstance(
            brain.true_monolithic_policy,
            RecurrentOptionAffordancePositionFeedbackTrueMonolithicNetwork,
        )
        decision = brain.act_inference(_build_observation(), sample=False)
        self.assertEqual(decision.option_logits.shape, (len(OPTION_NAMES),))
        self.assertEqual(decision.phase_logits.shape, (8,))
        self.assertIsNotNone(decision.phase_prediction)

    def test_true_monolithic_option_affordance_position_phase_teacher_option_margin_replay_policy_exposes_phase_and_option_logits(self) -> None:
        brain = SpiderBrain(
            seed=39,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="true_monolithic_option_affordance_position_phase_teacher_option_margin_replay_policy",
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
                direct_policy_continuation_replay_passes=2,
                direct_policy_continuation_replay_lr_scale=0.5,
                direct_policy_continuation_margin_weight=1.0,
            ),
        )
        self.assertIsInstance(
            brain.true_monolithic_policy,
            RecurrentOptionAffordancePositionFeedbackTrueMonolithicNetwork,
        )
        decision = brain.act_inference(_build_observation(), sample=False)
        self.assertEqual(decision.option_logits.shape, (len(OPTION_NAMES),))
        self.assertEqual(decision.phase_logits.shape, (8,))
        self.assertIsNotNone(decision.phase_prediction)

    def test_true_monolithic_option_affordance_position_phase_option_feedback_teacher_option_replay_policy_exposes_phase_and_option_logits(self) -> None:
        brain = SpiderBrain(
            seed=40,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="true_monolithic_option_affordance_position_phase_option_feedback_teacher_option_replay_policy",
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
                direct_policy_phase_option_feedback=True,
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
                direct_policy_continuation_replay_passes=2,
                direct_policy_continuation_replay_lr_scale=0.5,
            ),
        )
        self.assertIsInstance(
            brain.true_monolithic_policy,
            RecurrentOptionAffordancePositionFeedbackTrueMonolithicNetwork,
        )
        decision = brain.act_inference(_build_observation(), sample=False)
        self.assertEqual(decision.option_logits.shape, (len(OPTION_NAMES),))
        self.assertEqual(decision.phase_logits.shape, (8,))
        self.assertIsNotNone(decision.phase_prediction)

    def test_true_monolithic_option_affordance_position_phase_option_transition_teacher_option_replay_policy_exposes_phase_and_option_logits(self) -> None:
        brain = SpiderBrain(
            seed=42,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="true_monolithic_option_affordance_position_phase_option_transition_teacher_option_replay_policy",
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
                direct_policy_option_transition_feedback=True,
                direct_policy_continuation_replay_passes=2,
                direct_policy_continuation_replay_lr_scale=0.5,
            ),
        )
        self.assertIsInstance(
            brain.true_monolithic_policy,
            RecurrentOptionAffordancePositionFeedbackTrueMonolithicNetwork,
        )
        decision = brain.act_inference(_build_observation(), sample=False)
        self.assertEqual(decision.option_logits.shape, (len(OPTION_NAMES),))
        self.assertEqual(decision.phase_logits.shape, (8,))
        self.assertIsNotNone(decision.phase_prediction)

    def test_true_monolithic_option_affordance_position_phase_option_cooldown_teacher_option_replay_policy_exposes_phase_and_option_logits(self) -> None:
        brain = SpiderBrain(
            seed=45,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="true_monolithic_option_affordance_position_phase_option_cooldown_teacher_option_replay_policy",
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
                direct_policy_option_termination_cooldown=True,
                direct_policy_continuation_replay_passes=2,
                direct_policy_continuation_replay_lr_scale=0.5,
            ),
        )
        self.assertIsInstance(
            brain.true_monolithic_policy,
            RecurrentOptionAffordancePositionFeedbackTrueMonolithicNetwork,
        )
        decision = brain.act_inference(_build_observation(), sample=False)
        self.assertEqual(decision.option_logits.shape, (len(OPTION_NAMES),))
        self.assertEqual(decision.phase_logits.shape, (8,))
        self.assertIsNotNone(decision.phase_prediction)

    def test_true_monolithic_option_affordance_position_phase_option_action_teacher_option_replay_policy_exposes_phase_and_option_logits(self) -> None:
        brain = SpiderBrain(
            seed=47,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="true_monolithic_option_affordance_position_phase_option_action_teacher_option_replay_policy",
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
                direct_policy_option_action_head=True,
                direct_policy_continuation_replay_passes=2,
                direct_policy_continuation_replay_lr_scale=0.5,
            ),
        )
        self.assertIsInstance(
            brain.true_monolithic_policy,
            RecurrentOptionAffordancePositionFeedbackTrueMonolithicNetwork,
        )
        decision = brain.act_inference(_build_observation(), sample=False)
        self.assertEqual(decision.option_logits.shape, (len(OPTION_NAMES),))
        self.assertEqual(decision.phase_logits.shape, (8,))
        self.assertIsNotNone(decision.phase_prediction)

    def test_true_monolithic_option_affordance_position_phase_option_decoder_teacher_option_replay_policy_exposes_phase_and_option_logits(self) -> None:
        brain = SpiderBrain(
            seed=49,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="true_monolithic_option_affordance_position_phase_option_decoder_teacher_option_replay_policy",
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
                direct_policy_option_decoder_state=True,
                direct_policy_continuation_replay_passes=2,
                direct_policy_continuation_replay_lr_scale=0.5,
            ),
        )
        self.assertIsInstance(
            brain.true_monolithic_policy,
            RecurrentOptionAffordancePositionFeedbackTrueMonolithicNetwork,
        )
        decision = brain.act_inference(_build_observation(), sample=False)
        self.assertEqual(decision.option_logits.shape, (len(OPTION_NAMES),))
        self.assertEqual(decision.phase_logits.shape, (8,))
        self.assertIsNotNone(decision.phase_prediction)

    def test_true_monolithic_option_affordance_position_phase_option_decoder_feedback_teacher_option_replay_policy_exposes_phase_and_option_logits(self) -> None:
        brain = SpiderBrain(
            seed=51,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="true_monolithic_option_affordance_position_phase_option_decoder_feedback_teacher_option_replay_policy",
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
                direct_policy_phase_option_feedback=True,
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
                direct_policy_option_decoder_state=True,
                direct_policy_continuation_replay_passes=2,
                direct_policy_continuation_replay_lr_scale=0.5,
            ),
        )
        self.assertIsInstance(
            brain.true_monolithic_policy,
            RecurrentOptionAffordancePositionFeedbackTrueMonolithicNetwork,
        )
        decision = brain.act_inference(_build_observation(), sample=False)
        self.assertEqual(decision.option_logits.shape, (len(OPTION_NAMES),))
        self.assertEqual(decision.phase_logits.shape, (8,))
        self.assertIsNotNone(decision.phase_prediction)

    def test_true_monolithic_option_affordance_position_phase_option_dynamics_teacher_option_replay_policy_exposes_phase_and_option_logits(self) -> None:
        brain = SpiderBrain(
            seed=52,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="true_monolithic_option_affordance_position_phase_option_dynamics_teacher_option_replay_policy",
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
                direct_policy_option_decoder_state=True,
                direct_policy_option_recurrent_dynamics=True,
                direct_policy_continuation_replay_passes=2,
                direct_policy_continuation_replay_lr_scale=0.5,
            ),
        )
        self.assertIsInstance(
            brain.true_monolithic_policy,
            RecurrentOptionAffordancePositionFeedbackTrueMonolithicNetwork,
        )
        decision = brain.act_inference(_build_observation(), sample=False)
        self.assertEqual(decision.option_logits.shape, (len(OPTION_NAMES),))
        self.assertEqual(decision.phase_logits.shape, (8,))
        self.assertIsNotNone(decision.phase_prediction)
