from __future__ import annotations

from .shared import *

from spider_cortex_sim.direct_policy_capabilities import (
    DirectPolicyCapabilities,
    derive_direct_policy_capabilities,
)
from spider_cortex_sim.bus import MessageBus


class DirectPolicyCapabilitiesTest(unittest.TestCase):
    def _full_local_config(self) -> BrainAblationConfig:
        return BrainAblationConfig(
            name="direct_policy_capability_test",
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
            direct_policy_local_affordance_inputs=True,
            direct_policy_local_spatial_inputs=True,
            direct_policy_local_transition_inputs=True,
            direct_policy_local_transition_rollout_inputs=True,
            direct_policy_local_geodesic_inputs=True,
            direct_policy_transition_prediction_head=True,
            direct_policy_transition_prediction_feedback=True,
            direct_policy_transition_rollout_prediction_head=True,
            direct_policy_transition_rollout_prediction_feedback=True,
            direct_policy_handoff_teacher=True,
            direct_policy_handoff_option_teacher=True,
        )

    def test_config_derives_direct_policy_capability_groups(self) -> None:
        config = self._full_local_config()
        capabilities = derive_direct_policy_capabilities(config)

        self.assertIsInstance(capabilities, DirectPolicyCapabilities)
        self.assertEqual(capabilities.network.hidden_dims, (32,))
        self.assertTrue(capabilities.network.recurrent)
        self.assertTrue(capabilities.network.event_attention)
        self.assertTrue(capabilities.heads.phase)
        self.assertTrue(capabilities.heads.affordance)
        self.assertTrue(capabilities.heads.transition_prediction)
        self.assertTrue(capabilities.heads.transition_rollout_prediction)
        self.assertTrue(capabilities.local_inputs.affordance)
        self.assertTrue(capabilities.local_inputs.spatial)
        self.assertTrue(capabilities.local_inputs.transition)
        self.assertTrue(capabilities.local_inputs.transition_rollout)
        self.assertTrue(capabilities.local_inputs.geodesic)
        self.assertTrue(capabilities.teachers.handoff)
        self.assertTrue(capabilities.teachers.handoff_option)
        self.assertEqual(
            capabilities.local_inputs.input_dim(),
            DIRECT_POLICY_LOCAL_AFFORDANCE_INPUT_DIM
            + DIRECT_POLICY_LOCAL_SPATIAL_INPUT_DIM
            + DIRECT_POLICY_LOCAL_TRANSITION_INPUT_DIM
            + DIRECT_POLICY_LOCAL_TRANSITION_ROLLOUT_INPUT_DIM
            + DIRECT_POLICY_LOCAL_GEODESIC_INPUT_DIM,
        )

        metadata = capabilities.architecture_metadata()
        self.assertEqual(metadata["direct_policy_hidden_dims"], [32])
        self.assertTrue(metadata["direct_policy_local_affordance_inputs"])
        self.assertTrue(metadata["direct_policy_local_geodesic_inputs"])
        self.assertTrue(metadata["direct_policy_transition_prediction_head"])
        self.assertTrue(metadata["direct_policy_handoff_teacher"])
        self.assertTrue(hasattr(config, "direct_policy_local_affordance_inputs"))

    def test_spiderbrain_exposes_capabilities_for_network_and_input_wiring(self) -> None:
        config = self._full_local_config()
        brain = SpiderBrain(seed=131, module_dropout=0.0, config=config)
        capabilities = brain.direct_policy_capabilities
        observation = _build_observation()

        self.assertEqual(capabilities, derive_direct_policy_capabilities(brain.config))
        assert brain.true_monolithic_policy is not None
        expected_input_dim = (
            sum(spec.input_dim for spec in MODULE_INTERFACES)
            + capabilities.local_inputs.input_dim()
        )
        self.assertEqual(brain.true_monolithic_policy.input_dim, expected_input_dim)
        self.assertEqual(
            brain._build_monolithic_observation(observation).shape[0],
            expected_input_dim,
        )

    def test_action_trace_metadata_includes_local_geodesic_inputs(self) -> None:
        brain = SpiderBrain(
            seed=132,
            module_dropout=0.0,
            config=self._full_local_config(),
        )
        bus = MessageBus()

        brain.act_inference(_build_observation(), bus=bus, sample=False)

        payload = bus.topic_messages("action.selection")[0].payload
        self.assertTrue(
            payload["architecture_metadata"]["direct_policy_local_geodesic_inputs"]
        )
