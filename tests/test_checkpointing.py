import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from spider_cortex_sim.ablations import BrainAblationConfig
from spider_cortex_sim.agent import SpiderBrain
from spider_cortex_sim.checkpointing import (
    CheckpointPenaltyMode,
    CheckpointSelectionConfig,
    build_checkpointing_summary,
    checkpoint_candidate_composite_score,
    checkpoint_candidate_sort_key,
    checkpoint_preload_fingerprint,
    checkpoint_run_fingerprint,
    file_sha256,
    jsonify_observation,
    mean_reward_from_behavior_payload,
    persist_checkpoint_pair,
    resolve_checkpoint_load_dir,
)

from tests.fixtures.checkpointing import _write_checkpoint_artifact

class ResolveCheckpointLoadDirValidationTest(unittest.TestCase):
    def test_invalid_checkpoint_selection_raises_value_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(
                ValueError,
                "checkpoint_selection.*typo",
            ):
                resolve_checkpoint_load_dir(
                    Path(tmpdir),
                    checkpoint_selection="typo",
                )

class CheckpointSelectionPenaltyConfigTest(unittest.TestCase):
    def test_direct_penalty_config_normalizes_mode_and_weights(self) -> None:
        """
        Verify that CheckpointSelectionConfig normalizes the penalty mode and preserves penalty weights.

        Asserts that providing penalty_mode="direct" results in the enum value CheckpointPenaltyMode.DIRECT, that to_summary() reports "direct" for penalty_mode, and that override and dominance penalty weights are retained.
        """
        config = CheckpointSelectionConfig(
            metric="scenario_success_rate",
            override_penalty_weight=0.4,
            dominance_penalty_weight=0.2,
            penalty_mode="direct",
        )

        self.assertEqual(config.penalty_mode, CheckpointPenaltyMode.DIRECT)
        self.assertEqual(config.to_summary()["penalty_mode"], "direct")
        self.assertAlmostEqual(config.override_penalty_weight, 0.4)
        self.assertAlmostEqual(config.dominance_penalty_weight, 0.2)

    def test_direct_mode_penalizes_high_override_with_same_success_rate(self) -> None:
        low_override = {
            "scenario_success_rate": 0.8,
            "episode_success_rate": 0.6,
            "mean_reward": 1.0,
            "evaluation_summary": {
                "mean_final_reflex_override_rate": 0.1,
                "mean_reflex_dominance": 0.0,
            },
            "episode": 1,
        }
        high_override = {
            "scenario_success_rate": 0.8,
            "episode_success_rate": 0.6,
            "mean_reward": 1.0,
            "evaluation_summary": {
                "mean_final_reflex_override_rate": 0.7,
                "mean_reflex_dominance": 0.0,
            },
            "episode": 2,
        }
        config = CheckpointSelectionConfig(
            metric="scenario_success_rate",
            override_penalty_weight=1.0,
            penalty_mode=CheckpointPenaltyMode.DIRECT,
        )

        self.assertGreater(
            checkpoint_candidate_sort_key(
                low_override,
                selection_config=config,
            ),
            checkpoint_candidate_sort_key(
                high_override,
                selection_config=config,
            ),
        )

    def test_tiebreaker_mode_preserves_legacy_six_tuple(self) -> None:
        candidate = {
            "scenario_success_rate": 0.5,
            "episode_success_rate": 0.4,
            "mean_reward": 1.0,
            "evaluation_summary": {
                "mean_final_reflex_override_rate": 0.2,
                "mean_reflex_dominance": 0.3,
            },
            "episode": 7,
        }

        key = checkpoint_candidate_sort_key(
            candidate,
            selection_config=CheckpointSelectionConfig(
                metric="scenario_success_rate",
                penalty_mode=CheckpointPenaltyMode.TIEBREAKER,
            ),
        )

        self.assertEqual(key, (0.5, 0.4, 1.0, -0.2, -0.3, 7))

    def test_direct_mode_composite_score_uses_penalty_formula(self) -> None:
        candidate = {
            "scenario_success_rate": 0.8,
            "episode_success_rate": 0.4,
            "mean_reward": 1.0,
            "evaluation_summary": {
                "mean_final_reflex_override_rate": 0.25,
                "mean_reflex_dominance": 0.5,
            },
            "episode": 7,
        }

        key = checkpoint_candidate_sort_key(
            candidate,
            selection_config=CheckpointSelectionConfig(
                metric="scenario_success_rate",
                override_penalty_weight=0.4,
                dominance_penalty_weight=0.2,
                penalty_mode=CheckpointPenaltyMode.DIRECT,
            ),
        )

        self.assertAlmostEqual(key[0], 0.8 - (0.4 * 0.25) - (0.2 * 0.5))
        self.assertEqual(key[1:4], (0.8, 0.4, 1.0))

class SimulationMeanRewardFromPayloadTest(unittest.TestCase):
    """Tests for mean_reward_from_behavior_payload()."""

    def test_empty_payload_returns_zero(self) -> None:
        result = mean_reward_from_behavior_payload({})
        self.assertEqual(result, 0.0)

    def test_empty_legacy_scenarios_returns_zero(self) -> None:
        result = mean_reward_from_behavior_payload(
            {"legacy_scenarios": {}}
        )
        self.assertEqual(result, 0.0)

    def test_non_dict_legacy_scenarios_returns_zero(self) -> None:
        result = mean_reward_from_behavior_payload(
            {"legacy_scenarios": None}
        )
        self.assertEqual(result, 0.0)

    def test_single_scenario_mean_reward(self) -> None:
        payload = {
            "legacy_scenarios": {
                "night_rest": {"mean_reward": 2.5},
            }
        }
        result = mean_reward_from_behavior_payload(payload)
        self.assertAlmostEqual(result, 2.5)

    def test_multiple_scenarios_averaged(self) -> None:
        payload = {
            "legacy_scenarios": {
                "night_rest": {"mean_reward": 1.0},
                "food_deprivation": {"mean_reward": 3.0},
            }
        }
        result = mean_reward_from_behavior_payload(payload)
        self.assertAlmostEqual(result, 2.0)

    def test_non_dict_scenario_data_skipped(self) -> None:
        payload = {
            "legacy_scenarios": {
                "night_rest": {"mean_reward": 4.0},
                "bad_entry": "not_a_dict",
            }
        }
        result = mean_reward_from_behavior_payload(payload)
        self.assertAlmostEqual(result, 4.0)

    def test_missing_mean_reward_key_defaults_to_zero(self) -> None:
        payload = {
            "legacy_scenarios": {
                "night_rest": {"other_key": 99},
            }
        }
        result = mean_reward_from_behavior_payload(payload)
        self.assertAlmostEqual(result, 0.0)


class TrueMonolithicRecurrentCheckpointingTest(unittest.TestCase):
    def test_save_load_round_trip_resets_live_hidden_state_and_restores_weights(self) -> None:
        config = BrainAblationConfig(
            name="true_monolithic_recurrent_policy",
            architecture="true_monolithic",
            module_dropout=0.0,
            enable_reflexes=False,
            enable_auxiliary_targets=False,
            enable_food_direction_bias=True,
            use_learned_arbitration=False,
            warm_start_scale=0.0,
            direct_policy_hidden_dims=(32,),
            direct_policy_recurrent=True,
        )
        source = SpiderBrain(seed=31, module_dropout=0.0, config=config)
        target = SpiderBrain(seed=32, module_dropout=0.0, config=config)
        recurrent = source.true_monolithic_policy
        loaded = target.true_monolithic_policy
        self.assertIsNotNone(recurrent)
        self.assertIsNotNone(loaded)
        recurrent.hidden_state[:] = 0.75
        with tempfile.TemporaryDirectory() as tmpdir:
            source.save(tmpdir)
            target.load(tmpdir)
        np.testing.assert_allclose(loaded.hidden_state, np.zeros(loaded.hidden_dim, dtype=float))
        np.testing.assert_allclose(loaded.W_xh, recurrent.W_xh)
        np.testing.assert_allclose(loaded.W_hh, recurrent.W_hh)
        np.testing.assert_allclose(loaded.W2_policy, recurrent.W2_policy)

    def test_phase_head_round_trip_restores_phase_weights(self) -> None:
        config = BrainAblationConfig(
            name="true_monolithic_recurrent_phase_policy",
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
        )
        source = SpiderBrain(seed=33, module_dropout=0.0, config=config)
        target = SpiderBrain(seed=34, module_dropout=0.0, config=config)
        recurrent = source.true_monolithic_policy
        loaded = target.true_monolithic_policy
        self.assertIsNotNone(recurrent)
        self.assertIsNotNone(loaded)
        with tempfile.TemporaryDirectory() as tmpdir:
            source.save(tmpdir)
            target.load(tmpdir)
        np.testing.assert_allclose(loaded.W2_phase, recurrent.W2_phase)
        np.testing.assert_allclose(loaded.b2_phase, recurrent.b2_phase)

    def test_position_phase_replay_round_trip_restores_phase_and_option_weights(self) -> None:
        config = BrainAblationConfig(
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
        )
        source = SpiderBrain(seed=39, module_dropout=0.0, config=config)
        target = SpiderBrain(seed=40, module_dropout=0.0, config=config)
        recurrent = source.true_monolithic_policy
        loaded = target.true_monolithic_policy
        self.assertIsNotNone(recurrent)
        self.assertIsNotNone(loaded)
        with tempfile.TemporaryDirectory() as tmpdir:
            source.save(tmpdir)
            target.load(tmpdir)
        np.testing.assert_allclose(loaded.W2_option, recurrent.W2_option)
        np.testing.assert_allclose(loaded.b2_option, recurrent.b2_option)
        np.testing.assert_allclose(loaded.W2_phase, recurrent.W2_phase)
        np.testing.assert_allclose(loaded.b2_phase, recurrent.b2_phase)

    def test_position_phase_option_feedback_replay_round_trip_restores_phase_feedback_weights(self) -> None:
        config = BrainAblationConfig(
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
        )
        source = SpiderBrain(seed=41, module_dropout=0.0, config=config)
        target = SpiderBrain(seed=42, module_dropout=0.0, config=config)
        recurrent = source.true_monolithic_policy
        loaded = target.true_monolithic_policy
        self.assertIsNotNone(recurrent)
        self.assertIsNotNone(loaded)
        with tempfile.TemporaryDirectory() as tmpdir:
            source.save(tmpdir)
            target.load(tmpdir)
        np.testing.assert_allclose(
            loaded.W2_phase_option_feedback,
            recurrent.W2_phase_option_feedback,
        )
        np.testing.assert_allclose(
            loaded.b2_phase_option_feedback,
            recurrent.b2_phase_option_feedback,
        )

    def test_position_phase_option_transition_replay_round_trip_restores_transition_weights(self) -> None:
        config = BrainAblationConfig(
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
        )
        source = SpiderBrain(seed=43, module_dropout=0.0, config=config)
        target = SpiderBrain(seed=44, module_dropout=0.0, config=config)
        recurrent = source.true_monolithic_policy
        loaded = target.true_monolithic_policy
        self.assertIsNotNone(recurrent)
        self.assertIsNotNone(loaded)
        with tempfile.TemporaryDirectory() as tmpdir:
            source.save(tmpdir)
            target.load(tmpdir)
        np.testing.assert_allclose(
            loaded.W2_option_transition_feedback,
            recurrent.W2_option_transition_feedback,
        )
        np.testing.assert_allclose(
            loaded.b2_option_transition_feedback,
            recurrent.b2_option_transition_feedback,
        )

    def test_position_phase_option_action_replay_round_trip_restores_action_head_weights(self) -> None:
        config = BrainAblationConfig(
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
        )
        source = SpiderBrain(seed=45, module_dropout=0.0, config=config)
        target = SpiderBrain(seed=46, module_dropout=0.0, config=config)
        recurrent = source.true_monolithic_policy
        loaded = target.true_monolithic_policy
        self.assertIsNotNone(recurrent)
        self.assertIsNotNone(loaded)
        with tempfile.TemporaryDirectory() as tmpdir:
            source.save(tmpdir)
            target.load(tmpdir)
        np.testing.assert_allclose(
            loaded.W2_option_action_head,
            recurrent.W2_option_action_head,
        )
        np.testing.assert_allclose(
            loaded.b2_option_action_head,
            recurrent.b2_option_action_head,
        )

    def test_position_phase_option_decoder_replay_round_trip_restores_decoder_weights(self) -> None:
        config = BrainAblationConfig(
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
        )
        source = SpiderBrain(seed=47, module_dropout=0.0, config=config)
        target = SpiderBrain(seed=48, module_dropout=0.0, config=config)
        recurrent = source.true_monolithic_policy
        loaded = target.true_monolithic_policy
        self.assertIsNotNone(recurrent)
        self.assertIsNotNone(loaded)
        with tempfile.TemporaryDirectory() as tmpdir:
            source.save(tmpdir)
            target.load(tmpdir)
        np.testing.assert_allclose(
            loaded.W_option_decoder_state,
            recurrent.W_option_decoder_state,
        )
        np.testing.assert_allclose(
            loaded.b_option_decoder_state,
            recurrent.b_option_decoder_state,
        )

    def test_position_phase_option_dynamics_replay_round_trip_restores_recurrent_dynamics_weights(self) -> None:
        config = BrainAblationConfig(
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
        )
        source = SpiderBrain(seed=49, module_dropout=0.0, config=config)
        target = SpiderBrain(seed=50, module_dropout=0.0, config=config)
        recurrent = source.true_monolithic_policy
        loaded = target.true_monolithic_policy
        self.assertIsNotNone(recurrent)
        self.assertIsNotNone(loaded)
        with tempfile.TemporaryDirectory() as tmpdir:
            source.save(tmpdir)
            target.load(tmpdir)
        np.testing.assert_allclose(
            loaded.W_option_recurrent_dynamics,
            recurrent.W_option_recurrent_dynamics,
        )
        np.testing.assert_allclose(
            loaded.b_option_recurrent_dynamics,
            recurrent.b_option_recurrent_dynamics,
        )

    def test_position_phase_option_dynamics_sequence_replay_round_trip_restores_sequence_head_weights(self) -> None:
        config = BrainAblationConfig(
            name="true_monolithic_option_affordance_position_phase_option_dynamics_sequence_teacher_option_replay_policy",
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
            direct_policy_option_sequence_head=True,
            direct_policy_continuation_replay_passes=2,
            direct_policy_continuation_replay_lr_scale=0.5,
        )
        source = SpiderBrain(seed=58, module_dropout=0.0, config=config)
        target = SpiderBrain(seed=59, module_dropout=0.0, config=config)
        recurrent = source.true_monolithic_policy
        loaded = target.true_monolithic_policy
        self.assertIsNotNone(recurrent)
        self.assertIsNotNone(loaded)
        with tempfile.TemporaryDirectory() as tmpdir:
            source.save(tmpdir)
            target.load(tmpdir)
        np.testing.assert_allclose(
            loaded.W2_option_sequence_head,
            recurrent.W2_option_sequence_head,
        )
        np.testing.assert_allclose(
            loaded.b2_option_sequence_head,
            recurrent.b2_option_sequence_head,
        )

    def test_position_phase_option_dynamics_decoder_recurrent_replay_round_trip_restores_decoder_recurrent_weights(self) -> None:
        config = BrainAblationConfig(
            name="true_monolithic_option_affordance_position_phase_option_dynamics_decoder_recurrent_teacher_option_replay_policy",
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
            direct_policy_option_decoder_recurrent_state=True,
            direct_policy_continuation_replay_passes=2,
            direct_policy_continuation_replay_lr_scale=0.5,
        )
        source = SpiderBrain(seed=62, module_dropout=0.0, config=config)
        target = SpiderBrain(seed=63, module_dropout=0.0, config=config)
        recurrent = source.true_monolithic_policy
        loaded = target.true_monolithic_policy
        self.assertIsNotNone(recurrent)
        self.assertIsNotNone(loaded)
        with tempfile.TemporaryDirectory() as tmpdir:
            source.save(tmpdir)
            target.load(tmpdir)
        np.testing.assert_allclose(
            loaded.W_option_decoder_recurrent_state,
            recurrent.W_option_decoder_recurrent_state,
        )
        np.testing.assert_allclose(
            loaded.b_option_decoder_recurrent_state,
            recurrent.b_option_decoder_recurrent_state,
        )

    def test_position_phase_option_dynamics_decoder_recurrent_action_transition_replay_round_trip_restores_action_transition_weights(self) -> None:
        config = BrainAblationConfig(
            name="true_monolithic_option_affordance_position_phase_option_dynamics_decoder_recurrent_action_transition_teacher_option_replay_policy",
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
            direct_policy_option_decoder_recurrent_state=True,
            direct_policy_option_action_transition_state=True,
            direct_policy_continuation_replay_passes=2,
            direct_policy_continuation_replay_lr_scale=0.5,
        )
        source = SpiderBrain(seed=64, module_dropout=0.0, config=config)
        target = SpiderBrain(seed=65, module_dropout=0.0, config=config)
        recurrent = source.true_monolithic_policy
        loaded = target.true_monolithic_policy
        self.assertIsNotNone(recurrent)
        self.assertIsNotNone(loaded)
        with tempfile.TemporaryDirectory() as tmpdir:
            source.save(tmpdir)
            target.load(tmpdir)
        np.testing.assert_allclose(
            loaded.W_option_action_transition_state,
            recurrent.W_option_action_transition_state,
        )
        np.testing.assert_allclose(
            loaded.b_option_action_transition_state,
            recurrent.b_option_action_transition_state,
        )

    def test_position_phase_option_dynamics_decoder_recurrent_action_controller_replay_round_trip_restores_action_controller_weights(self) -> None:
        config = BrainAblationConfig(
            name="true_monolithic_option_affordance_position_phase_option_dynamics_decoder_recurrent_action_controller_teacher_option_replay_policy",
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
            direct_policy_option_decoder_recurrent_state=True,
            direct_policy_option_action_controller_state=True,
            direct_policy_continuation_replay_passes=2,
            direct_policy_continuation_replay_lr_scale=0.5,
        )
        source = SpiderBrain(seed=68, module_dropout=0.0, config=config)
        target = SpiderBrain(seed=69, module_dropout=0.0, config=config)
        recurrent = source.true_monolithic_policy
        loaded = target.true_monolithic_policy
        self.assertIsNotNone(recurrent)
        self.assertIsNotNone(loaded)
        with tempfile.TemporaryDirectory() as tmpdir:
            source.save(tmpdir)
            target.load(tmpdir)
        np.testing.assert_allclose(
            loaded.W_option_action_controller_decoder,
            recurrent.W_option_action_controller_decoder,
        )
        np.testing.assert_allclose(
            loaded.W_option_action_controller_prev,
            recurrent.W_option_action_controller_prev,
        )
        np.testing.assert_allclose(
            loaded.W_option_action_controller_action,
            recurrent.W_option_action_controller_action,
        )
        np.testing.assert_allclose(
            loaded.b_option_action_controller,
            recurrent.b_option_action_controller,
        )
        np.testing.assert_allclose(
            loaded.W2_option_action_controller_head,
            recurrent.W2_option_action_controller_head,
        )
        np.testing.assert_allclose(
            loaded.b2_option_action_controller_head,
            recurrent.b2_option_action_controller_head,
        )

    def test_position_phase_option_dynamics_decoder_recurrent_action_token_replay_round_trip_restores_action_token_weights(self) -> None:
        config = BrainAblationConfig(
            name="true_monolithic_option_affordance_position_phase_option_dynamics_decoder_recurrent_action_token_teacher_option_replay_policy",
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
            direct_policy_option_decoder_recurrent_state=True,
            direct_policy_option_action_token_decoder=True,
            direct_policy_continuation_replay_passes=2,
            direct_policy_continuation_replay_lr_scale=0.5,
        )
        source = SpiderBrain(seed=72, module_dropout=0.0, config=config)
        target = SpiderBrain(seed=73, module_dropout=0.0, config=config)
        recurrent = source.true_monolithic_policy
        loaded = target.true_monolithic_policy
        self.assertIsNotNone(recurrent)
        self.assertIsNotNone(loaded)
        with tempfile.TemporaryDirectory() as tmpdir:
            source.save(tmpdir)
            target.load(tmpdir)
        np.testing.assert_allclose(
            loaded.W_option_action_token_decoder,
            recurrent.W_option_action_token_decoder,
        )
        np.testing.assert_allclose(
            loaded.W_option_action_token_prev,
            recurrent.W_option_action_token_prev,
        )
        np.testing.assert_allclose(
            loaded.W_option_action_token_action,
            recurrent.W_option_action_token_action,
        )
        np.testing.assert_allclose(
            loaded.b_option_action_token,
            recurrent.b_option_action_token,
        )

    def test_position_phase_option_dynamics_decoder_recurrent_action_core_replay_round_trip_restores_action_core_weights(self) -> None:
        config = BrainAblationConfig(
            name="true_monolithic_option_affordance_position_phase_option_dynamics_decoder_recurrent_action_core_teacher_option_replay_policy",
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
            direct_policy_option_decoder_recurrent_state=True,
            direct_policy_option_action_recurrent_core=True,
            direct_policy_continuation_replay_passes=2,
            direct_policy_continuation_replay_lr_scale=0.5,
        )
        source = SpiderBrain(seed=76, module_dropout=0.0, config=config)
        target = SpiderBrain(seed=77, module_dropout=0.0, config=config)
        recurrent = source.true_monolithic_policy
        loaded = target.true_monolithic_policy
        self.assertIsNotNone(recurrent)
        self.assertIsNotNone(loaded)
        with tempfile.TemporaryDirectory() as tmpdir:
            source.save(tmpdir)
            target.load(tmpdir)
        np.testing.assert_allclose(
            loaded.W_option_action_policy_decoder,
            recurrent.W_option_action_policy_decoder,
        )
        np.testing.assert_allclose(
            loaded.W_option_action_policy_prev,
            recurrent.W_option_action_policy_prev,
        )
        np.testing.assert_allclose(
            loaded.W_option_action_policy_action,
            recurrent.W_option_action_policy_action,
        )
        np.testing.assert_allclose(
            loaded.b_option_action_policy,
            recurrent.b_option_action_policy,
        )
        np.testing.assert_allclose(
            loaded.W2_action_policy_core,
            recurrent.W2_action_policy_core,
        )
        np.testing.assert_allclose(
            loaded.b2_action_policy_core,
            recurrent.b2_action_policy_core,
        )

    def test_position_phase_option_dynamics_separate_action_recurrent_head_replay_round_trip_restores_action_core_weights(self) -> None:
        config = BrainAblationConfig(
            name="true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_recurrent_head_teacher_option_replay_policy",
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
            direct_policy_option_decoder_recurrent_state=True,
            direct_policy_option_action_recurrent_core=True,
            direct_policy_option_action_separate_recurrent_head=True,
            direct_policy_continuation_replay_passes=2,
            direct_policy_continuation_replay_lr_scale=0.5,
        )
        source = SpiderBrain(seed=80, module_dropout=0.0, config=config)
        target = SpiderBrain(seed=81, module_dropout=0.0, config=config)
        recurrent = source.true_monolithic_policy
        loaded = target.true_monolithic_policy
        self.assertIsNotNone(recurrent)
        self.assertIsNotNone(loaded)
        with tempfile.TemporaryDirectory() as tmpdir:
            source.save(tmpdir)
            target.load(tmpdir)
        np.testing.assert_allclose(
            loaded.W_option_action_policy_decoder,
            recurrent.W_option_action_policy_decoder,
        )
        np.testing.assert_allclose(
            loaded.W2_action_policy_core,
            recurrent.W2_action_policy_core,
        )

    def test_position_phase_option_dynamics_separate_action_policy_path_replay_round_trip_restores_action_policy_path_weights(self) -> None:
        config = BrainAblationConfig(
            name="true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_policy_path_teacher_option_replay_policy",
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
            direct_policy_option_decoder_recurrent_state=True,
            direct_policy_option_action_recurrent_core=True,
            direct_policy_option_action_separate_recurrent_head=True,
            direct_policy_option_action_separate_policy_path=True,
            direct_policy_continuation_replay_passes=2,
            direct_policy_continuation_replay_lr_scale=0.5,
        )
        source = SpiderBrain(seed=84, module_dropout=0.0, config=config)
        target = SpiderBrain(seed=85, module_dropout=0.0, config=config)
        recurrent = source.true_monolithic_policy
        loaded = target.true_monolithic_policy
        self.assertIsNotNone(recurrent)
        self.assertIsNotNone(loaded)
        with tempfile.TemporaryDirectory() as tmpdir:
            source.save(tmpdir)
            target.load(tmpdir)
        np.testing.assert_allclose(
            loaded.W_action_policy_path_input,
            recurrent.W_action_policy_path_input,
        )
        np.testing.assert_allclose(
            loaded.W_action_policy_path_prev,
            recurrent.W_action_policy_path_prev,
        )
        np.testing.assert_allclose(
            loaded.W_action_policy_path_action,
            recurrent.W_action_policy_path_action,
        )
        np.testing.assert_allclose(
            loaded.b_action_policy_path,
            recurrent.b_action_policy_path,
        )
        np.testing.assert_allclose(
            loaded.W2_action_policy_path,
            recurrent.W2_action_policy_path,
        )
        np.testing.assert_allclose(
            loaded.b2_action_policy_path,
            recurrent.b2_action_policy_path,
        )

    def test_position_phase_option_dynamics_separate_action_backbone_replay_round_trip_restores_action_backbone_weights(self) -> None:
        config = BrainAblationConfig(
            name="true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_backbone_teacher_option_replay_policy",
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
            direct_policy_option_decoder_recurrent_state=True,
            direct_policy_option_action_recurrent_core=True,
            direct_policy_option_action_separate_recurrent_head=True,
            direct_policy_option_action_separate_policy_path=True,
            direct_policy_option_action_separate_backbone=True,
            direct_policy_continuation_replay_passes=2,
            direct_policy_continuation_replay_lr_scale=0.5,
        )
        source = SpiderBrain(seed=88, module_dropout=0.0, config=config)
        target = SpiderBrain(seed=89, module_dropout=0.0, config=config)
        recurrent = source.true_monolithic_policy
        loaded = target.true_monolithic_policy
        self.assertIsNotNone(recurrent)
        self.assertIsNotNone(loaded)
        with tempfile.TemporaryDirectory() as tmpdir:
            source.save(tmpdir)
            target.load(tmpdir)
        np.testing.assert_allclose(
            loaded.W_action_backbone_input,
            recurrent.W_action_backbone_input,
        )
        np.testing.assert_allclose(
            loaded.W_action_backbone_prev,
            recurrent.W_action_backbone_prev,
        )
        np.testing.assert_allclose(
            loaded.W_action_backbone_action,
            recurrent.W_action_backbone_action,
        )
        np.testing.assert_allclose(
            loaded.b_action_backbone,
            recurrent.b_action_backbone,
        )
        np.testing.assert_allclose(
            loaded.W2_action_backbone,
            recurrent.W2_action_backbone,
        )
        np.testing.assert_allclose(
            loaded.b2_action_backbone,
            recurrent.b2_action_backbone,
        )

    def test_executive_option_policy_round_trip_restores_feedback_and_action_backbone_weights(self) -> None:
        config = BrainAblationConfig(
            name="true_monolithic_executive_option_policy",
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
        )
        source = SpiderBrain(seed=90, module_dropout=0.0, config=config)
        target = SpiderBrain(seed=91, module_dropout=0.0, config=config)
        recurrent = source.true_monolithic_policy
        loaded = target.true_monolithic_policy
        self.assertIsNotNone(recurrent)
        self.assertIsNotNone(loaded)
        with tempfile.TemporaryDirectory() as tmpdir:
            source.save(tmpdir)
            target.load(tmpdir)
        np.testing.assert_allclose(
            loaded.W2_phase_option_feedback,
            recurrent.W2_phase_option_feedback,
        )
        np.testing.assert_allclose(
            loaded.b2_phase_option_feedback,
            recurrent.b2_phase_option_feedback,
        )
        np.testing.assert_allclose(
            loaded.W2_option_transition_feedback,
            recurrent.W2_option_transition_feedback,
        )
        np.testing.assert_allclose(
            loaded.b2_option_transition_feedback,
            recurrent.b2_option_transition_feedback,
        )
        np.testing.assert_allclose(
            loaded.W2_action_backbone,
            recurrent.W2_action_backbone,
        )
        np.testing.assert_allclose(
            loaded.b2_action_backbone,
            recurrent.b2_action_backbone,
        )

    def test_executive_option_physiology_gating_round_trip_preserves_flag(self) -> None:
        config = BrainAblationConfig(
            name="true_monolithic_executive_option_physiology_gated_policy",
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
        )
        source = SpiderBrain(seed=92, module_dropout=0.0, config=config)
        target = SpiderBrain(seed=93, module_dropout=0.0, config=config)
        source_net = source.true_monolithic_policy
        target_net = target.true_monolithic_policy
        self.assertIsNotNone(source_net)
        self.assertIsNotNone(target_net)
        self.assertTrue(source_net.executive_physiology_option_gating)
        with tempfile.TemporaryDirectory() as tmpdir:
            source.save(tmpdir)
            target.load(tmpdir)
        self.assertTrue(target_net.executive_physiology_option_gating)

    def test_executive_option_release_round_trip_preserves_action_gate_flag(self) -> None:
        config = BrainAblationConfig(
            name="true_monolithic_executive_option_release_policy",
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
        )
        source = SpiderBrain(seed=94, module_dropout=0.0, config=config)
        target = SpiderBrain(seed=95, module_dropout=0.0, config=config)
        source_net = source.true_monolithic_policy
        target_net = target.true_monolithic_policy
        self.assertIsNotNone(source_net)
        self.assertIsNotNone(target_net)
        self.assertTrue(source_net.executive_affordance_action_gating)
        with tempfile.TemporaryDirectory() as tmpdir:
            source.save(tmpdir)
            target.load(tmpdir)
        self.assertTrue(target_net.executive_affordance_action_gating)

    def test_executive_option_masked_release_round_trip_preserves_mask_flag(self) -> None:
        config = BrainAblationConfig(
            name="true_monolithic_executive_option_masked_release_policy",
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
        )
        source = SpiderBrain(seed=96, module_dropout=0.0, config=config)
        target = SpiderBrain(seed=97, module_dropout=0.0, config=config)
        source_net = source.true_monolithic_policy
        target_net = target.true_monolithic_policy
        self.assertIsNotNone(source_net)
        self.assertIsNotNone(target_net)
        self.assertTrue(source_net.executive_option_action_masking)
        with tempfile.TemporaryDirectory() as tmpdir:
            source.save(tmpdir)
            target.load(tmpdir)
        self.assertTrue(target_net.executive_option_action_masking)

    def test_executive_option_event_release_round_trip_preserves_latch_flag(self) -> None:
        config = BrainAblationConfig(
            name="true_monolithic_executive_option_event_release_policy",
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
        )
        source = SpiderBrain(seed=98, module_dropout=0.0, config=config)
        target = SpiderBrain(seed=99, module_dropout=0.0, config=config)
        source_net = source.true_monolithic_policy
        target_net = target.true_monolithic_policy
        self.assertIsNotNone(source_net)
        self.assertIsNotNone(target_net)
        self.assertTrue(source_net.executive_event_release_latching)
        with tempfile.TemporaryDirectory() as tmpdir:
            source.save(tmpdir)
            target.load(tmpdir)
        self.assertTrue(target_net.executive_event_release_latching)

    def test_executive_option_transition_release_round_trip_preserves_commitment_flag(self) -> None:
        config = BrainAblationConfig(
            name="true_monolithic_executive_option_transition_release_policy",
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
        )
        source = SpiderBrain(seed=100, module_dropout=0.0, config=config)
        target = SpiderBrain(seed=101, module_dropout=0.0, config=config)
        source_net = source.true_monolithic_policy
        target_net = target.true_monolithic_policy
        self.assertIsNotNone(source_net)
        self.assertIsNotNone(target_net)
        self.assertTrue(source_net.executive_event_release_action_commitment)
        with tempfile.TemporaryDirectory() as tmpdir:
            source.save(tmpdir)
            target.load(tmpdir)
        self.assertTrue(target_net.executive_event_release_action_commitment)

    def test_executive_option_release_phase_round_trip_preserves_phase_flag(self) -> None:
        config = BrainAblationConfig(
            name="true_monolithic_executive_option_release_phase_policy",
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
        )
        source = SpiderBrain(seed=102, module_dropout=0.0, config=config)
        target = SpiderBrain(seed=103, module_dropout=0.0, config=config)
        source_net = source.true_monolithic_policy
        target_net = target.true_monolithic_policy
        self.assertIsNotNone(source_net)
        self.assertIsNotNone(target_net)
        self.assertTrue(source_net.executive_release_phase_state)
        with tempfile.TemporaryDirectory() as tmpdir:
            source.save(tmpdir)
            target.load(tmpdir)
        self.assertTrue(target_net.executive_release_phase_state)

    def test_executive_option_release_progression_round_trip_preserves_progression_flag(self) -> None:
        config = BrainAblationConfig(
            name="true_monolithic_executive_option_release_progression_policy",
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
            direct_policy_executive_release_progression=True,
        )
        source = SpiderBrain(seed=104, module_dropout=0.0, config=config)
        target = SpiderBrain(seed=105, module_dropout=0.0, config=config)
        source_net = source.true_monolithic_policy
        target_net = target.true_monolithic_policy
        self.assertIsNotNone(source_net)
        self.assertIsNotNone(target_net)
        self.assertTrue(source_net.executive_release_progression)
        with tempfile.TemporaryDirectory() as tmpdir:
            source.save(tmpdir)
            target.load(tmpdir)
        self.assertTrue(target_net.executive_release_progression)

    def test_executive_option_release_exit_contract_round_trip_preserves_flag(self) -> None:
        config = BrainAblationConfig(
            name="true_monolithic_executive_option_release_exit_contract_policy",
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
        )
        source = SpiderBrain(seed=106, module_dropout=0.0, config=config)
        target = SpiderBrain(seed=107, module_dropout=0.0, config=config)
        source_net = source.true_monolithic_policy
        target_net = target.true_monolithic_policy
        self.assertIsNotNone(source_net)
        self.assertIsNotNone(target_net)
        self.assertTrue(source_net.executive_release_exit_contract)
        with tempfile.TemporaryDirectory() as tmpdir:
            source.save(tmpdir)
            target.load(tmpdir)
        self.assertTrue(target_net.executive_release_exit_contract)

    def test_executive_option_release_substate_round_trip_preserves_flag(self) -> None:
        config = BrainAblationConfig(
            name="true_monolithic_executive_option_release_substate_policy",
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
        )
        source = SpiderBrain(seed=108, module_dropout=0.0, config=config)
        target = SpiderBrain(seed=109, module_dropout=0.0, config=config)
        source_net = source.true_monolithic_policy
        target_net = target.true_monolithic_policy
        self.assertIsNotNone(source_net)
        self.assertIsNotNone(target_net)
        self.assertTrue(source_net.executive_release_substate_progression)
        with tempfile.TemporaryDirectory() as tmpdir:
            source.save(tmpdir)
            target.load(tmpdir)
        self.assertTrue(target_net.executive_release_substate_progression)

    def test_executive_option_post_exit_continuation_round_trip_preserves_flag(self) -> None:
        config = BrainAblationConfig(
            name="true_monolithic_executive_option_post_exit_continuation_policy",
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
        )
        source = SpiderBrain(seed=110, module_dropout=0.0, config=config)
        target = SpiderBrain(seed=111, module_dropout=0.0, config=config)
        source_net = source.true_monolithic_policy
        target_net = target.true_monolithic_policy
        self.assertIsNotNone(source_net)
        self.assertIsNotNone(target_net)
        self.assertTrue(source_net.executive_post_exit_continuation)
        with tempfile.TemporaryDirectory() as tmpdir:
            source.save(tmpdir)
            target.load(tmpdir)
        self.assertTrue(target_net.executive_post_exit_continuation)

    def test_executive_option_post_exit_food_guidance_round_trip_preserves_flag(self) -> None:
        config = BrainAblationConfig(
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
        )
        source = SpiderBrain(seed=113, module_dropout=0.0, config=config)
        target = SpiderBrain(seed=114, module_dropout=0.0, config=config)
        source_net = source.true_monolithic_policy
        target_net = target.true_monolithic_policy
        self.assertIsNotNone(source_net)
        self.assertIsNotNone(target_net)
        self.assertTrue(source_net.executive_post_exit_food_guidance)
        with tempfile.TemporaryDirectory() as tmpdir:
            source.save(tmpdir)
            target.load(tmpdir)
        self.assertTrue(target_net.executive_post_exit_food_guidance)

    def test_executive_option_post_exit_food_commitment_round_trip_preserves_flag(self) -> None:
        config = BrainAblationConfig(
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
        )
        source = SpiderBrain(seed=116, module_dropout=0.0, config=config)
        target = SpiderBrain(seed=117, module_dropout=0.0, config=config)
        source_net = source.true_monolithic_policy
        target_net = target.true_monolithic_policy
        self.assertIsNotNone(source_net)
        self.assertIsNotNone(target_net)
        self.assertTrue(source_net.executive_post_exit_food_commitment)
        with tempfile.TemporaryDirectory() as tmpdir:
            source.save(tmpdir)
            target.load(tmpdir)
        self.assertTrue(target_net.executive_post_exit_food_commitment)

    def test_executive_option_post_exit_food_progression_round_trip_preserves_flag(self) -> None:
        config = BrainAblationConfig(
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
        )
        source = SpiderBrain(seed=119, module_dropout=0.0, config=config)
        target = SpiderBrain(seed=120, module_dropout=0.0, config=config)
        source_net = source.true_monolithic_policy
        target_net = target.true_monolithic_policy
        self.assertIsNotNone(source_net)
        self.assertIsNotNone(target_net)
        self.assertTrue(source_net.executive_post_exit_food_progression)
        with tempfile.TemporaryDirectory() as tmpdir:
            source.save(tmpdir)
            target.load(tmpdir)
        self.assertTrue(target_net.executive_post_exit_food_progression)

    def test_executive_option_post_exit_food_heading_progression_round_trip_preserves_flag(self) -> None:
        config = BrainAblationConfig(
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
        )
        source = SpiderBrain(seed=122, module_dropout=0.0, config=config)
        target = SpiderBrain(seed=123, module_dropout=0.0, config=config)
        source_net = source.true_monolithic_policy
        target_net = target.true_monolithic_policy
        self.assertIsNotNone(source_net)
        self.assertIsNotNone(target_net)
        self.assertTrue(source_net.executive_post_exit_food_heading_progression)
        with tempfile.TemporaryDirectory() as tmpdir:
            source.save(tmpdir)
            target.load(tmpdir)
        self.assertTrue(target_net.executive_post_exit_food_heading_progression)

    def test_executive_option_post_exit_smell_progression_round_trip_preserves_flag(self) -> None:
        config = BrainAblationConfig(
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
        )
        source = SpiderBrain(seed=125, module_dropout=0.0, config=config)
        target = SpiderBrain(seed=126, module_dropout=0.0, config=config)
        source_net = source.true_monolithic_policy
        target_net = target.true_monolithic_policy
        self.assertIsNotNone(source_net)
        self.assertIsNotNone(target_net)
        self.assertTrue(source_net.executive_post_exit_smell_progression)
        with tempfile.TemporaryDirectory() as tmpdir:
            source.save(tmpdir)
            target.load(tmpdir)
        self.assertTrue(target_net.executive_post_exit_smell_progression)

    def test_executive_option_post_exit_corridor_progression_round_trip_preserves_flag(self) -> None:
        config = BrainAblationConfig(
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
        )
        source = SpiderBrain(seed=225, module_dropout=0.0, config=config)
        target = SpiderBrain(seed=226, module_dropout=0.0, config=config)
        source_net = source.true_monolithic_policy
        target_net = target.true_monolithic_policy
        self.assertIsNotNone(source_net)
        self.assertIsNotNone(target_net)
        self.assertTrue(source_net.executive_post_exit_corridor_progression)
        with tempfile.TemporaryDirectory() as tmpdir:
            source.save(tmpdir)
            target.load(tmpdir)
        self.assertTrue(target_net.executive_post_exit_corridor_progression)

    def test_executive_option_post_exit_corridor_affordance_progression_round_trip_preserves_flag(self) -> None:
        config = BrainAblationConfig(
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
        )
        source = SpiderBrain(seed=227, module_dropout=0.0, config=config)
        target = SpiderBrain(seed=228, module_dropout=0.0, config=config)
        source_net = source.true_monolithic_policy
        target_net = target.true_monolithic_policy
        self.assertIsNotNone(source_net)
        self.assertIsNotNone(target_net)
        self.assertTrue(source_net.executive_post_exit_corridor_affordance_progression)
        with tempfile.TemporaryDirectory() as tmpdir:
            source.save(tmpdir)
            target.load(tmpdir)
        self.assertTrue(target_net.executive_post_exit_corridor_affordance_progression)

    def test_executive_option_post_food_return_round_trip_preserves_flag(self) -> None:
        config = BrainAblationConfig(
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
        )
        source = SpiderBrain(seed=229, module_dropout=0.0, config=config)
        target = SpiderBrain(seed=230, module_dropout=0.0, config=config)
        source_net = source.true_monolithic_policy
        target_net = target.true_monolithic_policy
        self.assertIsNotNone(source_net)
        self.assertIsNotNone(target_net)
        self.assertTrue(source_net.executive_post_food_return)
        with tempfile.TemporaryDirectory() as tmpdir:
            source.save(tmpdir)
            target.load(tmpdir)
        self.assertTrue(target_net.executive_post_food_return)

    def test_executive_option_post_food_vector_return_round_trip_preserves_flag(self) -> None:
        config = BrainAblationConfig(
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
        )
        source = SpiderBrain(seed=231, module_dropout=0.0, config=config)
        target = SpiderBrain(seed=232, module_dropout=0.0, config=config)
        source_net = source.true_monolithic_policy
        target_net = target.true_monolithic_policy
        self.assertIsNotNone(source_net)
        self.assertIsNotNone(target_net)
        self.assertTrue(source_net.executive_post_food_vector_return)
        with tempfile.TemporaryDirectory() as tmpdir:
            source.save(tmpdir)
            target.load(tmpdir)
        self.assertTrue(target_net.executive_post_food_vector_return)

    def test_executive_option_post_food_path_return_round_trip_preserves_flag(self) -> None:
        config = BrainAblationConfig(
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
        )
        source = SpiderBrain(seed=233, module_dropout=0.0, config=config)
        target = SpiderBrain(seed=234, module_dropout=0.0, config=config)
        source_net = source.true_monolithic_policy
        target_net = target.true_monolithic_policy
        self.assertIsNotNone(source_net)
        self.assertIsNotNone(target_net)
        self.assertTrue(source_net.executive_post_food_path_return)
        with tempfile.TemporaryDirectory() as tmpdir:
            source.save(tmpdir)
            target.load(tmpdir)
        self.assertTrue(target_net.executive_post_food_path_return)

    def test_event_attention_round_trip_restores_attention_weights(self) -> None:
        config = BrainAblationConfig(
            name="true_monolithic_event_attention_policy",
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
        )
        source = SpiderBrain(seed=35, module_dropout=0.0, config=config)
        target = SpiderBrain(seed=36, module_dropout=0.0, config=config)
        recurrent = source.true_monolithic_policy
        loaded = target.true_monolithic_policy
        self.assertIsNotNone(recurrent)
        self.assertIsNotNone(loaded)
        with tempfile.TemporaryDirectory() as tmpdir:
            source.save(tmpdir)
            target.load(tmpdir)
        np.testing.assert_allclose(loaded.W_query, recurrent.W_query)
        np.testing.assert_allclose(loaded.W_key, recurrent.W_key)
        np.testing.assert_allclose(loaded.W_value, recurrent.W_value)
        np.testing.assert_allclose(
            loaded.event_type_embeddings,
            recurrent.event_type_embeddings,
        )

    def test_option_head_round_trip_restores_option_weights(self) -> None:
        config = BrainAblationConfig(
            name="true_monolithic_option_policy",
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
        )
        source = SpiderBrain(seed=37, module_dropout=0.0, config=config)
        target = SpiderBrain(seed=38, module_dropout=0.0, config=config)
        recurrent = source.true_monolithic_policy
        loaded = target.true_monolithic_policy
        self.assertIsNotNone(recurrent)
        self.assertIsNotNone(loaded)
        recurrent.hidden_state[:] = 0.25
        with tempfile.TemporaryDirectory() as tmpdir:
            source.save(tmpdir)
            target.load(tmpdir)
        np.testing.assert_allclose(loaded.hidden_state, np.zeros(loaded.hidden_dim, dtype=float))
        np.testing.assert_allclose(loaded.W2_option, recurrent.W2_option)
        np.testing.assert_allclose(loaded.b2_option, recurrent.b2_option)
        np.testing.assert_allclose(
            loaded.option_action_bias,
            recurrent.option_action_bias,
        )

    def test_affordance_head_round_trip_restores_affordance_weights(self) -> None:
        config = BrainAblationConfig(
            name="true_monolithic_option_affordance_policy",
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
        )
        source = SpiderBrain(seed=39, module_dropout=0.0, config=config)
        target = SpiderBrain(seed=40, module_dropout=0.0, config=config)
        recurrent = source.true_monolithic_policy
        loaded = target.true_monolithic_policy
        self.assertIsNotNone(recurrent)
        self.assertIsNotNone(loaded)
        with tempfile.TemporaryDirectory() as tmpdir:
            source.save(tmpdir)
            target.load(tmpdir)
        np.testing.assert_allclose(
            loaded.W2_affordance_blocked,
            recurrent.W2_affordance_blocked,
        )
        np.testing.assert_allclose(
            loaded.b2_affordance_blocked,
            recurrent.b2_affordance_blocked,
        )
        np.testing.assert_allclose(
            loaded.W2_affordance_role,
            recurrent.W2_affordance_role,
        )
        np.testing.assert_allclose(
            loaded.b2_affordance_role,
            recurrent.b2_affordance_role,
        )

    def test_affordance_feedback_round_trip_restores_feedback_weights(self) -> None:
        config = BrainAblationConfig(
            name="true_monolithic_option_affordance_feedback_policy",
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
        )
        source = SpiderBrain(seed=41, module_dropout=0.0, config=config)
        target = SpiderBrain(seed=42, module_dropout=0.0, config=config)
        recurrent = source.true_monolithic_policy
        loaded = target.true_monolithic_policy
        self.assertIsNotNone(recurrent)
        self.assertIsNotNone(loaded)
        with tempfile.TemporaryDirectory() as tmpdir:
            source.save(tmpdir)
            target.load(tmpdir)
        np.testing.assert_allclose(
            loaded.W_affordance_feedback,
            recurrent.W_affordance_feedback,
        )
        np.testing.assert_allclose(
            loaded.b_affordance_feedback,
            recurrent.b_affordance_feedback,
        )
        np.testing.assert_allclose(
            loaded.W2_policy_feedback,
            recurrent.W2_policy_feedback,
        )
        np.testing.assert_allclose(
            loaded.b2_policy_feedback,
            recurrent.b2_policy_feedback,
        )
        np.testing.assert_allclose(
            loaded.W2_option_feedback,
            recurrent.W2_option_feedback,
        )
        np.testing.assert_allclose(
            loaded.b2_option_feedback,
            recurrent.b2_option_feedback,
        )

    def test_geometry_head_round_trip_restores_geometry_weights(self) -> None:
        config = BrainAblationConfig(
            name="true_monolithic_option_affordance_geometry_policy",
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
        )
        source = SpiderBrain(seed=43, module_dropout=0.0, config=config)
        target = SpiderBrain(seed=44, module_dropout=0.0, config=config)
        recurrent = source.true_monolithic_policy
        loaded = target.true_monolithic_policy
        self.assertIsNotNone(recurrent)
        self.assertIsNotNone(loaded)
        with tempfile.TemporaryDirectory() as tmpdir:
            source.save(tmpdir)
            target.load(tmpdir)
        np.testing.assert_allclose(
            loaded.W2_geometry,
            recurrent.W2_geometry,
        )
        np.testing.assert_allclose(
            loaded.b2_geometry,
            recurrent.b2_geometry,
        )

    def test_topology_head_round_trip_restores_shelter_column_weights(self) -> None:
        config = BrainAblationConfig(
            name="true_monolithic_option_affordance_topology_policy",
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
            direct_policy_shelter_column_head=True,
        )
        source = SpiderBrain(seed=45, module_dropout=0.0, config=config)
        target = SpiderBrain(seed=46, module_dropout=0.0, config=config)
        recurrent = source.true_monolithic_policy
        loaded = target.true_monolithic_policy
        self.assertIsNotNone(recurrent)
        self.assertIsNotNone(loaded)
        with tempfile.TemporaryDirectory() as tmpdir:
            source.save(tmpdir)
            target.load(tmpdir)
        np.testing.assert_allclose(
            loaded.W2_shelter_column,
            recurrent.W2_shelter_column,
        )
        np.testing.assert_allclose(
            loaded.b2_shelter_column,
            recurrent.b2_shelter_column,
        )

    def test_position_head_round_trip_restores_shelter_position_weights(self) -> None:
        config = BrainAblationConfig(
            name="true_monolithic_option_affordance_position_policy",
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
        )
        source = SpiderBrain(seed=47, module_dropout=0.0, config=config)
        target = SpiderBrain(seed=48, module_dropout=0.0, config=config)
        recurrent = source.true_monolithic_policy
        loaded = target.true_monolithic_policy
        self.assertIsNotNone(recurrent)
        self.assertIsNotNone(loaded)
        with tempfile.TemporaryDirectory() as tmpdir:
            source.save(tmpdir)
            target.load(tmpdir)
        np.testing.assert_allclose(
            loaded.W2_shelter_position,
            recurrent.W2_shelter_position,
        )
        np.testing.assert_allclose(
            loaded.b2_shelter_position,
            recurrent.b2_shelter_position,
        )

    def test_local_affordance_input_branch_round_trip_preserves_input_dim(self) -> None:
        config = BrainAblationConfig(
            name="true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_backbone_local_affordance_post_rest_probe_replayable_teacher_distill_option_replay_policy",
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
            direct_policy_handoff_teacher=True,
            direct_policy_handoff_option_teacher=True,
            direct_policy_post_rest_action_teacher=True,
            direct_policy_post_rest_release_sequence_teacher=True,
            direct_policy_post_rest_release_sequence_replay_boost=True,
            direct_policy_post_rest_release_sequence_distill=True,
            direct_policy_post_rest_probe_distillation=True,
            direct_policy_post_rest_probe_sequence_distillation=True,
            direct_policy_post_rest_probe_family_distillation=True,
            direct_policy_post_rest_probe_handoff_distillation=True,
            direct_policy_post_rest_probe_trajectory_distillation=True,
            direct_policy_post_rest_probe_cycle_distillation=True,
            direct_policy_post_rest_probe_trace_distillation=True,
            direct_policy_post_rest_probe_rollout_distillation=True,
            direct_policy_post_rest_probe_replayable_teacher_distillation=True,
            direct_policy_option_decoder_state=True,
            direct_policy_option_recurrent_dynamics=True,
            direct_policy_option_decoder_recurrent_state=True,
            direct_policy_option_action_recurrent_core=True,
            direct_policy_option_action_separate_recurrent_head=True,
            direct_policy_option_action_separate_policy_path=True,
            direct_policy_option_action_separate_backbone=True,
            direct_policy_continuation_replay_passes=2,
            direct_policy_continuation_replay_lr_scale=0.5,
        )
        source = SpiderBrain(seed=49, module_dropout=0.0, config=config)
        target = SpiderBrain(seed=50, module_dropout=0.0, config=config)
        recurrent = source.true_monolithic_policy
        loaded = target.true_monolithic_policy
        self.assertIsNotNone(recurrent)
        self.assertIsNotNone(loaded)
        with tempfile.TemporaryDirectory() as tmpdir:
            source.save(tmpdir)
            target.load(tmpdir)
        self.assertEqual(int(loaded.input_dim), int(recurrent.input_dim))

    def test_local_spatial_input_branch_round_trip_preserves_input_dim(self) -> None:
        config = BrainAblationConfig(
            name="true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_backbone_local_spatial_post_rest_probe_replayable_teacher_distill_option_replay_policy",
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
            direct_policy_handoff_teacher=True,
            direct_policy_handoff_option_teacher=True,
            direct_policy_post_rest_action_teacher=True,
            direct_policy_post_rest_release_sequence_teacher=True,
            direct_policy_post_rest_release_sequence_replay_boost=True,
            direct_policy_post_rest_release_sequence_distill=True,
            direct_policy_post_rest_probe_distillation=True,
            direct_policy_post_rest_probe_sequence_distillation=True,
            direct_policy_post_rest_probe_family_distillation=True,
            direct_policy_post_rest_probe_handoff_distillation=True,
            direct_policy_post_rest_probe_trajectory_distillation=True,
            direct_policy_post_rest_probe_cycle_distillation=True,
            direct_policy_post_rest_probe_trace_distillation=True,
            direct_policy_post_rest_probe_rollout_distillation=True,
            direct_policy_post_rest_probe_replayable_teacher_distillation=True,
            direct_policy_option_decoder_state=True,
            direct_policy_option_recurrent_dynamics=True,
            direct_policy_option_decoder_recurrent_state=True,
            direct_policy_option_action_recurrent_core=True,
            direct_policy_option_action_separate_recurrent_head=True,
            direct_policy_option_action_separate_policy_path=True,
            direct_policy_option_action_separate_backbone=True,
            direct_policy_continuation_replay_passes=2,
            direct_policy_continuation_replay_lr_scale=0.5,
        )
        source = SpiderBrain(seed=51, module_dropout=0.0, config=config)
        target = SpiderBrain(seed=52, module_dropout=0.0, config=config)
        recurrent = source.true_monolithic_policy
        loaded = target.true_monolithic_policy
        self.assertIsNotNone(recurrent)
        self.assertIsNotNone(loaded)
        with tempfile.TemporaryDirectory() as tmpdir:
            source.save(tmpdir)
            target.load(tmpdir)
        self.assertEqual(int(loaded.input_dim), int(recurrent.input_dim))

    def test_local_transition_input_branch_round_trip_preserves_input_dim(self) -> None:
        config = BrainAblationConfig(
            name="true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_backbone_local_transition_post_rest_probe_replayable_teacher_distill_option_replay_policy",
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
            direct_policy_handoff_teacher=True,
            direct_policy_handoff_option_teacher=True,
            direct_policy_post_rest_action_teacher=True,
            direct_policy_post_rest_release_sequence_teacher=True,
            direct_policy_post_rest_release_sequence_replay_boost=True,
            direct_policy_post_rest_release_sequence_distill=True,
            direct_policy_post_rest_probe_distillation=True,
            direct_policy_post_rest_probe_sequence_distillation=True,
            direct_policy_post_rest_probe_family_distillation=True,
            direct_policy_post_rest_probe_handoff_distillation=True,
            direct_policy_post_rest_probe_trajectory_distillation=True,
            direct_policy_post_rest_probe_cycle_distillation=True,
            direct_policy_post_rest_probe_trace_distillation=True,
            direct_policy_post_rest_probe_rollout_distillation=True,
            direct_policy_post_rest_probe_replayable_teacher_distillation=True,
            direct_policy_option_decoder_state=True,
            direct_policy_option_recurrent_dynamics=True,
            direct_policy_option_decoder_recurrent_state=True,
            direct_policy_option_action_recurrent_core=True,
            direct_policy_option_action_separate_recurrent_head=True,
            direct_policy_option_action_separate_policy_path=True,
            direct_policy_option_action_separate_backbone=True,
            direct_policy_continuation_replay_passes=2,
            direct_policy_continuation_replay_lr_scale=0.5,
        )
        source = SpiderBrain(seed=53, module_dropout=0.0, config=config)
        target = SpiderBrain(seed=54, module_dropout=0.0, config=config)
        recurrent = source.true_monolithic_policy
        loaded = target.true_monolithic_policy
        self.assertIsNotNone(recurrent)
        self.assertIsNotNone(loaded)
        with tempfile.TemporaryDirectory() as tmpdir:
            source.save(tmpdir)
            target.load(tmpdir)
        self.assertEqual(int(loaded.input_dim), int(recurrent.input_dim))

    def test_local_transition_rollout_input_branch_round_trip_preserves_input_dim(self) -> None:
        config = BrainAblationConfig(
            name="true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_backbone_local_transition_rollout_post_rest_probe_replayable_teacher_distill_option_replay_policy",
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
            direct_policy_handoff_teacher=True,
            direct_policy_handoff_option_teacher=True,
            direct_policy_post_rest_action_teacher=True,
            direct_policy_post_rest_release_sequence_teacher=True,
            direct_policy_post_rest_release_sequence_replay_boost=True,
            direct_policy_post_rest_release_sequence_distill=True,
            direct_policy_post_rest_probe_distillation=True,
            direct_policy_post_rest_probe_sequence_distillation=True,
            direct_policy_post_rest_probe_family_distillation=True,
            direct_policy_post_rest_probe_handoff_distillation=True,
            direct_policy_post_rest_probe_trajectory_distillation=True,
            direct_policy_post_rest_probe_cycle_distillation=True,
            direct_policy_post_rest_probe_trace_distillation=True,
            direct_policy_post_rest_probe_rollout_distillation=True,
            direct_policy_post_rest_probe_replayable_teacher_distillation=True,
            direct_policy_option_decoder_state=True,
            direct_policy_option_recurrent_dynamics=True,
            direct_policy_option_decoder_recurrent_state=True,
            direct_policy_option_action_recurrent_core=True,
            direct_policy_option_action_separate_recurrent_head=True,
            direct_policy_option_action_separate_policy_path=True,
            direct_policy_option_action_separate_backbone=True,
            direct_policy_continuation_replay_passes=2,
            direct_policy_continuation_replay_lr_scale=0.5,
        )
        source = SpiderBrain(seed=55, module_dropout=0.0, config=config)
        target = SpiderBrain(seed=56, module_dropout=0.0, config=config)
        recurrent = source.true_monolithic_policy
        loaded = target.true_monolithic_policy
        self.assertIsNotNone(recurrent)
        self.assertIsNotNone(loaded)
        with tempfile.TemporaryDirectory() as tmpdir:
            source.save(tmpdir)
            target.load(tmpdir)
        self.assertEqual(int(loaded.input_dim), int(recurrent.input_dim))

    def test_local_geodesic_input_branch_round_trip_preserves_input_dim(self) -> None:
        config = BrainAblationConfig(
            name="true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_backbone_geodesic_inputs_post_rest_probe_replayable_teacher_distill_option_replay_policy",
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
            direct_policy_handoff_teacher=True,
            direct_policy_handoff_option_teacher=True,
            direct_policy_post_rest_action_teacher=True,
            direct_policy_post_rest_release_sequence_teacher=True,
            direct_policy_post_rest_release_sequence_replay_boost=True,
            direct_policy_post_rest_release_sequence_distill=True,
            direct_policy_post_rest_probe_distillation=True,
            direct_policy_post_rest_probe_sequence_distillation=True,
            direct_policy_post_rest_probe_family_distillation=True,
            direct_policy_post_rest_probe_handoff_distillation=True,
            direct_policy_post_rest_probe_trajectory_distillation=True,
            direct_policy_post_rest_probe_cycle_distillation=True,
            direct_policy_post_rest_probe_trace_distillation=True,
            direct_policy_post_rest_probe_rollout_distillation=True,
            direct_policy_post_rest_probe_replayable_teacher_distillation=True,
            direct_policy_option_decoder_state=True,
            direct_policy_option_recurrent_dynamics=True,
            direct_policy_option_decoder_recurrent_state=True,
            direct_policy_option_action_recurrent_core=True,
            direct_policy_option_action_separate_recurrent_head=True,
            direct_policy_option_action_separate_policy_path=True,
            direct_policy_option_action_separate_backbone=True,
            direct_policy_continuation_replay_passes=2,
            direct_policy_continuation_replay_lr_scale=0.5,
        )
        source = SpiderBrain(seed=57, module_dropout=0.0, config=config)
        target = SpiderBrain(seed=58, module_dropout=0.0, config=config)
        recurrent = source.true_monolithic_policy
        loaded = target.true_monolithic_policy
        self.assertIsNotNone(recurrent)
        self.assertIsNotNone(loaded)
        with tempfile.TemporaryDirectory() as tmpdir:
            source.save(tmpdir)
            target.load(tmpdir)
        self.assertEqual(int(loaded.input_dim), int(recurrent.input_dim))

    def test_local_geodesic_margin_branch_round_trip_preserves_input_dim(self) -> None:
        config = BrainAblationConfig(
            name="true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_backbone_geodesic_inputs_post_rest_probe_replayable_teacher_distill_option_margin_replay_policy",
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
            direct_policy_handoff_teacher=True,
            direct_policy_handoff_option_teacher=True,
            direct_policy_post_rest_action_teacher=True,
            direct_policy_post_rest_release_sequence_teacher=True,
            direct_policy_post_rest_release_sequence_replay_boost=True,
            direct_policy_post_rest_release_sequence_distill=True,
            direct_policy_post_rest_probe_distillation=True,
            direct_policy_post_rest_probe_sequence_distillation=True,
            direct_policy_post_rest_probe_family_distillation=True,
            direct_policy_post_rest_probe_handoff_distillation=True,
            direct_policy_post_rest_probe_trajectory_distillation=True,
            direct_policy_post_rest_probe_cycle_distillation=True,
            direct_policy_post_rest_probe_trace_distillation=True,
            direct_policy_post_rest_probe_rollout_distillation=True,
            direct_policy_post_rest_probe_replayable_teacher_distillation=True,
            direct_policy_option_decoder_state=True,
            direct_policy_option_recurrent_dynamics=True,
            direct_policy_option_decoder_recurrent_state=True,
            direct_policy_option_action_recurrent_core=True,
            direct_policy_option_action_separate_recurrent_head=True,
            direct_policy_option_action_separate_policy_path=True,
            direct_policy_option_action_separate_backbone=True,
            direct_policy_continuation_replay_passes=2,
            direct_policy_continuation_replay_lr_scale=0.5,
            direct_policy_continuation_margin_weight=1.0,
        )
        source = SpiderBrain(seed=59, module_dropout=0.0, config=config)
        target = SpiderBrain(seed=60, module_dropout=0.0, config=config)
        recurrent = source.true_monolithic_policy
        loaded = target.true_monolithic_policy
        self.assertIsNotNone(recurrent)
        self.assertIsNotNone(loaded)
        with tempfile.TemporaryDirectory() as tmpdir:
            source.save(tmpdir)
            target.load(tmpdir)
        self.assertEqual(int(loaded.input_dim), int(recurrent.input_dim))

    def test_local_geodesic_option_transition_branch_round_trip_preserves_input_dim(self) -> None:
        config = BrainAblationConfig(
            name="true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_backbone_geodesic_inputs_option_transition_feedback_post_rest_probe_replayable_teacher_distill_option_replay_policy",
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
            direct_policy_handoff_teacher=True,
            direct_policy_handoff_option_teacher=True,
            direct_policy_post_rest_action_teacher=True,
            direct_policy_post_rest_release_sequence_teacher=True,
            direct_policy_post_rest_release_sequence_replay_boost=True,
            direct_policy_post_rest_release_sequence_distill=True,
            direct_policy_post_rest_probe_distillation=True,
            direct_policy_post_rest_probe_sequence_distillation=True,
            direct_policy_post_rest_probe_family_distillation=True,
            direct_policy_post_rest_probe_handoff_distillation=True,
            direct_policy_post_rest_probe_trajectory_distillation=True,
            direct_policy_post_rest_probe_cycle_distillation=True,
            direct_policy_post_rest_probe_trace_distillation=True,
            direct_policy_post_rest_probe_rollout_distillation=True,
            direct_policy_post_rest_probe_replayable_teacher_distillation=True,
            direct_policy_option_decoder_state=True,
            direct_policy_option_recurrent_dynamics=True,
            direct_policy_option_decoder_recurrent_state=True,
            direct_policy_option_action_recurrent_core=True,
            direct_policy_option_action_separate_recurrent_head=True,
            direct_policy_option_action_separate_policy_path=True,
            direct_policy_option_action_separate_backbone=True,
            direct_policy_option_transition_feedback=True,
            direct_policy_continuation_replay_passes=2,
            direct_policy_continuation_replay_lr_scale=0.5,
        )
        source = SpiderBrain(seed=61, module_dropout=0.0, config=config)
        target = SpiderBrain(seed=62, module_dropout=0.0, config=config)
        recurrent = source.true_monolithic_policy
        loaded = target.true_monolithic_policy
        self.assertIsNotNone(recurrent)
        self.assertIsNotNone(loaded)
        with tempfile.TemporaryDirectory() as tmpdir:
            source.save(tmpdir)
            target.load(tmpdir)
        self.assertEqual(int(loaded.input_dim), int(recurrent.input_dim))

    def test_local_geodesic_phase_option_feedback_branch_round_trip_preserves_input_dim(self) -> None:
        config = BrainAblationConfig(
            name="true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_backbone_geodesic_inputs_phase_option_feedback_post_rest_probe_replayable_teacher_distill_option_replay_policy",
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
            direct_policy_handoff_teacher=True,
            direct_policy_handoff_option_teacher=True,
            direct_policy_post_rest_action_teacher=True,
            direct_policy_post_rest_release_sequence_teacher=True,
            direct_policy_post_rest_release_sequence_replay_boost=True,
            direct_policy_post_rest_release_sequence_distill=True,
            direct_policy_post_rest_probe_distillation=True,
            direct_policy_post_rest_probe_sequence_distillation=True,
            direct_policy_post_rest_probe_family_distillation=True,
            direct_policy_post_rest_probe_handoff_distillation=True,
            direct_policy_post_rest_probe_trajectory_distillation=True,
            direct_policy_post_rest_probe_cycle_distillation=True,
            direct_policy_post_rest_probe_trace_distillation=True,
            direct_policy_post_rest_probe_rollout_distillation=True,
            direct_policy_post_rest_probe_replayable_teacher_distillation=True,
            direct_policy_option_decoder_state=True,
            direct_policy_option_recurrent_dynamics=True,
            direct_policy_option_decoder_recurrent_state=True,
            direct_policy_option_action_recurrent_core=True,
            direct_policy_option_action_separate_recurrent_head=True,
            direct_policy_option_action_separate_policy_path=True,
            direct_policy_option_action_separate_backbone=True,
            direct_policy_phase_option_feedback=True,
            direct_policy_continuation_replay_passes=2,
            direct_policy_continuation_replay_lr_scale=0.5,
        )
        source = SpiderBrain(seed=63, module_dropout=0.0, config=config)
        target = SpiderBrain(seed=64, module_dropout=0.0, config=config)
        recurrent = source.true_monolithic_policy
        loaded = target.true_monolithic_policy
        self.assertIsNotNone(recurrent)
        self.assertIsNotNone(loaded)
        with tempfile.TemporaryDirectory() as tmpdir:
            source.save(tmpdir)
            target.load(tmpdir)
        self.assertEqual(int(loaded.input_dim), int(recurrent.input_dim))

    def test_local_geodesic_option_sequence_branch_round_trip_preserves_input_dim(self) -> None:
        config = BrainAblationConfig(
            name="true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_backbone_geodesic_inputs_option_sequence_head_post_rest_probe_replayable_teacher_distill_option_replay_policy",
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
            direct_policy_handoff_teacher=True,
            direct_policy_handoff_option_teacher=True,
            direct_policy_post_rest_action_teacher=True,
            direct_policy_post_rest_release_sequence_teacher=True,
            direct_policy_post_rest_release_sequence_replay_boost=True,
            direct_policy_post_rest_release_sequence_distill=True,
            direct_policy_post_rest_probe_distillation=True,
            direct_policy_post_rest_probe_sequence_distillation=True,
            direct_policy_post_rest_probe_family_distillation=True,
            direct_policy_post_rest_probe_handoff_distillation=True,
            direct_policy_post_rest_probe_trajectory_distillation=True,
            direct_policy_post_rest_probe_cycle_distillation=True,
            direct_policy_post_rest_probe_trace_distillation=True,
            direct_policy_post_rest_probe_rollout_distillation=True,
            direct_policy_post_rest_probe_replayable_teacher_distillation=True,
            direct_policy_option_decoder_state=True,
            direct_policy_option_recurrent_dynamics=True,
            direct_policy_option_decoder_recurrent_state=True,
            direct_policy_option_action_recurrent_core=True,
            direct_policy_option_action_separate_recurrent_head=True,
            direct_policy_option_action_separate_policy_path=True,
            direct_policy_option_action_separate_backbone=True,
            direct_policy_option_sequence_head=True,
            direct_policy_continuation_replay_passes=2,
            direct_policy_continuation_replay_lr_scale=0.5,
        )
        source = SpiderBrain(seed=65, module_dropout=0.0, config=config)
        target = SpiderBrain(seed=66, module_dropout=0.0, config=config)
        recurrent = source.true_monolithic_policy
        loaded = target.true_monolithic_policy
        self.assertIsNotNone(recurrent)
        self.assertIsNotNone(loaded)
        with tempfile.TemporaryDirectory() as tmpdir:
            source.save(tmpdir)
            target.load(tmpdir)
        self.assertEqual(int(loaded.input_dim), int(recurrent.input_dim))

    def test_local_geodesic_trace_distill_branch_round_trip_preserves_input_dim(self) -> None:
        config = BrainAblationConfig(
            name="true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_backbone_geodesic_inputs_post_rest_probe_trace_distill_option_replay_policy",
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
            direct_policy_handoff_teacher=True,
            direct_policy_handoff_option_teacher=True,
            direct_policy_post_rest_action_teacher=True,
            direct_policy_post_rest_release_sequence_teacher=True,
            direct_policy_post_rest_release_sequence_replay_boost=True,
            direct_policy_post_rest_release_sequence_distill=True,
            direct_policy_post_rest_probe_distillation=True,
            direct_policy_post_rest_probe_sequence_distillation=True,
            direct_policy_post_rest_probe_family_distillation=True,
            direct_policy_post_rest_probe_handoff_distillation=True,
            direct_policy_post_rest_probe_trajectory_distillation=True,
            direct_policy_post_rest_probe_cycle_distillation=True,
            direct_policy_post_rest_probe_trace_distillation=True,
            direct_policy_option_decoder_state=True,
            direct_policy_option_recurrent_dynamics=True,
            direct_policy_option_decoder_recurrent_state=True,
            direct_policy_option_action_recurrent_core=True,
            direct_policy_option_action_separate_recurrent_head=True,
            direct_policy_option_action_separate_policy_path=True,
            direct_policy_option_action_separate_backbone=True,
            direct_policy_continuation_replay_passes=2,
            direct_policy_continuation_replay_lr_scale=0.5,
        )
        source = SpiderBrain(seed=67, module_dropout=0.0, config=config)
        target = SpiderBrain(seed=68, module_dropout=0.0, config=config)
        recurrent = source.true_monolithic_policy
        loaded = target.true_monolithic_policy
        self.assertIsNotNone(recurrent)
        self.assertIsNotNone(loaded)
        with tempfile.TemporaryDirectory() as tmpdir:
            source.save(tmpdir)
            target.load(tmpdir)
        self.assertEqual(int(loaded.input_dim), int(recurrent.input_dim))

    def test_local_geodesic_action_token_branch_round_trip_preserves_input_dim(self) -> None:
        config = BrainAblationConfig(
            name="true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_backbone_geodesic_inputs_action_token_post_rest_probe_replayable_teacher_distill_option_replay_policy",
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
            direct_policy_handoff_teacher=True,
            direct_policy_handoff_option_teacher=True,
            direct_policy_post_rest_action_teacher=True,
            direct_policy_post_rest_release_sequence_teacher=True,
            direct_policy_post_rest_release_sequence_replay_boost=True,
            direct_policy_post_rest_release_sequence_distill=True,
            direct_policy_post_rest_probe_distillation=True,
            direct_policy_post_rest_probe_sequence_distillation=True,
            direct_policy_post_rest_probe_family_distillation=True,
            direct_policy_post_rest_probe_handoff_distillation=True,
            direct_policy_post_rest_probe_trajectory_distillation=True,
            direct_policy_post_rest_probe_cycle_distillation=True,
            direct_policy_post_rest_probe_trace_distillation=True,
            direct_policy_post_rest_probe_rollout_distillation=True,
            direct_policy_post_rest_probe_replayable_teacher_distillation=True,
            direct_policy_option_decoder_state=True,
            direct_policy_option_recurrent_dynamics=True,
            direct_policy_option_decoder_recurrent_state=True,
            direct_policy_option_action_recurrent_core=True,
            direct_policy_option_action_separate_recurrent_head=True,
            direct_policy_option_action_separate_policy_path=True,
            direct_policy_option_action_separate_backbone=True,
            direct_policy_option_action_token_decoder=True,
            direct_policy_continuation_replay_passes=2,
            direct_policy_continuation_replay_lr_scale=0.5,
        )
        source = SpiderBrain(seed=69, module_dropout=0.0, config=config)
        target = SpiderBrain(seed=70, module_dropout=0.0, config=config)
        recurrent = source.true_monolithic_policy
        loaded = target.true_monolithic_policy
        self.assertIsNotNone(recurrent)
        self.assertIsNotNone(loaded)
        with tempfile.TemporaryDirectory() as tmpdir:
            source.save(tmpdir)
            target.load(tmpdir)
        self.assertEqual(int(loaded.input_dim), int(recurrent.input_dim))

    def test_transition_prediction_branch_round_trip_restores_transition_weights(self) -> None:
        config = BrainAblationConfig(
            name="true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_backbone_transition_prediction_feedback_post_rest_probe_replayable_teacher_distill_option_replay_policy",
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
            direct_policy_handoff_teacher=True,
            direct_policy_handoff_option_teacher=True,
            direct_policy_post_rest_action_teacher=True,
            direct_policy_post_rest_release_sequence_teacher=True,
            direct_policy_post_rest_release_sequence_replay_boost=True,
            direct_policy_post_rest_release_sequence_distill=True,
            direct_policy_post_rest_probe_distillation=True,
            direct_policy_post_rest_probe_sequence_distillation=True,
            direct_policy_post_rest_probe_family_distillation=True,
            direct_policy_post_rest_probe_handoff_distillation=True,
            direct_policy_post_rest_probe_trajectory_distillation=True,
            direct_policy_post_rest_probe_cycle_distillation=True,
            direct_policy_post_rest_probe_trace_distillation=True,
            direct_policy_post_rest_probe_rollout_distillation=True,
            direct_policy_post_rest_probe_replayable_teacher_distillation=True,
            direct_policy_option_decoder_state=True,
            direct_policy_option_recurrent_dynamics=True,
            direct_policy_option_decoder_recurrent_state=True,
            direct_policy_option_action_recurrent_core=True,
            direct_policy_option_action_separate_recurrent_head=True,
            direct_policy_option_action_separate_policy_path=True,
            direct_policy_option_action_separate_backbone=True,
            direct_policy_continuation_replay_passes=2,
            direct_policy_continuation_replay_lr_scale=0.5,
        )
        source = SpiderBrain(seed=57, module_dropout=0.0, config=config)
        target = SpiderBrain(seed=58, module_dropout=0.0, config=config)
        recurrent = source.true_monolithic_policy
        loaded = target.true_monolithic_policy
        self.assertIsNotNone(recurrent)
        self.assertIsNotNone(loaded)
        with tempfile.TemporaryDirectory() as tmpdir:
            source.save(tmpdir)
            target.load(tmpdir)
        np.testing.assert_allclose(
            loaded.W2_transition_prediction,
            recurrent.W2_transition_prediction,
        )
        np.testing.assert_allclose(
            loaded.b2_transition_prediction,
            recurrent.b2_transition_prediction,
        )
        np.testing.assert_allclose(
            loaded.W_transition_prediction_feedback,
            recurrent.W_transition_prediction_feedback,
        )
        np.testing.assert_allclose(
            loaded.b_transition_prediction_feedback,
            recurrent.b_transition_prediction_feedback,
        )

    def test_transition_rollout_prediction_branch_round_trip_restores_transition_weights(self) -> None:
        config = BrainAblationConfig(
            name="true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_backbone_transition_rollout_prediction_feedback_post_rest_probe_replayable_teacher_distill_option_replay_policy",
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
            direct_policy_handoff_teacher=True,
            direct_policy_handoff_option_teacher=True,
            direct_policy_post_rest_action_teacher=True,
            direct_policy_post_rest_release_sequence_teacher=True,
            direct_policy_post_rest_release_sequence_replay_boost=True,
            direct_policy_post_rest_release_sequence_distill=True,
            direct_policy_post_rest_probe_distillation=True,
            direct_policy_post_rest_probe_sequence_distillation=True,
            direct_policy_post_rest_probe_family_distillation=True,
            direct_policy_post_rest_probe_handoff_distillation=True,
            direct_policy_post_rest_probe_trajectory_distillation=True,
            direct_policy_post_rest_probe_cycle_distillation=True,
            direct_policy_post_rest_probe_trace_distillation=True,
            direct_policy_post_rest_probe_rollout_distillation=True,
            direct_policy_post_rest_probe_replayable_teacher_distillation=True,
            direct_policy_option_decoder_state=True,
            direct_policy_option_recurrent_dynamics=True,
            direct_policy_option_decoder_recurrent_state=True,
            direct_policy_option_action_recurrent_core=True,
            direct_policy_option_action_separate_recurrent_head=True,
            direct_policy_option_action_separate_policy_path=True,
            direct_policy_option_action_separate_backbone=True,
            direct_policy_continuation_replay_passes=2,
            direct_policy_continuation_replay_lr_scale=0.5,
        )
        source = SpiderBrain(seed=59, module_dropout=0.0, config=config)
        target = SpiderBrain(seed=60, module_dropout=0.0, config=config)
        recurrent = source.true_monolithic_policy
        loaded = target.true_monolithic_policy
        self.assertIsNotNone(recurrent)
        self.assertIsNotNone(loaded)
        with tempfile.TemporaryDirectory() as tmpdir:
            source.save(tmpdir)
            target.load(tmpdir)
        np.testing.assert_allclose(
            loaded.W2_transition_rollout_prediction,
            recurrent.W2_transition_rollout_prediction,
        )
        np.testing.assert_allclose(
            loaded.b2_transition_rollout_prediction,
            recurrent.b2_transition_rollout_prediction,
        )
        np.testing.assert_allclose(
            loaded.W_transition_rollout_prediction_feedback,
            recurrent.W_transition_rollout_prediction_feedback,
        )
        np.testing.assert_allclose(
            loaded.b_transition_rollout_prediction_feedback,
            recurrent.b_transition_rollout_prediction_feedback,
        )

class SimulationCheckpointSortKeyTest(unittest.TestCase):
    """Tests for checkpoint_candidate_sort_key()."""

    def test_invalid_metric_raises_value_error(self) -> None:
        candidate = {
            "scenario_success_rate": 0.5,
            "episode_success_rate": 0.5,
            "mean_reward": 0.5,
            "episode": 1,
        }
        with self.assertRaises(ValueError):
            checkpoint_candidate_sort_key(
                candidate, primary_metric="invalid_metric"
            )

    def test_tiebreaker_config_preserves_legacy_tuple(self) -> None:
        candidate = {
            "scenario_success_rate": 0.5,
            "episode_success_rate": 0.4,
            "mean_reward": 1.0,
            "evaluation_summary": {
                "mean_final_reflex_override_rate": 0.2,
                "mean_reflex_dominance": 0.3,
            },
            "episode": 7,
        }
        key = checkpoint_candidate_sort_key(
            candidate,
            selection_config=CheckpointSelectionConfig(
                metric="scenario_success_rate",
                penalty_mode=CheckpointPenaltyMode.TIEBREAKER,
            ),
        )
        self.assertEqual(key, (0.5, 0.4, 1.0, -0.2, -0.3, 7))

    def test_direct_penalty_mode_prepends_composite_score(self) -> None:
        candidate = {
            "scenario_success_rate": 0.8,
            "episode_success_rate": 0.4,
            "mean_reward": 1.0,
            "evaluation_summary": {
                "mean_final_reflex_override_rate": 0.25,
                "mean_reflex_dominance": 0.5,
            },
            "episode": 7,
        }
        key = checkpoint_candidate_sort_key(
            candidate,
            selection_config=CheckpointSelectionConfig(
                metric="scenario_success_rate",
                override_penalty_weight=0.4,
                dominance_penalty_weight=0.2,
                penalty_mode=CheckpointPenaltyMode.DIRECT,
            ),
        )
        self.assertAlmostEqual(key[0], 0.6)
        self.assertEqual(key[1:4], (0.8, 0.4, 1.0))
        self.assertEqual(key[4:], (-0.25, -0.5, 7))

    def test_direct_penalty_can_override_primary_metric_gap(self) -> None:
        high_dependence = {
            "scenario_success_rate": 0.8,
            "episode_success_rate": 0.5,
            "mean_reward": 0.5,
            "evaluation_summary": {
                "mean_final_reflex_override_rate": 0.5,
                "mean_reflex_dominance": 0.0,
            },
            "episode": 1,
        }
        low_dependence = {
            "scenario_success_rate": 0.7,
            "episode_success_rate": 0.5,
            "mean_reward": 0.5,
            "evaluation_summary": {
                "mean_final_reflex_override_rate": 0.0,
                "mean_reflex_dominance": 0.0,
            },
            "episode": 2,
        }
        selection_config = CheckpointSelectionConfig(
            metric="scenario_success_rate",
            override_penalty_weight=1.0,
            penalty_mode=CheckpointPenaltyMode.DIRECT,
        )

        self.assertGreater(
            checkpoint_candidate_sort_key(
                low_dependence,
                selection_config=selection_config,
            ),
            checkpoint_candidate_sort_key(
                high_dependence,
                selection_config=selection_config,
            ),
        )

    def test_scenario_success_rate_primary_metric(self) -> None:
        high = {
            "scenario_success_rate": 0.9,
            "episode_success_rate": 0.1,
            "mean_reward": 0.1,
            "episode": 1,
        }
        low = {
            "scenario_success_rate": 0.1,
            "episode_success_rate": 0.9,
            "mean_reward": 0.9,
            "episode": 2,
        }
        key_high = checkpoint_candidate_sort_key(
            high, primary_metric="scenario_success_rate"
        )
        key_low = checkpoint_candidate_sort_key(
            low, primary_metric="scenario_success_rate"
        )
        self.assertGreater(key_high, key_low)

    def test_episode_success_rate_primary_metric(self) -> None:
        a = {
            "scenario_success_rate": 0.9,
            "episode_success_rate": 0.1,
            "mean_reward": 0.5,
            "episode": 1,
        }
        b = {
            "scenario_success_rate": 0.2,
            "episode_success_rate": 0.8,
            "mean_reward": 0.5,
            "episode": 2,
        }
        key_a = checkpoint_candidate_sort_key(
            a, primary_metric="episode_success_rate"
        )
        key_b = checkpoint_candidate_sort_key(
            b, primary_metric="episode_success_rate"
        )
        self.assertGreater(key_b, key_a)

    def test_episode_tiebreaker_favors_later_episode(self) -> None:
        # Same metric scores but different episode number
        early = {
            "scenario_success_rate": 0.5,
            "episode_success_rate": 0.5,
            "mean_reward": 0.5,
            "episode": 3,
        }
        late = {
            "scenario_success_rate": 0.5,
            "episode_success_rate": 0.5,
            "mean_reward": 0.5,
            "episode": 10,
        }
        key_early = checkpoint_candidate_sort_key(
            early, primary_metric="scenario_success_rate"
        )
        key_late = checkpoint_candidate_sort_key(
            late, primary_metric="scenario_success_rate"
        )
        self.assertGreater(key_late, key_early)

    def test_reflex_dependence_penalty_precedes_episode_tiebreaker(self) -> None:
        low_reflex = {
            "scenario_success_rate": 0.5,
            "episode_success_rate": 0.5,
            "mean_reward": 0.5,
            "evaluation_summary": {
                "mean_final_reflex_override_rate": 0.1,
                "mean_reflex_dominance": 0.1,
            },
            "episode": 3,
        }
        high_reflex = {
            "scenario_success_rate": 0.5,
            "episode_success_rate": 0.5,
            "mean_reward": 0.5,
            "evaluation_summary": {
                "mean_final_reflex_override_rate": 0.9,
                "mean_reflex_dominance": 0.9,
            },
            "episode": 10,
        }
        key_low_reflex = checkpoint_candidate_sort_key(
            low_reflex, primary_metric="scenario_success_rate"
        )
        key_high_reflex = checkpoint_candidate_sort_key(
            high_reflex, primary_metric="scenario_success_rate"
        )
        self.assertGreater(key_low_reflex, key_high_reflex)

    def test_high_reflex_override_rate_lowers_checkpoint_ranking(self) -> None:
        low_override = {
            "scenario_success_rate": 0.5,
            "episode_success_rate": 0.5,
            "mean_reward": 0.5,
            "evaluation_summary": {
                "mean_final_reflex_override_rate": 0.0,
                "mean_reflex_dominance": 0.1,
            },
            "episode": 5,
        }
        high_override = {
            "scenario_success_rate": 0.5,
            "episode_success_rate": 0.5,
            "mean_reward": 0.5,
            "evaluation_summary": {
                "mean_final_reflex_override_rate": 0.8,
                "mean_reflex_dominance": 0.1,
            },
            "episode": 5,
        }

        self.assertGreater(
            checkpoint_candidate_sort_key(
                low_override,
                primary_metric="scenario_success_rate",
            ),
            checkpoint_candidate_sort_key(
                high_override,
                primary_metric="scenario_success_rate",
            ),
        )

    def test_high_reflex_dominance_lowers_checkpoint_ranking(self) -> None:
        low_dominance = {
            "scenario_success_rate": 0.5,
            "episode_success_rate": 0.5,
            "mean_reward": 0.5,
            "evaluation_summary": {
                "mean_final_reflex_override_rate": 0.1,
                "mean_reflex_dominance": 0.0,
            },
            "episode": 5,
        }
        high_dominance = {
            "scenario_success_rate": 0.5,
            "episode_success_rate": 0.5,
            "mean_reward": 0.5,
            "evaluation_summary": {
                "mean_final_reflex_override_rate": 0.1,
                "mean_reflex_dominance": 0.8,
            },
            "episode": 5,
        }

        self.assertGreater(
            checkpoint_candidate_sort_key(
                low_dominance,
                primary_metric="scenario_success_rate",
            ),
            checkpoint_candidate_sort_key(
                high_dominance,
                primary_metric="scenario_success_rate",
            ),
        )

    def test_sort_key_is_comparable_for_checkpoint_ranking(self) -> None:
        lower_success = {
            "scenario_success_rate": 0.6,
            "episode_success_rate": 0.6,
            "mean_reward": 0.6,
            "episode": 5,
        }
        higher_success = {
            "scenario_success_rate": 0.7,
            "episode_success_rate": 0.6,
            "mean_reward": 0.6,
            "episode": 5,
        }
        lower_key = checkpoint_candidate_sort_key(
            lower_success,
            primary_metric="scenario_success_rate",
        )
        higher_key = checkpoint_candidate_sort_key(
            higher_success,
            primary_metric="scenario_success_rate",
        )
        self.assertLess(lower_key, higher_key)

    def test_missing_metric_keys_default_to_zero(self) -> None:
        missing_metrics = {"episode": 1}
        explicit_zero_metrics = {
            "scenario_success_rate": 0.0,
            "episode_success_rate": 0.0,
            "mean_reward": 0.0,
            "episode": 1,
        }
        positive_metric = {
            "scenario_success_rate": 0.0,
            "episode_success_rate": 0.0,
            "mean_reward": 0.1,
            "episode": 1,
        }

        missing_key = checkpoint_candidate_sort_key(
            missing_metrics,
            primary_metric="mean_reward",
        )
        explicit_zero_key = checkpoint_candidate_sort_key(
            explicit_zero_metrics,
            primary_metric="mean_reward",
        )
        positive_key = checkpoint_candidate_sort_key(
            positive_metric,
            primary_metric="mean_reward",
        )
        self.assertEqual(missing_key, explicit_zero_key)
        self.assertLess(missing_key, positive_key)

    def test_checkpoint_metric_changes_selection_priority(self) -> None:
        candidate_a = {
            "scenario_success_rate": 0.9,
            "episode_success_rate": 0.1,
            "mean_reward": 0.2,
            "episode": 1,
        }
        candidate_b = {
            "scenario_success_rate": 0.2,
            "episode_success_rate": 0.8,
            "mean_reward": 0.7,
            "episode": 2,
        }

        scenario_choice = max(
            [candidate_a, candidate_b],
            key=lambda item: checkpoint_candidate_sort_key(
                item,
                primary_metric="scenario_success_rate",
            ),
        )
        reward_choice = max(
            [candidate_a, candidate_b],
            key=lambda item: checkpoint_candidate_sort_key(
                item,
                primary_metric="mean_reward",
            ),
        )

        self.assertIs(scenario_choice, candidate_a)
        self.assertIs(reward_choice, candidate_b)

class CheckpointSelectionConfigValidationTest(unittest.TestCase):
    """Tests for CheckpointSelectionConfig validation in __post_init__."""

    def test_invalid_metric_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            CheckpointSelectionConfig(metric="invalid_metric")

    def test_invalid_penalty_mode_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            CheckpointSelectionConfig(
                metric="scenario_success_rate",
                penalty_mode="bogus_mode",
            )

    def test_infinite_override_weight_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            CheckpointSelectionConfig(
                metric="scenario_success_rate",
                override_penalty_weight=float("inf"),
            )

    def test_negative_override_weight_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            CheckpointSelectionConfig(
                metric="scenario_success_rate",
                override_penalty_weight=-0.1,
            )

    def test_negative_dominance_weight_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            CheckpointSelectionConfig(
                metric="scenario_success_rate",
                dominance_penalty_weight=-1.0,
            )

    def test_nan_dominance_weight_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            CheckpointSelectionConfig(
                metric="scenario_success_rate",
                dominance_penalty_weight=float("nan"),
            )

    def test_all_valid_metrics_accepted(self) -> None:
        for metric in ("scenario_success_rate", "episode_success_rate", "mean_reward"):
            with self.subTest(metric=metric):
                config = CheckpointSelectionConfig(metric=metric)
                self.assertEqual(config.metric, metric)

    def test_to_summary_returns_all_expected_keys(self) -> None:
        config = CheckpointSelectionConfig(
            metric="episode_success_rate",
            override_penalty_weight=0.3,
            dominance_penalty_weight=0.1,
            penalty_mode="tiebreaker",
        )
        summary = config.to_summary()
        self.assertEqual(summary["metric"], "episode_success_rate")
        self.assertAlmostEqual(float(summary["override_penalty_weight"]), 0.3)
        self.assertAlmostEqual(float(summary["dominance_penalty_weight"]), 0.1)
        self.assertEqual(summary["penalty_mode"], "tiebreaker")

    def test_string_penalty_mode_normalized_to_enum(self) -> None:
        config = CheckpointSelectionConfig(
            metric="scenario_success_rate",
            penalty_mode="tiebreaker",
        )
        self.assertEqual(config.penalty_mode, CheckpointPenaltyMode.TIEBREAKER)

    def test_build_summary_rejects_mixed_config_and_legacy_kwargs(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not allow mixing"):
            build_checkpointing_summary(
                selection="best",
                checkpoint_interval=5,
                selection_scenario_episodes=2,
                selection_config=CheckpointSelectionConfig(
                    metric="scenario_success_rate",
                ),
                override_penalty_weight=1.0,
            )

class ResolveCheckpointLoadDirTest(unittest.TestCase):
    """Tests for resolve_checkpoint_load_dir()."""

    def test_none_checkpoint_dir_returns_none(self) -> None:
        result = resolve_checkpoint_load_dir(None, checkpoint_selection="best")
        self.assertIsNone(result)

    def test_checkpoint_selection_none_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = resolve_checkpoint_load_dir(
                tmpdir, checkpoint_selection="none"
            )
        self.assertIsNone(result)

    def test_best_selection_prefers_best_subdir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            best_dir = root / "best"
            best_dir.mkdir()
            (best_dir / "metadata.json").write_text("{}", encoding="utf-8")
            last_dir = root / "last"
            last_dir.mkdir()
            (last_dir / "metadata.json").write_text("{}", encoding="utf-8")

            result = resolve_checkpoint_load_dir(root, checkpoint_selection="best")
            self.assertEqual(result, best_dir)

    def test_last_selection_prefers_last_subdir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            best_dir = root / "best"
            best_dir.mkdir()
            (best_dir / "metadata.json").write_text("{}", encoding="utf-8")
            last_dir = root / "last"
            last_dir.mkdir()
            (last_dir / "metadata.json").write_text("{}", encoding="utf-8")

            result = resolve_checkpoint_load_dir(root, checkpoint_selection="last")
            self.assertEqual(result, last_dir)

    def test_best_selection_falls_back_to_last_when_best_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            last_dir = root / "last"
            last_dir.mkdir()
            (last_dir / "metadata.json").write_text("{}", encoding="utf-8")

            result = resolve_checkpoint_load_dir(root, checkpoint_selection="best")
            self.assertEqual(result, last_dir)

    def test_best_selection_falls_back_to_root_when_both_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "metadata.json").write_text("{}", encoding="utf-8")

            result = resolve_checkpoint_load_dir(root, checkpoint_selection="best")
            self.assertEqual(result, root)

    def test_returns_none_when_no_metadata_found(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            # Root exists as directory but has no metadata.json and no subdirs
            result = resolve_checkpoint_load_dir(root, checkpoint_selection="best")
            self.assertIsNone(result)

    def test_accepts_string_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "metadata.json").write_text("{}", encoding="utf-8")
            result = resolve_checkpoint_load_dir(str(tmpdir), checkpoint_selection="last")
            self.assertEqual(result, root)

class CheckpointCandidateCompositeScoreTest(unittest.TestCase):
    """Tests for checkpoint_candidate_composite_score()."""

    def test_zero_penalty_weights_returns_primary_metric_value(self) -> None:
        candidate = {
            "scenario_success_rate": 0.75,
            "episode_success_rate": 0.5,
            "mean_reward": 1.0,
            "evaluation_summary": {
                "mean_final_reflex_override_rate": 0.5,
                "mean_reflex_dominance": 0.5,
            },
            "episode": 5,
        }
        config = CheckpointSelectionConfig(
            metric="scenario_success_rate",
            override_penalty_weight=0.0,
            dominance_penalty_weight=0.0,
            penalty_mode=CheckpointPenaltyMode.TIEBREAKER,
        )
        score = checkpoint_candidate_composite_score(candidate, config)
        self.assertAlmostEqual(score, 0.75)

    def test_composite_score_penalizes_override_rate(self) -> None:
        candidate = {
            "scenario_success_rate": 0.8,
            "episode_success_rate": 0.5,
            "mean_reward": 1.0,
            "evaluation_summary": {
                "mean_final_reflex_override_rate": 0.4,
                "mean_reflex_dominance": 0.0,
            },
            "episode": 5,
        }
        config = CheckpointSelectionConfig(
            metric="scenario_success_rate",
            override_penalty_weight=0.5,
            dominance_penalty_weight=0.0,
            penalty_mode=CheckpointPenaltyMode.TIEBREAKER,
        )
        # 0.8 - (0.5 * 0.4) = 0.8 - 0.2 = 0.6
        score = checkpoint_candidate_composite_score(candidate, config)
        self.assertAlmostEqual(score, 0.6)

    def test_composite_score_penalizes_dominance(self) -> None:
        candidate = {
            "scenario_success_rate": 0.9,
            "episode_success_rate": 0.5,
            "mean_reward": 1.0,
            "evaluation_summary": {
                "mean_final_reflex_override_rate": 0.0,
                "mean_reflex_dominance": 0.5,
            },
            "episode": 5,
        }
        config = CheckpointSelectionConfig(
            metric="scenario_success_rate",
            override_penalty_weight=0.0,
            dominance_penalty_weight=0.2,
            penalty_mode=CheckpointPenaltyMode.TIEBREAKER,
        )
        # 0.9 - (0.2 * 0.5) = 0.9 - 0.1 = 0.8
        score = checkpoint_candidate_composite_score(candidate, config)
        self.assertAlmostEqual(score, 0.8)

    def test_composite_score_uses_primary_metric_regardless_of_mode(self) -> None:
        candidate = {
            "scenario_success_rate": 0.6,
            "episode_success_rate": 0.9,
            "mean_reward": 1.5,
            "evaluation_summary": {},
            "episode": 1,
        }
        config_tiebreaker = CheckpointSelectionConfig(
            metric="episode_success_rate",
            penalty_mode=CheckpointPenaltyMode.TIEBREAKER,
        )
        config_direct = CheckpointSelectionConfig(
            metric="episode_success_rate",
            penalty_mode=CheckpointPenaltyMode.DIRECT,
        )
        # Both configs should produce composite score based on episode_success_rate
        score_tb = checkpoint_candidate_composite_score(candidate, config_tiebreaker)
        score_direct = checkpoint_candidate_composite_score(candidate, config_direct)
        self.assertAlmostEqual(score_tb, 0.9)
        self.assertAlmostEqual(score_direct, 0.9)

    def test_tiebreaker_mode_is_overridden_for_composite_calculation(self) -> None:
        """composite_score always uses direct penalty mode internally."""
        candidate = {
            "scenario_success_rate": 0.7,
            "episode_success_rate": 0.5,
            "mean_reward": 1.0,
            "evaluation_summary": {
                "mean_final_reflex_override_rate": 0.3,
                "mean_reflex_dominance": 0.1,
            },
            "episode": 1,
        }
        config = CheckpointSelectionConfig(
            metric="scenario_success_rate",
            override_penalty_weight=1.0,
            dominance_penalty_weight=1.0,
            penalty_mode=CheckpointPenaltyMode.TIEBREAKER,
        )
        # 0.7 - (1.0 * 0.3) - (1.0 * 0.1) = 0.7 - 0.3 - 0.1 = 0.3
        score = checkpoint_candidate_composite_score(candidate, config)
        self.assertAlmostEqual(score, 0.3)
