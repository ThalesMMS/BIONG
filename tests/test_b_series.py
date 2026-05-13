import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from spider_cortex_sim.ablations import BrainAblationConfig, resolve_ablation_configs
from spider_cortex_sim.agent import SpiderBrain
from spider_cortex_sim.b_series import (
    B_CURRENT_BRIDGE_EFFECTIVE_LEVEL,
    B_CURRENT_BRIDGE_SELECTION_SOURCE,
    B_SEMANTIC_ACTIONS,
    B_SEMANTIC_ACTION_TO_INDEX,
    bridge_b_semantic_action,
)
from spider_cortex_sim.b_series_legacy import (
    LEGACY_B0_ACTIONS,
    LegacyB0Simulation,
)
from spider_cortex_sim.interfaces import (
    ACTION_CONTEXT_INTERFACE,
    ACTION_TO_INDEX,
    MODULE_INTERFACE_BY_NAME,
    MODULE_INTERFACES,
    MOTOR_CONTEXT_INTERFACE,
)
from spider_cortex_sim.world import ACTIONS


def _bridge_observation() -> dict[str, object]:
    return {
        "meta": {
            "on_food": False,
            "on_shelter": False,
            "local_affordances": {
                "MOVE_UP": {"blocked": False},
                "MOVE_DOWN": {"blocked": False},
                "MOVE_LEFT": {"blocked": False},
                "MOVE_RIGHT": {"blocked": False},
            },
            "local_transition_consequences": {
                "MOVE_UP": {
                    "food_dist_delta": -1.0,
                    "shelter_dist_delta": -1.0,
                },
                "MOVE_DOWN": {
                    "food_dist_delta": 0.0,
                    "shelter_dist_delta": 0.0,
                },
                "MOVE_LEFT": {
                    "food_dist_delta": 0.25,
                    "shelter_dist_delta": 1.5,
                },
                "MOVE_RIGHT": {
                    "food_dist_delta": 1.0,
                    "shelter_dist_delta": 0.2,
                },
            },
            "local_geodesic_consequences": {
                "MOVE_LEFT": {
                    "exit_geodesic_delta": 1.0,
                    "deep_geodesic_delta": 1.0,
                },
                "MOVE_RIGHT": {
                    "exit_geodesic_delta": 0.0,
                    "deep_geodesic_delta": 0.0,
                },
            },
        }
    }


def _set_module_values(
    observation: dict[str, object],
    module_name: str,
    values: dict[str, float],
) -> None:
    spec = MODULE_INTERFACE_BY_NAME[module_name]
    vector = np.asarray(observation[spec.observation_key], dtype=float).copy()
    signal_to_index = {name: idx for idx, name in enumerate(spec.signal_names)}
    for name, value in values.items():
        vector[signal_to_index[name]] = float(value)
    observation[spec.observation_key] = vector


def _brain_observation(
    meta: dict[str, object] | None = None,
    *,
    hunger: dict[str, float] | None = None,
    sleep: dict[str, float] | None = None,
    threat: dict[str, float] | None = None,
) -> dict[str, object]:
    observation: dict[str, object] = {
        spec.observation_key: np.zeros(spec.input_dim, dtype=float)
        for spec in MODULE_INTERFACES
    }
    observation[ACTION_CONTEXT_INTERFACE.observation_key] = np.zeros(
        ACTION_CONTEXT_INTERFACE.input_dim,
        dtype=float,
    )
    observation[MOTOR_CONTEXT_INTERFACE.observation_key] = np.zeros(
        MOTOR_CONTEXT_INTERFACE.input_dim,
        dtype=float,
    )
    observation["meta"] = meta or _bridge_observation()["meta"]
    if hunger:
        _set_module_values(observation, "hunger_center", hunger)
    if sleep:
        _set_module_values(observation, "sleep_center", sleep)
    if threat:
        _set_module_values(observation, "threat_center", threat)
    return observation


def _b0_config(name: str = "b0_current_bridge_policy") -> BrainAblationConfig:
    return resolve_ablation_configs([name])[0]


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

    def test_legacy_variant_is_not_routed_through_current_world_brain(self) -> None:
        config = _b0_config("b0_legacy_semantic_policy")
        with self.assertRaisesRegex(ValueError, "LegacyB0Simulation"):
            SpiderBrain(seed=1, module_dropout=0.0, config=config)


class BSeriesBridgeTest(unittest.TestCase):
    def test_move_to_food_selects_primitive_food_progress(self) -> None:
        decision = bridge_b_semantic_action("MOVE_TO_FOOD", _bridge_observation())
        self.assertEqual(decision.primitive_action, "MOVE_RIGHT")
        self.assertEqual(decision.reason, "food_progress")
        self.assertAlmostEqual(decision.food_delta_used, 1.0)

    def test_move_to_food_inside_shelter_first_favors_exit_geodesic(self) -> None:
        observation = _bridge_observation()
        meta = observation["meta"]
        assert isinstance(meta, dict)
        meta["on_shelter"] = True
        meta["shelter_role"] = "deep"
        meta["shelter_role_level"] = 1.0
        geodesics = meta["local_geodesic_consequences"]
        transitions = meta["local_transition_consequences"]
        assert isinstance(geodesics, dict)
        assert isinstance(transitions, dict)
        geodesics["MOVE_UP"] = {
            "exit_geodesic_delta": 1.0,
            "deep_geodesic_delta": -1.0,
            "next_on_exit_target": False,
            "next_on_deep_target": False,
        }
        transitions["MOVE_UP"] = {
            "food_dist_delta": -1.0,
            "shelter_dist_delta": 0.0,
            "predator_dist_delta": 1.0,
            "next_cell_has_food": False,
        }
        decision = bridge_b_semantic_action("MOVE_TO_FOOD", observation)
        self.assertEqual(decision.primitive_action, "MOVE_UP")
        self.assertEqual(decision.reason, "food_exit_to_outside")

    def test_move_to_food_uses_memory_vector_after_shelter_exit(self) -> None:
        observation = _bridge_observation()
        meta = observation["meta"]
        assert isinstance(meta, dict)
        meta["shelter_role"] = "outside"
        meta["memory_vectors"] = {
            "food": {"dx": 0.60, "dy": 0.10, "age": 0.0, "ttl": 30},
        }
        affordances = meta["local_affordances"]
        transitions = meta["local_transition_consequences"]
        assert isinstance(affordances, dict)
        assert isinstance(transitions, dict)
        affordances["MOVE_DOWN"] = {"blocked": False, "next_role": "entrance"}
        transitions["MOVE_DOWN"] = {
            "food_dist_delta": 1.0,
            "shelter_dist_delta": 1.0,
            "predator_dist_delta": 0.0,
            "next_cell_has_food": False,
        }
        transitions["MOVE_RIGHT"] = {
            "food_dist_delta": 0.0,
            "shelter_dist_delta": -1.0,
            "predator_dist_delta": 0.0,
            "next_cell_has_food": False,
        }

        decision = bridge_b_semantic_action("MOVE_TO_FOOD", observation)

        self.assertEqual(decision.primitive_action, "MOVE_RIGHT")
        self.assertEqual(decision.reason, "food_memory_vector")

    def test_move_to_food_does_not_reenter_shelter_without_food(self) -> None:
        observation = _bridge_observation()
        meta = observation["meta"]
        assert isinstance(meta, dict)
        meta["shelter_role"] = "outside"
        affordances = meta["local_affordances"]
        transitions = meta["local_transition_consequences"]
        assert isinstance(affordances, dict)
        assert isinstance(transitions, dict)
        affordances["MOVE_DOWN"] = {"blocked": False, "next_role": "entrance"}
        transitions["MOVE_DOWN"] = {
            "food_dist_delta": 1.0,
            "shelter_dist_delta": 1.0,
            "predator_dist_delta": 0.0,
            "next_cell_has_food": False,
        }
        transitions["MOVE_RIGHT"] = {
            "food_dist_delta": 0.25,
            "shelter_dist_delta": -1.0,
            "predator_dist_delta": 0.0,
            "next_cell_has_food": False,
        }
        transitions["MOVE_LEFT"] = {
            "food_dist_delta": 0.0,
            "shelter_dist_delta": 0.0,
            "predator_dist_delta": 0.0,
            "next_cell_has_food": False,
        }

        decision = bridge_b_semantic_action("MOVE_TO_FOOD", observation)

        self.assertEqual(decision.primitive_action, "MOVE_RIGHT")

    def test_move_to_shelter_selects_primitive_shelter_progress(self) -> None:
        decision = bridge_b_semantic_action("MOVE_TO_SHELTER", _bridge_observation())
        self.assertEqual(decision.primitive_action, "MOVE_LEFT")
        self.assertEqual(decision.reason, "shelter_progress")
        self.assertAlmostEqual(decision.shelter_delta_used, 1.5)

    def test_move_to_shelter_holds_when_already_deep(self) -> None:
        observation = _bridge_observation()
        meta = observation["meta"]
        assert isinstance(meta, dict)
        meta["on_shelter"] = True
        meta["shelter_role"] = "deep"
        meta["shelter_role_level"] = 1.0
        decision = bridge_b_semantic_action("MOVE_TO_SHELTER", observation)
        self.assertEqual(decision.primitive_action, "STAY")
        self.assertEqual(decision.reason, "already_deep_shelter")

    def test_blocked_winning_move_is_masked_without_semantic_rewrite(self) -> None:
        observation = _bridge_observation()
        meta = observation["meta"]
        assert isinstance(meta, dict)
        affordances = meta["local_affordances"]
        assert isinstance(affordances, dict)
        affordances["MOVE_RIGHT"] = {"blocked": True}
        decision = bridge_b_semantic_action("MOVE_TO_FOOD", observation)
        self.assertEqual(decision.semantic_action, "MOVE_TO_FOOD")
        self.assertEqual(decision.primitive_action, "MOVE_LEFT")
        self.assertTrue(decision.blocked_mask["MOVE_RIGHT"])
        self.assertEqual(decision.external_override_count, 0)

    def test_non_movement_semantic_actions_map_to_stay(self) -> None:
        for semantic_action in ("STAY", "EAT", "SLEEP"):
            with self.subTest(semantic_action=semantic_action):
                decision = bridge_b_semantic_action(
                    semantic_action,
                    _bridge_observation(),
                )
                self.assertEqual(decision.primitive_action, "STAY")

    def test_explore_selects_unblocked_primitive_move(self) -> None:
        decision = bridge_b_semantic_action("EXPLORE", _bridge_observation())
        self.assertIn(decision.primitive_action, {"MOVE_UP", "MOVE_DOWN", "MOVE_LEFT", "MOVE_RIGHT"})


class BSeriesRuntimeTest(unittest.TestCase):
    def test_b0_current_bridge_never_emits_semantic_action_to_world(self) -> None:
        brain = SpiderBrain(seed=11, module_dropout=0.0, config=_b0_config())
        assert brain.b_series_policy is not None
        brain.b_series_policy.b2_policy[:] = -10.0
        brain.b_series_policy.b2_policy[
            B_SEMANTIC_ACTION_TO_INDEX["SLEEP"]
        ] = 10.0

        decision = brain.act_inference(
            _brain_observation(
                hunger={
                    "hunger": 0.82,
                    "food_visible": 1.0,
                    "food_certainty": 1.0,
                },
                sleep={"health": 1.0},
            ),
            sample=False,
        )

        self.assertEqual(decision.learned_semantic_action, "SLEEP")
        self.assertEqual(decision.semantic_action, "MOVE_TO_FOOD")
        self.assertEqual(decision.b_effective_level, B_CURRENT_BRIDGE_EFFECTIVE_LEVEL)
        self.assertEqual(
            decision.semantic_action_source,
            B_CURRENT_BRIDGE_SELECTION_SOURCE,
        )
        self.assertEqual(decision.semantic_override_count, 1)
        self.assertEqual(decision.bridge_primitive_action, "MOVE_RIGHT")
        self.assertEqual(ACTIONS[decision.action_idx], "MOVE_RIGHT")
        self.assertNotIn(decision.semantic_action, ACTIONS)
        self.assertEqual(decision.external_override_count, 0)

    def test_b0_current_rest_phase_maps_to_stay_without_final_bias(self) -> None:
        brain = SpiderBrain(seed=12, module_dropout=0.0, config=_b0_config())
        assert brain.b_series_policy is not None
        brain.b_series_policy.b2_policy[:] = -10.0
        brain.b_series_policy.b2_policy[
            B_SEMANTIC_ACTION_TO_INDEX["MOVE_TO_FOOD"]
        ] = 10.0

        meta = _bridge_observation()["meta"]
        assert isinstance(meta, dict)
        meta["on_shelter"] = True
        meta["shelter_role"] = "deep"
        meta["shelter_role_level"] = 1.0
        geodesics = meta["local_geodesic_consequences"]
        transitions = meta["local_transition_consequences"]
        assert isinstance(geodesics, dict)
        assert isinstance(transitions, dict)
        geodesics["MOVE_UP"] = {
            "exit_geodesic_delta": 1.0,
            "deep_geodesic_delta": -1.0,
            "next_on_exit_target": False,
            "next_on_deep_target": False,
        }
        geodesics["MOVE_LEFT"] = {
            "exit_geodesic_delta": 0.0,
            "deep_geodesic_delta": 0.0,
            "next_on_exit_target": False,
            "next_on_deep_target": True,
        }
        transitions["MOVE_UP"] = {
            "food_dist_delta": -1.0,
            "shelter_dist_delta": 0.0,
            "predator_dist_delta": 1.0,
            "next_cell_has_food": False,
        }
        decision = brain.act_inference(
            _brain_observation(
                meta,
                sleep={
                    "fatigue": 0.90,
                    "hunger": 0.12,
                    "on_shelter": 1.0,
                    "night": 1.0,
                    "health": 1.0,
                    "sleep_debt": 0.90,
                    "shelter_role_level": 1.0,
                },
            ),
            sample=False,
        )

        self.assertEqual(decision.learned_semantic_action, "MOVE_TO_FOOD")
        self.assertEqual(decision.semantic_action, "SLEEP")
        self.assertEqual(decision.bridge_primitive_action, "STAY")
        self.assertEqual(ACTIONS[decision.action_idx], "STAY")
        self.assertEqual(decision.semantic_override_count, 1)

    def test_b0_current_critical_hunger_can_leave_deep_shelter_under_residual_smell(self) -> None:
        brain = SpiderBrain(seed=13, module_dropout=0.0, config=_b0_config())
        assert brain.b_series_policy is not None
        brain.b_series_policy.b2_policy[:] = -10.0
        brain.b_series_policy.b2_policy[B_SEMANTIC_ACTION_TO_INDEX["STAY"]] = 10.0

        meta = _bridge_observation()["meta"]
        assert isinstance(meta, dict)
        meta["on_shelter"] = True
        meta["shelter_role"] = "deep"
        meta["shelter_role_level"] = 1.0
        geodesics = meta["local_geodesic_consequences"]
        transitions = meta["local_transition_consequences"]
        assert isinstance(geodesics, dict)
        assert isinstance(transitions, dict)
        geodesics["MOVE_UP"] = {
            "exit_geodesic_delta": 1.0,
            "deep_geodesic_delta": -1.0,
            "next_on_exit_target": False,
            "next_on_deep_target": False,
        }
        geodesics["MOVE_LEFT"] = {
            "exit_geodesic_delta": 0.0,
            "deep_geodesic_delta": 0.0,
            "next_on_exit_target": False,
            "next_on_deep_target": True,
        }
        transitions["MOVE_UP"] = {
            "food_dist_delta": -1.0,
            "shelter_dist_delta": 0.0,
            "predator_dist_delta": 1.0,
            "next_cell_has_food": False,
        }
        decision = brain.act_inference(
            _brain_observation(
                meta,
                hunger={"hunger": 0.92},
                sleep={
                    "fatigue": 0.12,
                    "hunger": 0.92,
                    "on_shelter": 1.0,
                    "health": 1.0,
                    "shelter_role_level": 1.0,
                },
                threat={"predator_smell_strength": 0.70},
            ),
            sample=False,
        )

        self.assertEqual(decision.learned_semantic_action, "STAY")
        self.assertEqual(decision.semantic_action, "MOVE_TO_FOOD")
        self.assertEqual(decision.bridge_primitive_action, "MOVE_UP")
        self.assertEqual(ACTIONS[decision.action_idx], "MOVE_UP")


class BSeriesCheckpointTest(unittest.TestCase):
    def test_checkpoint_save_load_preserves_b_series_metadata_and_weights(self) -> None:
        config = _b0_config()
        source = SpiderBrain(seed=21, module_dropout=0.0, config=config)
        target = SpiderBrain(seed=22, module_dropout=0.0, config=config)
        assert source.b_series_policy is not None
        assert target.b_series_policy is not None
        source.b_series_policy.b2_policy[:] = np.arange(
            len(B_SEMANTIC_ACTIONS),
            dtype=float,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = source.save(Path(tmpdir) / "b0")
            loaded = target.load(checkpoint)
            metadata = json.loads((checkpoint / "metadata.json").read_text())

        self.assertIn("b_series_policy", loaded)
        self.assertEqual(
            metadata["modules"]["b_series_policy"]["semantic_actions"],
            list(B_SEMANTIC_ACTIONS),
        )
        self.assertEqual(metadata["modules"]["b_series_policy"]["b_level"], 0)
        self.assertEqual(metadata["modules"]["b_series_policy"]["b_mode"], "current_bridge")
        np.testing.assert_allclose(
            source.b_series_policy.b2_policy,
            target.b_series_policy.b2_policy,
        )

    def test_b1_partial_transfer_reports_coverage(self) -> None:
        source_config = _b0_config()
        source = SpiderBrain(seed=31, module_dropout=0.0, config=source_config)
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = source.save(Path(tmpdir) / "b0")
            b1_config = BrainAblationConfig(
                name="b1_transfer_smoke",
                architecture="b_series",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                use_learned_arbitration=False,
                enable_food_direction_bias=False,
                warm_start_scale=0.0,
                credit_strategy="broadcast",
                disabled_modules=(),
                reflex_scale=0.0,
                module_reflex_scales={},
                b_level=1,
                b_mode="current_bridge",
                b_parent_level=0,
                b_hidden_dim=40,
                b_transfer_source_checkpoint=str(checkpoint),
            )
            target = SpiderBrain(seed=32, module_dropout=0.0, config=b1_config)

        report = target.b_series_transfer_report
        self.assertIsNotNone(report)
        assert report is not None
        self.assertGreaterEqual(float(report["coverage"]), 0.50)
        self.assertEqual(report["target_b_level"], 1)
        self.assertEqual(report["parent_level"], 0)
        self.assertIn("W1", report["partially_loaded_keys"])


class BSeriesLegacyHarnessTest(unittest.TestCase):
    def test_legacy_harness_runs_short_training_smoke(self) -> None:
        sim = LegacyB0Simulation(max_steps=20, seed=5)
        summary, trace = sim.train(
            2,
            evaluation_episodes=1,
            capture_evaluation_trace=True,
        )

        self.assertEqual(summary["b_level"], 0)
        self.assertEqual(summary["b_mode"], "legacy_semantic")
        self.assertEqual(summary["semantic_actions"], list(B_SEMANTIC_ACTIONS))
        self.assertEqual(summary["evaluation"]["episodes"], 1)
        self.assertGreater(len(trace), 0)


if __name__ == "__main__":
    unittest.main()
