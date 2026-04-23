from __future__ import annotations

import json
import unittest

import numpy as np

from spider_cortex_sim.ablations import BrainAblationConfig
from spider_cortex_sim.agent import SpiderBrain
from spider_cortex_sim.interfaces import (
    ACTION_CONTEXT_INTERFACE,
    MOTOR_CONTEXT_INTERFACE,
)
from spider_cortex_sim.scenarios.trace_core import _trace_action_selection_payloads
from spider_cortex_sim.simulation import SpiderSimulation

from tests.fixtures.action_center import _blank_obs, _module_interface, _vector_for


class FrozenProposerIntegrationTest(unittest.TestCase):
    def _brain_config(self) -> BrainAblationConfig:
        return BrainAblationConfig(
            name="frozen_proposer_integration",
            architecture="modular",
            module_dropout=0.0,
            enable_reflexes=False,
            enable_auxiliary_targets=False,
            use_learned_arbitration=True,
            enable_deterministic_guards=False,
        )

    def _make_brain(self, seed: int = 7) -> SpiderBrain:
        return SpiderBrain(
            seed=seed,
            module_dropout=0.0,
            module_lr=0.02,
            motor_lr=0.03,
            arbitration_lr=0.02,
            config=self._brain_config(),
        )

    def _make_simulation(self, *, seed: int = 7, max_steps: int = 30) -> SpiderSimulation:
        return SpiderSimulation(
            seed=seed,
            max_steps=max_steps,
            module_dropout=0.0,
            brain_config=self._brain_config(),
        )

    def _freeze_all_proposers(self, brain: SpiderBrain) -> list[str]:
        if brain.module_bank is None:
            raise AssertionError("Frozen-proposer tests require the modular architecture.")
        brain.freeze_proposers()
        return brain.frozen_module_names()

    def _make_conflict_observation(self, valence_type: str) -> dict[str, np.ndarray]:
        obs = _blank_obs()
        if valence_type == "threat":
            action_context = {
                "hunger": 0.94,
                "fatigue": 0.16,
                "health": 1.0,
                "recent_pain": 0.72,
                "recent_contact": 0.86,
                "on_food": 0.0,
                "on_shelter": 0.0,
                "predator_visible": 0.97,
                "predator_certainty": 0.93,
                "day": 1.0,
                "night": 0.0,
                "sleep_debt": 0.14,
                "shelter_role_level": 0.0,
            }
            motor_context = {
                "on_food": 0.0,
                "on_shelter": 0.0,
                "predator_visible": 0.97,
                "predator_certainty": 0.93,
                "day": 1.0,
                "night": 0.0,
                "shelter_role_level": 0.0,
                "heading_dx": 0.0,
                "heading_dy": -1.0,
                "terrain_difficulty": 0.08,
                "fatigue": 0.16,
                "momentum": 0.05,
            }
            per_module = {
                "visual_cortex": {
                    "food_visible": 0.9,
                    "food_certainty": 0.84,
                    "predator_visible": 0.97,
                    "predator_certainty": 0.93,
                    "day": 1.0,
                },
                "sensory_cortex": {
                    "hunger": 0.94,
                    "recent_pain": 0.72,
                    "recent_contact": 0.86,
                    "food_smell_strength": 0.79,
                    "predator_smell_strength": 0.85,
                    "light": 1.0,
                },
                "hunger_center": {
                    "hunger": 0.94,
                    "food_visible": 0.9,
                    "food_certainty": 0.84,
                    "food_smell_strength": 0.79,
                    "food_memory_age": 0.08,
                },
                "sleep_center": {
                    "fatigue": 0.16,
                    "sleep_debt": 0.14,
                    "night": 0.0,
                    "on_shelter": 0.0,
                    "shelter_role_level": 0.0,
                    "shelter_trace_strength": 0.0,
                    "shelter_memory_age": 1.0,
                },
                "alert_center": {
                    "predator_visible": 0.97,
                    "predator_certainty": 0.93,
                    "predator_smell_strength": 0.85,
                    "predator_motion_salience": 0.94,
                    "recent_pain": 0.72,
                    "recent_contact": 0.86,
                },
            }
        elif valence_type == "sleep":
            action_context = {
                "hunger": 0.12,
                "fatigue": 0.96,
                "health": 1.0,
                "recent_pain": 0.0,
                "recent_contact": 0.0,
                "on_food": 0.0,
                "on_shelter": 1.0,
                "predator_visible": 0.0,
                "predator_certainty": 0.0,
                "day": 0.0,
                "night": 1.0,
                "sleep_debt": 0.96,
                "shelter_role_level": 0.94,
            }
            motor_context = {
                "on_food": 0.0,
                "on_shelter": 1.0,
                "predator_visible": 0.0,
                "predator_certainty": 0.0,
                "day": 0.0,
                "night": 1.0,
                "shelter_role_level": 0.94,
                "heading_dx": 0.0,
                "heading_dy": 0.0,
                "terrain_difficulty": 0.02,
                "fatigue": 0.96,
                "momentum": 0.0,
            }
            per_module = {
                "visual_cortex": {
                    "food_visible": 0.08,
                    "food_certainty": 0.08,
                    "predator_visible": 0.0,
                    "predator_certainty": 0.0,
                    "night": 1.0,
                },
                "sensory_cortex": {
                    "hunger": 0.12,
                    "fatigue": 0.96,
                    "food_smell_strength": 0.06,
                    "predator_smell_strength": 0.0,
                },
                "hunger_center": {
                    "hunger": 0.12,
                    "food_visible": 0.08,
                    "food_certainty": 0.08,
                    "food_smell_strength": 0.06,
                    "food_memory_age": 0.85,
                },
                "sleep_center": {
                    "fatigue": 0.96,
                    "sleep_debt": 0.96,
                    "night": 1.0,
                    "on_shelter": 1.0,
                    "shelter_role_level": 0.94,
                    "shelter_trace_strength": 0.92,
                    "shelter_memory_age": 0.04,
                },
                "alert_center": {
                    "predator_visible": 0.0,
                    "predator_certainty": 0.0,
                    "predator_smell_strength": 0.0,
                    "predator_motion_salience": 0.0,
                },
            }
        elif valence_type == "hunger":
            action_context = {
                "hunger": 0.99,
                "fatigue": 0.34,
                "health": 1.0,
                "recent_pain": 0.0,
                "recent_contact": 0.0,
                "on_food": 0.0,
                "on_shelter": 0.0,
                "predator_visible": 0.0,
                "predator_certainty": 0.0,
                "day": 1.0,
                "night": 0.0,
                "sleep_debt": 0.28,
                "shelter_role_level": 0.0,
            }
            motor_context = {
                "on_food": 0.0,
                "on_shelter": 0.0,
                "predator_visible": 0.0,
                "predator_certainty": 0.0,
                "day": 1.0,
                "night": 0.0,
                "shelter_role_level": 0.0,
                "heading_dx": 1.0,
                "heading_dy": 0.0,
                "terrain_difficulty": 0.06,
                "fatigue": 0.34,
                "momentum": 0.12,
            }
            per_module = {
                "visual_cortex": {
                    "food_visible": 0.92,
                    "food_certainty": 0.87,
                    "day": 1.0,
                },
                "sensory_cortex": {
                    "hunger": 0.99,
                    "fatigue": 0.34,
                    "food_smell_strength": 0.82,
                    "food_smell_dx": 1.0,
                },
                "hunger_center": {
                    "hunger": 0.99,
                    "food_visible": 0.92,
                    "food_certainty": 0.87,
                    "food_smell_strength": 0.82,
                    "food_memory_age": 0.05,
                },
                "sleep_center": {
                    "fatigue": 0.34,
                    "sleep_debt": 0.28,
                    "night": 0.0,
                    "on_shelter": 0.0,
                    "shelter_role_level": 0.0,
                    "shelter_trace_strength": 0.08,
                    "shelter_memory_age": 0.8,
                },
                "alert_center": {
                    "predator_visible": 0.0,
                    "predator_certainty": 0.0,
                    "predator_smell_strength": 0.0,
                    "predator_motion_salience": 0.0,
                },
            }
        else:
            raise ValueError(f"Unknown conflict observation type: {valence_type}")

        obs[ACTION_CONTEXT_INTERFACE.observation_key] = _vector_for(
            ACTION_CONTEXT_INTERFACE,
            **action_context,
        )
        obs[MOTOR_CONTEXT_INTERFACE.observation_key] = _vector_for(
            MOTOR_CONTEXT_INTERFACE,
            **motor_context,
        )
        for module_name, updates in per_module.items():
            interface = _module_interface(module_name)
            obs[interface.observation_key] = _vector_for(interface, **updates)
        return obs

    def _numeric_state(self, network: object) -> dict[str, np.ndarray]:
        arrays: dict[str, np.ndarray] = {}
        for name, value in network.state_dict().items():
            if isinstance(value, np.ndarray):
                arrays[name] = np.asarray(value, dtype=float).copy()
        return arrays

    def _state_changed(
        self,
        before: dict[str, np.ndarray],
        after: dict[str, np.ndarray],
    ) -> bool:
        if before.keys() != after.keys():
            return True
        return any(
            not np.allclose(before[name], after[name])
            for name in before
        )

    def _train_on_observations(
        self,
        brain: SpiderBrain,
        observations: tuple[dict[str, np.ndarray], ...],
        *,
        steps: int = 6,
    ) -> list[dict[str, object]]:
        stats_by_step: list[dict[str, object]] = []
        for idx in range(steps):
            obs = observations[idx % len(observations)]
            step = brain.act(obs, sample=True)
            stats = brain.learn(
                step,
                reward=1.0,
                next_observation=obs,
                done=True,
            )
            stats_by_step.append(stats)
        return stats_by_step

    def _message_payload(self, item: dict[str, object], sender: str, topic: str) -> dict[str, object] | None:
        messages = item.get("messages", [])
        if not isinstance(messages, list):
            return None
        for message in messages:
            if not isinstance(message, dict):
                continue
            if message.get("sender") != sender or message.get("topic") != topic:
                continue
            payload = message.get("payload")
            if isinstance(payload, dict):
                return payload
        return None

    def _classify_failure_mode(
        self,
        trace: list[dict[str, object]],
        expected_valence: str,
    ) -> str:
        saw_matching_valence = False
        saw_proposer_failure = False
        saw_integration_failure = False
        saw_success = False
        for item in trace:
            selection = self._message_payload(item, "action_center", "action.selection")
            if selection is None:
                continue
            if selection.get("winning_valence") != expected_valence:
                continue
            saw_matching_valence = True
            selected_intent = selection.get("selected_intent")
            if not isinstance(selected_intent, str) or not selected_intent:
                saw_proposer_failure = True
                continue
            execution = self._message_payload(item, "motor_cortex", "action.execution")
            if execution is None:
                saw_integration_failure = True
                continue
            selected_action = execution.get("selected_action")
            executed_action = execution.get("executed_action")
            if selected_action != selected_intent or executed_action != selected_intent:
                saw_integration_failure = True
                continue
            saw_success = True
        if saw_proposer_failure:
            return "proposer_failure"
        if saw_integration_failure:
            return "integration_failure"
        if saw_success:
            return "success"
        return "integration_failure" if not saw_matching_valence else "success"

    def _train_conflict_scenario(
        self,
        scenario_name: str,
        *,
        episodes: int,
    ) -> SpiderSimulation:
        sim = self._make_simulation(seed=7, max_steps=30)
        self._freeze_all_proposers(sim.brain)
        for episode_idx in range(episodes):
            sim.run_episode(
                episode_idx,
                training=True,
                sample=True,
                capture_trace=False,
                scenario_name=scenario_name,
            )
        return sim

    def test_frozen_proposers_skip_weight_updates_but_downstream_networks_change(self) -> None:
        brain = self._make_brain()
        frozen_names = self._freeze_all_proposers(brain)
        threat_obs = self._make_conflict_observation("threat")
        sleep_obs = self._make_conflict_observation("sleep")
        hunger_obs = self._make_conflict_observation("hunger")

        proposer_before = {
            name: self._numeric_state(brain.module_bank.modules[name])
            for name in frozen_names
        }
        action_center_before = self._numeric_state(brain.action_center)
        motor_cortex_before = self._numeric_state(brain.motor_cortex)
        arbitration_before = self._numeric_state(brain.arbitration_network)

        stats_history = self._train_on_observations(
            brain,
            (threat_obs, sleep_obs, hunger_obs),
            steps=8,
        )

        self.assertEqual(brain.frozen_module_names(), frozen_names)
        for name in frozen_names:
            with self.subTest(module=name):
                self.assertFalse(
                    self._state_changed(
                        proposer_before[name],
                        self._numeric_state(brain.module_bank.modules[name]),
                    )
                )
                for step_stats in stats_history:
                    self.assertEqual(step_stats["module_gradient_norms"][name], 0.0)

        self.assertTrue(
            self._state_changed(action_center_before, self._numeric_state(brain.action_center))
        )
        self.assertTrue(
            self._state_changed(motor_cortex_before, self._numeric_state(brain.motor_cortex))
        )
        self.assertTrue(
            self._state_changed(arbitration_before, self._numeric_state(brain.arbitration_network))
        )

    def test_food_vs_predator_conflict_prefers_threat_with_frozen_proposers(self) -> None:
        brain = self._make_brain()
        self._freeze_all_proposers(brain)

        decision = brain.act(self._make_conflict_observation("threat"), sample=False)

        self.assertEqual(decision.arbitration_decision.winning_valence, "threat")
        self.assertLess(
            decision.arbitration_decision.module_gates["hunger_center"],
            0.5,
        )

    def test_sleep_vs_exploration_conflict_prefers_sleep_with_frozen_proposers(self) -> None:
        brain = self._make_brain()
        self._freeze_all_proposers(brain)

        decision = brain.act(self._make_conflict_observation("sleep"), sample=False)

        self.assertEqual(decision.arbitration_decision.winning_valence, "sleep")
        self.assertLess(
            decision.arbitration_decision.module_gates["visual_cortex"],
            0.6,
        )
        self.assertLess(
            decision.arbitration_decision.module_gates["sensory_cortex"],
            0.7,
        )

    def test_food_vs_predator_trace_uses_action_selection_payloads(self) -> None:
        sim = self._train_conflict_scenario("food_vs_predator_conflict", episodes=20)
        _, trace = sim.run_episode(
            999,
            training=False,
            sample=False,
            capture_trace=True,
            scenario_name="food_vs_predator_conflict",
        )
        payloads = _trace_action_selection_payloads(trace)
        failure_mode = self._classify_failure_mode(trace, "threat")

        self.assertGreater(len(payloads), 0, msg=failure_mode)
        for payload in payloads:
            self.assertIn("winning_valence", payload)
            self.assertIn("module_gates", payload)
            self.assertIn("evidence", payload)
        self.assertTrue(
            any(
                payload.get("winning_valence") == "threat"
                and payload.get("dominant_module") in {"alert_center", "threat_center"}
                and float(payload["module_gates"].get("hunger_center", 1.0)) < 0.5
                for payload in payloads
            ),
            msg=failure_mode,
        )

    def test_sleep_vs_exploration_trace_identifies_sleep_dominance(self) -> None:
        sim = self._train_conflict_scenario("sleep_vs_exploration_conflict", episodes=20)
        _, trace = sim.run_episode(
            0,
            training=False,
            sample=False,
            capture_trace=True,
            scenario_name="sleep_vs_exploration_conflict",
        )
        payloads = _trace_action_selection_payloads(trace)
        failure_mode = self._classify_failure_mode(trace, "sleep")

        self.assertGreater(len(payloads), 0, msg=failure_mode)
        self.assertTrue(
            any(
                payload.get("winning_valence") == "sleep"
                and payload.get("dominant_module") == "sleep_center"
                and float(payload["module_gates"].get("visual_cortex", 1.0)) < 0.6
                and float(payload["module_gates"].get("sensory_cortex", 1.0)) < 0.7
                for payload in payloads
            ),
            msg=failure_mode,
        )

    def test_classify_failure_mode_reports_success(self) -> None:
        trace = [
            {
                "messages": [
                    {
                        "sender": "action_center",
                        "topic": "action.selection",
                        "payload": {
                            "winning_valence": "threat",
                            "selected_intent": "MOVE_DOWN",
                            "module_gates": {"hunger_center": 0.2},
                            "evidence": {"threat": {"predator_visible": 1.0}},
                        },
                    },
                    {
                        "sender": "motor_cortex",
                        "topic": "action.execution",
                        "payload": {
                            "selected_action": "MOVE_DOWN",
                            "executed_action": "MOVE_DOWN",
                        },
                    },
                ]
            }
        ]

        self.assertEqual(self._classify_failure_mode(trace, "threat"), "success")

    def test_classify_failure_mode_reports_integration_failure_for_missing_execution(self) -> None:
        trace = [
            {
                "messages": [
                    {
                        "sender": "action_center",
                        "topic": "action.selection",
                        "payload": {
                            "winning_valence": "sleep",
                            "selected_intent": "STAY",
                            "module_gates": {"visual_cortex": 0.4},
                            "evidence": {"sleep": {"sleep_debt": 0.95}},
                        },
                    },
                ]
            }
        ]

        self.assertEqual(
            self._classify_failure_mode(trace, "sleep"),
            "integration_failure",
        )

    def test_classify_failure_mode_prefers_proposer_failure_across_multi_step_trace(self) -> None:
        trace = [
            {
                "messages": [
                    {
                        "sender": "action_center",
                        "topic": "action.selection",
                        "payload": {
                            "winning_valence": "sleep",
                            "selected_intent": "STAY",
                            "module_gates": {"visual_cortex": 0.4},
                            "evidence": {"sleep": {"sleep_debt": 0.95}},
                        },
                    },
                    {
                        "sender": "motor_cortex",
                        "topic": "action.execution",
                        "payload": {
                            "selected_action": "STAY",
                            "executed_action": "STAY",
                        },
                    },
                ]
            },
            {
                "messages": [
                    {
                        "sender": "action_center",
                        "topic": "action.selection",
                        "payload": {
                            "winning_valence": "sleep",
                            "module_gates": {"visual_cortex": 0.4},
                            "evidence": {"sleep": {"sleep_debt": 0.98}},
                        },
                    },
                    {
                        "sender": "motor_cortex",
                        "topic": "action.execution",
                        "payload": {
                            "selected_action": "MOVE_RIGHT",
                            "executed_action": "MOVE_RIGHT",
                        },
                    },
                ]
            },
        ]

        self.assertEqual(
            self._classify_failure_mode(trace, "sleep"),
            "proposer_failure",
        )

    def test_classify_failure_mode_prefers_integration_failure_across_multi_step_trace(self) -> None:
        trace = [
            {
                "messages": [
                    {
                        "sender": "action_center",
                        "topic": "action.selection",
                        "payload": {
                            "winning_valence": "sleep",
                            "selected_intent": "STAY",
                            "module_gates": {"visual_cortex": 0.4},
                            "evidence": {"sleep": {"sleep_debt": 0.95}},
                        },
                    },
                    {
                        "sender": "motor_cortex",
                        "topic": "action.execution",
                        "payload": {
                            "selected_action": "STAY",
                            "executed_action": "STAY",
                        },
                    },
                ]
            },
            {
                "messages": [
                    {
                        "sender": "action_center",
                        "topic": "action.selection",
                        "payload": {
                            "winning_valence": "sleep",
                            "selected_intent": "STAY",
                            "module_gates": {"visual_cortex": 0.4},
                            "evidence": {"sleep": {"sleep_debt": 0.98}},
                        },
                    },
                ]
            },
        ]

        self.assertEqual(
            self._classify_failure_mode(trace, "sleep"),
            "integration_failure",
        )

    def test_classify_failure_mode_reports_proposer_failure_for_missing_selected_intent(self) -> None:
        trace = [
            {
                "messages": [
                    {
                        "sender": "action_center",
                        "topic": "action.selection",
                        "payload": {
                            "winning_valence": "sleep",
                            "module_gates": {"visual_cortex": 0.4},
                            "evidence": {"sleep": {"sleep_debt": 0.95}},
                        },
                    },
                    {
                        "sender": "motor_cortex",
                        "topic": "action.execution",
                        "payload": {
                            "selected_action": "MOVE_RIGHT",
                            "executed_action": "MOVE_RIGHT",
                        },
                    },
                ]
            }
        ]

        self.assertEqual(
            self._classify_failure_mode(trace, "sleep"),
            "proposer_failure",
        )

    def test_classify_failure_mode_reports_integration_failure_for_mismatched_execution(self) -> None:
        trace = [
            {
                "messages": [
                    {
                        "sender": "action_center",
                        "topic": "action.selection",
                        "payload": {
                            "winning_valence": "sleep",
                            "selected_intent": "STAY",
                            "module_gates": {"visual_cortex": 0.4},
                            "evidence": {"sleep": {"sleep_debt": 0.95}},
                        },
                    },
                    {
                        "sender": "motor_cortex",
                        "topic": "action.execution",
                        "payload": {
                            "selected_action": "MOVE_RIGHT",
                            "executed_action": "MOVE_RIGHT",
                        },
                    },
                ]
            }
        ]

        self.assertEqual(
            self._classify_failure_mode(trace, "sleep"),
            "integration_failure",
        )

    def test_classify_failure_mode_reports_integration_failure(self) -> None:
        trace = [
            {
                "messages": [
                    {
                        "sender": "action_center",
                        "topic": "action.selection",
                        "payload": {
                            "winning_valence": "exploration",
                            "selected_intent": "MOVE_RIGHT",
                            "module_gates": {"visual_cortex": 0.9},
                            "evidence": {"sleep": {"sleep_debt": 0.95}},
                        },
                    }
                ]
            }
        ]

        self.assertEqual(
            self._classify_failure_mode(trace, "sleep"),
            "integration_failure",
        )

    def test_summary_records_frozen_modules_and_is_json_serializable(self) -> None:
        sim = self._make_simulation(seed=7, max_steps=16)
        frozen_names = self._freeze_all_proposers(sim.brain)
        training_stats, _ = sim.run_episode(
            0,
            training=True,
            sample=True,
            capture_trace=False,
            scenario_name="food_vs_predator_conflict",
        )

        summary = sim._build_summary([training_stats], [])

        self.assertEqual(summary["frozen_modules"], frozen_names)
        self.assertEqual(summary["config"]["frozen_modules"], frozen_names)
        dumped = json.dumps(summary)
        self.assertIsInstance(dumped, str)
        self.assertGreater(len(dumped), 0)
        self.assertIsInstance(json.loads(dumped), dict)


if __name__ == "__main__":
    unittest.main()
