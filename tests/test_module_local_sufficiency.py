from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from spider_cortex_sim.interfaces import ACTION_TO_INDEX, get_interface_variant
from spider_cortex_sim.module_local_tasks import (
    ALERT_CENTER_TASKS,
    ALL_LOCAL_TASKS,
    HUNGER_CENTER_TASKS,
    SENSORY_CORTEX_TASKS,
    SLEEP_CENTER_TASKS,
    VISUAL_CORTEX_TASKS,
    ModuleVariantComparison,
    VariantSufficiencyResult,
    _favor_margin,
    build_task_observation,
    build_variant_sufficiency_report,
    check_interface_sufficiency,
    evaluate_variant_sufficiency,
    evaluate_local_task,
    make_local_task_network,
)
from spider_cortex_sim.modules import MODULE_HIDDEN_DIMS, VARIANT_HIDDEN_DIMS, get_variant_hidden_dim
from spider_cortex_sim.nn import softmax
from tests.fixtures.local_tasks import (
    _assert_action_favored,
    _create_fresh_network,
    _get_interface_for_module,
)


def _expected_indices(task) -> tuple[int, ...]:
    return tuple(ACTION_TO_INDEX[action_name] for action_name in task.expected_favored_actions)


def _target_probs(task, action_dim: int) -> np.ndarray:
    target = np.zeros(action_dim, dtype=float)
    expected_indices = _expected_indices(task)
    target[list(expected_indices)] = 1.0 / float(len(expected_indices))
    return target


class ModuleLocalSufficiencyTest(unittest.TestCase):
    def test_interface_signals_exist_for_all_tasks(self) -> None:
        for module_name, tasks in ALL_LOCAL_TASKS.items():
            interface = _get_interface_for_module(module_name)
            for task in tasks:
                is_sufficient, missing = check_interface_sufficiency(task, interface)
                self.assertTrue(is_sufficient, f"{module_name}:{task.name} missing {missing}")
                self.assertEqual(missing, [])

    def _assert_module_tasks_are_evaluable(self, tasks: tuple, seed_offset: int) -> None:
        for index, task in enumerate(tasks):
            interface = _get_interface_for_module(task.module_name)
            network = _create_fresh_network(task.module_name, seed=seed_offset + index)
            result = evaluate_local_task(task, network, interface, max_steps=250, lr=0.05)
            self.assertNotEqual(
                result.status,
                "interface_insufficient",
                f"{task.module_name}:{task.name} should expose all required signals.",
            )
            if result.status == "passed" and len(task.expected_favored_actions) == 1:
                _assert_action_favored(self, result.final_logits, task.expected_favored_actions, margin=0.0)

    def test_hunger_center_local_tasks(self) -> None:
        self._assert_module_tasks_are_evaluable(HUNGER_CENTER_TASKS, seed_offset=100)

    def test_sleep_center_local_tasks(self) -> None:
        self._assert_module_tasks_are_evaluable(SLEEP_CENTER_TASKS, seed_offset=200)

    def test_alert_center_local_tasks(self) -> None:
        self._assert_module_tasks_are_evaluable(ALERT_CENTER_TASKS, seed_offset=300)

    def test_visual_cortex_local_tasks(self) -> None:
        self._assert_module_tasks_are_evaluable(VISUAL_CORTEX_TASKS, seed_offset=400)

    def test_sensory_cortex_local_tasks(self) -> None:
        self._assert_module_tasks_are_evaluable(SENSORY_CORTEX_TASKS, seed_offset=500)


class DirectionCompatibilityTest(unittest.TestCase):
    def test_logits_are_finite_for_all_tasks(self) -> None:
        for module_name, tasks in ALL_LOCAL_TASKS.items():
            interface = _get_interface_for_module(module_name)
            for index, task in enumerate(tasks):
                observation = build_task_observation(task, interface)
                network = _create_fresh_network(module_name, seed=1000 + index)
                logits = network.forward(observation, store_cache=False)
                self.assertTrue(np.all(np.isfinite(logits)), f"{module_name}:{task.name} produced non-finite logits")

    def test_output_shape_matches_action_space(self) -> None:
        for module_name, tasks in ALL_LOCAL_TASKS.items():
            interface = _get_interface_for_module(module_name)
            for index, task in enumerate(tasks):
                observation = build_task_observation(task, interface)
                network = _create_fresh_network(module_name, seed=1100 + index)
                logits = network.forward(observation, store_cache=False)
                self.assertEqual(logits.shape, (len(ACTION_TO_INDEX),))

    def test_softmax_produces_valid_probabilities(self) -> None:
        for module_name, tasks in ALL_LOCAL_TASKS.items():
            interface = _get_interface_for_module(module_name)
            for index, task in enumerate(tasks):
                observation = build_task_observation(task, interface)
                network = _create_fresh_network(module_name, seed=1200 + index)
                logits = network.forward(observation, store_cache=False)
                probs = softmax(logits)
                self.assertTrue(np.all(probs >= 0.0))
                self.assertAlmostEqual(float(np.sum(probs)), 1.0, places=7)


class LocalTaskLearningTest(unittest.TestCase):
    def test_training_changes_logits(self) -> None:
        task = HUNGER_CENTER_TASKS[0]
        interface = _get_interface_for_module(task.module_name)
        observation = build_task_observation(task, interface)
        network = _create_fresh_network(task.module_name, seed=1300)
        before = network.forward(observation, store_cache=True).copy()
        probs = softmax(before)
        network.backward(probs - _target_probs(task, before.shape[0]), lr=0.05)
        after = network.forward(observation, store_cache=False)
        self.assertFalse(np.allclose(before, after))

    def test_multiple_training_steps_converge(self) -> None:
        task = SLEEP_CENTER_TASKS[1]
        interface = _get_interface_for_module(task.module_name)
        observation = build_task_observation(task, interface)
        network = _create_fresh_network(task.module_name, seed=1400)
        expected_indices = _expected_indices(task)

        initial_logits = network.forward(observation, store_cache=False).copy()
        initial_margin = _favor_margin(initial_logits, expected_indices)
        for _ in range(120):
            logits = network.forward(observation, store_cache=True)
            probs = softmax(logits)
            network.backward(probs - _target_probs(task, logits.shape[0]), lr=0.05)
        final_logits = network.forward(observation, store_cache=False)
        final_margin = _favor_margin(final_logits, expected_indices)

        self.assertGreater(final_margin, initial_margin)
        _assert_action_favored(self, final_logits, task.expected_favored_actions, margin=0.5)

    def test_gradient_respects_expected_direction(self) -> None:
        task = SLEEP_CENTER_TASKS[1]
        logits = np.zeros(len(ACTION_TO_INDEX), dtype=float)
        probs = softmax(logits)
        grad = probs - _target_probs(task, logits.shape[0])
        updated_logits = logits - 0.1 * grad

        expected_indices = _expected_indices(task)
        unexpected_indices = [idx for idx in range(logits.shape[0]) if idx not in expected_indices]
        for expected_idx in expected_indices:
            self.assertLess(grad[expected_idx], 0.0)
            self.assertGreater(updated_logits[expected_idx], logits[expected_idx])
        for unexpected_idx in unexpected_indices:
            self.assertGreater(grad[unexpected_idx], 0.0)
            self.assertLess(updated_logits[unexpected_idx], logits[unexpected_idx])


class VariantHiddenDimTest(unittest.TestCase):
    def test_variant_hidden_dims_scale_down_with_interface_size(self) -> None:
        expected_dims = {
            ("visual_cortex", 1): 5,
            ("visual_cortex", 2): 11,
            ("visual_cortex", 3): 26,
            ("visual_cortex", 4): 32,
            ("sensory_cortex", 1): 12,
            ("sensory_cortex", 2): 19,
            ("sensory_cortex", 3): 26,
            ("sensory_cortex", 4): 28,
            ("hunger_center", 1): 8,
            ("hunger_center", 2): 12,
            ("hunger_center", 3): 19,
            ("hunger_center", 4): 26,
            ("sleep_center", 1): 4,
            ("sleep_center", 2): 6,
            ("sleep_center", 3): 12,
            ("sleep_center", 4): 24,
            ("alert_center", 1): 4,
            ("alert_center", 2): 6,
            ("alert_center", 3): 10,
            ("alert_center", 4): 28,
        }
        self.assertEqual(VARIANT_HIDDEN_DIMS, expected_dims)

    def test_get_variant_hidden_dim_matches_registry_and_canonical_level(self) -> None:
        for (module_name, level), hidden_dim in VARIANT_HIDDEN_DIMS.items():
            self.assertEqual(get_variant_hidden_dim(module_name, level), hidden_dim)
            self.assertLessEqual(hidden_dim, MODULE_HIDDEN_DIMS[module_name])
        self.assertEqual(get_variant_hidden_dim("sleep_center", 4), MODULE_HIDDEN_DIMS["sleep_center"])
        self.assertEqual(get_variant_hidden_dim("visual_cortex", 4), MODULE_HIDDEN_DIMS["visual_cortex"])
        self.assertEqual(get_variant_hidden_dim("sensory_cortex", 4), MODULE_HIDDEN_DIMS["sensory_cortex"])


class VariantSufficiencyWorkflowTest(unittest.TestCase):
    def test_build_variant_sufficiency_report_identifies_minimal_level(self) -> None:
        results = [
            VariantSufficiencyResult(
                module_name="hunger_center",
                level=1,
                interface_name="hunger_v1",
                input_dim=5,
                tasks_attempted=0,
                tasks_passed=0,
                tasks_insufficient=2,
                task_details=tuple(),
            ),
            VariantSufficiencyResult(
                module_name="hunger_center",
                level=2,
                interface_name="hunger_v2",
                input_dim=8,
                tasks_attempted=1,
                tasks_passed=1,
                tasks_insufficient=1,
                task_details=tuple(),
            ),
            VariantSufficiencyResult(
                module_name="hunger_center",
                level=3,
                interface_name="hunger_v3",
                input_dim=13,
                tasks_attempted=2,
                tasks_passed=2,
                tasks_insufficient=0,
                task_details=tuple(),
            ),
            VariantSufficiencyResult(
                module_name="hunger_center",
                level=4,
                interface_name="hunger_center",
                input_dim=18,
                tasks_attempted=2,
                tasks_passed=2,
                tasks_insufficient=0,
                task_details=tuple(),
            ),
        ]

        report = build_variant_sufficiency_report("hunger_center", results)

        self.assertEqual(report["module_name"], "hunger_center")
        self.assertEqual(report["task_count"], len(HUNGER_CENTER_TASKS))
        self.assertEqual(report["minimal_sufficient_level"], 3)
        self.assertEqual(report["minimal_sufficient_interface"], "hunger_v3")
        self.assertEqual(report["levels_evaluated"], [1, 2, 3, 4])
        self.assertAlmostEqual(report["levels"][2]["pass_rate"], 1.0)
        self.assertAlmostEqual(report["levels"][2]["coverage_rate"], 1.0)

    def test_evaluate_variant_sufficiency_returns_structured_comparison(self) -> None:
        comparison = evaluate_variant_sufficiency(
            "hunger_center",
            rng=np.random.default_rng(123),
        )

        self.assertIsInstance(comparison, ModuleVariantComparison)
        self.assertEqual(comparison.module_name, "hunger_center")
        self.assertEqual([result.level for result in comparison.results], [1, 2, 3, 4])

        for result in comparison.results:
            interface = get_interface_variant("hunger_center", result.level)
            self.assertEqual(result.interface_name, interface.name)
            self.assertEqual(result.input_dim, interface.input_dim)
            self.assertEqual(
                result.tasks_attempted + result.tasks_insufficient,
                len(HUNGER_CENTER_TASKS),
            )
            self.assertEqual(len(result.task_details), len(HUNGER_CENTER_TASKS))

        insufficient_counts = {
            result.level: result.tasks_insufficient
            for result in comparison.results
        }
        self.assertGreater(insufficient_counts[1], 0)
        self.assertGreater(insufficient_counts[2], 0)
        self.assertGreater(insufficient_counts[3], 0)
        self.assertEqual(insufficient_counts[4], 0)
        self.assertEqual(comparison.report["module_name"], "hunger_center")
        self.assertEqual(comparison.report["levels_evaluated"], [1, 2, 3, 4])

    def test_evaluate_variant_sufficiency_uses_hidden_dim_override(self) -> None:
        with patch(
            "spider_cortex_sim.module_local_tasks.make_local_task_network",
            wraps=make_local_task_network,
        ) as mocked_make_network:
            comparison = evaluate_variant_sufficiency(
                "alert_center",
                rng=np.random.default_rng(7),
                hidden_dim=3,
            )

        self.assertEqual(comparison.module_name, "alert_center")
        self.assertEqual(len(comparison.results), 4)
        self.assertGreater(mocked_make_network.call_count, 0)
        self.assertTrue(
            all(call.kwargs["hidden_dim"] == 3 for call in mocked_make_network.call_args_list)
        )

    def test_visual_and_sensory_variant_sufficiency_cover_all_levels(self) -> None:
        for module_name in ("visual_cortex", "sensory_cortex"):
            with self.subTest(module_name=module_name):
                comparison = evaluate_variant_sufficiency(
                    module_name,
                    rng=np.random.default_rng(321),
                )
                self.assertEqual([result.level for result in comparison.results], [1, 2, 3, 4])
                self.assertEqual(comparison.report["module_name"], module_name)
                self.assertEqual(comparison.report["levels_evaluated"], [1, 2, 3, 4])
                self.assertEqual(
                    comparison.report["task_count"],
                    len(ALL_LOCAL_TASKS[module_name]),
                )
