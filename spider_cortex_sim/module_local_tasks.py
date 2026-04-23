from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .interfaces import (
    ACTION_TO_INDEX,
    MODULE_INTERFACE_BY_NAME,
    ModuleInterface,
    get_interface_variant,
    get_variant_levels,
)
from .modules import MODULE_HIDDEN_DIMS, get_variant_hidden_dim
from .nn import ProposalNetwork, softmax


@dataclass(frozen=True)
class LocalTask:
    name: str
    module_name: str
    observation_key: str
    signal_values: dict[str, float]
    expected_favored_actions: tuple[str, ...]
    description: str


@dataclass(frozen=True)
class LocalTaskResult:
    status: str
    details: str
    initial_logits: np.ndarray
    final_logits: np.ndarray
    training_steps_used: int


@dataclass(frozen=True)
class VariantSufficiencyResult:
    module_name: str
    level: int
    interface_name: str
    input_dim: int
    tasks_attempted: int
    tasks_passed: int
    tasks_insufficient: int
    task_details: tuple[dict[str, object], ...]


@dataclass(frozen=True)
class ModuleVariantComparison:
    module_name: str
    results: tuple[VariantSufficiencyResult, ...]
    report: dict[str, object]


def get_task_interface(task: LocalTask) -> ModuleInterface:
    try:
        interface = MODULE_INTERFACE_BY_NAME[task.module_name]
    except KeyError as exc:
        raise ValueError(f"Unknown module local task module '{task.module_name}'.") from exc
    if task.observation_key != interface.observation_key:
        raise ValueError(
            f"Task '{task.name}' observation_key '{task.observation_key}' does not match "
            f"interface '{interface.name}' observation_key '{interface.observation_key}'."
        )
    return interface


def check_interface_sufficiency(task: LocalTask, interface: ModuleInterface) -> tuple[bool, list[str]]:
    available = set(interface.signal_names)
    missing = sorted(name for name in task.signal_values if name not in available)
    return (not missing, missing)


def build_task_observation(task: LocalTask, interface: ModuleInterface) -> np.ndarray:
    observation = np.zeros(interface.input_dim, dtype=float)
    index_by_name = {
        signal.name: idx
        for idx, signal in enumerate(interface.inputs)
    }

    for signal_name in interface.signal_names:
        if signal_name.endswith("_age") and signal_name not in task.signal_values:
            observation[index_by_name[signal_name]] = 1.0

    for signal_name, value in task.signal_values.items():
        observation[index_by_name[signal_name]] = float(value)
    return observation


def make_local_task_network(
    task: LocalTask,
    *,
    interface: ModuleInterface | None = None,
    rng: np.random.Generator,
    hidden_dim: int | None = None,
) -> ProposalNetwork:
    interface = interface or get_task_interface(task)
    resolved_hidden_dim = hidden_dim or MODULE_HIDDEN_DIMS[task.module_name]
    return ProposalNetwork(
        input_dim=interface.input_dim,
        hidden_dim=resolved_hidden_dim,
        output_dim=len(ACTION_TO_INDEX),
        rng=rng,
        name=f"{task.module_name}_local_task",
    )


def _expected_action_indices(task: LocalTask) -> tuple[int, ...]:
    return tuple(ACTION_TO_INDEX[action_name] for action_name in task.expected_favored_actions)


def _favor_margin(logits: np.ndarray, expected_indices: tuple[int, ...]) -> float:
    expected_max = float(np.max(logits[list(expected_indices)]))
    unexpected_indices = [
        idx
        for idx in range(logits.shape[0])
        if idx not in expected_indices
    ]
    if not unexpected_indices:
        return float("inf")
    unexpected_max = float(np.max(logits[unexpected_indices]))
    return expected_max - unexpected_max


def _task_passed(logits: np.ndarray, expected_indices: tuple[int, ...]) -> bool:
    return _favor_margin(logits, expected_indices) > 1e-6


def _top_action_name(logits: np.ndarray) -> str:
    action_by_index = {idx: name for name, idx in ACTION_TO_INDEX.items()}
    return action_by_index[int(np.argmax(logits))]


def evaluate_local_task(
    task: LocalTask,
    network: ProposalNetwork,
    interface: ModuleInterface,
    *,
    max_steps: int = 100,
    lr: float = 0.01,
) -> LocalTaskResult:
    is_sufficient, missing = check_interface_sufficiency(task, interface)
    if not is_sufficient:
        missing_str = ", ".join(missing)
        zeros = np.zeros(len(ACTION_TO_INDEX), dtype=float)
        return LocalTaskResult(
            status="interface_insufficient",
            details=f"Missing interface signals for task '{task.name}': {missing_str}.",
            initial_logits=zeros.copy(),
            final_logits=zeros.copy(),
            training_steps_used=0,
        )

    observation = build_task_observation(task, interface)
    expected_indices = _expected_action_indices(task)
    target_probs = np.zeros(len(ACTION_TO_INDEX), dtype=float)
    target_probs[list(expected_indices)] = 1.0 / float(len(expected_indices))

    initial_logits = network.forward(observation, store_cache=True).copy()
    if _task_passed(initial_logits, expected_indices):
        margin = _favor_margin(initial_logits, expected_indices)
        return LocalTaskResult(
            status="passed",
            details=(
                f"Task '{task.name}' already favors {task.expected_favored_actions} "
                f"(top_action={_top_action_name(initial_logits)}, margin={margin:.3f})."
            ),
            initial_logits=initial_logits,
            final_logits=initial_logits.copy(),
            training_steps_used=0,
        )

    logits = initial_logits.copy()
    final_logits = initial_logits.copy()
    for step in range(max_steps):
        probs = softmax(logits)
        grad_logits = probs - target_probs
        network.backward(grad_logits, lr=lr)
        final_logits = network.forward(observation, store_cache=True).copy()
        if _task_passed(final_logits, expected_indices):
            margin = _favor_margin(final_logits, expected_indices)
            return LocalTaskResult(
                status="passed",
                details=(
                    f"Task '{task.name}' learned after {step + 1} steps "
                    f"(top_action={_top_action_name(final_logits)}, margin={margin:.3f})."
                ),
                initial_logits=initial_logits,
                final_logits=final_logits,
                training_steps_used=step + 1,
            )
        logits = final_logits

    margin = _favor_margin(final_logits, expected_indices)
    return LocalTaskResult(
        status="network_insufficient",
        details=(
            f"Task '{task.name}' did not favor {task.expected_favored_actions} after {max_steps} steps "
            f"(top_action={_top_action_name(final_logits)}, margin={margin:.3f})."
        ),
        initial_logits=initial_logits,
        final_logits=final_logits,
        training_steps_used=max_steps,
    )


def evaluate_module_local_tasks(
    module_name: str,
    *,
    rng: np.random.Generator,
    hidden_dim: int | None = None,
    max_steps: int = 100,
    lr: float = 0.01,
) -> list[tuple[LocalTask, LocalTaskResult]]:
    if module_name not in ALL_LOCAL_TASKS:
        raise ValueError(f"Unknown module local task suite '{module_name}'.")
    results: list[tuple[LocalTask, LocalTaskResult]] = []
    for task in ALL_LOCAL_TASKS[module_name]:
        interface = get_task_interface(task)
        network = make_local_task_network(task, rng=rng, hidden_dim=hidden_dim)
        result = evaluate_local_task(task, network, interface, max_steps=max_steps, lr=lr)
        results.append((task, result))
    return results


def build_variant_sufficiency_report(
    module_name: str,
    results: list[VariantSufficiencyResult],
) -> dict[str, object]:
    """
    Build a serializable sufficiency summary across interface levels for one module.
    """
    if not results:
        raise ValueError(f"No variant sufficiency results provided for module '{module_name}'.")

    unexpected_modules = sorted({result.module_name for result in results if result.module_name != module_name})
    if unexpected_modules:
        raise ValueError(
            f"Variant sufficiency report for '{module_name}' received mismatched modules: {unexpected_modules}."
        )

    total_tasks = len(ALL_LOCAL_TASKS[module_name])
    minimal_sufficient_level: int | None = None
    minimal_sufficient_interface: str | None = None
    level_entries: list[dict[str, object]] = []

    for result in sorted(results, key=lambda item: item.level):
        tasks_failed = result.tasks_attempted - result.tasks_passed
        pass_rate = (
            float(result.tasks_passed) / float(result.tasks_attempted)
            if result.tasks_attempted > 0
            else 0.0
        )
        coverage_rate = float(result.tasks_attempted) / float(total_tasks)
        all_tasks_sufficient = (
            result.tasks_attempted == total_tasks and result.tasks_insufficient == 0
        )
        all_tasks_passed = all_tasks_sufficient and result.tasks_passed == total_tasks
        if minimal_sufficient_level is None and all_tasks_passed:
            minimal_sufficient_level = result.level
            minimal_sufficient_interface = result.interface_name

        level_entries.append(
            {
                "level": result.level,
                "interface_name": result.interface_name,
                "input_dim": result.input_dim,
                "signal_count": result.input_dim,
                "tasks_attempted": result.tasks_attempted,
                "tasks_passed": result.tasks_passed,
                "tasks_failed": tasks_failed,
                "tasks_insufficient": result.tasks_insufficient,
                "pass_rate": pass_rate,
                "coverage_rate": coverage_rate,
                "all_tasks_sufficient": all_tasks_sufficient,
                "all_tasks_passed": all_tasks_passed,
                "task_details": list(result.task_details),
            }
        )

    return {
        "module_name": module_name,
        "task_count": total_tasks,
        "levels_evaluated": [entry["level"] for entry in level_entries],
        "minimal_sufficient_level": minimal_sufficient_level,
        "minimal_sufficient_interface": minimal_sufficient_interface,
        "levels": level_entries,
    }


def evaluate_variant_sufficiency(
    module_name: str,
    rng: np.random.Generator,
    hidden_dim: int | None = None,
) -> ModuleVariantComparison:
    """
    Evaluate local tasks across all registered interface sufficiency levels for one module.
    """
    if module_name not in ALL_LOCAL_TASKS:
        raise ValueError(f"Unknown module local task suite '{module_name}'.")

    tasks = ALL_LOCAL_TASKS[module_name]
    task_seeds = [
        int(rng.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))
        for _ in tasks
    ]
    results: list[VariantSufficiencyResult] = []

    for level in get_variant_levels(module_name):
        interface = get_interface_variant(module_name, level)
        resolved_hidden_dim = (
            hidden_dim if hidden_dim is not None else get_variant_hidden_dim(module_name, level)
        )
        task_details: list[dict[str, object]] = []
        tasks_attempted = 0
        tasks_passed = 0
        tasks_insufficient = 0

        for task, task_seed in zip(tasks, task_seeds, strict=True):
            is_sufficient, missing = check_interface_sufficiency(task, interface)
            if not is_sufficient:
                tasks_insufficient += 1
                task_details.append(
                    {
                        "task_name": task.name,
                        "status": "interface_insufficient",
                        "missing_signals": missing,
                        "expected_favored_actions": list(task.expected_favored_actions),
                    }
                )
                continue

            tasks_attempted += 1
            network = make_local_task_network(
                task,
                interface=interface,
                rng=np.random.default_rng(task_seed),
                hidden_dim=resolved_hidden_dim,
            )
            result = evaluate_local_task(
                task,
                network,
                interface,
                max_steps=250,
                lr=0.05,
            )
            if result.status == "passed":
                tasks_passed += 1
            task_details.append(
                {
                    "task_name": task.name,
                    "status": result.status,
                    "missing_signals": [],
                    "expected_favored_actions": list(task.expected_favored_actions),
                    "training_steps_used": result.training_steps_used,
                    "details": result.details,
                }
            )

        results.append(
            VariantSufficiencyResult(
                module_name=module_name,
                level=level,
                interface_name=interface.name,
                input_dim=interface.input_dim,
                tasks_attempted=tasks_attempted,
                tasks_passed=tasks_passed,
                tasks_insufficient=tasks_insufficient,
                task_details=tuple(task_details),
            )
        )

    return ModuleVariantComparison(
        module_name=module_name,
        results=tuple(results),
        report=build_variant_sufficiency_report(module_name, results),
    )


HUNGER_CENTER_LOCAL_TASKS: tuple[LocalTask, ...] = (
    LocalTask(
        name="food_smell_right",
        module_name="hunger_center",
        observation_key="hunger",
        signal_values={
            "hunger": 1.0,
            "food_smell_dx": 1.0,
            "food_smell_strength": 1.0,
        },
        expected_favored_actions=("MOVE_RIGHT", "ORIENT_RIGHT"),
        description="High hunger with a strong food scent on the right should bias rightward pursuit.",
    ),
    LocalTask(
        name="food_visible_ahead",
        module_name="hunger_center",
        observation_key="hunger",
        signal_values={
            "hunger": 1.0,
            "food_visible": 1.0,
            "food_certainty": 1.0,
            "food_dx": 0.5,
            "food_dy": -0.5,
        },
        expected_favored_actions=("MOVE_UP", "MOVE_RIGHT"),
        description="Visible food ahead-right under high hunger should bias approach actions.",
    ),
)

HUNGER_CENTER_TASKS = HUNGER_CENTER_LOCAL_TASKS


SLEEP_CENTER_LOCAL_TASKS: tuple[LocalTask, ...] = (
    LocalTask(
        name="shelter_left_at_night",
        module_name="sleep_center",
        observation_key="sleep",
        signal_values={
            "fatigue": 1.0,
            "night": 1.0,
            "shelter_memory_dx": -1.0,
        },
        expected_favored_actions=("MOVE_LEFT", "ORIENT_LEFT"),
        description="High fatigue at night with remembered shelter left should bias leftward return.",
    ),
    LocalTask(
        name="on_shelter_tired",
        module_name="sleep_center",
        observation_key="sleep",
        signal_values={
            "fatigue": 1.0,
            "night": 1.0,
            "on_shelter": 1.0,
            "shelter_role_level": 1.0,
        },
        expected_favored_actions=("STAY",),
        description="When already sheltered at night and tired, the module should prefer resting in place.",
    ),
)

SLEEP_CENTER_TASKS = SLEEP_CENTER_LOCAL_TASKS


ALERT_CENTER_LOCAL_TASKS: tuple[LocalTask, ...] = (
    LocalTask(
        name="predator_right_flee",
        module_name="alert_center",
        observation_key="alert",
        signal_values={
            "predator_visible": 1.0,
            "predator_certainty": 1.0,
            "predator_dx": 1.0,
            "visual_predator_threat": 1.0,
            "dominant_predator_visual": 1.0,
        },
        expected_favored_actions=("MOVE_LEFT", "ORIENT_LEFT"),
        description="A visible predator on the right should bias escape or orient-away behavior.",
    ),
)

ALERT_CENTER_TASKS = ALERT_CENTER_LOCAL_TASKS


VISUAL_CORTEX_LOCAL_TASKS: tuple[LocalTask, ...] = (
    LocalTask(
        name="food_visible_right",
        module_name="visual_cortex",
        observation_key="visual",
        signal_values={
            "food_visible": 1.0,
            "food_dx": 1.0,
            "heading_dx": 0.0,
            "heading_dy": -1.0,
        },
        expected_favored_actions=("MOVE_RIGHT", "ORIENT_RIGHT"),
        description="Visible food to the right should bias rightward approach or orientation.",
    ),
    LocalTask(
        name="shelter_visible_left_scan",
        module_name="visual_cortex",
        observation_key="visual",
        signal_values={
            "shelter_visible": 1.0,
            "shelter_dx": -1.0,
            "heading_dx": 1.0,
            "heading_dy": 0.0,
        },
        expected_favored_actions=("MOVE_LEFT", "ORIENT_LEFT"),
        description="A visible shelter on the left should bias return or scan toward the refuge.",
    ),
    LocalTask(
        name="predator_above_high_threat",
        module_name="visual_cortex",
        observation_key="visual",
        signal_values={
            "predator_visible": 1.0,
            "predator_dy": -1.0,
            "predator_motion_salience": 1.0,
            "visual_predator_threat": 1.0,
            "night": 1.0,
        },
        expected_favored_actions=("MOVE_DOWN", "ORIENT_DOWN"),
        description="A high-salience visual predator above should bias escape or orient-away behavior.",
    ),
)

VISUAL_CORTEX_TASKS = VISUAL_CORTEX_LOCAL_TASKS


SENSORY_CORTEX_LOCAL_TASKS: tuple[LocalTask, ...] = (
    LocalTask(
        name="pain_with_predator_smell",
        module_name="sensory_cortex",
        observation_key="sensory",
        signal_values={
            "recent_pain": 1.0,
            "recent_contact": 1.0,
            "predator_smell_strength": 1.0,
            "predator_smell_dx": 1.0,
        },
        expected_favored_actions=("MOVE_LEFT", "ORIENT_LEFT"),
        description="Pain plus predator smell on the right should bias withdrawal or orient-away behavior.",
    ),
    LocalTask(
        name="food_smell_right_when_hungry",
        module_name="sensory_cortex",
        observation_key="sensory",
        signal_values={
            "hunger": 1.0,
            "food_smell_strength": 1.0,
            "food_smell_dx": 1.0,
        },
        expected_favored_actions=("MOVE_RIGHT", "ORIENT_RIGHT"),
        description="Strong food smell on the right under high hunger should bias appetitive approach.",
    ),
    LocalTask(
        name="low_light_predator_food_conflict",
        module_name="sensory_cortex",
        observation_key="sensory",
        signal_values={
            "recent_pain": 1.0,
            "predator_smell_strength": 1.0,
            "predator_smell_dx": -1.0,
            "food_smell_strength": 1.0,
            "food_smell_dx": 1.0,
            "hunger": 1.0,
            "fatigue": 0.6,
            "light": 0.0,
        },
        expected_favored_actions=("MOVE_LEFT", "ORIENT_LEFT"),
        description="Under low light, simultaneous appetitive and threat smell should still preserve a coherent escape heading away from the predator cue.",
    ),
)

SENSORY_CORTEX_TASKS = SENSORY_CORTEX_LOCAL_TASKS


ALL_LOCAL_TASKS: dict[str, tuple[LocalTask, ...]] = {
    "hunger_center": HUNGER_CENTER_LOCAL_TASKS,
    "sleep_center": SLEEP_CENTER_LOCAL_TASKS,
    "alert_center": ALERT_CENTER_LOCAL_TASKS,
    "visual_cortex": VISUAL_CORTEX_LOCAL_TASKS,
    "sensory_cortex": SENSORY_CORTEX_LOCAL_TASKS,
}
