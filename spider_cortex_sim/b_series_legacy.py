from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .b_series import B_SEMANTIC_ACTIONS, B_SEMANTIC_ACTION_TO_INDEX
from .bus import MessageBus
from .nn import MotorNetwork, ProposalNetwork, one_hot, softmax


LEGACY_B0_ACTIONS: Sequence[str] = B_SEMANTIC_ACTIONS
LEGACY_B0_ACTION_TO_INDEX = B_SEMANTIC_ACTION_TO_INDEX
LEGACY_B0_MOVE_DELTAS = (
    (0, -1),
    (0, 1),
    (-1, 0),
    (1, 0),
)


@dataclass
class LegacyB0State:
    x: int
    y: int
    hunger: float
    fatigue: float
    health: float
    recent_pain: float
    last_reward: float
    total_reward: float
    food_eaten: int
    sleep_events: int
    shelter_entries: int
    alert_events: int
    steps_alive: int
    last_action: str


class LegacyB0World:
    """Isolated B0 semantic world inspired by commit 36191725."""

    def __init__(
        self,
        *,
        width: int = 12,
        height: int = 12,
        food_count: int = 4,
        day_length: int = 18,
        night_length: int = 12,
        seed: int = 0,
    ) -> None:
        self.width = int(width)
        self.height = int(height)
        self.food_count = int(food_count)
        self.day_length = int(day_length)
        self.night_length = int(night_length)
        self.cycle_length = self.day_length + self.night_length
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)
        self.tick = 0
        self.shelter_cells = self._make_shelter_cells()
        self.food_positions: list[tuple[int, int]] = []
        self.state = LegacyB0State(
            x=0,
            y=0,
            hunger=0.0,
            fatigue=0.0,
            health=1.0,
            recent_pain=0.0,
            last_reward=0.0,
            total_reward=0.0,
            food_eaten=0,
            sleep_events=0,
            shelter_entries=0,
            alert_events=0,
            steps_alive=0,
            last_action="STAY",
        )

    def _make_shelter_cells(self) -> set[tuple[int, int]]:
        cx = self.width // 2
        cy = self.height // 2
        cells: set[tuple[int, int]] = set()
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                x = min(max(cx + dx, 0), self.width - 1)
                y = min(max(cy + dy, 0), self.height - 1)
                cells.add((x, y))
        return cells

    def reset(self, seed: int | None = None) -> dict[str, np.ndarray]:
        self.rng = np.random.default_rng(self.seed if seed is None else seed)
        self.tick = 0
        start_x, start_y = sorted(self.shelter_cells)[len(self.shelter_cells) // 2]
        self.state = LegacyB0State(
            x=start_x,
            y=start_y,
            hunger=float(self.rng.uniform(0.35, 0.55)),
            fatigue=float(self.rng.uniform(0.15, 0.35)),
            health=1.0,
            recent_pain=0.0,
            last_reward=0.0,
            total_reward=0.0,
            food_eaten=0,
            sleep_events=0,
            shelter_entries=0,
            alert_events=0,
            steps_alive=0,
            last_action="STAY",
        )
        self.food_positions = []
        for _ in range(self.food_count):
            self.food_positions.append(self._random_empty_cell())
        return self.observe()

    def _random_empty_cell(self) -> tuple[int, int]:
        while True:
            x = int(self.rng.integers(0, self.width))
            y = int(self.rng.integers(0, self.height))
            candidate = (x, y)
            if candidate in self.shelter_cells:
                continue
            if candidate == (self.state.x, self.state.y):
                continue
            if candidate in self.food_positions:
                continue
            return candidate

    def is_night(self, tick: int | None = None) -> bool:
        phase = (self.tick if tick is None else int(tick)) % self.cycle_length
        return phase >= self.day_length

    def light_level(self) -> float:
        return 0.0 if self.is_night() else 1.0

    def phase_features(self) -> tuple[float, float]:
        phase = (self.tick % self.cycle_length) / float(self.cycle_length)
        angle = 2.0 * np.pi * phase
        return float(np.sin(angle)), float(np.cos(angle))

    def on_shelter(self) -> bool:
        return (self.state.x, self.state.y) in self.shelter_cells

    def on_food(self) -> bool:
        return (self.state.x, self.state.y) in self.food_positions

    def nearest(
        self,
        positions: Iterable[tuple[int, int]],
    ) -> tuple[tuple[int, int], int]:
        px, py = self.state.x, self.state.y
        best: tuple[int, int] | None = None
        best_dist = 10**9
        for x, y in positions:
            dist = abs(x - px) + abs(y - py)
            if dist < best_dist:
                best = (x, y)
                best_dist = dist
        if best is None:
            return (px, py), 0
        return best, int(best_dist)

    def _relative(self, target: tuple[int, int]) -> tuple[float, float, float]:
        max_dx = max(1, self.width - 1)
        max_dy = max(1, self.height - 1)
        dx = (target[0] - self.state.x) / max_dx
        dy = (target[1] - self.state.y) / max_dy
        dist = (
            abs(target[0] - self.state.x) + abs(target[1] - self.state.y)
        ) / float(self.width + self.height)
        return float(dx), float(dy), float(dist)

    def observe(self) -> dict[str, np.ndarray]:
        food_pos, food_dist = self.nearest(self.food_positions)
        shelter_pos, shelter_dist = self.nearest(self.shelter_cells)
        food_dx, food_dy, food_dist_norm = self._relative(food_pos)
        shelter_dx, shelter_dy, shelter_dist_norm = self._relative(shelter_pos)
        phase_sin, phase_cos = self.phase_features()
        on_shelter = float(self.on_shelter())
        on_food = float(self.on_food())
        night = float(self.is_night())
        light = self.light_level()
        x_norm = self.state.x / float(max(1, self.width - 1))
        y_norm = self.state.y / float(max(1, self.height - 1))
        open_exposed = float(bool(night) and not self.on_shelter())
        obs = {
            "visual": np.array(
                [
                    food_dx,
                    food_dy,
                    food_dist_norm,
                    shelter_dx,
                    shelter_dy,
                    shelter_dist_norm,
                    on_food,
                    on_shelter,
                    light,
                    phase_sin,
                    phase_cos,
                ],
                dtype=float,
            ),
            "sensory": np.array(
                [
                    self.state.hunger,
                    self.state.fatigue,
                    self.state.health,
                    on_shelter,
                    on_food,
                    night,
                    self.state.recent_pain,
                    self.state.last_reward,
                    x_norm,
                    y_norm,
                    phase_sin,
                    phase_cos,
                ],
                dtype=float,
            ),
            "hunger": np.array(
                [
                    self.state.hunger,
                    food_dx,
                    food_dy,
                    food_dist_norm,
                    on_food,
                    self.state.health,
                    light,
                    phase_cos,
                ],
                dtype=float,
            ),
            "sleep": np.array(
                [
                    self.state.fatigue,
                    shelter_dx,
                    shelter_dy,
                    shelter_dist_norm,
                    on_shelter,
                    night,
                    light,
                    phase_cos,
                ],
                dtype=float,
            ),
            "alert": np.array(
                [
                    night,
                    open_exposed,
                    self.state.recent_pain,
                    self.state.health,
                    self.state.hunger,
                    self.state.fatigue,
                    shelter_dx,
                    shelter_dy,
                    shelter_dist_norm,
                ],
                dtype=float,
            ),
            "motor_extra": np.array(
                [
                    self.state.hunger,
                    self.state.fatigue,
                    self.state.health,
                    night,
                    on_shelter,
                    on_food,
                    open_exposed,
                    x_norm,
                    y_norm,
                    phase_sin,
                    phase_cos,
                ],
                dtype=float,
            ),
        }
        obs["meta"] = {
            "food_dist": food_dist,
            "shelter_dist": shelter_dist,
            "night": bool(night),
            "on_shelter": bool(on_shelter),
            "on_food": bool(on_food),
        }
        return obs

    def _respawn_food(self, eaten_position: tuple[int, int]) -> None:
        self.food_positions = [p for p in self.food_positions if p != eaten_position]
        self.food_positions.append(self._random_empty_cell())

    def _move(self, dx: int, dy: int) -> bool:
        new_x = int(np.clip(self.state.x + dx, 0, self.width - 1))
        new_y = int(np.clip(self.state.y + dy, 0, self.height - 1))
        moved = (new_x, new_y) != (self.state.x, self.state.y)
        self.state.x = new_x
        self.state.y = new_y
        return moved

    def _step_towards(self, target: tuple[int, int]) -> bool:
        dx = target[0] - self.state.x
        dy = target[1] - self.state.y
        if dx == 0 and dy == 0:
            return False
        if abs(dx) > abs(dy):
            return self._move(int(np.sign(dx)), 0)
        if abs(dy) > abs(dx):
            return self._move(0, int(np.sign(dy)))
        if self.rng.random() < 0.5:
            return self._move(int(np.sign(dx)), 0)
        return self._move(0, int(np.sign(dy)))

    def _step_explore(self) -> bool:
        order = list(range(len(LEGACY_B0_MOVE_DELTAS)))
        self.rng.shuffle(order)
        for idx in order:
            dx, dy = LEGACY_B0_MOVE_DELTAS[idx]
            if self._move(dx, dy):
                return True
        return False

    def step(
        self,
        action_idx: int,
    ) -> tuple[dict[str, np.ndarray], float, bool, dict[str, object]]:
        action_name = LEGACY_B0_ACTIONS[int(action_idx)]
        was_on_shelter = self.on_shelter()
        night = self.is_night()
        _, prev_food_dist = self.nearest(self.food_positions)
        _, prev_shelter_dist = self.nearest(self.shelter_cells)
        reward = 0.0
        info: dict[str, object] = {
            "action": action_name,
            "ate": False,
            "slept": False,
            "pain": False,
        }

        moved = False
        if action_name == "MOVE_TO_FOOD":
            target, _ = self.nearest(self.food_positions)
            moved = self._step_towards(target)
            reward -= 0.015
        elif action_name == "MOVE_TO_SHELTER":
            target, _ = self.nearest(self.shelter_cells)
            moved = self._step_towards(target)
            reward -= 0.015
        elif action_name == "EXPLORE":
            moved = self._step_explore()
            reward -= 0.020
        elif action_name == "STAY":
            reward -= 0.010

        self.state.hunger += 0.012 + (0.006 if moved else 0.003)
        self.state.fatigue += 0.008 + (0.006 if moved else 0.002)
        if night and action_name != "SLEEP":
            self.state.fatigue += 0.015
        reward -= 0.10 * self.state.hunger
        reward -= 0.08 * self.state.fatigue

        if action_name == "EAT":
            if self.on_food():
                hunger_before = self.state.hunger
                self.state.hunger = max(0.0, self.state.hunger - 0.55)
                self.state.health = min(1.0, self.state.health + 0.04)
                reward += 0.30 + 3.00 * hunger_before
                self.state.food_eaten += 1
                info["ate"] = True
                self._respawn_food((self.state.x, self.state.y))
            else:
                reward -= 0.15
        elif action_name == "SLEEP":
            if self.on_shelter():
                fatigue_before = self.state.fatigue
                restore = 0.34 if night else 0.18
                self.state.fatigue = max(0.0, self.state.fatigue - restore)
                self.state.health = min(
                    1.0,
                    self.state.health + (0.06 if night else 0.02),
                )
                sleep_drive = max(fatigue_before - 0.35, 0.0)
                reward += -0.15 + 3.00 * sleep_drive + (0.55 if night else 0.00)
                reward -= 0.50 * self.state.hunger
                self.state.sleep_events += 1
                info["slept"] = True
            else:
                reward -= 0.80

        _, new_food_dist = self.nearest(self.food_positions)
        _, new_shelter_dist = self.nearest(self.shelter_cells)
        food_progress = float(prev_food_dist - new_food_dist)
        shelter_progress = float(prev_shelter_dist - new_shelter_dist)
        if action_name == "MOVE_TO_FOOD" and self.state.hunger > 0.25:
            reward += 0.20 * food_progress
        elif action_name == "MOVE_TO_SHELTER" and (self.state.fatigue > 0.35 or night):
            reward += 0.18 * shelter_progress
        elif action_name == "EXPLORE" and moved and self.state.hunger > 0.35 and not night:
            reward += 0.04

        on_shelter_now = self.on_shelter()
        if on_shelter_now and not was_on_shelter:
            self.state.shelter_entries += 1
            reward += 0.10
        if night and on_shelter_now:
            reward += 0.08
        if night and not on_shelter_now:
            reward -= 0.20
            if self.rng.random() < 0.12:
                damage = float(self.rng.uniform(0.08, 0.20))
                self.state.health = max(0.0, self.state.health - damage)
                self.state.recent_pain = min(1.0, self.state.recent_pain + 0.75)
                self.state.alert_events += 1
                reward -= 1.15
                info["pain"] = True
        else:
            self.state.recent_pain *= 0.55

        if self.state.hunger > 0.70:
            excess = (self.state.hunger - 0.70) / 0.30
            self.state.health = max(0.0, self.state.health - 0.10 * excess)
            reward -= 0.45 * excess
        if self.state.fatigue > 0.78:
            excess = (self.state.fatigue - 0.78) / 0.22
            self.state.health = max(0.0, self.state.health - 0.08 * excess)
            reward -= 0.35 * excess

        self.state.hunger = float(np.clip(self.state.hunger, 0.0, 1.0))
        self.state.fatigue = float(np.clip(self.state.fatigue, 0.0, 1.0))
        self.state.health = float(np.clip(self.state.health, 0.0, 1.0))
        self.state.last_reward = float(reward)
        self.state.total_reward += float(reward)
        self.state.steps_alive += 1
        self.state.last_action = action_name
        self.tick += 1

        done = self.state.health <= 0.0
        if done:
            reward -= 5.0
            self.state.last_reward = float(reward)
            self.state.total_reward += -5.0
        info["state"] = asdict(self.state)
        return self.observe(), float(reward), bool(done), info

    def state_dict(self) -> dict[str, object]:
        return asdict(self.state)


@dataclass(frozen=True)
class LegacyB0ModuleSpec:
    name: str
    observation_key: str
    input_dim: int
    hidden_dim: int


@dataclass
class LegacyB0ModuleResult:
    name: str
    observation_key: str
    logits: np.ndarray
    probs: np.ndarray


LEGACY_B0_MODULE_SPECS: tuple[LegacyB0ModuleSpec, ...] = (
    LegacyB0ModuleSpec("visual_cortex", "visual", 11, 24),
    LegacyB0ModuleSpec("sensory_cortex", "sensory", 12, 24),
    LegacyB0ModuleSpec("hunger_center", "hunger", 8, 20),
    LegacyB0ModuleSpec("sleep_center", "sleep", 8, 20),
    LegacyB0ModuleSpec("alert_center", "alert", 9, 20),
)


class LegacyB0ModuleBank:
    def __init__(self, action_dim: int, rng: np.random.Generator) -> None:
        self.specs = list(LEGACY_B0_MODULE_SPECS)
        self.modules = {
            spec.name: ProposalNetwork(
                input_dim=spec.input_dim,
                hidden_dim=spec.hidden_dim,
                output_dim=action_dim,
                rng=rng,
                name=spec.name,
            )
            for spec in self.specs
        }

    def forward(
        self,
        observation: Dict[str, np.ndarray],
        *,
        store_cache: bool = True,
    ) -> list[LegacyB0ModuleResult]:
        outputs: list[LegacyB0ModuleResult] = []
        for spec in self.specs:
            logits = self.modules[spec.name].forward(
                observation[spec.observation_key],
                store_cache=store_cache,
            )
            outputs.append(
                LegacyB0ModuleResult(
                    name=spec.name,
                    observation_key=spec.observation_key,
                    logits=logits,
                    probs=softmax(logits),
                )
            )
        return outputs

    def backward(self, grad_logits: np.ndarray, lr: float) -> None:
        for spec in self.specs:
            self.modules[spec.name].backward(grad_logits, lr=lr)

    def parameter_norms(self) -> dict[str, float]:
        return {name: module.parameter_norm() for name, module in self.modules.items()}


@dataclass
class LegacyB0BrainStep:
    module_results: list[LegacyB0ModuleResult]
    motor_correction_logits: np.ndarray
    total_logits: np.ndarray
    policy: np.ndarray
    value: float
    action_idx: int
    motor_input: np.ndarray


class LegacyB0Brain:
    def __init__(
        self,
        *,
        seed: int = 0,
        gamma: float = 0.96,
        module_lr: float = 0.010,
        motor_lr: float = 0.012,
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.action_dim = len(LEGACY_B0_ACTIONS)
        self.module_bank = LegacyB0ModuleBank(
            action_dim=self.action_dim,
            rng=self.rng,
        )
        motor_input_dim = self.action_dim * len(self.module_bank.specs) + 11
        self.motor_cortex = MotorNetwork(
            input_dim=motor_input_dim,
            hidden_dim=32,
            output_dim=self.action_dim,
            rng=self.rng,
            name="motor_cortex",
        )
        self.gamma = float(gamma)
        self.module_lr = float(module_lr)
        self.motor_lr = float(motor_lr)

    def _build_motor_input(
        self,
        module_results: list[LegacyB0ModuleResult],
        observation: dict[str, np.ndarray],
    ) -> np.ndarray:
        logits_flat = np.concatenate([result.logits for result in module_results], axis=0)
        return np.concatenate([logits_flat, observation["motor_extra"]], axis=0)

    def act(
        self,
        observation: dict[str, np.ndarray],
        bus: MessageBus | None = None,
        *,
        sample: bool = True,
    ) -> LegacyB0BrainStep:
        module_results = self.module_bank.forward(observation, store_cache=True)
        if bus is not None:
            for result in module_results:
                bus.publish(
                    sender=result.name,
                    topic="action.proposal",
                    payload={
                        "semantic_action_logits": result.logits.round(6).tolist(),
                        "semantic_action_probs": result.probs.round(6).tolist(),
                    },
                )
        motor_input = self._build_motor_input(module_results, observation)
        motor_correction_logits, value = self.motor_cortex.forward(
            motor_input,
            store_cache=True,
        )
        total_logits = motor_correction_logits.copy()
        for result in module_results:
            total_logits += result.logits
        policy = softmax(total_logits)
        if sample:
            action_idx = int(self.rng.choice(self.action_dim, p=policy))
        else:
            action_idx = int(np.argmax(policy))
        if bus is not None:
            bus.publish(
                sender="motor_cortex",
                topic="action.selection",
                payload={
                    "b_level": 0,
                    "b_mode": "legacy_semantic",
                    "semantic_logits": total_logits.round(6).tolist(),
                    "policy": policy.round(6).tolist(),
                    "semantic_action": LEGACY_B0_ACTIONS[action_idx],
                    "value_estimate": round(float(value), 6),
                },
            )
        return LegacyB0BrainStep(
            module_results=module_results,
            motor_correction_logits=motor_correction_logits,
            total_logits=total_logits,
            policy=policy,
            value=float(value),
            action_idx=action_idx,
            motor_input=motor_input,
        )

    def estimate_value(self, observation: dict[str, np.ndarray]) -> float:
        module_results = self.module_bank.forward(observation, store_cache=False)
        motor_input = self._build_motor_input(module_results, observation)
        _, value = self.motor_cortex.forward(motor_input, store_cache=False)
        return float(value)

    def learn(
        self,
        decision: LegacyB0BrainStep,
        reward: float,
        next_observation: dict[str, np.ndarray],
        done: bool,
    ) -> dict[str, float]:
        next_value = 0.0 if done else self.estimate_value(next_observation)
        td_target = reward + self.gamma * next_value
        advantage = float(np.clip(td_target - decision.value, -4.0, 4.0))
        grad_policy_logits = advantage * (
            decision.policy - one_hot(decision.action_idx, self.action_dim)
        )
        self.module_bank.backward(grad_policy_logits, lr=self.module_lr)
        grad_value = decision.value - td_target
        self.motor_cortex.backward(
            grad_policy_logits=grad_policy_logits,
            grad_value=grad_value,
            lr=self.motor_lr,
        )
        entropy = -float(np.sum(decision.policy * np.log(decision.policy + 1e-8)))
        return {
            "reward": float(reward),
            "td_target": float(td_target),
            "td_error": float(advantage),
            "value": float(decision.value),
            "next_value": float(next_value),
            "entropy": entropy,
        }

    def parameter_norms(self) -> dict[str, float]:
        norms = self.module_bank.parameter_norms()
        norms["motor_cortex"] = self.motor_cortex.parameter_norm()
        return norms


@dataclass
class LegacyB0EpisodeStats:
    episode: int
    training: bool
    total_reward: float
    steps: int
    food_eaten: int
    sleep_events: int
    shelter_entries: int
    alert_events: int
    alive: bool
    final_hunger: float
    final_fatigue: float
    final_health: float


class LegacyB0Simulation:
    def __init__(
        self,
        *,
        width: int = 12,
        height: int = 12,
        food_count: int = 4,
        day_length: int = 18,
        night_length: int = 12,
        max_steps: int = 120,
        seed: int = 0,
        gamma: float = 0.96,
        module_lr: float = 0.010,
        motor_lr: float = 0.012,
    ) -> None:
        self.seed = int(seed)
        self.world = LegacyB0World(
            width=width,
            height=height,
            food_count=food_count,
            day_length=day_length,
            night_length=night_length,
            seed=seed,
        )
        self.brain = LegacyB0Brain(
            seed=seed,
            gamma=gamma,
            module_lr=module_lr,
            motor_lr=motor_lr,
        )
        self.max_steps = int(max_steps)
        self.bus = MessageBus()

    def run_episode(
        self,
        episode_index: int,
        *,
        training: bool,
        sample: bool,
        capture_trace: bool = False,
    ) -> tuple[LegacyB0EpisodeStats, list[dict[str, object]]]:
        episode_seed = self.seed + 997 * (episode_index + 1)
        observation = self.world.reset(seed=episode_seed)
        trace: list[dict[str, object]] = []
        total_reward = 0.0
        for step in range(self.max_steps):
            self.bus.set_tick(step)
            self.bus.publish(
                sender="environment",
                topic="observation",
                payload={"state": self.world.state_dict(), "meta": observation["meta"]},
            )
            decision = self.brain.act(observation, self.bus, sample=sample)
            next_observation, reward, done, info = self.world.step(decision.action_idx)
            if training:
                learn_stats = self.brain.learn(
                    decision,
                    reward,
                    next_observation,
                    done,
                )
                self.bus.publish(
                    sender="learning",
                    topic="td_update",
                    payload=learn_stats,
                )
            if capture_trace:
                trace.append(
                    {
                        "episode": episode_index,
                        "tick": step,
                        "training": training,
                        "b_level": 0,
                        "b_mode": "legacy_semantic",
                        "semantic_action": LEGACY_B0_ACTIONS[decision.action_idx],
                        "state": self.world.state_dict(),
                        "reward": float(reward),
                        "info": info,
                        "done": bool(done),
                        "messages": self.bus.serialize_current_tick(),
                    }
                )
            total_reward += float(reward)
            observation = next_observation
            if done:
                break
        state = self.world.state
        return (
            LegacyB0EpisodeStats(
                episode=episode_index,
                training=training,
                total_reward=float(total_reward),
                steps=state.steps_alive,
                food_eaten=state.food_eaten,
                sleep_events=state.sleep_events,
                shelter_entries=state.shelter_entries,
                alert_events=state.alert_events,
                alive=state.health > 0.0,
                final_hunger=float(state.hunger),
                final_fatigue=float(state.fatigue),
                final_health=float(state.health),
            ),
            trace,
        )

    def train(
        self,
        episodes: int,
        *,
        evaluation_episodes: int = 3,
        capture_evaluation_trace: bool = True,
    ) -> tuple[dict[str, object], list[dict[str, object]]]:
        training_history: list[LegacyB0EpisodeStats] = []
        for episode in range(int(episodes)):
            stats, _ = self.run_episode(
                episode,
                training=True,
                sample=True,
                capture_trace=False,
            )
            training_history.append(stats)
        evaluation_history: list[LegacyB0EpisodeStats] = []
        evaluation_trace: list[dict[str, object]] = []
        for idx in range(int(evaluation_episodes)):
            stats, trace = self.run_episode(
                int(episodes) + idx,
                training=False,
                sample=False,
                capture_trace=capture_evaluation_trace and idx == evaluation_episodes - 1,
            )
            evaluation_history.append(stats)
            if trace:
                evaluation_trace = trace
        return self._build_summary(training_history, evaluation_history), evaluation_trace

    def _build_summary(
        self,
        training_history: list[LegacyB0EpisodeStats],
        evaluation_history: list[LegacyB0EpisodeStats],
    ) -> dict[str, object]:
        last_window = training_history[-min(10, len(training_history)) :]
        return {
            "b_level": 0,
            "b_mode": "legacy_semantic",
            "semantic_actions": list(LEGACY_B0_ACTIONS),
            "training": {
                "episodes": len(training_history),
                "mean_reward_last_window": mean(
                    stat.total_reward for stat in last_window
                )
                if last_window
                else 0.0,
                "mean_food_last_window": mean(stat.food_eaten for stat in last_window)
                if last_window
                else 0.0,
                "mean_sleep_last_window": mean(
                    stat.sleep_events for stat in last_window
                )
                if last_window
                else 0.0,
                "survival_rate_last_window": mean(
                    1.0 if stat.alive else 0.0 for stat in last_window
                )
                if last_window
                else 0.0,
                "history_tail": [asdict(stat) for stat in last_window],
            },
            "evaluation": {
                "episodes": len(evaluation_history),
                "mean_reward": mean(stat.total_reward for stat in evaluation_history)
                if evaluation_history
                else 0.0,
                "mean_food": mean(stat.food_eaten for stat in evaluation_history)
                if evaluation_history
                else 0.0,
                "mean_sleep": mean(stat.sleep_events for stat in evaluation_history)
                if evaluation_history
                else 0.0,
                "survival_rate": mean(
                    1.0 if stat.alive else 0.0 for stat in evaluation_history
                )
                if evaluation_history
                else 0.0,
                "episodes_detail": [asdict(stat) for stat in evaluation_history],
            },
            "parameter_norms": self.brain.parameter_norms(),
        }


def run_b0_legacy_training(
    *,
    episodes: int = 120,
    evaluation_episodes: int = 3,
    max_steps: int = 120,
    seed: int = 0,
    output_dir: str | Path | None = None,
) -> dict[str, object]:
    sim = LegacyB0Simulation(max_steps=max_steps, seed=seed)
    summary, trace = sim.train(
        int(episodes),
        evaluation_episodes=int(evaluation_episodes),
        capture_evaluation_trace=True,
    )
    if output_dir is not None:
        import json

        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "b0_legacy_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        with (directory / "b0_legacy_trace.jsonl").open("w", encoding="utf-8") as handle:
            for item in trace:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    return summary


__all__ = [
    "LEGACY_B0_ACTIONS",
    "LEGACY_B0_ACTION_TO_INDEX",
    "LegacyB0Brain",
    "LegacyB0EpisodeStats",
    "LegacyB0Simulation",
    "LegacyB0World",
    "run_b0_legacy_training",
]
