from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from ..agent import BrainStep
from ..metrics import EpisodeMetricAccumulator
from ..predator import PREDATOR_STATES
from ..simulation import SpiderSimulation
from ..world import REWARD_COMPONENT_NAMES
from .constants import (
    CELL_SIZE,
    DEFAULT_BRAIN_DIR,
    DEFAULT_SPEED_IDX,
    PANEL_WIDTH,
    TICK_SPEEDS,
    TOP_BAR_HEIGHT,
)


class GUIController:
    def __init__(self, sim: SpiderSimulation) -> None:
        self.sim = sim
        self.world = sim.world
        self.brain = sim.brain
        self.bus = sim.bus

        grid_w = self.world.width * CELL_SIZE
        self.win_w = grid_w + PANEL_WIDTH

        self.grid_offset_x = 0
        self.grid_offset_y = TOP_BAR_HEIGHT
        self.panel_x = grid_w
        self.panel_y = TOP_BAR_HEIGHT

        self.running = True
        self.paused = True
        self.step_requested = False
        self.speed_idx = DEFAULT_SPEED_IDX
        self.tick_timer = 0.0

        self.phase: str = "training"  # "training", "evaluation", "done"
        self.current_episode = 0
        self.total_train_episodes = 0
        self.total_eval_episodes = 0
        self.max_steps = sim.max_steps

        self.current_step = 0
        self.episode_reward = 0.0
        self.episode_done = False
        self.observation: Optional[Dict] = None
        self.last_decision: Optional[BrainStep] = None
        self.last_info: Optional[Dict] = None
        self.last_reward = 0.0
        self.reward_history: List[float] = []
        self.training_rewards: List[float] = []
        self.episode_metrics = EpisodeMetricAccumulator(REWARD_COMPONENT_NAMES, PREDATOR_STATES)
        self.show_visibility_overlay = False
        self.show_smell_overlay = False
        self.panel_scroll = 0
        self.panel_content_height = 0

        # Status toast
        self.toast_text: str = ""
        self.toast_timer: float = 0.0
        self.toast_is_error: bool = False

    def configure_run(self, train_episodes: int, eval_episodes: int) -> None:
        self.total_train_episodes = train_episodes
        self.total_eval_episodes = eval_episodes
        self.phase = "training"
        self.current_episode = 0
        self._start_episode()

    def _start_episode(self) -> None:
        is_training = self.phase == "training"
        if is_training:
            episode_seed = self.sim.seed + 997 * (self.current_episode + 1)
        else:
            episode_seed = self.sim.seed + 997 * (self.total_train_episodes + self.current_episode + 1)

        self.observation = self.world.reset(seed=episode_seed)
        self.current_step = 0
        self.episode_reward = 0.0
        self.episode_done = False
        self.last_decision = None
        self.last_info = None
        self.last_reward = 0.0
        self.reward_history.clear()
        self.episode_metrics = EpisodeMetricAccumulator(REWARD_COMPONENT_NAMES, PREDATOR_STATES)

    def _do_step(self) -> None:
        if self.episode_done or self.observation is None:
            return

        is_training = self.phase == "training"
        self.bus.set_tick(self.current_step)
        self.bus.publish(
            sender="environment",
            topic="observation",
            payload={
                "state": self.world.state_dict(),
                "meta": self.observation["meta"],
            },
        )

        decision = self.brain.act(self.observation, self.bus, sample=is_training)
        predator_state_before = self.world.lizard.mode
        next_obs, reward, done, info = self.world.step(decision.action_idx)

        if is_training:
            terminal = done or (self.current_step + 1) >= self.max_steps
            learn_stats = self.brain.learn(decision, reward, next_obs, terminal)
            self.bus.publish(sender="learning", topic="td_update", payload=learn_stats)

        self.last_decision = decision
        self.last_info = info
        self.last_reward = reward
        self.episode_reward += reward
        self.reward_history.append(reward)
        self.episode_metrics.record_transition(
            step=self.current_step,
            observation_meta=self.observation["meta"],
            next_meta=next_obs["meta"],
            info=info,
            state=self.world.state,
            predator_state_before=predator_state_before,
            predator_state=self.world.lizard.mode,
        )
        self.observation = next_obs
        self.current_step += 1

        if done:
            self.episode_done = True

        if self.current_step >= self.max_steps:
            self.episode_done = True

    def _advance_episode(self) -> None:
        """
        Advance to the next episode and update the GUI's phase and episode state.

        During training, records the just-finished episode's reward, increments the training episode counter,
        and either starts the next training episode or switches to evaluation mode (resetting the episode index)
        when the configured number of training episodes is reached. During evaluation, increments the evaluation
        episode counter and either starts the next evaluation episode or marks the run as done and pauses the GUI
        when the configured number of evaluation episodes is reached.
        """
        if self.phase == "training":
            self.training_rewards.append(self.episode_reward)
            self.current_episode += 1
            if self.current_episode >= self.total_train_episodes:
                self.phase = "evaluation"
                self.current_episode = 0
            self._start_episode()
        elif self.phase == "evaluation":
            self.current_episode += 1
            if self.current_episode >= self.total_eval_episodes:
                self.phase = "done"
                self.paused = True
            else:
                self._start_episode()

    def _skip_training(self) -> None:
        """
        Run the remainder of the training phase as fast as possible without rendering.

        This repeatedly advances the environment by performing steps until each training episode finishes,
        advances episode bookkeeping after each episode, and when all training episodes complete it starts
        the next episode (evaluation or a new training episode depending on the phase).
        """
        while self.phase == "training":
            while not self.episode_done:
                self._do_step()
            self._advance_episode()

    def skip_training(self) -> None:
        self._skip_training()

    def _save_brain(self, directory: str | Path | None = None) -> None:
        """
        Save the current brain to disk and display a toast message indicating success or failure.

        If `directory` is provided, it is used as the target path; otherwise `DEFAULT_BRAIN_DIR` is used. On success a toast showing the save path is displayed. Expected filesystem errors are reported via toast; unexpected exceptions are allowed to propagate.

        Parameters:
            directory (str | Path | None): Optional directory or path to save the brain. If `None`, the default brain directory is used.
        """
        path = Path(directory) if directory else Path(DEFAULT_BRAIN_DIR)
        try:
            self.brain.save(path)
            self._show_toast(f"Brain saved to {path}/", is_error=False)
        except (OSError, json.JSONDecodeError, zipfile.BadZipFile) as exc:
            self._show_toast(f"Save error: {exc}", is_error=True)

    def save_brain(self, directory: str | Path | None = None) -> None:
        self._save_brain(directory)

    def _load_brain(self, directory: str | Path | None = None, modules: Sequence[str] | None = None) -> None:
        """
        Load a saved brain state into the simulation and show a toast reporting the outcome.

        Parameters:
            directory (str | Path | None): Filesystem path or directory containing the saved brain. If None, uses the default brain directory.
            modules (Sequence[str] | None): Optional list of module names to load from the saved brain; if None, loads all available modules.

        Notes:
            On success displays a toast with the list of loaded modules. Expected load failures are displayed via toast; unexpected exceptions are allowed to propagate.
        """
        path = Path(directory) if directory else Path(DEFAULT_BRAIN_DIR)
        try:
            loaded = self.brain.load(path, modules=modules)
            self._show_toast(f"Loaded: {', '.join(loaded)}", is_error=False)
        except (
            FileNotFoundError,
            PermissionError,
            OSError,
            json.JSONDecodeError,
            zipfile.BadZipFile,
            KeyError,
            ValueError,
        ) as exc:
            self._show_toast(f"Load error: {exc}", is_error=True)

    def load_brain(
        self,
        directory: str | Path | None = None,
        modules: Sequence[str] | None = None,
    ) -> None:
        self._load_brain(directory, modules=modules)

    def _show_toast(self, text: str, duration: float = 3.5, *, is_error: bool = False) -> None:
        """
        Display a transient toast notification in the GUI.

        Parameters:
            text (str): Message to display in the toast.
            duration (float): Time in seconds the toast remains visible (default 3.5).
            is_error (bool): Whether the toast should render with error styling.
        """
        self.toast_text = text
        self.toast_timer = duration
        self.toast_is_error = is_error

    def show_toast(
        self,
        text: str,
        duration: float = 3.5,
        *,
        is_error: bool = False,
    ) -> None:
        self._show_toast(text, duration, is_error=is_error)

    def tick(self, dt: float) -> None:
        self.update_toast(dt)
        if not self.paused or self.step_requested:
            if self.phase == "done":
                return
            if self.episode_done:
                self._advance_episode()
                return
            if self.step_requested:
                self._do_step()
                self.step_requested = False
                return
            self.tick_timer += dt
            interval = 1.0 / TICK_SPEEDS[self.speed_idx]
            while self.tick_timer >= interval and not self.episode_done:
                self._do_step()
                self.tick_timer -= interval

    def update_toast(self, dt: float) -> None:
        if self.toast_timer > 0:
            self.toast_timer -= dt

    def request_quit(self) -> None:
        self.running = False

    def toggle_pause(self) -> None:
        self.paused = not self.paused

    def request_step(self) -> None:
        self.step_requested = True
        self.paused = True

    def increase_speed(self) -> None:
        self.speed_idx = min(self.speed_idx + 1, len(TICK_SPEEDS) - 1)

    def decrease_speed(self) -> None:
        self.speed_idx = max(self.speed_idx - 1, 0)

    def toggle_visibility_overlay(self) -> None:
        self.show_visibility_overlay = not self.show_visibility_overlay

    def toggle_smell_overlay(self) -> None:
        self.show_smell_overlay = not self.show_smell_overlay

    def scroll_panel(self, delta: int) -> None:
        grid_h = self.world.height * CELL_SIZE
        max_scroll = max(0, self.panel_content_height - grid_h)
        self.panel_scroll = max(0, min(self.panel_scroll + delta, max_scroll))

    def _restart(self) -> None:
        """
        Replaces the simulation's brain with a fresh instance using the same hyperparameters and restarts training from episode zero.

        Resets phase to "training", sets the current episode index to 0, clears accumulated training rewards, and starts a new episode with the freshly constructed brain.
        """
        self.sim.brain = type(self.sim.brain)(
            seed=self.sim.seed,
            gamma=self.brain.gamma,
            module_lr=self.brain.module_lr,
            motor_lr=self.brain.motor_lr,
            module_dropout=self.brain.module_dropout,
            config=self.brain.config,
            operational_profile=self.sim.operational_profile,
        )
        self.brain = self.sim.brain
        self.phase = "training"
        self.current_episode = 0
        self.training_rewards.clear()
        self._start_episode()

    def restart(self) -> None:
        self._restart()

    # -----------------------------------------------------------------------
    # Drawing
    # -----------------------------------------------------------------------
