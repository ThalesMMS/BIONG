from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ..agent import BrainStep
from ..metrics import EpisodeMetricAccumulator
from ..predator import PREDATOR_STATES
from ..simulation import SpiderSimulation
from ..world import REWARD_COMPONENT_NAMES
from .constants import (
    BOTTOM_BAR_HEIGHT,
    CELL_SIZE,
    DEFAULT_BRAIN_DIR,
    DEFAULT_EVOLUTION_SNAPSHOT_DIR,
    DEFAULT_SPEED_IDX,
    LEFT_SIDEBAR_WIDTH,
    PANEL_WIDTH,
    TICK_SPEEDS,
    TOP_BAR_HEIGHT,
)
from .models import (
    GUI_MODEL_SPECS,
    GUIRunConfig,
    GUIRuntimeAdapter,
    adapter_for_existing_simulation,
    build_runtime_adapter,
)


class GUIController:
    def __init__(
        self,
        sim: SpiderSimulation | None = None,
        *,
        run_config: GUIRunConfig | None = None,
        model_id: str = "modular_full",
    ) -> None:
        if sim is None and run_config is None:
            run_config = GUIRunConfig()
        self.run_config = (
            run_config if run_config is not None else GUIRunConfig.from_simulation(sim)
        )
        self.runtime: GUIRuntimeAdapter = (
            adapter_for_existing_simulation(sim, model_id=model_id)
            if sim is not None
            else build_runtime_adapter(self.run_config, model_id)
        )
        self.available_model_specs = GUI_MODEL_SPECS
        self.active_model = self.runtime.spec
        self.sim = self.runtime.sim
        self.world = self.runtime.world
        self.brain = self.runtime.brain
        self.bus = self.runtime.bus

        self.cell_size = CELL_SIZE
        self.left_sidebar_width = LEFT_SIDEBAR_WIDTH
        self.panel_width = PANEL_WIDTH

        grid_w = self.world.width * self.cell_size
        self.win_w = self.left_sidebar_width + grid_w + self.panel_width
        self.win_h = TOP_BAR_HEIGHT + (self.world.height * self.cell_size) + BOTTOM_BAR_HEIGHT

        self.grid_offset_x = self.left_sidebar_width
        self.grid_offset_y = TOP_BAR_HEIGHT
        self.panel_x = self.left_sidebar_width + grid_w
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
        self.max_steps = self.runtime.max_steps

        self.current_step = 0
        self.episode_reward = 0.0
        self.episode_done = False
        self.observation: Optional[Dict] = None
        self.last_decision: Optional[BrainStep | Any] = None
        self.last_info: Optional[Dict] = None
        self.last_reward = 0.0
        self.reward_history: List[float] = []
        self.training_rewards: List[float] = []
        self.episode_metrics = EpisodeMetricAccumulator(REWARD_COMPONENT_NAMES, PREDATOR_STATES)
        self.show_visibility_overlay = False
        self.show_smell_overlay = False
        self.panel_scroll = 0
        self.panel_content_height = 0
        self.ui_scale = 1.0
        self.last_evolution_snapshot_path: str | None = None

        # Status toast
        self.toast_text: str = ""
        self.toast_timer: float = 0.0
        self.toast_is_error: bool = False

        # Window resize (handled by GUI/renderer)
        self.resize_requested = False
        self.requested_win_size: tuple[int, int] | None = None

    def request_resize(self, win_w: int, win_h: int) -> None:
        self.resize_requested = True
        self.requested_win_size = (win_w, win_h)

    def compute_fit_cell_size(
        self,
        win_w: int,
        win_h: int,
        *,
        panel_width: int,
        min_cell: int = 8,
        max_cell: int = CELL_SIZE,
    ) -> int:
        available_w = max(1, win_w - panel_width - self.left_sidebar_width)
        available_h = max(1, win_h - TOP_BAR_HEIGHT - BOTTOM_BAR_HEIGHT)

        fit_w = available_w // max(1, self.world.width)
        fit_h = available_h // max(1, self.world.height)
        fit = min(fit_w, fit_h)

        if fit < min_cell:
            return min_cell
        if fit > max_cell:
            return max_cell
        return int(fit)

    def apply_window_size(self, win_w: int, win_h: int) -> None:
        self.win_w = win_w
        self.win_h = win_h

        min_panel = 240
        max_panel = PANEL_WIDTH
        self.panel_width = max(min_panel, min(max_panel, win_w // 3))

        self.cell_size = self.compute_fit_cell_size(win_w, win_h, panel_width=self.panel_width)
        self.ui_scale = max(0.85, min(1.4, self.cell_size / CELL_SIZE))

        grid_w = self.world.width * self.cell_size
        self.grid_offset_x = self.left_sidebar_width
        self.grid_offset_y = TOP_BAR_HEIGHT
        self.panel_x = self.left_sidebar_width + grid_w
        self.panel_y = TOP_BAR_HEIGHT

    def consume_resize_request(self) -> tuple[int, int] | None:
        if not self.resize_requested:
            return None
        self.resize_requested = False
        return self.requested_win_size

    def _bind_runtime(self, runtime: GUIRuntimeAdapter) -> None:
        self.runtime = runtime
        self.active_model = runtime.spec
        self.sim = runtime.sim
        self.world = runtime.world
        self.brain = runtime.brain
        self.bus = runtime.bus
        self.max_steps = runtime.max_steps
        self.panel_scroll = 0
        self.panel_content_height = 0

    def apply_model(self, model_id: str) -> None:
        runtime = build_runtime_adapter(self.run_config, model_id)
        self._bind_runtime(runtime)
        self.phase = "training"
        self.current_episode = 0
        self.training_rewards.clear()
        self.reward_history.clear()
        self.last_decision = None
        self.last_info = None
        self.last_reward = 0.0
        self.episode_reward = 0.0
        self.episode_done = False
        self.tick_timer = 0.0
        self.step_requested = False
        self._start_episode()
        self._show_toast(
            f"Model reset: {self.active_model.label}",
            is_error=False,
        )

    def model_audit_fields(self) -> dict[str, object]:
        return self.runtime.audit_fields(self.last_decision)

    def configure_run(self, train_episodes: int, eval_episodes: int) -> None:
        self.total_train_episodes = train_episodes
        self.total_eval_episodes = eval_episodes
        self.phase = "training"
        self.current_episode = 0
        self._start_episode()

    def _start_episode(self) -> None:
        is_training = self.phase == "training"
        if is_training:
            episode_seed = self.runtime.seed + 997 * (self.current_episode + 1)
        else:
            episode_seed = self.runtime.seed + 997 * (
                self.total_train_episodes + self.current_episode + 1
            )

        self.observation = dict(self.runtime.reset(seed=episode_seed))
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

        decision = self.runtime.act(self.observation, sample=is_training)
        predator_state_before = (
            self.world.lizard.mode if self.runtime.supports_current_metrics else None
        )
        next_obs, reward, done, info = self.runtime.step(decision.action_idx)

        if is_training:
            terminal = done or (self.current_step + 1) >= self.max_steps
            learn_stats = self.runtime.learn(decision, reward, next_obs, terminal)
            self.bus.publish(sender="learning", topic="td_update", payload=learn_stats)

        self.last_decision = decision
        self.last_info = info
        self.last_reward = reward
        self.episode_reward += reward
        self.reward_history.append(reward)
        if self.runtime.supports_current_metrics:
            self.episode_metrics.record_transition(
                step=self.current_step,
                observation_meta=self.observation["meta"],
                next_meta=next_obs["meta"],
                info=info,
                state=self.world.state,
                predator_state_before=predator_state_before,
                predator_state=self.world.lizard.mode,
            )
        self.observation = dict(next_obs)
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
            if not self.runtime.supports_current_metrics:
                self._show_toast(
                    "Use Save Evolution Source for B0 legacy.",
                    is_error=True,
                )
                return
            self.brain.save(path)
            self._show_toast(f"Brain saved to {path}/", is_error=False)
        except (OSError, json.JSONDecodeError, zipfile.BadZipFile) as exc:
            self._show_toast(f"Save error: {exc}", is_error=True)

    def save_brain(self, directory: str | Path | None = None) -> None:
        self._save_brain(directory)

    def _controller_snapshot_state(self) -> dict[str, object]:
        return {
            "phase": self.phase,
            "current_episode": int(self.current_episode),
            "total_train_episodes": int(self.total_train_episodes),
            "total_eval_episodes": int(self.total_eval_episodes),
            "current_step": int(self.current_step),
            "episode_reward": float(self.episode_reward),
            "last_reward": float(self.last_reward),
            "last_action_idx": (
                None
                if self.last_decision is None
                else int(self.last_decision.action_idx)
            ),
            "audit": self.model_audit_fields(),
        }

    def save_evolution_snapshot(
        self,
        directory: str | Path | None = None,
    ) -> Path | None:
        base_dir = Path(directory) if directory is not None else Path(
            DEFAULT_EVOLUTION_SNAPSHOT_DIR
        )
        try:
            path = self.runtime.save_evolution_snapshot(
                base_dir,
                controller_state=self._controller_snapshot_state(),
            )
        except (OSError, json.JSONDecodeError, zipfile.BadZipFile, ValueError) as exc:
            self._show_toast(f"Evolution save error: {exc}", is_error=True)
            return None
        self.last_evolution_snapshot_path = str(path)
        self._show_toast(f"Evolution source saved: {path}", is_error=False)
        return path

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
            if not self.runtime.supports_current_metrics:
                self._show_toast("Load is not supported for B0 legacy.", is_error=True)
                return
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
        grid_h = self.world.height * self.cell_size
        max_scroll = max(0, self.panel_content_height - grid_h)
        self.panel_scroll = max(0, min(self.panel_scroll + delta, max_scroll))

    def _restart(self) -> None:
        """
        Replaces the simulation's brain with a fresh instance using the same hyperparameters and restarts training from episode zero.

        Resets phase to "training", sets the current episode index to 0, clears accumulated training rewards, and starts a new episode with the freshly constructed brain.
        """
        runtime = build_runtime_adapter(self.run_config, self.active_model.id)
        self._bind_runtime(runtime)
        self.phase = "training"
        self.current_episode = 0
        self.training_rewards.clear()
        self._start_episode()
        self._show_toast(f"Restarted: {self.active_model.label}", is_error=False)

    def restart(self) -> None:
        self._restart()

    # -----------------------------------------------------------------------
    # Drawing
    # -----------------------------------------------------------------------
