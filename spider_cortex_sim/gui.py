"""Pygame graphical interface for visualizing the neuro-modular spider simulation."""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import pygame
except ImportError:
    print(
        "pygame is required for the graphical interface.\n"
        "Install it with:  pip install pygame"
    )
    sys.exit(1)

from .agent import BrainStep
from .bus import MessageBus
from .maps import BLOCKED, CLUTTER, NARROW
from .metrics import EpisodeMetricAccumulator
from .noise import NoiseConfig
from .predator import PREDATOR_STATES
from .simulation import SpiderSimulation
from .world import ACTIONS, REWARD_COMPONENT_NAMES, SpiderWorld

# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------
COLOR_BG_DAY = (200, 220, 180)
COLOR_BG_NIGHT = (30, 30, 60)
COLOR_GRID = (160, 175, 140)
COLOR_GRID_NIGHT = (50, 50, 80)
COLOR_GROUND = (225, 235, 210)
COLOR_GROUND_NIGHT = (40, 40, 70)
COLOR_SHELTER = (160, 130, 90)
COLOR_SHELTER_NIGHT = (90, 70, 50)
COLOR_SHELTER_ENTRANCE = (176, 138, 92)
COLOR_SHELTER_INTERIOR = (144, 110, 76)
COLOR_SHELTER_DEEP = (112, 84, 60)
COLOR_FOOD = (200, 60, 60)
COLOR_LIZARD_BODY = (60, 140, 70)
COLOR_LIZARD_EYE = (255, 240, 100)
COLOR_SPIDER_BODY = (50, 50, 50)
COLOR_SPIDER_EYES = (220, 30, 30)
COLOR_SPIDER_LEGS = (70, 70, 70)
COLOR_BLOCKED = (70, 72, 80)
COLOR_CLUTTER = (152, 168, 132)
COLOR_CLUTTER_NIGHT = (64, 78, 74)
COLOR_NARROW = (188, 170, 120)
COLOR_NARROW_NIGHT = (104, 92, 66)

COLOR_PANEL_BG = (30, 30, 35)
COLOR_PANEL_BORDER = (60, 60, 70)
COLOR_TEXT = (220, 220, 220)
COLOR_TEXT_DIM = (140, 140, 150)
COLOR_TEXT_TITLE = (255, 220, 100)

COLOR_BAR_HUNGER = (220, 140, 40)
COLOR_BAR_FATIGUE = (100, 140, 220)
COLOR_BAR_HEALTH = (80, 200, 100)
COLOR_BAR_BG = (55, 55, 60)

COLOR_ACTION_ACTIVE = (100, 220, 130)
COLOR_ACTION_INACTIVE = (80, 80, 90)

COLOR_MODULE_COLORS = [
    (130, 180, 255),   # visual_cortex
    (255, 180, 100),   # sensory_cortex
    (255, 120, 120),   # hunger_center
    (120, 160, 255),   # sleep_center
    (255, 220, 80),    # alert_center
    (160, 255, 160),   # motor_cortex
]

COLOR_REWARD_POS = (80, 220, 120)
COLOR_REWARD_NEG = (220, 80, 80)

COLOR_BUTTON = (60, 60, 70)
COLOR_BUTTON_HOVER = (80, 80, 95)
COLOR_BUTTON_TEXT = (200, 200, 210)

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------
CELL_SIZE = 56
PANEL_WIDTH = 420
TOP_BAR_HEIGHT = 48
BOTTOM_BAR_HEIGHT = 56

DEFAULT_BRAIN_DIR = "spider_brain"
TICK_SPEEDS = [2, 4, 8, 15, 30, 60]
DEFAULT_SPEED_IDX = 2


def _lerp_color(c1: Tuple[int, ...], c2: Tuple[int, ...], t: float) -> Tuple[int, ...]:
    return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))


class Button:
    """Simple button for the bottom bar."""

    def __init__(self, text: str, x: int, y: int, w: int, h: int) -> None:
        """
        Initialize a Button with label text and rectangular position/size.
        
        Parameters:
            text (str): Label displayed on the button.
            x (int): X-coordinate of the button's top-left corner.
            y (int): Y-coordinate of the button's top-left corner.
            w (int): Width of the button in pixels.
            h (int): Height of the button in pixels.
        """
        self.text = text
        self.rect = pygame.Rect(x, y, w, h)
        self.hovered = False

    def draw(self, surface: pygame.Surface, font: pygame.font.Font) -> None:
        color = COLOR_BUTTON_HOVER if self.hovered else COLOR_BUTTON
        pygame.draw.rect(surface, color, self.rect, border_radius=6)
        pygame.draw.rect(surface, COLOR_PANEL_BORDER, self.rect, width=1, border_radius=6)
        label = font.render(self.text, True, COLOR_BUTTON_TEXT)
        lx = self.rect.x + (self.rect.width - label.get_width()) // 2
        ly = self.rect.y + (self.rect.height - label.get_height()) // 2
        surface.blit(label, (lx, ly))

    def update_hover(self, mouse_pos: Tuple[int, int]) -> None:
        self.hovered = self.rect.collidepoint(mouse_pos)


class SpiderGUI:
    """Main simulation viewer window."""

    def __init__(self, sim: SpiderSimulation) -> None:
        """
        Initialize the GUI for a SpiderSimulation and set up Pygame state, layout, runtime state, and UI controls.
        
        Initializes Pygame, creates the display surface and clock, prepares fonts, computes grid and panel layout from the simulation world, and initializes runtime bookkeeping used by the GUI (phase/episode/step counters, toggles, toast state, and reward/metric accumulators). Also constructs the bottom-bar control buttons and stores references to the simulation objects (world, brain, bus).
        
        Parameters:
            sim (SpiderSimulation): The simulation instance whose world, brain, and bus the GUI will visualize and control.
        """
        self.sim = sim
        self.world = sim.world
        self.brain = sim.brain
        self.bus = sim.bus

        grid_w = self.world.width * CELL_SIZE
        grid_h = self.world.height * CELL_SIZE
        self.win_w = grid_w + PANEL_WIDTH
        self.win_h = TOP_BAR_HEIGHT + grid_h + BOTTOM_BAR_HEIGHT

        pygame.init()
        self.screen = pygame.display.set_mode((self.win_w, self.win_h))
        pygame.display.set_caption("Neuro-Modular Spider Simulation")
        self.clock = pygame.time.Clock()

        self.font_sm = pygame.font.SysFont("monospace", 13)
        self.font_md = pygame.font.SysFont("monospace", 15, bold=True)
        self.font_lg = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_xl = pygame.font.SysFont("monospace", 22, bold=True)

        self.grid_offset_x = 0
        self.grid_offset_y = TOP_BAR_HEIGHT
        self.panel_x = grid_w
        self.panel_y = TOP_BAR_HEIGHT

        # State
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

        # Buttons
        btn_y = TOP_BAR_HEIGHT + grid_h + 4
        btn_h = 24
        self.btn_pause = Button("⏸ Pause", 10, btn_y, 100, btn_h)
        self.btn_step = Button("⏭ Step", 116, btn_y, 90, btn_h)
        self.btn_slower = Button("◀ Slower", 212, btn_y, 90, btn_h)
        self.btn_faster = Button("Faster ▶", 308, btn_y, 90, btn_h)
        self.btn_restart = Button("↻ Restart", 404, btn_y, 100, btn_h)
        self.btn_save = Button("💾 Save", 510, btn_y, 96, btn_h)
        self.btn_load = Button("📂 Load", 612, btn_y, 100, btn_h)
        self.buttons = [
            self.btn_pause, self.btn_step, self.btn_slower, self.btn_faster,
            self.btn_restart, self.btn_save, self.btn_load,
        ]

    def launch(self, train_episodes: int, eval_episodes: int) -> None:
        """
        Start the GUI-driven training and evaluation run and enter the main event loop.
        
        Parameters:
            train_episodes (int): Number of training episodes to run before switching to evaluation.
            eval_episodes (int): Number of evaluation episodes to run after training.
        """
        self.total_train_episodes = train_episodes
        self.total_eval_episodes = eval_episodes
        self.phase = "training"
        self.current_episode = 0
        self._start_episode()
        self._main_loop()

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
            learn_stats = self.brain.learn(decision, reward, next_obs, done)
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
        self._start_episode()

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
        except OSError as exc:
            self._show_toast(f"Save error: {exc}", is_error=True)

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

    def _main_loop(self) -> None:
        while self.running:
            dt = self.clock.tick(60) / 1000.0
            self._handle_events()

            if self.toast_timer > 0:
                self.toast_timer -= dt

            if not self.paused or self.step_requested:
                if self.phase == "done":
                    pass
                elif self.episode_done:
                    self._advance_episode()
                else:
                    if self.step_requested:
                        self._do_step()
                        self.step_requested = False
                    else:
                        self.tick_timer += dt
                        interval = 1.0 / TICK_SPEEDS[self.speed_idx]
                        while self.tick_timer >= interval and not self.episode_done:
                            self._do_step()
                            self.tick_timer -= interval

            self._draw()
            pygame.display.flip()

        pygame.quit()

    def _handle_events(self) -> None:
        mouse_pos = pygame.mouse.get_pos()
        for btn in self.buttons:
            btn.update_hover(mouse_pos)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_RIGHT or event.key == pygame.K_n:
                    self.step_requested = True
                    self.paused = True
                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    self.speed_idx = min(self.speed_idx + 1, len(TICK_SPEEDS) - 1)
                elif event.key == pygame.K_MINUS:
                    self.speed_idx = max(self.speed_idx - 1, 0)
                elif event.key == pygame.K_s:
                    mods = pygame.key.get_mods()
                    if mods & pygame.KMOD_CTRL or mods & pygame.KMOD_META:
                        self._save_brain()
                    elif self.phase == "training":
                        self._skip_training()
                elif event.key == pygame.K_l:
                    mods = pygame.key.get_mods()
                    if mods & pygame.KMOD_CTRL or mods & pygame.KMOD_META:
                        self._load_brain()
                elif event.key == pygame.K_r:
                    self._restart()
                elif event.key == pygame.K_v:
                    self.show_visibility_overlay = not self.show_visibility_overlay
                elif event.key == pygame.K_m:
                    self.show_smell_overlay = not self.show_smell_overlay
                elif event.key == pygame.K_PAGEDOWN:
                    self._scroll_panel(80)
                elif event.key == pygame.K_PAGEUP:
                    self._scroll_panel(-80)
            elif event.type == pygame.MOUSEWHEEL:
                # Scroll panel when mouse is over the panel area
                mx, _ = pygame.mouse.get_pos()
                if mx >= self.panel_x:
                    self._scroll_panel(-event.y * 30)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self.btn_pause.rect.collidepoint(event.pos):
                    self.paused = not self.paused
                elif self.btn_step.rect.collidepoint(event.pos):
                    self.step_requested = True
                    self.paused = True
                elif self.btn_slower.rect.collidepoint(event.pos):
                    self.speed_idx = max(self.speed_idx - 1, 0)
                elif self.btn_faster.rect.collidepoint(event.pos):
                    self.speed_idx = min(self.speed_idx + 1, len(TICK_SPEEDS) - 1)
                elif self.btn_restart.rect.collidepoint(event.pos):
                    self._restart()
                elif self.btn_save.rect.collidepoint(event.pos):
                    self._save_brain()
                elif self.btn_load.rect.collidepoint(event.pos):
                    self._load_brain()

    def _scroll_panel(self, delta: int) -> None:
        """Scroll the panel content by *delta* pixels, clamping to valid range."""
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

    # -----------------------------------------------------------------------
    # Drawing
    # -----------------------------------------------------------------------
    def _draw(self) -> None:
        """
        Render the entire GUI frame on the main screen surface.
        
        Clears the screen background and draws the top bar, world grid, right-side panel, and bottom control bar in that order.
        """
        self.screen.fill(COLOR_PANEL_BG)
        self._draw_top_bar()
        self._draw_grid()
        self._draw_panel()
        self._draw_bottom_bar()

    def _night_t(self) -> float:
        return 1.0 if self.world.is_night() else 0.0

    def _draw_top_bar(self) -> None:
        """
        Draw the GUI's top information bar showing the simulation phase, episode progress, current tick speed, and run/pause state.
        
        Renders a background rectangle across the top of the window and places:
        - a phase label ("TRAINING", "EVALUATION", or "DONE") with episode progress when applicable,
        - the current simulation speed in ticks per second,
        - the current run state ("PAUSED" or "RUNNING").
        """
        rect = pygame.Rect(0, 0, self.win_w, TOP_BAR_HEIGHT)
        pygame.draw.rect(self.screen, (40, 40, 48), rect)

        # Phase and episode
        if self.phase == "training":
            phase_text = f"TRAINING  ep {self.current_episode + 1}/{self.total_train_episodes}"
        elif self.phase == "evaluation":
            phase_text = f"EVALUATION  ep {self.current_episode + 1}/{self.total_eval_episodes}"
        else:
            phase_text = "DONE"
        label = self.font_lg.render(phase_text, True, COLOR_TEXT_TITLE)
        self.screen.blit(label, (12, 14))

        # Speed
        speed_text = f"Speed: {TICK_SPEEDS[self.speed_idx]} t/s"
        sl = self.font_sm.render(speed_text, True, COLOR_TEXT_DIM)
        self.screen.blit(sl, (self.win_w - 160, 8))

        state_text = "⏸ PAUSED" if self.paused else "▶ RUNNING"
        st = self.font_sm.render(state_text, True, COLOR_TEXT)
        self.screen.blit(st, (self.win_w - 160, 28))

    def _draw_grid(self) -> None:
        """
        Draws the simulation grid and all related visual overlays onto the GUI screen.
        
        Renders each world cell (terrain, clutter, narrow passages, shelter variants) with day/night color interpolation; draws shelter roofs and doors, grid lines, smell and visibility overlays when enabled, food items, the predator (lizard) and the spider, and a soft nighttime tint over the entire grid proportional to night intensity.
        """
        nt = self._night_t()
        bg = _lerp_color(COLOR_BG_DAY, COLOR_BG_NIGHT, nt)
        ground = _lerp_color(COLOR_GROUND, COLOR_GROUND_NIGHT, nt)
        shelter_c = _lerp_color(COLOR_SHELTER, COLOR_SHELTER_NIGHT, nt)
        grid_c = _lerp_color(COLOR_GRID, COLOR_GRID_NIGHT, nt)

        grid_w = self.world.width * CELL_SIZE
        grid_h = self.world.height * CELL_SIZE
        grid_rect = pygame.Rect(self.grid_offset_x, self.grid_offset_y, grid_w, grid_h)
        pygame.draw.rect(self.screen, bg, grid_rect)

        ox, oy = self.grid_offset_x, self.grid_offset_y

        # Cells
        for cy in range(self.world.height):
            for cx in range(self.world.width):
                rx = ox + cx * CELL_SIZE
                ry = oy + cy * CELL_SIZE
                cell_rect = pygame.Rect(rx, ry, CELL_SIZE, CELL_SIZE)
                pos = (cx, cy)
                terrain = self.world.terrain_at(pos)
                if terrain == BLOCKED:
                    pygame.draw.rect(self.screen, COLOR_BLOCKED, cell_rect)
                elif terrain == CLUTTER:
                    clutter_c = _lerp_color(COLOR_CLUTTER, COLOR_CLUTTER_NIGHT, nt)
                    pygame.draw.rect(self.screen, clutter_c, cell_rect)
                    for offset in (10, 26, 40):
                        pygame.draw.circle(self.screen, _lerp_color((96, 112, 80), (80, 96, 86), nt), (rx + offset, ry + 18 + (offset % 8)), 4)
                elif terrain == NARROW:
                    narrow_c = _lerp_color(COLOR_NARROW, COLOR_NARROW_NIGHT, nt)
                    pygame.draw.rect(self.screen, narrow_c, cell_rect)
                    pygame.draw.rect(
                        self.screen,
                        _lerp_color((122, 108, 72), (88, 78, 54), nt),
                        pygame.Rect(rx + 10, ry + CELL_SIZE // 2 - 4, CELL_SIZE - 20, 8),
                        border_radius=4,
                    )
                elif pos in self.world.shelter_deep_cells:
                    pygame.draw.rect(self.screen, _lerp_color(COLOR_SHELTER_DEEP, COLOR_SHELTER_NIGHT, nt), cell_rect)
                elif pos in self.world.shelter_interior_cells:
                    pygame.draw.rect(self.screen, _lerp_color(COLOR_SHELTER_INTERIOR, COLOR_SHELTER_NIGHT, nt), cell_rect)
                elif pos in self.world.shelter_entrance_cells:
                    pygame.draw.rect(self.screen, _lerp_color(COLOR_SHELTER_ENTRANCE, COLOR_SHELTER_NIGHT, nt), cell_rect)
                else:
                    pygame.draw.rect(self.screen, ground, cell_rect)

                if pos in self.world.shelter_cells:
                    roof_color = _lerp_color((120, 90, 60), (70, 50, 35), nt)
                    pygame.draw.polygon(self.screen, roof_color, [
                        (rx + CELL_SIZE // 2, ry + 4),
                        (rx + 6, ry + CELL_SIZE // 2 - 2),
                        (rx + CELL_SIZE - 6, ry + CELL_SIZE // 2 - 2),
                    ])
                    if pos in self.world.shelter_entrance_cells:
                        door_color = _lerp_color((90, 60, 30), (50, 35, 20), nt)
                        door_rect = pygame.Rect(rx + CELL_SIZE // 2 - 5, ry + CELL_SIZE // 2, 10, CELL_SIZE // 2 - 4)
                        pygame.draw.rect(self.screen, door_color, door_rect, border_radius=2)

                pygame.draw.rect(self.screen, grid_c, cell_rect, width=1)

        if self.show_smell_overlay:
            food_smell = self.world.smell_field("food")
            predator_smell = self.world.smell_field("predator")
            for cy in range(self.world.height):
                for cx in range(self.world.width):
                    fx = food_smell[cy][cx]
                    px = predator_smell[cy][cx]
                    if fx <= 0.01 and px <= 0.01:
                        continue
                    overlay = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
                    overlay.fill((int(220 * fx), 60, int(220 * px), int(90 * max(fx, px))))
                    self.screen.blit(overlay, (ox + cx * CELL_SIZE, oy + cy * CELL_SIZE))

        if self.show_visibility_overlay:
            visibility = self.world.visibility_overlay()
            for cx, cy in visibility["occluded"]:
                overlay = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
                overlay.fill((220, 180, 60, 48))
                self.screen.blit(overlay, (ox + cx * CELL_SIZE, oy + cy * CELL_SIZE))
            for cx, cy in visibility["visible"]:
                overlay = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
                overlay.fill((80, 220, 120, 42))
                self.screen.blit(overlay, (ox + cx * CELL_SIZE, oy + cy * CELL_SIZE))

        # Food
        for fx, fy in self.world.food_positions:
            cx = ox + fx * CELL_SIZE + CELL_SIZE // 2
            cy_pos = oy + fy * CELL_SIZE + CELL_SIZE // 2
            # Fly or insect body
            pygame.draw.circle(self.screen, (80, 60, 40), (cx, cy_pos), 8)
            pygame.draw.circle(self.screen, COLOR_FOOD, (cx, cy_pos), 6)
            # Wings
            pygame.draw.ellipse(self.screen, (255, 200, 200, 180),
                                pygame.Rect(cx - 12, cy_pos - 8, 10, 6))
            pygame.draw.ellipse(self.screen, (255, 200, 200, 180),
                                pygame.Rect(cx + 2, cy_pos - 8, 10, 6))

        # Predator lizard
        self._draw_lizard(
            ox + self.world.lizard.x * CELL_SIZE + CELL_SIZE // 2,
            oy + self.world.lizard.y * CELL_SIZE + CELL_SIZE // 2,
        )

        # Spider
        self._draw_spider(
            ox + self.world.state.x * CELL_SIZE + CELL_SIZE // 2,
            oy + self.world.state.y * CELL_SIZE + CELL_SIZE // 2,
        )

        # Soft nighttime overlay
        if nt > 0:
            overlay = pygame.Surface((grid_w, grid_h), pygame.SRCALPHA)
            overlay.fill((0, 0, 30, int(60 * nt)))
            self.screen.blit(overlay, (ox, oy))


    def _draw_lizard(self, cx: int, cy: int) -> None:
        """
        Draws a simplified lizard icon centered at the given grid pixel coordinates.
        
        Parameters:
            cx (int): X coordinate of the lizard's center in pixels.
            cy (int): Y coordinate of the lizard's center in pixels.
        """
        tail = [(cx - 16, cy + 2), (cx - 26, cy + 6), (cx - 22, cy + 10)]
        pygame.draw.lines(self.screen, COLOR_LIZARD_BODY, False, tail, 4)
        body_rect = pygame.Rect(cx - 12, cy - 8, 24, 16)
        pygame.draw.ellipse(self.screen, COLOR_LIZARD_BODY, body_rect)
        head_rect = pygame.Rect(cx + 6, cy - 6, 14, 12)
        pygame.draw.ellipse(self.screen, (80, 170, 90), head_rect)
        for lx, ly in [(-8, -8), (-8, 8), (4, -8), (4, 8)]:
            pygame.draw.line(self.screen, COLOR_LIZARD_BODY, (cx + lx, cy), (cx + lx - 8, cy + ly), 2)
        pygame.draw.circle(self.screen, COLOR_LIZARD_EYE, (cx + 12, cy - 2), 2)

    def _draw_spider(self, cx: int, cy: int) -> None:
        # Legs
        """
        Draws a stylized spider centered at the given screen coordinates.
        
        Parameters:
            cx (int): X coordinate of the spider center (pixels).
            cy (int): Y coordinate of the spider center (pixels).
        """
        leg_offsets = [
            (-14, -10), (-16, -3), (-15, 5), (-12, 12),
            (14, -10), (16, -3), (15, 5), (12, 12),
        ]
        for lx, ly in leg_offsets:
            pygame.draw.line(self.screen, COLOR_SPIDER_LEGS, (cx, cy), (cx + lx, cy + ly), 2)

        # Body (abdomen + cephalothorax)
        pygame.draw.circle(self.screen, COLOR_SPIDER_BODY, (cx, cy + 4), 9)
        pygame.draw.circle(self.screen, (60, 60, 60), (cx, cy - 4), 7)

        # Eyes
        pygame.draw.circle(self.screen, COLOR_SPIDER_EYES, (cx - 3, cy - 7), 2)
        pygame.draw.circle(self.screen, COLOR_SPIDER_EYES, (cx + 3, cy - 7), 2)

        # Eye highlights
        pygame.draw.circle(self.screen, (255, 255, 255), (cx - 2, cy - 8), 1)
        pygame.draw.circle(self.screen, (255, 255, 255), (cx + 4, cy - 8), 1)

    def _draw_panel(self) -> None:
        """
        Render the right-side information panel with the spider's state, memory, diagnostics, and metrics.
        
        Displays labeled sections including:
        - SPIDER STATE: tick, day/night, position, map/profile, shelter role, status bars (hunger, fatigue, sleep debt, health), counts and predator/lizard info, and overlay toggles.
        - MEMORY: stored targets and ages for food, predator, shelter, and escape; predator vision and memory vectors when available.
        - REWARD: last reward, episode total, top reward components, and a mini reward chart.
        - CURRENT ACTION: available actions, their probabilities, and the currently selected action.
        - CORTICAL MODULES: per-module name, top action and confidence, reflex info, and miniature probability bars.
        - PROGRESS: recent training reward mean (when training data exists).
        - METRICS: episode-level statistics such as night role distribution, predator latency and event counts, and predator state occupancy/tick counts.
        
        No value is returned.
        """
        grid_h = self.world.height * CELL_SIZE
        # Create an off-screen surface tall enough for all content
        panel_surf_h = max(grid_h, 1600)
        panel_surf = pygame.Surface((PANEL_WIDTH, panel_surf_h))
        panel_surf.fill(COLOR_PANEL_BG)

        x0 = 14
        y = 10
        pw = PANEL_WIDTH - 28
        metric_snapshot = self.episode_metrics.snapshot()
        # Temporarily redirect drawing to panel_surf
        _orig_screen = self.screen
        self.screen = panel_surf

        # --- Status ---
        y = self._draw_section_title("SPIDER STATE", x0, y)
        st = self.world.state
        phase = "NIGHT 🌙" if self.world.is_night() else "DAY ☀"
        y = self._draw_text(f"Tick: {self.world.tick}   Phase: {phase}", x0, y, COLOR_TEXT)
        y = self._draw_text(f"Position: ({st.x}, {st.y})", x0, y, COLOR_TEXT_DIM)
        y = self._draw_text(f"Map: {self.world.map_template_name}   Profile: {self.world.reward_profile}", x0, y, COLOR_TEXT_DIM)
        y = self._draw_text(f"Shelter: {self.world.shelter_role_at(self.world.spider_pos())}", x0, y, COLOR_TEXT_DIM)

        y += 4
        y = self._draw_bar("Hunger", st.hunger, x0, y, pw, COLOR_BAR_HUNGER)
        y = self._draw_bar("Fatigue", st.fatigue, x0, y, pw, COLOR_BAR_FATIGUE)
        y = self._draw_bar("Sleep debt", st.sleep_debt, x0, y, pw, (160, 120, 230))
        y = self._draw_bar("Health", st.health, x0, y, pw, COLOR_BAR_HEALTH)

        y += 4
        y = self._draw_text(f"Food: {st.food_eaten}   Sleep: {st.sleep_events}   Shelter entries: {st.shelter_entries}", x0, y, COLOR_TEXT_DIM)
        y = self._draw_text(f"Predator: contacts={st.predator_contacts}  sightings={st.predator_sightings}  escapes={st.predator_escapes}", x0, y, COLOR_TEXT_DIM)
        y = self._draw_text(f"Lizard: ({self.world.lizard.x}, {self.world.lizard.y})  mode={self.world.lizard.mode}", x0, y, COLOR_TEXT_DIM)
        y = self._draw_text(
            f"Lizard target: patrol={self.world.lizard.patrol_target} wait={self.world.lizard.wait_target}",
            x0,
            y,
            COLOR_TEXT_DIM,
        )
        y = self._draw_text(
            f"Overlays: vision={'on' if self.show_visibility_overlay else 'off'}  smell={'on' if self.show_smell_overlay else 'off'}",
            x0,
            y,
            COLOR_TEXT_DIM,
        )

        y += 6
        y = self._draw_section_title("MEMORY", x0, y)
        y = self._draw_text(f"Food: target={st.food_memory.target} age={st.food_memory.age}", x0, y, COLOR_TEXT_DIM)
        y = self._draw_text(f"Predator: target={st.predator_memory.target} age={st.predator_memory.age}", x0, y, COLOR_TEXT_DIM)
        y = self._draw_text(f"Shelter: target={st.shelter_memory.target} age={st.shelter_memory.age}", x0, y, COLOR_TEXT_DIM)
        y = self._draw_text(f"Escape: target={st.escape_memory.target} age={st.escape_memory.age}", x0, y, COLOR_TEXT_DIM)
        if self.observation is not None:
            vision = self.observation["meta"]["vision"]
            y = self._draw_text(
                f"Predator vision: c={vision['predator']['certainty']:.2f} occ={vision['predator']['occluded']:.0f}",
                x0,
                y,
                COLOR_TEXT_DIM,
            )
            y = self._draw_text(
                f"Vectors: food={self.observation['meta']['memory_vectors']['food']} shelter={self.observation['meta']['memory_vectors']['shelter']}",
                x0,
                y,
                COLOR_TEXT_DIM,
            )

        # --- Reward ---
        y += 6
        y = self._draw_section_title("REWARD", x0, y)
        rc = COLOR_REWARD_POS if self.last_reward >= 0 else COLOR_REWARD_NEG
        y = self._draw_text(f"Last: {self.last_reward:+.3f}   Total: {self.episode_reward:.2f}", x0, y, rc)
        if self.last_info is not None:
            components = sorted(
                self.last_info["reward_components"].items(),
                key=lambda item: abs(item[1]),
                reverse=True,
            )
            for name, value in components[:4]:
                if abs(value) < 1e-6:
                    continue
                color = COLOR_REWARD_POS if value >= 0.0 else COLOR_REWARD_NEG
                y = self._draw_text(f"{name}: {value:+.3f}", x0, y, color)

        # Mini reward chart
        y += 2
        y = self._draw_reward_chart(x0, y, pw, 40)

        # --- Current action ---
        y += 6
        y = self._draw_section_title("CURRENT ACTION", x0, y)
        if self.last_decision is not None:
            action_name = ACTIONS[self.last_decision.action_idx]
            for i, a in enumerate(ACTIONS):
                prob = self.last_decision.policy[i]
                is_selected = (a == action_name)
                color = COLOR_ACTION_ACTIVE if is_selected else COLOR_ACTION_INACTIVE
                marker = "▶" if is_selected else " "
                y = self._draw_text(f" {marker} {a:<18} {prob:.3f}", x0, y, color)
        else:
            y = self._draw_text("  (waiting...)", x0, y, COLOR_TEXT_DIM)

        # --- Neural modules ---
        y += 6
        y = self._draw_section_title("CORTICAL MODULES", x0, y)
        if self.last_decision is not None:
            for i, result in enumerate(self.last_decision.module_results):
                color = COLOR_MODULE_COLORS[i % len(COLOR_MODULE_COLORS)]
                best_action = ACTIONS[int(np.argmax(result.probs))]
                confidence = float(np.max(result.probs))
                short_name = result.name.replace("_", " ").title()
                y = self._draw_text(f"  {short_name}", x0, y, color)
                reflex_text = "no reflex"
                if result.reflex is not None:
                    reflex_text = f"{result.reflex.action} · {result.reflex.reason}"
                y = self._draw_text(
                    f"    top={best_action} ({confidence:.2f})  reflex={reflex_text}",
                    x0,
                    y,
                    COLOR_TEXT_DIM,
                )
                # Mini probability bars
                bar_y = y
                bar_h = 8
                for j, prob in enumerate(result.probs):
                    bx = x0 + 12 + j * (pw - 24) // len(ACTIONS)
                    bw = max(1, (pw - 24) // len(ACTIONS) - 2)
                    filled_h = int(bar_h * prob)
                    pygame.draw.rect(self.screen, (50, 50, 55), pygame.Rect(bx, bar_y, bw, bar_h))
                    pygame.draw.rect(self.screen, color, pygame.Rect(bx, bar_y + bar_h - filled_h, bw, filled_h))
                y = bar_y + bar_h + 4

        # --- Training progress ---
        if self.training_rewards:
            y += 4
            y = self._draw_section_title("PROGRESS", x0, y)
            window = self.training_rewards[-min(20, len(self.training_rewards)):]
            avg = sum(window) / len(window)
            y = self._draw_text(f"Mean over last {len(window)} eps: {avg:.2f}", x0, y, COLOR_TEXT_DIM)

        y += 6
        y = self._draw_section_title("METRICS", x0, y)
        role_dist = metric_snapshot["night_role_distribution"]
        y = self._draw_text(
            "Night: "
            f"out={role_dist['outside']:.2f} ent={role_dist['entrance']:.2f} "
            f"in={role_dist['inside']:.2f} deep={role_dist['deep']:.2f}",
            x0,
            y,
            COLOR_TEXT_DIM,
        )
        y = self._draw_text(
            f"Predator latency: {metric_snapshot['mean_predator_response_latency']:.2f}  events={metric_snapshot['predator_response_events']}",
            x0,
            y,
            COLOR_TEXT_DIM,
        )
        occupancy = metric_snapshot["predator_state_occupancy"]
        for state_name in PREDATOR_STATES:
            y = self._draw_text(
                f"{state_name:<12} occ={occupancy.get(state_name, 0.0):.2f} ticks={metric_snapshot['predator_state_ticks'].get(state_name, 0)}",
                x0,
                y,
                COLOR_TEXT_DIM,
            )

        # Restore screen and blit the scrolled panel surface
        self.screen = _orig_screen
        self.panel_content_height = y + 10
        # Clamp scroll
        max_scroll = max(0, self.panel_content_height - grid_h)
        self.panel_scroll = max(0, min(self.panel_scroll, max_scroll))
        # Draw panel background and border on real screen
        panel_rect = pygame.Rect(self.panel_x, self.panel_y, PANEL_WIDTH, grid_h)
        pygame.draw.rect(self.screen, COLOR_PANEL_BG, panel_rect)
        pygame.draw.line(self.screen, COLOR_PANEL_BORDER,
                         (self.panel_x, self.panel_y),
                         (self.panel_x, self.panel_y + grid_h))
        # Blit the visible portion of the panel surface
        source_rect = pygame.Rect(0, self.panel_scroll, PANEL_WIDTH, grid_h)
        self.screen.blit(panel_surf, (self.panel_x, self.panel_y), source_rect)
        # Scrollbar indicator
        if self.panel_content_height > grid_h:
            sb_x = self.panel_x + PANEL_WIDTH - 6
            sb_h = max(20, int(grid_h * grid_h / self.panel_content_height))
            sb_travel = grid_h - sb_h
            sb_y = self.panel_y + int(sb_travel * self.panel_scroll / max_scroll) if max_scroll > 0 else self.panel_y
            pygame.draw.rect(self.screen, (80, 80, 100), pygame.Rect(sb_x, sb_y, 4, sb_h), border_radius=2)

    def _draw_section_title(self, text: str, x: int, y: int) -> int:
        label = self.font_md.render(text, True, COLOR_TEXT_TITLE)
        self.screen.blit(label, (x, y))
        y += label.get_height() + 2
        pygame.draw.line(self.screen, COLOR_PANEL_BORDER, (x, y), (x + PANEL_WIDTH - 28, y))
        return y + 4

    def _draw_text(self, text: str, x: int, y: int, color: Tuple[int, ...]) -> int:
        label = self.font_sm.render(text, True, color)
        self.screen.blit(label, (x, y))
        return y + label.get_height() + 1

    def _draw_bar(self, name: str, value: float, x: int, y: int, w: int, color: Tuple[int, ...]) -> int:
        label = self.font_sm.render(f"{name}: {value:.2f}", True, COLOR_TEXT)
        self.screen.blit(label, (x, y))
        bar_x = x + 120
        bar_w = w - 120
        bar_h = 12
        pygame.draw.rect(self.screen, COLOR_BAR_BG, pygame.Rect(bar_x, y + 1, bar_w, bar_h), border_radius=3)
        fill_w = int(bar_w * min(max(value, 0), 1))
        if fill_w > 0:
            pygame.draw.rect(self.screen, color, pygame.Rect(bar_x, y + 1, fill_w, bar_h), border_radius=3)
        return y + bar_h + 5

    def _draw_reward_chart(self, x: int, y: int, w: int, h: int) -> int:
        """
        Draws a compact reward history chart at the given panel rectangle and returns the vertical position after the chart.
        
        Renders a bordered chart background, a horizontal zero line, and vertical bars for recent rewards. Positive rewards are drawn above the zero line using the positive reward color; negative rewards are drawn below using the negative reward color. Bars are normalized by the largest absolute visible reward and clipped to the chart height.
        
        Parameters:
            x (int): X coordinate of the chart's top-left corner.
            y (int): Y coordinate of the chart's top-left corner.
            w (int): Width of the chart in pixels.
            h (int): Height of the chart in pixels.
        
        Returns:
            int: The y coordinate immediately below the drawn chart (i.e., y + h).
        """
        pygame.draw.rect(self.screen, (40, 40, 45), pygame.Rect(x, y, w, h), border_radius=3)
        if len(self.reward_history) < 2:
            return y + h

        visible = self.reward_history[-w:]
        if not visible:
            return y + h

        max_abs = max(abs(v) for v in visible)
        if max_abs < 0.01:
            max_abs = 1.0
        mid_y = y + h // 2
        # Zero line
        pygame.draw.line(self.screen, COLOR_PANEL_BORDER, (x, mid_y), (x + w, mid_y))

        step_w = max(1, w / len(visible))
        for i, val in enumerate(visible):
            px = int(x + i * step_w)
            normalized = val / max_abs
            bar_h_px = int(abs(normalized) * (h // 2 - 2))
            if val >= 0:
                color = COLOR_REWARD_POS
                ry = mid_y - bar_h_px
            else:
                color = COLOR_REWARD_NEG
                ry = mid_y
            if bar_h_px > 0:
                pygame.draw.rect(self.screen, color, pygame.Rect(px, ry, max(1, int(step_w) - 1), bar_h_px))
        return y + h

    def _draw_bottom_bar(self) -> None:
        """
        Draw the bottom control bar including interactive buttons, shortcut hints, and a transient status toast.
        
        Updates the pause button label to reflect current pause state, renders all bottom-bar buttons, displays left and right-aligned shortcut/help text, and if a toast is active renders a centered status message using explicit success or error styling.
        """
        grid_h = self.world.height * CELL_SIZE
        bar_y = TOP_BAR_HEIGHT + grid_h
        rect = pygame.Rect(0, bar_y, self.win_w, BOTTOM_BAR_HEIGHT)
        pygame.draw.rect(self.screen, (40, 40, 48), rect)

        self.btn_pause.text = "▶ Run" if self.paused else "⏸ Pause"
        for btn in self.buttons:
            btn.draw(self.screen, self.font_sm)

        # Shortcuts
        help_text = "Space=pause  →/N=step  +/-=speed  V=vision  M=smell  S=skip training  R=restart  Q=quit"
        ht = self.font_sm.render(help_text, True, COLOR_TEXT_DIM)
        self.screen.blit(ht, (10, bar_y + 30))

        help2 = "Ctrl+S=save brain  Ctrl+L=load brain"
        ht2 = self.font_sm.render(help2, True, COLOR_TEXT_DIM)
        self.screen.blit(ht2, (self.win_w - ht2.get_width() - 10, bar_y + 30))

        # Status toast
        if self.toast_timer > 0 and self.toast_text:
            alpha = min(1.0, self.toast_timer / 0.5)
            toast_color = (220, 80, 80) if self.toast_is_error else (100, 200, 130)
            toast_surf = self.font_md.render(self.toast_text, True, toast_color)
            tx = (self.win_w - toast_surf.get_width()) // 2
            ty = bar_y - 22
            pygame.draw.rect(self.screen, (30, 30, 35),
                             pygame.Rect(tx - 8, ty - 2, toast_surf.get_width() + 16, toast_surf.get_height() + 4),
                             border_radius=4)
            self.screen.blit(toast_surf, (tx, ty))


def run_gui(
    *,
    episodes: int = 180,
    eval_episodes: int = 3,
    width: int = 12,
    height: int = 12,
    food_count: int = 4,
    day_length: int = 18,
    night_length: int = 12,
    max_steps: int = 120,
    seed: int = 7,
    gamma: float = 0.96,
    module_lr: float = 0.010,
    motor_lr: float = 0.012,
    module_dropout: float = 0.05,
    reward_profile: str = "classic",
    map_template: str = "central_burrow",
    operational_profile: str = "default_v1",
    noise_profile: str | NoiseConfig = "none",
    load_brain: str | Path | None = None,
    load_modules: Sequence[str] | None = None,
) -> None:
    """
    Launch the interactive Pygame GUI and run training/evaluation using a configured SpiderSimulation.
    
    Initializes a SpiderSimulation with the provided environment and learning hyperparameters, optionally loads a saved brain (or selected modules), then starts the GUI and runs the requested number of training and evaluation episodes.
    
    Parameters:
        width (int): Grid width of the simulated world in cells.
        height (int): Grid height of the simulated world in cells.
        episodes (int): Number of training episodes to run before switching to evaluation.
        eval_episodes (int): Number of evaluation episodes to run during the evaluation phase.
        food_count (int): Number of food items spawned in the world.
        day_length (int): Number of ticks in the day portion of the cycle.
        night_length (int): Number of ticks in the night portion of the cycle.
        max_steps (int): Maximum steps allowed per episode; episodes end early if this is reached.
        seed (int): Base random seed used to initialize the simulation and episode sequence.
        gamma (float): Discount factor used by the learning system.
        module_lr (float): Learning rate for proposal modules.
        motor_lr (float): Learning rate for the motor cortex.
        module_dropout (float): Dropout probability applied to proposal modules during training.
        reward_profile (str): Reward shaping profile name used by the simulator.
        map_template (str): World layout template name used when constructing each episode.
        operational_profile (str): Operational profile name controlling runtime behavior defaults.
        noise_profile (str | NoiseConfig): Noise configuration for environment stochasticity; may be a profile name or a NoiseConfig object.
        load_brain (str | Path | None): Path to a saved brain state to load before launching; if None, no brain is loaded.
        load_modules (Sequence[str] | None): Specific module names to load from the saved brain; if None, all modules are loaded.
    """
    sim = SpiderSimulation(
        width=width,
        height=height,
        food_count=food_count,
        day_length=day_length,
        night_length=night_length,
        max_steps=max_steps,
        seed=seed,
        gamma=gamma,
        module_lr=module_lr,
        motor_lr=motor_lr,
        module_dropout=module_dropout,
        reward_profile=reward_profile,
        map_template=map_template,
        operational_profile=operational_profile,
        noise_profile=noise_profile,
    )
    if load_brain is not None:
        loaded = sim.brain.load(load_brain, modules=load_modules)
        print(f"Loaded modules: {loaded}")
    gui = SpiderGUI(sim)
    gui.launch(train_episodes=episodes, eval_episodes=eval_episodes)
