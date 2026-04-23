"""Pygame graphical interface for visualizing the neuro-modular simulation."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from ..noise import NoiseConfig
from ..capacity_profiles import CapacityProfile
from ..simulation import SpiderSimulation
from .constants import BOTTOM_BAR_HEIGHT, CELL_SIZE, TOP_BAR_HEIGHT
from .controller import GUIController
from .events import EventHandler
from .pygame_compat import require_pygame
from .rendering import Renderer
from .widgets import Button


class SpiderGUI:
    """Main simulation viewer window."""

    def __init__(self, sim: SpiderSimulation) -> None:
        self.pygame = require_pygame()
        self.controller = GUIController(sim)
        grid_h = sim.world.height * CELL_SIZE
        self.win_w = self.controller.win_w
        self.win_h = TOP_BAR_HEIGHT + grid_h + BOTTOM_BAR_HEIGHT

        self.pygame.init()
        self.screen = self.pygame.display.set_mode((self.win_w, self.win_h))
        self.pygame.display.set_caption("Neuro-Modular Spider Simulation")
        self.clock = self.pygame.time.Clock()

        fonts = {
            "sm": self.pygame.font.SysFont("monospace", 13),
            "md": self.pygame.font.SysFont("monospace", 15, bold=True),
            "lg": self.pygame.font.SysFont("monospace", 18, bold=True),
            "xl": self.pygame.font.SysFont("monospace", 22, bold=True),
        }

        self.buttons = self._build_buttons(grid_h)
        self.events = EventHandler(
            controller=self.controller,
            buttons=self.buttons,
        )
        self.renderer = Renderer(
            surface=self.screen,
            fonts=fonts,
            controller=self.controller,
            buttons=self.buttons,
        )

    def _build_buttons(self, grid_h: int) -> dict[str, Button]:
        btn_y = TOP_BAR_HEIGHT + grid_h + 4
        btn_h = 24
        return {
            "pause": Button("⏸ Pause", 10, btn_y, 100, btn_h),
            "step": Button("⏭ Step", 116, btn_y, 90, btn_h),
            "slower": Button("◀ Slower", 212, btn_y, 90, btn_h),
            "faster": Button("Faster ▶", 308, btn_y, 90, btn_h),
            "restart": Button("↻ Restart", 404, btn_y, 100, btn_h),
            "save": Button("💾 Save", 510, btn_y, 96, btn_h),
            "load": Button("📂 Load", 612, btn_y, 100, btn_h),
        }

    def launch(self, train_episodes: int, eval_episodes: int) -> None:
        self.controller.configure_run(train_episodes, eval_episodes)
        self._main_loop()

    def _main_loop(self) -> None:
        while self.controller.running:
            dt = self.clock.tick(60) / 1000.0
            self.events.handle_events()
            self.controller.tick(dt)
            self.renderer.draw()
            self.pygame.display.flip()
        self.pygame.quit()


def run_gui(
    *,
    episodes: int = 180,
    eval_episodes: int = 3,
    width: int = 12,
    height: int = 12,
    food_count: int = 4,
    day_length: int = 36,
    night_length: int = 24,
    max_steps: int = 120,
    seed: int = 7,
    gamma: float = 0.96,
    module_lr: float = 0.010,
    motor_lr: float = 0.012,
    module_dropout: float = 0.05,
    capacity_profile: str | CapacityProfile | None = None,
    reward_profile: str = "classic",
    map_template: str = "central_burrow",
    operational_profile: str = "default_v1",
    noise_profile: str | NoiseConfig = "none",
    load_brain: str | Path | None = None,
    load_modules: Sequence[str] | None = None,
) -> None:
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
        capacity_profile=capacity_profile,
        reward_profile=reward_profile,
        map_template=map_template,
        operational_profile=operational_profile,
        noise_profile=noise_profile,
    )
    gui = SpiderGUI(sim)
    if load_brain is not None:
        gui.controller.load_brain(load_brain, modules=load_modules)
    gui.launch(train_episodes=episodes, eval_episodes=eval_episodes)


__all__ = ["Button", "GUIController", "Renderer", "SpiderGUI", "run_gui"]
