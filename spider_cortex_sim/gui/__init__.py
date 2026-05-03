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
from .pygame_compat import pygame, require_pygame
from .rendering import Renderer
from .widgets import Button


class SpiderGUI:
    """Main simulation viewer window."""

    def __init__(self, sim: SpiderSimulation) -> None:
        self.pygame = require_pygame()
        self.pygame.init()
        self.controller = GUIController(sim)
        self.controller.apply_window_size(
            self.controller.win_w,
            TOP_BAR_HEIGHT + (sim.world.height * CELL_SIZE) + BOTTOM_BAR_HEIGHT,
        )
        grid_h = sim.world.height * self.controller.cell_size

        self.fonts = self._create_fonts()
        self.win_w = self.controller.win_w
        self.win_h = self.controller.win_h

        self.screen = self.pygame.display.set_mode(
            (self.win_w, self.win_h),
            self.pygame.RESIZABLE,
        )
        self.pygame.display.set_caption("Neuro-Modular Spider Simulation")
        self.clock = self.pygame.time.Clock()


        self.buttons = self._build_buttons(grid_h)
        self.events = EventHandler(
            controller=self.controller,
            buttons=self.buttons,
        )
        self.renderer = Renderer(
            surface=self.screen,
            fonts=self.fonts,
            controller=self.controller,
            buttons=self.buttons,
        )

    def _build_buttons(self, grid_h: int) -> dict[str, Button]:
        btn_y = TOP_BAR_HEIGHT + grid_h + 4
        btn_h = 24
        min_padding = 10
        preferred_gap = 6
        button_specs = [
            ("pause", "⏸ Pause", 100),
            ("step", "⏭ Step", 90),
            ("slower", "◀ Slower", 90),
            ("faster", "Faster ▶", 90),
            ("restart", "↻ Restart", 100),
            ("save", "💾 Save", 96),
            ("load", "📂 Load", 100),
        ]
        gap_count = len(button_specs) - 1
        total_button_w = sum(width for _, _, width in button_specs)
        min_padding_current = min_padding
        gap = preferred_gap
        available_w = max(0, self.win_w - 2 * min_padding_current)
        while (
            min_padding_current > 0
            and total_button_w + gap * gap_count > available_w
        ):
            min_padding_current -= 1
            available_w = max(0, self.win_w - 2 * min_padding_current)
        if total_button_w + gap * gap_count > available_w:
            gap = 0
            available_w = max(0, self.win_w - 2 * min_padding_current)

        scaled_w = max(0, available_w - gap * gap_count)
        scale = min(1.0, scaled_w / total_button_w)
        widths = [max(0, round(width * scale)) for _, _, width in button_specs]

        overflow = sum(widths) + gap * gap_count - available_w
        idx = len(widths) - 1
        while overflow > 0 and idx >= 0:
            shrink = min(overflow, widths[idx])
            widths[idx] -= shrink
            overflow -= shrink
            idx -= 1

        x = min_padding_current
        buttons: dict[str, Button] = {}
        for (name, text, _), width in zip(button_specs, widths, strict=True):
            buttons[name] = Button(text, x, btn_y, width, btn_h)
            x += width + gap
        return {
            name: buttons[name]
            for name, _, _ in button_specs
        }

    def launch(self, train_episodes: int, eval_episodes: int) -> None:
        self.controller.configure_run(train_episodes, eval_episodes)
        self._main_loop()

    def _main_loop(self) -> None:
        while self.controller.running:
            dt = self.clock.tick(60) / 1000.0
            self.events.handle_events()

            size = self.controller.consume_resize_request()
            if size is not None:
                self.win_w, self.win_h = size
                self.controller.apply_window_size(self.win_w, self.win_h)
                self.win_w = self.controller.win_w
                self.win_h = self.controller.win_h
                self.screen = self.pygame.display.set_mode(
                    (self.win_w, self.win_h),
                    self.pygame.RESIZABLE,
                )
                self.renderer.screen = self.screen

                self.fonts = self._create_fonts()
                self.renderer.set_fonts(self.fonts)
                self._rebuild_buttons()

            self.controller.tick(dt)
            self.renderer.draw()
            self.pygame.display.flip()
        self.pygame.quit()

    def _create_fonts(self) -> dict[str, pygame.font.Font]:
        return {
            "sm": self.pygame.font.SysFont("monospace", max(11, int(13 * self.controller.ui_scale))),
            "md": self.pygame.font.SysFont("monospace", max(12, int(15 * self.controller.ui_scale)), bold=True),
            "lg": self.pygame.font.SysFont("monospace", max(14, int(18 * self.controller.ui_scale)), bold=True),
            "xl": self.pygame.font.SysFont("monospace", max(16, int(22 * self.controller.ui_scale)), bold=True),
        }

    def _rebuild_buttons(self) -> None:
        grid_h = self.controller.world.height * self.controller.cell_size
        self.buttons = self._build_buttons(grid_h)
        self.events = EventHandler(
            controller=self.controller,
            buttons=self.buttons,
        )
        self.renderer.buttons = self.buttons
        self.renderer.btn_pause = self.buttons["pause"]


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
