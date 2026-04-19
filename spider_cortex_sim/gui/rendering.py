from __future__ import annotations

from typing import Tuple

import numpy as np

from ..maps import BLOCKED, CLUTTER, NARROW
from ..predator import PREDATOR_STATES
from ..world import ACTIONS
from .constants import (
    BOTTOM_BAR_HEIGHT,
    CELL_SIZE,
    PANEL_WIDTH,
    TICK_SPEEDS,
    TOP_BAR_HEIGHT,
)
from .pygame_compat import pygame

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
    (130, 180, 255),
    (255, 180, 100),
    (255, 120, 120),
    (120, 160, 255),
    (255, 220, 80),
    (160, 255, 160),
]

COLOR_REWARD_POS = (80, 220, 120)
COLOR_REWARD_NEG = (220, 80, 80)

COLOR_BUTTON = (60, 60, 70)
COLOR_BUTTON_HOVER = (80, 80, 95)
COLOR_BUTTON_TEXT = (200, 200, 210)

MIN_PANEL_SURF_H = 1600

_CONTROLLER_DELEGATED_NAMES = {
    "current_episode",
    "episode_metrics",
    "episode_reward",
    "grid_offset_x",
    "grid_offset_y",
    "last_decision",
    "last_info",
    "last_reward",
    "observation",
    "panel_x",
    "panel_y",
    "paused",
    "phase",
    "reward_history",
    "show_smell_overlay",
    "show_visibility_overlay",
    "speed_idx",
    "toast_is_error",
    "toast_text",
    "toast_timer",
    "total_eval_episodes",
    "total_train_episodes",
    "training_rewards",
    "win_w",
    "world",
}

def _lerp_color(c1: Tuple[int, ...], c2: Tuple[int, ...], t: float) -> Tuple[int, ...]:
    if len(c1) != len(c2):
        raise ValueError("Color tuples must have the same length.")
    return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))


class Renderer:
    def __init__(
        self,
        *,
        surface: pygame.Surface,
        fonts: dict[str, pygame.font.Font],
        controller,
        buttons,
    ) -> None:
        self.screen = surface
        self.controller = controller
        self.buttons = buttons
        self.font_sm = fonts["sm"]
        self.font_md = fonts["md"]
        self.font_lg = fonts["lg"]
        self.font_xl = fonts["xl"]
        self._night_overlay = None
        self._night_overlay_size = None
        self.btn_pause = buttons["pause"]

    def __getattr__(self, name: str):
        if name not in _CONTROLLER_DELEGATED_NAMES:
            raise AttributeError(
                f"{type(self).__name__!s} object has no attribute {name!r}"
            )
        return getattr(self.controller, name)

    @property
    def panel_scroll(self) -> int:
        return self.controller.panel_scroll

    @panel_scroll.setter
    def panel_scroll(self, value: int) -> None:
        self.controller.panel_scroll = value

    @property
    def panel_content_height(self) -> int:
        return self.controller.panel_content_height

    @panel_content_height.setter
    def panel_content_height(self, value: int) -> None:
        self.controller.panel_content_height = value

    def draw(self) -> None:
        self._draw()

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
            smell_overlay_surface = pygame.Surface(
                (CELL_SIZE, CELL_SIZE),
                pygame.SRCALPHA,
            )
            for cy in range(self.world.height):
                for cx in range(self.world.width):
                    fx = food_smell[cy][cx]
                    px = predator_smell[cy][cx]
                    if fx <= 0.01 and px <= 0.01:
                        continue
                    smell_overlay_surface.fill(
                        (int(220 * fx), 60, int(220 * px), int(90 * max(fx, px)))
                    )
                    self.screen.blit(
                        smell_overlay_surface,
                        (ox + cx * CELL_SIZE, oy + cy * CELL_SIZE),
                    )

        if self.show_visibility_overlay:
            visibility = self.world.visibility_overlay()
            occluded_overlay_surface = pygame.Surface(
                (CELL_SIZE, CELL_SIZE),
                pygame.SRCALPHA,
            )
            visible_overlay_surface = pygame.Surface(
                (CELL_SIZE, CELL_SIZE),
                pygame.SRCALPHA,
            )
            occluded_overlay_surface.fill((220, 180, 60, 48))
            visible_overlay_surface.fill((80, 220, 120, 42))
            for cx, cy in visibility["occluded"]:
                self.screen.blit(
                    occluded_overlay_surface,
                    (ox + cx * CELL_SIZE, oy + cy * CELL_SIZE),
                )
            for cx, cy in visibility["visible"]:
                self.screen.blit(
                    visible_overlay_surface,
                    (ox + cx * CELL_SIZE, oy + cy * CELL_SIZE),
                )

        # Food
        for fx, fy in self.world.food_positions:
            cx = ox + fx * CELL_SIZE + CELL_SIZE // 2
            cy_pos = oy + fy * CELL_SIZE + CELL_SIZE // 2
            # Fly or insect body
            pygame.draw.circle(self.screen, (80, 60, 40), (cx, cy_pos), 8)
            pygame.draw.circle(self.screen, COLOR_FOOD, (cx, cy_pos), 6)
            # Wings
            pygame.draw.ellipse(self.screen, (255, 200, 200),
                                pygame.Rect(cx - 12, cy_pos - 8, 10, 6))
            pygame.draw.ellipse(self.screen, (255, 200, 200),
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
            overlay_size = (grid_w, grid_h)
            if (
                self._night_overlay is None
                or self._night_overlay_size != overlay_size
            ):
                self._night_overlay = pygame.Surface(overlay_size, pygame.SRCALPHA)
                self._night_overlay_size = overlay_size
            self._night_overlay.fill((0, 0, 30, int(60 * nt)))
            self.screen.blit(self._night_overlay, (ox, oy))

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
        panel_surf_h = max(grid_h, MIN_PANEL_SURF_H)
        panel_surf = pygame.Surface((PANEL_WIDTH, panel_surf_h))
        panel_surf.fill(COLOR_PANEL_BG)

        x0 = 14
        y = 10
        pw = PANEL_WIDTH - 28
        metric_snapshot = self.episode_metrics.snapshot()

        # --- Status ---
        y = self._draw_section_title(panel_surf, "SPIDER STATE", x0, y)
        st = self.world.state
        phase = "NIGHT 🌙" if self.world.is_night() else "DAY ☀"
        y = self._draw_text(panel_surf, f"Tick: {self.world.tick}   Phase: {phase}", x0, y, COLOR_TEXT)
        y = self._draw_text(panel_surf, f"Position: ({st.x}, {st.y})", x0, y, COLOR_TEXT_DIM)
        y = self._draw_text(panel_surf, f"Map: {self.world.map_template_name}   Profile: {self.world.reward_profile}", x0, y, COLOR_TEXT_DIM)
        y = self._draw_text(panel_surf, f"Shelter: {self.world.shelter_role_at(self.world.spider_pos())}", x0, y, COLOR_TEXT_DIM)

        y += 4
        y = self._draw_bar(panel_surf, "Hunger", st.hunger, x0, y, pw, COLOR_BAR_HUNGER)
        y = self._draw_bar(panel_surf, "Fatigue", st.fatigue, x0, y, pw, COLOR_BAR_FATIGUE)
        y = self._draw_bar(panel_surf, "Sleep debt", st.sleep_debt, x0, y, pw, (160, 120, 230))
        y = self._draw_bar(panel_surf, "Health", st.health, x0, y, pw, COLOR_BAR_HEALTH)

        y += 4
        y = self._draw_text(panel_surf, f"Food: {st.food_eaten}   Sleep: {st.sleep_events}   Shelter entries: {st.shelter_entries}", x0, y, COLOR_TEXT_DIM)
        y = self._draw_text(panel_surf, f"Predator: contacts={st.predator_contacts}  sightings={st.predator_sightings}  escapes={st.predator_escapes}", x0, y, COLOR_TEXT_DIM)
        y = self._draw_text(panel_surf, f"Lizard: ({self.world.lizard.x}, {self.world.lizard.y})  mode={self.world.lizard.mode}", x0, y, COLOR_TEXT_DIM)
        y = self._draw_text(panel_surf,
            f"Lizard target: patrol={self.world.lizard.patrol_target} wait={self.world.lizard.wait_target}",
            x0,
            y,
            COLOR_TEXT_DIM,
        )
        y = self._draw_text(panel_surf,
            f"Overlays: vision={'on' if self.show_visibility_overlay else 'off'}  smell={'on' if self.show_smell_overlay else 'off'}",
            x0,
            y,
            COLOR_TEXT_DIM,
        )

        y += 6
        y = self._draw_section_title(panel_surf, "MEMORY", x0, y)
        y = self._draw_text(panel_surf, f"Food: target={st.food_memory.target} age={st.food_memory.age}", x0, y, COLOR_TEXT_DIM)
        y = self._draw_text(panel_surf, f"Predator: target={st.predator_memory.target} age={st.predator_memory.age}", x0, y, COLOR_TEXT_DIM)
        y = self._draw_text(panel_surf, f"Shelter: target={st.shelter_memory.target} age={st.shelter_memory.age}", x0, y, COLOR_TEXT_DIM)
        y = self._draw_text(panel_surf, f"Escape: target={st.escape_memory.target} age={st.escape_memory.age}", x0, y, COLOR_TEXT_DIM)
        if self.observation is not None:
            vision = self.observation["meta"]["vision"]
            y = self._draw_text(panel_surf,
                f"Predator vision: c={vision['predator']['certainty']:.2f} occ={vision['predator']['occluded']:.0f}",
                x0,
                y,
                COLOR_TEXT_DIM,
            )
            y = self._draw_text(panel_surf,
                f"Vectors: food={self.observation['meta']['memory_vectors']['food']} shelter={self.observation['meta']['memory_vectors']['shelter']}",
                x0,
                y,
                COLOR_TEXT_DIM,
            )

        # --- Reward ---
        y += 6
        y = self._draw_section_title(panel_surf, "REWARD", x0, y)
        rc = COLOR_REWARD_POS if self.last_reward >= 0 else COLOR_REWARD_NEG
        y = self._draw_text(panel_surf, f"Last: {self.last_reward:+.3f}   Total: {self.episode_reward:.2f}", x0, y, rc)
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
                y = self._draw_text(panel_surf, f"{name}: {value:+.3f}", x0, y, color)

        # Mini reward chart
        y += 2
        y = self._draw_reward_chart(panel_surf, x0, y, pw, 40)

        # --- Current action ---
        y += 6
        y = self._draw_section_title(panel_surf, "CURRENT ACTION", x0, y)
        if self.last_decision is not None:
            action_name = ACTIONS[self.last_decision.action_idx]
            for i, a in enumerate(ACTIONS):
                prob = self.last_decision.policy[i]
                is_selected = (a == action_name)
                color = COLOR_ACTION_ACTIVE if is_selected else COLOR_ACTION_INACTIVE
                marker = "▶" if is_selected else " "
                y = self._draw_text(panel_surf, f" {marker} {a:<18} {prob:.3f}", x0, y, color)
        else:
            y = self._draw_text(panel_surf, "  (waiting...)", x0, y, COLOR_TEXT_DIM)

        # --- Neural modules ---
        y += 6
        y = self._draw_section_title(panel_surf, "CORTICAL MODULES", x0, y)
        if self.last_decision is not None:
            for i, result in enumerate(self.last_decision.module_results):
                color = COLOR_MODULE_COLORS[i % len(COLOR_MODULE_COLORS)]
                best_action = ACTIONS[int(np.argmax(result.probs))]
                confidence = float(np.max(result.probs))
                short_name = result.name.replace("_", " ").title()
                y = self._draw_text(panel_surf, f"  {short_name}", x0, y, color)
                reflex_text = "no reflex"
                if result.reflex is not None:
                    reflex_text = f"{result.reflex.action} · {result.reflex.reason}"
                y = self._draw_text(panel_surf,
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
                    pygame.draw.rect(panel_surf, (50, 50, 55), pygame.Rect(bx, bar_y, bw, bar_h))
                    pygame.draw.rect(panel_surf, color, pygame.Rect(bx, bar_y + bar_h - filled_h, bw, filled_h))
                y = bar_y + bar_h + 4

        # --- Training progress ---
        if self.training_rewards:
            y += 4
            y = self._draw_section_title(panel_surf, "PROGRESS", x0, y)
            window = self.training_rewards[-min(20, len(self.training_rewards)):]
            avg = sum(window) / len(window)
            y = self._draw_text(panel_surf, f"Mean over last {len(window)} eps: {avg:.2f}", x0, y, COLOR_TEXT_DIM)

        y += 6
        y = self._draw_section_title(panel_surf, "METRICS", x0, y)
        role_dist = metric_snapshot["night_role_distribution"]
        y = self._draw_text(panel_surf,
            "Night: "
            f"out={role_dist['outside']:.2f} ent={role_dist['entrance']:.2f} "
            f"in={role_dist['inside']:.2f} deep={role_dist['deep']:.2f}",
            x0,
            y,
            COLOR_TEXT_DIM,
        )
        y = self._draw_text(panel_surf,
            f"Predator latency: {metric_snapshot['mean_predator_response_latency']:.2f}  events={metric_snapshot['predator_response_events']}",
            x0,
            y,
            COLOR_TEXT_DIM,
        )
        occupancy = metric_snapshot["predator_state_occupancy"]
        for state_name in PREDATOR_STATES:
            y = self._draw_text(panel_surf,
                f"{state_name:<12} occ={occupancy.get(state_name, 0.0):.2f} ticks={metric_snapshot['predator_state_ticks'].get(state_name, 0)}",
                x0,
                y,
                COLOR_TEXT_DIM,
            )

        # Blit the scrolled panel surface
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

    def _draw_section_title(
        self,
        surface: pygame.Surface,
        text: str,
        x: int,
        y: int,
    ) -> int:
        label = self.font_md.render(text, True, COLOR_TEXT_TITLE)
        surface.blit(label, (x, y))
        y += label.get_height() + 2
        pygame.draw.line(surface, COLOR_PANEL_BORDER, (x, y), (x + PANEL_WIDTH - 28, y))
        return y + 4

    def _draw_text(
        self,
        surface: pygame.Surface,
        text: str,
        x: int,
        y: int,
        color: Tuple[int, ...],
    ) -> int:
        label = self.font_sm.render(text, True, color)
        surface.blit(label, (x, y))
        return y + label.get_height() + 1

    def _draw_bar(
        self,
        surface: pygame.Surface,
        name: str,
        value: float,
        x: int,
        y: int,
        w: int,
        color: Tuple[int, ...],
    ) -> int:
        label = self.font_sm.render(f"{name}: {value:.2f}", True, COLOR_TEXT)
        surface.blit(label, (x, y))
        bar_x = x + 120
        bar_w = w - 120
        bar_h = 12
        pygame.draw.rect(surface, COLOR_BAR_BG, pygame.Rect(bar_x, y + 1, bar_w, bar_h), border_radius=3)
        fill_w = int(bar_w * min(max(value, 0), 1))
        if fill_w > 0:
            pygame.draw.rect(surface, color, pygame.Rect(bar_x, y + 1, fill_w, bar_h), border_radius=3)
        return y + bar_h + 5

    def _draw_reward_chart(
        self,
        surface: pygame.Surface,
        x: int,
        y: int,
        w: int,
        h: int,
    ) -> int:
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
        pygame.draw.rect(surface, (40, 40, 45), pygame.Rect(x, y, w, h), border_radius=3)
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
        pygame.draw.line(surface, COLOR_PANEL_BORDER, (x, mid_y), (x + w, mid_y))

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
                pygame.draw.rect(surface, color, pygame.Rect(px, ry, max(1, int(step_w) - 1), bar_h_px))
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
        for btn in self.buttons.values():
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
            alpha_value = int(alpha * 255)
            toast_color = (220, 80, 80) if self.toast_is_error else (100, 200, 130)
            toast_surf = self.font_md.render(
                self.toast_text,
                True,
                toast_color,
            ).convert_alpha()
            toast_surf.set_alpha(alpha_value)
            tx = (self.win_w - toast_surf.get_width()) // 2
            ty = bar_y - 22
            bg_rect = pygame.Rect(
                tx - 8,
                ty - 2,
                toast_surf.get_width() + 16,
                toast_surf.get_height() + 4,
            )
            bg_surf = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(
                bg_surf,
                (30, 30, 35, alpha_value),
                bg_surf.get_rect(),
                border_radius=4,
            )
            self.screen.blit(bg_surf, bg_rect.topleft)
            self.screen.blit(toast_surf, (tx, ty))
