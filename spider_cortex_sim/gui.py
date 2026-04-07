"""Interface gráfica Pygame para visualização da simulação neuro-modular da aranha."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import pygame
except ImportError:
    print(
        "pygame é necessário para a interface gráfica.\n"
        "Instale com:  pip install pygame"
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
# Paleta de cores
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
# Constantes de layout
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
    """Botão simples para a barra inferior."""

    def __init__(self, text: str, x: int, y: int, w: int, h: int) -> None:
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
    """Janela principal de visualização da simulação."""

    def __init__(self, sim: SpiderSimulation) -> None:
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

        # Estado
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

        # Toast de status
        self.toast_text: str = ""
        self.toast_timer: float = 0.0

        # Botões
        btn_y = TOP_BAR_HEIGHT + grid_h + 4
        btn_h = 24
        self.btn_pause = Button("⏸ Pausar", 10, btn_y, 100, btn_h)
        self.btn_step = Button("⏭ Passo", 116, btn_y, 90, btn_h)
        self.btn_slower = Button("◀ Lento", 212, btn_y, 90, btn_h)
        self.btn_faster = Button("Rápido ▶", 308, btn_y, 90, btn_h)
        self.btn_restart = Button("↻ Reiniciar", 404, btn_y, 100, btn_h)
        self.btn_save = Button("💾 Salvar", 510, btn_y, 96, btn_h)
        self.btn_load = Button("📂 Carregar", 612, btn_y, 100, btn_h)
        self.buttons = [
            self.btn_pause, self.btn_step, self.btn_slower, self.btn_faster,
            self.btn_restart, self.btn_save, self.btn_load,
        ]

    def launch(self, train_episodes: int, eval_episodes: int) -> None:
        """Ponto de entrada principal – treina e avalia com visualização."""
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
            predator_state=self.world.lizard.mode,
        )
        self.observation = next_obs
        self.current_step += 1

        if done:
            self.episode_done = True

        if self.current_step >= self.max_steps:
            self.episode_done = True

    def _advance_episode(self) -> None:
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
        """Roda o treinamento restante sem renderizar."""
        while self.phase == "training":
            while not self.episode_done:
                self._do_step()
            self._advance_episode()
        self._start_episode()

    def _save_brain(self, directory: str | Path | None = None) -> None:
        path = Path(directory) if directory else Path(DEFAULT_BRAIN_DIR)
        try:
            self.brain.save(path)
            self._show_toast(f"Cérebro salvo em {path}/")
        except Exception as exc:
            self._show_toast(f"Erro ao salvar: {exc}")

    def _load_brain(self, directory: str | Path | None = None, modules: Sequence[str] | None = None) -> None:
        path = Path(directory) if directory else Path(DEFAULT_BRAIN_DIR)
        try:
            loaded = self.brain.load(path, modules=modules)
            self._show_toast(f"Carregado: {', '.join(loaded)}")
        except Exception as exc:
            self._show_toast(f"Erro ao carregar: {exc}")

    def _show_toast(self, text: str, duration: float = 3.5) -> None:
        self.toast_text = text
        self.toast_timer = duration

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

    def _restart(self) -> None:
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
    # Desenho
    # -----------------------------------------------------------------------
    def _draw(self) -> None:
        self.screen.fill(COLOR_PANEL_BG)
        self._draw_top_bar()
        self._draw_grid()
        self._draw_panel()
        self._draw_bottom_bar()

    def _night_t(self) -> float:
        return 1.0 if self.world.is_night() else 0.0

    def _draw_top_bar(self) -> None:
        rect = pygame.Rect(0, 0, self.win_w, TOP_BAR_HEIGHT)
        pygame.draw.rect(self.screen, (40, 40, 48), rect)

        # Fase e episódio
        if self.phase == "training":
            phase_text = f"TREINAMENTO  ep {self.current_episode + 1}/{self.total_train_episodes}"
        elif self.phase == "evaluation":
            phase_text = f"AVALIAÇÃO  ep {self.current_episode + 1}/{self.total_eval_episodes}"
        else:
            phase_text = "CONCLUÍDO"
        label = self.font_lg.render(phase_text, True, COLOR_TEXT_TITLE)
        self.screen.blit(label, (12, 14))

        # Velocidade
        speed_text = f"Vel: {TICK_SPEEDS[self.speed_idx]} t/s"
        sl = self.font_sm.render(speed_text, True, COLOR_TEXT_DIM)
        self.screen.blit(sl, (self.win_w - 160, 8))

        state_text = "⏸ PAUSADO" if self.paused else "▶ RODANDO"
        st = self.font_sm.render(state_text, True, COLOR_TEXT)
        self.screen.blit(st, (self.win_w - 160, 28))

    def _draw_grid(self) -> None:
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

        # Células
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

        # Comida
        for fx, fy in self.world.food_positions:
            cx = ox + fx * CELL_SIZE + CELL_SIZE // 2
            cy_pos = oy + fy * CELL_SIZE + CELL_SIZE // 2
            # Corpo da mosca/inseto
            pygame.draw.circle(self.screen, (80, 60, 40), (cx, cy_pos), 8)
            pygame.draw.circle(self.screen, COLOR_FOOD, (cx, cy_pos), 6)
            # Asas
            pygame.draw.ellipse(self.screen, (255, 200, 200, 180),
                                pygame.Rect(cx - 12, cy_pos - 8, 10, 6))
            pygame.draw.ellipse(self.screen, (255, 200, 200, 180),
                                pygame.Rect(cx + 2, cy_pos - 8, 10, 6))

        # Lagarto predador
        self._draw_lizard(
            ox + self.world.lizard.x * CELL_SIZE + CELL_SIZE // 2,
            oy + self.world.lizard.y * CELL_SIZE + CELL_SIZE // 2,
        )

        # Aranha
        self._draw_spider(
            ox + self.world.state.x * CELL_SIZE + CELL_SIZE // 2,
            oy + self.world.state.y * CELL_SIZE + CELL_SIZE // 2,
        )

        # Overlay noturno suave
        if nt > 0:
            overlay = pygame.Surface((grid_w, grid_h), pygame.SRCALPHA)
            overlay.fill((0, 0, 30, int(60 * nt)))
            self.screen.blit(overlay, (ox, oy))


    def _draw_lizard(self, cx: int, cy: int) -> None:
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
        # Pernas
        leg_offsets = [
            (-14, -10), (-16, -3), (-15, 5), (-12, 12),
            (14, -10), (16, -3), (15, 5), (12, 12),
        ]
        for lx, ly in leg_offsets:
            pygame.draw.line(self.screen, COLOR_SPIDER_LEGS, (cx, cy), (cx + lx, cy + ly), 2)

        # Corpo (abdômen + cefalotórax)
        pygame.draw.circle(self.screen, COLOR_SPIDER_BODY, (cx, cy + 4), 9)
        pygame.draw.circle(self.screen, (60, 60, 60), (cx, cy - 4), 7)

        # Olhos
        pygame.draw.circle(self.screen, COLOR_SPIDER_EYES, (cx - 3, cy - 7), 2)
        pygame.draw.circle(self.screen, COLOR_SPIDER_EYES, (cx + 3, cy - 7), 2)

        # Brilho nos olhos
        pygame.draw.circle(self.screen, (255, 255, 255), (cx - 2, cy - 8), 1)
        pygame.draw.circle(self.screen, (255, 255, 255), (cx + 4, cy - 8), 1)

    def _draw_panel(self) -> None:
        grid_h = self.world.height * CELL_SIZE
        panel_rect = pygame.Rect(self.panel_x, self.panel_y, PANEL_WIDTH, grid_h)
        pygame.draw.rect(self.screen, COLOR_PANEL_BG, panel_rect)
        pygame.draw.line(self.screen, COLOR_PANEL_BORDER,
                         (self.panel_x, self.panel_y),
                         (self.panel_x, self.panel_y + grid_h))

        x0 = self.panel_x + 14
        y = self.panel_y + 10
        pw = PANEL_WIDTH - 28
        metric_snapshot = self.episode_metrics.snapshot()

        # --- Status ---
        y = self._draw_section_title("ESTADO DA ARANHA", x0, y)
        st = self.world.state
        phase = "NOITE 🌙" if self.world.is_night() else "DIA ☀"
        y = self._draw_text(f"Tick: {self.world.tick}   Fase: {phase}", x0, y, COLOR_TEXT)
        y = self._draw_text(f"Posição: ({st.x}, {st.y})", x0, y, COLOR_TEXT_DIM)
        y = self._draw_text(f"Mapa: {self.world.map_template_name}   Perfil: {self.world.reward_profile}", x0, y, COLOR_TEXT_DIM)
        y = self._draw_text(f"Abrigo: {self.world.shelter_role_at(self.world.spider_pos())}", x0, y, COLOR_TEXT_DIM)

        y += 4
        y = self._draw_bar("Fome", st.hunger, x0, y, pw, COLOR_BAR_HUNGER)
        y = self._draw_bar("Fadiga", st.fatigue, x0, y, pw, COLOR_BAR_FATIGUE)
        y = self._draw_bar("Dívida sono", st.sleep_debt, x0, y, pw, (160, 120, 230))
        y = self._draw_bar("Saúde", st.health, x0, y, pw, COLOR_BAR_HEALTH)

        y += 4
        y = self._draw_text(f"Comida: {st.food_eaten}   Sono: {st.sleep_events}   Abrigos: {st.shelter_entries}", x0, y, COLOR_TEXT_DIM)
        y = self._draw_text(f"Predador: contatos={st.predator_contacts}  avistamentos={st.predator_sightings}  fugas={st.predator_escapes}", x0, y, COLOR_TEXT_DIM)
        y = self._draw_text(f"Lagarto: ({self.world.lizard.x}, {self.world.lizard.y})  modo={self.world.lizard.mode}", x0, y, COLOR_TEXT_DIM)
        y = self._draw_text(
            f"Alvo lagarto: patrulha={self.world.lizard.patrol_target} espera={self.world.lizard.wait_target}",
            x0,
            y,
            COLOR_TEXT_DIM,
        )
        y = self._draw_text(
            f"Overlays: vis={'on' if self.show_visibility_overlay else 'off'}  cheiro={'on' if self.show_smell_overlay else 'off'}",
            x0,
            y,
            COLOR_TEXT_DIM,
        )

        y += 6
        y = self._draw_section_title("MEMÓRIA", x0, y)
        y = self._draw_text(f"Comida: alvo={st.food_memory.target} idade={st.food_memory.age}", x0, y, COLOR_TEXT_DIM)
        y = self._draw_text(f"Predador: alvo={st.predator_memory.target} idade={st.predator_memory.age}", x0, y, COLOR_TEXT_DIM)
        y = self._draw_text(f"Abrigo: alvo={st.shelter_memory.target} idade={st.shelter_memory.age}", x0, y, COLOR_TEXT_DIM)
        y = self._draw_text(f"Fuga: alvo={st.escape_memory.target} idade={st.escape_memory.age}", x0, y, COLOR_TEXT_DIM)
        if self.observation is not None:
            vision = self.observation["meta"]["vision"]
            y = self._draw_text(
                f"Visão predador: c={vision['predator']['certainty']:.2f} occ={vision['predator']['occluded']:.0f}",
                x0,
                y,
                COLOR_TEXT_DIM,
            )
            y = self._draw_text(
                f"Vetores: comida={self.observation['meta']['memory_vectors']['food']} abrigo={self.observation['meta']['memory_vectors']['shelter']}",
                x0,
                y,
                COLOR_TEXT_DIM,
            )

        # --- Recompensa ---
        y += 6
        y = self._draw_section_title("RECOMPENSA", x0, y)
        rc = COLOR_REWARD_POS if self.last_reward >= 0 else COLOR_REWARD_NEG
        y = self._draw_text(f"Última: {self.last_reward:+.3f}   Total: {self.episode_reward:.2f}", x0, y, rc)
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

        # Mini gráfico de recompensa
        y += 2
        y = self._draw_reward_chart(x0, y, pw, 40)

        # --- Ação ---
        y += 6
        y = self._draw_section_title("AÇÃO ATUAL", x0, y)
        if self.last_decision is not None:
            action_name = ACTIONS[self.last_decision.action_idx]
            for i, a in enumerate(ACTIONS):
                prob = self.last_decision.policy[i]
                is_selected = (a == action_name)
                color = COLOR_ACTION_ACTIVE if is_selected else COLOR_ACTION_INACTIVE
                marker = "▶" if is_selected else " "
                y = self._draw_text(f" {marker} {a:<18} {prob:.3f}", x0, y, color)
        else:
            y = self._draw_text("  (aguardando...)", x0, y, COLOR_TEXT_DIM)

        # --- Módulos neurais ---
        y += 6
        y = self._draw_section_title("MÓDULOS CORTICAIS", x0, y)
        if self.last_decision is not None:
            for i, result in enumerate(self.last_decision.module_results):
                color = COLOR_MODULE_COLORS[i % len(COLOR_MODULE_COLORS)]
                best_action = ACTIONS[int(np.argmax(result.probs))]
                confidence = float(np.max(result.probs))
                short_name = result.name.replace("_", " ").title()
                y = self._draw_text(f"  {short_name}", x0, y, color)
                reflex_text = "sem reflexo"
                if result.reflex is not None:
                    reflex_text = f"{result.reflex.action} · {result.reflex.reason}"
                y = self._draw_text(
                    f"    topo={best_action} ({confidence:.2f})  reflexo={reflex_text}",
                    x0,
                    y,
                    COLOR_TEXT_DIM,
                )
                # Mini barras de probabilidade
                bar_y = y
                bar_h = 8
                for j, prob in enumerate(result.probs):
                    bx = x0 + 12 + j * (pw - 24) // len(ACTIONS)
                    bw = max(1, (pw - 24) // len(ACTIONS) - 2)
                    filled_h = int(bar_h * prob)
                    pygame.draw.rect(self.screen, (50, 50, 55), pygame.Rect(bx, bar_y, bw, bar_h))
                    pygame.draw.rect(self.screen, color, pygame.Rect(bx, bar_y + bar_h - filled_h, bw, filled_h))
                y = bar_y + bar_h + 4

        # --- Progresso do treino ---
        if self.training_rewards:
            y += 4
            y = self._draw_section_title("PROGRESSO", x0, y)
            window = self.training_rewards[-min(20, len(self.training_rewards)):]
            avg = sum(window) / len(window)
            y = self._draw_text(f"Média últimos {len(window)} ep: {avg:.2f}", x0, y, COLOR_TEXT_DIM)

        y += 6
        y = self._draw_section_title("MÉTRICAS", x0, y)
        role_dist = metric_snapshot["night_role_distribution"]
        y = self._draw_text(
            "Noite: "
            f"out={role_dist['outside']:.2f} ent={role_dist['entrance']:.2f} "
            f"in={role_dist['inside']:.2f} deep={role_dist['deep']:.2f}",
            x0,
            y,
            COLOR_TEXT_DIM,
        )
        y = self._draw_text(
            f"Latência predador: {metric_snapshot['mean_predator_response_latency']:.2f}  eventos={metric_snapshot['predator_response_events']}",
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
        # Linha zero
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
        grid_h = self.world.height * CELL_SIZE
        bar_y = TOP_BAR_HEIGHT + grid_h
        rect = pygame.Rect(0, bar_y, self.win_w, BOTTOM_BAR_HEIGHT)
        pygame.draw.rect(self.screen, (40, 40, 48), rect)

        self.btn_pause.text = "▶ Rodar" if self.paused else "⏸ Pausar"
        for btn in self.buttons:
            btn.draw(self.screen, self.font_sm)

        # Atalhos
        help_text = "Espaço=pausar  →/N=passo  +/-=vel  V=visão  M=cheiro  S=pular treino  R=reiniciar  Q=sair"
        ht = self.font_sm.render(help_text, True, COLOR_TEXT_DIM)
        self.screen.blit(ht, (10, bar_y + 30))

        help2 = "Ctrl+S=salvar cérebro  Ctrl+L=carregar cérebro"
        ht2 = self.font_sm.render(help2, True, COLOR_TEXT_DIM)
        self.screen.blit(ht2, (self.win_w - ht2.get_width() - 10, bar_y + 30))

        # Toast de status
        if self.toast_timer > 0 and self.toast_text:
            alpha = min(1.0, self.toast_timer / 0.5)
            toast_color = (100, 200, 130) if "Erro" not in self.toast_text else (220, 80, 80)
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
    Launch a Pygame GUI for a SpiderSimulation configured with the given simulation and brain parameters.
    
    Initializes a SpiderSimulation with the provided environment and learning hyperparameters, optionally loads a saved brain (or selected modules), then opens the interactive GUI and runs the specified number of training and evaluation episodes.
    
    Parameters:
        episodes (int): Number of training episodes to run before switching to evaluation.
        eval_episodes (int): Number of evaluation episodes to run during the evaluation phase.
        width (int): World grid width in cells.
        height (int): World grid height in cells.
        food_count (int): Number of food items placed in the world at reset.
        day_length (int): Duration of daytime in ticks.
        night_length (int): Duration of nighttime in ticks.
        max_steps (int): Maximum steps allowed per episode.
        seed (int): Base random seed for environment and brain initialization.
        gamma (float): Discount factor used by the brain.
        module_lr (float): Learning rate for cortical/module parameters.
        motor_lr (float): Learning rate for motor/action parameters.
        module_dropout (float): Dropout probability applied to module outputs during training.
        reward_profile (str): Named reward profile to configure reward components.
        map_template (str): Identifier of the map template used for world layout.
        operational_profile (str): Runtime profile name that configures simulation/brain behavior.
        noise_profile (str | NoiseConfig): Noise configuration applied to environment stochasticity (accepts a profile name or a NoiseConfig).
        load_brain (str | Path | None): Path to a saved brain state to load before launching; if None, no load is performed.
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
        print(f"Módulos carregados: {loaded}")
    gui = SpiderGUI(sim)
    gui.launch(train_episodes=episodes, eval_episodes=eval_episodes)
