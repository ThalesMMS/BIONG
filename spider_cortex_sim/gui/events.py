from __future__ import annotations

from .pygame_compat import pygame


class EventHandler:
    def __init__(self, *, controller, buttons) -> None:
        self.controller = controller
        self.buttons = buttons
        self.btn_pause = buttons["pause"]
        self.btn_step = buttons["step"]
        self.btn_slower = buttons["slower"]
        self.btn_faster = buttons["faster"]
        self.btn_restart = buttons["restart"]
        self.btn_save = buttons["save"]
        self.btn_load = buttons["load"]

    def handle_events(self) -> None:
        mouse_pos = pygame.mouse.get_pos()
        for btn in self.buttons.values():
            btn.update_hover(mouse_pos)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.controller.request_quit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                    self.controller.request_quit()
                elif event.key == pygame.K_SPACE:
                    self.controller.toggle_pause()
                elif event.key == pygame.K_RIGHT or event.key == pygame.K_n:
                    self.controller.request_step()
                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    self.controller.increase_speed()
                elif event.key == pygame.K_MINUS:
                    self.controller.decrease_speed()
                elif event.key == pygame.K_s:
                    mods = pygame.key.get_mods()
                    if mods & pygame.KMOD_CTRL or mods & pygame.KMOD_META:
                        self.controller.save_brain()
                    elif self.controller.phase == "training":
                        self.controller.skip_training()
                elif event.key == pygame.K_l:
                    mods = pygame.key.get_mods()
                    if mods & pygame.KMOD_CTRL or mods & pygame.KMOD_META:
                        self.controller.load_brain()
                elif event.key == pygame.K_r:
                    self.controller.restart()
                elif event.key == pygame.K_v:
                    self.controller.toggle_visibility_overlay()
                elif event.key == pygame.K_m:
                    self.controller.toggle_smell_overlay()
                elif event.key == pygame.K_PAGEDOWN:
                    self._scroll_panel(80)
                elif event.key == pygame.K_PAGEUP:
                    self._scroll_panel(-80)
            elif event.type == pygame.MOUSEWHEEL:
                # Scroll panel when mouse is over the panel area
                mx, _ = pygame.mouse.get_pos()
                if mx >= self.controller.panel_x:
                    self._scroll_panel(-event.y * 30)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self.btn_pause.is_clicked(event.pos):
                    self.controller.toggle_pause()
                elif self.btn_step.is_clicked(event.pos):
                    self.controller.request_step()
                elif self.btn_slower.is_clicked(event.pos):
                    self.controller.decrease_speed()
                elif self.btn_faster.is_clicked(event.pos):
                    self.controller.increase_speed()
                elif self.btn_restart.is_clicked(event.pos):
                    self.controller.restart()
                elif self.btn_save.is_clicked(event.pos):
                    self.controller.save_brain()
                elif self.btn_load.is_clicked(event.pos):
                    self.controller.load_brain()

    def _scroll_panel(self, delta: int) -> None:
        """Scroll the panel content by *delta* pixels, clamping to valid range."""
        self.controller.scroll_panel(delta)
