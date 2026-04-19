from __future__ import annotations

from typing import Tuple

from .pygame_compat import pygame
from .rendering import (
    COLOR_BUTTON,
    COLOR_BUTTON_HOVER,
    COLOR_BUTTON_TEXT,
    COLOR_PANEL_BORDER,
)


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

    def is_clicked(self, mouse_pos: Tuple[int, int]) -> bool:
        return self.rect.collidepoint(mouse_pos)
