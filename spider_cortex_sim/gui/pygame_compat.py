from __future__ import annotations

import sys

try:
    import pygame
except ModuleNotFoundError as exc:
    if exc.name != "pygame":
        raise
    pygame = None


def require_pygame():
    if pygame is None:
        print(
            "pygame is required for the graphical interface.\n"
            "Install it with:  pip install pygame-ce",
            file=sys.stderr,
        )
        sys.exit(1)
    return pygame


__all__ = ["pygame", "require_pygame"]
