"""Arcade cabinet dressing: chunky pixel text, scanlines, blinking prompts.

The pixel look comes from rendering text small and scaling it up with nearest
neighbour, so it works with whatever fonts the board happens to have — including
Hangul, which no bitmap arcade font would cover.

Everything is one pre-rendered surface or one blit per frame. The Jetson's frame
budget belongs to inference and to pushing video over the display link.
"""

from __future__ import annotations

import pygame

# Phosphor palette: near-black tube, saturated ink.
BLACK = (8, 8, 12)
PANEL = (16, 16, 26)
TEXT = (232, 240, 220)
MUTED = (112, 120, 112)
AMBER = (255, 200, 40)
GREEN = (80, 255, 120)
MAGENTA = (255, 60, 160)
CYAN = (60, 220, 255)


def pixel_text(surface, text, font, color, center, scale: int = 3,
               shadow: tuple[int, int, int] | None = None):
    """Draw text as fat pixels. Returns the blitted rect."""
    small = font.render(text, True, color)
    big = pygame.transform.scale(
        small, (small.get_width() * scale, small.get_height() * scale)
    )
    rect = big.get_rect(center=center)
    if shadow is not None:
        dark = font.render(text, True, shadow)
        dark = pygame.transform.scale(dark, big.get_size())
        surface.blit(dark, rect.move(scale, scale))
    surface.blit(big, rect)
    return rect


class Scanlines:
    """A CRT's dark line between every scan row, pre-rendered once."""

    def __init__(self, size: tuple[int, int], gap: int = 3, alpha: int = 52):
        self.surface = pygame.Surface(size, pygame.SRCALPHA)
        for y in range(0, size[1], gap):
            self.surface.fill((0, 0, 0, alpha), (0, y, size[0], 1))

    def draw(self, surface: pygame.Surface) -> None:
        surface.blit(self.surface, (0, 0))


def blink(now: int, period: int = 700) -> bool:
    """True for half of every period, for prompts that should flash."""
    return (now // period) % 2 == 0


def draw_frame(surface, rect, color, width: int = 3) -> None:
    """A hard-edged border, the way a cabinet bezel frames the screen."""
    pygame.draw.rect(surface, color, rect, width=width)
