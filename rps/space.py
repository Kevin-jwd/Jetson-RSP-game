"""Space dressing for the UI: a drifting starfield, neon glow, and a vignette.

Everything here is pre-rendered or drawn with plain rectangles. The Jetson has to
spend its frame budget on inference and on pushing video over the display link,
so the decoration must not compete for it.
"""

from __future__ import annotations

import random

import pygame

STAR_COLORS = ((255, 255, 255), (200, 220, 255), (255, 230, 210), (180, 200, 255))


class Starfield:
    """Slowly drifting, twinkling stars that fill whatever rect they are drawn in."""

    def __init__(self, size: tuple[int, int], count: int = 90):
        self.w, self.h = size
        self.stars = []
        for _ in range(count):
            self.stars.append([
                random.uniform(0, self.w),          # x
                random.uniform(0, self.h),          # y
                random.choice((1, 1, 1, 2)),        # radius, mostly single pixels
                random.uniform(0.25, 1.0),          # brightness
                random.uniform(0, 6.28),            # twinkle phase
                random.uniform(0.6, 2.2),           # twinkle speed
                random.choice(STAR_COLORS),
            ])
        self.drift = 6.0  # px/s, just enough to notice over a round

    def update(self, dt: float) -> None:
        for star in self.stars:
            star[4] += star[5] * dt
            star[1] += self.drift * dt * star[3]
            if star[1] > self.h:
                star[1] -= self.h
                star[0] = random.uniform(0, self.w)

    def draw(self, surface: pygame.Surface, origin: tuple[int, int] = (0, 0)) -> None:
        import math

        ox, oy = origin
        for x, y, r, base, phase, _, color in self.stars:
            # sin twinkle, never fully dark: a blinking field is distracting.
            level = base * (0.55 + 0.45 * math.sin(phase))
            c = tuple(min(255, int(v * level)) for v in color)
            surface.fill(c, (ox + int(x), oy + int(y), r, r))


def glow_text(surface, text, font, color, center, glow=(60, 160, 255), spread: int = 3) -> None:
    """Text with a neon halo, drawn as a few offset copies rather than a blur."""
    image = font.render(text, True, color)
    rect = image.get_rect(center=center)

    halo = font.render(text, True, glow)
    halo.set_alpha(70)
    for dx, dy in ((-spread, 0), (spread, 0), (0, -spread), (0, spread),
                   (-spread, -spread), (spread, spread)):
        surface.blit(halo, rect.move(dx, dy))
    surface.blit(image, rect)


def vignette(size: tuple[int, int], strength: int = 150) -> pygame.Surface:
    """A darkened border for the video, so it reads as a porthole.

    Built once as a stack of nested rectangles: cheap, and the banding is
    invisible at this strength.
    """
    w, h = size
    surface = pygame.Surface(size, pygame.SRCALPHA)
    steps = 24
    band = max(w, h) // 12
    for i in range(steps):
        alpha = int(strength * (1 - i / steps) ** 2)
        inset = i * band // steps
        pygame.draw.rect(
            surface, (0, 0, 0, alpha),
            (inset, inset, w - inset * 2, h - inset * 2),
            width=max(1, band // steps + 1), border_radius=10,
        )
    return surface
