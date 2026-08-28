"""A small particle burst for the winning hand.

Written for this game rather than taken from a library: the reference project
(github.com/orkslayergamedev/pygame-particles) ships no license, so only its
approach is borrowed — a direction vector scaled by speed and delta time, an
alpha fade, and culling once a particle is spent.

Circle surfaces are cached per colour and radius and reused with ``set_alpha``;
building one surface per particle per frame is the thing that would cost frames
on the Jetson.
"""

from __future__ import annotations

import math
import random

import pygame

GRAVITY = 300.0        # px/s^2, enough for a visible arc over one round
DRAG = 0.86            # velocity kept per second
LIFE = (2.0, 3.0)      # seconds
SPEED = (75.0, 300.0)  # px/s
RADIUS = (2, 5)

# Vivid hues that stay readable against the camera image.
RAINBOW = (
    (255, 90, 90), (255, 165, 60), (255, 225, 70), (110, 230, 130),
    (90, 200, 255), (120, 130, 255), (220, 120, 255),
)

_circles: dict[tuple[tuple[int, int, int], int], pygame.Surface] = {}


def _circle(color: tuple[int, int, int], radius: int) -> pygame.Surface:
    key = (color, radius)
    surface = _circles.get(key)
    if surface is None:
        surface = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(surface, color, (radius, radius), radius)
        _circles[key] = surface
    return surface


class Particle:
    __slots__ = ("x", "y", "vx", "vy", "color", "radius", "life", "age")

    def __init__(self, x: float, y: float, color: tuple[int, int, int]):
        angle = random.uniform(0, math.tau)
        speed = random.uniform(*SPEED)
        self.x, self.y = x, y
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.color = color
        self.radius = random.randint(*RADIUS)
        self.life = random.uniform(*LIFE)
        self.age = 0.0

    def update(self, dt: float) -> bool:
        """Advance one step; returns False once the particle is spent."""
        self.age += dt
        if self.age >= self.life:
            return False
        decay = DRAG ** dt
        self.vx *= decay
        self.vy = self.vy * decay + GRAVITY * dt
        self.x += self.vx * dt
        self.y += self.vy * dt
        return True

    def draw(self, surface: pygame.Surface) -> None:
        image = _circle(self.color, self.radius)
        image.set_alpha(int(255 * (1.0 - self.age / self.life)))
        surface.blit(image, (self.x - self.radius, self.y - self.radius))


class Particles:
    """Every live particle on screen."""

    def __init__(self):
        self.items: list[Particle] = []

    def burst(self, center, color, count: int = 110) -> None:
        """``color`` is one RGB triple, or a palette to pick from per particle."""
        palette = [color] if isinstance(color[0], int) else list(color)
        x, y = center
        self.items.extend(Particle(x, y, random.choice(palette)) for _ in range(count))

    def update(self, dt: float) -> None:
        self.items = [p for p in self.items if p.update(dt)]

    def draw(self, surface: pygame.Surface) -> None:
        for particle in self.items:
            particle.draw(surface)

    def clear(self) -> None:
        self.items.clear()
