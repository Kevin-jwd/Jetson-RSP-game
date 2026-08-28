"""Arcade cabinet dressing: chunky pixel text, scanlines, blinking prompts.

The pixel look comes from rendering text small and scaling it up with nearest
neighbour, so it works with whatever fonts the board happens to have — including
Hangul, which no bitmap arcade font would cover.

Everything is one pre-rendered surface or one blit per frame. The Jetson's frame
budget belongs to inference and to pushing video over the display link.
"""

from __future__ import annotations

import numpy as np
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
NEON_MINT = (80, 255, 200)
NEON_PINK = (255, 70, 180)


def pixel_surface(text, font, color, scale: int,
                  shadow: tuple[int, int, int] | None = None,
                  outline: tuple[int, int, int] | None = None) -> pygame.Surface:
    """Render text as fat pixels onto its own transparent surface.

    Blockiness comes from two choices: antialiasing off, so no grey edge pixels
    survive, and ``pygame.transform.scale``, which is nearest neighbour. The
    smaller the font and the larger the scale, the coarser the result.
    """
    def block(ink):
        small = font.render(text, False, ink)
        return pygame.transform.scale(
            small, (small.get_width() * scale, small.get_height() * scale)
        )

    body = block(color)
    step = max(2, scale)
    pad = step * 2
    out = pygame.Surface((body.get_width() + pad * 2, body.get_height() + pad * 2),
                         pygame.SRCALPHA)
    at = (pad, pad)

    if outline is not None:
        edge = block(outline)
        for dx, dy in ((-step, 0), (step, 0), (0, -step), (0, step),
                       (-step, -step), (step, -step), (-step, step), (step, step)):
            out.blit(edge, (at[0] + dx, at[1] + dy))
    if shadow is not None:
        out.blit(block(shadow), (at[0] + scale, at[1] + scale))
    out.blit(body, at)
    return out


def pixel_text(surface, text, font, color, center, scale: int = 3,
               shadow: tuple[int, int, int] | None = None,
               outline: tuple[int, int, int] | None = None):
    """Draw fat-pixel text centred at ``center``. Returns the blitted rect."""
    image = pixel_surface(text, font, color, scale, shadow, outline)
    rect = image.get_rect(center=center)
    surface.blit(image, rect)
    return rect


def perspective(image: pygame.Surface, top: float = 0.5, bands: int = 20,
                squash: float = 0.55) -> pygame.Surface:
    """Tilt a surface away from the viewer, the opening-crawl way.

    Sliced into horizontal bands, each scaled narrower and shorter towards the
    top. It is not a real projection, but at this size nothing else reads as
    different — and it stays nearest-neighbour, so the pixels survive.
    """
    w, h = image.get_size()
    band_h = h / bands
    heights = [band_h * (squash + (1 - squash) * (i + 1) / bands) for i in range(bands)]
    out = pygame.Surface((w, round(sum(heights)) + bands), pygame.SRCALPHA)

    y = 0.0
    for i in range(bands):
        strip = image.subsurface((0, round(i * band_h), w,
                                  max(1, round(band_h) if i < bands - 1 else h - round(i * band_h))))
        ratio = top + (1 - top) * (i + 1) / bands
        target = (max(1, round(w * ratio)), max(1, round(heights[i])))
        out.blit(pygame.transform.scale(strip, target), (round((w - target[0]) / 2), round(y)))
        y += heights[i]
    return out


class Scanlines:
    """A CRT grid, pre-rendered once.

    Horizontal scan lines alone make the picture read as rows of pixels but not
    columns, so an aperture grille goes in as well — fainter, because the real
    thing is, and because two full-strength grids would eat the image.
    """

    def __init__(self, size: tuple[int, int], gap: int = 3, alpha: int = 52,
                 vgap: int = 3, valpha: int = 30):
        self.surface = pygame.Surface(size, pygame.SRCALPHA)
        for y in range(0, size[1], gap):
            self.surface.fill((0, 0, 0, alpha), (0, y, size[0], 1))
        for x in range(0, size[0], vgap):
            self.surface.fill((0, 0, 0, valpha), (x, 0, 1, size[1]))

    def draw(self, surface: pygame.Surface) -> None:
        surface.blit(self.surface, (0, 0))


def blink(now: int, period: int = 700) -> bool:
    """True for half of every period, for prompts that should flash."""
    return (now // period) % 2 == 0


def draw_frame(surface, rect, color, width: int = 3) -> None:
    """A hard-edged border, the way a cabinet bezel frames the screen."""
    pygame.draw.rect(surface, color, rect, width=width)


def hug_outline(image: pygame.Surface, origin, pad: int, bands: int = 12):
    """Points of a stepped outline that follows what is actually drawn.

    A plain rectangle or trapezoid around tilted text sits at odds with the
    glyphs. Walking the ink's left and right extent band by band gives a shape
    that wraps the word instead, and the steps read as pixels rather than as a
    smooth diagonal.
    """
    alpha = pygame.surfarray.array_alpha(image)      # (width, height)
    w, h = image.get_size()
    step = max(1, h // bands)
    ox, oy = origin

    left, right = [], []
    for y0 in range(0, h, step):
        y1 = min(h, y0 + step)
        cols = np.nonzero(alpha[:, y0:y1].max(axis=1))[0]
        if not len(cols):
            continue
        left.append((int(cols[0]) - pad, y0, y1))
        right.append((int(cols[-1]) + pad, y0, y1))
    if not left:
        return []

    # Take each band's extent out to its neighbours' as well. Without this the
    # horizontal segment that steps from one band to the next can cut straight
    # through the glyphs of the band it is stepping into.
    def dilate(rows, pick):
        out = []
        for i, (x, y0, y1) in enumerate(rows):
            near = [rows[j][0] for j in (i - 1, i, i + 1) if 0 <= j < len(rows)]
            out.append((pick(near), y0, y1))
        return out

    left = dilate(left, min)
    right = dilate(right, max)

    points = [(ox + right[0][0], oy + right[0][1] - pad)]   # above the first row
    for x, y0, y1 in right:                                 # down the right edge
        points += [(ox + x, oy + y0), (ox + x, oy + y1)]
    points.append((ox + right[-1][0], oy + right[-1][2] + pad))
    points.append((ox + left[-1][0], oy + left[-1][2] + pad))
    for x, y0, y1 in reversed(left):                        # back up the left edge
        points += [(ox + x, oy + y1), (ox + x, oy + y0)]
    points.append((ox + left[0][0], oy + left[0][1] - pad))
    return points


def draw_hug(surface, points, color, width: int = 3) -> None:
    if len(points) >= 3:
        pygame.draw.polygon(surface, color, points, width)


def draw_trapezoid(surface, rect, top: float, color, width: int = 3) -> None:
    """A band that follows a tilted surface: narrower at the top by ``top``.

    Used to wrap text that has been through ``perspective`` — a plain rectangle
    around it would sit at odds with the tilt.
    """
    cx = rect.centerx
    half_bottom = rect.width / 2
    half_top = half_bottom * top
    pygame.draw.polygon(surface, color, [
        (cx - half_top, rect.top), (cx + half_top, rect.top),
        (cx + half_bottom, rect.bottom), (cx - half_bottom, rect.bottom),
    ], width)
