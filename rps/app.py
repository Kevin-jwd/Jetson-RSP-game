"""pygame front-end: webcam -> detection -> two-player verdict."""

from __future__ import annotations

import cv2
import numpy as np
import pygame

from .detector import Detector
from .logic import DRAW, LOSE, WIN, play

VIEW_W = 640
PANEL_H = 170

BG = (18, 18, 22)
PANEL = (28, 28, 34)
DIVIDER = (55, 55, 64)
TEXT = (235, 235, 240)
MUTED = (130, 130, 145)

RESULT_COLORS = {WIN: (90, 220, 140), LOSE: (240, 95, 95), DRAW: (225, 200, 90)}


def _font(size: int, bold: bool = False) -> pygame.font.Font:
    return pygame.font.SysFont("consolas,dejavusansmono,couriernew", size, bold=bold)


def _blit_centered(surface, text, font, color, center) -> None:
    img = font.render(text, True, color)
    surface.blit(img, img.get_rect(center=center))


class Game:
    def __init__(self, model_path: str, camera: int = 0, conf: float = 0.5, mirror: bool = True):
        self.detector = Detector(model_path, conf_thres=conf)
        self.mirror = mirror

        self.cap = cv2.VideoCapture(camera)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            raise RuntimeError(f"could not open camera {camera}")

        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("camera opened but returned no frame")

        h, w = frame.shape[:2]
        self.view = (VIEW_W, round(VIEW_W * h / w))

        pygame.init()
        pygame.display.set_caption("Rock Paper Scissors")
        self.screen = pygame.display.set_mode((self.view[0], self.view[1] + PANEL_H))
        self.clock = pygame.time.Clock()

        self.f_small = _font(18)
        self.f_label = _font(34, bold=True)
        self.f_result = _font(52, bold=True)

    def run(self) -> None:
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    elif event.key == pygame.K_m:
                        self.mirror = not self.mirror

            ok, frame = self.cap.read()
            if not ok:
                continue
            if self.mirror:
                frame = cv2.flip(frame, 1)

            detections = self.detector.detect(frame)
            round_ = play(detections)

            self.screen.fill(BG)
            self._draw_view(frame, round_)
            self._draw_panel(round_)
            pygame.display.flip()
            self.clock.tick(60)

        self.cap.release()
        pygame.quit()

    def _draw_view(self, frame: np.ndarray, round_) -> None:
        sx = self.view[0] / frame.shape[1]
        sy = self.view[1] / frame.shape[0]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        surface = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
        self.screen.blit(pygame.transform.smoothscale(surface, self.view), (0, 0))

        if round_ is None:
            return

        for det, result in ((round_.left, round_.left_result), (round_.right, round_.right_result)):
            x1, y1, x2, y2 = det.box
            rect = pygame.Rect(x1 * sx, y1 * sy, (x2 - x1) * sx, (y2 - y1) * sy)
            color = RESULT_COLORS[result]
            pygame.draw.rect(self.screen, color, rect, width=3, border_radius=6)

            caption = self.f_small.render(f"{det.label} {det.conf:.2f}", True, BG)
            tag_top = rect.top - 24 if rect.top >= 24 else rect.top
            tag = pygame.Rect(rect.left, tag_top, caption.get_width() + 12, 24)
            pygame.draw.rect(self.screen, color, tag, border_radius=4)
            self.screen.blit(caption, (tag.left + 6, tag.top + 3))

    def _draw_panel(self, round_) -> None:
        top = self.view[1]
        pygame.draw.rect(self.screen, PANEL, (0, top, self.view[0], PANEL_H))

        if round_ is None:
            _blit_centered(
                self.screen, "show two hands", self.f_label, MUTED,
                (self.view[0] // 2, top + PANEL_H // 2 - 12),
            )
            self._draw_hint(top)
            return

        pygame.draw.line(
            self.screen, DIVIDER,
            (self.view[0] // 2, top + 18), (self.view[0] // 2, top + PANEL_H - 40),
        )

        halves = (
            (self.view[0] // 4, "LEFT", round_.left, round_.left_result),
            (self.view[0] * 3 // 4, "RIGHT", round_.right, round_.right_result),
        )
        for cx, side, det, result in halves:
            _blit_centered(self.screen, side, self.f_small, MUTED, (cx, top + 28))
            _blit_centered(self.screen, det.label.upper(), self.f_label, TEXT, (cx, top + 64))
            _blit_centered(self.screen, result, self.f_result, RESULT_COLORS[result], (cx, top + 112))

        self._draw_hint(top)

    def _draw_hint(self, top: int) -> None:
        hint = f"{self.clock.get_fps():4.1f} fps   [m] mirror   [q] quit"
        self.screen.blit(self.f_small.render(hint, True, MUTED), (14, top + PANEL_H - 26))
