"""pygame front-end: webcam -> detection -> two-player verdict.

A round is played, not polled: press 시작 and the game chants 가위-바위-보, reads
both hands on the beat, freezes that frame with the verdict, then starts the next
round. It keeps going until 종료.
"""

from __future__ import annotations

import cv2
import numpy as np
import pygame

from .detector import Detector
from .logic import DRAW, LOSE, WIN, play
from .particles import RAINBOW, Particles

VIEW_W = 640
PANEL_H = 190

BG = (18, 18, 22)
PANEL = (28, 28, 34)
DIVIDER = (55, 55, 64)
TEXT = (235, 235, 240)
MUTED = (130, 130, 145)
ACCENT = (90, 160, 240)
BUTTON = (44, 44, 54)
BUTTON_HOVER = (60, 60, 74)

RESULT_COLORS = {WIN: (90, 220, 140), LOSE: (240, 95, 95), DRAW: (225, 200, 90)}

# Round timing, in milliseconds.
BEAT_MS = 700          # one chant beat: 가위 / 바위 / 보!
SHOOT_MS = 1500        # how long to keep looking for two hands after the beat
RESULT_MS = 2500       # how long the frozen verdict stays up

IDLE, COUNTDOWN, SHOOT, RESULT = "idle", "countdown", "shoot", "result"

# Fonts that can draw Hangul, in the order they are usually installed.
KOREAN_FONTS = "notosanscjkkr,notosanskr,nanumgothic,nanumbarungothic,malgungothic,applegothic"
MONO_FONTS = "consolas,dejavusansmono,couriernew"


def _blit_centered(surface, text, font, color, center) -> None:
    img = font.render(text, True, color)
    surface.blit(img, img.get_rect(center=center))


class Button:
    def __init__(self, rect: pygame.Rect, label: str):
        self.rect = rect
        self.label = label

    def draw(self, screen, font, mouse) -> None:
        hover = self.rect.collidepoint(mouse)
        pygame.draw.rect(screen, BUTTON_HOVER if hover else BUTTON, self.rect, border_radius=8)
        pygame.draw.rect(screen, DIVIDER, self.rect, width=1, border_radius=8)
        _blit_centered(screen, self.label, font, TEXT, self.rect.center)

    def hit(self, pos) -> bool:
        return self.rect.collidepoint(pos)


class Game:
    def __init__(self, model_path: str, camera: int = 0, conf: float = 0.5, mirror: bool = True,
                 class_names: list[str] | None = None):
        self.detector = Detector(model_path, conf_thres=conf, class_names=class_names)
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

        # Without a Hangul font the chant would render as empty boxes, so romanise.
        self.hangul = pygame.font.match_font(KOREAN_FONTS) is not None
        self.f_small = pygame.font.SysFont(MONO_FONTS, 18)
        self.f_button = self._ui_font(22, bold=True)
        self.f_label = self._ui_font(34, bold=True)
        self.f_big = self._ui_font(64, bold=True)
        self.f_result = self._ui_font(46, bold=True)

        self.txt_start = "시작" if self.hangul else "START"
        self.txt_stop = "정지" if self.hangul else "STOP"
        self.txt_quit = "종료" if self.hangul else "QUIT"
        self.chant = ["가위", "바위", "보!"] if self.hangul else ["GAWI", "BAWI", "BO!"]
        self.txt_press = "시작을 누르세요" if self.hangul else "press START"
        self.txt_nohands = "두 손이 안 보여요" if self.hangul else "no two hands"

        by = self.view[1] + PANEL_H - 52
        self.btn_play = Button(pygame.Rect(16, by, 130, 40), self.txt_start)
        self.btn_quit = Button(pygame.Rect(self.view[0] - 146, by, 130, 40), self.txt_quit)

        self.state = IDLE
        self.since = 0
        self.playing = False
        self.round_ = None
        self.shot = None          # frame frozen at the moment of the verdict
        self.shot_dets: list = []
        self.particles = Particles()

    def _ui_font(self, size: int, bold: bool = False) -> pygame.font.Font:
        name = KOREAN_FONTS if self.hangul else MONO_FONTS
        return pygame.font.SysFont(name, size, bold=bold)

    # --- round flow -------------------------------------------------------

    def _enter(self, state: str) -> None:
        self.state = state
        self.since = pygame.time.get_ticks()

    def _toggle_play(self) -> None:
        self.playing = not self.playing
        self.btn_play.label = self.txt_stop if self.playing else self.txt_start
        self._enter(COUNTDOWN if self.playing else IDLE)
        if not self.playing:
            self.round_ = None
            self.shot = None
            self.particles.clear()

    def _advance(self, detections, frame, now: int) -> None:
        elapsed = now - self.since

        if self.state == COUNTDOWN and elapsed >= BEAT_MS * len(self.chant):
            self._enter(SHOOT)

        elif self.state == SHOOT:
            round_ = play(detections)
            if round_ is not None:
                self.round_ = round_
                self.shot = frame.copy()
                self.shot_dets = detections
                self._celebrate(round_, frame.shape)
                self._enter(RESULT)
            elif elapsed >= SHOOT_MS:
                self.round_ = None
                self.shot = frame.copy()
                self.shot_dets = detections
                self._enter(RESULT)

        elif self.state == RESULT and elapsed >= RESULT_MS:
            self.particles.clear()
            self._enter(COUNTDOWN if self.playing else IDLE)

    def _celebrate(self, round_, shape) -> None:
        """Fire particles over the winning hand. A draw has no winner to cheer."""
        if round_.left_result == DRAW:
            return
        winner = round_.left if round_.left_result == WIN else round_.right
        sx = self.view[0] / shape[1]
        sy = self.view[1] / shape[0]
        x1, y1, x2, y2 = winner.box
        self.particles.burst(((x1 + x2) / 2 * sx, (y1 + y2) / 2 * sy), RAINBOW)

    def run(self) -> None:
        running = True
        while running:
            now = pygame.time.get_ticks()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if self.btn_quit.hit(event.pos):
                        running = False
                    elif self.btn_play.hit(event.pos):
                        self._toggle_play()
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    elif event.key == pygame.K_SPACE:
                        self._toggle_play()
                    elif event.key == pygame.K_m:
                        self.mirror = not self.mirror

            ok, frame = self.cap.read()
            if not ok:
                continue
            if self.mirror:
                frame = cv2.flip(frame, 1)

            # Only the beats that matter need inference: idling on a menu should
            # not keep the GPU busy.
            detections = self.detector.detect(frame) if self.state in (COUNTDOWN, SHOOT) else []
            self._advance(detections, frame, now)
            self.particles.update(self.clock.get_time() / 1000.0)

            self.screen.fill(BG)
            if self.state == RESULT and self.shot is not None:
                self._draw_view(self.shot, self.shot_dets, self.round_)
            else:
                self._draw_view(frame, detections, None)
            self._draw_panel(now, len(detections))
            pygame.display.flip()
            self.clock.tick(60)

        self.cap.release()
        pygame.quit()

    # --- drawing ----------------------------------------------------------

    def _draw_view(self, frame: np.ndarray, detections, round_) -> None:
        sx = self.view[0] / frame.shape[1]
        sy = self.view[1] / frame.shape[0]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        surface = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
        self.screen.blit(pygame.transform.smoothscale(surface, self.view), (0, 0))

        # Draw every detection, not just the two that made a round: an unmatched
        # box is the difference between "no hand seen" and "only one hand seen".
        players = {} if round_ is None else {
            id(round_.left): round_.left_result, id(round_.right): round_.right_result,
        }

        for det in detections:
            x1, y1, x2, y2 = det.box
            rect = pygame.Rect(x1 * sx, y1 * sy, (x2 - x1) * sx, (y2 - y1) * sy)
            result = players.get(id(det))
            color = RESULT_COLORS[result] if result else MUTED
            pygame.draw.rect(self.screen, color, rect, width=3, border_radius=6)

            caption = self.f_small.render(f"{det.label} {det.conf:.2f}", True, BG)
            tag_top = rect.top - 24 if rect.top >= 24 else rect.top
            tag = pygame.Rect(rect.left, tag_top, caption.get_width() + 12, 24)
            pygame.draw.rect(self.screen, color, tag, border_radius=4)
            self.screen.blit(caption, (tag.left + 6, tag.top + 3))

        # Clip to the video area so particles never spill onto the panel.
        self.screen.set_clip(pygame.Rect(0, 0, *self.view))
        self.particles.draw(self.screen)
        self.screen.set_clip(None)

        if self.state == COUNTDOWN:
            self._draw_chant()

    def _draw_chant(self) -> None:
        """The chant sits over the video so players watch the camera, not the panel."""
        beat = min((pygame.time.get_ticks() - self.since) // BEAT_MS, len(self.chant) - 1)
        word = self.chant[beat]

        img = self.f_big.render(word, True, TEXT)
        rect = img.get_rect(center=(self.view[0] // 2, self.view[1] // 2))
        shade = pygame.Surface((rect.width + 48, rect.height + 24), pygame.SRCALPHA)
        shade.fill((0, 0, 0, 150))
        self.screen.blit(shade, shade.get_rect(center=rect.center))
        self.screen.blit(img, rect)

    def _draw_panel(self, now: int, found: int) -> None:
        top = self.view[1]
        pygame.draw.rect(self.screen, PANEL, (0, top, self.view[0], PANEL_H))
        mid = self.view[0] // 2

        if self.state == RESULT and self.round_ is not None:
            pygame.draw.line(
                self.screen, DIVIDER, (mid, top + 14), (mid, top + PANEL_H - 66),
            )
            halves = (
                (self.view[0] // 4, "LEFT", self.round_.left, self.round_.left_result),
                (self.view[0] * 3 // 4, "RIGHT", self.round_.right, self.round_.right_result),
            )
            for cx, side, det, result in halves:
                _blit_centered(self.screen, side, self.f_small, MUTED, (cx, top + 24))
                _blit_centered(self.screen, det.label.upper(), self.f_label, TEXT, (cx, top + 56))
                _blit_centered(
                    self.screen, result, self.f_result, RESULT_COLORS[result], (cx, top + 100),
                )
        else:
            if self.state == RESULT:
                message, color = self.txt_nohands, RESULT_COLORS[LOSE]
            elif self.state == SHOOT:
                message, color = "...", ACCENT
            elif self.state == COUNTDOWN:
                message, color = self.chant[
                    min((now - self.since) // BEAT_MS, len(self.chant) - 1)
                ], ACCENT
            else:
                message, color = self.txt_press, MUTED
            _blit_centered(self.screen, message, self.f_label, color, (mid, top + 62))

        mouse = pygame.mouse.get_pos()
        self.btn_play.draw(self.screen, self.f_button, mouse)
        self.btn_quit.draw(self.screen, self.f_button, mouse)

        hint = f"{self.clock.get_fps():4.1f} fps  {found} det  [space] start/stop  [m] mirror  [q] quit"
        self.screen.blit(self.f_small.render(hint, True, MUTED), (16, top + PANEL_H - 84))
