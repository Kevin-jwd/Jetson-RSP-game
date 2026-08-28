"""pygame front-end: webcam -> detection -> verdict.

The game opens on a title screen with two modes. A round is played, not polled:
the game chants 가위-바위-보, reads the hands on the beat, freezes that frame with
the verdict, and stops there — 재시도 plays again, 종료 quits.

vs Person judges two hands in the frame. vs AI reads one hand and plays a random
move against it; the AI's move is drawn over the left of the video but never goes
through the detector, since the game already knows what it picked.
"""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import pygame

from .detector import Detector
from .logic import BEATS, DRAW, LOSE, WIN, judge, play
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
SHOOT_MS = 1500        # how long to keep looking for hands after the beat

MENU, COUNTDOWN, SHOOT, RESULT = "menu", "countdown", "shoot", "result"
PERSON, AI = "person", "ai"

ASSETS = Path(__file__).resolve().parent.parent / "assets"

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
        self.f_title = self._ui_font(52, bold=True)
        self.f_big = self._ui_font(64, bold=True)
        self.f_result = self._ui_font(46, bold=True)

        self.txt_title = "가위 바위 보" if self.hangul else "ROCK PAPER SCISSORS"
        self.txt_retry = "재시도" if self.hangul else "RETRY"
        self.txt_quit = "종료" if self.hangul else "QUIT"
        self.chant = ["가위", "바위", "보!"] if self.hangul else ["GAWI", "BAWI", "BO!"]
        self.txt_nohands = "손이 안 보여요" if self.hangul else "no hands"

        cx, by = self.view[0] // 2, self.view[1] + 100
        self.btn_ai = Button(pygame.Rect(cx - 210, by, 190, 46), "vs AI")
        self.btn_person = Button(pygame.Rect(cx + 20, by, 190, 46), "vs Person")
        self.btn_retry = Button(pygame.Rect(cx - 210, by, 190, 46), self.txt_retry)
        self.btn_quit = Button(pygame.Rect(cx + 20, by, 190, 46), self.txt_quit)
        self.btn_stop = Button(pygame.Rect(self.view[0] - 146, by, 130, 46), self.txt_quit)

        self.images = self._load_images()

        self.state = MENU
        self.mode = PERSON
        self.since = 0
        self.round_ = None        # two-hand verdict, vs Person
        self.duel = None          # (human, ai move, human result, ai result), vs AI
        self.ai_move = None       # what the AI picked this round
        self.shot = None          # frame frozen at the moment of the verdict
        self.shot_dets: list = []
        self.particles = Particles()

    def _ui_font(self, size: int, bold: bool = False) -> pygame.font.Font:
        name = KOREAN_FONTS if self.hangul else MONO_FONTS
        return pygame.font.SysFont(name, size, bold=bold)

    def _load_images(self) -> dict[str, pygame.Surface]:
        """Hand images for the AI, scaled to a bit over half the view height.

        Missing files are not fatal: the AI's move is drawn as a labelled card
        instead, so the game stays playable before the artwork exists.
        """
        images: dict[str, pygame.Surface] = {}
        target = self.view[1] * 0.55
        for move in BEATS:
            for suffix in (".png", ".jpg", ".jpeg", ".webp"):
                path = ASSETS / f"{move}{suffix}"
                if not path.exists():
                    continue
                image = pygame.image.load(str(path)).convert_alpha()
                scale = target / max(image.get_width(), image.get_height())
                images[move] = pygame.transform.smoothscale(
                    image, (round(image.get_width() * scale), round(image.get_height() * scale)),
                )
                break
        missing = [m for m in BEATS if m not in images]
        if missing:
            print(f"no AI images for {missing} in {ASSETS}; drawing name cards instead")
        return images

    # --- round flow -------------------------------------------------------

    def _enter(self, state: str) -> None:
        self.state = state
        self.since = pygame.time.get_ticks()

    def _start(self, mode: str) -> None:
        self.mode = mode
        self.round_ = None
        self.duel = None
        # Picked up front so the AI's hand can be shown even when the camera
        # never got a good look at the player's.
        self.ai_move = random.choice(list(BEATS)) if mode == AI else None
        self.shot = None
        self.shot_dets = []
        self.particles.clear()
        self._enter(COUNTDOWN)

    def _to_menu(self) -> None:
        """Leaving a round returns to the title; only q closes the program."""
        self.particles.clear()
        self._enter(MENU)

    def _buttons(self) -> list[Button]:
        if self.state == MENU:
            return [self.btn_ai, self.btn_person]
        if self.state == RESULT:
            return [self.btn_retry, self.btn_quit]
        return [self.btn_stop]

    def _advance(self, detections, frame, now: int) -> None:
        elapsed = now - self.since

        if self.state == COUNTDOWN and elapsed >= BEAT_MS * len(self.chant):
            self._enter(SHOOT)

        elif self.state == SHOOT:
            decided = self._judge_ai(detections) if self.mode == AI else self._judge_person(detections)
            if decided or elapsed >= SHOOT_MS:
                self.shot = frame.copy()
                self.shot_dets = detections
                if decided:
                    self._celebrate(frame.shape)
                self._enter(RESULT)

    def _judge_person(self, detections) -> bool:
        self.round_ = play(detections)
        return self.round_ is not None

    def _judge_ai(self, detections) -> bool:
        """One hand against a random move. The AI's pick never touches the detector."""
        hands = [d for d in detections if d.label in BEATS]
        if not hands:
            return False
        human = max(hands, key=lambda d: d.conf)
        ai_result, human_result = judge(self.ai_move, human.label)
        self.duel = (human, self.ai_move, human_result, ai_result)
        return True

    def _celebrate(self, shape) -> None:
        """Fire particles over the winner. A draw has no winner to cheer."""
        sx = self.view[0] / shape[1]
        sy = self.view[1] / shape[0]

        def over(det):
            x1, y1, x2, y2 = det.box
            return ((x1 + x2) / 2 * sx, (y1 + y2) / 2 * sy)

        if self.mode == AI:
            if self.duel is None:
                return
            human, _, human_result, _ = self.duel
            if human_result == DRAW:
                return
            self.particles.burst(over(human) if human_result == WIN else self._ai_center(), RAINBOW)

        elif self.round_ is not None and self.round_.left_result != DRAW:
            winner = self.round_.left if self.round_.left_result == WIN else self.round_.right
            self.particles.burst(over(winner), RAINBOW)

    def _ai_center(self) -> tuple[float, float]:
        """The AI always plays from the left half of the view."""
        return (self.view[0] * 0.25, self.view[1] * 0.5)

    def run(self) -> None:
        running = True
        while running:
            now = pygame.time.get_ticks()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if self.state == MENU:
                        if self.btn_ai.hit(event.pos):
                            self._start(AI)
                        elif self.btn_person.hit(event.pos):
                            self._start(PERSON)
                    elif self.state == RESULT:
                        if self.btn_retry.hit(event.pos):
                            self._start(self.mode)
                        elif self.btn_quit.hit(event.pos):
                            self._to_menu()
                    elif self.btn_stop.hit(event.pos):
                        self._to_menu()
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    elif event.key == pygame.K_r and self.state == RESULT:
                        self._start(self.mode)
                    elif event.key == pygame.K_m:
                        self.mirror = not self.mirror

            ok, frame = self.cap.read()
            if not ok:
                continue
            if self.mirror:
                frame = cv2.flip(frame, 1)

            # Only the beats that matter need inference: a title screen should
            # not keep the GPU busy.
            detections = self.detector.detect(frame) if self.state in (COUNTDOWN, SHOOT) else []
            self._advance(detections, frame, now)
            self.particles.update(self.clock.get_time() / 1000.0)

            self.screen.fill(BG)
            if self.state == RESULT and self.shot is not None:
                self._draw_view(self.shot, self.shot_dets, verdict=True)
            else:
                self._draw_view(frame, detections, verdict=False)
            self._draw_panel(now, len(detections))
            pygame.display.flip()
            self.clock.tick(60)

        self.cap.release()
        pygame.quit()

    # --- drawing ----------------------------------------------------------

    def _box_colors(self, verdict: bool) -> dict[int, str]:
        if not verdict:
            return {}
        if self.mode == AI:
            return {} if self.duel is None else {id(self.duel[0]): self.duel[2]}
        if self.round_ is None:
            return {}
        return {id(self.round_.left): self.round_.left_result,
                id(self.round_.right): self.round_.right_result}

    def _draw_view(self, frame: np.ndarray, detections, verdict: bool) -> None:
        sx = self.view[0] / frame.shape[1]
        sy = self.view[1] / frame.shape[0]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        surface = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
        self.screen.blit(pygame.transform.smoothscale(surface, self.view), (0, 0))

        # Draw every detection, not only the ones that made a verdict: an
        # unmatched box is the difference between "no hand" and "one hand".
        players = self._box_colors(verdict)
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

        if verdict and self.mode == AI and self.ai_move is not None:
            self._draw_ai_move(self.ai_move, self.duel[3] if self.duel else None)

        # Clip to the video area so particles never spill onto the panel.
        self.screen.set_clip(pygame.Rect(0, 0, *self.view))
        self.particles.draw(self.screen)
        self.screen.set_clip(None)

        if self.state == MENU:
            self._draw_title()
        elif self.state == COUNTDOWN:
            self._draw_chant()

    def _draw_ai_move(self, move: str, result: str | None) -> None:
        """The AI's hand, overlaid on the left of the video.

        ``result`` is None when the player's hand was never read: the AI still
        played a move, so show it, just without a verdict colour.
        """
        center = self._ai_center()
        color = RESULT_COLORS[result] if result else MUTED
        image = self.images.get(move)

        if image is not None:
            rect = image.get_rect(center=center)
            pygame.draw.rect(self.screen, color, rect.inflate(16, 16), width=3, border_radius=10)
            self.screen.blit(image, rect)
        else:
            rect = pygame.Rect(0, 0, round(self.view[0] * 0.36), round(self.view[1] * 0.45))
            rect.center = center
            card = pygame.Surface(rect.size, pygame.SRCALPHA)
            card.fill((0, 0, 0, 170))
            self.screen.blit(card, rect)
            pygame.draw.rect(self.screen, color, rect, width=3, border_radius=10)
            _blit_centered(self.screen, move.upper(), self.f_label, color, rect.center)

        _blit_centered(self.screen, "AI", self.f_small, color, (center[0], rect.top - 14))

    def _draw_title(self) -> None:
        shade = pygame.Surface(self.view, pygame.SRCALPHA)
        shade.fill((0, 0, 0, 120))
        self.screen.blit(shade, (0, 0))
        _blit_centered(self.screen, self.txt_title, self.f_title, TEXT,
                       (self.view[0] // 2, self.view[1] // 2))

    def _draw_chant(self) -> None:
        """The chant sits over the video so players watch the camera, not the panel."""
        beat = min((pygame.time.get_ticks() - self.since) // BEAT_MS, len(self.chant) - 1)
        img = self.f_big.render(self.chant[beat], True, TEXT)
        rect = img.get_rect(center=(self.view[0] // 2, self.view[1] // 2))
        shade = pygame.Surface((rect.width + 48, rect.height + 24), pygame.SRCALPHA)
        shade.fill((0, 0, 0, 150))
        self.screen.blit(shade, shade.get_rect(center=rect.center))
        self.screen.blit(img, rect)

    def _draw_panel(self, now: int, found: int) -> None:
        top = self.view[1]
        pygame.draw.rect(self.screen, PANEL, (0, top, self.view[0], PANEL_H))
        mid = self.view[0] // 2

        if self.state == RESULT:
            sides = self._result_sides()
            if sides is None:
                _blit_centered(self.screen, self.txt_nohands, self.f_label,
                               RESULT_COLORS[LOSE], (mid, top + 46))
                if self.mode == AI and self.ai_move is not None:
                    _blit_centered(self.screen, f"AI: {self.ai_move.upper()}", self.f_small,
                                   MUTED, (mid, top + 84))
            else:
                pygame.draw.line(self.screen, DIVIDER, (mid, top + 10), (mid, top + 88))
                for cx, name, move, result in sides:
                    _blit_centered(self.screen, name, self.f_small, MUTED, (cx, top + 16))
                    _blit_centered(self.screen, move.upper(), self.f_label, TEXT, (cx, top + 44))
                    _blit_centered(self.screen, result, self.f_result,
                                   RESULT_COLORS[result], (cx, top + 82))
        elif self.state == COUNTDOWN:
            beat = min((now - self.since) // BEAT_MS, len(self.chant) - 1)
            _blit_centered(self.screen, self.chant[beat], self.f_label, ACCENT, (mid, top + 46))
        elif self.state == SHOOT:
            _blit_centered(self.screen, "...", self.f_label, ACCENT, (mid, top + 46))

        mouse = pygame.mouse.get_pos()
        for button in self._buttons():
            button.draw(self.screen, self.f_button, mouse)

        hint = f"{self.clock.get_fps():4.1f} fps  {found} det  [r] retry  [m] mirror  [q] quit"
        self.screen.blit(self.f_small.render(hint, True, MUTED), (16, top + PANEL_H - 26))

    def _result_sides(self):
        """(x, name, move, result) per side, or None when nothing was judged."""
        quarter, three_quarters = self.view[0] // 4, self.view[0] * 3 // 4

        if self.mode == AI:
            if self.duel is None:
                return None
            human, ai_move, human_result, ai_result = self.duel
            return ((quarter, "AI", ai_move, ai_result),
                    (three_quarters, "YOU", human.label, human_result))

        if self.round_ is None:
            return None
        return ((quarter, "LEFT", self.round_.left.label, self.round_.left_result),
                (three_quarters, "RIGHT", self.round_.right.label, self.round_.right_result))
