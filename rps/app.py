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
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pygame

from .detector import Detector
from .logic import BEATS, DRAW, LOSE, WIN, Round, judge
from .particles import RAINBOW, Particles
from .space import Starfield, glow_text, vignette

VIEW_W = 640
PANEL_H = 190

# Deep space: near-black with a blue cast, neon accents.
BG = (6, 8, 20)
PANEL = (11, 14, 32)
DIVIDER = (44, 52, 96)
TEXT = (226, 232, 255)
MUTED = (118, 128, 172)
ACCENT = (110, 200, 255)
BUTTON = (22, 26, 56)
BUTTON_HOVER = (38, 46, 88)
GLOW = (70, 140, 255)

RESULT_COLORS = {WIN: (90, 240, 190), LOSE: (255, 90, 150), DRAW: (255, 205, 100)}

# Round timing, in milliseconds.
BEAT_MS = 1000         # one chant beat: 가위 / 바위 / 보!
VOTE_MS = 700          # how long to watch before judging
SHOOT_MS = 1500        # give up if no hand is seen at all in this long
RESULT_MS = 3000       # vs AI only: how long the verdict stays before the next round

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
        pygame.draw.rect(screen, BUTTON_HOVER if hover else BUTTON, self.rect, border_radius=10)
        pygame.draw.rect(screen, ACCENT if hover else DIVIDER, self.rect,
                         width=2 if hover else 1, border_radius=10)
        if hover:
            glow_text(screen, self.label, font, TEXT, self.rect.center, ACCENT, 2)
        else:
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
        self.txt_solo = "1인용" if self.hangul else "1 PLAYER"
        self.txt_duo = "2인용" if self.hangul else "2 PLAYERS"

        cx, by = self.view[0] // 2, self.view[1] + 106
        self.btn_ai = Button(pygame.Rect(cx - 210, by, 190, 46), self.txt_solo)
        self.btn_person = Button(pygame.Rect(cx + 20, by, 190, 46), self.txt_duo)
        self.btn_retry = Button(pygame.Rect(cx - 210, by, 190, 46), self.txt_retry)
        self.btn_quit = Button(pygame.Rect(cx + 20, by, 190, 46), self.txt_quit)
        self.btn_stop = Button(pygame.Rect(self.view[0] - 146, by, 130, 46), self.txt_quit)

        self.images = self._load_images()
        self.stars = Starfield((self.view[0], self.view[1] + PANEL_H), count=110)
        self.vignette = vignette(self.view)

        self.state = MENU
        self.mode = PERSON
        self.since = 0
        self.round_ = None        # two-hand verdict, vs Person
        self.duel = None          # (human, ai move, human result, ai result), vs AI
        self.ai_move = None       # what the AI picked this round
        self.shot = None          # frame frozen at the moment of the verdict
        self.shot_dets: list = []
        self.votes = None         # per hand: {label: (frames, summed conf)}
        self.last_hands: list = []
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
        self.votes = None
        self.last_hands = []
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
            # vs AI starts the next round on its own, so 재시도 would do nothing.
            return [self.btn_stop] if self.mode == AI else [self.btn_retry, self.btn_quit]
        return [self.btn_stop]

    def _advance(self, detections, frame, now: int) -> None:
        elapsed = now - self.since

        if self.state == COUNTDOWN and elapsed >= BEAT_MS * len(self.chant):
            self._enter(SHOOT)

        elif self.state == SHOOT:
            self._collect(detections)
            if (elapsed >= VOTE_MS and self.votes) or elapsed >= SHOOT_MS:
                self.shot = frame.copy()
                self.shot_dets = detections
                if self._decide():
                    self._celebrate(frame.shape)
                self._enter(RESULT)

        elif self.state == RESULT and self.mode == AI and elapsed >= RESULT_MS:
            self._start(AI)

    def _collect(self, detections) -> None:
        """Vote on each hand's move rather than trusting one frame.

        A hand opening from rock to paper passes through something the model
        reads as scissors, and the first frame that parses is exactly when that
        happens. Counting frames over VOTE_MS drops those in-between reads.
        """
        hands = [d for d in detections if d.label in BEATS]
        if self.mode == AI:
            hands = hands[:1] if len(hands) == 1 else (
                [max(hands, key=lambda d: d.conf)] if hands else [])
        elif len(hands) >= 2:
            hands = sorted(hands, key=lambda d: d.conf, reverse=True)[:2]
            hands.sort(key=lambda d: d.cx)
        else:
            hands = []
        if not hands:
            return

        if self.votes is None:
            self.votes = [{} for _ in hands]
        if len(self.votes) != len(hands):
            return

        # Hands are ordered left to right, so slot 0 stays the same player even
        # as the boxes move between frames.
        for slot, det in zip(self.votes, hands):
            frames, conf = slot.get(det.label, (0, 0.0))
            slot[det.label] = (frames + 1, conf + det.conf)
        self.last_hands = hands

    def _decide(self) -> bool:
        """Turn the votes into a verdict. Ties fall to the higher summed confidence."""
        if not self.votes:
            return False

        labels = [max(slot.items(), key=lambda kv: kv[1])[0] for slot in self.votes]
        hands = [replace(det, label=label) for det, label in zip(self.last_hands, labels)]
        # Show the boxes that were voted on, with the labels the vote settled on.
        self.shot_dets = hands

        if self.mode == AI:
            human = hands[0]
            ai_result, human_result = judge(self.ai_move, human.label)
            self.duel = (human, self.ai_move, human_result, ai_result)
        else:
            left, right = hands
            left_result, right_result = judge(left.label, right.label)
            self.round_ = Round(left, right, left_result, right_result)
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
            dt = self.clock.get_time() / 1000.0
            self.particles.update(dt)
            self.stars.update(dt)

            self.screen.fill(BG)
            self.stars.draw(self.screen)
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
        self.screen.blit(self.vignette, (0, 0))

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
        shade.fill((4, 6, 18, 170))
        self.screen.blit(shade, (0, 0))
        # Stars again over the dimmed video, so the title floats in space.
        self.stars.draw(self.screen)
        glow_text(self.screen, self.txt_title, self.f_title, TEXT,
                  (self.view[0] // 2, self.view[1] // 2), GLOW, 4)

    def _draw_chant(self) -> None:
        """The chant sits over the video so players watch the camera, not the panel."""
        beat = min((pygame.time.get_ticks() - self.since) // BEAT_MS, len(self.chant) - 1)
        center = (self.view[0] // 2, self.view[1] // 2)
        img = self.f_big.render(self.chant[beat], True, TEXT)
        rect = img.get_rect(center=center)
        shade = pygame.Surface((rect.width + 56, rect.height + 28), pygame.SRCALPHA)
        shade.fill((4, 6, 18, 165))
        self.screen.blit(shade, shade.get_rect(center=rect.center))
        glow_text(self.screen, self.chant[beat], self.f_big, TEXT, center, ACCENT, 4)

    def _draw_panel(self, now: int, found: int) -> None:
        top = self.view[1]
        plate = pygame.Surface((self.view[0], PANEL_H), pygame.SRCALPHA)
        plate.fill((*PANEL, 232))
        self.screen.blit(plate, (0, top))
        pygame.draw.line(self.screen, ACCENT, (0, top), (self.view[0], top))
        mid = self.view[0] // 2

        if self.state == RESULT:
            sides = self._result_sides()
            if sides is None:
                _blit_centered(self.screen, self.txt_nohands, self.f_label,
                               RESULT_COLORS[LOSE], (mid, top + 44))
                if self.mode == AI and self.ai_move is not None:
                    _blit_centered(self.screen, f"AI: {self.ai_move.upper()}", self.f_small,
                                   MUTED, (mid, top + 84))
            else:
                pygame.draw.line(self.screen, DIVIDER, (mid, top + 8), (mid, top + 84))
                for cx, name, move, result in sides:
                    _blit_centered(self.screen, name, self.f_small, MUTED, (cx, top + 12))
                    _blit_centered(self.screen, move.upper(), self.f_label, TEXT, (cx, top + 38))
                    glow_text(self.screen, result, self.f_result,
                              RESULT_COLORS[result], (cx, top + 74), RESULT_COLORS[result], 3)
        elif self.state == COUNTDOWN:
            beat = min((now - self.since) // BEAT_MS, len(self.chant) - 1)
            _blit_centered(self.screen, self.chant[beat], self.f_label, ACCENT, (mid, top + 46))
        elif self.state == SHOOT:
            voted = max((sum(f for f, _ in slot.values()) for slot in self.votes or []), default=0)
            _blit_centered(self.screen, "." * (1 + voted % 3), self.f_label, ACCENT, (mid, top + 46))

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
