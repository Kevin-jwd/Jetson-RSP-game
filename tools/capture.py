"""Capture and label training frames in bursts, on the Jetson itself.

Labelling a detection set by hand is slow, but the two halves of a label have
very different costs. Boxes come from the current engine — it misreads classes,
yet still puts the box in the right place. Classes come from you, declared once
before a burst: hold a pose, record a hundred frames, and every frame is
labelled without another keypress.

    python tools/capture.py --model models/best.engine --out RPS_Real

Set the pose with 1/2/3 (TAB switches which hand in two-hand mode), then SPACE
to record. Frames where the engine does not find exactly the expected number of
hands are skipped rather than saved wrong. B records background frames with no
hands, which teach the model not to fire on faces and clutter.

Keys: 1/2/3 class · tab hand · h one/two hands · space burst · b background
      m mirror · q quit
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pygame

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rps.detector import CLASS_NAMES  # noqa: E402

# Same order the game reads back, so captured labels need no remapping.
CLASSES = [CLASS_NAMES[i] for i in sorted(CLASS_NAMES)]

VIEW_W = 800
PANEL_H = 170

BG = (18, 18, 22)
PANEL = (28, 28, 34)
TEXT = (235, 235, 240)
MUTED = (130, 130, 145)
BUTTON = (44, 44, 54)
BUTTON_HOVER = (60, 60, 74)
ACTIVE = (255, 210, 80)
RECORDING = (240, 95, 95)
OK = (110, 230, 130)
CLASS_COLORS = [(90, 200, 255), (110, 230, 130), (240, 140, 200)]


def _blit(surface, text, font, color, pos, center=False):
    img = font.render(text, True, color)
    surface.blit(img, img.get_rect(center=pos) if center else img.get_rect(topleft=pos))


class Capture:
    def __init__(self, args):
        self.out = Path(args.out)
        self.img_dir = self.out / "images" / args.split
        self.lbl_dir = self.out / "labels" / args.split
        self.img_dir.mkdir(parents=True, exist_ok=True)
        self.lbl_dir.mkdir(parents=True, exist_ok=True)

        from rps.detector import Detector
        # Deliberately low: a box in the right place is what matters here, and a
        # missed hand costs a skipped frame.
        self.detector = Detector(args.model, conf_thres=args.conf)

        self.cap = cv2.VideoCapture(args.camera)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            raise SystemExit(f"could not open camera {args.camera}")
        ok, frame = self.cap.read()
        if not ok:
            raise SystemExit("camera opened but returned no frame")

        h, w = frame.shape[:2]
        self.view = (VIEW_W, round(VIEW_W * h / w))
        self.mirror = not args.no_mirror
        self.burst = args.burst
        self.interval = args.interval
        self.patience = args.patience

        pygame.init()
        pygame.display.set_caption("RPS capture")
        self.screen = pygame.display.set_mode((self.view[0], self.view[1] + PANEL_H))
        self.clock = pygame.time.Clock()
        self.f_small = pygame.font.SysFont("consolas,dejavusansmono", 17)
        self.f_button = pygame.font.SysFont("consolas,dejavusansmono", 20, bold=True)
        self.f_label = pygame.font.SysFont("consolas,dejavusansmono", 28, bold=True)

        self.two_hands = True
        self.slots = [1, 1]        # class per hand, left to right
        self.active = 0
        self.recording = None      # None, "hands" or "background"
        self.left_to_record = 0
        self.next_shot = 0.0
        self.deadline = 0.0
        self.saved = 0
        self.skipped = 0
        self.per_class = [0] * len(CLASSES)
        self.message = ""

        y = self.view[1] + 14
        self.class_buttons = [(pygame.Rect(14 + i * 132, y, 124, 40), i) for i in range(len(CLASSES))]
        self.btn_hands = pygame.Rect(14 + len(CLASSES) * 132 + 20, y, 150, 40)
        self.btn_burst = pygame.Rect(self.view[0] - 320, y, 150, 40)
        self.btn_bg = pygame.Rect(self.view[0] - 156, y, 142, 40)

        self._count_existing()

    def _count_existing(self) -> None:
        """Continue a session instead of restarting the numbering."""
        for path in self.lbl_dir.glob("*.txt"):
            self.saved += 1
            for line in path.read_text().splitlines():
                if line.strip():
                    self.per_class[int(line.split()[0])] += 1

    # --- capture ----------------------------------------------------------

    @property
    def expected(self) -> int:
        return 2 if self.two_hands else 1

    def _assign(self, boxes) -> list[tuple[int, tuple[int, int, int, int]]]:
        """Pair the engine's boxes with the classes declared for this burst.

        Left to right, because that is the only ordering the operator can hold in
        their head while posing.
        """
        boxes = sorted(boxes, key=lambda d: (d.box[0] + d.box[2]) / 2)
        return [(self.slots[i], b.box) for i, b in enumerate(boxes)]

    def _record(self, frame) -> None:
        now = time.monotonic()
        if now > self.deadline:
            # Skipped frames do not count down, so a burst the engine never
            # matches would otherwise wait forever.
            self.recording = None
            self.message = f"gave up: {self.left_to_record} frames short, {self.skipped} skipped"
            return
        if now < self.next_shot:
            return
        self.next_shot = now + self.interval

        if self.recording == "background":
            pairs = []
        else:
            detections = self.detector.detect(frame)
            if len(detections) != self.expected:
                # Saving a frame whose hands were not all found would teach the
                # model that a real hand is background.
                self.skipped += 1
                self.message = f"skipped: saw {len(detections)}, expected {self.expected}"
                return
            pairs = self._assign(detections)

        self._save(frame, pairs)
        self.left_to_record -= 1
        if self.left_to_record <= 0:
            self.recording = None
            self.message = f"burst done: {self.saved} saved, {self.skipped} skipped"

    def _save(self, frame, pairs) -> None:
        h, w = frame.shape[:2]
        stem = f"rps_{time.strftime('%Y%m%d_%H%M%S')}_{self.saved:05d}"
        cv2.imwrite(str(self.img_dir / f"{stem}.jpg"), frame)

        lines = []
        for cls, (x1, y1, x2, y2) in pairs:
            cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
            bw, bh = abs(x2 - x1) / w, abs(y2 - y1) / h
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            self.per_class[cls] += 1
        # An empty file is a valid label: it marks a background image.
        (self.lbl_dir / f"{stem}.txt").write_text(
            ("\n".join(lines) + "\n") if lines else "", encoding="utf-8"
        )
        self.saved += 1

    def _write_yaml(self) -> None:
        (self.out / "data.yaml").write_text(
            f"path: {self.out.resolve()}\n"
            f"train: images/train\nval: images/valid\n"
            f"\nnc: {len(CLASSES)}\nnames: {CLASSES}\n",
            encoding="utf-8",
        )

    def _start(self, kind: str) -> None:
        if self.recording:
            self.recording = None
            self.message = "stopped"
            return
        self.recording = kind
        self.left_to_record = self.burst
        self.next_shot = 0.0
        self.deadline = time.monotonic() + self.burst * self.interval + self.patience
        self.message = ""

    # --- input ------------------------------------------------------------

    def _click(self, pos) -> None:
        for rect, cls in self.class_buttons:
            if rect.collidepoint(pos):
                self.slots[self.active] = cls
                return
        if self.btn_hands.collidepoint(pos):
            self.two_hands = not self.two_hands
            self.active = 0
        elif self.btn_burst.collidepoint(pos):
            self._start("hands")
        elif self.btn_bg.collidepoint(pos):
            self._start("background")

    def _key(self, event) -> bool:
        if event.key == pygame.K_q:
            return False
        if event.key == pygame.K_m:
            self.mirror = not self.mirror
        elif event.key == pygame.K_SPACE:
            self._start("hands")
        elif event.key == pygame.K_b:
            self._start("background")
        elif event.key == pygame.K_h:
            self.two_hands = not self.two_hands
            self.active = 0
        elif event.key == pygame.K_TAB and self.two_hands:
            self.active = 1 - self.active
        elif pygame.K_1 <= event.key <= pygame.K_0 + len(CLASSES):
            self.slots[self.active] = event.key - pygame.K_1
        return True

    def run(self) -> None:
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self._click(event.pos)
                elif event.type == pygame.KEYDOWN:
                    running = self._key(event)

            ok, frame = self.cap.read()
            if not ok:
                continue
            if self.mirror:
                frame = cv2.flip(frame, 1)

            # Preview always runs the detector so the boxes you would record are
            # on screen before you commit to a burst.
            detections = [] if self.recording == "background" else self.detector.detect(frame)
            if self.recording:
                self._record(frame)

            self.screen.fill(BG)
            self._draw_view(frame, detections)
            self._draw_panel()
            pygame.display.flip()
            self.clock.tick(60)

        self._write_yaml()
        self.cap.release()
        pygame.quit()
        print(f"{self.saved} images in {self.img_dir} ({self.skipped} skipped)")
        for name, n in zip(CLASSES, self.per_class):
            print(f"  {name:9s} {n} boxes")

    # --- drawing ----------------------------------------------------------

    def _draw_view(self, frame, detections) -> None:
        sx = self.view[0] / frame.shape[1]
        sy = self.view[1] / frame.shape[0]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        surface = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
        self.screen.blit(pygame.transform.smoothscale(surface, self.view), (0, 0))

        matched = len(detections) == self.expected
        for i, (cls, box) in enumerate(self._assign(detections) if matched else
                                       [(None, d.box) for d in detections]):
            x1, y1, x2, y2 = box
            rect = pygame.Rect(x1 * sx, y1 * sy, (x2 - x1) * sx, (y2 - y1) * sy)
            color = CLASS_COLORS[cls % len(CLASS_COLORS)] if cls is not None else MUTED
            pygame.draw.rect(self.screen, color, rect, width=3, border_radius=6)
            if cls is not None:
                tag = self.f_small.render(CLASSES[cls], True, BG)
                back = pygame.Rect(rect.left, max(rect.top - 22, 0), tag.get_width() + 10, 22)
                pygame.draw.rect(self.screen, color, back, border_radius=4)
                self.screen.blit(tag, (back.left + 5, back.top + 2))

        if self.recording:
            label = f"REC {self.left_to_record}"
            pygame.draw.rect(self.screen, RECORDING, (0, 0, *self.view), width=5)
            _blit(self.screen, label, self.f_label, RECORDING, (16, 12))
        elif self.recording is None and not matched:
            _blit(self.screen, f"need {self.expected} hands, see {len(detections)}",
                  self.f_small, MUTED, (16, 12))

    def _draw_panel(self) -> None:
        top = self.view[1]
        pygame.draw.rect(self.screen, PANEL, (0, top, self.view[0], PANEL_H))
        mouse = pygame.mouse.get_pos()

        for rect, cls in self.class_buttons:
            hover = rect.collidepoint(mouse)
            pygame.draw.rect(self.screen, BUTTON_HOVER if hover else BUTTON, rect, border_radius=8)
            pygame.draw.rect(self.screen, CLASS_COLORS[cls], rect, width=2, border_radius=8)
            _blit(self.screen, f"{cls + 1} {CLASSES[cls]}", self.f_button, TEXT,
                  rect.center, center=True)

        for rect, label in ((self.btn_hands, "1 HAND" if not self.two_hands else "2 HANDS"),
                            (self.btn_burst, "STOP" if self.recording else "BURST"),
                            (self.btn_bg, "BACKGROUND")):
            hover = rect.collidepoint(mouse)
            pygame.draw.rect(self.screen, BUTTON_HOVER if hover else BUTTON, rect, border_radius=8)
            _blit(self.screen, label, self.f_button, TEXT, rect.center, center=True)

        if self.two_hands:
            pose = "  ".join(
                f"{'>' if i == self.active else ' '}{side}:{CLASSES[self.slots[i]]}"
                for i, side in enumerate(("LEFT", "RIGHT"))
            )
        else:
            pose = f"HAND:{CLASSES[self.slots[0]]}"
        _blit(self.screen, pose, self.f_label, ACTIVE, (14, top + 66))

        counts = "  ".join(f"{n}:{c}" for n, c in zip(CLASSES, self.per_class))
        _blit(self.screen, f"{self.saved} images  {self.skipped} skipped   {counts}",
              self.f_small, OK, (14, top + 104))
        if self.message:
            _blit(self.screen, self.message, self.f_small, MUTED, (14, top + 124))
        _blit(self.screen, "1/2/3 class  tab hand  h one/two  space burst  b background  m mirror  q quit",
              self.f_small, MUTED, (14, top + PANEL_H - 22))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="engine used to propose boxes")
    parser.add_argument("--out", default="RPS_Real", help="dataset root to write")
    parser.add_argument("--split", default="train", choices=["train", "valid", "test"])
    parser.add_argument("--burst", type=int, default=60, help="frames per burst")
    parser.add_argument("--interval", type=float, default=0.15,
                        help="seconds between saved frames, so a burst is not one pose")
    parser.add_argument("--patience", type=float, default=20.0,
                        help="seconds of slack before a burst gives up")
    parser.add_argument("--conf", type=float, default=0.2)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=320, help="match the game's capture size")
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--no-mirror", action="store_true")
    Capture(parser.parse_args()).run()


if __name__ == "__main__":
    main()
