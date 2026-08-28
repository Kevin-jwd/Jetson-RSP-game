"""Hand detection on a TensorRT engine.

Ultralytics loads the ``.engine`` built on this board and handles preprocessing
and NMS itself, so this module only turns its results into ``Detection`` values
for the game. The rest of the game only ever calls ``detect(frame)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# The engine's own class list is unreliable, so map ids to names here.
CLASS_NAMES = {0: "scissors", 1: "rock", 2: "paper"}

IMGSZ = 320


@dataclass(frozen=True)
class Detection:
    label: str
    conf: float
    box: tuple[int, int, int, int]  # x1, y1, x2, y2 in frame pixels

    @property
    def cx(self) -> float:
        return (self.box[0] + self.box[2]) / 2


class Detector:
    """YOLO11 detector running on a TensorRT engine.

    An engine is tied to the GPU and TensorRT version it was built with, so it
    has to be built on the Jetson itself.
    """

    def __init__(self, model_path: str, conf_thres: float = 0.5, iou_thres: float = 0.45):
        from ultralytics import YOLO

        self.model = YOLO(model_path, task="detect")
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

    def detect(self, frame: np.ndarray) -> list[Detection]:
        results = self.model(
            frame, imgsz=IMGSZ, conf=self.conf_thres, iou=self.iou_thres,
            device=0, verbose=False,
        )
        boxes = results[0].boxes
        if boxes is None or not len(boxes):
            return []

        xyxy = boxes.xyxy.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)

        # Unknown ids are kept and labelled with the raw id: dropping them would
        # look exactly like "nothing detected" if the engine's order ever changes.
        return [
            Detection(CLASS_NAMES.get(c, str(c)), float(p), tuple(b))
            for b, p, c in zip(xyxy, confs, classes)
        ]
