"""Hand detection backends.

The rest of the game only ever calls ``Detector.detect(frame) -> list[Detection]``.
That keeps the ONNX backend used on the PC swappable for a TensorRT one on the
Jetson without touching the game logic or the UI.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

import cv2
import numpy as np
import onnxruntime as ort


@dataclass(frozen=True)
class Detection:
    label: str
    conf: float
    box: tuple[int, int, int, int]  # x1, y1, x2, y2 in frame pixels

    @property
    def cx(self) -> float:
        return (self.box[0] + self.box[2]) / 2


def _letterbox(frame: np.ndarray, size: int):
    """Resize keeping aspect ratio, pad to a square. Returns the canvas and the
    transform needed to map boxes back onto the original frame."""
    h, w = frame.shape[:2]
    r = min(size / h, size / w)
    nh, nw = round(h * r), round(w * r)
    left, top = (size - nw) // 2, (size - nh) // 2

    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[top:top + nh, left:left + nw] = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return canvas, r, left, top


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> list[int]:
    """Class-agnostic NMS: one hand should produce exactly one label."""
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        ix1 = np.maximum(x1[i], x1[rest])
        iy1 = np.maximum(y1[i], y1[rest])
        ix2 = np.minimum(x2[i], x2[rest])
        iy2 = np.minimum(y2[i], y2[rest])
        inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou < iou_thres]
    return keep


class OnnxDetector:
    """YOLO11 detector running on onnxruntime."""

    def __init__(self, model_path: str, conf_thres: float = 0.35, iou_thres: float = 0.45):
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        spec = self.session.get_inputs()[0]
        self.input_name = spec.name
        self.size = spec.shape[2]

        # Read class names off the model instead of hardcoding them: this model
        # is ordered {0: scissors, 1: rock, 2: paper}, which is easy to get wrong.
        meta = self.session.get_modelmeta().custom_metadata_map
        names = ast.literal_eval(meta["names"])
        self.names = [names[i] for i in range(len(names))]

    def detect(self, frame: np.ndarray) -> list[Detection]:
        canvas, r, pad_x, pad_y = _letterbox(frame, self.size)
        blob = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0

        out = self.session.run(None, {self.input_name: blob})[0][0].T  # (anchors, 4 + nc)

        scores = out[:, 4:]
        conf = scores.max(axis=1)
        keep = conf >= self.conf_thres
        if not keep.any():
            return []

        out, conf, cls = out[keep], conf[keep], scores[keep].argmax(axis=1)

        cx, cy, w, h = out[:, 0], out[:, 1], out[:, 2], out[:, 3]
        boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / r
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / r

        fh, fw = frame.shape[:2]
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, fw)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, fh)

        return [
            Detection(self.names[cls[i]], float(conf[i]), tuple(boxes[i].astype(int)))
            for i in _nms(boxes, conf, self.iou_thres)
        ]
