"""Hand detection backends.

The rest of the game only ever calls ``Detector.detect(frame) -> list[Detection]``.
``create_detector()`` picks the backend from the file extension: ``.onnx`` runs on
onnxruntime (PC), ``.engine`` on TensorRT (Jetson). The game logic and the UI are
the same either way.
"""

from __future__ import annotations

import ast
import json
import struct
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


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


class _YoloDetector:
    """Shared letterbox / decode / NMS. Backends only supply ``_infer``."""

    conf_thres: float
    iou_thres: float
    size: int
    names: list[str]

    def _infer(self, blob: np.ndarray) -> np.ndarray:
        """Run the network on a (1, 3, size, size) blob -> (1, 4 + nc, anchors)."""
        raise NotImplementedError

    def detect(self, frame: np.ndarray) -> list[Detection]:
        canvas, r, pad_x, pad_y = _letterbox(frame, self.size)
        blob = np.ascontiguousarray(
            canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        )

        out = self._infer(blob)[0].T  # (anchors, 4 + nc)

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


class OnnxDetector(_YoloDetector):
    """YOLO11 detector running on onnxruntime."""

    def __init__(self, model_path: str, conf_thres: float = 0.35, iou_thres: float = 0.45):
        import onnxruntime as ort

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

    def _infer(self, blob: np.ndarray) -> np.ndarray:
        return self.session.run(None, {self.input_name: blob})[0]


def _split_engine(path: str) -> tuple[bytes, dict]:
    """Split an ultralytics-exported .engine into (serialized engine, metadata).

    Ultralytics prefixes the engine with a 4-byte little-endian length and a JSON
    header holding ``imgsz`` and ``names``. A plain trtexec engine has no header,
    so fall back to the whole file and no metadata.
    """
    raw = Path(path).read_bytes()
    if len(raw) > 4:
        n = struct.unpack("<I", raw[:4])[0]
        if 0 < n < len(raw) - 4:
            try:
                return raw[4 + n:], json.loads(raw[4:4 + n].decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                pass
    return raw, {}


class TrtDetector(_YoloDetector):
    """YOLO11 detector running on a TensorRT engine.

    An engine is tied to the GPU and the TensorRT version it was built with, so it
    has to be produced on the Jetson itself.
    """

    def __init__(self, model_path: str, conf_thres: float = 0.35, iou_thres: float = 0.45):
        import tensorrt as trt

        from .cuda import CudaMemory

        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        serialized, meta = _split_engine(model_path)
        self.engine = trt.Runtime(trt.Logger(trt.Logger.ERROR)).deserialize_cuda_engine(serialized)
        if self.engine is None:
            raise RuntimeError(f"could not load engine {model_path}; rebuild it on this board")
        self.context = self.engine.create_execution_context()

        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        modes = [self.engine.get_tensor_mode(n) for n in names]
        self.in_name = next(n for n, m in zip(names, modes) if m == trt.TensorIOMode.INPUT)
        self.out_name = next(n for n, m in zip(names, modes) if m == trt.TensorIOMode.OUTPUT)

        in_shape = tuple(self.context.get_tensor_shape(self.in_name))
        out_shape = tuple(self.context.get_tensor_shape(self.out_name))
        self.size = in_shape[2]

        class_names = meta.get("names")
        if class_names:
            class_names = {int(k): v for k, v in class_names.items()}
            self.names = [class_names[i] for i in range(len(class_names))]
        else:
            # A trtexec engine carries no class list; fall back to this model's order.
            self.names = ["scissors", "rock", "paper"][: out_shape[1] - 4]

        self.mem = CudaMemory()
        self.out = np.empty(out_shape, dtype=np.float32)
        self.d_in = self.mem.alloc(int(np.prod(in_shape)) * 4)
        self.d_out = self.mem.alloc(self.out.nbytes)
        self.context.set_tensor_address(self.in_name, self.d_in)
        self.context.set_tensor_address(self.out_name, self.d_out)

    def _infer(self, blob: np.ndarray) -> np.ndarray:
        self.mem.htod(self.d_in, blob)
        self.context.execute_async_v3(self.mem.stream)
        self.mem.dtoh(self.out, self.d_out)
        self.mem.sync()
        return self.out


def create_detector(model_path: str, conf_thres: float = 0.35, iou_thres: float = 0.45):
    """Pick the backend from the model file extension."""
    if Path(model_path).suffix.lower() in (".engine", ".plan", ".trt"):
        return TrtDetector(model_path, conf_thres, iou_thres)
    return OnnxDetector(model_path, conf_thres, iou_thres)
