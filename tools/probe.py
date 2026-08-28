"""Headless check: what does the engine actually return for camera frames?

Runs at a very low confidence threshold and prints every box, so an empty game
screen can be told apart from a threshold that is simply set too high.

    python tools/probe.py --model models/rps_yolo11n_2.engine
"""

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rps.detector import CLASS_NAMES, IMGSZ  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/rps_yolo11n_2.engine")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--frames", type=int, default=30)
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model, task="detect")
    print(f"model names: {getattr(model, 'names', None)}")
    print(f"our mapping: {CLASS_NAMES}  imgsz={IMGSZ}")

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise SystemExit(f"could not open camera {args.camera}")

    for i in range(args.frames):
        ok, frame = cap.read()
        if not ok:
            print("frame read failed")
            break

        results = model(frame, imgsz=IMGSZ, conf=args.conf, iou=0.45, device=0, verbose=False)
        boxes = results[0].boxes
        n = 0 if boxes is None else len(boxes)
        print(f"[{i:02d}] frame={frame.shape} boxes={n}", flush=True)
        for b, p, c in zip(
            boxes.xyxy.cpu().numpy().astype(int),
            boxes.conf.cpu().numpy(),
            boxes.cls.cpu().numpy().astype(int),
        ):
            print(f"      cls={c} conf={p:.3f} box={tuple(b)}")

    cap.release()


if __name__ == "__main__":
    main()
