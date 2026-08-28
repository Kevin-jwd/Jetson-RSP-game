"""Headless check: what does the engine actually return for camera frames?

Runs at a very low confidence threshold and prints the engine's shapes plus every
box it produces, so an empty game screen can be told apart from a threshold that
is simply set too high.

    python tools/probe.py --model models/rps_yolo11n_2.engine
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rps.detector import Detector  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/rps_yolo11n_2.engine")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--frames", type=int, default=20)
    args = parser.parse_args()

    det = Detector(args.model, conf_thres=args.conf)
    print(f"input  {det.in_name}: {tuple(det.context.get_tensor_shape(det.in_name))}")
    print(f"output {det.out_name}: {det.out.shape}")
    print(f"names: {det.names}")

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

        detections = det.detect(frame)
        # Best raw score in the whole output, ignoring the threshold entirely.
        best = float(det.out[0].T[:, 4:].max())
        print(f"[{i:02d}] frame={frame.shape} kept={len(detections)} best_score={best:.3f}", flush=True)
        for d in detections:
            print(f"      {d.label:>9} conf={d.conf:.3f} box={d.box}")

    cap.release()


if __name__ == "__main__":
    main()
