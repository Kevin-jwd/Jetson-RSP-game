"""Headless check: what does the engine actually return for camera frames?

Runs at a very low confidence threshold and prints the engine's shapes plus every
box it produces, so an empty game screen can be told apart from a threshold that
is simply set too high.

    python tools/probe.py --model models/rps_yolo11n_2.engine
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rps.detector import Detector  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/rps_yolo11s.engine")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--delay", type=float, default=3.0, help="seconds to get a hand in view")
    parser.add_argument("--classes", help="class order of the engine, comma separated")
    parser.add_argument("--no-flip-tta", action="store_true")
    args = parser.parse_args()

    det = Detector(args.model, conf_thres=args.conf,
                   class_names=args.classes.split(",") if args.classes else None,
                   flip_tta=not args.no_flip_tta)
    print(f"input  {det.in_name}: {tuple(det.context.get_tensor_shape(det.in_name))} {det.in_dtype}")
    print(f"output {det.out_name}: {det.out.shape} {det.out_dtype}")
    print(f"names: {det.names}")

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise SystemExit(f"could not open camera {args.camera}")

    # Does the engine react to its input at all? Two very different frames must
    # not produce the same score; if they do, the blob never reaches the engine.
    black = np.zeros((240, 320, 3), np.uint8)
    noise = np.random.randint(0, 256, (240, 320, 3), dtype=np.uint8)
    det.detect(black)
    black_score = float(det.out[0].T[:, 4:].max())
    det.detect(noise)
    noise_score = float(det.out[0].T[:, 4:].max())
    print(f"sanity: black={black_score:.4f} noise={noise_score:.4f}"
          f"{'  <-- input is being ignored' if black_score == noise_score else ''}")

    if args.delay:
        print(f"show your hands... starting in {args.delay:.0f}s", flush=True)
        for _ in range(int(args.delay * 30)):  # keep draining so frames stay fresh
            cap.read()
        time.sleep(0.1)

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
