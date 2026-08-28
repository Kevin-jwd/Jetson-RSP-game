"""Live camera preview with detection boxes, for checking framing and speed.

The game only runs inference at the moment of the verdict, so it is no longer
the place to find out whether your hands sit in frame, how confident the model
is on each of them, or how many frames per second inference costs. This is.

    python test/preview.py --model models/rps_yolo11s.engine

Keys: `m` mirror the display, `f` toggle the two-orientation inference,
      `s` save the current frame, `q` quit.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rps.detector import Detector  # noqa: E402

COLORS = {"paper": (255, 200, 60), "rock": (90, 240, 130), "scissors": (200, 120, 255)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="models/rps_yolo11s.engine")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--classes", help="class order of the engine, comma separated")
    parser.add_argument("--no-flip-tta", action="store_true")
    args = parser.parse_args()

    det = Detector(args.model, conf_thres=args.conf,
                   class_names=args.classes.split(",") if args.classes else None,
                   flip_tta=not args.no_flip_tta)
    print(f"input {det.size}x{det.size} {det.in_dtype} | names {det.names}")

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise SystemExit(f"could not open camera {args.camera}")

    cv2.namedWindow("preview", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("preview", args.width, args.height)

    mirror, saved, fps, last = True, 0, 0.0, None
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        start = time.perf_counter()
        detections = det.detect(frame)
        infer_ms = (time.perf_counter() - start) * 1000

        # Whole-loop fps, smoothed: the number that matters is the one the game
        # would show, not how fast one inference ran.
        now = time.perf_counter()
        if last is not None:
            instant = 1.0 / max(now - last, 1e-6)
            fps = instant if fps == 0.0 else 0.9 * fps + 0.1 * instant
        last = now

        # Detect on the camera image, mirror only what is shown, exactly as the
        # game does — otherwise the preview would not match what it sees.
        if mirror:
            w = frame.shape[1]
            detections = [(d.label, d.conf, (w - d.box[2], d.box[1], w - d.box[0], d.box[3]))
                          for d in detections]
            frame = cv2.flip(frame, 1)
        else:
            detections = [(d.label, d.conf, d.box) for d in detections]

        for label, conf, (x1, y1, x2, y2) in detections:
            color = COLORS.get(label, (200, 200, 200))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, max(y1 - 8, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

        head = (f"{infer_ms:5.1f} ms  {fps:4.1f} fps  {len(detections)} det"
                f"  tta={'on' if det.flip_tta else 'off'}  mirror={'on' if mirror else 'off'}")
        cv2.putText(frame, head, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (60, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("preview", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("m"):
            mirror = not mirror
        if key == ord("f"):
            det.flip_tta = not det.flip_tta
        if key == ord("s"):
            name = f"preview_{saved:03d}.jpg"
            cv2.imwrite(name, frame)
            print(f"saved {name}")
            saved += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
