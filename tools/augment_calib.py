"""Suggest augmentation values by measuring the real camera against the dataset.

Augmentation only helps where it covers the variation the model will actually
meet. So measure both sides — brightness, saturation, hand size and hand
position — and report the gain range needed to bridge the gap.

    python tools/augment_calib.py --model models/best.engine --dataset RPS_Dataset

The numbers are a starting point, not an optimum: they say "cover this much
variation", not "this maximises mAP". Only low-level differences can be closed
this way; a different background or hand shape still needs real data.

ultralytics applies hsv_h/s/v as multiplicative gains within 1 +/- value, scale
as a resize factor within 1 +/- value, and translate as a fraction of the image.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Cover this much of each distribution; the tails are usually motion blur or a
# hand halfway out of frame, which are not worth widening every augmentation for.
LOW, HIGH = 10, 90


def hsv_stats(image: np.ndarray) -> tuple[float, float]:
    """Mean saturation and value, 0..1."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return float(hsv[:, :, 1].mean()) / 255.0, float(hsv[:, :, 2].mean()) / 255.0


def label_for(image: Path) -> Path:
    """The YOLO label beside an image: .../images/<split>/x.jpg -> .../labels/<split>/x.txt

    Built from path parts rather than string replacement so it works on Windows
    paths too, where the separator is not "/".
    """
    parts = list(image.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    return Path(*parts).with_suffix(".txt")


def summarise(name: str, values: list[float]) -> tuple[float, float, float]:
    if not values:
        print(f"  {name:12s} no samples")
        return (0.0, 0.0, 0.0)
    arr = np.asarray(values, dtype=np.float64)
    lo, hi = np.percentile(arr, [LOW, HIGH])
    print(f"  {name:12s} n={len(arr):4d}  median={np.median(arr):.3f}  "
          f"p{LOW}={lo:.3f}  p{HIGH}={hi:.3f}")
    return float(np.median(arr)), float(lo), float(hi)


def gain(camera: tuple[float, float, float], dataset: tuple[float, float, float]) -> float:
    """How far the camera's range sits from the dataset's centre, as a gain."""
    ref = dataset[0]
    if ref <= 1e-6:
        return 0.0
    return max(abs(camera[1] / ref - 1.0), abs(camera[2] / ref - 1.0))


def read_camera(args) -> tuple[list, list, list, list]:
    """Brightness, saturation, hand size and hand offset seen by the webcam."""
    from rps.detector import Detector

    detector = Detector(args.model, conf_thres=args.conf) if args.model else None

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise SystemExit(f"could not open camera {args.camera}")

    sats, vals, sizes, offsets = [], [], [], []
    print(f"reading {args.frames} frames; move your hands around, near and far")
    for i in range(args.frames):
        ok, frame = cap.read()
        if not ok:
            break
        s, v = hsv_stats(frame)
        sats.append(s)
        vals.append(v)

        if detector is not None:
            h, w = frame.shape[:2]
            for det in detector.detect(frame):
                x1, y1, x2, y2 = det.box
                # Size as a fraction of the frame, comparable across resolutions.
                sizes.append(np.sqrt(abs(x2 - x1) * abs(y2 - y1) / (w * h)))
                offsets.append(max(abs((x1 + x2) / 2 / w - 0.5),
                                   abs((y1 + y2) / 2 / h - 0.5)) * 2)
        if (i + 1) % 30 == 0:
            print(f"  {i + 1}/{args.frames} frames, {len(sizes)} hands", flush=True)

    cap.release()
    return sats, vals, sizes, offsets


def read_dataset(root: Path, limit: int) -> tuple[list, list, list, list]:
    """The same measurements over the training images and their label files."""
    images = [p for p in root.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES]
    if not images:
        raise SystemExit(f"no images under {root}")
    if len(images) > limit:
        images = list(np.random.default_rng(0).choice(images, limit, replace=False))

    sats, vals, sizes, offsets = [], [], [], []
    for path in images:
        image = cv2.imread(str(path))
        if image is None:
            continue
        s, v = hsv_stats(image)
        sats.append(s)
        vals.append(v)

        # Labels are already normalised, so no image size is needed here.
        label = label_for(Path(path))
        if not label.exists():
            continue
        for line in label.read_text().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            cx, cy, bw, bh = (float(p) for p in parts[1:5])
            sizes.append(np.sqrt(bw * bh))
            offsets.append(max(abs(cx - 0.5), abs(cy - 0.5)) * 2)
    return sats, vals, sizes, offsets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, help="training dataset root")
    parser.add_argument("--model", help="engine used to find hands in the camera frames")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--frames", type=int, default=180)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--limit", type=int, default=400, help="dataset images to sample")
    args = parser.parse_args()

    cam = read_camera(args)
    data = read_dataset(Path(args.dataset), args.limit)

    print("\ncamera")
    cam_sat = summarise("saturation", cam[0])
    cam_val = summarise("brightness", cam[1])
    cam_size = summarise("hand size", cam[2]) if cam[2] else None
    cam_off = summarise("hand offset", cam[3]) if cam[3] else None

    print("dataset")
    data_sat = summarise("saturation", data[0])
    data_val = summarise("brightness", data[1])
    data_size = summarise("hand size", data[2]) if data[2] else None
    data_off = summarise("hand offset", data[3]) if data[3] else None

    print("\nsuggested train() values")
    print(f"  hsv_s={min(gain(cam_sat, data_sat), 0.9):.2f}")
    print(f"  hsv_v={min(gain(cam_val, data_val), 0.9):.2f}")
    if cam_size and data_size:
        print(f"  scale={min(gain(cam_size, data_size), 0.9):.2f}")
    else:
        print("  scale=? (pass --model so hands can be found in the camera frames)")
    if cam_off and data_off:
        # translate has to reach the offset itself, not the ratio: it is already
        # a fraction of the image.
        print(f"  translate={min(max(cam_off[2] - data_off[0], 0.0) / 2, 0.5):.2f}")
    print("\nhsv_h and degrees are not measurable this way; keep hsv_h=0.015 and set")
    print("degrees from how tilted the hands actually are on screen.")


if __name__ == "__main__":
    main()
