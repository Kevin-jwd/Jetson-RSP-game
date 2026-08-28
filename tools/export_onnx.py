"""Export a trained YOLO detector to the ONNX this game's engine is built from.

Run this off the board (Colab or a PC) — it needs ultralytics and torch, which
the Jetson deliberately does not. The result is a static 1x3xNxN ONNX whose
output is (1, 4 + nc, anchors), which `trtexec` turns into the engine that
`rps/detector.py` loads.

    python tools/export_onnx.py --weights runs/detect/train/weights/best.pt

Then on the Jetson:

    /usr/src/tensorrt/bin/trtexec --onnx=models/rps_yolo11n.onnx \\
        --saveEngine=models/rps_yolo11n_2.engine --fp16
"""

import argparse
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, help="trained .pt weights")
    parser.add_argument("--imgsz", type=int, default=320, help="must match the game's engine")
    parser.add_argument("--opset", type=int, default=12)
    parser.add_argument("--out", default="models/rps_yolo11n.onnx")
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)
    print(f"classes: {model.names}")

    # dynamic=False keeps the shape static: the detector sizes its device buffers
    # off the engine's shape, and a fixed one keeps that unambiguous.
    exported = model.export(
        format="onnx", imgsz=args.imgsz, opset=args.opset, simplify=True, dynamic=False,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if Path(exported).resolve() != out.resolve():
        shutil.move(str(exported), out)
    print(f"wrote {out}")

    names = [model.names[i] for i in range(len(model.names))]
    print(f"\nclass order is {names}")
    print("if that differs from CLASS_NAMES in rps/detector.py, update it there.")


if __name__ == "__main__":
    main()
