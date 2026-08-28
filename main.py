"""Rock-paper-scissors on a webcam: one hand against the machine, or two hands."""

import argparse

from rps.app import Game


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/rps_yolo11s.engine")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--no-mirror", action="store_true")
    parser.add_argument(
        "--no-flip-tta", action="store_true",
        help="detect only on the camera image, instead of both orientations",
    )
    parser.add_argument(
        "--classes",
        help="class order of the engine, e.g. scissors,rock,paper for an older model",
    )
    args = parser.parse_args()

    classes = args.classes.split(",") if args.classes else None
    Game(args.model, args.camera, args.conf, mirror=not args.no_mirror,
         class_names=classes, flip_tta=not args.no_flip_tta).run()


if __name__ == "__main__":
    main()
