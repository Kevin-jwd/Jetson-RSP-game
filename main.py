"""Two-player rock-paper-scissors on a webcam."""

import argparse

from rps.app import Game


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/rps_yolo11n_2.engine")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--no-mirror", action="store_true")
    args = parser.parse_args()

    Game(args.model, args.camera, args.conf, mirror=not args.no_mirror).run()


if __name__ == "__main__":
    main()
