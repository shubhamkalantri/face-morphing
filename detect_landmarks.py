#!/usr/bin/env python3
"""Regenerate the landmark CSVs from the source images.

The CSVs are committed, so this is only needed to swap in different faces.
Requires dlib and the predictor model:

    ./scripts/download_predictor.sh
    python detect_landmarks.py
"""

import argparse
import os

import cv2

from morphing.landmarks import detect_and_save, load_landmarks
from morphing.triangulation import build_simplices, plot_triangulation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data", help="directory of .jpg faces")
    parser.add_argument(
        "--predictor",
        default="models/shape_predictor_68_face_landmarks.dat",
        help="path to the dlib 68-point model",
    )
    parser.add_argument(
        "--plot-dir",
        default=None,
        help="if set, also save triangulation overlays here",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.predictor):
        raise SystemExit(
            f"predictor not found at {args.predictor}\n"
            "run ./scripts/download_predictor.sh first"
        )

    detect_and_save(args.data_dir, args.predictor)

    if args.plot_dir:
        os.makedirs(args.plot_dir, exist_ok=True)
        landmarks = {
            name: load_landmarks(os.path.join(args.data_dir, f"{name}.csv"))
            for name in ("source", "destination")
        }
        # Both overlays draw the *shared* mesh, which is the one the morph
        # runs on. Triangulating each face separately would draw two
        # different meshes, neither of which is used.
        simplices = build_simplices(landmarks["source"], landmarks["destination"])
        for name, points in landmarks.items():
            image = cv2.cvtColor(
                cv2.imread(os.path.join(args.data_dir, f"{name}.jpg")),
                cv2.COLOR_BGR2RGB,
            )
            path = os.path.join(args.plot_dir, f"{name}_triangulation.png")
            plot_triangulation(image, points, path, simplices)
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
