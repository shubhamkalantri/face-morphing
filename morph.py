import argparse
import os

import cv2
import numpy as np

from morphing import meshless, rendering, triangulation
from morphing.landmarks import (
    interpolate_landmarks,
    load_landmarks,
    rescale_landmarks,
)

DEFAULT_SIZE = 512

def load_image(path: str, size: int) -> np.ndarray:
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]
    if height != width:
        raise ValueError(
            f"{path} is {width}x{height}; the morph pipeline requires a square "
            "image (the landmarks are rescaled by a single factor)"
        )
    if size != height or size != width:
        image = cv2.resize(image, (size, size))
    return image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        choices=["triangulation", "meshless"],
        default="triangulation",
        help="warping method (default: triangulation)",
    )
    parser.add_argument("--data-dir", default="data", help="images and landmark CSVs")
    parser.add_argument(
        "--frames-dir", default="frames", help="where frames are written"
    )
    parser.add_argument("--results-dir", default="results", help="where GIF/MP4 go")
    parser.add_argument(
        "--size",
        type=int,
        default=None,
        help="working resolution (default: 512 for triangulation, 256 for meshless, "
        "which is far slower per pixel)",
    )
    parser.add_argument("--frames", type=int, default=51, help="number of frames")
    parser.add_argument(
        "--fps", type=int, default=25, help="frame rate of the video and the GIF"
    )
    parser.add_argument(
        "--gif-width",
        type=int,
        default=None,
        help="width of the GIF in pixels (default: the working resolution)",
    )
    parser.add_argument(
        "--falloff",
        type=float,
        default=2.0,
        help="meshless distance falloff exponent a (default: 2.0)",
    )
    parser.add_argument("--no-video", action="store_true", help="skip the MP4")
    parser.add_argument("--no-gif", action="store_true", help="skip the GIF")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    size = args.size
    if size is None:
        size = DEFAULT_SIZE if args.method == "triangulation" else 256

    source_image = load_image(os.path.join(args.data_dir, "source.jpg"), size)
    destination_image = load_image(os.path.join(args.data_dir, "destination.jpg"), size)
    
    source_points = rescale_landmarks(
        load_landmarks(os.path.join(args.data_dir, "source.csv")), DEFAULT_SIZE, size
    )
    destination_points = rescale_landmarks(
        load_landmarks(os.path.join(args.data_dir, "destination.csv")),
        DEFAULT_SIZE,
        size,
    )

    for name, points in (
        ("source", source_points),
        ("destination", destination_points),
    ):
        if points.min() < 0 or points.max() > size - 1:
            raise ValueError(
                f"{name} landmarks fall outside a {size}x{size} image "
                f"(range {points.min():.1f}..{points.max():.1f}); the CSVs are "
                f"assumed to have been detected at {DEFAULT_SIZE}x{DEFAULT_SIZE}"
            )

    alphas, intermediate_points = interpolate_landmarks(
        source_points, destination_points, args.frames
    )

    if args.method == "triangulation":
        frames = triangulation.morph_sequence(
            source_image,
            destination_image,
            source_points,
            destination_points,
            intermediate_points,
            alphas,
        )
    else:
        frames = meshless.morph_sequence(
            source_image,
            destination_image,
            source_points,
            destination_points,
            intermediate_points,
            alphas,
            args.falloff,
        )

    frames_dir = os.path.join(args.frames_dir, args.method)
    rendering.save_frames(frames, frames_dir)
    print(f"wrote {len(frames)} frames to {frames_dir}/")

    os.makedirs(args.results_dir, exist_ok=True)
    if not args.no_video:
        path = os.path.join(args.results_dir, f"{args.method}.mp4")
        rendering.write_video(frames_dir, path, args.fps)
        print(f"wrote {path}")
    if not args.no_gif:
        path = os.path.join(args.results_dir, f"{args.method}.gif")
        rendering.write_gif(
            frames_dir, path, args.fps, args.gif_width if args.gif_width else size
        )
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
