import os
import shutil
import subprocess

import cv2
import numpy as np


def save_frames(frames: list[np.ndarray], directory: str) -> None:
    os.makedirs(directory, exist_ok=True)
    for i, frame in enumerate(frames):
        cv2.imwrite(
            os.path.join(directory, f"frame_{i}.png"),
            cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
        )


def _require_ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            "ffmpeg is not on PATH; install it or pass --no-video / --no-gif"
        )
    return ffmpeg


def write_video(frames_dir: str, output_path: str, fps: int = 25) -> None:
    subprocess.run(
        [
            _require_ffmpeg(),
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            str(fps),
            "-i",
            os.path.join(frames_dir, "frame_%d.png"),
            "-c:v",
            "libx264",
            "-r",
            str(fps),
            "-pix_fmt",
            "yuv420p",
            output_path,
        ],
        check=True,
    )


def write_gif(
    frames_dir: str, output_path: str, fps: int = 20, width: int = 256
) -> None:
    filters = (
        f"fps={fps},scale={width}:-1:flags=lanczos,"
        "split[s0][s1];[s0]palettegen=max_colors=96[p];"
        "[s1][p]paletteuse=dither=bayer:bayer_scale=4"
    )
    subprocess.run(
        [
            _require_ffmpeg(),
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            str(fps),
            "-i",
            os.path.join(frames_dir, "frame_%d.png"),
            "-vf",
            filters,
            output_path,
        ],
        check=True,
    )
