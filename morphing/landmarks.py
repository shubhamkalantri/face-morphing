import os
from glob import glob

import numpy as np
import pandas as pd


def boundary_points(width: int, height: int) -> np.ndarray:
    max_x, max_y = width - 1, height - 1
    mid_x, mid_y = max_x // 2, max_y // 2
    return np.array(
        [
            [0, 0],
            [0, mid_y],
            [mid_x, 0],
            [max_x, mid_y],
            [mid_x, max_y],
            [max_x, max_y],
            [0, max_y],
            [max_x, 0],
        ]
    )


def detect_landmarks(image_path: str, predictor_path: str) -> np.ndarray:
    import dlib

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    image = dlib.load_rgb_image(image_path)
    faces = detector(image, 0)
    if not faces:
        raise ValueError(f"No face detected in {image_path}")

    shape = predictor(image, faces[0])
    points = [(shape.part(i).x, shape.part(i).y) for i in range(68)]

    height, width = image.shape[:2]
    return np.vstack((np.array(points), boundary_points(width, height)))


def detect_and_save(faces_dir: str, predictor_path: str) -> None:
    for image_path in sorted(glob(os.path.join(faces_dir, "*.jpg"))):
        points = detect_landmarks(image_path, predictor_path)
        csv_path = os.path.splitext(image_path)[0] + ".csv"
        save_landmarks(points, csv_path)
        print(f"{image_path} -> {csv_path} ({len(points)} points)")


def save_landmarks(points: np.ndarray, csv_path: str) -> None:
    pd.DataFrame(points, columns=["x", "y"]).to_csv(csv_path, index=False)


def load_landmarks(csv_path: str) -> np.ndarray:
    frame = pd.read_csv(csv_path)
    return np.stack((frame["x"].to_numpy(), frame["y"].to_numpy())).T


def rescale_landmarks(points: np.ndarray, from_size: int, to_size: int) -> np.ndarray:
    return np.asarray(points, dtype=float) * ((to_size - 1) / (from_size - 1))


def interpolate_landmarks(
    source: np.ndarray, destination: np.ndarray, n_frames: int
) -> tuple[np.ndarray, np.ndarray]:
    alphas = np.linspace(0, 1, n_frames)
    intermediate = np.array(
        [(1 - alpha) * source + alpha * destination for alpha in alphas]
    )
    return alphas, intermediate
