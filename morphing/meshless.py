import numpy as np
from tqdm import tqdm

from .interpolation import bilinear_sample

EPSILON = 1e-6
_CHUNK = 16384


def kappa(points: np.ndarray, landmarks: np.ndarray, alpha: float = 2.0) -> np.ndarray:
    squared_distance = (points[:, 0, None] - landmarks[None, :, 0]) ** 2
    squared_distance += (points[:, 1, None] - landmarks[None, :, 1]) ** 2
    return 1.0 / (squared_distance**alpha + EPSILON)


def displacement_field(
    points: np.ndarray,
    source_points: np.ndarray,
    target_points: np.ndarray,
    alpha: float = 2.0,
) -> np.ndarray:
    offsets = target_points - source_points
    displacements = np.empty((points.shape[0], 2))
    total = np.empty((points.shape[0], 1))

    for start in range(0, points.shape[0], _CHUNK):
        block = points[start : start + _CHUNK]
        weights = kappa(block, source_points, alpha)
        total[start : start + _CHUNK] = np.sum(weights, axis=1, keepdims=True)
        displacements[start : start + _CHUNK] = weights @ offsets

    return np.divide(
        displacements,
        total,
        out=np.zeros_like(displacements),
        where=total > 1e-8,
    )


def morph_frame(
    source_image: np.ndarray,
    destination_image: np.ndarray,
    source_points: np.ndarray,
    destination_points: np.ndarray,
    intermediate_points: np.ndarray,
    alpha: float,
    falloff: float = 2.0,
) -> np.ndarray:
    height, width = source_image.shape[:2]

    ys, xs = np.mgrid[0:height, 0:width]
    grid = np.stack((xs.ravel(), ys.ravel()), axis=-1).astype(np.float64)

    to_source = grid + displacement_field(
        grid, intermediate_points, source_points, falloff
    )
    to_destination = grid + displacement_field(
        grid, intermediate_points, destination_points, falloff
    )

    np.clip(to_source[:, 0], 0, width - 1, out=to_source[:, 0])
    np.clip(to_source[:, 1], 0, height - 1, out=to_source[:, 1])
    np.clip(to_destination[:, 0], 0, width - 1, out=to_destination[:, 0])
    np.clip(to_destination[:, 1], 0, height - 1, out=to_destination[:, 1])

    source_color = bilinear_sample(source_image, to_source[:, 0], to_source[:, 1])
    dest_color = bilinear_sample(
        destination_image, to_destination[:, 0], to_destination[:, 1]
    )

    blended = np.round((1 - alpha) * source_color + alpha * dest_color).astype(np.uint8)

    return blended.reshape(source_image.shape)


def morph_sequence(
    source_image: np.ndarray,
    destination_image: np.ndarray,
    source_points: np.ndarray,
    destination_points: np.ndarray,
    intermediate_points: np.ndarray,
    alphas: np.ndarray,
    falloff: float = 2.0,
) -> list[np.ndarray]:
    frames = [None] * len(alphas)
    frames[0] = source_image
    frames[-1] = destination_image

    for i in tqdm(range(1, len(alphas) - 1), desc="meshless"):
        frames[i] = morph_frame(
            source_image,
            destination_image,
            source_points,
            destination_points,
            intermediate_points[i],
            alphas[i],
            falloff,
        )

    return frames
