import warnings

import numpy as np
from scipy.spatial import Delaunay
from tqdm import tqdm

from .interpolation import bilinear_sample

MIN_TRIANGLE_AREA = 1e-6

def affine_transform_matrix(source: np.ndarray, destination: np.ndarray) -> np.ndarray:
    A = np.zeros((6, 6))
    b = np.zeros((6, 1))

    for row, (x, y) in enumerate(source[:3]):
        A[row, 0:3] = [x, y, 1]
    A[3:, 3:] = A[:3, :3]

    b[0:3, 0] = destination[:3, 0]
    b[3:6, 0] = destination[:3, 1]

    return np.linalg.solve(A, b).reshape(2, 3)


def build_simplices(
    source_points: np.ndarray, destination_points: np.ndarray
) -> np.ndarray:
    return Delaunay((source_points + destination_points) / 2.0).simplices


def triangle_area(triangle: np.ndarray) -> float:
    (ax, ay), (bx, by), (cx, cy) = triangle[:3]
    return abs((bx - ax) * (cy - ay) - (cx - ax) * (by - ay)) / 2.0


def _apply(
    transform: np.ndarray, x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    return (
        transform[0, 0] * x + transform[0, 1] * y + transform[0, 2],
        transform[1, 0] * x + transform[1, 1] * y + transform[1, 2],
    )


def _barycentric_mask(
    triangle: np.ndarray, x: np.ndarray, y: np.ndarray
) -> np.ndarray:
    (ax, ay), (bx, by), (cx, cy) = triangle[:3]

    denominator = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
    first = ((by - cy) * (x - cx) + (cx - bx) * (y - cy)) / denominator
    second = ((cy - ay) * (x - cx) + (ax - cx) * (y - cy)) / denominator

    return (first >= 0) & (second >= 0) & (first + second <= 1)


def morph_frame(
    source_image: np.ndarray,
    destination_image: np.ndarray,
    source_points: np.ndarray,
    destination_points: np.ndarray,
    intermediate_points: np.ndarray,
    simplices: np.ndarray,
    alpha: float,
) -> np.ndarray:
    output = np.zeros_like(source_image)
    source_height, source_width = source_image.shape[:2]
    dest_height, dest_width = destination_image.shape[:2]

    for triangle in simplices:
        current_tri = intermediate_points[triangle]
        source_tri = source_points[triangle]
        dest_tri = destination_points[triangle]

        min_x, min_y = np.floor(current_tri.min(axis=0)).astype(int)
        max_x, max_y = np.ceil(current_tri.max(axis=0)).astype(int)

        min_x, min_y = max(min_x, 0), max(min_y, 0)
        max_x = min(max_x, source_width - 1)
        max_y = min(max_y, source_height - 1)

        if (
            min(
                triangle_area(current_tri),
                triangle_area(source_tri),
                triangle_area(dest_tri),
            )
            < MIN_TRIANGLE_AREA
        ):
            warnings.warn(
                f"skipping degenerate triangle {tuple(int(i) for i in triangle)}: "
                f"area {triangle_area(current_tri):.3g} px^2",
                RuntimeWarning,
                stacklevel=2,
            )
            continue

        to_source = affine_transform_matrix(current_tri, source_tri)
        to_destination = affine_transform_matrix(current_tri, dest_tri)

        # The whole bounding box is tested and warped at once: the per-pixel
        # Python loop this replaces cost about 200x more per frame.
        grid_y, grid_x = np.mgrid[min_y : max_y + 1, min_x : max_x + 1]
        grid_x = grid_x.ravel().astype(np.float64)
        grid_y = grid_y.ravel().astype(np.float64)

        inside = _barycentric_mask(current_tri, grid_x, grid_y)
        if not inside.any():
            continue

        px, py = grid_x[inside], grid_y[inside]

        sx, sy = _apply(to_source, px, py)
        dx, dy = _apply(to_destination, px, py)

        np.clip(sx, 0.0, source_width - 1, out=sx)
        np.clip(sy, 0.0, source_height - 1, out=sy)
        np.clip(dx, 0.0, dest_width - 1, out=dx)
        np.clip(dy, 0.0, dest_height - 1, out=dy)

        source_color = bilinear_sample(source_image, sx, sy)
        dest_color = bilinear_sample(destination_image, dx, dy)

        output[py.astype(np.intp), px.astype(np.intp)] = np.round(
            (1 - alpha) * source_color + alpha * dest_color
        ).astype(np.uint8)

    return output


def morph_sequence(
    source_image: np.ndarray,
    destination_image: np.ndarray,
    source_points: np.ndarray,
    destination_points: np.ndarray,
    intermediate_points: np.ndarray,
    alphas: np.ndarray,
) -> list[np.ndarray]:
    simplices = build_simplices(source_points, destination_points)

    frames = [None] * len(alphas)
    frames[0] = source_image
    frames[-1] = destination_image

    for i in tqdm(range(1, len(alphas) - 1), desc="triangulation"):
        frames[i] = morph_frame(
            source_image,
            destination_image,
            source_points,
            destination_points,
            intermediate_points[i],
            simplices,
            alphas[i],
        )

    return frames


def plot_triangulation(
    image: np.ndarray,
    points: np.ndarray,
    output_path: str,
    simplices: np.ndarray | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if simplices is None:
        simplices = Delaunay(points).simplices

    plt.figure(figsize=(4, 4))
    plt.imshow(image)
    plt.triplot(points[:, 0], points[:, 1], simplices, color="white", linewidth=0.7)
    plt.plot(points[:, 0], points[:, 1], "o", color="red", markersize=2.5)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(output_path, bbox_inches="tight", dpi=90)
    plt.close()
