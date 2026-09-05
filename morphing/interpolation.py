import numpy as np


def bilinear_sample(image: np.ndarray, x, y) -> np.ndarray:
    height, width = image.shape[:2]

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    x1 = np.clip(np.floor(x), 0, width - 2).astype(np.intp)
    y1 = np.clip(np.floor(y), 0, height - 2).astype(np.intp)
    x2, y2 = x1 + 1, y1 + 1

    dx = (x - x1)[..., None]
    dy = (y - y1)[..., None]

    return (
        (1 - dx) * (1 - dy) * image[y1, x1]
        + dx * (1 - dy) * image[y1, x2]
        + (1 - dx) * dy * image[y2, x1]
        + dx * dy * image[y2, x2]
    )
