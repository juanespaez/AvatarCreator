"""OpenCV adapter for the ImageLoader port."""
from pathlib import Path

import cv2
import numpy as np

from avatar_creator.ports.image_loader import ImageLoader


class OpenCvImageLoader(ImageLoader):
    """Reads images with OpenCV and always hands back RGB (OpenCV natively uses BGR)."""

    def load_photo(self, path: Path) -> np.ndarray:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(path)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def load_piece(self, path: Path, out_size: int = 224) -> np.ndarray | None:
        raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if raw is None:
            return None
        if raw.ndim == 3 and raw.shape[2] == 4:
            # Flatten onto WHITE: left alone, transparent pixels read as black and would
            # dominate the embedding.
            alpha = raw[:, :, 3:4].astype(np.float32) / 255.0
            rgb = cv2.cvtColor(raw[:, :, :3], cv2.COLOR_BGR2RGB).astype(np.float32)
            raw = (rgb * alpha + 255.0 * (1 - alpha)).astype(np.uint8)
        else:
            raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)

        # Square-pad with white before resizing so the part's aspect ratio is preserved.
        height, width = raw.shape[:2]
        side = max(height, width)
        canvas = np.full((side, side, 3), 255, np.uint8)
        top, left = (side - height) // 2, (side - width) // 2
        canvas[top : top + height, left : left + width] = raw
        return cv2.resize(canvas, (out_size, out_size), interpolation=cv2.INTER_CUBIC)

    def load_rgba(self, path: Path, size: tuple[int, int] | None = None) -> np.ndarray | None:
        raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if raw is None:
            return None
        # Normalise every input (gray / BGR / BGRA) to 4 channels so layers stack uniformly.
        if raw.ndim == 2:
            raw = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGRA)
        if raw.shape[2] == 3:
            raw = cv2.cvtColor(raw, cv2.COLOR_BGR2BGRA)
        rgba = np.dstack([cv2.cvtColor(raw[:, :, :3], cv2.COLOR_BGR2RGB), raw[:, :, 3]])
        if size:
            # INTER_AREA is the right filter for downscaling art without aliasing.
            rgba = cv2.resize(rgba, size, interpolation=cv2.INTER_AREA)
        return rgba
