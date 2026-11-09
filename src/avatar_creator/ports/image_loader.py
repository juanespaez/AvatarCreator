"""Contract: read images from disk into arrays."""
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class ImageLoader(ABC):
    """Reads the three kinds of image the pipeline needs. All outputs are RGB (not BGR)."""

    @abstractmethod
    def load_photo(self, path: Path) -> np.ndarray:
        """Load a photo as ``(H, W, 3)`` uint8 RGB. Raises FileNotFoundError if unreadable."""

    @abstractmethod
    def load_piece(self, path: Path, out_size: int = 224) -> np.ndarray | None:
        """Load a piece artwork as an ``(out_size, out_size, 3)`` RGB image, or None if unreadable.

        Pieces are isolated parts on transparency, so implementations flatten them onto white
        and square-pad them before resizing (see the encoders' expectations).
        """

    @abstractmethod
    def load_rgba(self, path: Path, size: tuple[int, int] | None = None) -> np.ndarray | None:
        """Load an image as RGBA (keeping transparency) for compositing, or None if unreadable."""
