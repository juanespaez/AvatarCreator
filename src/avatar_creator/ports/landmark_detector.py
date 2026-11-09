"""Contract: find facial landmarks in a photo."""
from abc import ABC, abstractmethod

import numpy as np


class LandmarkDetector(ABC):
    """Locates face landmarks so regions can be cropped.

    A port (not a concrete class) so the application never imports MediaPipe, and a fake
    detector can stand in for it in tests.
    """

    @abstractmethod
    def detect(self, rgb: np.ndarray) -> np.ndarray | None:
        """Return ``(N, 2)`` float32 pixel coordinates of the landmarks, or None if no face.

        Only ONE face is ever returned: ambiguous multi-face input is out of scope.
        """
