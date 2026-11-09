"""Roll-alignment of a face photo and its landmarks."""
import cv2
import numpy as np


class FaceAligner:
    """Rotates a face so the eye line is horizontal, and measures the face's scale.

    Pure image math (no I/O, no models), so it lives in the domain.

    WHY align: a tilted head would otherwise produce a tilted eye crop that the encoder reads
    as a different shape. Crops are taken from the aligned image so tilt does not matter.
    """

    #: MediaPipe indices of the outer eye corners. They drive both roll alignment and the
    #: inter-ocular distance (IOD), which is the scale unit for every crop padding.
    LEFT_EYE_OUTER: int = 263
    RIGHT_EYE_OUTER: int = 33

    def align(self, rgb: np.ndarray, landmarks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Rotate ``rgb`` about the eye-line midpoint so the outer eye corners are level.

        Args:
            rgb: ``(H, W, 3)`` uint8 RGB photo.
            landmarks: ``(N, 2)`` pixel coordinates in the original photo.

        Returns:
            The aligned image and the landmarks transformed into the aligned image's frame.
        """
        left, right = landmarks[self.LEFT_EYE_OUTER], landmarks[self.RIGHT_EYE_OUTER]
        dx, dy = (left - right)[0], (left - right)[1]
        angle = np.degrees(np.arctan2(dy, dx))
        center = tuple(((left + right) / 2).tolist())
        rotation = cv2.getRotationMatrix2D(center, angle, 1.0)

        height, width = rgb.shape[:2]
        # BORDER_REPLICATE (not black) so the corners opened by the rotation do not inject a
        # hard edge that would leak into hair/face-shape crops.
        aligned = cv2.warpAffine(
            rgb, rotation, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
        )
        homogeneous = np.hstack([landmarks, np.ones((len(landmarks), 1), np.float32)])
        return aligned, (homogeneous @ rotation.T).astype(np.float32)

    @classmethod
    def inter_ocular_distance(cls, landmarks: np.ndarray) -> float:
        """Distance between the outer eye corners, in pixels: the scale unit of all crops."""
        return float(np.linalg.norm(landmarks[cls.LEFT_EYE_OUTER] - landmarks[cls.RIGHT_EYE_OUTER]))
