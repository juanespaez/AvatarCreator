"""MediaPipe Face Mesh adapter for the LandmarkDetector port."""
import numpy as np

from avatar_creator.ports.landmark_detector import LandmarkDetector


class MediaPipeLandmarkDetector(LandmarkDetector):
    """Finds 478 face landmarks with MediaPipe Face Mesh.

    WHY Face Mesh instead of a 68-point model: 68-point models give ~6 points per eye and none
    for the hairline or cheek interior. Dense contours can be cropped from directly, and
    MediaPipe is real-time on CPU with no separate model download.
    """

    def __init__(self, min_detection_confidence: float = 0.5) -> None:
        self._min_detection_confidence = min_detection_confidence
        # Built on first use: importing/initialising MediaPipe is slow and a run served
        # entirely from the embedding cache may still need it only for the query photo.
        self._face_mesh = None

    def _mesh(self):
        if self._face_mesh is None:
            import mediapipe as mp

            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,  # every input is an independent photo, no tracking
                max_num_faces=1,  # multi-face is out of scope; ambiguous input should fail loudly
                refine_landmarks=True,  # +iris points (468 -> 478), cheap and gives eye scale
                min_detection_confidence=self._min_detection_confidence,
            )
        return self._face_mesh

    def detect(self, rgb: np.ndarray) -> np.ndarray | None:
        result = self._mesh().process(rgb)
        if not result.multi_face_landmarks:
            return None
        height, width = rgb.shape[:2]
        # MediaPipe returns coordinates normalised to [0, 1]; crops need pixels.
        landmarks = result.multi_face_landmarks[0].landmark
        return np.array([[p.x * width, p.y * height] for p in landmarks], dtype=np.float32)
