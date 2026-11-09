"""Raised when a photo contains no detectable face."""
from pathlib import Path


class NoFaceDetectedError(RuntimeError):
    """No face landmarks could be found in an image.

    The pipeline fails loudly here instead of indexing or matching garbage; callers decide
    whether to skip the photo (CLI) or abort (service).
    """

    def __init__(self, photo_path: Path | str = "") -> None:
        super().__init__(f"no face detected in {photo_path}")
        self.photo_path: Path | None = Path(photo_path) if photo_path else None
