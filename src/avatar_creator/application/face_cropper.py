"""Photo -> per-region crops."""
from pathlib import Path
from typing import Sequence

import numpy as np

from avatar_creator.domain.models.face_region import FaceRegion
from avatar_creator.domain.models.region_spec import RegionSpec
from avatar_creator.domain.services.face_aligner import FaceAligner
from avatar_creator.domain.services.region_cropper import RegionCropper
from avatar_creator.ports.image_loader import ImageLoader
from avatar_creator.ports.landmark_detector import LandmarkDetector


class FaceCropper:
    """Runs the crop pipeline: load -> detect landmarks -> roll-align -> crop each region.

    Used for BOTH query photos and reference photos. Sharing one code path is the point: the
    query embedding and the reference embedding must come from an identical transform, or
    comparing them is not apples-to-apples.
    """

    def __init__(
        self,
        image_loader: ImageLoader,
        landmark_detector: LandmarkDetector,
        face_aligner: FaceAligner | None = None,
        region_cropper: RegionCropper | None = None,
        out_size: int = 224,
    ) -> None:
        self._image_loader = image_loader
        self._landmark_detector = landmark_detector
        self._face_aligner = face_aligner or FaceAligner()
        self._region_cropper = region_cropper or RegionCropper()
        self._out_size = out_size

    def crop_photo(self, path: Path, specs: Sequence[RegionSpec]) -> dict[FaceRegion, np.ndarray]:
        """Crop ``specs``' regions from the photo at ``path``. See ``crop_image``."""
        return self.crop_image(self._image_loader.load_photo(path), specs)

    def crop_image(self, rgb: np.ndarray, specs: Sequence[RegionSpec]) -> dict[FaceRegion, np.ndarray]:
        """Crop each region in ``specs`` from an RGB image.

        Returns ``{}`` when no face is found, so callers can count and report failures instead
        of silently indexing garbage. Regions whose box comes out empty are simply absent.
        """
        landmarks = self._landmark_detector.detect(rgb)
        if landmarks is None:
            return {}
        aligned_rgb, aligned_landmarks = self._face_aligner.align(rgb, landmarks)
        crops: dict[FaceRegion, np.ndarray] = {}
        for spec in specs:
            crop = self._region_cropper.crop(aligned_rgb, aligned_landmarks, spec, self._out_size)
            if crop is not None:
                crops[spec.region] = crop
        return crops
