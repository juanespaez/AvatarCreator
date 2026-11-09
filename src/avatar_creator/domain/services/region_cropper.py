"""Cuts one region's square crop out of an aligned face."""
import cv2
import numpy as np

from avatar_creator.domain.models.region_spec import RegionSpec
from avatar_creator.domain.services.face_aligner import FaceAligner


class RegionCropper:
    """Turns a region spec + aligned landmarks into a fixed-size square RGB crop.

    Crops are square and padded relative to IOD (not image size), so a query crop and a
    reference crop produced from photos of any resolution are directly comparable.
    """

    def crop(
        self,
        rgb_aligned: np.ndarray,
        landmarks_aligned: np.ndarray,
        spec: RegionSpec,
        out_size: int = 224,
    ) -> np.ndarray | None:
        """Crop ``spec``'s region; returns None if the box ends up empty."""
        points = landmarks_aligned[list(spec.landmark_indices)]
        scale = FaceAligner.inter_ocular_distance(landmarks_aligned)

        x0, y0 = points.min(0)
        x1, y1 = points.max(0)
        # dy shifts the box in IOD units: hair sits ABOVE its anchor row, beard BELOW the lips.
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2 + spec.crop.dy * scale
        # Half-side of a SQUARE box -> no aspect distortion when resized to out_size.
        half = max(x1 - x0, y1 - y0) / 2 + spec.crop.pad * scale

        height, width = rgb_aligned.shape[:2]
        left, top = int(round(cx - half)), int(round(cy - half))
        right, bottom = int(round(cx + half)), int(round(cy + half))

        # Pad instead of clamping: clamping would change the crop's aspect and shift the region
        # off-centre - exactly the kind of silent inconsistency that poisons a retrieval index.
        pad_left, pad_top = max(0, -left), max(0, -top)
        pad_right, pad_bottom = max(0, right - width), max(0, bottom - height)
        if pad_left or pad_top or pad_right or pad_bottom:
            rgb_aligned = cv2.copyMakeBorder(
                rgb_aligned, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REPLICATE
            )
            # Padding shifted the image origin, so shift the box with it.
            left += pad_left
            right += pad_left
            top += pad_top
            bottom += pad_top

        patch = rgb_aligned[top:bottom, left:right]
        if patch.size == 0:
            return None
        return cv2.resize(patch, (out_size, out_size), interpolation=cv2.INTER_CUBIC)
