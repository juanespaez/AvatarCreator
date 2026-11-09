"""Everything that defines a region, in one place."""
from dataclasses import dataclass

from avatar_creator.domain.models.crop_config import CropConfig
from avatar_creator.domain.models.encoder_kind import EncoderKind
from avatar_creator.domain.models.face_region import FaceRegion


@dataclass(frozen=True)
class RegionSpec:
    """The full definition of a region: where it is, how to frame it, how to describe it.

    The notebook scattered these three facts across three dictionaries (``REGION_IDX``,
    ``CROP_CFG``, ``ENCODER``). Keeping them in one value object means adding or tuning a
    region touches exactly one place.

    Attributes:
        region: Which region this spec describes.
        landmark_indices: MediaPipe Face Mesh canonical point indices whose bounding box
            anchors the crop. They are stable across MediaPipe versions.
        crop: Framing of the square crop.
        encoder: Which encoder turns the crop into a vector.
    """

    region: FaceRegion
    landmark_indices: tuple[int, ...]
    crop: CropConfig
    encoder: EncoderKind
