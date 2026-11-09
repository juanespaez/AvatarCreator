"""Mouth recognizer."""
from avatar_creator.application.recognizers.region_recognizer import RegionRecognizer
from avatar_creator.domain.models.crop_config import CropConfig
from avatar_creator.domain.models.encoder_kind import EncoderKind
from avatar_creator.domain.models.face_region import FaceRegion
from avatar_creator.domain.models.region_spec import RegionSpec


class MouthRecognizer(RegionRecognizer):
    """Matches the lips (outer and inner contour) against the mouth pieces, using ArcFace."""

    spec = RegionSpec(
        region=FaceRegion.MOUTH,
        landmark_indices=(61, 291, 0, 17, 13, 14, 78, 308, 37, 267, 84, 314, 181, 405, 146, 375),
        crop=CropConfig(pad=0.30, dy=0.00),
        encoder=EncoderKind.ARCFACE,
    )
