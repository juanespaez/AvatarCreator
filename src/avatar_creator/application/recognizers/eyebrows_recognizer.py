"""Eyebrows recognizer."""
from avatar_creator.application.recognizers.region_recognizer import RegionRecognizer
from avatar_creator.domain.models.crop_config import CropConfig
from avatar_creator.domain.models.encoder_kind import EncoderKind
from avatar_creator.domain.models.face_region import FaceRegion
from avatar_creator.domain.models.region_spec import RegionSpec


class EyebrowsRecognizer(RegionRecognizer):
    """Matches both eyebrows against the eyebrow pieces, using ArcFace."""

    spec = RegionSpec(
        region=FaceRegion.EYEBROWS,
        landmark_indices=(
            70, 63, 105, 66, 107, 55, 65, 52, 53, 46,  # right eyebrow
            300, 293, 334, 296, 336, 285, 295, 282, 283, 276,  # left eyebrow
        ),
        crop=CropConfig(pad=0.35, dy=0.00),
        encoder=EncoderKind.ARCFACE,
    )
