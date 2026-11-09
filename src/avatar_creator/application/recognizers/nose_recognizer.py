"""Nose recognizer."""
from avatar_creator.application.recognizers.region_recognizer import RegionRecognizer
from avatar_creator.domain.models.crop_config import CropConfig
from avatar_creator.domain.models.encoder_kind import EncoderKind
from avatar_creator.domain.models.face_region import FaceRegion
from avatar_creator.domain.models.region_spec import RegionSpec


class NoseRecognizer(RegionRecognizer):
    """Matches the nose (bridge to nostrils) against the nose pieces, using ArcFace."""

    spec = RegionSpec(
        region=FaceRegion.NOSE,
        landmark_indices=(168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 98, 97, 326, 327, 115, 344, 220, 440),
        crop=CropConfig(pad=0.25, dy=0.00),
        encoder=EncoderKind.ARCFACE,
    )
