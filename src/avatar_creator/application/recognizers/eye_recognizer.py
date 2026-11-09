"""Eye recognizer."""
from avatar_creator.application.recognizers.region_recognizer import RegionRecognizer
from avatar_creator.domain.models.crop_config import CropConfig
from avatar_creator.domain.models.encoder_kind import EncoderKind
from avatar_creator.domain.models.face_region import FaceRegion
from avatar_creator.domain.models.region_spec import RegionSpec


class EyeRecognizer(RegionRecognizer):
    """Matches both eyes (one crop spanning them) against the eye pieces, using ArcFace."""

    spec = RegionSpec(
        region=FaceRegion.EYES,
        landmark_indices=(
            # right eye ring
            33, 133, 160, 159, 158, 157, 173, 155, 154, 153, 145, 144, 163, 7,
            # left eye ring
            362, 263, 387, 386, 385, 384, 398, 382, 381, 380, 374, 373, 390, 249,
        ),
        crop=CropConfig(pad=0.30, dy=0.00),
        encoder=EncoderKind.ARCFACE,
    )
