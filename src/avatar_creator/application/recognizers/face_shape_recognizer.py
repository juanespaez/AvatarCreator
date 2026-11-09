"""Face-shape recognizer."""
from avatar_creator.application.recognizers.region_recognizer import RegionRecognizer
from avatar_creator.domain.models.crop_config import CropConfig
from avatar_creator.domain.models.encoder_kind import EncoderKind
from avatar_creator.domain.models.face_region import FaceRegion
from avatar_creator.domain.models.region_spec import RegionSpec


class FaceShapeRecognizer(RegionRecognizer):
    """Matches the whole face outline (jaw, cheeks, forehead) against the face-shape pieces."""

    spec = RegionSpec(
        region=FaceRegion.FACE_SHAPE,
        # Jawline points 0-16 plus the temples, chin and forehead anchors.
        landmark_indices=tuple(range(0, 17)) + (234, 454, 10, 152, 127, 356, 132, 361, 172, 397),
        crop=CropConfig(pad=0.10, dy=0.00),  # tight pad: the outline itself is the signal
        encoder=EncoderKind.ARCFACE,
    )
