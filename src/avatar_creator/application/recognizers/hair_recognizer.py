"""Hair recognizer."""
from avatar_creator.application.recognizers.region_recognizer import RegionRecognizer
from avatar_creator.domain.models.crop_config import CropConfig
from avatar_creator.domain.models.encoder_kind import EncoderKind
from avatar_creator.domain.models.face_region import FaceRegion
from avatar_creator.domain.models.region_spec import RegionSpec


class HairRecognizer(RegionRecognizer):
    """Matches the hair above the forehead against the hair pieces, using CLIP.

    The anchor is only the forehead/hairline landmark row; the large ``pad`` and the upward
    ``dy`` shift move the crop onto the hair itself, which has no landmarks.
    """

    spec = RegionSpec(
        region=FaceRegion.HAIR,
        landmark_indices=(10, 109, 338, 67, 297, 54, 284, 21, 251, 103, 332),
        crop=CropConfig(pad=0.90, dy=-0.75),
        encoder=EncoderKind.CLIP,
    )
