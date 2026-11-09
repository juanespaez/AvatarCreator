"""Skin-tone recognizer."""
from avatar_creator.application.recognizers.region_recognizer import RegionRecognizer
from avatar_creator.domain.models.crop_config import CropConfig
from avatar_creator.domain.models.encoder_kind import EncoderKind
from avatar_creator.domain.models.face_region import FaceRegion
from avatar_creator.domain.models.region_spec import RegionSpec


class SkinToneRecognizer(RegionRecognizer):
    """Matches the cheek colour against skin-tone swatches by deltaE2000 (no neural network).

    Nothing here is special-cased: the SKIN encoder and the colour-distance searcher injected
    by the composition root make the shared ``recognize`` flow work in colour space.
    """

    spec = RegionSpec(
        region=FaceRegion.SKIN_TONE,
        # Cheek interior: avoids lips, brows and eyes.
        landmark_indices=(50, 280, 205, 425, 101, 330, 116, 345),
        crop=CropConfig(pad=0.15, dy=0.00),
        encoder=EncoderKind.SKIN,
    )
