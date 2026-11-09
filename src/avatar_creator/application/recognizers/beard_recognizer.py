"""Beard recognizer."""
from avatar_creator.application.recognizers.region_recognizer import RegionRecognizer
from avatar_creator.domain.models.crop_config import CropConfig
from avatar_creator.domain.models.encoder_kind import EncoderKind
from avatar_creator.domain.models.face_region import FaceRegion
from avatar_creator.domain.models.region_spec import RegionSpec


class BeardRecognizer(RegionRecognizer):
    """Matches the lower face (lip line down to the chin) against the beard pieces, using CLIP.

    CLIP rather than ArcFace because a beard is discriminated by texture and colour, not
    geometry. The crop is shifted DOWN (``dy``) because a beard has no landmarks of its own:
    it sits below the lip line.
    """

    spec = RegionSpec(
        region=FaceRegion.BEARD,
        landmark_indices=(
            61, 291, 17, 18, 200, 199, 175, 152, 148, 176, 149, 150,
            136, 172, 58, 132, 377, 400, 378, 379, 365, 397, 288, 361,
        ),
        crop=CropConfig(pad=0.20, dy=0.10),
        encoder=EncoderKind.CLIP,
    )
