"""Enum-keyed controller that builds the recognizer for each region."""
from typing import Callable, Mapping

from avatar_creator.application.recognizers.beard_recognizer import BeardRecognizer
from avatar_creator.application.recognizers.eye_recognizer import EyeRecognizer
from avatar_creator.application.recognizers.eyebrows_recognizer import EyebrowsRecognizer
from avatar_creator.application.recognizers.face_shape_recognizer import FaceShapeRecognizer
from avatar_creator.application.recognizers.hair_recognizer import HairRecognizer
from avatar_creator.application.recognizers.mouth_recognizer import MouthRecognizer
from avatar_creator.application.recognizers.nose_recognizer import NoseRecognizer
from avatar_creator.application.recognizers.region_recognizer import RegionRecognizer
from avatar_creator.application.recognizers.skin_tone_recognizer import SkinToneRecognizer
from avatar_creator.domain.models.encoder_kind import EncoderKind
from avatar_creator.domain.models.face_region import FaceRegion
from avatar_creator.ports.region_encoder import RegionEncoder
from avatar_creator.ports.similarity_searcher import SimilaritySearcher


class RecognizerRegistry:
    """Maps each ``FaceRegion`` to its recognizer class and builds them wired with their parts.

    This is the single place that knows which recognizer serves which region, so the
    ``AvatarRecognizer`` facade never branches on region. To support a new region, add the
    enum member, one recognizer subclass, and one line to ``RECOGNIZER_TYPES``.
    """

    RECOGNIZER_TYPES: dict[FaceRegion, type[RegionRecognizer]] = {
        FaceRegion.FACE_SHAPE: FaceShapeRecognizer,
        FaceRegion.SKIN_TONE: SkinToneRecognizer,
        FaceRegion.EYES: EyeRecognizer,
        FaceRegion.EYEBROWS: EyebrowsRecognizer,
        FaceRegion.NOSE: NoseRecognizer,
        FaceRegion.MOUTH: MouthRecognizer,
        FaceRegion.BEARD: BeardRecognizer,
        FaceRegion.HAIR: HairRecognizer,
    }

    def __init__(
        self,
        encoders: Mapping[EncoderKind, RegionEncoder],
        searcher_factory: Callable[[EncoderKind], SimilaritySearcher],
    ) -> None:
        """
        Args:
            encoders: One shared encoder per kind (models are heavy; share them).
            searcher_factory: Builds a NEW searcher for a kind - searchers are stateful, so
                each recognizer needs its own.
        """
        self._encoders = encoders
        self._searcher_factory = searcher_factory

    def create(self, region: FaceRegion) -> RegionRecognizer:
        """Build the recognizer for one region."""
        recognizer_type = self.RECOGNIZER_TYPES[region]
        kind = recognizer_type.spec.encoder
        return recognizer_type(self._encoders[kind], self._searcher_factory(kind))

    def create_all(self) -> dict[FaceRegion, RegionRecognizer]:
        """Build every recognizer, in z-order (the order of the ``FaceRegion`` enum)."""
        return {region: self.create(region) for region in FaceRegion}
