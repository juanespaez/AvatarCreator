"""One line of the routing evaluation table."""
from dataclasses import dataclass

from avatar_creator.domain.models.encoder_kind import EncoderKind
from avatar_creator.domain.models.evaluation_result import EvaluationResult
from avatar_creator.domain.models.face_region import FaceRegion


@dataclass(frozen=True)
class RoutingRow:
    """How well ``encoder`` retrieves the right piece in ``region`` (one table row)."""

    region: FaceRegion
    encoder: EncoderKind
    result: EvaluationResult
