"""The outcome of recognizing one photo."""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from avatar_creator.domain.models.face_region import FaceRegion
from avatar_creator.domain.models.match import Match


@dataclass
class AvatarSelection:
    """Per-region top-k matches for one photo, plus the crops that produced them.

    Regions are chosen by independent argmax (each region ignores the others). Joint
    selection - e.g. hair colour constraining eyebrow colour - is a known open question.

    Attributes:
        photo_path: The query photo.
        crops: Every region crop that could be extracted (kept for previews/debugging).
        matches: Ranked matches per region; regions without an index are absent.
    """

    photo_path: Path
    crops: dict[FaceRegion, np.ndarray] = field(default_factory=dict)
    matches: dict[FaceRegion, list[Match]] = field(default_factory=dict)

    def top1(self, region: FaceRegion) -> Match | None:
        """Best match for ``region``, or None if the region has no hits."""
        hits = self.matches.get(region)
        return hits[0] if hits else None

    def regions_in_z_order(self) -> list[FaceRegion]:
        """Regions that have at least one hit, bottom layer first."""
        return [region for region in FaceRegion if self.matches.get(region)]
