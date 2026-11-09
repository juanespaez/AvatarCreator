"""How a region's square crop is framed."""
from dataclasses import dataclass


@dataclass(frozen=True)
class CropConfig:
    """Framing of one region's square crop, expressed in inter-ocular-distance (IOD) units.

    Using IOD instead of pixels makes a crop mean the same thing on a selfie and on a 4K
    portrait, which is what makes query crops and reference crops comparable.

    Attributes:
        pad: Fraction of IOD added around the landmark bounding box.
        dy:  Vertical shift of the crop centre, in IOD units (negative = up). Hair and beard
             need it because their content is not where their landmarks are.
    """

    pad: float = 0.30
    dy: float = 0.0
