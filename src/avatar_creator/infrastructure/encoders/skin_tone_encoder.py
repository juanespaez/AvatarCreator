"""Median-CIELAB colour adapter for the RegionEncoder port."""
from typing import Sequence

import numpy as np

from avatar_creator.domain.models.encoder_kind import EncoderKind
from avatar_creator.ports.region_encoder import RegionEncoder


class SkinToneEncoder(RegionEncoder):
    """Describes a cheek patch by 3 numbers: its median CIELAB colour.

    No neural network on purpose: skin tone is a 3-number problem. A perceptually uniform
    colour space is more accurate, ~100x cheaper and fully inspectable, unlike a 512-d
    embedding that cannot be debugged when it goes wrong.
    """

    @property
    def kind(self) -> EncoderKind:
        return EncoderKind.SKIN

    def _embed(self, crops_rgb: Sequence[np.ndarray]) -> np.ndarray:
        from skimage import color

        out = []
        for crop in crops_rgb:
            lab = color.rgb2lab(crop.astype(np.float32) / 255.0)
            # Median, not mean: robust to the stray shadow, mole or specular highlight that a
            # mean would smear across the whole estimate.
            out.append(np.median(lab.reshape(-1, 3), axis=0))
        # NOT L2-normalised: this is a metric (colour) space, not a cosine space.
        return np.asarray(out, dtype=np.float32)
