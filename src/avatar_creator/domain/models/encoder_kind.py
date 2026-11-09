"""Which kind of vector a region is turned into."""
from enum import Enum


class EncoderKind(Enum):
    """The three ways a crop can be described numerically.

    * ``ARCFACE`` - 512-d face-geometry embedding (eyes, eyebrows, nose, mouth, face shape).
    * ``CLIP``    - 512-d appearance/texture embedding (hair, beard).
    * ``SKIN``    - 3-number median CIELAB colour, compared with deltaE2000 (skin tone).
                    It is a metric space, not a cosine space, so it is searched differently.
    """

    ARCFACE = "arcface"
    CLIP = "clip"
    SKIN = "skin"
