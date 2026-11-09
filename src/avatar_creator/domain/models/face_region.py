"""The facial regions the avatar is assembled from."""
from enum import Enum


class FaceRegion(Enum):
    """Every region of the face that is matched against its own bank of Lego pieces.

    The enum value is the folder slug used under ``pieces/``, ``refs/`` and ``parked/``,
    so ``FaceRegion("eyes")`` maps a folder name straight to a region (and back via ``.value``).

    Definition order is meaningful: it is the z-order used when compositing the avatar
    (earlier = drawn first / underneath, later = drawn on top), exactly like the notebook's
    ``CATEGORIES`` list. Iterating ``FaceRegion`` therefore yields regions bottom-to-top.

    To add a new region (e.g. ears): add a member here, add one ``RegionRecognizer`` subclass
    and register it in ``RecognizerRegistry`` - nothing else changes.
    """

    FACE_SHAPE = "face_shape"
    SKIN_TONE = "skin_tone"
    EYES = "eyes"
    EYEBROWS = "eyebrows"
    NOSE = "nose"
    MOUTH = "mouth"
    BEARD = "beard"
    HAIR = "hair"
