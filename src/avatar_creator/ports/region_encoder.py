"""Contract: turn region crops into vectors."""
from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np

from avatar_creator.domain.models.embedding_bank import EMBEDDING_DIM
from avatar_creator.domain.models.encoder_kind import EncoderKind


class RegionEncoder(ABC):
    """Describes crops numerically. One implementation per ``EncoderKind``.

    Implementations load their (heavy) model lazily on first use, so building the whole
    pipeline is cheap and a fully cached run never loads a model at all.
    """

    @property
    @abstractmethod
    def kind(self) -> EncoderKind:
        """Which kind of vector this encoder produces."""

    def embed(self, crops_rgb: Sequence[np.ndarray]) -> np.ndarray:
        """Encode crops into an ``(N, D)`` float32 matrix (template method).

        The empty-input guard lives here once instead of in every subclass (DRY).
        """
        if len(crops_rgb) == 0:
            return np.zeros((0, EMBEDDING_DIM), np.float32)
        return self._embed(crops_rgb)

    @abstractmethod
    def _embed(self, crops_rgb: Sequence[np.ndarray]) -> np.ndarray:
        """Encode a NON-EMPTY list of ``(H, W, 3)`` uint8 RGB crops."""
