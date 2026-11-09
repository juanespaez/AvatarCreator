"""Base class of every per-region recognizer."""
from abc import ABC, abstractmethod

import numpy as np

from avatar_creator.domain.models.embedding_bank import EmbeddingBank
from avatar_creator.domain.models.face_region import FaceRegion
from avatar_creator.domain.models.match import Match
from avatar_creator.domain.models.region_spec import RegionSpec
from avatar_creator.ports.region_encoder import RegionEncoder
from avatar_creator.ports.similarity_searcher import SimilaritySearcher


class RegionRecognizer(ABC):
    """Recognizes ONE facial region: crop in, ranked Lego pieces out.

    Template-method design. Everything that is the same for every region (hold a bank,
    encode a crop, search) lives here once (DRY). The only thing a subclass supplies is its
    ``spec``: which landmarks anchor the region, how to frame the crop and which encoder
    describes it (SRP: a subclass is the single place that defines its region).

    The encoder is shared between recognizers (one ArcFace model serves five regions), while
    each recognizer owns its own searcher because a searcher holds one region's index.
    """

    def __init__(self, encoder: RegionEncoder, searcher: SimilaritySearcher) -> None:
        if encoder.kind is not self.spec.encoder:
            # Fail at wiring time: a CLIP vector searched in an ArcFace bank would silently
            # produce plausible-looking nonsense rather than an error.
            raise ValueError(
                f"{type(self).__name__} needs a {self.spec.encoder.value} encoder, got {encoder.kind.value}"
            )
        self._encoder = encoder
        self._searcher = searcher
        self._bank: EmbeddingBank | None = None

    @property
    @abstractmethod
    def spec(self) -> RegionSpec:
        """Definition of this recognizer's region (landmarks, crop framing, encoder)."""

    @property
    def region(self) -> FaceRegion:
        """The region this recognizer is responsible for."""
        return self.spec.region

    @property
    def bank(self) -> EmbeddingBank | None:
        """The bank currently loaded, or None before ``load_bank``."""
        return self._bank

    @property
    def is_ready(self) -> bool:
        """True once a bank is loaded and the recognizer can answer queries."""
        return self._bank is not None

    def load_bank(self, bank: EmbeddingBank) -> None:
        """Install ``bank`` and prepare the searcher (build the index) for it."""
        if bank.encoder is not self.spec.encoder:
            raise ValueError(
                f"{self.region.value} expects {self.spec.encoder.value} vectors, bank holds {bank.encoder.value}"
            )
        self._searcher.fit(bank)
        self._bank = bank

    def recognize(self, crop_rgb: np.ndarray, k: int = 3) -> list[Match]:
        """Return the ``k`` best pieces for one region crop, best first.

        Raises:
            RuntimeError: if no bank has been loaded yet.
        """
        if not self.is_ready:
            raise RuntimeError(f"{type(self).__name__} has no bank loaded; call load_bank() first")
        query = self._encoder.embed([crop_rgb])[0]
        return self._searcher.search(query, k)
