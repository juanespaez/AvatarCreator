"""Contract: cache built embedding banks."""
from abc import ABC, abstractmethod

from avatar_creator.domain.models.embedding_bank import EmbeddingBank
from avatar_creator.domain.models.encoder_kind import EncoderKind
from avatar_creator.domain.models.face_region import FaceRegion
from avatar_creator.domain.models.match_mode import MatchMode


class EmbeddingBankStore(ABC):
    """Persists banks so re-running does not re-encode the whole piece bank.

    The cache key is (region, encoder, mode). It does NOT notice a changed crop config or
    CLIP checkpoint: anything that changes the query-side transform invalidates the bank-side
    vectors, so clear the cache after changing a ``RegionSpec``.
    """

    @abstractmethod
    def load(self, region: FaceRegion, encoder: EncoderKind, mode: MatchMode) -> EmbeddingBank | None:
        """Return the cached bank, or None when there is none."""

    @abstractmethod
    def save(self, region: FaceRegion, bank: EmbeddingBank) -> None:
        """Persist ``bank`` under its own (region, encoder, mode) key."""
