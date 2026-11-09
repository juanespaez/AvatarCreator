"""Contract: discover the piece bank, reference photos and query photos."""
from abc import ABC, abstractmethod
from pathlib import Path

from avatar_creator.domain.models.asset import Asset
from avatar_creator.domain.models.face_region import FaceRegion


class AssetRepository(ABC):
    """Where the data lives (data contract: ``pieces/``, ``refs/``, ``queries/``)."""

    @abstractmethod
    def pieces(self, region: FaceRegion) -> list[Asset]:
        """The Lego pieces for ``region``; the id is the file stem. Empty if none."""

    @abstractmethod
    def references(self, region: FaceRegion) -> list[Asset]:
        """Real photos labelled with a piece id for ``region``; the id is the parent folder."""

    @abstractmethod
    def find_piece(self, region: FaceRegion, piece_id: str) -> Asset | None:
        """The piece called ``piece_id`` in ``region``, or None."""

    @abstractmethod
    def list_queries(self) -> list[Path]:
        """Every query photo available, sorted."""

    @abstractmethod
    def resolve_query(self, name: str) -> Path:
        """Turn a path or a bare name (extension optional) into a real query photo path.

        Raises FileNotFoundError if nothing matches.
        """
