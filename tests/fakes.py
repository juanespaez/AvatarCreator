"""Tiny in-memory stand-ins for the ports, so tests need no models, no GPU and no real photos."""
from pathlib import Path

import numpy as np

from avatar_creator.domain.models.asset import Asset
from avatar_creator.domain.models.embedding_bank import EmbeddingBank
from avatar_creator.domain.models.encoder_kind import EncoderKind
from avatar_creator.domain.models.face_region import FaceRegion
from avatar_creator.domain.models.match_mode import MatchMode
from avatar_creator.ports.asset_repository import AssetRepository
from avatar_creator.ports.embedding_bank_store import EmbeddingBankStore
from avatar_creator.ports.image_loader import ImageLoader
from avatar_creator.ports.landmark_detector import LandmarkDetector
from avatar_creator.ports.region_encoder import RegionEncoder


def synthetic_landmarks(width: int = 400, height: int = 400) -> np.ndarray:
    """478 points scattered around the image centre, with level eyes 100px apart.

    Real positions do not matter for plumbing tests; what matters is that every MediaPipe index
    exists and the outer eye corners (33 and 263) are a sensible distance apart.
    """
    rng = np.random.default_rng(0)
    points = rng.uniform([width * 0.3, height * 0.3], [width * 0.7, height * 0.7], size=(478, 2))
    points[33] = [width / 2 - 50, height / 2]
    points[263] = [width / 2 + 50, height / 2]
    return points.astype(np.float32)


class FakeLandmarkDetector(LandmarkDetector):
    """Returns fixed landmarks, or None to simulate a photo with no face."""

    def __init__(self, landmarks: np.ndarray | None) -> None:
        self._landmarks = landmarks

    def detect(self, rgb: np.ndarray) -> np.ndarray | None:
        return self._landmarks


class FakeImageLoader(ImageLoader):
    """Serves a flat-colour photo/piece whose colour is derived from the file name."""

    def load_photo(self, path: Path) -> np.ndarray:
        return np.full((400, 400, 3), 128, np.uint8)

    def load_piece(self, path: Path, out_size: int = 224) -> np.ndarray | None:
        return np.full((out_size, out_size, 3), 200, np.uint8)

    def load_rgba(self, path: Path, size: tuple[int, int] | None = None) -> np.ndarray | None:
        side = size[0] if size else 8
        layer = np.zeros((side, side, 4), np.uint8)
        layer[..., :3] = 255 if "white" in path.name else 10
        layer[..., 3] = 255
        return layer


class FakeEncoder(RegionEncoder):
    """Maps a crop to a 4-d one-hot-ish vector from its mean brightness (deterministic)."""

    def __init__(self, kind: EncoderKind = EncoderKind.ARCFACE) -> None:
        self._kind = kind

    @property
    def kind(self) -> EncoderKind:
        return self._kind

    def _embed(self, crops_rgb) -> np.ndarray:
        out = []
        for crop in crops_rgb:
            vector = np.zeros(4, np.float32)
            vector[int(crop.mean()) % 4] = 1.0
            out.append(vector)
        return np.stack(out)


class InMemoryAssetRepository(AssetRepository):
    """Assets declared directly in the test."""

    def __init__(self, pieces=None, references=None) -> None:
        self._pieces: dict[FaceRegion, list[Asset]] = pieces or {}
        self._references: dict[FaceRegion, list[Asset]] = references or {}

    def pieces(self, region: FaceRegion) -> list[Asset]:
        return self._pieces.get(region, [])

    def references(self, region: FaceRegion) -> list[Asset]:
        return self._references.get(region, [])

    def find_piece(self, region: FaceRegion, piece_id: str) -> Asset | None:
        return next((a for a in self.pieces(region) if a.piece_id == piece_id), None)

    def list_queries(self) -> list[Path]:
        return []

    def resolve_query(self, name: str) -> Path:
        return Path(name)


class InMemoryBankStore(EmbeddingBankStore):
    """Dict-backed cache that counts how many times it was asked to save."""

    def __init__(self) -> None:
        self.banks: dict[tuple, EmbeddingBank] = {}
        self.saves = 0

    def load(self, region: FaceRegion, encoder: EncoderKind, mode: MatchMode) -> EmbeddingBank | None:
        return self.banks.get((region, encoder, mode))

    def save(self, region: FaceRegion, bank: EmbeddingBank) -> None:
        self.saves += 1
        self.banks[(region, bank.encoder, bank.mode)] = bank
