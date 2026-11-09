"""Filesystem adapter for the AssetRepository port."""
from pathlib import Path

from avatar_creator.domain.models.asset import Asset
from avatar_creator.domain.models.face_region import FaceRegion
from avatar_creator.infrastructure.filesystem.project_paths import ProjectPaths
from avatar_creator.ports.asset_repository import AssetRepository


class FilesystemAssetRepository(AssetRepository):
    """Reads pieces, references and queries from the folder layout in ``ProjectPaths``.

    The folder name IS the label (``refs/<region>/<piece_id>/``): no CSV or annotation tool.
    Adding a piece = dropping a PNG in ``pieces/``; nothing is retrained, the index is rebuilt.
    """

    IMAGE_EXTENSIONS: frozenset[str] = frozenset({".png", ".jpg", ".jpeg", ".webp"})

    def __init__(self, paths: ProjectPaths | None = None) -> None:
        self._paths = paths or ProjectPaths()

    def _is_image(self, path: Path) -> bool:
        return path.suffix.lower() in self.IMAGE_EXTENSIONS

    def pieces(self, region: FaceRegion) -> list[Asset]:
        folder = self._paths.pieces_dir / region.value
        if not folder.exists():
            return []
        # piece_id is the file stem and is the join key with refs/ - keep them identical or
        # refs silently orphan. Sorted so index order (and therefore ties) is deterministic.
        return [
            Asset(stem, path)
            for stem, path in sorted((p.stem, p) for p in folder.glob("*") if self._is_image(p))
        ]

    def references(self, region: FaceRegion) -> list[Asset]:
        folder = self._paths.refs_dir / region.value
        if not folder.exists():
            return []
        assets: list[Asset] = []
        for piece_dir in sorted(folder.iterdir()):
            if piece_dir.is_dir():
                # The label comes from the PARENT folder, so many photos share one piece id.
                assets += [
                    Asset(piece_dir.name, photo)
                    for photo in sorted(piece_dir.glob("*"))
                    if self._is_image(photo)
                ]
        return assets

    def find_piece(self, region: FaceRegion, piece_id: str) -> Asset | None:
        return next((asset for asset in self.pieces(region) if asset.piece_id == piece_id), None)

    def list_queries(self) -> list[Path]:
        folder = self._paths.queries_dir
        if not folder.exists():
            return []
        return [p for p in sorted(folder.glob("*")) if self._is_image(p)]

    def resolve_query(self, name: str) -> Path:
        direct = Path(name)
        if direct.exists():
            return direct
        # Convenience: "curry" finds queries/curry.jpg (any known extension).
        candidates = [self._paths.queries_dir / name] + [
            self._paths.queries_dir / f"{name}{ext}" for ext in sorted(self.IMAGE_EXTENSIONS)
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(f"No image {name!r} found (also looked in {self._paths.queries_dir})")
