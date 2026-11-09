"""Text report of what data is available (notebook cell 2)."""
from avatar_creator.domain.models.face_region import FaceRegion
from avatar_creator.ports.asset_repository import AssetRepository


class DataReport:
    """Counts pieces and reference photos per region and flags orphaned references.

    An orphan is a reference labelled with a piece id that has no artwork in ``pieces/``. It is
    a silent accuracy killer: it can win a search and then fail to composite.
    """

    def __init__(self, assets: AssetRepository) -> None:
        self._assets = assets

    def render(self) -> str:
        lines = [f"{'category':<12}{'pieces':>8}{'refs':>8}{'refs/piece':>12}"]
        warnings: list[str] = []
        for region in FaceRegion:
            pieces = self._assets.pieces(region)
            refs = self._assets.references(region)
            labelled = {asset.piece_id for asset in refs}
            per_piece = len(refs) / len(labelled) if labelled else 0
            lines.append(f"{region.value:<12}{len(pieces):>8}{len(refs):>8}{per_piece:>12.1f}")

            orphans = labelled - {asset.piece_id for asset in pieces}
            if orphans:
                warnings.append(
                    f"WARNING {region.value}: refs reference unknown piece_ids -> {sorted(orphans)[:5]}"
                )
        return "\n".join(lines + warnings)
