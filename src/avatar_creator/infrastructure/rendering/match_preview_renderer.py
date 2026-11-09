"""Draws query crops above their top-1 matched pieces."""
from pathlib import Path

import numpy as np

from avatar_creator.domain.models.avatar_selection import AvatarSelection
from avatar_creator.infrastructure.rendering.figure_renderer import FigureRenderer
from avatar_creator.ports.asset_repository import AssetRepository
from avatar_creator.ports.image_loader import ImageLoader


class MatchPreviewRenderer(FigureRenderer):
    """Top row: the query crop per region. Bottom row: the top-1 matched piece.

    Lets a match be eyeballed for plausibility instead of trusted on cosine score alone.
    """

    def __init__(self, image_loader: ImageLoader, assets: AssetRepository) -> None:
        self._image_loader = image_loader
        self._assets = assets

    def render(self, selection: AvatarSelection, out_dir: Path) -> Path | None:
        regions = selection.regions_in_z_order()
        if not regions:
            return None

        plt = self._pyplot()
        fig, axes = plt.subplots(2, len(regions), figsize=(2.2 * len(regions), 5.0))
        if len(regions) == 1:
            axes = axes.reshape(2, 1)  # keep 2-D indexing even with a single column

        for column, region in enumerate(regions):
            axes[0, column].imshow(selection.crops[region])
            axes[0, column].set_title(region.value, fontsize=9)
            axes[0, column].axis("off")

            best = selection.matches[region][0]
            asset = self._assets.find_piece(region, best.piece_id)
            piece_image = self._image_loader.load_piece(asset.path, 224) if asset else None
            if piece_image is not None:  # a piece id with no artwork on disk is shown blank
                axes[1, column].imshow(piece_image)
            axes[1, column].set_title(f"{best.piece_id}\n{best.score:.2f}", fontsize=8)
            axes[1, column].axis("off")

        axes[0, 0].set_ylabel("query", fontsize=9)
        axes[1, 0].set_ylabel("top-1 match", fontsize=9)
        fig.suptitle(selection.photo_path.name, fontsize=11)
        plt.tight_layout()
        return self._save(fig, out_dir / f"{selection.photo_path.stem}_match.png")
