"""Draws the region crops of one photo (notebook cell 5)."""
from pathlib import Path
from typing import Mapping

import numpy as np

from avatar_creator.domain.models.face_region import FaceRegion
from avatar_creator.infrastructure.rendering.figure_renderer import FigureRenderer


class CropPreviewRenderer(FigureRenderer):
    """One row of crops, one per region.

    This is the visual check to run BEFORE building any index: if the crops are wrong, every
    number downstream is meaningless. ``hair`` and ``beard`` offsets are the likeliest to need
    tuning for a given photo framing.
    """

    def render(self, photo_path: Path, crops: Mapping[FaceRegion, np.ndarray], out_dir: Path) -> Path:
        plt = self._pyplot()
        fig, axes = plt.subplots(1, len(crops), figsize=(2.0 * len(crops), 2.4))
        axes = np.atleast_1d(axes)  # subplots returns a bare Axes when there is only one column
        for ax, (region, image) in zip(axes, crops.items()):
            ax.imshow(image)
            ax.set_title(region.value, fontsize=9)
            ax.axis("off")
        fig.suptitle(photo_path.name, fontsize=10)
        plt.tight_layout()
        return self._save(fig, out_dir / f"{photo_path.stem}_crops.png")
