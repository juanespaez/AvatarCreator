"""Draws input photo next to the assembled avatar (notebook cell 11)."""
from pathlib import Path

import numpy as np

from avatar_creator.infrastructure.rendering.figure_renderer import FigureRenderer


class AvatarPreviewRenderer(FigureRenderer):
    """Side-by-side "input" vs "assembled" image."""

    def render(self, photo_path: Path, photo_rgb: np.ndarray, avatar_rgba: np.ndarray, out_dir: Path) -> Path:
        plt = self._pyplot()
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(photo_rgb)
        axes[0].set_title("input")
        axes[0].axis("off")
        axes[1].imshow(avatar_rgba)
        axes[1].set_title("assembled")
        axes[1].axis("off")
        plt.tight_layout()
        return self._save(fig, out_dir / f"{photo_path.stem}_avatar.png")
