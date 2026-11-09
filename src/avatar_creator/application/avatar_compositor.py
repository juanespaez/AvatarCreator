"""Selection -> assembled avatar image (notebook cell 11)."""
import numpy as np

from avatar_creator.domain.models.avatar_selection import AvatarSelection
from avatar_creator.domain.models.face_region import FaceRegion
from avatar_creator.ports.asset_repository import AssetRepository
from avatar_creator.ports.image_loader import ImageLoader


class AvatarCompositor:
    """Stacks the chosen pieces into one RGBA avatar, in ``FaceRegion`` z-order.

    Honest limitation: correct layering needs an *anchor convention* in the assets - each PNG
    pre-registered on a common full canvas at a common scale. Without that, no code can place
    an eye correctly, because "where does this eye go" is not recoverable from the pixels.
    This class therefore assumes pre-registered canvas-sized layers and does a plain alpha
    stack (the notebook's ``canvas`` layout). The notebook's ``anchored`` layout was only a
    centred placeholder and is intentionally not ported.
    """

    def __init__(self, image_loader: ImageLoader, assets: AssetRepository, canvas_size: int = 512) -> None:
        self._image_loader = image_loader
        self._assets = assets
        self._canvas_size = canvas_size

    def compose(self, selection: AvatarSelection) -> np.ndarray:
        """Return a ``(canvas, canvas, 4)`` uint8 RGBA avatar built from each region's top-1."""
        canvas = np.zeros((self._canvas_size, self._canvas_size, 4), np.uint8)
        for region in FaceRegion:  # enum order == z-order: later regions are drawn on top
            if region is FaceRegion.SKIN_TONE:
                continue  # skin tone is a colour swatch, not a layer with artwork to draw
            best = selection.top1(region)
            if best is None:
                continue
            asset = self._assets.find_piece(region, best.piece_id)
            if asset is None:
                continue  # id matched (e.g. via a reference photo) but no artwork exists for it
            layer = self._image_loader.load_rgba(asset.path, (self._canvas_size, self._canvas_size))
            if layer is not None:
                canvas = self._alpha_over(canvas, layer)
        return canvas

    @staticmethod
    def _alpha_over(destination: np.ndarray, source: np.ndarray) -> np.ndarray:
        """Draw ``source`` over ``destination`` using ``source``'s alpha (standard "over")."""
        alpha = source[:, :, 3:4].astype(np.float32) / 255.0
        destination[:, :, :3] = (source[:, :, :3] * alpha + destination[:, :, :3] * (1 - alpha)).astype(np.uint8)
        destination[:, :, 3:4] = np.maximum(destination[:, :, 3:4], source[:, :, 3:4])
        return destination
