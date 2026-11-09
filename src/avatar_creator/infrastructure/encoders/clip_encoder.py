"""CLIP (open_clip ViT-B/32) adapter for the RegionEncoder port."""
import logging
from typing import Sequence

import numpy as np

from avatar_creator.domain.models.encoder_kind import EncoderKind
from avatar_creator.ports.region_encoder import RegionEncoder

logger = logging.getLogger(__name__)


class ClipEncoder(RegionEncoder):
    """512-d CLIP image-tower embeddings for appearance regions (hair, beard).

    Used where the discriminating signal is texture and colour rather than geometry.
    open_clip is used instead of the HF wrapper because it exposes the raw image tower and
    lets the checkpoint be swapped (e.g. ViT-L/14) by changing one string.
    """

    #: Recorded in the exported manifest so vectors can be traced to their model.
    MODEL_LABEL: str = "ViT-B-32/laion2b_s34b_b79k"

    def __init__(
        self,
        device: str = "cpu",
        batch_size: int = 64,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
    ) -> None:
        self._device = device
        self._batch_size = batch_size
        self._model_name = model_name
        self._pretrained = pretrained
        self._model = None  # loaded lazily
        self._preprocess = None

    @property
    def kind(self) -> EncoderKind:
        return EncoderKind.CLIP

    def _load(self) -> None:
        import open_clip

        logger.info("Downloading/loading CLIP (%s, %s) ...", self._model_name, self._pretrained)
        model, _, preprocess = open_clip.create_model_and_transforms(
            self._model_name, pretrained=self._pretrained
        )
        self._model = model.to(self._device).eval()  # eval(): no dropout/batchnorm updates
        self._preprocess = preprocess
        logger.info("CLIP ready.")

    def _embed(self, crops_rgb: Sequence[np.ndarray]) -> np.ndarray:
        import torch
        from PIL import Image

        if self._model is None:
            self._load()
        out = []
        with torch.no_grad():
            # Batched so a big bank does not need all crops as tensors in memory at once.
            for start in range(0, len(crops_rgb), self._batch_size):
                chunk = crops_rgb[start : start + self._batch_size]
                tensor = torch.stack([self._preprocess(Image.fromarray(c)) for c in chunk]).to(self._device)
                features = self._model.encode_image(tensor).float()
                # L2-normalise so cosine similarity == inner product.
                out.append((features / features.norm(dim=-1, keepdim=True)).cpu().numpy())
        return np.concatenate(out).astype(np.float32)
