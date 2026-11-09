"""ArcFace (insightface buffalo_l) adapter for the RegionEncoder port."""
import logging
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from avatar_creator.domain.models.encoder_kind import EncoderKind
from avatar_creator.ports.region_encoder import RegionEncoder

logger = logging.getLogger(__name__)


class ArcFaceEncoder(RegionEncoder):
    """512-d ArcFace embeddings for geometric regions (eyes, eyebrows, nose, mouth, face shape).

    The recognition head is loaded directly, bypassing insightface's ``FaceAnalysis`` wrapper
    for inference: that wrapper re-detects a face inside the image, but our input is already a
    crop of ONE region and no detector will find a face in a picture of an eyebrow.

    Caveat worth remembering: ArcFace is trained on whole aligned faces; feeding it an isolated
    crop is off-distribution. Run the ``evaluate`` command to measure whether it holds up.
    """

    #: Recorded in the exported manifest so vectors can be traced to their model.
    MODEL_LABEL: str = "buffalo_l/w600k_r50"

    def __init__(self, device: str = "cpu") -> None:
        self._device = device
        self._model = None  # loaded lazily, see _load()

    @property
    def kind(self) -> EncoderKind:
        return EncoderKind.ARCFACE

    def _load(self):
        from insightface.app import FaceAnalysis
        from insightface.model_zoo import get_model

        logger.info("Downloading/loading ArcFace (buffalo_l) ...")
        # Side effect we want: FaceAnalysis downloads the buffalo_l pack into ~/.insightface.
        FaceAnalysis(name="buffalo_l")
        model_path = str(Path.home() / ".insightface/models/buffalo_l/w600k_r50.onnx")
        model = get_model(model_path)
        model.prepare(ctx_id=0 if self._device == "cuda" else -1)
        logger.info("ArcFace ready.")
        return model

    def _embed(self, crops_rgb: Sequence[np.ndarray]) -> np.ndarray:
        if self._model is None:
            self._model = self._load()
        batch = []
        for crop in crops_rgb:
            # Same preprocessing the model was trained with: 112x112, BGR, (x-127.5)/127.5.
            image = cv2.resize(crop, (112, 112), interpolation=cv2.INTER_CUBIC)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image = (image.astype(np.float32) - 127.5) / 127.5
            batch.append(image.transpose(2, 0, 1))  # HWC -> CHW
        blob = np.stack(batch).astype(np.float32)
        features = self._model.session.run(None, {self._model.session.get_inputs()[0].name: blob})[0]
        # L2-normalise so cosine similarity == inner product (what FAISS IndexFlatIP computes).
        return features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-9)
