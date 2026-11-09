"""Assets -> embedding bank (cache aware)."""
import logging
from typing import Mapping, Sequence

import numpy as np

from avatar_creator.application.face_cropper import FaceCropper
from avatar_creator.domain.models.asset import Asset
from avatar_creator.domain.models.embedding_bank import EmbeddingBank
from avatar_creator.domain.models.encoder_kind import EncoderKind
from avatar_creator.domain.models.match_mode import MatchMode
from avatar_creator.domain.models.region_spec import RegionSpec
from avatar_creator.ports.embedding_bank_store import EmbeddingBankStore
from avatar_creator.ports.image_loader import ImageLoader
from avatar_creator.ports.region_encoder import RegionEncoder

logger = logging.getLogger(__name__)


class EmbeddingBankBuilder:
    """Turns a region's assets into an ``EmbeddingBank``, reusing the cache when possible.

    How the "crop" for each asset is produced depends on the mode:

    * DIRECT    - the asset is piece artwork: flatten it onto white and square-pad it. There
                  is no face to detect in an isolated part.
    * REFERENCE - the asset is a real photo: run the full landmark crop pipeline for the
                  region, exactly as for a query.
    """

    def __init__(
        self,
        face_cropper: FaceCropper,
        image_loader: ImageLoader,
        encoders: Mapping[EncoderKind, RegionEncoder],
        store: EmbeddingBankStore,
    ) -> None:
        self._face_cropper = face_cropper
        self._image_loader = image_loader
        self._encoders = encoders
        self._store = store

    def build(self, spec: RegionSpec, assets: Sequence[Asset], mode: MatchMode) -> EmbeddingBank:
        """Return the bank for ``spec``'s region in a CONCRETE ``mode`` (not HYBRID).

        Args:
            spec: The region definition (its encoder decides the vector type).
            assets: Pieces (DIRECT) or labelled reference photos (REFERENCE).
            mode: DIRECT or REFERENCE. Resolve HYBRID per region before calling.
        """
        if mode is MatchMode.HYBRID:
            raise ValueError("HYBRID must be resolved to DIRECT or REFERENCE before building a bank")

        cached = self._store.load(spec.region, spec.encoder, mode)
        if cached is not None:
            return cached

        ids: list[str] = []
        crops: list[np.ndarray] = []
        failed = 0
        for asset in assets:
            crop = self._crop_for(spec, asset, mode)
            if crop is None:
                failed += 1  # counted, not hidden: failures are reported by the caller
                continue
            ids.append(asset.piece_id)
            crops.append(crop)

        vectors = self._encoders[spec.encoder].embed(crops)
        bank = EmbeddingBank(ids=ids, vectors=vectors, mode=mode, encoder=spec.encoder, n_failed=failed)
        self._store.save(spec.region, bank)
        return bank

    def _crop_for(self, spec: RegionSpec, asset: Asset, mode: MatchMode) -> np.ndarray | None:
        if mode is MatchMode.DIRECT:
            return self._image_loader.load_piece(asset.path)
        return self._face_cropper.crop_photo(asset.path, [spec]).get(spec.region)
