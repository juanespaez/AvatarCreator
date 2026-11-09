"""Pickle-file adapter for the EmbeddingBankStore port."""
import pickle
from pathlib import Path

from avatar_creator.domain.models.embedding_bank import EmbeddingBank
from avatar_creator.domain.models.encoder_kind import EncoderKind
from avatar_creator.domain.models.face_region import FaceRegion
from avatar_creator.domain.models.match_mode import MatchMode
from avatar_creator.ports.embedding_bank_store import EmbeddingBankStore


class PickleEmbeddingBankStore(EmbeddingBankStore):
    """Caches banks as ``<region>__<encoder>__<mode>.pkl`` inside a cache directory.

    The on-disk format is a plain dict (not a pickled domain object) and the file names are
    identical to the notebook's and the old scripts', so existing caches keep working and the
    format does not break when domain classes move.

    Only load caches this project wrote: unpickling untrusted files can execute code.
    """

    def __init__(self, cache_dir: Path) -> None:
        self._cache_dir = cache_dir

    def _path(self, region: FaceRegion, encoder: EncoderKind, mode: MatchMode) -> Path:
        return self._cache_dir / f"{region.value}__{encoder.value}__{mode.value}.pkl"

    def load(self, region: FaceRegion, encoder: EncoderKind, mode: MatchMode) -> EmbeddingBank | None:
        path = self._path(region, encoder, mode)
        if not path.exists():
            return None
        with open(path, "rb") as handle:
            # SAFE-BY-SCOPE: the file is in this project's own _cache/ (written by save() below),
            # and pickle is kept only so caches from the notebook/old scripts stay valid.
            data = pickle.load(handle)
        return EmbeddingBank(
            ids=data["ids"],
            vectors=data["vectors"],
            mode=MatchMode(data["mode"]),
            encoder=EncoderKind(data["encoder"]),
            n_failed=data["n_failed"],
        )

    def save(self, region: FaceRegion, bank: EmbeddingBank) -> None:
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        data = dict(
            ids=bank.ids,
            vectors=bank.vectors,
            mode=bank.mode.value,
            encoder=bank.encoder.value,
            n_failed=bank.n_failed,
        )
        with open(self._path(region, bank.encoder, bank.mode), "wb") as handle:
            pickle.dump(data, handle)
