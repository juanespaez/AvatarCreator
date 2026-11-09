"""Writes the deployable artifacts (notebook cell 12)."""
import json
from pathlib import Path
from typing import Mapping

import faiss
import numpy as np

from avatar_creator.domain.models.embedding_bank import EmbeddingBank
from avatar_creator.domain.models.encoder_kind import EncoderKind
from avatar_creator.domain.models.face_region import FaceRegion
from avatar_creator.domain.models.match_mode import MatchMode
from avatar_creator.domain.models.region_spec import RegionSpec
from avatar_creator.infrastructure.encoders.arcface_encoder import ArcFaceEncoder
from avatar_creator.infrastructure.encoders.clip_encoder import ClipEncoder


class ArtifactExporter:
    """Persists vectors, ids, FAISS indices and the config that produced them.

    The manifest is saved WITH the vectors on purpose: an embedding file is meaningless without
    knowing which encoder, crop padding and match mode produced it, and mixing two generations
    of vectors in one index yields plausible-looking wrong answers rather than an error. A
    future service loads these at startup instead of re-encoding the bank.
    """

    def __init__(self, out_dir: Path) -> None:
        self._out_dir = out_dir

    def export(
        self,
        banks: Mapping[FaceRegion, EmbeddingBank],
        specs: Mapping[FaceRegion, RegionSpec],
        counts: Mapping[FaceRegion, tuple[int, int]],
        match_mode: MatchMode,
        min_refs: int,
    ) -> list[Path]:
        """Write everything and return the files created.

        Args:
            banks: Fitted bank per region.
            specs: Spec per region (for the encoder routing and crop config in the manifest).
            counts: ``(n_pieces, n_refs)`` per region, for the manifest.
            match_mode: The mode policy the banks were built with.
            min_refs: The HYBRID threshold in force.
        """
        self._out_dir.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []

        manifest = dict(
            categories=[region.value for region in specs],
            encoder={region.value: spec.encoder.value for region, spec in specs.items()},
            match_mode=match_mode.value,
            min_refs=min_refs,
            crop_cfg={
                region.value: dict(pad=spec.crop.pad, dy=spec.crop.dy) for region, spec in specs.items()
            },
            clip_checkpoint=ClipEncoder.MODEL_LABEL,
            arcface=ArcFaceEncoder.MODEL_LABEL,
            counts={
                region.value: dict(pieces=pieces, refs=refs) for region, (pieces, refs) in counts.items()
            },
        )
        written.append(self._write_text("manifest.json", json.dumps(manifest, indent=2)))

        for region, bank in banks.items():
            vectors_path = self._out_dir / f"{region.value}.vectors.npy"
            np.save(vectors_path, bank.vectors)
            written.append(vectors_path)
            written.append(self._write_text(f"{region.value}.ids.json", json.dumps(bank.ids)))
            # Skin tone lives in colour space, not cosine space, so it has no FAISS index.
            if bank.encoder is not EncoderKind.SKIN and len(bank) > 0:
                index = faiss.IndexFlatIP(bank.vectors.shape[1])
                index.add(bank.vectors.astype(np.float32))
                index_path = self._out_dir / f"{region.value}.faiss"
                faiss.write_index(index, str(index_path))
                written.append(index_path)
        return written

    def _write_text(self, name: str, text: str) -> Path:
        path = self._out_dir / name
        path.write_text(text)
        return path
