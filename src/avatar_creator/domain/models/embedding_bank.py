"""The searchable vectors of one region."""
from dataclasses import dataclass, field

import numpy as np

from avatar_creator.domain.models.encoder_kind import EncoderKind
from avatar_creator.domain.models.match_mode import MatchMode

#: Width of the ArcFace / CLIP vectors. Also used for empty banks so shapes stay consistent.
EMBEDDING_DIM = 512


@dataclass
class EmbeddingBank:
    """Vectors (and their piece ids) for one region, built in one mode with one encoder.

    ``ids[i]`` is the piece id of ``vectors[i]``. In REFERENCE mode several vectors share one
    id (many photos per piece); a search therefore returns *pieces*, not vectors.

    Attributes:
        ids: Piece id per vector.
        vectors: ``(N, D)`` float32 matrix; D is 512 for ArcFace/CLIP and 3 for skin (Lab).
        mode: Concrete mode the bank was built in (never HYBRID).
        encoder: Encoder that produced the vectors. Vectors from different encoders (or
            different crop configs) must never be mixed - they give plausible-looking wrong
            answers instead of errors.
        n_failed: Source images that could not be cropped/read and were left out.
    """

    ids: list[str] = field(default_factory=list)
    vectors: np.ndarray = field(default_factory=lambda: np.zeros((0, EMBEDDING_DIM), np.float32))
    mode: MatchMode = MatchMode.DIRECT
    encoder: EncoderKind = EncoderKind.ARCFACE
    n_failed: int = 0

    def __len__(self) -> int:
        return len(self.ids)

    @property
    def unique_pieces(self) -> int:
        """Number of distinct pieces represented (<= len(self) in REFERENCE mode)."""
        return len(set(self.ids))
