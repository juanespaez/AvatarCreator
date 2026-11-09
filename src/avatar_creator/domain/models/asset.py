"""A labelled image file."""
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Asset:
    """An image on disk together with the piece it stands for.

    For a *piece* the id is the file stem (``apple_01``). For a *reference photo* the id is
    the parent folder name (``refs/eyes/apple_01/person_a.jpg`` -> ``apple_01``), so several
    assets can share one id. The id is the join key between ``pieces/`` and ``refs/``.
    """

    piece_id: str
    path: Path
