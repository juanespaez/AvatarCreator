"""The project's folder layout."""
from pathlib import Path


class ProjectPaths:
    """Resolves every folder the pipeline reads or writes from one root.

    Layout (the data contract from the README)::

        root/
          pieces/<region>/<piece_id>.png          the Lego bank (what gets retrieved)
          refs/<region>/<piece_id>/<photo>.jpg    real photos labelled by folder name
          queries/<photo>.jpg                     full-face photos to run on
          _cache/                                 embedding cache + previews (reproducible)
          artifacts/                              exported vectors / indices / manifest
    """

    #: ``.../src/avatar_creator/infrastructure/filesystem/project_paths.py`` -> repository root.
    DEFAULT_ROOT: Path = Path(__file__).resolve().parents[4]

    def __init__(self, root: Path | None = None, cache_dir: Path | None = None) -> None:
        self.root: Path = Path(root) if root else self.DEFAULT_ROOT
        # The cache can be redirected (e.g. to force a from-scratch rebuild) without moving data.
        self.cache_dir: Path = Path(cache_dir) if cache_dir else self.root / "_cache"

    @property
    def pieces_dir(self) -> Path:
        return self.root / "pieces"

    @property
    def refs_dir(self) -> Path:
        return self.root / "refs"

    @property
    def queries_dir(self) -> Path:
        return self.root / "queries"

    @property
    def previews_dir(self) -> Path:
        return self.cache_dir / "previews"

    @property
    def artifacts_dir(self) -> Path:
        return self.root / "artifacts"
