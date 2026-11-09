"""Shared base for the matplotlib renderers."""
from pathlib import Path


class FigureRenderer:
    """Base class holding the one thing every renderer does the same way: saving a figure.

    Rendering is headless (``Agg`` backend) and always writes a PNG, so it works over SSH, in
    CI and on machines with no display, and previews can be opened from disk afterwards.
    """

    @staticmethod
    def _pyplot():
        # Imported lazily and switched to Agg BEFORE pyplot is first imported; matplotlib is
        # only needed by commands that draw, and must never open a GUI window.
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt

    def _save(self, fig, dest: Path) -> Path:
        """Write ``fig`` to ``dest`` (creating folders) and release it."""
        dest.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(dest, dpi=150)
        self._pyplot().close(fig)  # figures hold memory until closed - matters in batch runs
        return dest
