"""Accuracy of one (region, encoder) pair."""
from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationResult:
    """Leave-one-out retrieval accuracy for one region with one encoder.

    Attributes:
        n: Reference photos evaluated.
        n_pieces: Distinct pieces among them.
        top1: Fraction whose true piece came back at rank 1.
        top3: Fraction whose true piece came back within rank 3.
        chance: ``1 / n_pieces`` - what random guessing would score.
    """

    n: int
    n_pieces: int
    top1: float
    top3: float
    chance: float

    @property
    def above_chance(self) -> bool:
        """True when top-1 beats random by more than 2x (the notebook's verdict rule)."""
        return self.top1 > 2 * self.chance
