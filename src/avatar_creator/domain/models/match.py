"""One search hit."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Match:
    """A candidate piece and how close it is to the query.

    ``score`` always follows "higher = closer": cosine similarity in [-1, 1] for vector
    regions, and *negated* deltaE for skin tone, so callers never need a special case.
    """

    piece_id: str
    score: float
