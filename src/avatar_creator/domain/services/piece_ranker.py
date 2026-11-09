"""Collapses vector-level hits into piece-level ranking."""
from typing import Iterable

from avatar_creator.domain.models.match import Match


class PieceRanker:
    """Ranks *pieces*, not vectors.

    Both searchers (cosine and colour distance) need identical aggregation, so it lives here
    once (DRY). In REFERENCE mode several vectors share one piece id, hence the aggregation.
    """

    def top_k(self, scored: Iterable[tuple[str, float]], k: int) -> list[Match]:
        """Keep each piece's best score, then return the ``k`` best pieces.

        Aggregation is ``max`` rather than ``mean`` on purpose: one very close reference is
        stronger evidence than several mediocre ones, and ``mean`` would punish a piece for
        having many diverse reference photos - the piece you most want to win.

        Args:
            scored: ``(piece_id, score)`` pairs, higher = closer, in any order. Insertion order
                breaks ties (stable sort), so feed best-first for reproducible results.
            k: How many pieces to return.
        """
        best: dict[str, float] = {}
        for piece_id, score in scored:
            if piece_id not in best or score > best[piece_id]:
                best[piece_id] = score
        ranked = sorted(best.items(), key=lambda item: -item[1])[:k]
        return [Match(piece_id, score) for piece_id, score in ranked]
