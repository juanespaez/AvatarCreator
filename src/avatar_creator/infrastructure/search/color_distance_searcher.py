"""CIEDE2000 colour-distance adapter for the SimilaritySearcher port."""
import numpy as np

from avatar_creator.domain.models.embedding_bank import EmbeddingBank
from avatar_creator.domain.models.match import Match
from avatar_creator.domain.services.piece_ranker import PieceRanker
from avatar_creator.ports.similarity_searcher import SimilaritySearcher


class ColorDistanceSearcher(SimilaritySearcher):
    """Ranks pieces by deltaE2000 between CIELAB colours (no FAISS: not a cosine space).

    deltaE2000 is the standard perceptual colour difference; ~1.0 is a just-noticeable
    difference. Scores are NEGATED distances so that "higher = closer" holds like everywhere
    else and callers need no special case.
    """

    def __init__(self, ranker: PieceRanker | None = None) -> None:
        self._ranker = ranker or PieceRanker()
        self._bank: EmbeddingBank = EmbeddingBank()

    def fit(self, bank: EmbeddingBank) -> None:
        self._bank = bank

    @staticmethod
    def _distance(query: np.ndarray, bank_vectors: np.ndarray) -> np.ndarray:
        from skimage.color import deltaE_ciede2000

        return np.array([deltaE_ciede2000(query, b) for b in bank_vectors], dtype=np.float32)

    def search(self, query: np.ndarray, k: int) -> list[Match]:
        distances = self._distance(query, self._bank.vectors)
        # Closest first, so a piece's first sighting is its best and PieceRanker keeps it.
        order = np.argsort(distances)
        return self._ranker.top_k(((self._bank.ids[i], -float(distances[i])) for i in order), k)

    def pairwise(self, vectors: np.ndarray) -> np.ndarray:
        return -np.array([self._distance(v, vectors) for v in vectors])
