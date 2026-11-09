"""FAISS exact-cosine adapter for the SimilaritySearcher port."""
import faiss
import numpy as np

from avatar_creator.domain.models.embedding_bank import EmbeddingBank
from avatar_creator.domain.models.match import Match
from avatar_creator.domain.services.piece_ranker import PieceRanker
from avatar_creator.ports.similarity_searcher import SimilaritySearcher


class FaissCosineSearcher(SimilaritySearcher):
    """Exact cosine search with ``IndexFlatIP`` over L2-normalised vectors.

    One searcher (hence one index) per region is the structural point of the design: a global
    index would let a beard win an eye query. Banks are tiny, so exact brute force is correct -
    approximate search would only add recall error.
    """

    def __init__(self, ranker: PieceRanker | None = None) -> None:
        self._ranker = ranker or PieceRanker()
        self._index: faiss.IndexFlatIP | None = None
        self._ids: list[str] = []

    def fit(self, bank: EmbeddingBank) -> None:
        self._ids = list(bank.ids)
        if len(bank) == 0:
            self._index = None  # nothing to search; search() then returns no matches
            return
        index = faiss.IndexFlatIP(bank.vectors.shape[1])  # inner product == cosine when normalised
        index.add(bank.vectors.astype(np.float32))
        self._index = index

    def search(self, query: np.ndarray, k: int) -> list[Match]:
        if self._index is None:
            return []
        # Over-fetch: with many refs per piece the top-N vectors can all belong to ONE piece, so
        # fetching only k vectors would return fewer than k distinct pieces.
        n_fetch = min(len(self._ids), max(k * 10, 50))
        sims, idxs = self._index.search(query.reshape(1, -1).astype(np.float32), n_fetch)
        scored = [
            (self._ids[i], float(sim))
            for sim, i in zip(sims[0], idxs[0])
            if i >= 0  # FAISS pads with -1 when fewer than n_fetch results exist
        ]
        return self._ranker.top_k(scored, k)

    def pairwise(self, vectors: np.ndarray) -> np.ndarray:
        return vectors @ vectors.T
