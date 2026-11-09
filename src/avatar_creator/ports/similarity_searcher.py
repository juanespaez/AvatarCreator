"""Contract: nearest-piece search over one region's bank."""
from abc import ABC, abstractmethod

import numpy as np

from avatar_creator.domain.models.embedding_bank import EmbeddingBank
from avatar_creator.domain.models.match import Match


class SimilaritySearcher(ABC):
    """Strategy for comparing a query vector against a region's bank.

    Cosine (FAISS) and colour distance (deltaE) are different metric spaces, hence different
    strategies. Every strategy follows "higher score = closer", so callers never special-case.
    A searcher is stateful (it holds the index of the bank it was fitted on): use one
    instance per region.
    """

    @abstractmethod
    def fit(self, bank: EmbeddingBank) -> None:
        """Prepare the searcher (e.g. build an index) for ``bank``."""

    @abstractmethod
    def search(self, query: np.ndarray, k: int) -> list[Match]:
        """Return the top-``k`` distinct pieces for one query vector, best first."""

    @abstractmethod
    def pairwise(self, vectors: np.ndarray) -> np.ndarray:
        """Return the ``(N, N)`` similarity matrix of ``vectors`` with themselves.

        Used by the leave-one-out evaluation. Needs no ``fit``.
        """
