"""Leave-one-out A/B evaluation of encoders per region (notebook cell 10)."""
from collections import Counter
from typing import Callable, Mapping, Sequence

import numpy as np

from avatar_creator.application.face_cropper import FaceCropper
from avatar_creator.domain.models.encoder_kind import EncoderKind
from avatar_creator.domain.models.evaluation_result import EvaluationResult
from avatar_creator.domain.models.face_region import FaceRegion
from avatar_creator.domain.models.region_spec import RegionSpec
from avatar_creator.domain.models.routing_row import RoutingRow
from avatar_creator.ports.asset_repository import AssetRepository
from avatar_creator.ports.region_encoder import RegionEncoder
from avatar_creator.ports.similarity_searcher import SimilaritySearcher


class RoutingEvaluator:
    """Measures which encoder retrieves the right piece best, per region.

    This turns the encoder routing table from an assumption into a measurement.

    Protocol: for every reference photo, embed it, remove THAT PHOTO (not the whole piece) from
    the candidate set, search, and check whether the true piece comes back at rank 1 / rank 3.
    Needs >= 2 refs per piece: with a single ref, removing it removes the only correct answer
    and accuracy is 0 by construction.

    How to read the result:
        * ArcFace >> CLIP on a region -> the routing was right for it.
        * CLIP >= ArcFace            -> flip that region's encoder.
        * Both near ``chance``       -> the problem is the crop or the labels, not the encoder.
    """

    #: Fewer usable photos than this and the percentages are meaningless.
    MIN_USABLE: int = 5

    def __init__(
        self,
        assets: AssetRepository,
        face_cropper: FaceCropper,
        encoders: Mapping[EncoderKind, RegionEncoder],
        searcher_factory: Callable[[EncoderKind], SimilaritySearcher],
        max_n: int = 400,
    ) -> None:
        self._assets = assets
        self._face_cropper = face_cropper
        self._encoders = encoders
        self._searcher_factory = searcher_factory
        self._max_n = max_n

    def evaluate_all(self, specs: Mapping[FaceRegion, RegionSpec]) -> list[RoutingRow]:
        """Evaluate every region with every candidate encoder; regions lacking data are omitted."""
        rows: list[RoutingRow] = []
        for spec in specs.values():
            rows.extend(self.evaluate_region(spec))
        return rows

    def evaluate_region(self, spec: RegionSpec) -> list[RoutingRow]:
        """Evaluate one region with each candidate encoder (crops are computed once and shared)."""
        refs = self._assets.references(spec.region)
        counts = Counter(asset.piece_id for asset in refs)
        usable = [asset for asset in refs if counts[asset.piece_id] >= 2][: self._max_n]
        if len(usable) < self.MIN_USABLE:
            return []

        ids: list[str] = []
        crops: list[np.ndarray] = []
        for asset in usable:
            crop = self._face_cropper.crop_photo(asset.path, [spec]).get(spec.region)
            if crop is not None:
                ids.append(asset.piece_id)
                crops.append(crop)
        if len(crops) < self.MIN_USABLE:
            return []

        rows: list[RoutingRow] = []
        for kind in self._candidate_encoders(spec.region):
            vectors = self._encoders[kind].embed(crops)
            similarity = self._searcher_factory(kind).pairwise(vectors)
            rows.append(RoutingRow(spec.region, kind, self._score(similarity, ids)))
        return rows

    @staticmethod
    def best_routing(rows: Sequence[RoutingRow]) -> dict[FaceRegion, tuple[EncoderKind, float]]:
        """The best encoder per region by top-1 (the first encoder wins exact ties)."""
        best: dict[FaceRegion, tuple[EncoderKind, float]] = {}
        for row in rows:
            if row.result.top1 > best.get(row.region, (None, -1))[1]:
                best[row.region] = (row.encoder, row.result.top1)
        return best

    @staticmethod
    def _candidate_encoders(region: FaceRegion) -> list[EncoderKind]:
        """ArcFace and CLIP compete everywhere; the colour metric only makes sense for skin."""
        kinds = [EncoderKind.ARCFACE, EncoderKind.CLIP]
        return kinds + [EncoderKind.SKIN] if region is FaceRegion.SKIN_TONE else kinds

    @staticmethod
    def _score(similarity: np.ndarray, ids: Sequence[str]) -> EvaluationResult:
        """Top-1 / top-3 retrieval accuracy from an (N, N) similarity matrix (higher = closer)."""
        labels = np.array(ids)
        similarity = similarity.copy()
        # Leave-one-out: a photo must not retrieve itself.
        np.fill_diagonal(similarity, -1e9)

        top1 = top3 = 0
        for row in range(len(labels)):
            ranked: list[str] = []
            for j in np.argsort(-similarity[row]):
                # Collapse vectors into distinct pieces: many photos share one piece id.
                if labels[j] not in ranked:
                    ranked.append(labels[j])
                if len(ranked) == 3:
                    break
            top1 += ranked[0] == labels[row]
            top3 += labels[row] in ranked

        n = len(labels)
        n_pieces = len(set(ids))
        return EvaluationResult(n=n, n_pieces=n_pieces, top1=top1 / n, top3=top3 / n, chance=1 / n_pieces)
