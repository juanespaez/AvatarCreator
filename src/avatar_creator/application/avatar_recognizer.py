"""Facade: the one object a caller needs to turn a photo into piece picks."""
import logging
from pathlib import Path

from avatar_creator.application.embedding_bank_builder import EmbeddingBankBuilder
from avatar_creator.application.face_cropper import FaceCropper
from avatar_creator.application.recognizers.region_recognizer import RegionRecognizer
from avatar_creator.domain.models.avatar_selection import AvatarSelection
from avatar_creator.domain.models.embedding_bank import EmbeddingBank
from avatar_creator.domain.models.face_region import FaceRegion
from avatar_creator.domain.models.match_mode import MatchMode
from avatar_creator.domain.no_face_detected_error import NoFaceDetectedError
from avatar_creator.ports.asset_repository import AssetRepository

logger = logging.getLogger(__name__)


class AvatarRecognizer:
    """Facade over the whole recognition pipeline.

    Hides crop extraction, bank building and per-region search behind two calls::

        recognizer.fit()                        # build (or load cached) indices once
        selection = recognizer.recognize(photo) # photo -> top-k pieces per region

    It delegates every region-specific decision to the ``RegionRecognizer`` objects, so it
    contains no ``if region is EYES`` branching: regions are looked up through the
    ``FaceRegion`` enum.
    """

    def __init__(
        self,
        recognizers: dict[FaceRegion, RegionRecognizer],
        face_cropper: FaceCropper,
        bank_builder: EmbeddingBankBuilder,
        assets: AssetRepository,
        mode: MatchMode = MatchMode.DIRECT,
        min_refs: int = 3,
    ) -> None:
        """
        Args:
            recognizers: One recognizer per region, in z-order.
            face_cropper: Extracts region crops from the query photo.
            bank_builder: Builds/loads each region's embedding bank.
            assets: Where pieces and reference photos come from.
            mode: DIRECT, REFERENCE or HYBRID (HYBRID is resolved per region).
            min_refs: Labelled refs a region needs before HYBRID trusts REFERENCE mode.
        """
        self._recognizers = recognizers
        self._face_cropper = face_cropper
        self._bank_builder = bank_builder
        self._assets = assets
        self._mode = mode
        self._min_refs = min_refs

    @property
    def recognizers(self) -> dict[FaceRegion, RegionRecognizer]:
        return self._recognizers

    @property
    def mode(self) -> MatchMode:
        return self._mode

    @property
    def min_refs(self) -> int:
        return self._min_refs

    def fit(self) -> dict[FaceRegion, EmbeddingBank]:
        """Build every region's bank and load it into its recognizer.

        Regions with nothing to index are skipped (and logged) rather than failing the whole
        run: the pipeline degrades to the regions that do have data.

        Returns:
            The banks that were loaded, by region.
        """
        banks: dict[FaceRegion, EmbeddingBank] = {}
        for region, recognizer in self._recognizers.items():
            pieces = self._assets.pieces(region)
            refs = self._assets.references(region)
            mode = self._mode.resolve(len(refs), self._min_refs)
            source, noun = (pieces, "pieces") if mode is MatchMode.DIRECT else (refs, "refs")
            if not source:
                logger.info("skip %s: no %s", region.value, noun)
                continue

            spec = recognizer.spec
            logger.info(
                "building index: %s (%s, %d %s, mode=%s) ...",
                region.value, spec.encoder.value, len(source), noun, mode.value,
            )
            bank = self._bank_builder.build(spec, source, mode)
            recognizer.load_bank(bank)
            banks[region] = bank
            logger.info("  -> %d vectors, %d failed", len(bank), bank.n_failed)
        return banks

    def recognize(self, photo_path: Path, k: int = 3) -> AvatarSelection:
        """Pick the ``k`` best pieces per region for one photo.

        Raises:
            NoFaceDetectedError: if the photo contains no detectable face.
        """
        specs = [recognizer.spec for recognizer in self._recognizers.values()]
        crops = self._face_cropper.crop_photo(photo_path, specs)
        if not crops:
            raise NoFaceDetectedError(photo_path)

        selection = AvatarSelection(photo_path=photo_path, crops=crops)
        for region, recognizer in self._recognizers.items():
            # A region is skipped if it has no bank (nothing indexed) or no crop (box was empty).
            if recognizer.is_ready and region in crops:
                selection.matches[region] = recognizer.recognize(crops[region], k)
        return selection
