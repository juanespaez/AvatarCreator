"""Composition root: the only place that knows every concrete class."""
from functools import cached_property

from avatar_creator.application.avatar_compositor import AvatarCompositor
from avatar_creator.application.avatar_recognizer import AvatarRecognizer
from avatar_creator.application.embedding_bank_builder import EmbeddingBankBuilder
from avatar_creator.application.face_cropper import FaceCropper
from avatar_creator.application.recognizer_registry import RecognizerRegistry
from avatar_creator.application.routing_evaluator import RoutingEvaluator
from avatar_creator.domain.models.encoder_kind import EncoderKind
from avatar_creator.domain.models.match_mode import MatchMode
from avatar_creator.infrastructure.encoders.arcface_encoder import ArcFaceEncoder
from avatar_creator.infrastructure.encoders.clip_encoder import ClipEncoder
from avatar_creator.infrastructure.encoders.skin_tone_encoder import SkinToneEncoder
from avatar_creator.infrastructure.filesystem.filesystem_asset_repository import FilesystemAssetRepository
from avatar_creator.infrastructure.filesystem.project_paths import ProjectPaths
from avatar_creator.infrastructure.imaging.opencv_image_loader import OpenCvImageLoader
from avatar_creator.infrastructure.landmarks.mediapipe_landmark_detector import MediaPipeLandmarkDetector
from avatar_creator.infrastructure.persistence.artifact_exporter import ArtifactExporter
from avatar_creator.infrastructure.persistence.pickle_embedding_bank_store import PickleEmbeddingBankStore
from avatar_creator.infrastructure.rendering.avatar_preview_renderer import AvatarPreviewRenderer
from avatar_creator.infrastructure.rendering.crop_preview_renderer import CropPreviewRenderer
from avatar_creator.infrastructure.rendering.match_preview_renderer import MatchPreviewRenderer
from avatar_creator.infrastructure.search.color_distance_searcher import ColorDistanceSearcher
from avatar_creator.infrastructure.search.faiss_cosine_searcher import FaissCosineSearcher
from avatar_creator.ports.asset_repository import AssetRepository
from avatar_creator.ports.image_loader import ImageLoader
from avatar_creator.ports.landmark_detector import LandmarkDetector
from avatar_creator.ports.region_encoder import RegionEncoder
from avatar_creator.ports.similarity_searcher import SimilaritySearcher


class PipelineContainer:
    """Wires ports to adapters (manual dependency injection, no framework needed at this size).

    Everything is a ``cached_property``: built on first access, then reused. Combined with the
    adapters' own lazy model loading, asking for ``avatar_recognizer`` costs nothing until a
    model is actually needed. Swap an adapter here (e.g. another landmark detector) and no
    other layer changes.
    """

    def __init__(
        self,
        paths: ProjectPaths | None = None,
        device: str = "cpu",
        mode: MatchMode = MatchMode.DIRECT,
        min_refs: int = 3,
    ) -> None:
        self.paths = paths or ProjectPaths()
        self._device = device
        self._mode = mode
        self._min_refs = min_refs

    # ---- infrastructure adapters ---------------------------------------------------------
    @cached_property
    def image_loader(self) -> ImageLoader:
        return OpenCvImageLoader()

    @cached_property
    def landmark_detector(self) -> LandmarkDetector:
        return MediaPipeLandmarkDetector()

    @cached_property
    def assets(self) -> AssetRepository:
        return FilesystemAssetRepository(self.paths)

    @cached_property
    def encoders(self) -> dict[EncoderKind, RegionEncoder]:
        # One shared instance per kind: ArcFace alone serves five regions.
        return {
            EncoderKind.ARCFACE: ArcFaceEncoder(self._device),
            EncoderKind.CLIP: ClipEncoder(self._device),
            EncoderKind.SKIN: SkinToneEncoder(),
        }

    @staticmethod
    def make_searcher(kind: EncoderKind) -> SimilaritySearcher:
        """A fresh searcher for ``kind`` (stateful, so never shared between regions)."""
        return ColorDistanceSearcher() if kind is EncoderKind.SKIN else FaissCosineSearcher()

    # ---- application services ------------------------------------------------------------
    @cached_property
    def face_cropper(self) -> FaceCropper:
        return FaceCropper(self.image_loader, self.landmark_detector)

    @cached_property
    def avatar_recognizer(self) -> AvatarRecognizer:
        registry = RecognizerRegistry(self.encoders, self.make_searcher)
        bank_builder = EmbeddingBankBuilder(
            self.face_cropper,
            self.image_loader,
            self.encoders,
            PickleEmbeddingBankStore(self.paths.cache_dir),
        )
        return AvatarRecognizer(
            registry.create_all(), self.face_cropper, bank_builder, self.assets, self._mode, self._min_refs
        )

    @cached_property
    def compositor(self) -> AvatarCompositor:
        return AvatarCompositor(self.image_loader, self.assets)

    @cached_property
    def evaluator(self) -> RoutingEvaluator:
        return RoutingEvaluator(self.assets, self.face_cropper, self.encoders, self.make_searcher)

    @cached_property
    def exporter(self) -> ArtifactExporter:
        return ArtifactExporter(self.paths.artifacts_dir)

    # ---- renderers -----------------------------------------------------------------------
    @cached_property
    def crop_preview_renderer(self) -> CropPreviewRenderer:
        return CropPreviewRenderer()

    @cached_property
    def match_preview_renderer(self) -> MatchPreviewRenderer:
        return MatchPreviewRenderer(self.image_loader, self.assets)

    @cached_property
    def avatar_preview_renderer(self) -> AvatarPreviewRenderer:
        return AvatarPreviewRenderer()
