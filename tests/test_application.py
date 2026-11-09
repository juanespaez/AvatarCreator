"""Use cases, wired to fakes: recognizers, registry, facade, compositor, evaluator."""
import unittest
from pathlib import Path

import numpy as np

from avatar_creator.application.avatar_compositor import AvatarCompositor
from avatar_creator.application.avatar_recognizer import AvatarRecognizer
from avatar_creator.application.embedding_bank_builder import EmbeddingBankBuilder
from avatar_creator.application.face_cropper import FaceCropper
from avatar_creator.application.recognizer_registry import RecognizerRegistry
from avatar_creator.application.recognizers.eye_recognizer import EyeRecognizer
from avatar_creator.application.recognizers.region_recognizer import RegionRecognizer
from avatar_creator.application.routing_evaluator import RoutingEvaluator
from avatar_creator.domain.models.asset import Asset
from avatar_creator.domain.models.avatar_selection import AvatarSelection
from avatar_creator.domain.models.embedding_bank import EmbeddingBank
from avatar_creator.domain.models.encoder_kind import EncoderKind
from avatar_creator.domain.models.face_region import FaceRegion
from avatar_creator.domain.models.match import Match
from avatar_creator.domain.models.match_mode import MatchMode
from avatar_creator.domain.no_face_detected_error import NoFaceDetectedError
from avatar_creator.infrastructure.search.faiss_cosine_searcher import FaissCosineSearcher
from tests.fakes import (
    FakeEncoder,
    FakeImageLoader,
    FakeLandmarkDetector,
    InMemoryAssetRepository,
    InMemoryBankStore,
    synthetic_landmarks,
)

ENCODERS = {kind: FakeEncoder(kind) for kind in EncoderKind}


def make_searcher(kind: EncoderKind) -> FaissCosineSearcher:
    return FaissCosineSearcher()


class RecognizerRegistryTest(unittest.TestCase):
    def test_every_region_has_a_recognizer_serving_that_region(self) -> None:
        recognizers = RecognizerRegistry(ENCODERS, make_searcher).create_all()
        self.assertEqual(list(recognizers), list(FaceRegion))  # all regions, in z-order
        for region, recognizer in recognizers.items():
            self.assertIs(recognizer.region, region)

    def test_spec_matches_the_original_notebook_config(self) -> None:
        by_region = {r: t.spec for r, t in RecognizerRegistry.RECOGNIZER_TYPES.items()}
        self.assertEqual(by_region[FaceRegion.HAIR].crop.pad, 0.90)
        self.assertEqual(by_region[FaceRegion.HAIR].crop.dy, -0.75)
        self.assertEqual(by_region[FaceRegion.BEARD].crop.dy, 0.10)
        encoders = {r.value: s.encoder.value for r, s in by_region.items()}
        self.assertEqual(
            encoders,
            {"eyes": "arcface", "eyebrows": "arcface", "nose": "arcface", "mouth": "arcface",
             "face_shape": "arcface", "hair": "clip", "beard": "clip", "skin_tone": "skin"},
        )
        self.assertEqual(len(by_region[FaceRegion.EYES].landmark_indices), 28)
        self.assertEqual(len(by_region[FaceRegion.FACE_SHAPE].landmark_indices), 27)


class RegionRecognizerTest(unittest.TestCase):
    def test_rejects_an_encoder_of_the_wrong_kind(self) -> None:
        with self.assertRaises(ValueError):
            EyeRecognizer(FakeEncoder(EncoderKind.CLIP), FaissCosineSearcher())

    def test_rejects_a_bank_from_another_encoder(self) -> None:
        recognizer = EyeRecognizer(FakeEncoder(), FaissCosineSearcher())
        with self.assertRaises(ValueError):
            recognizer.load_bank(EmbeddingBank(encoder=EncoderKind.CLIP))

    def test_recognize_before_load_bank_is_an_error(self) -> None:
        with self.assertRaises(RuntimeError):
            EyeRecognizer(FakeEncoder(), FaissCosineSearcher()).recognize(np.zeros((4, 4, 3), np.uint8))

    def test_recognize_returns_the_nearest_piece(self) -> None:
        recognizer = EyeRecognizer(FakeEncoder(), FaissCosineSearcher())
        # FakeEncoder maps a crop of brightness 5 to vector index 5 % 4 == 1.
        vectors = np.eye(4, dtype=np.float32)
        recognizer.load_bank(EmbeddingBank(ids=["p0", "p1", "p2", "p3"], vectors=vectors))
        hits = recognizer.recognize(np.full((4, 4, 3), 5, np.uint8), k=2)
        self.assertEqual(hits[0].piece_id, "p1")


class AvatarRecognizerTest(unittest.TestCase):
    def _build(self, pieces, references=None, mode=MatchMode.DIRECT, landmarks=True, store=None):
        loader = FakeImageLoader()
        cropper = FaceCropper(loader, FakeLandmarkDetector(synthetic_landmarks() if landmarks else None))
        store = store or InMemoryBankStore()
        builder = EmbeddingBankBuilder(cropper, loader, ENCODERS, store)
        assets = InMemoryAssetRepository(pieces, references)
        recognizers = RecognizerRegistry(ENCODERS, make_searcher).create_all()
        return AvatarRecognizer(recognizers, cropper, builder, assets, mode, min_refs=2), store

    def test_fit_skips_regions_without_data_and_recognize_covers_the_rest(self) -> None:
        facade, _ = self._build({FaceRegion.EYES: [Asset("e1", Path("e1.png")), Asset("e2", Path("e2.png"))]})
        banks = facade.fit()
        self.assertEqual(list(banks), [FaceRegion.EYES])
        selection = facade.recognize(Path("photo.jpg"), k=2)
        self.assertEqual(list(selection.matches), [FaceRegion.EYES])
        self.assertEqual(len(selection.crops), 8)  # every region is still cropped
        self.assertIsNotNone(selection.top1(FaceRegion.EYES))

    def test_no_face_raises(self) -> None:
        facade, _ = self._build({FaceRegion.EYES: [Asset("e1", Path("e1.png"))]}, landmarks=False)
        facade.fit()
        with self.assertRaises(NoFaceDetectedError):
            facade.recognize(Path("photo.jpg"))

    def test_hybrid_picks_references_only_where_there_are_enough(self) -> None:
        pieces = {r: [Asset(f"{r.value}_1", Path("x.png"))] for r in (FaceRegion.EYES, FaceRegion.NOSE)}
        refs = {FaceRegion.EYES: [Asset("eyes_1", Path("r1.jpg")), Asset("eyes_1", Path("r2.jpg"))]}
        facade, _ = self._build(pieces, refs, mode=MatchMode.HYBRID)  # min_refs = 2
        banks = facade.fit()
        self.assertIs(banks[FaceRegion.EYES].mode, MatchMode.REFERENCE)
        self.assertIs(banks[FaceRegion.NOSE].mode, MatchMode.DIRECT)

    def test_banks_come_from_the_cache_the_second_time(self) -> None:
        pieces = {FaceRegion.EYES: [Asset("e1", Path("e1.png"))]}
        facade, store = self._build(pieces)
        facade.fit()
        self.assertEqual(store.saves, 1)
        facade.fit()
        self.assertEqual(store.saves, 1)  # built once, then served from the store

    def test_hybrid_cannot_be_built_directly(self) -> None:
        facade, _ = self._build({})
        builder = facade._bank_builder
        with self.assertRaises(ValueError):
            builder.build(EyeRecognizer.spec, [], MatchMode.HYBRID)


class AvatarCompositorTest(unittest.TestCase):
    def test_later_regions_are_drawn_on_top_and_skin_is_skipped(self) -> None:
        assets = InMemoryAssetRepository(
            {
                FaceRegion.FACE_SHAPE: [Asset("shape", Path("dark.png"))],
                FaceRegion.SKIN_TONE: [Asset("swatch", Path("white_swatch.png"))],
                FaceRegion.HAIR: [Asset("hair", Path("white_hair.png"))],
            }
        )
        selection = AvatarSelection(
            Path("p.jpg"),
            matches={
                FaceRegion.FACE_SHAPE: [Match("shape", 1.0)],
                FaceRegion.SKIN_TONE: [Match("swatch", 1.0)],
                FaceRegion.HAIR: [Match("hair", 1.0)],
            },
        )
        canvas = AvatarCompositor(FakeImageLoader(), assets, canvas_size=8).compose(selection)
        self.assertEqual(canvas.shape, (8, 8, 4))
        self.assertEqual(int(canvas[0, 0, 0]), 255)  # hair (white, last in z-order) is on top

    def test_regions_with_no_artwork_are_skipped_silently(self) -> None:
        selection = AvatarSelection(Path("p.jpg"), matches={FaceRegion.EYES: [Match("ghost", 1.0)]})
        canvas = AvatarCompositor(FakeImageLoader(), InMemoryAssetRepository(), canvas_size=4).compose(selection)
        self.assertEqual(int(canvas.sum()), 0)


class RoutingEvaluatorTest(unittest.TestCase):
    def test_score_counts_top1_top3_and_chance_with_leave_one_out(self) -> None:
        ids = ["a", "a", "b", "b"]
        # Each photo's most similar OTHER photo is its sibling -> perfect retrieval.
        similarity = np.array(
            [[1, 0.9, 0.1, 0.1], [0.9, 1, 0.1, 0.1], [0.1, 0.1, 1, 0.9], [0.1, 0.1, 0.9, 1]], np.float32
        )
        result = RoutingEvaluator._score(similarity, ids)
        self.assertEqual((result.n, result.n_pieces, result.top1, result.top3, result.chance), (4, 2, 1.0, 1.0, 0.5))

    def test_a_photo_cannot_retrieve_itself(self) -> None:
        ids = ["a", "b"]
        similarity = np.array([[1.0, 0.0], [0.0, 1.0]], np.float32)  # only self-similarity is high
        result = RoutingEvaluator._score(similarity, ids)
        self.assertEqual(result.top1, 0.0)

    def test_not_enough_refs_gives_no_rows(self) -> None:
        loader = FakeImageLoader()
        cropper = FaceCropper(loader, FakeLandmarkDetector(synthetic_landmarks()))
        refs = {FaceRegion.EYES: [Asset("a", Path("1.jpg")), Asset("a", Path("2.jpg")), Asset("b", Path("3.jpg"))]}
        evaluator = RoutingEvaluator(InMemoryAssetRepository(references=refs), cropper, ENCODERS, make_searcher)
        self.assertEqual(evaluator.evaluate_region(EyeRecognizer.spec), [])

    def test_best_routing_prefers_the_higher_top1_and_first_on_ties(self) -> None:
        from avatar_creator.domain.models.evaluation_result import EvaluationResult
        from avatar_creator.domain.models.routing_row import RoutingRow

        def row(kind: EncoderKind, top1: float) -> RoutingRow:
            return RoutingRow(FaceRegion.EYES, kind, EvaluationResult(10, 5, top1, 1.0, 0.2))

        best = RoutingEvaluator.best_routing([row(EncoderKind.ARCFACE, 0.4), row(EncoderKind.CLIP, 0.7)])
        self.assertEqual(best[FaceRegion.EYES], (EncoderKind.CLIP, 0.7))
        tie = RoutingEvaluator.best_routing([row(EncoderKind.ARCFACE, 0.5), row(EncoderKind.CLIP, 0.5)])
        self.assertIs(tie[FaceRegion.EYES][0], EncoderKind.ARCFACE)

    def test_full_evaluation_runs_end_to_end_on_fakes(self) -> None:
        loader = FakeImageLoader()
        cropper = FaceCropper(loader, FakeLandmarkDetector(synthetic_landmarks()))
        refs = {FaceRegion.EYES: [Asset(pid, Path(f"{pid}{n}.jpg")) for pid in "ab" for n in range(3)]}
        evaluator = RoutingEvaluator(InMemoryAssetRepository(references=refs), cropper, ENCODERS, make_searcher)
        rows = evaluator.evaluate_region(EyeRecognizer.spec)
        self.assertEqual([r.encoder for r in rows], [EncoderKind.ARCFACE, EncoderKind.CLIP])
        self.assertEqual(rows[0].result.n, 6)


class CliSmokeTest(unittest.TestCase):
    def test_unknown_photo_is_a_clean_error_not_a_traceback(self) -> None:
        import contextlib
        import io
        import tempfile

        from avatar_creator.presentation.avatar_cli import AvatarCli

        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stderr(io.StringIO()) as err:
            code = AvatarCli().run(["--root", tmp, "preview-crops", "missing_photo"])
        self.assertEqual(code, 2)
        self.assertIn("missing_photo", err.getvalue())


if __name__ == "__main__":
    unittest.main()
