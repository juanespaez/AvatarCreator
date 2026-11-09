"""Adapters that touch real libraries/filesystem but need no ML models."""
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from avatar_creator.domain.models.embedding_bank import EmbeddingBank
from avatar_creator.domain.models.encoder_kind import EncoderKind
from avatar_creator.domain.models.face_region import FaceRegion
from avatar_creator.domain.models.match_mode import MatchMode
from avatar_creator.infrastructure.encoders.skin_tone_encoder import SkinToneEncoder
from avatar_creator.infrastructure.filesystem.filesystem_asset_repository import FilesystemAssetRepository
from avatar_creator.infrastructure.filesystem.project_paths import ProjectPaths
from avatar_creator.infrastructure.imaging.opencv_image_loader import OpenCvImageLoader
from avatar_creator.infrastructure.persistence.pickle_embedding_bank_store import PickleEmbeddingBankStore
from avatar_creator.infrastructure.search.color_distance_searcher import ColorDistanceSearcher
from avatar_creator.infrastructure.search.faiss_cosine_searcher import FaissCosineSearcher


def write_png(path: Path, rgba: tuple[int, int, int, int], size: tuple[int, int] = (10, 20)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((size[0], size[1], 4), np.uint8)
    image[:] = rgba  # OpenCV order is BGRA
    cv2.imwrite(str(path), image)


class FaissCosineSearcherTest(unittest.TestCase):
    def setUp(self) -> None:
        vectors = np.eye(3, dtype=np.float32)
        # Two vectors for piece "a" (many refs per piece), one for "b".
        self.bank = EmbeddingBank(ids=["a", "a", "b"], vectors=vectors[[0, 1, 2]])
        self.searcher = FaissCosineSearcher()
        self.searcher.fit(self.bank)

    def test_returns_distinct_pieces_best_first(self) -> None:
        hits = self.searcher.search(np.array([0, 1, 0], np.float32), k=3)
        self.assertEqual([m.piece_id for m in hits], ["a", "b"])  # 'a' collapsed, only 2 pieces exist
        self.assertAlmostEqual(hits[0].score, 1.0, places=5)

    def test_empty_bank_returns_no_matches(self) -> None:
        searcher = FaissCosineSearcher()
        searcher.fit(EmbeddingBank())
        self.assertEqual(searcher.search(np.zeros(512, np.float32), k=3), [])

    def test_pairwise_is_cosine_for_normalised_vectors(self) -> None:
        np.testing.assert_allclose(self.searcher.pairwise(self.bank.vectors), np.eye(3), atol=1e-6)


class ColorDistanceSearcherTest(unittest.TestCase):
    def test_nearest_colour_wins_and_scores_are_negated_distance(self) -> None:
        labs = np.array([[50, 0, 0], [80, 10, 10], [20, -5, -5]], np.float32)
        searcher = ColorDistanceSearcher()
        searcher.fit(EmbeddingBank(ids=["mid", "light", "dark"], vectors=labs, encoder=EncoderKind.SKIN))
        hits = searcher.search(np.array([79, 10, 10], np.float32), k=2)
        self.assertEqual(hits[0].piece_id, "light")
        self.assertLessEqual(hits[0].score, 0)  # negated distance: higher (closer to 0) = closer
        self.assertGreater(hits[0].score, hits[1].score)

    def test_pairwise_diagonal_is_best(self) -> None:
        labs = np.array([[50, 0, 0], [80, 10, 10]], np.float32)
        matrix = ColorDistanceSearcher().pairwise(labs)
        self.assertEqual(matrix[0, 0], 0)
        self.assertLess(matrix[0, 1], 0)


class SkinToneEncoderTest(unittest.TestCase):
    def test_flat_colour_patch_gives_its_lab_value(self) -> None:
        patch = np.full((8, 8, 3), 255, np.uint8)  # pure white
        vector = SkinToneEncoder().embed([patch])[0]
        self.assertAlmostEqual(float(vector[0]), 100.0, delta=0.5)  # L* of white
        self.assertEqual(vector.shape, (3,))


class OpenCvImageLoaderTest(unittest.TestCase):
    def test_transparent_piece_is_flattened_onto_white_and_squared(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "piece.png"
            write_png(path, (0, 0, 0, 0))  # fully transparent black
            image = OpenCvImageLoader().load_piece(path, out_size=32)
        self.assertEqual(image.shape, (32, 32, 3))
        self.assertTrue((image == 255).all())  # transparency became white, not black

    def test_unreadable_files_give_none_or_raise(self) -> None:
        loader = OpenCvImageLoader()
        self.assertIsNone(loader.load_piece(Path("does_not_exist.png")))
        self.assertIsNone(loader.load_rgba(Path("does_not_exist.png")))
        with self.assertRaises(FileNotFoundError):
            loader.load_photo(Path("does_not_exist.png"))

    def test_load_rgba_resizes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "layer.png"
            write_png(path, (1, 2, 3, 255))
            layer = OpenCvImageLoader().load_rgba(path, (16, 16))
        self.assertEqual(layer.shape, (16, 16, 4))
        self.assertEqual(tuple(layer[0, 0]), (3, 2, 1, 255))  # BGRA on disk -> RGBA in memory


class FilesystemAssetRepositoryTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        for name in ("b_02.png", "a_01.png"):
            write_png(self.root / "pieces" / "eyes" / name, (0, 0, 0, 255))
        (self.root / "pieces" / "eyes" / "notes.txt").write_text("ignored")
        write_png(self.root / "refs" / "eyes" / "a_01" / "p1.jpg", (0, 0, 0, 255))
        write_png(self.root / "refs" / "eyes" / "a_01" / "p2.png", (0, 0, 0, 255))
        write_png(self.root / "queries" / "me.png", (0, 0, 0, 255))
        self.repo = FilesystemAssetRepository(ProjectPaths(self.root))

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_pieces_are_sorted_images_only_with_stem_ids(self) -> None:
        self.assertEqual([a.piece_id for a in self.repo.pieces(FaceRegion.EYES)], ["a_01", "b_02"])

    def test_references_take_their_id_from_the_parent_folder(self) -> None:
        self.assertEqual([a.piece_id for a in self.repo.references(FaceRegion.EYES)], ["a_01", "a_01"])

    def test_missing_folders_mean_no_assets(self) -> None:
        self.assertEqual(self.repo.pieces(FaceRegion.HAIR), [])
        self.assertEqual(self.repo.references(FaceRegion.HAIR), [])

    def test_find_piece(self) -> None:
        self.assertEqual(self.repo.find_piece(FaceRegion.EYES, "b_02").path.name, "b_02.png")
        self.assertIsNone(self.repo.find_piece(FaceRegion.EYES, "zzz"))

    def test_resolve_query_accepts_bare_name_without_extension(self) -> None:
        self.assertEqual(self.repo.resolve_query("me").name, "me.png")
        with self.assertRaises(FileNotFoundError):
            self.repo.resolve_query("nobody")


class PickleEmbeddingBankStoreTest(unittest.TestCase):
    def test_round_trip_and_miss(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PickleEmbeddingBankStore(Path(tmp) / "cache")
            self.assertIsNone(store.load(FaceRegion.EYES, EncoderKind.ARCFACE, MatchMode.DIRECT))
            bank = EmbeddingBank(["a"], np.ones((1, 4), np.float32), MatchMode.DIRECT, EncoderKind.ARCFACE, 2)
            store.save(FaceRegion.EYES, bank)
            loaded = store.load(FaceRegion.EYES, EncoderKind.ARCFACE, MatchMode.DIRECT)
            self.assertEqual(loaded.ids, ["a"])
            self.assertEqual(loaded.n_failed, 2)
            self.assertIs(loaded.mode, MatchMode.DIRECT)
            # File name matches the notebook/old-script cache naming, so old caches stay valid.
            self.assertTrue((Path(tmp) / "cache" / "eyes__arcface__direct.pkl").exists())

    def test_reads_the_legacy_dict_format(self) -> None:
        import pickle

        with tempfile.TemporaryDirectory() as tmp:
            legacy = dict(ids=["x"], vectors=np.zeros((1, 512), np.float32), mode="reference", encoder="clip", n_failed=0)
            (Path(tmp) / "hair__clip__reference.pkl").write_bytes(pickle.dumps(legacy))
            bank = PickleEmbeddingBankStore(Path(tmp)).load(FaceRegion.HAIR, EncoderKind.CLIP, MatchMode.REFERENCE)
            self.assertIs(bank.mode, MatchMode.REFERENCE)
            self.assertIs(bank.encoder, EncoderKind.CLIP)


class ArtifactExporterTest(unittest.TestCase):
    def test_writes_manifest_vectors_ids_and_faiss_but_no_index_for_skin(self) -> None:
        from avatar_creator.application.recognizers.eye_recognizer import EyeRecognizer
        from avatar_creator.application.recognizers.skin_tone_recognizer import SkinToneRecognizer
        from avatar_creator.infrastructure.persistence.artifact_exporter import ArtifactExporter

        banks = {
            FaceRegion.EYES: EmbeddingBank(["a"], np.eye(1, 512, dtype=np.float32)),
            FaceRegion.SKIN_TONE: EmbeddingBank(["p"], np.zeros((1, 3), np.float32), encoder=EncoderKind.SKIN),
        }
        specs = {FaceRegion.EYES: EyeRecognizer.spec, FaceRegion.SKIN_TONE: SkinToneRecognizer.spec}
        counts = {FaceRegion.EYES: (1, 0), FaceRegion.SKIN_TONE: (1, 0)}
        with tempfile.TemporaryDirectory() as tmp:
            ArtifactExporter(Path(tmp)).export(banks, specs, counts, MatchMode.DIRECT, 3)
            names = {p.name for p in Path(tmp).iterdir()}
            manifest = json.loads((Path(tmp) / "manifest.json").read_text())
        self.assertIn("eyes.faiss", names)
        self.assertNotIn("skin_tone.faiss", names)
        self.assertEqual(manifest["encoder"]["eyes"], "arcface")
        self.assertEqual(manifest["crop_cfg"]["eyes"], {"pad": 0.30, "dy": 0.0})


if __name__ == "__main__":
    unittest.main()
