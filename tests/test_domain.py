"""Domain rules: enums, ranking, alignment and cropping geometry."""
import unittest

import numpy as np

from avatar_creator.domain.models.crop_config import CropConfig
from avatar_creator.domain.models.encoder_kind import EncoderKind
from avatar_creator.domain.models.evaluation_result import EvaluationResult
from avatar_creator.domain.models.face_region import FaceRegion
from avatar_creator.domain.models.match_mode import MatchMode
from avatar_creator.domain.models.region_spec import RegionSpec
from avatar_creator.domain.services.face_aligner import FaceAligner
from avatar_creator.domain.services.piece_ranker import PieceRanker
from avatar_creator.domain.services.region_cropper import RegionCropper
from tests.fakes import synthetic_landmarks


class MatchModeTest(unittest.TestCase):
    def test_hybrid_uses_references_only_when_there_are_enough(self) -> None:
        self.assertIs(MatchMode.HYBRID.resolve(n_refs=3, min_refs=3), MatchMode.REFERENCE)
        self.assertIs(MatchMode.HYBRID.resolve(n_refs=2, min_refs=3), MatchMode.DIRECT)

    def test_explicit_modes_are_never_overridden(self) -> None:
        self.assertIs(MatchMode.DIRECT.resolve(n_refs=99, min_refs=3), MatchMode.DIRECT)
        self.assertIs(MatchMode.REFERENCE.resolve(n_refs=0, min_refs=3), MatchMode.REFERENCE)


class FaceRegionTest(unittest.TestCase):
    def test_enum_order_is_the_original_z_order(self) -> None:
        self.assertEqual(
            [r.value for r in FaceRegion],
            ["face_shape", "skin_tone", "eyes", "eyebrows", "nose", "mouth", "beard", "hair"],
        )

    def test_folder_slug_round_trips(self) -> None:
        self.assertIs(FaceRegion("eyes"), FaceRegion.EYES)


class PieceRankerTest(unittest.TestCase):
    def test_keeps_best_score_per_piece_and_orders_descending(self) -> None:
        scored = [("a", 0.2), ("b", 0.9), ("a", 0.5), ("c", 0.1)]
        ranked = PieceRanker().top_k(scored, k=2)
        self.assertEqual([(m.piece_id, m.score) for m in ranked], [("b", 0.9), ("a", 0.5)])

    def test_max_not_mean(self) -> None:
        # "a" has one great ref and many poor ones; it must still beat "b".
        scored = [("a", 0.9)] + [("a", 0.0)] * 10 + [("b", 0.5)] * 3
        self.assertEqual(PieceRanker().top_k(scored, k=1)[0].piece_id, "a")

    def test_ties_keep_first_seen_order(self) -> None:
        ranked = PieceRanker().top_k([("x", 1.0), ("y", 1.0)], k=2)
        self.assertEqual([m.piece_id for m in ranked], ["x", "y"])


class EvaluationResultTest(unittest.TestCase):
    def test_above_chance_needs_double_the_chance_rate(self) -> None:
        self.assertTrue(EvaluationResult(10, 4, 0.6, 0.9, 0.25).above_chance)
        self.assertFalse(EvaluationResult(10, 4, 0.5, 0.9, 0.25).above_chance)


class FaceAlignerTest(unittest.TestCase):
    def test_alignment_levels_a_tilted_eye_line(self) -> None:
        landmarks = synthetic_landmarks()
        landmarks[263] = landmarks[33] + [90, 40]  # tilted eye line
        image = np.zeros((400, 400, 3), np.uint8)
        _, aligned = FaceAligner().align(image, landmarks)
        self.assertAlmostEqual(float(aligned[263][1]), float(aligned[33][1]), places=1)

    def test_inter_ocular_distance(self) -> None:
        self.assertAlmostEqual(FaceAligner.inter_ocular_distance(synthetic_landmarks()), 100.0, places=3)


class RegionCropperTest(unittest.TestCase):
    SPEC = RegionSpec(FaceRegion.NOSE, (1, 2, 3), CropConfig(pad=0.25), EncoderKind.ARCFACE)

    def test_output_is_square_and_requested_size(self) -> None:
        crop = RegionCropper().crop(np.zeros((400, 400, 3), np.uint8), synthetic_landmarks(), self.SPEC, out_size=64)
        self.assertEqual(crop.shape, (64, 64, 3))

    def test_box_hanging_off_the_image_is_padded_not_clamped(self) -> None:
        landmarks = synthetic_landmarks()
        landmarks[[1, 2, 3]] = [[2, 2], [4, 4], [3, 3]]  # region in the top-left corner
        spec = RegionSpec(FaceRegion.NOSE, (1, 2, 3), CropConfig(pad=1.0), EncoderKind.ARCFACE)
        crop = RegionCropper().crop(np.full((400, 400, 3), 77, np.uint8), landmarks, spec, out_size=32)
        self.assertEqual(crop.shape, (32, 32, 3))  # still a full square, not a truncated one


if __name__ == "__main__":
    unittest.main()
