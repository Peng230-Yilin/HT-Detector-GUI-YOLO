import sys
import itertools
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GUI_ROOT = PROJECT_ROOT / "Peng1.0_GUI"
sys.path.insert(0, str(GUI_ROOT))

from batch_state import (  # noqa: E402
    BatchState,
    ImageItem,
    ImageStatus,
    SampleError,
    SampleErrorType,
    SampleResult,
    assign_batch_numbers,
    assign_image_numbers,
    build_image_items,
    natural_sort_paths,
    pair_cuvettes_and_liquids,
    sort_spatially,
)


def sample(x, y, image_order=1, source_file="image.png"):
    box = (x, y, x + 20, y + 40)
    return SampleResult(
        image_order=image_order,
        source_file=source_file,
        cuvette_box=box,
        liquid_box=(x + 4, y + 10, x + 16, y + 35),
        roi_box=(x + 6, y + 15, x + 14, y + 30),
        red=1.0,
        green=2.0,
        blue=3.0,
    )


class NaturalSortTests(unittest.TestCase):
    def test_empty_and_single(self):
        self.assertEqual(natural_sort_paths([]), [])
        self.assertEqual(natural_sort_paths(["image1.PNG"]), ["image1.PNG"])

    def test_natural_order_and_image_order(self):
        values = ["image10.png", "image1.png", "image2.png"]
        items = build_image_items(values)
        self.assertEqual([item.original_filename for item in items], [
            "image1.png", "image2.png", "image10.png"
        ])
        self.assertEqual([item.image_order for item in items], [1, 2, 3])

    def test_case_extension_directories_and_duplicates_are_deterministic(self):
        values = [
            str(Path("z") / "IMAGE2.jpg"),
            str(Path("a") / "image2.JPG"),
            str(Path("a") / "image10.png"),
            str(Path("a") / "image2.JPG"),
        ]
        ordered = natural_sort_paths(values)
        self.assertEqual(ordered[0], values[1])
        self.assertEqual(ordered[1], values[3])
        self.assertEqual(ordered[2], values[0])
        self.assertEqual(ordered[3], values[2])

    def test_original_filename_is_not_modified(self):
        original = str(Path("folder") / "My IMAGE02.JpG")
        item = build_image_items([original])[0]
        self.assertEqual(item.original_filename, "My IMAGE02.JpG")
        self.assertEqual(original, str(Path("folder") / "My IMAGE02.JpG"))

    def test_batch_state_factory(self):
        state = BatchState.from_paths(["image2.png", "image1.png"])
        self.assertEqual([item.image_order for item in state.images], [1, 2])
        self.assertEqual(state.current_image_index, 0)

    def test_empty_batch_index_and_current_image(self):
        state = BatchState()
        self.assertIsNone(state.current_image_index)
        self.assertIsNone(state.current_image)
        with self.assertRaises(IndexError):
            state.current_image_index = 0

    def test_current_image_index_contract(self):
        images = build_image_items(["a.png", "b.png"])
        state = BatchState(images=images)
        self.assertIs(state.current_image, state.images[0])
        state.current_image_index = 1
        self.assertIs(state.current_image, state.images[1])
        for invalid in (-1, len(images), 100):
            with self.assertRaises(IndexError):
                state.current_image_index = invalid

    def test_invalid_constructor_index_and_list_mutation(self):
        images = build_image_items(["a.png"])
        with self.assertRaises(IndexError):
            BatchState(images=images, current_image_index=-1)
        with self.assertRaises(IndexError):
            BatchState(images=images, current_image_index=1)
        with self.assertRaises(IndexError):
            BatchState(images=[], current_image_index=0)
        state = BatchState(images=images)
        state.images.clear()
        with self.assertRaises(IndexError):
            _ = state.current_image


class SpatialSortTests(unittest.TestCase):
    def test_single_row_left_to_right(self):
        ordered = sort_spatially([sample(100, 10), sample(10, 12), sample(50, 8)])
        self.assertEqual([value.cuvette_box[0] for value in ordered], [10, 50, 100])

    def test_two_rows_top_then_bottom_and_shuffled_input(self):
        values = [sample(80, 100), sample(70, 10), sample(10, 100), sample(20, 10)]
        ordered = sort_spatially(values)
        self.assertEqual(
            [(value.cuvette_box[0], value.cuvette_box[1]) for value in ordered],
            [(20, 10), (70, 10), (10, 100), (80, 100)],
        )


class PairingTests(unittest.TestCase):
    def test_unique_center_containment_pair(self):
        result = pair_cuvettes_and_liquids(
            [(0, 0, 30, 80)], [(5, 20, 25, 70)], source_file="one.png"
        )
        self.assertEqual(len(result.pairs), 1)
        self.assertEqual(result.errors, [])

    def test_horizontal_overlap_fallback_can_pair_uniquely(self):
        result = pair_cuvettes_and_liquids(
            [(0, 0, 30, 30)], [(5, 35, 25, 60)]
        )
        self.assertEqual(len(result.pairs), 1)
        self.assertEqual(result.errors, [])

    def test_no_candidate_records_both_unmatched_targets(self):
        result = pair_cuvettes_and_liquids(
            [(0, 0, 20, 40)], [(100, 0, 120, 40)]
        )
        self.assertEqual(result.pairs, [])
        self.assertEqual(
            {error.error_type for error in result.errors},
            {SampleErrorType.UNMATCHED_CUVETTE, SampleErrorType.UNMATCHED_LIQUID},
        )

    def test_multiple_candidates_are_ambiguous_not_forced(self):
        result = pair_cuvettes_and_liquids(
            [(0, 0, 30, 80), (20, 0, 50, 80)], [(22, 20, 28, 70)]
        )
        self.assertEqual(result.pairs, [])
        self.assertIn(SampleErrorType.AMBIGUOUS_LIQUID,
                      {error.error_type for error in result.errors})
        self.assertTrue(all(error.no_in_image is None and error.batch_no is None
                            for error in result.errors))

    def test_all_input_permutations_have_identical_complete_result(self):
        cuvettes = [(0, 0, 30, 80), (20, 0, 50, 80), (100, 0, 130, 80)]
        liquids = [(22, 20, 28, 70), (105, 20, 125, 70)]
        results = [
            pair_cuvettes_and_liquids(cuvette_order, liquid_order)
            for cuvette_order in itertools.permutations(cuvettes)
            for liquid_order in itertools.permutations(liquids)
        ]
        self.assertTrue(all(result == results[0] for result in results[1:]))
        self.assertEqual(
            results[0].errors[0].related_cuvette_boxes,
            sorted(results[0].errors[0].related_cuvette_boxes),
        )


class NumberingTests(unittest.TestCase):
    def test_valid_error_valid_numbers_only_valid_samples(self):
        left = sample(0, 0)
        right = sample(100, 0)
        error = SampleError(
            1, "image.png", SampleErrorType.INVALID_ROI, "empty ROI",
            position=(60, 20),
        )
        image = ImageItem("image.png", "image.png", 1, status=ImageStatus.COMPLETED,
                          samples=[right, left], errors=[error])
        assign_batch_numbers([image])
        self.assertEqual([value.no_in_image for value in image.samples], [1, 2])
        self.assertEqual([value.batch_no for value in image.samples], [1, 2])
        self.assertIsNone(error.no_in_image)
        self.assertIsNone(error.batch_no)

    def test_per_image_resets_and_batch_continues(self):
        first = ImageItem("a.png", "a.png", 1, status=ImageStatus.COMPLETED,
                          samples=[sample(0, 0, 1, "a.png")])
        second = ImageItem("b.png", "b.png", 2, status=ImageStatus.COMPLETED, samples=[
            sample(60, 0, 2, "b.png"), sample(0, 0, 2, "b.png")
        ])
        self.assertEqual(assign_batch_numbers([second, first]), 3)
        self.assertEqual([value.no_in_image for value in first.samples], [1])
        self.assertEqual([value.no_in_image for value in second.samples], [1, 2])
        self.assertEqual([value.batch_no for value in first.samples], [1])
        self.assertEqual([value.batch_no for value in second.samples], [2, 3])

    def test_failed_image_does_not_consume_batch_number(self):
        failed = ImageItem("a.png", "a.png", 1, status=ImageStatus.FAILED)
        valid = ImageItem("b.png", "b.png", 2, status=ImageStatus.COMPLETED,
                          samples=[sample(0, 0, 2, "b.png")])
        self.assertEqual(assign_batch_numbers([failed, valid]), 1)
        self.assertEqual(valid.samples[0].batch_no, 1)

    def test_failed_stale_sample_is_cleared_without_batch_gap(self):
        first = ImageItem("a.png", "a.png", 1, status=ImageStatus.COMPLETED,
                          samples=[sample(0, 0), sample(40, 0)])
        stale = sample(0, 0, 2, "b.png")
        stale.batch_no = 99
        failed = ImageItem("b.png", "b.png", 2, status=ImageStatus.FAILED,
                           samples=[stale])
        last = ImageItem("c.png", "c.png", 3, status=ImageStatus.COMPLETED,
                         samples=[sample(0, 0), sample(40, 0), sample(80, 0)])
        self.assertEqual(assign_batch_numbers([first, failed, last]), 5)
        self.assertEqual([value.batch_no for value in first.samples], [1, 2])
        self.assertIsNone(stale.batch_no)
        self.assertEqual([value.batch_no for value in last.samples], [3, 4, 5])
        self.assertEqual(assign_batch_numbers([first, failed, last]), 5)

    def test_assign_image_numbers_is_spatial(self):
        ordered = assign_image_numbers([sample(100, 0), sample(0, 0)])
        self.assertEqual([value.cuvette_box[0] for value in ordered], [0, 100])
        self.assertEqual([value.no_in_image for value in ordered], [1, 2])


class CompatibilityTests(unittest.TestCase):
    def test_legacy_single_image_target_fields_are_unchanged(self):
        value = sample(0, 0)
        value.no_in_image = 1
        target = value.legacy_target(4.5)
        self.assertEqual(set(target), {
            "No.", "Con.", "Red", "Green", "Blue",
            "cuvette_box", "liquid_box", "rgb_roi",
        })
        self.assertEqual(target["No."], 1)
        self.assertEqual(target["Con."], 4.5)


if __name__ == "__main__":
    unittest.main()
