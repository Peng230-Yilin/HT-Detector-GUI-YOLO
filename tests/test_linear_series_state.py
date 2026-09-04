import math
import sys
from collections.abc import Mapping
from dataclasses import FrozenInstanceError, fields, is_dataclass, replace
from decimal import Decimal
from enum import Enum
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GUI_ROOT = PROJECT_ROOT / "Peng1.0_GUI"
sys.path.insert(0, str(GUI_ROOT))

from linear_series_state import (  # noqa: E402
    LinearImageItem,
    LinearImageStatus,
    LinearSampleError,
    LinearSampleItem,
    LinearSampleStatus,
    LinearSeriesInvariantError,
    LinearSeriesPhase,
    LinearSeriesState,
    RegressionInput,
    RegressionValidationError,
    assign_series_numbers,
    build_image_items,
    natural_sort_paths,
    normalize_concentration,
    parse_concentration,
)


def rgb(red, green=None, blue=None, **extra):
    value = {
        "red": red,
        "green": red + 1 if green is None else green,
        "blue": red + 2 if blue is None else blue,
    }
    value.update(extra)
    return value


def completed_state(paths, counts):
    state = LinearSeriesState.from_paths(paths)
    for image, count in zip(state.images, counts):
        state.mark_image_processing(image.image_order)
        state.accept_image_result(
            image.image_order,
            [rgb(image.image_order * 10 + index) for index in range(count)],
        )
    return state


def set_concentrations(state, values):
    for sample, value in zip(state.all_samples, values):
        state.set_concentration(sample.sample_key, value)


def immutable_value(value):
    """Capture values recursively without retaining mutable domain aliases."""

    if isinstance(value, float):
        if math.isnan(value):
            return ("float", "nan")
        if math.isinf(value):
            return ("float", "inf" if value > 0 else "-inf")
        return value
    if value is None or isinstance(
            value, (str, bytes, int, bool, Decimal, Enum)):
        return value
    if is_dataclass(value):
        return (
            type(value).__qualname__,
            tuple(
                (item.name, immutable_value(getattr(value, item.name)))
                for item in fields(value)
            ),
        )
    if isinstance(value, Mapping):
        entries = [
            (immutable_value(key), immutable_value(item))
            for key, item in value.items()
        ]
        return ("mapping", tuple(sorted(entries, key=repr)))
    if isinstance(value, (list, tuple)):
        return ("sequence", tuple(immutable_value(item) for item in value))
    if isinstance(value, (set, frozenset)):
        return (
            "set",
            tuple(sorted((immutable_value(item) for item in value), key=repr)),
        )
    return ("identity", type(value).__qualname__, id(value))


def stored_state_value(state):
    return immutable_value(object.__getattribute__(state, "_data"))


def direct_sample(
        image_order=1,
        spatial_order=1,
        series_number=1,
        sample_key=None,
        normalized_path=None,
        original_file_name=None,
        **changes):
    normalized_path = normalized_path or "normalized/image{}.png".format(
        image_order
    )
    original_file_name = original_file_name or "image{}.png".format(
        image_order
    )
    values = {
        "sample_key": sample_key or "sample-{}-{}".format(
            image_order, spatial_order
        ),
        "image_order": image_order,
        "normalized_path": normalized_path,
        "original_file_name": original_file_name,
        "spatial_order": spatial_order,
        "red": float(spatial_order),
        "green": float(spatial_order + 1),
        "blue": float(spatial_order + 2),
        "series_number": series_number,
    }
    values.update(changes)
    return LinearSampleItem(**values)


def direct_error(
        image_order=1,
        error_index=1,
        error_key=None,
        normalized_path=None,
        original_file_name=None):
    return LinearSampleError(
        image_order=image_order,
        original_file_name=(
            original_file_name or "image{}.png".format(image_order)
        ),
        normalized_path=(
            normalized_path or "normalized/image{}.png".format(image_order)
        ),
        reason="sample failed",
        error_type="invalid_sample",
        spatial_order=error_index,
        error_key=error_key or "error-{}-{}".format(image_order, error_index),
    )


def direct_image(
        image_order=1,
        status=LinearImageStatus.PENDING,
        samples=(),
        errors=(),
        failure_reason=None,
        image_key=None,
        selection_order=None,
        normalized_path=None,
        original_file_name=None):
    return LinearImageItem(
        normalized_path=(
            normalized_path or "normalized/image{}.png".format(image_order)
        ),
        original_file_name=(
            original_file_name or "image{}.png".format(image_order)
        ),
        image_order=image_order,
        selection_order=(
            image_order if selection_order is None else selection_order
        ),
        image_key=image_key or "image-key-{}".format(image_order),
        status=status,
        samples=samples,
        errors=errors,
        failure_reason=failure_reason,
    )


def changed_image(image, **changes):
    values = {
        "normalized_path": image.normalized_path,
        "original_file_name": image.original_file_name,
        "image_order": image.image_order,
        "selection_order": image.selection_order,
        "image_key": image.image_key,
        "status": image.status,
        "samples": image.samples,
        "errors": image.errors,
        "failure_reason": image.failure_reason,
    }
    values.update(changes)
    return LinearImageItem(**values)


def regression_kwargs(snapshot):
    return {item.name: getattr(snapshot, item.name) for item in fields(snapshot)}


class NaturalSortTests(unittest.TestCase):
    def test_empty_series(self):
        state = LinearSeriesState.from_paths([])
        self.assertEqual(state.images, ())
        self.assertEqual(state.all_samples, ())
        self.assertEqual(state.phase, LinearSeriesPhase.IDLE)
        self.assertFalse(state.busy)
        self.assertFalse(state.begin())
        with self.assertRaisesRegex(KeyError, "image_order"):
            state.image_for_order(1)

    def test_natural_sort_and_original_filename_extension(self):
        paths = ["image10.PNG", "image2.jpg", "image1.JpG"]
        items = build_image_items(paths)
        self.assertEqual(
            [item.original_file_name for item in items],
            ["image1.JpG", "image2.jpg", "image10.PNG"],
        )
        self.assertEqual([item.image_order for item in items], [1, 2, 3])
        self.assertEqual(paths, ["image10.PNG", "image2.jpg", "image1.JpG"])

    def test_mapping_constructor_sorts_before_validating_global_numbers(self):
        first_sample = direct_sample(1, series_number=1)
        second_sample = direct_sample(2, series_number=2)
        first_image = direct_image(
            1, status=LinearImageStatus.COMPLETED, samples=(first_sample,)
        )
        second_image = direct_image(
            2, status=LinearImageStatus.COMPLETED, samples=(second_sample,)
        )
        state = LinearSeriesState(
            images=(second_image, first_image),
            phase=LinearSeriesPhase.MAPPING,
        )
        self.assertEqual(
            tuple(image.image_order for image in state.images),
            (1, 2),
        )
        self.assertEqual(
            tuple(sample.series_number for sample in state.all_samples),
            (1, 2),
        )

    def test_leading_zero_case_and_path_tie_breaks_are_deterministic(self):
        values = [
            str(Path("z") / "IMAGE2.jpg"),
            str(Path("a") / "image02.jpg"),
            str(Path("a") / "image2.JPG"),
            str(Path("a") / "image10.jpg"),
        ]
        ordered = natural_sort_paths(values)
        self.assertEqual(ordered[0], values[2])
        self.assertEqual(ordered[1], values[0])
        self.assertEqual(ordered[2], values[1])
        self.assertEqual(ordered[3], values[3])

    def test_duplicate_path_is_two_distinct_image_tasks(self):
        selected = [str(Path("folder") / "same.png")] * 2
        state = LinearSeriesState.from_paths(selected)
        self.assertEqual(len(state.images), 2)
        self.assertEqual(
            state.images[0].normalized_path,
            state.images[1].normalized_path,
        )
        self.assertNotEqual(state.images[0].image_key, state.images[1].image_key)
        self.assertEqual([item.selection_order for item in state.images], [1, 2])

    def test_same_name_in_different_directories_stays_distinct(self):
        state = LinearSeriesState.from_paths([
            str(Path("left") / "sample.png"),
            str(Path("right") / "sample.png"),
        ])
        self.assertEqual(
            [item.original_file_name for item in state.images],
            ["sample.png", "sample.png"],
        )
        self.assertNotEqual(
            state.images[0].normalized_path,
            state.images[1].normalized_path,
        )
        self.assertNotEqual(state.images[0].image_key, state.images[1].image_key)


class SeriesAcceptanceTests(unittest.TestCase):
    def test_single_image_single_sample_defaults_included(self):
        state = completed_state(["one.png"], [1])
        sample = state.all_samples[0]
        self.assertEqual(sample.series_number, 1)
        self.assertEqual(sample.spatial_order, 1)
        self.assertTrue(sample.included)
        self.assertEqual(sample.concentration_text, "")
        self.assertEqual(state.images[0].status, LinearImageStatus.COMPLETED)
        self.assertEqual(state.phase, LinearSeriesPhase.MAPPING)

    def test_single_image_multiple_samples_preserves_confirmed_spatial_order(self):
        state = LinearSeriesState.from_paths(["one.png"])
        state.mark_image_processing(1)
        state.accept_image_result(1, [rgb(90), rgb(10), rgb(50)])
        self.assertEqual([sample.red for sample in state.all_samples], [90, 10, 50])
        self.assertEqual(
            [sample.spatial_order for sample in state.all_samples], [1, 2, 3]
        )
        self.assertEqual(
            [sample.series_number for sample in state.all_samples], [1, 2, 3]
        )

    def test_multiple_images_have_one_gap_free_series_numbering(self):
        state = completed_state(["image10.png", "image2.png"], [2, 3])
        self.assertEqual(
            [image.original_file_name for image in state.images],
            ["image2.png", "image10.png"],
        )
        self.assertEqual(
            [sample.series_number for sample in state.all_samples],
            [1, 2, 3, 4, 5],
        )
        self.assertEqual(
            [sample.image_order for sample in state.all_samples],
            [1, 1, 2, 2, 2],
        )

    def test_failed_image_and_invalid_rgb_do_not_consume_numbers(self):
        state = LinearSeriesState.from_paths(["a.png", "b.png", "c.png"])
        state.mark_image_processing(1)
        state.fail_image(1, "decode failed")
        state.mark_image_processing(2)
        state.accept_image_result(2, [rgb(1), rgb(math.nan), rgb(5)])
        state.mark_image_processing(3)
        state.accept_image_result(3, [rgb(9)])

        self.assertEqual(state.images[0].status, LinearImageStatus.FAILED)
        self.assertEqual(state.images[0].samples, ())
        self.assertEqual(
            [sample.spatial_order for sample in state.images[1].samples], [1, 3]
        )
        self.assertEqual(
            [sample.series_number for sample in state.all_samples], [1, 2, 3]
        )
        errors = state.images[0].errors + state.images[1].errors
        self.assertTrue(all(error.series_number is None for error in errors))
        self.assertTrue(all(error.sample_key is None for error in errors))
        set_concentrations(state, ["0", "1", "2"])
        snapshot = state.regression_input()
        self.assertEqual(snapshot.numbers, (1, 2, 3))
        self.assertEqual(
            snapshot.sample_keys,
            tuple(sample.sample_key for sample in state.all_samples),
        )
        self.assertTrue(all(
            error.error_key not in snapshot.sample_keys for error in errors
        ))

    def test_zero_valid_rgb_marks_image_failed_and_series_continues(self):
        state = LinearSeriesState.from_paths(["a.png", "b.png"])
        state.mark_image_processing(1)
        state.accept_image_result(1, [{"red": 1, "green": 2}])
        self.assertEqual(state.images[0].status, LinearImageStatus.FAILED)
        self.assertEqual(state.phase, LinearSeriesPhase.EXTRACTING)
        state.mark_image_processing(2)
        state.accept_image_result(2, [{"RGB": (1, 2, 3)}])
        self.assertEqual(state.all_samples[0].series_number, 1)
        self.assertEqual(state.phase, LinearSeriesPhase.MAPPING)

    def test_all_images_failed_enters_failed_phase_and_keeps_confirmed_reference(self):
        confirmed = {"formula": "previous"}
        state = LinearSeriesState.from_paths(
            ["a.png", "b.png"], last_confirmed_result=confirmed
        )
        state.mark_image_processing(1)
        state.fail_image(1, "decode failed")
        self.assertEqual(state.phase, LinearSeriesPhase.EXTRACTING)
        state.mark_image_processing(2)
        state.accept_image_result(2, [{"red": math.inf, "green": 2, "blue": 3}])
        self.assertEqual(state.phase, LinearSeriesPhase.FAILED)
        self.assertEqual(state.all_samples, ())
        self.assertIs(state.last_confirmed_result, confirmed)

    def test_worker_error_objects_are_copied_with_image_identity(self):
        state = LinearSeriesState.from_paths(["actual.png"])
        state.mark_image_processing(1)
        image = state.accept_image_result(1, [rgb(1)], [{
            "reason": "bad ROI",
            "error_type": "invalid_roi",
            "spatial_order": 4,
        }])
        error = image.errors[0]
        self.assertEqual(error.image_order, 1)
        self.assertEqual(error.original_file_name, "actual.png")
        self.assertEqual(error.normalized_path, image.normalized_path)
        self.assertEqual(error.reason, "bad ROI")
        self.assertEqual(error.error_type, "invalid_roi")
        self.assertEqual(error.spatial_order, 4)

    def test_repeated_path_sample_keys_are_unique_and_stable(self):
        paths = ["same.png", "same.png"]
        first = completed_state(paths, [2, 2])
        second = completed_state(paths, [2, 2])
        first_keys = [sample.sample_key for sample in first.all_samples]
        second_keys = [sample.sample_key for sample in second.all_samples]
        self.assertEqual(first_keys, second_keys)
        self.assertEqual(len(first_keys), len(set(first_keys)))
        self.assertEqual(first_keys, [
            "linear-sample:000001:000001",
            "linear-sample:000001:000002",
            "linear-sample:000002:000001",
            "linear-sample:000002:000002",
        ])
        self.assertNotIn(first.images[0].normalized_path, first_keys[0])

    def test_include_and_concentration_edits_never_renumber(self):
        state = completed_state(["one.png"], [3])
        before = {
            sample.sample_key: sample.series_number for sample in state.all_samples
        }
        middle = state.all_samples[1]
        state.set_included(middle.sample_key, False)
        state.set_concentration(middle.sample_key, " 1e0 ")
        updated = state.sample_for_key(middle.sample_key)
        self.assertTrue(middle.included)
        self.assertEqual(middle.concentration_text, "")
        self.assertFalse(updated.included)
        self.assertEqual(updated.concentration_text, " 1e0 ")
        self.assertEqual(
            {sample.sample_key: sample.series_number for sample in state.all_samples},
            before,
        )

    def test_cancel_discards_draft_and_preserves_confirmed_result(self):
        confirmed = {"formula": "old", "plot": object(), "save": True}
        state = LinearSeriesState.from_paths(
            ["one.png"], last_confirmed_result=confirmed
        )
        state.mark_image_processing(1)
        state.accept_image_result(1, [rgb(1)], [{"reason": "warning"}])
        self.assertTrue(state.images[0].samples)
        self.assertTrue(state.images[0].errors)
        state.cancel()
        self.assertEqual(state.phase, LinearSeriesPhase.CANCELLED)
        self.assertEqual(state.images[0].status, LinearImageStatus.CANCELLED)
        self.assertEqual(state.images[0].samples, ())
        self.assertEqual(state.images[0].errors, ())
        self.assertIsNone(state.images[0].failure_reason)
        self.assertIs(state.last_confirmed_result, confirmed)
        with self.assertRaisesRegex(RuntimeError, "Cancelled"):
            state.accept_image_result(1, [rgb(1)])

    def test_bad_iterables_do_not_partially_mutate_state(self):
        state = LinearSeriesState.from_paths(["one.png"])
        state.mark_image_processing(1)
        before = stored_state_value(state)
        with self.assertRaises(TypeError):
            state.accept_image_result(1, None)
        self.assertEqual(stored_state_value(state), before)
        image = state.images[0]
        self.assertEqual(state.phase, LinearSeriesPhase.EXTRACTING)
        self.assertEqual(image.status, LinearImageStatus.PROCESSING)
        self.assertEqual(image.samples, ())
        self.assertEqual(image.errors, ())

        with self.assertRaises(TypeError):
            state.accept_image_result(1, [rgb(1)], None)
        self.assertEqual(stored_state_value(state), before)
        self.assertEqual(state.phase, LinearSeriesPhase.EXTRACTING)
        self.assertEqual(state.images[0].status, LinearImageStatus.PROCESSING)
        self.assertEqual(state.images[0].samples, ())
        self.assertEqual(state.images[0].errors, ())


class StateInvariantTests(unittest.TestCase):
    def assert_invariant_code(self, code, action):
        with self.assertRaises(LinearSeriesInvariantError) as raised:
            action()
        self.assertEqual(raised.exception.code, code)

    def test_public_collections_and_domain_items_are_immutable(self):
        cuvette_box = [0, 1, 2, 3]
        liquid_box = [4, 5, 6, 7]
        roi_box = [8, 9, 10, 11]
        state = LinearSeriesState.from_paths(["one.png"])
        state.mark_image_processing(1)
        state.accept_image_result(1, [{
            "RGB": [1, 2, 3],
            "cuvette_box": cuvette_box,
            "liquid_box": liquid_box,
            "roi_box": roi_box,
        }], [{"reason": "warning"}])

        image = state.images[0]
        sample = image.samples[0]
        error = image.errors[0]
        for collection in (state.images, image.samples, image.errors):
            with self.subTest(collection=type(collection).__name__):
                self.assertIs(type(collection), tuple)
                with self.assertRaises(AttributeError):
                    collection.append(None)
                with self.assertRaises(AttributeError):
                    collection.clear()

        with self.assertRaises(AttributeError):
            state.images = ()
        for target, attribute, value in (
            (image, "status", LinearImageStatus.FAILED),
            (image, "image_order", 99),
            (image, "samples", ()),
            (sample, "sample_key", "changed"),
            (sample, "red", math.nan),
            (sample, "green", math.nan),
            (sample, "blue", math.nan),
            (sample, "cuvette_box", (9.0, 9.0, 9.0, 9.0)),
            (sample, "included", False),
            (sample, "concentration_text", "9"),
            (error, "reason", "changed"),
        ):
            with self.subTest(attribute=attribute), self.assertRaises(
                    (FrozenInstanceError, AttributeError)):
                setattr(target, attribute, value)

        cuvette_box[0] = 99
        liquid_box.clear()
        roi_box.append(12)
        self.assertEqual(sample.cuvette_box, (0.0, 1.0, 2.0, 3.0))
        self.assertEqual(sample.liquid_box, (4.0, 5.0, 6.0, 7.0))
        self.assertEqual(sample.roi_box, (8.0, 9.0, 10.0, 11.0))

        key = sample.sample_key
        state.set_included(key, False)
        state.set_concentration(key, " 1.00 ")
        updated = state.sample_for_key(key)
        self.assertFalse(updated.included)
        self.assertEqual(updated.concentration_text, " 1.00 ")
        self.assertTrue(sample.included)
        self.assertEqual(sample.concentration_text, "")

    def test_error_number_is_permanently_none(self):
        error = direct_error()
        self.assertIsNone(error.series_number)
        self.assertIsNone(error.number)
        self.assertIsNone(error.no)
        self.assertIsNone(error.sample_key)
        with self.assertRaises((FrozenInstanceError, AttributeError)):
            error.series_number = 77
        self.assertIsNone(error.series_number)

    def test_image_order_entry_points_reject_bool_and_invalid_values(self):
        state = LinearSeriesState.from_paths(["one.png"])
        before = stored_state_value(state)
        for value in (True, False, 0, -1, 1.0, "1", None):
            for method in (state.image_for_order, state.mark_image_processing):
                with self.subTest(value=value, method=method.__name__):
                    self.assert_invariant_code(
                        "invalid_image_order", lambda: method(value)
                    )
                    self.assertEqual(stored_state_value(state), before)
        for method in (state.image_for_order, state.mark_image_processing):
            with self.subTest(method=method.__name__), self.assertRaises(KeyError):
                method(2)
            self.assertEqual(stored_state_value(state), before)

    def test_constructor_rejects_invalid_phase_status_and_combinations(self):
        self.assert_invariant_code(
            "invalid_phase",
            lambda: LinearSeriesState(
                images=(direct_image(),), phase="idle"
            ),
        )
        self.assert_invariant_code(
            "invalid_image_status",
            lambda: direct_image(status="pending"),
        )

        phase_cases = (
            (LinearSeriesPhase.MAPPING, "invalid_mapping_state"),
            (LinearSeriesPhase.READY, "invalid_mapping_state"),
            (LinearSeriesPhase.FAILED, "invalid_failed_state"),
            (LinearSeriesPhase.CANCELLED, "invalid_cancelled_state"),
        )
        for phase, code in phase_cases:
            for status in (
                    LinearImageStatus.PENDING,
                    LinearImageStatus.PROCESSING):
                with self.subTest(phase=phase, status=status):
                    self.assert_invariant_code(
                        code,
                        lambda phase=phase, status=status: LinearSeriesState(
                            images=(direct_image(status=status),), phase=phase
                        ),
                    )

        self.assert_invariant_code(
            "completed_without_samples",
            lambda: LinearSeriesState(
                images=(direct_image(status=LinearImageStatus.COMPLETED),),
                phase=LinearSeriesPhase.MAPPING,
            ),
        )
        sample = direct_sample()
        for status, phase in (
            (LinearImageStatus.PENDING, LinearSeriesPhase.IDLE),
            (LinearImageStatus.PROCESSING, LinearSeriesPhase.EXTRACTING),
        ):
            with self.subTest(status=status):
                self.assert_invariant_code(
                    "nonterminal_image_has_results",
                    lambda status=status, phase=phase: LinearSeriesState(
                        images=(direct_image(status=status, samples=(sample,)),),
                        phase=phase,
                    ),
                )

        failed_error = direct_error()
        failed_image = direct_image(
            status=LinearImageStatus.FAILED,
            errors=(failed_error,),
            failure_reason="failed",
        )
        completed_image = direct_image(
            status=LinearImageStatus.COMPLETED,
            samples=(direct_sample(),),
        )
        terminal_cases = (
            (
                LinearSeriesPhase.MAPPING,
                failed_image,
                "invalid_mapping_state",
            ),
            (
                LinearSeriesPhase.FAILED,
                completed_image,
                "invalid_failed_state",
            ),
            (
                LinearSeriesPhase.CANCELLED,
                failed_image,
                "invalid_cancelled_state",
            ),
            (
                LinearSeriesPhase.IDLE,
                direct_image(status=LinearImageStatus.PROCESSING),
                "invalid_idle_state",
            ),
            (
                LinearSeriesPhase.EXTRACTING,
                completed_image,
                "invalid_extracting_state",
            ),
        )
        for phase, image, code in terminal_cases:
            with self.subTest(phase=phase, terminal=image.status):
                self.assert_invariant_code(
                    code,
                    lambda phase=phase, image=image: LinearSeriesState(
                        images=(image,), phase=phase
                    ),
                )

    def test_value_constructors_reject_bool_orders_and_invalid_sample_status(self):
        for factory in (
            lambda: direct_image(image_order=True),
            lambda: direct_sample(image_order=True),
            lambda: direct_error(image_order=True),
        ):
            with self.subTest(factory=factory):
                self.assert_invariant_code("invalid_image_order", factory)
        self.assert_invariant_code(
            "invalid_sample_status",
            lambda: direct_sample(status="valid"),
        )

    def test_constructor_rejects_two_processing_images(self):
        self.assert_invariant_code(
            "multiple_processing_images",
            lambda: LinearSeriesState(
                images=(
                    direct_image(1, status=LinearImageStatus.PROCESSING),
                    direct_image(2, status=LinearImageStatus.PROCESSING),
                ),
                phase=LinearSeriesPhase.EXTRACTING,
            ),
        )

    def test_constructor_rejects_duplicate_image_sample_and_error_keys(self):
        duplicate_order_images = (
            direct_image(1),
            direct_image(
                1,
                selection_order=2,
                image_key="image-key-other",
                normalized_path="normalized/other.png",
                original_file_name="other.png",
            ),
        )
        duplicate_image_keys = (
            direct_image(1, image_key="same-image-key"),
            direct_image(2, image_key="same-image-key"),
        )
        first_sample = direct_sample(1, series_number=1, sample_key="same-sample")
        second_sample = direct_sample(2, series_number=2, sample_key="same-sample")
        duplicate_sample_images = (
            direct_image(
                1, status=LinearImageStatus.COMPLETED, samples=(first_sample,)
            ),
            direct_image(
                2, status=LinearImageStatus.COMPLETED, samples=(second_sample,)
            ),
        )
        first_error = direct_error(1, error_key="same-error")
        second_error = direct_error(2, error_key="same-error")
        duplicate_error_images = (
            direct_image(
                1,
                status=LinearImageStatus.FAILED,
                errors=(first_error,),
                failure_reason="failed",
            ),
            direct_image(
                2,
                status=LinearImageStatus.FAILED,
                errors=(second_error,),
                failure_reason="failed",
            ),
        )
        cases = (
            ("duplicate_image_order", duplicate_order_images,
             LinearSeriesPhase.IDLE),
            ("duplicate_image_key", duplicate_image_keys,
             LinearSeriesPhase.IDLE),
            ("duplicate_sample_key", duplicate_sample_images,
             LinearSeriesPhase.MAPPING),
            ("duplicate_error_key", duplicate_error_images,
             LinearSeriesPhase.FAILED),
        )
        for code, images, phase in cases:
            with self.subTest(code=code):
                self.assert_invariant_code(
                    code,
                    lambda images=images, phase=phase: LinearSeriesState(
                        images=images, phase=phase
                    ),
                )

    def test_public_reads_and_updates_revalidate_corrupt_internal_graph(self):
        state = LinearSeriesState.from_paths(["one.png", "two.png"])
        first, second = state.images
        duplicate = changed_image(second, image_order=first.image_order)
        corrupt_data = replace(state._data, images=(first, duplicate))
        object.__setattr__(state, "_data", corrupt_data)
        before = stored_state_value(state)

        actions = (
            lambda: state.images,
            lambda: state.phase,
            lambda: state.all_samples,
            lambda: state.busy,
            lambda: state.image_for_order(1),
            lambda: state.mark_image_processing(1),
            state.validate,
        )
        for action in actions:
            with self.subTest(action=action):
                self.assert_invariant_code("duplicate_image_order", action)
                self.assertEqual(stored_state_value(state), before)

    def test_error_spatial_order_bool_is_rejected_on_access(self):
        state = LinearSeriesState.from_paths(["invalid.png"])
        state.mark_image_processing(1)
        state.accept_image_result(1, [{"RGB": [math.nan, 2, 3]}])
        error = state.images[0].errors[0]
        object.__setattr__(error, "spatial_order", True)
        before = stored_state_value(state)
        self.assert_invariant_code("invalid_spatial_order", state.validate)
        self.assertEqual(stored_state_value(state), before)

    def test_stored_rgb_and_geometry_require_exact_finite_float_values(self):
        rgb_state = completed_state(["rgb.png"], [1])
        rgb_sample = rgb_state.images[0].samples[0]
        object.__setattr__(rgb_sample, "red", Decimal("1"))
        rgb_before = stored_state_value(rgb_state)
        self.assert_invariant_code("invalid_rgb", rgb_state.validate)
        self.assertEqual(stored_state_value(rgb_state), rgb_before)

        box_state = LinearSeriesState.from_paths(["box.png"])
        box_state.mark_image_processing(1)
        box_state.accept_image_result(1, [{
            "RGB": [1, 2, 3],
            "roi_box": [0, 1, 2, 3],
        }])
        box_sample = box_state.images[0].samples[0]
        object.__setattr__(box_sample, "roi_box", (0, 1, 2, 3))
        box_before = stored_state_value(box_state)
        self.assert_invariant_code("invalid_geometry", box_state.validate)
        self.assertEqual(stored_state_value(box_state), box_before)


class StrictRgbContractTests(unittest.TestCase):
    def test_only_ordered_finite_numeric_rgb_triplets_are_accepted(self):
        for rgb_value in ([1, Decimal("2.5"), 3], (4, 5.5, Decimal("6"))):
            with self.subTest(rgb_value=rgb_value):
                state = LinearSeriesState.from_paths(["valid.png"])
                state.mark_image_processing(1)
                image = state.accept_image_result(1, [{"RGB": rgb_value}])
                self.assertEqual(image.status, LinearImageStatus.COMPLETED)
                self.assertEqual(len(image.samples), 1)
                sample = image.samples[0]
                self.assertTrue(all(
                    type(value) is float and math.isfinite(value)
                    for value in (sample.red, sample.green, sample.blue)
                ))
                self.assertEqual(sample.series_number, 1)
                self.assertEqual(image.errors, ())

    def test_invalid_rgb_types_shapes_and_channels_never_receive_numbers(self):
        bad_sources = (
            ("str", {"RGB": "123"}),
            ("bytes", {"RGB": b"123"}),
            ("bytearray", {"RGB": bytearray(b"123")}),
            ("bool", {"RGB": True}),
            ("set", {"RGB": {1, 2, 3}}),
            ("frozenset", {"RGB": frozenset((1, 2, 3))}),
            ("dict", {"RGB": {"red": 1, "green": 2, "blue": 3}}),
            ("too_short", {"RGB": [1, 2]}),
            ("too_long", {"RGB": [1, 2, 3, 4]}),
            ("channel_string", {"red": "1", "green": 2, "blue": 3}),
            ("channel_bool", {"red": True, "green": 2, "blue": 3}),
            ("nan", {"RGB": [math.nan, 2, 3]}),
            ("positive_infinity", {"RGB": [math.inf, 2, 3]}),
            ("negative_infinity", {"RGB": [-math.inf, 2, 3]}),
        )
        for label, source in bad_sources:
            with self.subTest(label=label):
                state = LinearSeriesState.from_paths(["invalid.png"])
                state.mark_image_processing(1)
                image = state.accept_image_result(1, [source])
                self.assertEqual(image.status, LinearImageStatus.FAILED)
                self.assertEqual(image.samples, ())
                self.assertEqual(len(image.errors), 1)
                error = image.errors[0]
                self.assertEqual(error.error_type, "invalid_rgb")
                self.assertIsNone(error.series_number)
                self.assertIsNone(error.sample_key)
                self.assertTrue(error.error_key)
                self.assertEqual(state.all_samples, ())
                self.assertEqual(state.phase, LinearSeriesPhase.FAILED)

    def test_rgb_fields_cannot_be_changed_to_non_finite_values(self):
        state = completed_state(["one.png"], [1])
        sample = state.all_samples[0]
        for attribute, value in (
            ("red", math.nan),
            ("green", math.inf),
            ("blue", -math.inf),
        ):
            with self.subTest(attribute=attribute), self.assertRaises(
                    FrozenInstanceError):
                setattr(sample, attribute, value)
        self.assertEqual(
            (sample.red, sample.green, sample.blue),
            (10.0, 11.0, 12.0),
        )


class AtomicStateUpdateTests(unittest.TestCase):
    def test_accept_result_identity_exception_is_atomic_and_retryable(self):
        state = LinearSeriesState.from_paths(["one.png"])
        state.mark_image_processing(1)
        before = stored_state_value(state)
        mismatched = rgb(4, image_order=99)
        with self.assertRaises(LinearSeriesInvariantError) as raised:
            state.accept_image_result(1, [rgb(1), mismatched])
        self.assertEqual(raised.exception.code, "mismatched_image_order")
        self.assertEqual(stored_state_value(state), before)
        self.assertEqual(state.images[0].status, LinearImageStatus.PROCESSING)
        self.assertEqual(state.images[0].samples, ())
        self.assertEqual(state.images[0].errors, ())

        image = state.accept_image_result(1, [rgb(1), rgb(4)])
        self.assertEqual(image.status, LinearImageStatus.COMPLETED)
        self.assertEqual(
            [sample.red for sample in image.samples],
            [1.0, 4.0],
        )

    def test_spoofed_secondary_identity_alias_is_not_ignored(self):
        state = LinearSeriesState.from_paths(["one.png"])
        state.mark_image_processing(1)
        image = state.images[0]
        source = rgb(
            1,
            normalized_path=image.normalized_path,
            path=image.normalized_path + ".spoof",
            original_file_name=image.original_file_name,
            image_name="spoof.png",
        )
        before = stored_state_value(state)
        with self.assertRaises(LinearSeriesInvariantError) as raised:
            state.accept_image_result(1, [source])
        self.assertIn(
            raised.exception.code,
            ("mismatched_normalized_path", "mismatched_original_file_name"),
        )
        self.assertEqual(stored_state_value(state), before)

    def test_numbering_failure_preserves_entire_graph_and_old_numbers(self):
        state = completed_state(["a.png", "b.png"], [2, 2])
        first_image, second_image = state.images
        first_key = first_image.samples[0].sample_key
        corrupted = replace(
            second_image.samples[0],
            sample_key=first_key,
            series_number=77,
        )
        corrupt_second = changed_image(
            second_image,
            samples=(corrupted, second_image.samples[1]),
        )
        object.__setattr__(
            state,
            "_data",
            replace(state._data, images=(first_image, corrupt_second)),
        )
        before = stored_state_value(state)

        with self.assertRaises(LinearSeriesInvariantError) as raised:
            state.assign_series_numbers()
        self.assertEqual(raised.exception.code, "duplicate_sample_key")
        self.assertEqual(stored_state_value(state), before)

    def test_numbering_and_regression_defensively_reject_corrupt_rgb(self):
        state = completed_state(["one.png"], [2])
        set_concentrations(state, ["0", "1"])
        image = state.images[0]
        corrupted = image.samples[0]
        object.__setattr__(corrupted, "red", math.nan)
        before = stored_state_value(state)

        for action in (state.assign_series_numbers, state.build_regression_input):
            with self.subTest(action=action.__name__):
                self.assert_invariant_code(action)
                self.assertEqual(stored_state_value(state), before)

    def assert_invariant_code(self, action):
        with self.assertRaises(LinearSeriesInvariantError) as raised:
            action()
        self.assertEqual(raised.exception.code, "invalid_rgb")


class NumberingHelperContractTests(unittest.TestCase):
    def test_helper_numbers_a_linear_series_state_and_returns_sample_count(self):
        state = LinearSeriesState.from_paths([
            "image1.png", "image2.png", "image3.png",
        ])
        state.mark_image_processing(1)
        state.accept_image_result(1, [rgb(10), rgb(20)])
        state.mark_image_processing(2)
        state.fail_image(2, "decode failed")
        state.mark_image_processing(3)
        state.accept_image_result(3, [rgb(30)])

        self.assertEqual(assign_series_numbers(state), 3)
        self.assertEqual(
            [sample.series_number for sample in state.all_samples],
            [1, 2, 3],
        )
        self.assertTrue(all(
            error.series_number is None
            for image in state.images
            for error in image.errors
        ))

    def test_helper_rejects_list_and_tuple_without_side_effects(self):
        state = completed_state(["one.png"], [2])
        image = state.images[0]
        data_id = id(object.__getattribute__(state, "_data"))
        state_before = stored_state_value(state)
        image_before = immutable_value(image)
        sample_ids = tuple(id(sample) for sample in image.samples)
        numbers_before = tuple(
            sample.series_number for sample in image.samples
        )

        for value in ([image], (image,), [], ()):
            with self.subTest(container_type=type(value), empty=not value):
                container_before = immutable_value(value)
                item_ids = tuple(id(item) for item in value)
                with self.assertRaisesRegex(
                        TypeError,
                        "^assign_series_numbers requires LinearSeriesState$",
                ):
                    assign_series_numbers(value)
                self.assertEqual(immutable_value(value), container_before)
                self.assertEqual(tuple(id(item) for item in value), item_ids)
                self.assertEqual(immutable_value(image), image_before)
                self.assertEqual(
                    tuple(id(sample) for sample in image.samples), sample_ids
                )
                self.assertEqual(
                    tuple(sample.series_number for sample in image.samples),
                    numbers_before,
                )
                self.assertEqual(
                    id(object.__getattribute__(state, "_data")), data_id
                )
                self.assertEqual(stored_state_value(state), state_before)


class DecimalParsingTests(unittest.TestCase):
    def test_integer_decimal_scientific_and_zero_are_accepted(self):
        self.assertEqual(parse_concentration("0"), Decimal("0"))
        self.assertEqual(parse_concentration("+0"), Decimal("0"))
        self.assertEqual(parse_concentration("-0"), Decimal("0"))
        self.assertEqual(normalize_concentration("+0"), "0")
        self.assertEqual(normalize_concentration("-0"), "0")
        self.assertEqual(parse_concentration(" 12 "), Decimal("12"))
        self.assertEqual(parse_concentration(".125"), Decimal("0.125"))
        self.assertEqual(parse_concentration("1.25e+2"), Decimal("125"))
        self.assertEqual(parse_concentration(7), Decimal("7"))
        self.assertEqual(parse_concentration(Decimal("2.500")), Decimal("2.5"))

    def test_equivalent_decimal_spellings_normalize_identically(self):
        values = ["1", "1.0", "1.00", "1e0"]
        self.assertEqual({parse_concentration(value) for value in values}, {Decimal(1)})
        self.assertEqual({normalize_concentration(value) for value in values}, {"1"})

    def test_empty_and_non_numeric_are_rejected(self):
        for value in ("", "   ", "abc", "1,2", "1_000", None, True, False):
            with self.subTest(value=value), self.assertRaises(
                    RegressionValidationError) as raised:
                parse_concentration(value)
            self.assertIn(
                raised.exception.code,
                ("empty_concentration", "invalid_concentration"),
            )

    def test_nan_and_infinity_are_rejected(self):
        values = ["NaN", "Infinity", "-Infinity", Decimal("NaN"), Decimal("Inf")]
        for value in values:
            with self.subTest(value=value), self.assertRaises(
                    RegressionValidationError):
                parse_concentration(value)

    def test_negative_is_rejected_but_signed_zero_is_allowed(self):
        with self.assertRaisesRegex(RegressionValidationError, "greater than"):
            parse_concentration("-0.0001")
        self.assertEqual(parse_concentration("-0"), Decimal(0))

    def test_huge_python_integer_and_text_conversion_errors_are_wrapped(self):
        with self.assertRaises(RegressionValidationError) as raised:
            parse_concentration(10 ** 5000)
        self.assertEqual(raised.exception.code, "invalid_concentration")

        class BadText:
            def __str__(self):
                raise OverflowError("conversion probe")

        with self.assertRaises(RegressionValidationError) as raised:
            parse_concentration(BadText())
        self.assertEqual(raised.exception.code, "invalid_concentration")


class RegressionValidationTests(unittest.TestCase):
    def test_valid_snapshot_has_exact_and_float_concentrations_and_rgb(self):
        state = completed_state(["one.png"], [3])
        set_concentrations(state, ["0", "2.5e-1", "1.50"])
        snapshot = state.build_regression_input()
        self.assertEqual(snapshot.numbers, (1, 2, 3))
        self.assertEqual(snapshot.concentration_texts, ("0", "2.5e-1", "1.50"))
        self.assertEqual(
            snapshot.normalized_concentration_texts,
            ("0", "0.25", "1.5"),
        )
        self.assertEqual(
            snapshot.decimal_concentrations,
            (Decimal("0"), Decimal("0.25"), Decimal("1.5")),
        )
        self.assertEqual(snapshot.concentrations, (0.0, 0.25, 1.5))
        self.assertEqual(snapshot.red_values, (10.0, 11.0, 12.0))
        self.assertEqual(snapshot.green_values, (11.0, 12.0, 13.0))
        self.assertEqual(snapshot.blue_values, (12.0, 13.0, 14.0))
        self.assertEqual(snapshot.included_count, 3)

    def test_only_included_rows_are_validated(self):
        state = completed_state(["one.png"], [4])
        set_concentrations(state, ["0", "1", "", "1.00"])
        state.set_included(state.all_samples[2].sample_key, False)
        state.set_included(state.all_samples[3].sample_key, False)
        snapshot = state.regression_input()
        self.assertEqual(snapshot.numbers, (1, 2))
        self.assertEqual(snapshot.concentrations, (0.0, 1.0))

    def test_included_invalid_values_identify_the_row(self):
        for bad in ("", "word", "NaN", "Infinity", "-1"):
            with self.subTest(bad=bad):
                state = completed_state(["bad.png"], [2])
                set_concentrations(state, [bad, "2"])
                first = state.all_samples[0]
                with self.assertRaises(RegressionValidationError) as raised:
                    state.regression_input()
                message = str(raised.exception)
                self.assertIn(first.sample_key, message)
                self.assertIn("Image Name=bad.png", message)
                self.assertIn("No.1", message)

    def test_equivalent_duplicate_reports_every_conflicting_sample(self):
        state = completed_state(["a.png", "b.png"], [2, 2])
        set_concentrations(state, ["1", "1.0", "1.00", "1e0"])
        with self.assertRaises(RegressionValidationError) as raised:
            state.regression_input()
        error = raised.exception
        self.assertEqual(error.code, "duplicate_concentration")
        self.assertEqual(
            set(error.sample_keys),
            {sample.sample_key for sample in state.all_samples},
        )
        for sample in state.all_samples:
            self.assertIn(sample.sample_key, str(error))
            self.assertIn("No.{}".format(sample.series_number), str(error))
        self.assertIn("Image Name=a.png", str(error))
        self.assertIn("Image Name=b.png", str(error))
        self.assertIn("at least 2 different", str(error))

    def test_multiple_duplicate_groups_have_complete_conflict_information(self):
        state = completed_state(["groups.png"], [6])
        set_concentrations(state, ["1", "1.0", "2", "2e0", "3", "4"])
        with self.assertRaises(RegressionValidationError) as raised:
            state.regression_input()
        error = raised.exception
        expected_conflicts = state.all_samples[:4]
        self.assertEqual(len(error.conflict_groups), 2)
        self.assertEqual(set(error.sample_keys), {
            sample.sample_key for sample in expected_conflicts
        })
        for sample in expected_conflicts:
            self.assertIn(sample.sample_key, str(error))
            self.assertIn("No.{}".format(sample.series_number), str(error))

    def test_fewer_than_two_included_samples_is_rejected(self):
        state = completed_state(["one.png"], [2])
        set_concentrations(state, ["0", "1"])
        included = state.all_samples[0]
        state.set_included(state.all_samples[1].sample_key, False)
        with self.assertRaises(RegressionValidationError) as raised:
            state.regression_input()
        error = raised.exception
        self.assertEqual(error.code, "insufficient_included_samples")
        self.assertEqual(error.sample_keys, (included.sample_key,))
        self.assertEqual(len(error.sample_details), 1)
        detail = error.sample_details[0]
        self.assertEqual(detail.sample_key, included.sample_key)
        self.assertEqual(detail.original_file_name, "one.png")
        self.assertEqual(detail.series_number, included.series_number)
        self.assertIn(included.sample_key, str(error))
        self.assertIn("Image Name=one.png", str(error))
        self.assertIn("No.1", str(error))

    def test_fewer_than_two_distinct_concentrations_is_rejected(self):
        state = completed_state(["one.png"], [2])
        set_concentrations(state, ["5", "5.0"])
        with self.assertRaises(RegressionValidationError) as raised:
            state.regression_input()
        self.assertIn("at least 2 different", str(raised.exception))

    def test_float_overflow_is_rejected_before_scipy_boundary(self):
        state = completed_state(["one.png"], [2])
        set_concentrations(state, ["1e9999", "1"])
        with self.assertRaises(RegressionValidationError) as raised:
            state.regression_input()
        self.assertEqual(raised.exception.code, "float_overflow")
        self.assertIn(state.all_samples[0].sample_key, str(raised.exception))
        self.assertEqual(
            raised.exception.sample_details[0].original_file_name,
            "one.png",
        )
        self.assertIn("No.1", str(raised.exception))

    def test_nonzero_decimal_float_underflow_is_rejected(self):
        state = completed_state(["one.png"], [2])
        set_concentrations(state, ["1e-9999", "1"])
        with self.assertRaises(RegressionValidationError) as raised:
            state.regression_input()
        self.assertEqual(raised.exception.code, "float_underflow")
        self.assertEqual(
            raised.exception.sample_keys,
            (state.all_samples[0].sample_key,),
        )
        self.assertIn("Image Name=one.png", str(raised.exception))
        self.assertIn("No.1", str(raised.exception))

    def test_distinct_decimals_that_fold_to_one_float_are_rejected(self):
        first = "1.00000000000000001"
        second = "1.00000000000000002"
        self.assertNotEqual(Decimal(first), Decimal(second))
        self.assertEqual(float(first), float(second))
        state = completed_state(["one.png"], [2])
        set_concentrations(state, [first, second])
        with self.assertRaises(RegressionValidationError) as raised:
            state.regression_input()
        error = raised.exception
        self.assertEqual(error.code, "float_collision")
        for sample in state.all_samples:
            self.assertIn(sample.sample_key, str(error))

    def test_snapshot_is_frozen_and_detached_from_later_ui_edits(self):
        state = completed_state(["one.png"], [2])
        set_concentrations(state, ["0", "1"])
        snapshot = state.regression_input()
        first = state.all_samples[0]
        state.set_concentration(first.sample_key, "99")
        state.set_included(first.sample_key, False)
        with self.assertRaises(FrozenInstanceError):
            first.red = 999
        self.assertEqual(snapshot.concentrations, (0.0, 1.0))
        self.assertEqual(snapshot.red_values, (10.0, 11.0))
        self.assertEqual(snapshot.green_values, (11.0, 12.0))
        self.assertEqual(snapshot.blue_values, (12.0, 13.0))
        self.assertEqual(snapshot.numbers, (1, 2))
        with self.assertRaises(FrozenInstanceError):
            snapshot.numbers = (9, 10)

    def test_snapshot_rows_preserve_identity_order_text_decimal_float_and_rgb(self):
        state = completed_state(["image10.png", "image2.png"], [2, 2])
        raw_texts = (" 0 ", "2.500e-1", "1.00", "2e0")
        set_concentrations(state, raw_texts)
        samples = state.all_samples
        snapshot = state.regression_input()

        actual_rows = tuple(zip(
            snapshot.sample_keys,
            snapshot.image_names,
            snapshot.normalized_paths,
            snapshot.numbers,
            snapshot.concentration_texts,
            snapshot.normalized_concentration_texts,
            snapshot.decimal_concentrations,
            snapshot.concentrations,
            snapshot.red_values,
            snapshot.green_values,
            snapshot.blue_values,
        ))
        expected_rows = tuple(
            (
                sample.sample_key,
                sample.original_file_name,
                sample.normalized_path,
                sample.series_number,
                raw_text,
                normalize_concentration(raw_text),
                parse_concentration(raw_text),
                float(parse_concentration(raw_text)),
                sample.red,
                sample.green,
                sample.blue,
            )
            for sample, raw_text in zip(samples, raw_texts)
        )
        self.assertEqual(actual_rows, expected_rows)
        self.assertEqual(snapshot.included_count, len(samples))

    def test_regression_input_rejects_outer_and_nested_mutable_values(self):
        state = completed_state(["one.png"], [2])
        set_concentrations(state, ["0", "1"])
        snapshot = state.regression_input()
        cases = (
            ("outer_list", "numbers", list(snapshot.numbers)),
            (
                "nested_list",
                "sample_keys",
                (["mutable"], snapshot.sample_keys[1]),
            ),
            (
                "nested_dict",
                "image_names",
                ({"name": "one.png"}, snapshot.image_names[1]),
            ),
            (
                "nested_set",
                "normalized_paths",
                ({snapshot.normalized_paths[0]}, snapshot.normalized_paths[1]),
            ),
        )
        for label, field_name, bad_value in cases:
            with self.subTest(label=label):
                values = regression_kwargs(snapshot)
                values[field_name] = bad_value
                with self.assertRaises(ValueError):
                    RegressionInput(**values)

        for item in fields(snapshot):
            self.assertIs(type(getattr(snapshot, item.name)), tuple)
        regression_fields = {item.name for item in fields(snapshot)}
        self.assertTrue({
            "cuvette_box", "liquid_box", "roi_box"
        }.isdisjoint(regression_fields))

    def test_regression_input_rejects_misaligned_rows_and_bad_scalars(self):
        state = completed_state(["one.png"], [2])
        set_concentrations(state, ["0", "1"])
        snapshot = state.regression_input()
        cases = (
            ("short_blue", "blue_values", snapshot.blue_values[:-1]),
            ("bool_number", "numbers", (True, 2)),
            ("float_decimal", "decimal_concentrations", (0.0, Decimal("1"))),
            ("nan_red", "red_values", (math.nan, snapshot.red_values[1])),
            ("int_green", "green_values", (1, snapshot.green_values[1])),
        )
        for label, field_name, bad_value in cases:
            with self.subTest(label=label):
                values = regression_kwargs(snapshot)
                values[field_name] = bad_value
                with self.assertRaises(ValueError):
                    RegressionInput(**values)

    def test_regression_input_requires_canonical_normalized_decimal_text(self):
        state = completed_state(["one.png"], [2])
        set_concentrations(state, ["0", "1"])
        snapshot = state.regression_input()
        cases = (
            (
                "trailing_zero",
                (Decimal("0"), Decimal("1.00")),
                ("0", "1.00"),
                ("0", "1.00"),
            ),
            (
                "signed_zero",
                (Decimal("-0"), Decimal("1")),
                ("-0", "1"),
                ("-0", "1"),
            ),
        )
        for label, decimals, raw_texts, normalized_texts in cases:
            with self.subTest(label=label):
                values = regression_kwargs(snapshot)
                values["decimal_concentrations"] = decimals
                values["concentration_texts"] = raw_texts
                values["normalized_concentration_texts"] = normalized_texts
                values["concentrations"] = tuple(float(item) for item in decimals)
                with self.assertRaises(ValueError):
                    RegressionInput(**values)

    def test_snapshot_is_detached_from_every_draft_value(self):
        state = LinearSeriesState.from_paths(["one.png"])
        state.mark_image_processing(1)
        state.accept_image_result(1, [
            {
                "RGB": [1, 2, 3],
                "cuvette_box": [0, 1, 2, 3],
                "liquid_box": [4, 5, 6, 7],
                "roi_box": [8, 9, 10, 11],
            },
            {
                "RGB": [4, 5, 6],
                "cuvette_box": [1, 2, 3, 4],
                "liquid_box": [5, 6, 7, 8],
                "roi_box": [9, 10, 11, 12],
            },
        ])
        set_concentrations(state, ["1.00", "2.500e-1"])
        snapshot = state.regression_input()
        before = immutable_value(snapshot)

        first_key = state.all_samples[0].sample_key
        state.set_included(first_key, False)
        state.set_concentration(first_key, "99")
        image = state.images[0]
        changed_samples = tuple(
            replace(
                sample,
                normalized_path="changed/path.png",
                original_file_name="changed.png",
                red=sample.red + 100,
                green=sample.green + 100,
                blue=sample.blue + 100,
                cuvette_box=(20.0, 21.0, 22.0, 23.0),
                liquid_box=(24.0, 25.0, 26.0, 27.0),
                roi_box=(28.0, 29.0, 30.0, 31.0),
            )
            for sample in image.samples
        )
        replacement = changed_image(
            image,
            normalized_path="changed/path.png",
            original_file_name="changed.png",
            samples=changed_samples,
        )
        candidate = LinearSeriesState(
            images=(replacement,), phase=LinearSeriesPhase.MAPPING
        )
        state._adopt_validated(candidate)

        self.assertEqual(immutable_value(snapshot), before)
        self.assertEqual(snapshot.concentration_texts, ("1.00", "2.500e-1"))
        self.assertEqual(snapshot.normalized_concentration_texts, ("1", "0.25"))
        self.assertEqual(snapshot.red_values, (1.0, 4.0))
        self.assertEqual(snapshot.green_values, (2.0, 5.0))
        self.assertEqual(snapshot.blue_values, (3.0, 6.0))
        self.assertEqual(snapshot.image_names, ("one.png", "one.png"))


if __name__ == "__main__":
    unittest.main()
