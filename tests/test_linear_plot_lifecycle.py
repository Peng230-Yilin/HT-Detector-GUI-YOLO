import ast
import copy
import inspect
import math
import unittest
from unittest.mock import patch

import batch_state as batch_state_module
import linear_plot_projection as projection_module
from batch_state import (
    BatchState,
    ConcentrationStatus,
    ImageItem,
    ImageStatus,
    NumberingMode,
    RegressionChannelModel,
    RegressionSet,
    SampleResult,
)
from linear_plot_projection import build_linear_plot_projection


def formula(slope, intercept, r_squared=0.95):
    return {
        "slope": slope,
        "intercept": intercept,
        "r": math.sqrt(r_squared),
        "R2": r_squared,
        "p": 0.02,
        "std_err": 0.2,
    }


def regression_set(source):
    return RegressionSet.from_formulas(
        {
            "R": formula(2.0, 1.0, 0.91),
            "G": formula(3.0, 2.0, 0.92),
            "B": formula(4.0, 3.0, 0.93),
        },
        source_id=source,
        valid_ranges=(0.0, 20.0),
    )


def calibration_samples():
    return [
        {
            "concentration": 1.0,
            "red": 3.0,
            "green": 5.0,
            "blue": 7.0,
            "included": True,
        },
        {
            "concentration": 2.0,
            "red": 5.0,
            "green": 8.0,
            "blue": 11.0,
            "included": True,
        },
    ]


def sample(image_order, filename, number, batch_number):
    return SampleResult(
        image_order=image_order,
        source_file=filename,
        cuvette_box=(0.0, 0.0, 10.0, 10.0),
        liquid_box=(1.0, 1.0, 9.0, 9.0),
        roi_box=(2.0, 2.0, 8.0, 8.0),
        red=21.0 + image_order,
        green=32.0 + image_order,
        blue=43.0 + image_order,
        con_r=10.0 + image_order,
        con_g=11.0 + image_order,
        con_b=12.0 + image_order,
        status_r=ConcentrationStatus.IN_RANGE,
        status_g=ConcentrationStatus.BELOW_RANGE,
        status_b=ConcentrationStatus.ABOVE_RANGE,
        no_in_image=number,
        batch_no=batch_number,
    )


def image(order, path, filename, samples, status=ImageStatus.COMPLETED):
    return ImageItem(
        path=path,
        original_filename=filename,
        image_order=order,
        status=status,
        samples=list(samples),
    )


def batch(regression, images, numbering=NumberingMode.PER_IMAGE):
    return BatchState(
        images=list(images),
        numbering_mode=numbering,
        regression_set=regression,
    )


def build(
    active,
    *,
    batch_value,
    mode="RGB",
    selected=1,
    calibration_revision=None,
    samples=None,
):
    return build_linear_plot_projection(
        active_regression_set=active,
        calibration_samples=(
            calibration_samples() if samples is None else samples
        ),
        calibration_revision=(
            active.revision
            if calibration_revision is None
            else calibration_revision
        ),
        batch_state=batch_value,
        plot_mode=mode,
        selected_image_order=selected,
    )


class LinearPlotLifecycleTests(unittest.TestCase):
    def test_matching_revision_projects_every_completed_formal_sample(self):
        active = regression_set("active")
        images = [
            image(
                1,
                "C:/batch/a.jpg",
                "a.jpg",
                [sample(1, "a.jpg", 1, 1), sample(1, "a.jpg", 2, 2)],
            ),
            image(
                2,
                "C:/batch/b.jpg",
                "b.jpg",
                [sample(2, "b.jpg", 1, 3)],
            ),
        ]
        result = build(
            active,
            batch_value=batch(
                active, images, numbering=NumberingMode.CONTINUOUS_BATCH
            ),
            selected=2,
        )
        self.assertFalse(result.detection_points_stale)
        self.assertEqual(result.detection_regression_revision, active.revision)
        self.assertEqual(len(result.detection_points), 9)
        self.assertEqual(
            sum(point.is_current_image for point in result.detection_points), 3
        )

    def test_revision_mismatch_suppresses_detection_points_and_marks_stale(self):
        active = regression_set("active")
        stale = regression_set("stale")
        stale_batch = batch(stale, [
            image(
                1,
                "C:/batch/a.jpg",
                "a.jpg",
                [sample(1, "a.jpg", 1, 1)],
            )
        ])
        result = build(active, batch_value=stale_batch)
        self.assertEqual(result.regression_revision, active.revision)
        self.assertEqual(result.detection_regression_revision, stale.revision)
        self.assertTrue(result.detection_points_stale)
        self.assertEqual(result.detection_points, ())
        self.assertEqual(tuple(curve.channel for curve in result.curves), ("R", "G", "B"))

    def test_revision_mismatch_never_recalculates_old_detection_points(self):
        active = regression_set("active")
        stale = regression_set("stale")
        original = sample(1, "a.jpg", 5, 8)
        stale_batch = batch(stale, [
            image(1, "C:/batch/a.jpg", "a.jpg", [original])
        ])
        with patch.object(
            RegressionChannelModel,
            "calculate",
            side_effect=AssertionError("concentration recalculation is forbidden"),
        ):
            result = build(active, batch_value=stale_batch)
        self.assertTrue(result.detection_points_stale)
        self.assertEqual(result.detection_points, ())
        self.assertEqual((original.con_r, original.con_g, original.con_b), (11.0, 12.0, 13.0))

    def test_no_batch_has_no_detection_revision_and_is_not_stale(self):
        active = regression_set("active")
        result = build(active, batch_value=None)
        self.assertIsNone(result.numbering_mode)
        self.assertIsNone(result.detection_regression_revision)
        self.assertFalse(result.detection_points_stale)
        self.assertEqual(result.detection_points, ())

    def test_empty_matching_batch_is_not_stale(self):
        active = regression_set("active")
        result = build(active, batch_value=batch(active, []))
        self.assertFalse(result.detection_points_stale)
        self.assertEqual(result.detection_points, ())

    def test_empty_mismatched_batch_records_the_real_revision_conflict(self):
        active = regression_set("active")
        stale = regression_set("stale")
        result = build(active, batch_value=batch(stale, []))
        self.assertTrue(result.detection_points_stale)
        self.assertEqual(result.detection_regression_revision, stale.revision)
        self.assertEqual(result.detection_points, ())

    def test_calibration_and_detection_revision_gates_are_independent(self):
        active = regression_set("active")
        stale = regression_set("stale")
        current_batch = batch(active, [
            image(
                1,
                "C:/batch/a.jpg",
                "a.jpg",
                [sample(1, "a.jpg", 1, 1)],
            )
        ])
        result = build(
            active,
            batch_value=current_batch,
            calibration_revision=stale.revision,
        )
        self.assertTrue(result.calibration_points_stale)
        self.assertTrue(all(not curve.calibration_points for curve in result.curves))
        self.assertFalse(result.detection_points_stale)
        self.assertEqual(len(result.detection_points), 3)

    def test_legacy_targets_and_display_pollution_are_never_read(self):
        active = regression_set("active")
        stored_sample = sample(1, "a.jpg", 17, 31)
        stored_image = image(
            1,
            "C:/batch/a.jpg",
            "a.jpg",
            [stored_sample],
        )
        state = batch(active, [stored_image])
        state.targets = [{"No.": -100, "Con.": -200.0}]
        state.current_gui_table = [["poison"]]
        state.lcd_value = 999999
        stored_image.targets = [{"No.": -300, "Con.": -400.0}]
        before = build(active, batch_value=state)
        state.targets[0]["No."] = 999999
        state.current_gui_table[0][0] = "changed"
        state.lcd_value = -1
        stored_image.targets[0]["Con."] = 999999.0
        after = build(active, batch_value=state)
        self.assertEqual(before, after)
        self.assertEqual(
            {(point.no_in_image, point.batch_no)
             for point in after.detection_points},
            {(17, 31)},
        )

    def test_projection_construction_does_not_mutate_batch_or_samples(self):
        active = regression_set("active")
        state = batch(active, [
            image(
                1,
                "C:/batch/a.jpg",
                "a.jpg",
                [sample(1, "a.jpg", 9, 15)],
            )
        ], numbering=NumberingMode.CONTINUOUS_BATCH)
        before = copy.deepcopy(state)
        build(active, batch_value=state, selected=77)
        self.assertEqual(state, before)
        self.assertEqual(state.current_image_index, before.current_image_index)

    def test_projection_is_detached_from_later_batch_mutation(self):
        active = regression_set("active")
        stored_sample = sample(1, "a.jpg", 4, 44)
        stored_image = image(
            1,
            "C:/batch/a.jpg",
            "a.jpg",
            [stored_sample],
        )
        state = batch(active, [stored_image])
        result = build(active, batch_value=state)
        expected = copy.deepcopy(result)
        stored_sample.red = 999.0
        stored_sample.con_r = 999.0
        stored_sample.no_in_image = 999
        stored_image.original_filename = "mutated.jpg"
        stored_image.samples.clear()
        state.images.clear()
        state.numbering_mode = NumberingMode.CONTINUOUS_BATCH
        self.assertEqual(result, expected)

    def test_builder_never_calls_formula_or_number_assignment_helpers(self):
        active = regression_set("active")
        state = batch(active, [
            image(
                1,
                "C:/batch/a.jpg",
                "a.jpg",
                [sample(1, "a.jpg", 6, 60)],
            )
        ])
        with patch.object(
            RegressionChannelModel,
            "calculate",
            side_effect=AssertionError("formula calculation is forbidden"),
        ), patch.object(
            batch_state_module,
            "assign_image_numbers",
            side_effect=AssertionError("image numbering is forbidden"),
        ), patch.object(
            batch_state_module,
            "assign_batch_numbers",
            side_effect=AssertionError("batch numbering is forbidden"),
        ):
            result = build(active, batch_value=state, mode="R")
        point = result.detection_points[0]
        self.assertEqual(
            (point.concentration, point.intensity, point.no_in_image, point.batch_no),
            (11.0, 22.0, 6, 60),
        )

    def test_missing_and_invalid_regression_statuses_are_copied_to_curves(self):
        active = RegressionSet.from_formulas(
            {
                "R": None,
                "G": formula(0.0, 2.0),
                "B": {"slope": "invalid"},
            },
            source_id="unavailable",
            valid_ranges=(0.0, 1.0),
        )
        result = build(
            active,
            batch_value=None,
            samples=[],
            calibration_revision=active.revision,
        )
        self.assertEqual(
            tuple(curve.unavailable_status for curve in result.curves),
            (
                ConcentrationStatus.MISSING_REGRESSION,
                ConcentrationStatus.INVALID_SLOPE,
                ConcentrationStatus.INVALID_REGRESSION,
            ),
        )

    def test_module_direct_imports_stay_within_pure_allowlist(self):
        tree = ast.parse(inspect.getsource(projection_module))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        self.assertLessEqual(
            imported,
            {"collections", "dataclasses", "math", "numbers", "typing", "batch_state"},
        )
        forbidden = {
            "PySide6", "PyQt5", "PyQt6", "matplotlib", "cv2", "numpy",
            "openpyxl", "pandas", "detectmain", "detectionwindow",
            "yolo_detection_worker", "time",
        }
        self.assertTrue(imported.isdisjoint(forbidden))

    def test_module_contains_no_io_clock_worker_gui_or_model_calls(self):
        tree = ast.parse(inspect.getsource(projection_module))
        forbidden_names = {
            "open", "perf_counter", "perf_counter_ns", "monotonic",
            "monotonic_ns", "QApplication", "QWidget", "QThread",
            "YoloDetectionWorker", "YOLO", "VideoCapture",
        }
        called_names = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name):
                called_names.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                called_names.add(node.func.attr)
        self.assertTrue(called_names.isdisjoint(forbidden_names))


if __name__ == "__main__":
    unittest.main()
