import math
import unittest
from dataclasses import FrozenInstanceError

from batch_state import (
    BatchState,
    ConcentrationStatus,
    ImageItem,
    ImageStatus,
    NumberingMode,
    RegressionSet,
    SampleResult,
)
from linear_plot_projection import (
    CalibrationPoint,
    CalibrationSample,
    build_linear_plot_projection,
)


def formula(slope, intercept, r_squared):
    return {
        "slope": slope,
        "intercept": intercept,
        "r": math.sqrt(r_squared),
        "R2": r_squared,
        "p": 0.01,
        "std_err": 0.1,
    }


def regression_set(source="projection-test"):
    return RegressionSet.from_formulas(
        {
            "R": formula(2.0, 10.0, 0.91),
            "G": formula(3.0, 20.0, 0.92),
            "B": formula(4.0, 30.0, 0.93),
        },
        source_id=source,
        valid_ranges={
            "R": (0.0, 10.0),
            "G": (1.0, 11.0),
            "B": (2.0, 12.0),
        },
    )


def calibration_samples():
    return [
        {
            "Con.": 1.0,
            "Red": 12.0,
            "Green": 23.0,
            "Blue": 34.0,
            "included": True,
        },
        {
            "Con.": 2.0,
            "Red": 14.0,
            "Green": 26.0,
            "Blue": 38.0,
            "included": False,
        },
        {
            "Con.": 3.0,
            "Red": 16.0,
            "Green": 29.0,
            "Blue": 42.0,
            "included": True,
        },
    ]


def sample_result(
    image_order=1,
    source_file="sample.jpg",
    no_in_image=7,
    batch_no=19,
    red=101.0,
    green=102.0,
    blue=103.0,
    con_r=1.1,
    con_g=2.2,
    con_b=3.3,
    status_r=ConcentrationStatus.IN_RANGE,
    status_g=ConcentrationStatus.BELOW_RANGE,
    status_b=ConcentrationStatus.ABOVE_RANGE,
):
    return SampleResult(
        image_order=image_order,
        source_file=source_file,
        cuvette_box=(0.0, 0.0, 10.0, 10.0),
        liquid_box=(1.0, 1.0, 9.0, 9.0),
        roi_box=(2.0, 2.0, 8.0, 8.0),
        red=red,
        green=green,
        blue=blue,
        con_r=con_r,
        con_g=con_g,
        con_b=con_b,
        status_r=status_r,
        status_g=status_g,
        status_b=status_b,
        no_in_image=no_in_image,
        batch_no=batch_no,
    )


def image_item(
    order=1,
    filename="sample.jpg",
    path="C:/samples/sample.jpg",
    samples=None,
    status=ImageStatus.COMPLETED,
):
    return ImageItem(
        path=path,
        original_filename=filename,
        image_order=order,
        status=status,
        samples=list(samples or []),
    )


def batch_state(regression, images, numbering=NumberingMode.PER_IMAGE):
    return BatchState(
        images=list(images),
        numbering_mode=numbering,
        regression_set=regression,
    )


def projection(
    regression,
    *,
    mode="RGB",
    samples=None,
    calibration_revision=None,
    batch=None,
    selected=1,
):
    return build_linear_plot_projection(
        active_regression_set=regression,
        calibration_samples=(
            calibration_samples() if samples is None else samples
        ),
        calibration_revision=(
            regression.revision
            if calibration_revision is None
            else calibration_revision
        ),
        batch_state=batch,
        plot_mode=mode,
        selected_image_order=selected,
    )


class LinearPlotProjectionTests(unittest.TestCase):
    def test_modes_have_exact_fixed_curve_order(self):
        regression = regression_set()
        expected = {
            "R": ("R",),
            "G": ("G",),
            "B": ("B",),
            "RGB": ("R", "G", "B"),
        }
        for mode, channels in expected.items():
            with self.subTest(mode=mode):
                result = projection(regression, mode=mode, batch=None)
                self.assertEqual(tuple(curve.channel for curve in result.curves), channels)

    def test_invalid_plot_modes_are_rejected(self):
        regression = regression_set()
        for mode in ("", "RG", "rgb", "Red", None, 1):
            with self.subTest(mode=mode):
                with self.assertRaises(ValueError):
                    projection(regression, mode=mode, batch=None)

    def test_curve_formula_r_squared_range_and_status_come_from_active_set(self):
        regression = regression_set()
        samples = calibration_samples()
        for value in samples:
            value["formulas"] = {
                "slope": -999.0,
                "intercept": -999.0,
                "R2": -999.0,
            }
            value["range_min"] = -999.0
            value["range_max"] = -998.0
        result = projection(regression, samples=samples, batch=None)
        for curve, channel in zip(result.curves, ("R", "G", "B")):
            model = regression.model(channel)
            self.assertEqual(curve.slope, model.slope)
            self.assertEqual(curve.intercept, model.intercept)
            self.assertEqual(curve.r_squared, model.r_squared)
            self.assertEqual(curve.range_min, model.valid_min)
            self.assertEqual(curve.range_max, model.valid_max)
            self.assertIs(curve.unavailable_status, model.unavailable_status)

    def test_only_included_calibration_samples_become_channel_points(self):
        regression = regression_set()
        result = projection(regression, batch=None)
        expected = {
            "R": ((1.0, 12.0), (3.0, 16.0)),
            "G": ((1.0, 23.0), (3.0, 29.0)),
            "B": ((1.0, 34.0), (3.0, 42.0)),
        }
        for curve in result.curves:
            self.assertEqual(
                tuple((point.concentration, point.intensity)
                      for point in curve.calibration_points),
                expected[curve.channel],
            )

    def test_calibration_revision_mismatch_rejects_only_standard_points(self):
        active = regression_set("active")
        stale = regression_set("stale")
        result = projection(
            active,
            calibration_revision=stale.revision,
            batch=None,
        )
        self.assertTrue(result.calibration_points_stale)
        self.assertEqual(result.calibration_revision, stale.revision)
        self.assertEqual(tuple(curve.channel for curve in result.curves), ("R", "G", "B"))
        self.assertTrue(all(curve.calibration_points == () for curve in result.curves))
        self.assertEqual(result.curves[0].slope, active.red.slope)

    def test_rgb_points_use_exact_channel_coordinates_and_shared_identity(self):
        regression = regression_set()
        sample = sample_result()
        batch = batch_state(regression, [image_item(samples=[sample])])
        result = projection(regression, batch=batch)
        self.assertEqual(len(result.detection_points), 3)
        expected = {
            "R": (1.1, 101.0),
            "G": (2.2, 102.0),
            "B": (3.3, 103.0),
        }
        identities = set()
        for point in result.detection_points:
            self.assertEqual(
                (point.concentration, point.intensity), expected[point.channel]
            )
            identities.add((point.image_order, point.no_in_image, point.batch_no))
        self.assertEqual(identities, {(1, 7, 19)})

    def test_single_channel_mode_never_projects_other_channels(self):
        regression = regression_set()
        batch = batch_state(
            regression,
            [image_item(samples=[sample_result()])],
        )
        for mode in ("R", "G", "B"):
            with self.subTest(mode=mode):
                result = projection(regression, mode=mode, batch=batch)
                self.assertEqual(
                    tuple(point.channel for point in result.detection_points),
                    (mode,),
                )

    def test_plottable_statuses_are_preserved_without_reclassification(self):
        regression = regression_set()
        statuses = (
            ConcentrationStatus.IN_RANGE,
            ConcentrationStatus.BELOW_RANGE,
            ConcentrationStatus.ABOVE_RANGE,
            ConcentrationStatus.RANGE_UNAVAILABLE,
        )
        images = []
        for order, status in enumerate(statuses, start=1):
            sample = sample_result(
                image_order=order,
                source_file="{}.jpg".format(order),
                status_r=status,
            )
            images.append(image_item(
                order=order,
                filename=sample.source_file,
                path="C:/samples/{}".format(sample.source_file),
                samples=[sample],
            ))
        batch = batch_state(
            regression, images, numbering=NumberingMode.CONTINUOUS_BATCH
        )
        result = projection(regression, mode="R", batch=batch)
        self.assertEqual(
            tuple(point.status for point in result.detection_points), statuses
        )

    def test_unplottable_channel_does_not_suppress_other_channels(self):
        regression = regression_set()
        sample = sample_result(
            con_r=None,
            status_r=ConcentrationStatus.INVALID_SLOPE,
            con_g=None,
            status_g=ConcentrationStatus.CALCULATION_FAILED,
            con_b=3.3,
            status_b=ConcentrationStatus.RANGE_UNAVAILABLE,
        )
        batch = batch_state(regression, [image_item(samples=[sample])])
        result = projection(regression, batch=batch)
        self.assertEqual(
            tuple(point.channel for point in result.detection_points), ("B",)
        )
        self.assertIs(
            result.detection_points[0].status,
            ConcentrationStatus.RANGE_UNAVAILABLE,
        )

    def test_none_nan_and_infinite_values_do_not_generate_points(self):
        regression = regression_set()
        sample = sample_result()
        sample.con_r = None
        sample.con_g = float("nan")
        sample.blue = float("inf")
        batch = batch_state(regression, [image_item(samples=[sample])])
        result = projection(regression, batch=batch)
        self.assertEqual(result.detection_points, ())

    def test_numbering_mode_and_stored_numbers_are_copied_not_reassigned(self):
        regression = regression_set()
        sample = sample_result(no_in_image=41, batch_no=907)
        batch = batch_state(
            regression,
            [image_item(samples=[sample])],
            numbering=NumberingMode.CONTINUOUS_BATCH,
        )
        result = projection(regression, mode="R", batch=batch)
        point = result.detection_points[0]
        self.assertIs(result.numbering_mode, NumberingMode.CONTINUOUS_BATCH)
        self.assertEqual((point.no_in_image, point.batch_no), (41, 907))
        self.assertEqual((sample.no_in_image, sample.batch_no), (41, 907))

    def test_same_filename_and_repeated_path_are_distinguished_by_image_order(self):
        regression = regression_set()
        repeated_path = "C:/samples/repeated/sample.jpg"
        images = []
        for order in (1, 2):
            sample = sample_result(
                image_order=order,
                source_file="sample.jpg",
                no_in_image=1,
                batch_no=order,
            )
            images.append(image_item(
                order=order,
                filename="sample.jpg",
                path=repeated_path,
                samples=[sample],
            ))
        result = projection(
            regression,
            mode="R",
            batch=batch_state(
                regression, images, numbering=NumberingMode.CONTINUOUS_BATCH
            ),
            selected=2,
        )
        self.assertEqual(
            tuple(point.image_order for point in result.detection_points), (1, 2)
        )
        self.assertEqual(
            tuple(point.is_current_image for point in result.detection_points),
            (False, True),
        )

    def test_missing_selected_order_does_not_fall_back_to_current_index(self):
        regression = regression_set()
        images = [
            image_item(
                order=1,
                samples=[sample_result(image_order=1)],
            ),
            image_item(
                order=2,
                filename="two.jpg",
                path="C:/samples/two.jpg",
                samples=[sample_result(image_order=2, source_file="two.jpg")],
            ),
        ]
        batch = batch_state(regression, images)
        batch.current_image_index = 1
        result = projection(regression, mode="R", batch=batch, selected=99)
        self.assertFalse(any(
            point.is_current_image for point in result.detection_points
        ))

    def test_failed_and_empty_completed_images_do_not_create_points(self):
        regression = regression_set()
        failed_sample = sample_result(image_order=1)
        images = [
            image_item(
                order=1,
                samples=[failed_sample],
                status=ImageStatus.FAILED,
            ),
            image_item(
                order=2,
                filename="empty.jpg",
                path="C:/samples/empty.jpg",
                samples=[],
            ),
        ]
        result = projection(
            regression,
            batch=batch_state(regression, images),
        )
        self.assertEqual(result.detection_points, ())

    def test_projection_and_nested_values_are_frozen_tuples(self):
        regression = regression_set()
        result = projection(regression, batch=None)
        self.assertIsInstance(result.curves, tuple)
        self.assertTrue(all(
            isinstance(curve.calibration_points, tuple) for curve in result.curves
        ))
        with self.assertRaises(FrozenInstanceError):
            result.plot_mode = "R"
        with self.assertRaises(FrozenInstanceError):
            result.curves[0].slope = 999.0
        with self.assertRaises(FrozenInstanceError):
            result.curves[0].calibration_points[0].intensity = 999.0

    def test_calibration_source_list_and_dict_mutation_cannot_change_projection(self):
        regression = regression_set()
        source = calibration_samples()
        result = projection(regression, samples=source, batch=None)
        expected = result.curves[0].calibration_points
        source[0]["Con."] = 999.0
        source[0]["Red"] = 999.0
        source[0]["included"] = False
        source.clear()
        self.assertEqual(result.curves[0].calibration_points, expected)
        self.assertEqual(
            expected,
            (CalibrationPoint(1.0, 12.0), CalibrationPoint(3.0, 16.0)),
        )

    def test_calibration_sample_is_frozen_and_normalizes_finite_values(self):
        sample = CalibrationSample(1, 2, 3, 4, True)
        self.assertEqual(
            (sample.concentration, sample.red, sample.green, sample.blue),
            (1.0, 2.0, 3.0, 4.0),
        )
        with self.assertRaises(FrozenInstanceError):
            sample.red = 100.0


if __name__ == "__main__":
    unittest.main()
