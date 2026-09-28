import ast
import copy
import unittest
from pathlib import Path

from batch_detection_controller import BatchDetectionController
from batch_state import (
    BatchState,
    ConcentrationStatus,
    DetectionScope,
    ImageItem,
    ImageStatus,
    NumberingMode,
    RegressionSet,
    SampleResult,
)
from linear_plot_projection import build_linear_plot_projection
from linear_plot_renderer import render_linear_plot
from tests.test_phase1c2_plot_integration import (
    PROTECTED_METHOD_HASHES,
    method_hash,
)


NORMAL_COLORS = {"R": "red", "G": "green", "B": "blue"}
OUT_OF_RANGE_COLORS = {"R": "#8B0000", "G": "#006400", "B": "#00008B"}


class RecordingAxes:
    def __init__(self):
        self.scatter_calls = []
        self.plot_calls = []
        self.annotation_calls = []
        self.legend_calls = []
        self.title = None
        self.title_kwargs = None
        self.text_calls = []
        self.transAxes = object()

    def clear(self):
        pass

    def scatter(self, x, y, **kwargs):
        call = {"x": tuple(x), "y": tuple(y), **kwargs}
        self.scatter_calls.append(call)
        return call

    def plot(self, x, y, **kwargs):
        call = {"x": tuple(x), "y": tuple(y), **kwargs}
        self.plot_calls.append(call)
        return (call,)

    def annotate(self, text, xy, **kwargs):
        call = {"text": str(text), "xy": tuple(xy), **kwargs}
        self.annotation_calls.append(call)
        return call

    def text(self, x, y, text, **kwargs):
        call = {"x": x, "y": y, "text": str(text), **kwargs}
        self.text_calls.append(call)
        return call

    def set_title(self, value, **kwargs):
        self.title = value
        self.title_kwargs = kwargs

    def set_xlabel(self, value, **kwargs):
        self.xlabel = value

    def set_ylabel(self, value, **kwargs):
        self.ylabel = value

    def tick_params(self, *args, **kwargs):
        self.tick_call = (args, kwargs)

    def grid(self, *args, **kwargs):
        self.grid_call = (args, kwargs)

    def legend(self, *args, **kwargs):
        call = (args, kwargs)
        self.legend_calls.append(call)
        return call


def formula(slope, intercept, r_squared):
    return {
        "slope": slope,
        "intercept": intercept,
        "r": r_squared ** 0.5,
        "R2": r_squared,
        "p": 0.01,
        "std_err": 0.1,
    }


def regression_set(source="repair4"):
    return RegressionSet.from_formulas(
        {
            "R": formula(2.0, 10.0, 0.91),
            "G": formula(3.0, 20.0, 0.92),
            "B": formula(4.0, 30.0, 0.93),
        },
        source_id=source,
        valid_ranges={"R": (1.0, 5.0), "G": (1.0, 5.0), "B": (1.0, 5.0)},
    )


def calibration_samples():
    return (
        {"Con.": 1.0, "Red": 12.0, "Green": 23.0, "Blue": 34.0, "included": True},
        {"Con.": 5.0, "Red": 20.0, "Green": 35.0, "Blue": 50.0, "included": True},
    )


def sample_result(
    *,
    image_order=1,
    source_file="one.png",
    no_in_image=1,
    batch_no=1,
    concentrations=(2.0, 2.5, 3.0),
    intensities=(14.0, 27.5, 42.0),
    statuses=(
        ConcentrationStatus.IN_RANGE,
        ConcentrationStatus.IN_RANGE,
        ConcentrationStatus.IN_RANGE,
    ),
    run_token=7,
):
    value = SampleResult(
        image_order=image_order,
        source_file=source_file,
        cuvette_box=(0.0, 0.0, 10.0, 20.0),
        liquid_box=(1.0, 2.0, 9.0, 18.0),
        roi_box=(2.0, 4.0, 8.0, 16.0),
        red=intensities[0],
        green=intensities[1],
        blue=intensities[2],
        con_r=concentrations[0],
        con_g=concentrations[1],
        con_b=concentrations[2],
        status_r=statuses[0],
        status_g=statuses[1],
        status_b=statuses[2],
        no_in_image=no_in_image,
        batch_no=batch_no,
    )
    value.detection_run_token = run_token
    return value


def image_item(order, samples, status=ImageStatus.COMPLETED):
    filename = "{}.png".format(order)
    return ImageItem(
        path="C:/samples/{}".format(filename),
        original_filename=filename,
        image_order=order,
        status=status,
        samples=list(samples),
    )


def projection(
    *,
    mode="RGB",
    images=(),
    numbering=NumberingMode.PER_IMAGE,
    selected=1,
    run_token=7,
):
    regression = regression_set()
    state = BatchState(
        images=list(images),
        numbering_mode=numbering,
        regression_set=regression,
    )
    state.detection_run_token = run_token
    return build_linear_plot_projection(
        active_regression_set=regression,
        calibration_samples=calibration_samples(),
        calibration_revision=regression.revision,
        batch_state=state,
        plot_mode=mode,
        selected_image_order=selected,
    )


def render(**kwargs):
    axes = RecordingAxes()
    value = projection(**kwargs)
    render_linear_plot(axes, value)
    return axes, value


def public_labels(axes):
    return [
        call["label"]
        for call in axes.scatter_calls + axes.plot_calls
        if call.get("label") not in (None, "_nolegend_")
    ]


def accepted_payload(task, count=1):
    return {
        "run_token": task.run_token,
        "job_token": task.job_token,
        "regression_revision": task.regression_revision,
        "image_order": task.image_order,
        "source_path": task.path,
        "source_file": task.source_file,
        "sample_results": [
            {
                "image_order": task.image_order,
                "source_file": task.source_file,
                "cuvette_box": (0.0, 0.0, 10.0, 20.0),
                "liquid_box": (1.0, 2.0, 9.0, 18.0),
                "roi_box": (2.0, 4.0, 8.0, 16.0),
                "red": 14.0,
                "green": 27.5,
                "blue": 42.0,
                "con_r": 2.0,
                "con_g": 2.5,
                "con_b": 3.0,
                "status_r": ConcentrationStatus.IN_RANGE.value,
                "status_g": ConcentrationStatus.IN_RANGE.value,
                "status_b": ConcentrationStatus.IN_RANGE.value,
                "no_in_image": index,
                "batch_no": task.batch_start_no + index - 1,
                "status": "valid",
                "warnings": [],
            }
            for index in range(1, count + 1)
        ],
        "sample_errors": [],
    }


class Repair4DetectionPlotSemanticsTests(unittest.TestCase):
    def test_01_rgb_without_detection_creates_no_legend(self):
        axes, _ = render(images=())
        self.assertEqual(axes.legend_calls, [])
        self.assertEqual(public_labels(axes), [])

    def test_02_rgb_detected_summary_uses_one_gray_diamond(self):
        axes, _ = render(images=(image_item(1, (sample_result(),)),))
        handles = [call for call in axes.scatter_calls if call.get("label") == "Detected (n=1)"]
        self.assertEqual(len(handles), 1)
        self.assertEqual((handles[0]["color"], handles[0]["marker"]), ("#666666", "D"))
        self.assertEqual(axes.legend_calls[0][1]["ncol"], 1)

    def test_03_rgb_out_of_range_summary_uses_one_gray_x_and_unique_sample_count(self):
        sample = sample_result(statuses=(
            ConcentrationStatus.BELOW_RANGE,
            ConcentrationStatus.ABOVE_RANGE,
            ConcentrationStatus.ABOVE_RANGE,
        ))
        axes, _ = render(images=(image_item(1, (sample,)),))
        handles = [call for call in axes.scatter_calls if call.get("label") == "Out of range (n=1)"]
        self.assertEqual(len(handles), 1)
        self.assertEqual((handles[0]["color"], handles[0]["marker"]), ("#666666", "x"))
        self.assertIn("Detected (n=1)", public_labels(axes))
        self.assertEqual(axes.legend_calls[0][1]["ncol"], 2)

    def test_04_rgb_legend_contains_no_channel_standard_or_equation_entries(self):
        axes, _ = render(images=(image_item(1, (sample_result(),)),))
        self.assertEqual(public_labels(axes), ["Detected (n=1)"])

    def test_05_in_range_points_and_numbers_use_channel_colored_diamonds(self):
        axes, _ = render(images=(image_item(1, (sample_result(),)),))
        actual = [
            call for call in axes.scatter_calls
            if call["x"] and call.get("marker") != "o"
        ]
        self.assertEqual(
            [(call["color"], call["marker"]) for call in actual],
            [(NORMAL_COLORS[channel], "D") for channel in ("R", "G", "B")],
        )
        self.assertEqual(len(axes.annotation_calls), 1)
        self.assertEqual(axes.annotation_calls[0]["color"], "red")

    def _assert_out_of_range_style(self, channel, color):
        statuses = [ConcentrationStatus.IN_RANGE] * 3
        statuses[("R", "G", "B").index(channel)] = ConcentrationStatus.ABOVE_RANGE
        axes, _ = render(
            mode=channel,
            images=(image_item(1, (sample_result(statuses=tuple(statuses)),)),),
        )
        actual = [call for call in axes.scatter_calls if call["x"] and call["marker"] != "o"]
        self.assertEqual(len(actual), 1)
        self.assertEqual((actual[0]["color"], actual[0]["marker"]), (color, "x"))
        self.assertEqual(len(axes.annotation_calls), 1)
        self.assertEqual(axes.annotation_calls[0]["color"], color)

    def test_06_r_out_of_range_point_and_number_are_dark_red_x(self):
        self._assert_out_of_range_style("R", "#8B0000")

    def test_07_g_out_of_range_point_and_number_are_dark_green_x(self):
        self._assert_out_of_range_style("G", "#006400")

    def test_08_b_out_of_range_point_and_number_are_dark_blue_x(self):
        self._assert_out_of_range_style("B", "#00008B")

    def test_09_fit_is_solid_in_range_and_dashed_only_for_visible_extrapolation(self):
        below = sample_result(
            no_in_image=1,
            batch_no=1,
            concentrations=(0.25, 2.0, 2.0),
            statuses=(ConcentrationStatus.BELOW_RANGE,) + (ConcentrationStatus.IN_RANGE,) * 2,
        )
        above = sample_result(
            no_in_image=2,
            batch_no=2,
            concentrations=(8.0, 2.0, 2.0),
            statuses=(ConcentrationStatus.ABOVE_RANGE,) + (ConcentrationStatus.IN_RANGE,) * 2,
        )
        axes, _ = render(mode="R", images=(image_item(1, (below, above)),))
        lines = [call for call in axes.plot_calls if call["color"] == "#E57373"]
        self.assertEqual([(call["x"], call.get("linestyle", "-")) for call in lines], [
            ((1.0, 5.0), "-"),
            ((0.25, 1.0), "--"),
            ((5.0, 8.0), "--"),
        ])

    def test_10_range_unavailable_keeps_numeric_channel_point_without_false_warning(self):
        sample = sample_result(statuses=(
            ConcentrationStatus.RANGE_UNAVAILABLE,
            ConcentrationStatus.IN_RANGE,
            ConcentrationStatus.IN_RANGE,
        ))
        axes, _ = render(mode="R", images=(image_item(1, (sample,)),))
        detected = [call for call in axes.scatter_calls if call["x"] and call["marker"] != "o"]
        self.assertEqual([(call["color"], call["marker"]) for call in detected], [("red", "D")])
        self.assertNotIn("Out of range (n=1)", public_labels(axes))
        self.assertFalse(any(call.get("linestyle") == "--" for call in axes.plot_calls))

    def test_11_rgb_draws_all_channel_points_but_annotates_each_sample_once(self):
        sample = sample_result(statuses=(
            ConcentrationStatus.BELOW_RANGE,
            ConcentrationStatus.IN_RANGE,
            ConcentrationStatus.ABOVE_RANGE,
        ))
        axes, _ = render(images=(image_item(1, (sample,)),))
        actual = [call for call in axes.scatter_calls if call["x"] and call["marker"] != "o"]
        self.assertEqual(sum(len(call["x"]) for call in actual), 3)
        self.assertEqual(len(axes.annotation_calls), 1)
        self.assertEqual(axes.annotation_calls[0]["color"], "#8B0000")

    def test_12_rgb_curves_standards_and_detection_points_are_retained(self):
        axes, _ = render(images=(image_item(1, (sample_result(),)),))
        standards = [call for call in axes.scatter_calls if call.get("marker") == "o"]
        detected = [call for call in axes.scatter_calls if call["x"] and call.get("marker") != "o"]
        solid_curves = [call for call in axes.plot_calls if call.get("linestyle", "-") == "-"]
        self.assertEqual((len(standards), len(solid_curves)), (3, 3))
        self.assertEqual(sum(len(call["x"]) for call in detected), 3)

    def test_13_continuous_numbering_projects_all_current_run_images(self):
        images = (
            image_item(1, (sample_result(image_order=1, source_file="1.png", batch_no=1),)),
            image_item(2, (sample_result(image_order=2, source_file="2.png", batch_no=2),)),
        )
        value = projection(images=images, numbering=NumberingMode.CONTINUOUS_BATCH, selected=2)
        self.assertEqual({point.image_order for point in value.detection_points}, {1, 2})
        self.assertEqual(len(value.detection_points), 6)

    def test_14_current_image_is_opaque_and_other_images_use_point_three_alpha(self):
        images = (
            image_item(1, (sample_result(image_order=1, source_file="1.png", batch_no=1),)),
            image_item(2, (sample_result(image_order=2, source_file="2.png", batch_no=2),)),
        )
        value = projection(images=images, numbering=NumberingMode.CONTINUOUS_BATCH, selected=2)
        by_image = {order: {point.alpha for point in value.detection_points if point.image_order == order}
                    for order in (1, 2)}
        self.assertEqual(by_image, {1: {0.30}, 2: {1.0}})

    def test_15_selection_changes_emphasis_without_recalculation(self):
        images = (
            image_item(1, (sample_result(image_order=1, source_file="1.png", batch_no=1),)),
            image_item(2, (sample_result(image_order=2, source_file="2.png", batch_no=2),)),
        )
        before_samples = copy.deepcopy(images)
        first = projection(images=images, numbering=NumberingMode.CONTINUOUS_BATCH, selected=1)
        second = projection(images=images, numbering=NumberingMode.CONTINUOUS_BATCH, selected=2)
        coordinates = lambda value: tuple(
            (point.image_order, point.channel, point.concentration, point.intensity)
            for point in value.detection_points
        )
        self.assertEqual(coordinates(first), coordinates(second))
        self.assertEqual(tuple(point.alpha for point in first.detection_points[:3]), (1.0,) * 3)
        self.assertEqual(tuple(point.alpha for point in second.detection_points[:3]), (0.30,) * 3)
        self.assertEqual(images, before_samples)

    def test_16_per_image_numbering_projects_only_selected_image(self):
        images = (
            image_item(1, (sample_result(image_order=1, source_file="1.png", no_in_image=1),)),
            image_item(2, (sample_result(image_order=2, source_file="2.png", no_in_image=1),)),
        )
        value = projection(images=images, numbering=NumberingMode.PER_IMAGE, selected=2)
        self.assertEqual({point.image_order for point in value.detection_points}, {2})

    def test_17_new_run_hides_prior_run_points_before_callbacks(self):
        old = sample_result(run_token=6)
        value = projection(images=(image_item(1, (old,)),), run_token=7)
        self.assertEqual(value.detection_points, ())

    def test_18_stale_duplicate_and_failed_callbacks_cannot_restore_prior_points(self):
        regression = regression_set("controller")
        state = BatchState.from_paths(("C:/samples/one.png",))
        state.detection_scope = DetectionScope.ALL_IMPORTED_IMAGES
        state.numbering_mode = NumberingMode.CONTINUOUS_BATCH
        controller = BatchDetectionController(state=state, clock_ns=lambda: 100)
        first = controller.begin(regression)
        first_payload = accepted_payload(first)
        self.assertTrue(controller.accept_payload(first_payload))
        self.assertIsNotNone(controller.finish_if_done())
        second = controller.begin(regression)
        self.assertEqual(controller.state.images[0].samples, [])
        self.assertFalse(controller.accept_payload(first_payload))
        failure = {
            "run_token": second.run_token,
            "job_token": second.job_token,
            "regression_revision": second.regression_revision,
            "image_order": second.image_order,
            "source_path": second.path,
            "source_file": second.source_file,
            "message": "failed",
        }
        self.assertTrue(controller.accept_failure(failure))
        self.assertFalse(controller.accept_failure(failure))
        self.assertEqual(controller.state.images[0].samples, [])

    def test_19_projection_reads_only_authoritative_sample_results(self):
        authoritative = sample_result(concentrations=(2.0, 2.5, 3.0))
        regression = regression_set("authority")
        state = BatchState(
            images=(image_item(1, (authoritative,)),),
            numbering_mode=NumberingMode.CONTINUOUS_BATCH,
            regression_set=regression,
        )
        state.detection_run_token = 7
        state.last_batch_result = {
            "targets": [{"Con.": 9999.0, "Red": -1.0, "Green": -2.0, "Blue": -3.0}]
        }
        value = build_linear_plot_projection(
            active_regression_set=regression,
            calibration_samples=calibration_samples(),
            calibration_revision=regression.revision,
            batch_state=state,
            plot_mode="RGB",
            selected_image_order=1,
        )
        self.assertEqual(
            tuple(point.concentration for point in value.detection_points),
            (2.0, 2.5, 3.0),
        )
        self.assertEqual(tuple(point.intensity for point in value.detection_points), (14.0, 27.5, 42.0))

    def test_20_single_channel_screen_save_excel_and_timing_contracts_remain_unchanged(self):
        axes, _ = render(mode="G", images=(image_item(1, (sample_result(),)),))
        self.assertEqual(public_labels(axes), ["Detected (n=1)"])
        self.assertEqual(axes.text_calls[0]["text"], "y = 3x + 20; R² = 0.920")

        protected = (
            "_build_detection_workbook_bytes",
            "_validate_detection_workbook_bytes",
            "_build_detection_export_bytes",
            "_commit_detection_export",
            "_save_detection_result",
            "_build_linear_workbook_bytes",
            "_validate_linear_workbook_bytes",
            "_build_linear_export_bytes",
            "_commit_linear_export",
            "_save_linear_result",
            "_refresh_detection_duration_display",
            "_refresh_linear_duration_display",
            "_set_shared_time_display",
            "_refresh_shared_time_display",
        )
        self.assertEqual(
            {name: method_hash(name) for name in protected},
            {name: PROTECTED_METHOD_HASHES[name] for name in protected},
        )

        source = Path("detectmain.py").read_text(encoding="utf-8")
        tree = ast.parse(source, filename="detectmain.py")
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "DetectMain")
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                      and node.name == "_build_linear_export_bytes")
        calls = [node.func.attr for node in ast.walk(method)
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]
        self.assertIn("_plot_regression_result", calls)
        self.assertIn("print_png", calls)


if __name__ == "__main__":
    unittest.main()
