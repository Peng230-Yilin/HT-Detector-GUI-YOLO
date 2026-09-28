import ast
import copy
import unittest
from pathlib import Path

from batch_state import ConcentrationStatus, NumberingMode
from linear_plot_projection import (
    CalibrationCurveView,
    CalibrationPoint,
    DetectionPlotPoint,
    LinearPlotProjection,
)
from linear_plot_renderer import render_linear_plot


COLORS = {"R": "red", "G": "green", "B": "blue"}


class RecordingAxes:
    def __init__(self):
        self.scatter_calls = []
        self.plot_calls = []
        self.title = None
        self.title_kwargs = None
        self.xlabel = None
        self.xlabel_kwargs = None
        self.ylabel = None
        self.ylabel_kwargs = None
        self.tick_calls = []
        self.legend_calls = []
        self.annotation_calls = []
        self.text_calls = []
        self.transAxes = object()

    def clear(self):
        pass

    def scatter(self, x, y, **kwargs):
        self.scatter_calls.append({"x": tuple(x), "y": tuple(y), **kwargs})

    def plot(self, x, y, **kwargs):
        self.plot_calls.append({"x": tuple(x), "y": tuple(y), **kwargs})

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
        self.xlabel_kwargs = kwargs

    def set_ylabel(self, value, **kwargs):
        self.ylabel = value
        self.ylabel_kwargs = kwargs

    def tick_params(self, *args, **kwargs):
        self.tick_calls.append((args, kwargs))

    def grid(self, *args, **kwargs):
        self.grid_call = (args, kwargs)

    def legend(self, *args, **kwargs):
        self.legend_calls.append((args, kwargs))


def curve(channel, slope, intercept, r_squared):
    offset = {"R": 0.0, "G": 10.0, "B": 20.0}[channel]
    return CalibrationCurveView(
        channel=channel,
        slope=slope,
        intercept=intercept,
        r_squared=r_squared,
        range_min=0.0,
        range_max=2.0,
        calibration_points=(
            CalibrationPoint(0.0, offset + 1.0),
            CalibrationPoint(2.0, offset + 5.0),
        ),
        unavailable_status=None,
    )


def detected(channel, concentration=1.0, intensity=3.0):
    return DetectionPlotPoint(
        image_order=1,
        original_filename="sample.jpg",
        no_in_image=1,
        batch_no=1,
        channel=channel,
        concentration=concentration,
        intensity=intensity,
        status=ConcentrationStatus.IN_RANGE,
        is_current_image=False,
    )


def projection(mode="RGB", points=(), selected_curve=None):
    curves = (
        curve("R", 2.0, 1.0, 0.91),
        curve("G", 3.0, 2.0, 0.92),
        curve("B", 4.0, 3.0, 0.93),
    )
    if selected_curve is not None:
        curves = (selected_curve,)
    elif mode != "RGB":
        curves = tuple(value for value in curves if value.channel == mode)
    return LinearPlotProjection(
        regression_revision="active-revision",
        plot_mode=mode,
        curves=curves,
        detection_points=tuple(points),
        numbering_mode=NumberingMode.PER_IMAGE,
        selected_image_order=1,
        calibration_revision="active-revision",
        detection_regression_revision="active-revision",
        calibration_points_stale=False,
        detection_points_stale=False,
    )


def legend_labels(axes):
    return [
        call["label"]
        for call in axes.scatter_calls + axes.plot_calls
        if call.get("label") and call["label"] != "_nolegend_"
    ]


class Repair3CompactPlotAnnotationTests(unittest.TestCase):
    def test_01_rgb_title_and_y_axis_are_compact(self):
        axes = RecordingAxes()
        render_linear_plot(axes, projection())
        self.assertEqual(axes.title, "Linear Regression – RGB")
        self.assertEqual(axes.ylabel, "Intensity")

    def test_02_rgb_without_detection_has_only_r_g_b_legend_entries(self):
        axes = RecordingAxes()
        render_linear_plot(axes, projection(points=()))
        self.assertEqual(legend_labels(axes), [])
        self.assertEqual(axes.legend_calls, [])

    def test_03_rgb_detection_adds_exactly_one_compact_detected_entry(self):
        points = tuple(detected(channel) for channel in ("R", "G", "B"))
        axes = RecordingAxes()
        render_linear_plot(axes, projection(points=points))
        labels = legend_labels(axes)
        self.assertEqual(labels, ["Detected (n=1)"])
        self.assertEqual(labels.count("Detected (n=1)"), 1)
        self.assertEqual(axes.legend_calls[0][1]["ncol"], 1)

    def test_04_rgb_legend_contains_no_formula_or_r_squared_text(self):
        axes = RecordingAxes()
        render_linear_plot(
            axes,
            projection(points=tuple(detected(channel) for channel in ("R", "G", "B"))),
        )
        text = " ".join(legend_labels(axes))
        for forbidden in ("y =", "R²", "fit", "regression", "Standards"):
            self.assertNotIn(forbidden, text)

    def test_05_rgb_legend_is_above_axes_compact_and_unframed(self):
        axes = RecordingAxes()
        render_linear_plot(
            axes,
            projection(points=tuple(detected(channel) for channel in ("R", "G", "B"))),
        )
        kwargs = axes.legend_calls[0][1]
        self.assertEqual(kwargs["loc"], "upper center")
        self.assertGreater(kwargs["bbox_to_anchor"][1], 1.0)
        self.assertLess(axes.title_kwargs["y"] - kwargs["bbox_to_anchor"][1], 0.2)
        self.assertGreater(axes.title_kwargs["y"], kwargs["bbox_to_anchor"][1])
        self.assertEqual(kwargs["ncol"], 1)
        self.assertFalse(kwargs["frameon"])
        self.assertEqual(kwargs["fontsize"], 9)
        self.assertLessEqual(kwargs["handlelength"], 1.5)
        self.assertLessEqual(kwargs["columnspacing"], 1.0)
        self.assertLessEqual(kwargs["handletextpad"], 0.5)
        self.assertLessEqual(kwargs["borderaxespad"], 0.2)

    def test_06_single_channel_retains_compact_complete_equation(self):
        axes = RecordingAxes()
        render_linear_plot(axes, projection("G", points=(detected("G"),)))
        self.assertEqual(legend_labels(axes), ["Detected (n=1)"])
        self.assertEqual(axes.text_calls[0]["text"], "y = 3x + 2; R² = 0.920")
        self.assertEqual(axes.legend_calls[0][1]["loc"], "best")

    def test_07_single_channel_numeric_display_uses_confirmed_precision(self):
        value = curve("R", -13.3157, 197.6986, 0.9539)
        axes = RecordingAxes()
        render_linear_plot(axes, projection("R", selected_curve=value))
        self.assertEqual(axes.text_calls[0]["text"], "y = -13.32x + 197.7; R² = 0.954")

        compact = curve("R", -0.0, -1.23456e-7, -0.0)
        axes = RecordingAxes()
        render_linear_plot(axes, projection("R", selected_curve=compact))
        label = axes.text_calls[0]["text"]
        self.assertEqual(label, "y = 0x - 1.235e-7; R² = 0.000")
        self.assertNotIn("+ -", label)

    def test_08_confirmed_font_sizes_are_applied_to_all_plot_text(self):
        axes = RecordingAxes()
        render_linear_plot(axes, projection("B", points=(detected("B"),)))
        self.assertEqual(axes.title_kwargs["fontsize"], 13)
        self.assertEqual(axes.xlabel_kwargs["fontsize"], 10)
        self.assertEqual(axes.ylabel_kwargs["fontsize"], 10)
        self.assertEqual(axes.tick_calls, [((), {"axis": "both", "labelsize": 9})])
        self.assertEqual(axes.legend_calls[0][1]["fontsize"], 9)
        self.assertEqual(axes.text_calls[0]["fontsize"], 9)

    def test_09_formatting_changes_no_projection_or_excel_authority(self):
        value = projection("R", points=(detected("R"),))
        before = copy.deepcopy(value)
        render_linear_plot(RecordingAxes(), value)
        self.assertEqual(value, before)
        tree = ast.parse(Path("linear_plot_renderer.py").read_text(encoding="utf-8"))
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
        self.assertTrue(
            (names | attributes).isdisjoint(
                {"openpyxl", "Workbook", "save_settings", "write", "print_png"}
            )
        )

    def test_10_screen_and_save_paths_reuse_the_same_renderer(self):
        axes = RecordingAxes()
        render_linear_plot(axes, projection("R"))
        self.assertEqual(len(axes.legend_calls), 0)

        source = Path("detectmain.py").read_text(encoding="utf-8")
        tree = ast.parse(source, filename="detectmain.py")
        cls = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "DetectMain"
        )
        methods = {
            node.name: node
            for node in cls.body
            if isinstance(node, ast.FunctionDef)
        }
        plot_calls = [
            node for node in ast.walk(methods["_plot_regression_result"])
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "render_linear_plot"
        ]
        save_calls = [
            node.func.attr
            for node in ast.walk(methods["_build_linear_export_bytes"])
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        ]
        self.assertEqual(len(plot_calls), 1)
        self.assertIn("_plot_regression_result", save_calls)
        self.assertIn("print_png", save_calls)
        self.assertNotIn("render_linear_plot", save_calls)


if __name__ == "__main__":
    unittest.main()
