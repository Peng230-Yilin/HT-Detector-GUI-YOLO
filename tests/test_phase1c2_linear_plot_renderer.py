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
REGRESSION_COLORS = {"R": "#E57373", "G": "#66BB6A", "B": "#64B5F6"}


class SpyAxes:
    def __init__(self):
        self.events = []
        self.scatter_calls = []
        self.plot_calls = []
        self.title = None
        self.xlabel = None
        self.ylabel = None
        self.legend_calls = 0
        self.title_kwargs = None
        self.xlabel_kwargs = None
        self.ylabel_kwargs = None
        self.tick_params_calls = []
        self.legend_kwargs = None
        self.annotation_calls = []
        self.text_calls = []
        self.transAxes = object()

    def clear(self):
        self.events.append(("clear",))

    def scatter(self, x, y, **kwargs):
        call = {"x": tuple(x), "y": tuple(y), **kwargs}
        self.scatter_calls.append(call)
        self.events.append(("scatter", kwargs.get("color"), kwargs.get("marker")))

    def plot(self, x, y, **kwargs):
        call = {"x": tuple(x), "y": tuple(y), **kwargs}
        self.plot_calls.append(call)
        self.events.append(("plot", kwargs.get("color")))

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
        self.tick_params_calls.append((args, kwargs))

    def grid(self, *args, **kwargs):
        self.grid_call = (args, kwargs)

    def legend(self, *args, **kwargs):
        self.legend_calls += 1
        self.legend_kwargs = kwargs


def curve(channel, slope, intercept, r_squared, unavailable=None):
    offset = {"R": 0.0, "G": 10.0, "B": 20.0}[channel]
    return CalibrationCurveView(
        channel=channel,
        slope=slope,
        intercept=intercept,
        r_squared=r_squared,
        range_min=0.0 if slope is not None else None,
        range_max=2.0 if slope is not None else None,
        calibration_points=(
            CalibrationPoint(0.0, offset + 1.0),
            CalibrationPoint(2.0, offset + 5.0),
        ),
        unavailable_status=unavailable,
    )


def detection(channel, concentration, intensity, status=ConcentrationStatus.IN_RANGE):
    return DetectionPlotPoint(
        image_order=1,
        original_filename="sample.jpg",
        no_in_image=1,
        batch_no=1,
        channel=channel,
        concentration=concentration,
        intensity=intensity,
        status=status,
        is_current_image=False,
    )


def projection(mode="RGB", curves=None, points=None, stale=False):
    defaults = (
        curve("R", 2.0, 1.0, 0.91),
        curve("G", 3.0, 2.0, 0.92),
        curve("B", 4.0, 3.0, 0.93),
    )
    selected_curves = defaults if curves is None else tuple(curves)
    if mode != "RGB" and curves is None:
        selected_curves = tuple(value for value in defaults if value.channel == mode)
    default_points = (
        detection("R", 1.0, 3.0),
        detection("G", 1.5, 6.5),
        detection("B", 2.0, 11.0),
    )
    return LinearPlotProjection(
        regression_revision="active-revision",
        plot_mode=mode,
        curves=selected_curves,
        detection_points=default_points if points is None else tuple(points),
        numbering_mode=NumberingMode.PER_IMAGE,
        selected_image_order=1,
        calibration_revision="active-revision",
        detection_regression_revision=(
            "stale-revision" if stale else "active-revision"
        ),
        calibration_points_stale=False,
        detection_points_stale=stale,
    )


class LinearPlotRendererTests(unittest.TestCase):
    def test_single_channel_modes_draw_only_the_selected_channel(self):
        expected = {
            "R": ("Linear Regression – R", "R Intensity"),
            "G": ("Linear Regression – G", "G Intensity"),
            "B": ("Linear Regression – B", "B Intensity"),
        }
        for mode, (title, ylabel) in expected.items():
            with self.subTest(mode=mode):
                axes = SpyAxes()
                render_linear_plot(axes, projection(mode))
                colors = {call["color"] for call in axes.scatter_calls + axes.plot_calls}
                self.assertEqual(colors, {COLORS[mode], REGRESSION_COLORS[mode]})
                self.assertEqual(axes.title, title)
                self.assertEqual(axes.xlabel, "Concentration")
                self.assertEqual(axes.ylabel, ylabel)

    def test_rgb_uses_fixed_red_green_blue_order(self):
        axes = SpyAxes()
        render_linear_plot(axes, projection())
        self.assertEqual(
            [call["color"] for call in axes.plot_calls],
            ["#E57373", "#66BB6A", "#64B5F6"],
        )
        self.assertEqual(axes.title, "Linear Regression – RGB")
        self.assertEqual(axes.ylabel, "Intensity")

    def test_three_distinct_formulas_do_not_cross_channels(self):
        axes = SpyAxes()
        render_linear_plot(axes, projection())
        by_color = {call["color"]: call for call in axes.plot_calls}
        self.assertEqual(by_color["#E57373"]["y"], (1.0, 5.0))
        self.assertEqual(by_color["#66BB6A"]["y"], (2.0, 8.0))
        self.assertEqual(by_color["#64B5F6"]["y"], (3.0, 11.0))

    def test_calibration_point_coordinates_are_exact(self):
        axes = SpyAxes()
        render_linear_plot(axes, projection("R"))
        used = next(call for call in axes.scatter_calls if call["marker"] == "o")
        self.assertEqual(used["x"], (0.0, 2.0))
        self.assertEqual(used["y"], (1.0, 5.0))

    def test_detection_point_coordinates_are_exact(self):
        axes = SpyAxes()
        render_linear_plot(axes, projection("G"))
        found = next(
            call for call in axes.scatter_calls
            if call["marker"] == "D" and call["x"]
        )
        self.assertEqual(found["x"], (1.5,))
        self.assertEqual(found["y"], (6.5,))

    def test_calibration_and_detection_markers_are_distinct(self):
        axes = SpyAxes()
        render_linear_plot(axes, projection("B"))
        self.assertEqual({call["marker"] for call in axes.scatter_calls}, {"o", "D"})

    def test_rgb_uses_fixed_channel_colors_for_every_artist(self):
        axes = SpyAxes()
        render_linear_plot(axes, projection())
        standard_artists = [
            call for call in axes.scatter_calls
            if call["x"] and call["marker"] == "o"
        ]
        detection_artists = [
            call for call in axes.scatter_calls
            if call["x"] and call["marker"] != "o"
        ]
        self.assertEqual(
            [call["color"] for call in standard_artists],
            [REGRESSION_COLORS[channel] for channel in ("R", "G", "B")],
        )
        for call in detection_artists:
            self.assertIn(call["color"], set(COLORS.values()))
        self.assertEqual(
            [call["color"] for call in axes.plot_calls],
            [REGRESSION_COLORS[channel] for channel in ("R", "G", "B")],
        )
        summary = next(call for call in axes.scatter_calls if not call["x"])
        self.assertEqual(summary["color"], "#666666")

    def test_detection_legend_entry_is_not_repeated_per_point(self):
        axes = SpyAxes()
        points = tuple(
            detection("R", float(index), float(index + 10))
            for index in range(1, 5)
        )
        render_linear_plot(axes, projection("R", points=points))
        labels = [call["label"] for call in axes.scatter_calls]
        self.assertEqual(labels.count("Detected (n=1)"), 1)

    def test_fit_legend_contains_projection_equation_and_r_squared(self):
        axes = SpyAxes()
        render_linear_plot(axes, projection("G"))
        self.assertEqual(axes.plot_calls[0]["label"], "_nolegend_")
        self.assertEqual(axes.text_calls[0]["text"], "y = 3x + 2; R² = 0.920")

    def test_unavailable_channel_does_not_stop_other_channels(self):
        unavailable = CalibrationCurveView(
            channel="R",
            slope=None,
            intercept=None,
            r_squared=None,
            range_min=None,
            range_max=None,
            calibration_points=(CalibrationPoint(1.0, 7.0),),
            unavailable_status=ConcentrationStatus.MISSING_REGRESSION,
        )
        axes = SpyAxes()
        render_linear_plot(
            axes,
            projection(curves=(unavailable, curve("G", 3.0, 2.0, 0.92), curve("B", 4.0, 3.0, 0.93))),
        )
        self.assertEqual([call["color"] for call in axes.plot_calls], ["#66BB6A", "#64B5F6"])
        self.assertTrue(any(call["color"] == "#E57373" for call in axes.scatter_calls))

    def test_no_detection_points_still_draws_the_curve(self):
        axes = SpyAxes()
        render_linear_plot(axes, projection("R", points=()))
        self.assertEqual(len(axes.plot_calls), 1)
        self.assertFalse(any(call["marker"] == "D" for call in axes.scatter_calls))

    def test_stale_projection_never_renders_old_detection_points(self):
        axes = SpyAxes()
        render_linear_plot(axes, projection("R", points=(), stale=True))
        self.assertFalse(any(call["marker"] == "D" for call in axes.scatter_calls))
        self.assertEqual(len(axes.plot_calls), 1)

    def test_range_unavailable_detection_point_remains_renderable(self):
        axes = SpyAxes()
        point = detection(
            "B", 8.5, 37.0, ConcentrationStatus.RANGE_UNAVAILABLE
        )
        render_linear_plot(axes, projection("B", points=(point,)))
        found = next(
            call for call in axes.scatter_calls
            if call["marker"] == "D" and call["x"]
        )
        self.assertEqual((found["x"], found["y"]), ((8.5,), (37.0,)))

    def test_renderer_does_not_modify_projection(self):
        value = projection()
        before = copy.deepcopy(value)
        render_linear_plot(SpyAxes(), value)
        self.assertEqual(value, before)

    def test_renderer_imports_only_projection_and_standard_library(self):
        source = Path("linear_plot_renderer.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".")[0])
        self.assertEqual(imports, {"linear_plot_projection"})
        forbidden = {
            "batch_state", "detectmain", "PySide6", "matplotlib", "openpyxl",
            "pandas", "cv2", "numpy", "time", "subprocess",
        }
        self.assertTrue(imports.isdisjoint(forbidden))

    def test_renderer_never_draws_canvas_or_saves_files(self):
        source = Path("linear_plot_renderer.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        calls = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                calls.add(node.func.attr)
        self.assertTrue(
            calls.isdisjoint({"draw", "draw_idle", "print_png", "savefig", "open", "write_bytes"})
        )
        axes = SpyAxes()
        render_linear_plot(axes, projection("R"))
        self.assertEqual(axes.legend_calls, 1)
        self.assertEqual(axes.legend_kwargs["fontsize"], 9)


if __name__ == "__main__":
    unittest.main()
