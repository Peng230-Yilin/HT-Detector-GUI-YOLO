import ast
import copy
import hashlib
import unittest
from pathlib import Path

from batch_state import ConcentrationStatus, NumberingMode, RegressionSet
from linear_plot_projection import (
    CalibrationCurveView,
    CalibrationPoint,
    DetectionPlotPoint,
    LinearPlotProjection,
    build_linear_plot_projection,
)
from linear_plot_renderer import render_linear_plot
from tests.test_phase1c2_plot_integration import (
    PROTECTED_METHOD_HASHES,
    method_hash,
)


NORMAL_COLORS = {"R": "red", "G": "green", "B": "blue"}
EDGE_COLORS = {"R": "#8B0000", "G": "#006400", "B": "#00008B"}
REGRESSION_COLORS = {"R": "#E57373", "G": "#66BB6A", "B": "#64B5F6"}
REPAIR6_DETECTED_MARKER_SIZE = 50
PROTECTED_IDENTITIES = {
    "linear_plot_projection.py": (16234, "c7eff50fbe54ecea35bd1cd68f120f3a4e833ca15b5d5079ed6b30c59422717c"),
    "detectmain.py": (161432, "886bf1acf76d49552b596fdf14bd0b72c20db8849639999f6c163c18a852080b"),
    "batch_state.py": (38462, "5c7a3cd2ae194c8953e7064facee043b14352ce9b87b41755e062dff6b3f61ce"),
    "batch_detection_controller.py": (13541, "e01f9468902e9f6334495fdb3597818def13b68525805e0e21e2be4ca67e6c70"),
    "yolo_detection_worker.py": (61855, "8cdf28a7bc70ffc71ea2b841bc0e56ca136ac5edf1dc85db3f459a39a9d4bdf8"),
    "interface_config.py": (14132, "f14de7c23a8be0040387afb3c1abb26317634b70f2bc14edf3ea4010780646d9"),
    "interface_settings_dialog.py": (11970, "3a332a0f4cf1f2a70f57f4ce3e98eab093c93f7454e12d3968db37debad7514a"),
    "ui/detectmain.ui": (15734, "cb533b3586f3c8c3e578a4340bffe11bec94091e2e59542d24e6757679648cf4"),
    "ui/ui_detectmain.py": (13945, "e342f9c59348e64ce6676ffd52c78e85745be5c0d90b95fa08b7d34ca62ecc29"),
}


class FakeFigure:
    def __init__(self):
        self.figsize = (15.0, 15.0)
        self.dpi = 100


class RecordingAxes:
    def __init__(self):
        self.figure = FakeFigure()
        self.transAxes = object()
        self.bounds = (0.14, 0.16, 0.82, 0.64)
        self.scatter_calls = []
        self.plot_calls = []
        self.annotation_calls = []
        self.text_calls = []
        self.legend_calls = []
        self.title = None
        self.title_kwargs = None

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
        self.xlabel_kwargs = kwargs

    def set_ylabel(self, value, **kwargs):
        self.ylabel = value
        self.ylabel_kwargs = kwargs

    def tick_params(self, *args, **kwargs):
        self.tick_call = (args, kwargs)

    def grid(self, *args, **kwargs):
        self.grid_call = (args, kwargs)

    def legend(self, *args, **kwargs):
        call = (args, kwargs)
        self.legend_calls.append(call)
        return call


def curve(channel, slope=None, intercept=None, r_squared=None):
    if slope is None:
        slope = {"R": -67.83, "G": 3.0, "B": 4.0}[channel]
    if intercept is None:
        intercept = {"R": 184.1, "G": 20.0, "B": 30.0}[channel]
    if r_squared is None:
        r_squared = {"R": 0.953, "G": 0.920, "B": 0.930}[channel]
    offset = {"R": 0.0, "G": 10.0, "B": 20.0}[channel]
    return CalibrationCurveView(
        channel=channel,
        slope=slope,
        intercept=intercept,
        r_squared=r_squared,
        range_min=1.0,
        range_max=5.0,
        calibration_points=(
            CalibrationPoint(1.0, offset + 2.0),
            CalibrationPoint(5.0, offset + 8.0),
        ),
        unavailable_status=None,
    )


def point(
    channel,
    status=ConcentrationStatus.IN_RANGE,
    concentration=2.0,
    intensity=14.0,
    no_in_image=1,
    batch_no=1,
    filename="sample.png",
):
    return DetectionPlotPoint(
        image_order=1,
        original_filename=filename,
        no_in_image=no_in_image,
        batch_no=batch_no,
        channel=channel,
        concentration=concentration,
        intensity=intensity,
        status=status,
        is_current_image=True,
        run_token=7,
        alpha=1.0,
    )


def projection(mode="RGB", points=()):
    channels = ("R", "G", "B") if mode == "RGB" else (mode,)
    return LinearPlotProjection(
        regression_revision="repair5",
        plot_mode=mode,
        curves=tuple(curve(channel) for channel in channels),
        detection_points=tuple(points),
        numbering_mode=NumberingMode.PER_IMAGE,
        selected_image_order=1,
        calibration_revision="repair5",
        detection_regression_revision="repair5",
        calibration_points_stale=False,
        detection_points_stale=False,
        detection_run_token=7,
    )


def render(mode="RGB", points=()):
    axes = RecordingAxes()
    value = projection(mode, points)
    render_linear_plot(axes, value)
    return axes, value


def public_labels(axes):
    return [
        call["label"]
        for call in axes.scatter_calls + axes.plot_calls
        if call.get("label") not in (None, "_nolegend_")
    ]


def actual_markers(axes, marker=None):
    return [
        call for call in axes.scatter_calls
        if call["x"] and (marker is None or call.get("marker") == marker)
    ]


def protected_identity_mismatches(overrides=None):
    overrides = {} if overrides is None else dict(overrides)
    mismatches = []
    for name, expected in PROTECTED_IDENTITIES.items():
        data = overrides.get(name, Path(name).read_bytes())
        actual = (len(data), hashlib.sha256(data).hexdigest())
        if actual != expected:
            mismatches.append((name, actual, expected))
    return tuple(mismatches)


class Repair5PlotClarityTests(unittest.TestCase):
    def _assert_normal_edge(self, channel):
        axes, _ = render(channel, (point(channel),))
        marker = actual_markers(axes, "D")[0]
        self.assertEqual(marker["color"], NORMAL_COLORS[channel])
        self.assertEqual(marker["edgecolors"], EDGE_COLORS[channel])
        self.assertEqual(marker["linewidths"], 1.5)

    def test_01_r_normal_diamond_has_dark_red_edge(self):
        self._assert_normal_edge("R")

    def test_02_g_normal_diamond_has_dark_green_edge(self):
        self._assert_normal_edge("G")

    def test_03_b_normal_diamond_has_dark_blue_edge(self):
        self._assert_normal_edge("B")

    def test_04_normal_diamond_marker_size_matches_repair4(self):
        for channel in ("R", "G", "B"):
            with self.subTest(channel=channel):
                axes, _ = render(channel, (point(channel),))
                self.assertEqual(actual_markers(axes, "D")[0]["s"], REPAIR6_DETECTED_MARKER_SIZE)

    def test_05_normal_diamond_is_above_lines_and_standards(self):
        points = (
            point("R"),
            point("R", ConcentrationStatus.BELOW_RANGE, 0.5, 150.0, 2, 2, "below.png"),
        )
        axes, _ = render("R", points)
        diamond = actual_markers(axes, "D")[0]
        standards = actual_markers(axes, "o")[0]
        self.assertGreater(diamond["zorder"], standards["zorder"])
        self.assertGreater(diamond["zorder"], max(call["zorder"] for call in axes.plot_calls))

    def test_06_solid_regression_lines_use_light_channel_colors(self):
        axes, _ = render()
        solids = [call for call in axes.plot_calls if call["linestyle"] == "-"]
        self.assertEqual([call["color"] for call in solids], [REGRESSION_COLORS[c] for c in ("R", "G", "B")])

    def test_07_dashed_extrapolation_uses_same_light_channel_color(self):
        for channel in ("R", "G", "B"):
            with self.subTest(channel=channel):
                axes, _ = render(channel, (
                    point(channel, ConcentrationStatus.BELOW_RANGE, 0.5, 14.0),
                ))
                dashed = [call for call in axes.plot_calls if call["linestyle"] == "--"]
                self.assertEqual([call["color"] for call in dashed], [REGRESSION_COLORS[channel]])

    def test_08_standard_points_remain_saturated_channel_colors(self):
        axes, _ = render()
        standards = actual_markers(axes, "o")
        self.assertEqual([call["color"] for call in standards], [REGRESSION_COLORS[c] for c in ("R", "G", "B")])
        self.assertEqual([call["edgecolors"] for call in standards], [REGRESSION_COLORS[c] for c in ("R", "G", "B")])

    def test_09_single_channel_without_detection_has_subtitle_and_no_legend(self):
        axes, _ = render("R")
        self.assertEqual(axes.title, "Linear Regression – R")
        self.assertEqual(axes.title_kwargs["fontsize"], 13)
        self.assertEqual(len(axes.text_calls), 1)
        subtitle = axes.text_calls[0]
        self.assertEqual(subtitle["text"], "y = -67.83x + 184.1; R² = 0.953")
        self.assertEqual((subtitle["x"], subtitle["fontsize"], subtitle["ha"]), (0.5, 9, "center"))
        self.assertNotIn("bbox", subtitle)
        self.assertEqual(axes.legend_calls, [])

    def test_10_single_channel_normal_only_legend_has_detected(self):
        axes, _ = render("R", (point("R"),))
        self.assertEqual(public_labels(axes), ["Detected (n=1)"])
        handle = next(call for call in axes.scatter_calls if call.get("label") == "Detected (n=1)")
        self.assertEqual((handle["color"], handle["edgecolors"], handle["linewidths"]), ("red", "#8B0000", 1.5))

    def test_11_single_channel_out_of_range_only_legend_has_warning(self):
        axes, _ = render("G", (
            point("G", ConcentrationStatus.ABOVE_RANGE, 7.0, 41.0),
        ))
        self.assertEqual(public_labels(axes), ["Out of range (n=1)"])

    def test_12_single_channel_mixed_legend_has_exactly_two_entries(self):
        axes, _ = render("B", (
            point("B"),
            point("B", ConcentrationStatus.BELOW_RANGE, 0.5, 32.0, 2, 2, "below.png"),
        ))
        self.assertEqual(public_labels(axes), ["Detected (n=1)", "Out of range (n=1)"])
        self.assertEqual(len(axes.legend_calls), 1)

    def test_13_single_channel_legend_excludes_standards_equation_and_r_squared(self):
        axes, _ = render("R", (point("R"),))
        legend_text = " ".join(public_labels(axes))
        for forbidden in ("Standards", "y =", "R²"):
            self.assertNotIn(forbidden, legend_text)
        self.assertEqual(axes.plot_calls[0]["label"], "_nolegend_")

    def test_14_rgb_without_detection_has_no_legend_or_equation(self):
        axes, _ = render()
        self.assertEqual(axes.legend_calls, [])
        self.assertEqual(axes.text_calls, [])
        self.assertNotIn("y =", " ".join(public_labels(axes)))

    def test_15_rgb_normal_detection_has_only_gray_detected_summary(self):
        points = tuple(point(channel) for channel in ("R", "G", "B"))
        axes, _ = render("RGB", points)
        self.assertEqual(public_labels(axes), ["Detected (n=1)"])
        handles = [call for call in axes.scatter_calls if not call["x"]]
        self.assertEqual([(call["color"], call["marker"]) for call in handles], [("#666666", "D")])

    def test_16_rgb_out_of_range_adds_gray_warning_summary(self):
        points = (
            point("R", ConcentrationStatus.ABOVE_RANGE, 7.0, 14.0),
            point("G"),
            point("B"),
        )
        axes, _ = render("RGB", points)
        self.assertEqual(public_labels(axes), ["Detected (n=1)", "Out of range (n=1)"])
        handles = [call for call in axes.scatter_calls if not call["x"]]
        self.assertEqual([(call["color"], call["marker"]) for call in handles], [("#666666", "D"), ("#666666", "x")])

    def test_17_rgb_actual_detection_points_are_never_gray(self):
        points = tuple(point(channel) for channel in ("R", "G", "B"))
        axes, _ = render("RGB", points)
        actual = actual_markers(axes, "D")
        self.assertEqual([call["color"] for call in actual], ["red", "green", "blue"])
        self.assertNotIn("#666666", [call["color"] for call in actual])

    def test_18_rgb_counts_unique_samples_not_channel_points(self):
        points = tuple(point(channel) for channel in ("R", "G", "B")) + tuple(
            point(
                channel,
                ConcentrationStatus.ABOVE_RANGE if channel == "R" else ConcentrationStatus.IN_RANGE,
                7.0 if channel == "R" else 3.0,
                20.0,
                2,
                2,
                "second.png",
            )
            for channel in ("R", "G", "B")
        )
        axes, _ = render("RGB", points)
        self.assertEqual(public_labels(axes), ["Detected (n=2)", "Out of range (n=1)"])

    def test_19_figure_layout_marker_sizes_and_line_widths_are_unchanged(self):
        axes = RecordingAxes()
        before = (axes.figure.figsize, axes.figure.dpi, axes.bounds)
        value = projection("R", (
            point("R"),
            point("R", ConcentrationStatus.BELOW_RANGE, 0.5, 14.0, 2, 2, "below.png"),
        ))
        render_linear_plot(axes, value)
        self.assertEqual((axes.figure.figsize, axes.figure.dpi, axes.bounds), before)
        self.assertEqual({call["linewidth"] for call in axes.plot_calls}, {2.2})
        self.assertEqual(actual_markers(axes, "o")[0]["s"], 36)
        self.assertEqual(actual_markers(axes, "D")[0]["s"], REPAIR6_DETECTED_MARKER_SIZE)
        source = Path("detectmain.py").read_text(encoding="utf-8")
        self.assertIn("plt.subplots(figsize=(15,15), dpi=100)", source)
        self.assertIn("left=0.14, right=0.96, bottom=0.16, top=0.80", source)

    def test_20_screen_and_saved_png_share_one_renderer(self):
        source = Path("detectmain.py").read_text(encoding="utf-8")
        tree = ast.parse(source, filename="detectmain.py")
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "DetectMain")
        methods = {node.name: node for node in cls.body if isinstance(node, ast.FunctionDef)}
        render_calls = [
            node for node in ast.walk(methods["_plot_regression_result"])
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "render_linear_plot"
        ]
        save_attributes = [
            node.func.attr for node in ast.walk(methods["_build_linear_export_bytes"])
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        ]
        self.assertEqual(len(render_calls), 1)
        self.assertIn("_plot_regression_result", save_attributes)
        self.assertIn("print_png", save_attributes)
        self.assertNotIn("render_linear_plot", save_attributes)

    def test_21_formal_projection_and_renderer_do_not_mutate_inputs(self):
        regression = RegressionSet.from_formulas(
            {
                "R": {"slope": 2.0, "intercept": 10.0, "r": 0.95, "R2": 0.91, "p": 0.01, "std_err": 0.1},
                "G": {"slope": 3.0, "intercept": 20.0, "r": 0.96, "R2": 0.92, "p": 0.01, "std_err": 0.1},
                "B": {"slope": 4.0, "intercept": 30.0, "r": 0.97, "R2": 0.93, "p": 0.01, "std_err": 0.1},
            },
            source_id="repair5",
            valid_ranges={"R": (1.0, 5.0), "G": (1.0, 5.0), "B": (1.0, 5.0)},
        )
        samples = [
            {"Con.": 1.0, "Red": 12.0, "Green": 23.0, "Blue": 34.0, "included": True},
            {"Con.": 5.0, "Red": 20.0, "Green": 35.0, "Blue": 50.0, "included": True},
        ]
        before_samples = copy.deepcopy(samples)
        value = build_linear_plot_projection(
            active_regression_set=regression,
            calibration_samples=samples,
            calibration_revision=regression.revision,
            batch_state=None,
            plot_mode="R",
            selected_image_order=None,
        )
        before_projection = copy.deepcopy(value)
        render_linear_plot(RecordingAxes(), value)
        self.assertEqual(samples, before_samples)
        self.assertEqual(value, before_projection)

    def test_22_excel_projection_state_ui_time_and_transaction_contracts_are_protected(self):
        self.assertEqual(protected_identity_mismatches(), ())
        self.assertEqual(
            {name: method_hash(name) for name in PROTECTED_METHOD_HASHES},
            PROTECTED_METHOD_HASHES,
        )


if __name__ == "__main__":
    unittest.main()
