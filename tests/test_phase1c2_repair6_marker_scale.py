import ast
import copy
import unittest
from pathlib import Path

from batch_state import ConcentrationStatus
from linear_plot_renderer import render_linear_plot
from tests.test_phase1c2_plot_integration import PROTECTED_METHOD_HASHES, method_hash
from tests.test_phase1c2_repair5_plot_clarity import (
    EDGE_COLORS,
    NORMAL_COLORS,
    PROTECTED_IDENTITIES,
    REGRESSION_COLORS,
    RecordingAxes,
    actual_markers,
    point,
    projection,
    protected_identity_mismatches,
    public_labels,
)


DETECTED_MARKER_SIZE = 50
STANDARD_MARKER_SIZE = 36
OUT_OF_RANGE_MARKER_SIZE = 65


def render_formal(mode="RGB", points=()):
    axes = RecordingAxes()
    value = projection(mode, points)
    render_linear_plot(axes, value)
    return axes, value


def standard_markers(axes):
    return actual_markers(axes, "o")


def detection_markers(axes, marker="D"):
    return actual_markers(axes, marker)


class Repair6MarkerScaleTests(unittest.TestCase):
    def test_01_r_g_b_normal_diamonds_use_size_50(self):
        for channel in ("R", "G", "B"):
            with self.subTest(channel=channel):
                axes, _ = render_formal(channel, (point(channel),))
                self.assertEqual(detection_markers(axes)[0]["s"], DETECTED_MARKER_SIZE)

    def test_02_single_and_rgb_diamonds_share_size_50(self):
        single, _ = render_formal("R", (point("R"),))
        rgb, _ = render_formal("RGB", tuple(point(channel) for channel in ("R", "G", "B")))
        self.assertEqual([call["s"] for call in detection_markers(single)], [50])
        self.assertEqual([call["s"] for call in detection_markers(rgb)], [50, 50, 50])

    def test_03_diamond_edges_and_linewidth_are_unchanged(self):
        for channel in ("R", "G", "B"):
            with self.subTest(channel=channel):
                axes, _ = render_formal(channel, (point(channel),))
                marker = detection_markers(axes)[0]
                self.assertEqual(marker["color"], NORMAL_COLORS[channel])
                self.assertEqual(marker["edgecolors"], EDGE_COLORS[channel])
                self.assertEqual(marker["linewidths"], 1.5)

    def test_04_diamond_zorder_remains_above_standards_and_lines(self):
        axes, _ = render_formal("R", (point("R"),))
        diamond = detection_markers(axes)[0]
        standard = standard_markers(axes)[0]
        self.assertGreater(diamond["zorder"], standard["zorder"])
        self.assertGreater(diamond["zorder"], max(call["zorder"] for call in axes.plot_calls))

    def test_05_r_g_b_standard_points_use_size_36(self):
        axes, _ = render_formal()
        self.assertEqual([call["s"] for call in standard_markers(axes)], [36, 36, 36])

    def test_06_single_and_rgb_standards_share_size_36(self):
        for channel in ("R", "G", "B"):
            with self.subTest(channel=channel):
                single, _ = render_formal(channel)
                self.assertEqual([call["s"] for call in standard_markers(single)], [36])
        rgb, _ = render_formal()
        self.assertEqual([call["s"] for call in standard_markers(rgb)], [36, 36, 36])

    def test_07_r_standard_face_and_edge_match_regression_color(self):
        axes, _ = render_formal("R")
        standard = standard_markers(axes)[0]
        self.assertEqual((standard["color"], standard["edgecolors"]), ("#E57373", "#E57373"))

    def test_08_g_standard_face_and_edge_match_regression_color(self):
        axes, _ = render_formal("G")
        standard = standard_markers(axes)[0]
        self.assertEqual((standard["color"], standard["edgecolors"]), ("#66BB6A", "#66BB6A"))

    def test_09_b_standard_face_and_edge_match_regression_color(self):
        axes, _ = render_formal("B")
        standard = standard_markers(axes)[0]
        self.assertEqual((standard["color"], standard["edgecolors"]), ("#64B5F6", "#64B5F6"))

    def test_10_standard_points_and_regression_lines_share_one_color_mapping(self):
        axes, _ = render_formal()
        self.assertEqual(
            [call["color"] for call in standard_markers(axes)],
            [call["color"] for call in axes.plot_calls],
        )
        source = Path("linear_plot_renderer.py").read_text(encoding="utf-8")
        tree = ast.parse(source, filename="linear_plot_renderer.py")
        renderer = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "render_linear_plot")
        standard_call = next(
            node for node in ast.walk(renderer)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "scatter"
            and any(keyword.arg == "marker" and isinstance(keyword.value, ast.Constant) and keyword.value.value == "o" for keyword in node.keywords)
        )
        keyword_values = {keyword.arg: ast.unparse(keyword.value) for keyword in standard_call.keywords}
        self.assertEqual(keyword_values["color"], "regression_color")
        self.assertEqual(keyword_values["edgecolors"], "regression_color")
        self.assertEqual(source.count("_REGRESSION_COLORS ="), 1)

    def test_11_standard_coordinates_count_alpha_and_marker_are_unchanged(self):
        axes, _ = render_formal()
        standards = standard_markers(axes)
        self.assertEqual(len(standards), 3)
        self.assertEqual([call["marker"] for call in standards], ["o", "o", "o"])
        self.assertEqual([call["x"] for call in standards], [(1.0, 5.0)] * 3)
        self.assertEqual([call["y"] for call in standards], [(2.0, 8.0), (12.0, 18.0), (22.0, 28.0)])
        self.assertTrue(all("alpha" not in call for call in standards))

    def test_12_solid_and_dashed_regression_geometry_and_style_are_unchanged(self):
        axes, _ = render_formal("R", (
            point("R", ConcentrationStatus.BELOW_RANGE, 0.5, 150.0),
        ))
        solid = next(call for call in axes.plot_calls if call["linestyle"] == "-")
        dashed = next(call for call in axes.plot_calls if call["linestyle"] == "--")
        self.assertEqual((solid["color"], solid["linewidth"], solid["x"]), ("#E57373", 2.2, (1.0, 5.0)))
        for actual, expected in zip(solid["y"], (116.27, -155.05)):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual((dashed["color"], dashed["linewidth"], dashed["x"]), ("#E57373", 2.2, (0.5, 1.0)))
        for actual, expected in zip(dashed["y"], (150.185, 116.27)):
            self.assertAlmostEqual(actual, expected)

    def test_13_out_of_range_x_size_color_and_status_are_unchanged(self):
        value = point("G", ConcentrationStatus.ABOVE_RANGE, 7.0, 41.0)
        axes, projection_value = render_formal("G", (value,))
        marker = detection_markers(axes, "x")[0]
        self.assertEqual((marker["s"], marker["color"], marker["marker"]), (65, "#006400", "x"))
        self.assertEqual(projection_value.detection_points[0].status, ConcentrationStatus.ABOVE_RANGE)

    def test_14_sample_number_color_font_position_and_offset_are_unchanged(self):
        axes, _ = render_formal("B", (point("B"),))
        annotation = axes.annotation_calls[0]
        self.assertEqual(annotation["color"], "blue")
        self.assertEqual(annotation["fontsize"], 9)
        self.assertEqual(annotation["xy"], (2.0, 14.0))
        self.assertEqual(annotation["xytext"], (5, 5))
        self.assertEqual(annotation["textcoords"], "offset points")

    def test_15_single_title_subtitle_and_dynamic_legend_are_unchanged(self):
        axes, _ = render_formal("R", (point("R"),))
        self.assertEqual(axes.title, "Linear Regression – R")
        self.assertEqual(axes.title_kwargs, {"fontsize": 13, "y": 1.10, "pad": 0})
        self.assertEqual(axes.text_calls[0]["text"], "y = -67.83x + 184.1; R² = 0.953")
        self.assertEqual(public_labels(axes), ["Detected (n=1)"])
        self.assertEqual(len(axes.legend_calls), 1)

    def test_16_rgb_gray_summary_sizes_and_unique_counts_are_preserved(self):
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
        axes, _ = render_formal("RGB", points)
        handles = [call for call in axes.scatter_calls if not call["x"]]
        self.assertEqual(public_labels(axes), ["Detected (n=2)", "Out of range (n=1)"])
        self.assertEqual(
            [(call["color"], call["marker"], call["s"]) for call in handles],
            [("#666666", "D", 50), ("#666666", "x", 65)],
        )

    def test_17_rgb_actual_points_remain_saturated_and_never_gray(self):
        axes, _ = render_formal("RGB", tuple(point(channel) for channel in ("R", "G", "B")))
        actual = detection_markers(axes)
        self.assertEqual([call["color"] for call in actual], ["red", "green", "blue"])
        self.assertEqual([call["s"] for call in actual], [50, 50, 50])
        self.assertNotIn("#666666", [call["color"] for call in actual])

    def test_18_figure_dpi_axes_bounds_and_layout_are_unchanged(self):
        axes = RecordingAxes()
        before = (axes.figure.figsize, axes.figure.dpi, axes.bounds)
        render_linear_plot(axes, projection("R", (point("R"),)))
        self.assertEqual((axes.figure.figsize, axes.figure.dpi, axes.bounds), before)
        source = Path("detectmain.py").read_text(encoding="utf-8")
        self.assertIn("plt.subplots(figsize=(15,15), dpi=100)", source)
        self.assertIn("left=0.14, right=0.96, bottom=0.16, top=0.80", source)

    def test_19_projection_is_not_modified(self):
        value = projection("RGB", tuple(point(channel) for channel in ("R", "G", "B")))
        before = copy.deepcopy(value)
        render_linear_plot(RecordingAxes(), value)
        self.assertEqual(value, before)

    def test_20_excel_ui_time_png_and_save_contracts_are_protected(self):
        self.assertEqual(set(PROTECTED_IDENTITIES), {
            "linear_plot_projection.py", "detectmain.py", "batch_state.py",
            "batch_detection_controller.py", "yolo_detection_worker.py",
            "interface_config.py", "interface_settings_dialog.py",
            "ui/detectmain.ui", "ui/ui_detectmain.py",
        })
        self.assertEqual(protected_identity_mismatches(), ())
        self.assertEqual(
            {name: method_hash(name) for name in PROTECTED_METHOD_HASHES},
            PROTECTED_METHOD_HASHES,
        )


if __name__ == "__main__":
    unittest.main()
