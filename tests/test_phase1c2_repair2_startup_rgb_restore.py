import ast
import copy
import io
import json
import math
import numbers
import types
import unittest
from pathlib import Path


DETECTMAIN_PATH = Path("detectmain.py")
CONFIG_PATH = Path("interface_config.py")
DETECTMAIN_TREE = ast.parse(
    DETECTMAIN_PATH.read_text(encoding="utf-8"), filename=str(DETECTMAIN_PATH)
)
CONFIG_TREE = ast.parse(
    CONFIG_PATH.read_text(encoding="utf-8"), filename=str(CONFIG_PATH)
)


def class_node(tree, name):
    return next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == name
    )


def function_node(tree, name):
    return next(
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    )


def method_node(class_value, name):
    return next(
        node for node in class_value.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    )


DETECTMAIN_CLASS = class_node(DETECTMAIN_TREE, "DetectMain")


def startup_nodes():
    init = method_node(DETECTMAIN_CLASS, "__init__")
    startup_load = next(
        node for node in init.body
        if isinstance(node, ast.Try)
        and any(
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Name)
            and child.func.id == "load_effective_settings"
            for child in ast.walk(node)
        )
    )
    mode_assignment = next(
        node for node in init.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "self"
            and target.attr == "_linear_plot_channel_mode"
            for target in node.targets
        )
    )
    return startup_load, mode_assignment


def official_modes():
    assignment = next(
        node for node in DETECTMAIN_TREE.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "_LINEAR_PLOT_CHANNEL_MODES"
            for target in node.targets
        )
    )
    return ast.literal_eval(assignment.value)


def compile_startup_mode_restore(load_effective_settings):
    startup_load, mode_assignment = startup_nodes()
    function = ast.FunctionDef(
        name="run_startup_mode_restore",
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg="self")],
            vararg=None,
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=[],
        ),
        body=[
            copy.deepcopy(startup_load),
            copy.deepcopy(mode_assignment),
            ast.Return(
                value=ast.Attribute(
                    value=ast.Name(id="self", ctx=ast.Load()),
                    attr="_linear_plot_channel_mode",
                    ctx=ast.Load(),
                )
            ),
        ],
        decorator_list=[],
        returns=None,
        type_comment=None,
    )
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace = {
        "load_effective_settings": load_effective_settings,
        "_LINEAR_PLOT_CHANNEL_MODES": official_modes(),
    }
    exec(compile(module, "<repair2-startup-mode-restore>", "exec"), namespace)
    return namespace[function.name]


def run_startup_snapshot(snapshot):
    calls = []

    def load_effective_settings(*, apply_to_module):
        calls.append(("load", apply_to_module))
        return copy.deepcopy(snapshot), [], None

    restore = compile_startup_mode_restore(load_effective_settings)
    harness = types.SimpleNamespace()
    result = restore(harness)
    return result, harness, calls


def compile_config_logic():
    wanted_assignments = {"ALLOWED_SETTINGS", "TEXT_ORDERS"}
    wanted_functions = {
        "_require_number",
        "_require_integer",
        "validate_settings",
        "merge_settings_overrides",
    }
    body = []
    for node in CONFIG_TREE.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in wanted_assignments
            for target in node.targets
        ):
            body.append(copy.deepcopy(node))
        elif isinstance(node, ast.FunctionDef) and node.name in wanted_functions:
            body.append(copy.deepcopy(node))
    module = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
    namespace = {"copy": copy, "math": math, "numbers": numbers}
    exec(compile(module, "<repair2-config-logic>", "exec"), namespace)
    return namespace


CONFIG_LOGIC = compile_config_logic()


def valid_settings():
    return {
        "detect_confidence": 0.05,
        "show_confidence": False,
        "x0_ratio": 0.1,
        "y0_ratio": 0.2,
        "x1_ratio": 0.8,
        "y1_ratio": 0.9,
        "color_channel": "B",
        "linear_plot_channel": "B",
        "rgb_calculate_accuracy": 16,
        "rgb_display_accuracy": 2,
        "con_display_accuracy": 2,
        "Order_Con_R_G_B": "ConGBR",
        "con_list": [0.0, 1.0],
        "linear_formula_point_matrix": [True, True],
    }


def load_config_snapshot(overrides):
    node = copy.deepcopy(function_node(CONFIG_TREE, "load_effective_settings"))
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    access_modes = []

    class FakePath:
        def is_file(self):
            return True

        def open(self, mode, encoding=None):
            access_modes.append((mode, encoding))
            return io.StringIO(json.dumps({"version": 1, "overrides": overrides}))

    namespace = {
        "Path": lambda _value: FakePath(),
        "CONFIG_VERSION": 1,
        "json": json,
        "load_interface_module": lambda: object(),
        "default_settings": lambda _module: valid_settings(),
        "configuration_path": lambda: (_ for _ in ()).throw(
            AssertionError("default path must not be used")
        ),
        "merge_settings_overrides": CONFIG_LOGIC["merge_settings_overrides"],
        "apply_settings": lambda _module, settings: settings,
    }
    exec(compile(module, "<repair2-load-effective-settings>", "exec"), namespace)
    snapshot, warnings, _interface = namespace["load_effective_settings"](
        config_file="saved-settings.json", apply_to_module=False
    )
    return snapshot, warnings, access_modes


class Repair2StartupRgbRestoreTests(unittest.TestCase):
    def test_01_all_four_valid_modes_survive_startup(self):
        for mode in ("R", "G", "B", "RGB"):
            with self.subTest(mode=mode):
                result, harness, calls = run_startup_snapshot(
                    {"linear_plot_channel": mode}
                )
                self.assertEqual(result, mode)
                self.assertEqual(harness._linear_plot_channel_mode, mode)
                self.assertEqual(calls, [("load", False)])

    def test_02_saved_rgb_survives_config_snapshot_and_startup(self):
        snapshot, warnings, access_modes = load_config_snapshot(
            {"color_channel": "G", "linear_plot_channel": "RGB"}
        )
        result, harness, calls = run_startup_snapshot(snapshot)
        self.assertEqual(warnings, [])
        self.assertEqual(access_modes, [("r", "utf-8")])
        self.assertEqual(snapshot["linear_plot_channel"], "RGB")
        self.assertEqual(result, "RGB")
        self.assertEqual(harness._linear_plot_channel_mode, "RGB")
        self.assertEqual(calls, [("load", False)])

    def test_03_missing_mode_falls_back_to_b(self):
        result, harness, _calls = run_startup_snapshot({})
        self.assertEqual(result, "B")
        self.assertEqual(harness._linear_plot_channel_mode, "B")

    def test_04_invalid_modes_fall_back_to_b(self):
        invalid_values = (None, "", "rgb", "RgB", True, 1, (), [], {})
        for value in invalid_values:
            with self.subTest(value=value):
                result, harness, _calls = run_startup_snapshot(
                    {"linear_plot_channel": value}
                )
                self.assertEqual(result, "B")
                self.assertEqual(harness._linear_plot_channel_mode, "B")

    def test_05_legacy_inherited_r_g_b_modes_survive_startup(self):
        for detection_mode in ("R", "G", "B"):
            with self.subTest(detection_mode=detection_mode):
                snapshot, warnings, access_modes = load_config_snapshot(
                    {"color_channel": detection_mode}
                )
                result, harness, _calls = run_startup_snapshot(snapshot)
                self.assertEqual(warnings, [])
                self.assertEqual(access_modes, [("r", "utf-8")])
                self.assertEqual(snapshot["linear_plot_channel"], detection_mode)
                self.assertEqual(result, detection_mode)
                self.assertEqual(
                    harness._linear_plot_channel_mode, detection_mode
                )

    def test_06_startup_selection_does_not_mutate_settings_or_write_files(self):
        snapshot = {"color_channel": "R", "linear_plot_channel": "RGB"}
        before = copy.deepcopy(snapshot)
        result, _harness, calls = run_startup_snapshot(snapshot)
        startup_load, mode_assignment = startup_nodes()
        called_names = {
            child.func.id
            for node in (startup_load, mode_assignment)
            for child in ast.walk(node)
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
        }
        called_attributes = {
            child.func.attr
            for node in (startup_load, mode_assignment)
            for child in ast.walk(node)
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute)
        }
        self.assertEqual(result, "RGB")
        self.assertEqual(snapshot, before)
        self.assertEqual(calls, [("load", False)])
        self.assertTrue(
            called_names.isdisjoint({"open", "save_settings", "write"})
        )
        self.assertTrue(
            called_attributes.isdisjoint(
                {"open", "write", "write_text", "write_bytes", "save"}
            )
        )

    def test_07_startup_selection_runs_no_redraw_clock_worker_model_or_camera(self):
        result, _harness, calls = run_startup_snapshot(
            {"linear_plot_channel": "RGB"}
        )
        startup_load, mode_assignment = startup_nodes()
        referenced_names = {
            child.id
            for node in (startup_load, mode_assignment)
            for child in ast.walk(node)
            if isinstance(child, ast.Name)
        }
        referenced_attributes = {
            child.attr
            for node in (startup_load, mode_assignment)
            for child in ast.walk(node)
            if isinstance(child, ast.Attribute)
        }
        forbidden = {
            "_plot_regression_result",
            "perf_counter",
            "monotonic",
            "regression_requested",
            "detection_requested",
            "worker",
            "model",
            "Camera",
            "VideoCapture",
        }
        self.assertEqual(result, "RGB")
        self.assertEqual(calls, [("load", False)])
        self.assertTrue(referenced_names.isdisjoint(forbidden))
        self.assertTrue(referenced_attributes.isdisjoint(forbidden))

    def test_08_rgb_plot_mode_remains_independent_from_detection_color_channel(self):
        for detection_mode in ("R", "G", "B"):
            with self.subTest(detection_mode=detection_mode):
                snapshot, warnings, _access_modes = load_config_snapshot(
                    {
                        "color_channel": detection_mode,
                        "linear_plot_channel": "RGB",
                    }
                )
                result, harness, _calls = run_startup_snapshot(snapshot)
                self.assertEqual(warnings, [])
                self.assertEqual(snapshot["color_channel"], detection_mode)
                self.assertNotEqual(snapshot["color_channel"], "RGB")
                self.assertEqual(result, "RGB")
                self.assertEqual(harness._linear_plot_channel_mode, "RGB")


if __name__ == "__main__":
    unittest.main()
