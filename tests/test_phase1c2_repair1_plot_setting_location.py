import ast
import copy
import hashlib
import io
import json
import math
import numbers
import types
import unittest
from pathlib import Path


DETECTMAIN_PATH = Path("detectmain.py")
WINDOW_PATH = Path("detectionwindow.py")
CONFIG_PATH = Path("interface_config.py")
DIALOG_PATH = Path("interface_settings_dialog.py")
WORKER_PATH = Path("yolo_detection_worker.py")

DETECTMAIN_SOURCE = DETECTMAIN_PATH.read_text(encoding="utf-8")
WINDOW_SOURCE = WINDOW_PATH.read_text(encoding="utf-8")
CONFIG_SOURCE = CONFIG_PATH.read_text(encoding="utf-8")
DIALOG_SOURCE = DIALOG_PATH.read_text(encoding="utf-8")
WORKER_SOURCE = WORKER_PATH.read_text(encoding="utf-8")

DETECTMAIN_TREE = ast.parse(DETECTMAIN_SOURCE, filename=str(DETECTMAIN_PATH))
WINDOW_TREE = ast.parse(WINDOW_SOURCE, filename=str(WINDOW_PATH))
CONFIG_TREE = ast.parse(CONFIG_SOURCE, filename=str(CONFIG_PATH))
DIALOG_TREE = ast.parse(DIALOG_SOURCE, filename=str(DIALOG_PATH))

PROTECTED_FILE_HASHES = {
    "linear_plot_projection.py": "c7eff50fbe54ecea35bd1cd68f120f3a4e833ca15b5d5079ed6b30c59422717c",
}
REPAIR5_RENDERER_IDENTITY = (
    12779,
    "e1e7daa436b9ca7dc192c6036a31ae446ab50688a3eddfecaa6d62b5c5d10fce",
)
PROTECTED_UI_HASHES = {
    "ui/detectmain.ui": "cb533b3586f3c8c3e578a4340bffe11bec94091e2e59542d24e6757679648cf4",
    "ui/ui_detectmain.py": "e342f9c59348e64ce6676ffd52c78e85745be5c0d90b95fa08b7d34ca62ecc29",
    "ui/detectwindow.ui": "58d77e803990e191cc3831a7533a80b0dba7c0249258638a4d7524df2f7a4b42",
    "ui/ui_detectwindow.py": "8550c0a4710d1919e6e0dcecfd9d67515593438973efb91eedf623ce01159f08",
}


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


def method_source(class_value, source, name):
    return ast.get_source_segment(source, method_node(class_value, name))


def compile_method(class_value, name, globals_value=None):
    node = copy.deepcopy(method_node(class_value, name))
    node.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    namespace = {} if globals_value is None else dict(globals_value)
    exec(compile(module, "<isolated:{}>".format(name), "exec"), namespace)
    return namespace[name]


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
    exec(compile(module, "<interface-config-pure-logic>", "exec"), namespace)
    return namespace


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


DETECTMAIN_CLASS = class_node(DETECTMAIN_TREE, "DetectMain")
WINDOW_CLASS = class_node(WINDOW_TREE, "DetectWindow")
DIALOG_CLASS = class_node(DIALOG_TREE, "InterfaceSettingsDialog")
CONFIG_LOGIC = compile_config_logic()


class ValueControl:
    def __init__(self, value):
        self._value = value

    def value(self):
        return self._value

    def currentText(self):
        return self._value

    def isChecked(self):
        return self._value


class Repair1PlotSettingLocationTests(unittest.TestCase):
    def test_01_detectionwindow_has_no_plot_channel_submenu(self):
        self.assertNotIn("_setup_linear_plot_channel_menu", WINDOW_SOURCE)
        self.assertNotIn("menuLinear_Plot_Channel", WINDOW_SOURCE)
        self.assertNotIn('addMenu("Plot Channel")', WINDOW_SOURCE)

    def test_02_plot_action_group_is_removed(self):
        self.assertNotIn("_linear_plot_channel_group", WINDOW_SOURCE)
        self.assertNotIn("bind_linear_plot_channel_action_sync", DETECTMAIN_SOURCE)

    def test_03_plot_actions_are_removed(self):
        self.assertNotIn("actionLinear_Plot_", WINDOW_SOURCE)
        self.assertNotIn("_linear_plot_channel_actions", WINDOW_SOURCE)

    def test_04_linear_numbering_menu_is_retained(self):
        setup = method_source(WINDOW_CLASS, WINDOW_SOURCE, "_setup_linear_menu")
        self.assertIn("actionLinear_Single_Image", setup)
        self.assertIn("actionLinear_Image_Series", setup)
        edit_setup = method_source(
            WINDOW_CLASS, WINDOW_SOURCE, "_setup_edit_camera_menu"
        )
        self.assertIn("menuLinear.menuAction()", edit_setup)

    def test_05_detection_color_channel_label_exists(self):
        init = method_source(DIALOG_CLASS, DIALOG_SOURCE, "__init__")
        self.assertIn('(\"Detection color channel\", self.color_channel)', init)

    def test_06_detection_selector_is_exact_r_g_b(self):
        init = method_node(DIALOG_CLASS, "__init__")
        choices = [
            ast.literal_eval(call.args[0])
            for call in ast.walk(init)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "addItems"
            and isinstance(call.func.value, ast.Attribute)
            and call.func.value.attr == "color_channel"
        ]
        self.assertEqual(choices, [("R", "G", "B")])

    def test_07_plot_channel_row_immediately_follows_detection_channel(self):
        init = method_source(DIALOG_CLASS, DIALOG_SOURCE, "__init__")
        expected = (
            '(\"Detection color channel\", self.color_channel),\n'
            '                (\"Linear plot channel\", self.linear_plot_channel),'
        )
        self.assertIn(expected, init)

    def test_08_plot_selector_is_exact_r_g_b_rgb(self):
        init = method_node(DIALOG_CLASS, "__init__")
        choices = [
            ast.literal_eval(call.args[0])
            for call in ast.walk(init)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "addItems"
            and isinstance(call.func.value, ast.Attribute)
            and call.func.value.attr == "linear_plot_channel"
        ]
        self.assertEqual(choices, [("R", "G", "B", "RGB")])

    def test_09_selectors_use_distinct_configuration_keys(self):
        settings_source = method_source(DIALOG_CLASS, DIALOG_SOURCE, "_settings")
        self.assertIn('"color_channel": self.color_channel.currentText()', settings_source)
        self.assertIn(
            '"linear_plot_channel": self.linear_plot_channel.currentText()',
            settings_source,
        )

    def test_10_legacy_config_migrates_from_valid_detection_channel(self):
        merged = CONFIG_LOGIC["merge_settings_overrides"](
            valid_settings(), {"color_channel": "G"}
        )
        self.assertEqual(merged["color_channel"], "G")
        self.assertEqual(merged["linear_plot_channel"], "G")

    def test_11_legacy_migration_has_no_write_path(self):
        node = copy.deepcopy(function_node(CONFIG_TREE, "load_effective_settings"))
        module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
        access_modes = []

        class FakePath:
            def is_file(self):
                return True

            def open(self, mode, encoding=None):
                access_modes.append((mode, encoding))
                return io.StringIO(json.dumps({
                    "version": 1,
                    "overrides": {"color_channel": "G"},
                }))

        namespace = {
            "Path": lambda _value: FakePath(),
            "CONFIG_VERSION": 1,
            "json": json,
            "load_interface_module": lambda: object(),
            "default_settings": lambda _module: valid_settings(),
            "configuration_path": lambda: (_ for _ in ()).throw(
                AssertionError("default path should not be used")
            ),
            "merge_settings_overrides": CONFIG_LOGIC["merge_settings_overrides"],
            "apply_settings": lambda _module, value: value,
        }
        exec(
            compile(module, "<isolated:load_effective_settings>", "exec"),
            namespace,
        )
        settings, warnings, _interface = namespace["load_effective_settings"](
            config_file="legacy.json", apply_to_module=False
        )
        self.assertEqual(warnings, [])
        self.assertEqual(settings["linear_plot_channel"], "G")
        self.assertEqual(access_modes, [("r", "utf-8")])

    def test_12_invalid_plot_channel_falls_back_to_b(self):
        merged = CONFIG_LOGIC["merge_settings_overrides"](
            valid_settings(),
            {"color_channel": "R", "linear_plot_channel": "invalid"},
        )
        self.assertEqual(merged["color_channel"], "R")
        self.assertEqual(merged["linear_plot_channel"], "B")

    def test_13_detection_color_channel_still_rejects_rgb(self):
        settings = valid_settings()
        settings["color_channel"] = "RGB"
        with self.assertRaisesRegex(ValueError, "color_channel must be R, G, or B"):
            CONFIG_LOGIC["validate_settings"](settings)

    def test_14_restore_defaults_resets_both_channels(self):
        restore = compile_method(DIALOG_CLASS, "_restore_defaults")
        captured = []
        defaults = valid_settings()
        harness = types.SimpleNamespace(
            _defaults=defaults,
            _set_values=lambda value: captured.append(dict(value)),
        )
        restore(harness)
        self.assertEqual(captured[0]["color_channel"], "B")
        self.assertEqual(captured[0]["linear_plot_channel"], "B")

    def test_15_cancel_changes_no_runtime_channel_and_does_not_redraw(self):
        calls = []

        class FakeDialog:
            def __init__(self, _parent):
                self.saved_settings = {
                    "color_channel": "R", "linear_plot_channel": "RGB"
                }

            def exec(self):
                return 0

            def deleteLater(self):
                calls.append("delete")

        opener = compile_method(
            WINDOW_CLASS,
            "_open_interface_settings",
            {
                "InterfaceSettingsDialog": FakeDialog,
                "QDialog": types.SimpleNamespace(Accepted=1),
            },
        )
        main = types.SimpleNamespace(
            _set_linear_plot_channel_mode=lambda mode: calls.append(("plot", mode))
        )
        harness = types.SimpleNamespace(
            _interface_settings_dialog=None, _detectMain=main
        )
        opener(harness)
        self.assertEqual(calls, ["delete"])

    def test_16_ok_collects_two_independent_channel_values(self):
        settings_method = compile_method(DIALOG_CLASS, "_settings")
        values = valid_settings()
        harness = types.SimpleNamespace(
            detect_confidence=ValueControl(values["detect_confidence"]),
            show_confidence=ValueControl(values["show_confidence"]),
            x0_ratio=ValueControl(values["x0_ratio"]),
            y0_ratio=ValueControl(values["y0_ratio"]),
            x1_ratio=ValueControl(values["x1_ratio"]),
            y1_ratio=ValueControl(values["y1_ratio"]),
            color_channel=ValueControl("R"),
            linear_plot_channel=ValueControl("RGB"),
            rgb_calculate_accuracy=ValueControl(values["rgb_calculate_accuracy"]),
            rgb_display_accuracy=ValueControl(values["rgb_display_accuracy"]),
            con_display_accuracy=ValueControl(values["con_display_accuracy"]),
            text_order=ValueControl(values["Order_Con_R_G_B"]),
            _calibration_values=lambda: (
                values["con_list"], values["linear_formula_point_matrix"]
            ),
        )
        settings = CONFIG_LOGIC["validate_settings"](settings_method(harness))
        self.assertEqual((settings["color_channel"], settings["linear_plot_channel"]), ("R", "RGB"))

        accept_node = copy.deepcopy(method_node(DIALOG_CLASS, "accept"))
        accept_node.decorator_list = []
        isolated_class = ast.ClassDef(
            name="IsolatedDialog",
            bases=[ast.Name(id="FakeDialogBase", ctx=ast.Load())],
            keywords=[],
            body=[accept_node],
            decorator_list=[],
        )
        module = ast.fix_missing_locations(
            ast.Module(body=[isolated_class], type_ignores=[])
        )
        saved = []
        messages = []

        class FakeDialogBase:
            def accept(self):
                self.base_accepted = True

        class FakeMessageBox:
            @staticmethod
            def critical(*args):
                raise AssertionError("valid settings must not show an error")

            @staticmethod
            def information(*args):
                messages.append(args)

        namespace = {
            "FakeDialogBase": FakeDialogBase,
            "validate_settings": CONFIG_LOGIC["validate_settings"],
            "save_settings": lambda value: saved.append(dict(value)),
            "QMessageBox": FakeMessageBox,
        }
        exec(compile(module, "<isolated:dialog-accept>", "exec"), namespace)
        isolated = namespace["IsolatedDialog"]()
        isolated._settings = lambda: dict(settings)
        isolated._saved_settings = None
        isolated.accept()
        self.assertEqual(
            (saved[0]["color_channel"], saved[0]["linear_plot_channel"]),
            ("R", "RGB"),
        )
        self.assertEqual(isolated._saved_settings, settings)
        self.assertTrue(isolated.base_accepted)
        self.assertEqual(len(messages), 1)

    def test_17_ok_passes_only_plot_setting_to_runtime_plot_setter(self):
        calls = []

        class FakeDialog:
            def __init__(self, _parent):
                self.saved_settings = {
                    "color_channel": "R", "linear_plot_channel": "RGB"
                }

            def exec(self):
                return 1

            def deleteLater(self):
                calls.append("delete")

        opener = compile_method(
            WINDOW_CLASS,
            "_open_interface_settings",
            {
                "InterfaceSettingsDialog": FakeDialog,
                "QDialog": types.SimpleNamespace(Accepted=1),
            },
        )
        main = types.SimpleNamespace(
            _set_linear_plot_channel_mode=lambda mode: calls.append(("plot", mode))
        )
        harness = types.SimpleNamespace(
            _interface_settings_dialog=None, _detectMain=main
        )
        opener(harness)
        self.assertEqual(calls, [("plot", "RGB"), "delete"])

    def test_18_detection_channel_retains_formal_worker_path(self):
        self.assertIn('color_channel = settings["color_channel"]', WORKER_SOURCE)
        self.assertIn('concentration_for(color_channel)', WORKER_SOURCE)
        self.assertIn('"Con.{}:{}".format(color_channel, con_text)', WORKER_SOURCE)

    def test_19_plot_setter_does_not_write_detection_setting(self):
        source = method_source(
            DETECTMAIN_CLASS, DETECTMAIN_SOURCE, "_set_linear_plot_channel_mode"
        )
        self.assertNotIn("color_channel", source)
        self.assertNotIn("settings", source)

    def test_20_plot_setter_has_no_configuration_write(self):
        source = method_source(
            DETECTMAIN_CLASS, DETECTMAIN_SOURCE, "_set_linear_plot_channel_mode"
        )
        self.assertNotIn("save_settings", source)
        self.assertNotIn("write", source)
        self.assertNotIn("open", source)

    def test_21_active_regression_causes_exactly_one_redraw(self):
        setter = compile_method(
            DETECTMAIN_CLASS,
            "_set_linear_plot_channel_mode",
            {"_LINEAR_PLOT_CHANNEL_MODES": ("R", "G", "B", "RGB")},
        )
        redraws = []
        harness = types.SimpleNamespace(
            _linear_plot_channel_mode="B",
            _active_linear_regression_set=lambda: object(),
            _plot_regression_result=lambda: redraws.append("draw"),
        )
        setter(harness, "RGB")
        self.assertEqual(redraws, ["draw"])

    def test_22_missing_active_regression_causes_no_redraw(self):
        setter = compile_method(
            DETECTMAIN_CLASS,
            "_set_linear_plot_channel_mode",
            {"_LINEAR_PLOT_CHANNEL_MODES": ("R", "G", "B", "RGB")},
        )
        redraws = []
        harness = types.SimpleNamespace(
            _linear_plot_channel_mode="B",
            _active_linear_regression_set=lambda: None,
            _plot_regression_result=lambda: redraws.append("draw"),
        )
        setter(harness, "R")
        self.assertEqual(redraws, [])

    def test_23_plot_setting_update_does_not_read_clock(self):
        sources = "\n".join((
            method_source(DETECTMAIN_CLASS, DETECTMAIN_SOURCE, "_set_linear_plot_channel_mode"),
            method_source(WINDOW_CLASS, WINDOW_SOURCE, "_open_interface_settings"),
        ))
        for forbidden in ("perf_counter", "monotonic", "active_elapsed_ms", "duration"):
            self.assertNotIn(forbidden, sources)

    def test_24_plot_setting_update_runs_no_worker_model_or_detection(self):
        sources = "\n".join((
            method_source(DETECTMAIN_CLASS, DETECTMAIN_SOURCE, "_set_linear_plot_channel_mode"),
            method_source(WINDOW_CLASS, WINDOW_SOURCE, "_open_interface_settings"),
        ))
        for forbidden in (
            "regression_requested", "detection_requested", "worker", "model",
            "calculate_all", "calculate(", "Camera",
        ):
            self.assertNotIn(forbidden, sources)

    def test_25_projection_hash_is_unchanged_and_renderer_change_is_repair3_only(self):
        actual = {
            path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for path in PROTECTED_FILE_HASHES
        }
        self.assertEqual(actual, PROTECTED_FILE_HASHES)
        self.assertEqual(
            (
                Path("linear_plot_renderer.py").stat().st_size,
                hashlib.sha256(Path("linear_plot_renderer.py").read_bytes()).hexdigest(),
            ),
            REPAIR5_RENDERER_IDENTITY,
        )

    def test_26_save_linear_current_view_path_is_unchanged(self):
        source = method_source(
            DETECTMAIN_CLASS, DETECTMAIN_SOURCE, "_build_linear_export_bytes"
        )
        self.assertEqual(
            hashlib.sha256(source.encode("utf-8")).hexdigest(),
            "fe509b7dd037bd3d04e5a38de23b4aacbdae75c43958ff922f28f779f7bcfe74",
        )
        self.assertLess(
            source.index("self._plot_regression_result()"),
            source.index("self.canvas.print_png"),
        )

    def test_27_both_result_tables_still_have_five_columns(self):
        assignment = next(
            node for node in DETECTMAIN_CLASS.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id == "CALIBRATION_TABLE_HEADERS"
                for target in node.targets
            )
        )
        self.assertEqual(
            ast.literal_eval(assignment.value),
            ("No.", "Con.", "Red", "Green", "Blue"),
        )

    def test_28_ui_and_generated_ui_files_are_unchanged(self):
        actual = {
            path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for path in PROTECTED_UI_HASHES
        }
        self.assertEqual(actual, PROTECTED_UI_HASHES)

    def test_29_new_test_creates_no_qt_or_gui_object(self):
        source = Path(__file__).read_text(encoding="utf-8")
        tree = ast.parse(source, filename=__file__)
        imports = set()
        calls = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".")[0])
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    calls.add(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    calls.add(node.func.attr)
        self.assertNotIn("Py" + "Side6", imports)
        self.assertTrue(calls.isdisjoint({
            "Q" + "Application", "Q" + "Widget", "Q" + "Thread",
            "Detect" + "Main", "InterfaceSettings" + "Dialog",
            "YoloDetection" + "Worker", "FigureCanvas" + "QTAgg",
        }))

    def test_30_project_has_only_one_plot_channel_control(self):
        production_sources = {
            path.name: path.read_text(encoding="utf-8")
            for path in Path(".").glob("*.py")
            if path.name not in {"interface_config.py", "detectmain.py"}
        }
        assignments = []
        for name, source in production_sources.items():
            tree = ast.parse(source, filename=name)
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Assign)
                    and any(
                        isinstance(target, ast.Attribute)
                        and target.attr == "linear_plot_channel"
                        for target in node.targets
                    )
                ):
                    assignments.append(name)
        self.assertEqual(assignments, ["interface_settings_dialog.py"])
        self.assertNotIn("Plot Channel", WINDOW_SOURCE)

    def test_31_new_install_plot_channel_default_is_b(self):
        source = ast.get_source_segment(
            CONFIG_SOURCE, function_node(CONFIG_TREE, "default_settings")
        )
        self.assertIn('raw_defaults["linear_plot_channel"] = "B"', source)


if __name__ == "__main__":
    unittest.main()
