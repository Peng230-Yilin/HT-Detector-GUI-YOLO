import ast
import copy
import hashlib
import math
import types
import unittest
from pathlib import Path

from batch_state import RegressionSet


DETECTMAIN_PATH = Path("detectmain.py")
WINDOW_PATH = Path("detectionwindow.py")
CONFIG_PATH = Path("interface_config.py")
SETTINGS_DIALOG_PATH = Path("interface_settings_dialog.py")
DETECTMAIN_SOURCE = DETECTMAIN_PATH.read_text(encoding="utf-8")
WINDOW_SOURCE = WINDOW_PATH.read_text(encoding="utf-8")
CONFIG_SOURCE = CONFIG_PATH.read_text(encoding="utf-8")
SETTINGS_DIALOG_SOURCE = SETTINGS_DIALOG_PATH.read_text(encoding="utf-8")
DETECTMAIN_TREE = ast.parse(DETECTMAIN_SOURCE, filename=str(DETECTMAIN_PATH))
WINDOW_TREE = ast.parse(WINDOW_SOURCE, filename=str(WINDOW_PATH))
CONFIG_TREE = ast.parse(CONFIG_SOURCE, filename=str(CONFIG_PATH))
SETTINGS_DIALOG_TREE = ast.parse(
    SETTINGS_DIALOG_SOURCE, filename=str(SETTINGS_DIALOG_PATH)
)

PROTECTED_METHOD_HASHES = {
    "_build_detection_workbook_bytes": "5626692e11037af8937d0329e9233c001bf124768ed0114207021019d4b2b3f9",
    "_validate_detection_workbook_bytes": "613d6c7797c0af5bb2d2c73b24e8b500b086a2c81aefd4e9cd3a108bf044ee47",
    "_build_detection_export_bytes": "ee630526c18084b4240c04cd451a9740c91166f303f581b4c31f26af97e65438",
    "_commit_detection_export": "af9a90a0449e0d4c59d806857b101032f93d18fcc88c197a240c353f610c71c4",
    "_save_detection_result": "9b9f7792ea5676d04a906c0de5c6babc6d15a3a9d2f229bdf71e6784d7bb2c40",
    "_build_linear_workbook_bytes": "ce91d3f7d495ca3fc3bd857b0ee631495b5e1f18bd13f2417aeb3bbda0163b15",
    "_validate_linear_workbook_bytes": "49d4582495311883bd1b1cf2843d8c36fc0888786beed1093f035fcd648c65fa",
    "_build_linear_export_bytes": "fe509b7dd037bd3d04e5a38de23b4aacbdae75c43958ff922f28f779f7bcfe74",
    "_commit_linear_export": "b3ad2f4d0b44298967476989947732fdc24637ae75ce27c979aa02449b0ea104",
    "_save_linear_result": "ae92c7d87a5da7a177525d5b9063ff362a17d130131c1e179dd6e2b888ad5c83",
    "_on_regression_finished": "f3efe3e5d8eeb98d96bc938dee7a879f57eb7b80db07a0b00868d239b8dd6cca",
    "_on_regression_failed": "ce6c4225c696cf9748f89f3b02cefeb3b22a9dc0954f456fdf5ce187f25a2d66",
    "_on_detection_finished": "39064cb4beac4c0604714aaf9a47b4fcd48176ff9445f203f051a497f410c5d2",
    "_on_detection_failed": "4464cf5eea825f21a53099e113ea3cc2984218ef61c44cbf4c5b133c7d59b57e",
    "_show_detection_image": "daaeb4afc698984db0dcdd8a96f95d194259aa961852f7c83e4b6b8ef1c1af98",
    "_refresh_detection_duration_display": "161fe9441d2142d9991316827b5a1003f9a27ae2800b0dedd415e5458ca71666",
    "_refresh_linear_duration_display": "2140285587401a11bcfcae49f77dcf0d1a42b1417076864782583a4e78f0793b",
    "_set_shared_time_display": "0830c265e9570ba9258a42f7453907d823c10b1f4f8e3579bd38478544270c3b",
    "_refresh_shared_time_display": "7371a4b600331895da7201563c4d406f30cdaf34f36aa436a18ef3499a030f37",
    "_apply_result_pane_view": "8d85b8467956725ae7aa9b1aa15b5be8775419972fbaea98db63f6ec5517f936",
}

PROTECTED_UI_HASHES = {
    "ui/detectmain.ui": "cb533b3586f3c8c3e578a4340bffe11bec94091e2e59542d24e6757679648cf4",
    "ui/ui_detectmain.py": "e342f9c59348e64ce6676ffd52c78e85745be5c0d90b95fa08b7d34ca62ecc29",
    "ui/detectwindow.ui": "58d77e803990e191cc3831a7533a80b0dba7c0249258638a4d7524df2f7a4b42",
    "ui/ui_detectwindow.py": "8550c0a4710d1919e6e0dcecfd9d67515593438973efb91eedf623ce01159f08",
}

PHASE1C1_IDENTITIES = {
    "linear_plot_projection.py": (16234, "c7eff50fbe54ecea35bd1cd68f120f3a4e833ca15b5d5079ed6b30c59422717c"),
    "tests/test_linear_plot_projection.py": (15434, "594a4ef9d23878e857ca5be894675834b93b9a995fa51af39f0a032b668a92c4"),
    "tests/test_linear_plot_lifecycle.py": (13753, "52177b00d1301f73c73c27bf39c2ad378507b20947ffbe3eb9e8d33dd1ebb207"),
}


def class_node(tree, name):
    return next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)


DETECTMAIN_CLASS = class_node(DETECTMAIN_TREE, "DetectMain")
WINDOW_CLASS = class_node(WINDOW_TREE, "DetectWindow")
SETTINGS_DIALOG_CLASS = class_node(SETTINGS_DIALOG_TREE, "InterfaceSettingsDialog")


def method_node(class_value, name):
    return next(
        node for node in class_value.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    )


def method_source(class_value, source, name):
    return ast.get_source_segment(source, method_node(class_value, name))


def method_hash(name):
    value = method_source(DETECTMAIN_CLASS, DETECTMAIN_SOURCE, name)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def compile_method(name, globals_value=None):
    node = copy.deepcopy(method_node(DETECTMAIN_CLASS, name))
    node.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    namespace = {} if globals_value is None else dict(globals_value)
    exec(compile(module, "<detectmain-method:{}>".format(name), "exec"), namespace)
    return namespace[name]


def compile_startup_mode_restore(load_effective_settings):
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
    modes_assignment = next(
        node for node in DETECTMAIN_TREE.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "_LINEAR_PLOT_CHANNEL_MODES"
            for target in node.targets
        )
    )
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
        "_LINEAR_PLOT_CHANNEL_MODES": ast.literal_eval(modes_assignment.value),
    }
    exec(compile(module, "<detectmain-startup-mode-restore>", "exec"), namespace)
    return namespace[function.name]


def regression(source):
    def formula(slope, intercept, r2):
        return {
            "slope": slope,
            "intercept": intercept,
            "r": math.sqrt(r2),
            "R2": r2,
            "p": 0.01,
            "std_err": 0.1,
        }
    return RegressionSet.from_formulas(
        {
            "R": formula(2.0, 1.0, 0.91),
            "G": formula(3.0, 2.0, 0.92),
            "B": formula(4.0, 3.0, 0.93),
        },
        source_id=source,
        valid_ranges=(0.0, 10.0),
    )


class Phase1C2PlotIntegrationTests(unittest.TestCase):
    def test_linear_plot_channel_has_one_runtime_authority_field(self):
        assignments = []
        for filename, tree in (("detectmain", DETECTMAIN_TREE), ("window", WINDOW_TREE)):
            for node in ast.walk(tree):
                targets = []
                if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if isinstance(target, ast.Attribute) and target.attr == "_linear_plot_channel_mode":
                        assignments.append(filename)
        self.assertEqual(assignments, ["detectmain", "detectmain"])

    def test_only_exact_r_g_b_rgb_modes_are_accepted(self):
        constant = next(
            node for node in DETECTMAIN_TREE.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "_LINEAR_PLOT_CHANNEL_MODES" for target in node.targets)
        )
        self.assertEqual(ast.literal_eval(constant.value), ("R", "G", "B", "RGB"))
        setter = compile_method(
            "_set_linear_plot_channel_mode",
            {"_LINEAR_PLOT_CHANNEL_MODES": ("R", "G", "B", "RGB")},
        )
        harness = types.SimpleNamespace(
            _linear_plot_channel_mode="B",
            _active_linear_regression_set=lambda: None,
        )
        for valid in ("R", "G", "B", "RGB"):
            self.assertEqual(setter(harness, valid), valid)
        for invalid in ("", "rgb", "RgB", None, True, 1):
            with self.assertRaises(ValueError):
                setter(harness, invalid)

    def test_plot_channel_menu_is_removed_from_detectionwindow(self):
        self.assertNotIn("_setup_linear_plot_channel_menu", WINDOW_SOURCE)
        self.assertNotIn("menuLinear_Plot_Channel", WINDOW_SOURCE)
        self.assertNotIn("actionLinear_Plot_", WINDOW_SOURCE)
        edit_setup = method_source(WINDOW_CLASS, WINDOW_SOURCE, "_setup_edit_camera_menu")
        self.assertIn("edit_menu.addAction(self._uiWindow.menuLinear.menuAction())", edit_setup)

    def test_interface_settings_places_plot_channel_below_detection_channel(self):
        init = method_source(
            SETTINGS_DIALOG_CLASS, SETTINGS_DIALOG_SOURCE, "__init__"
        )
        detection_label = '("Detection color channel", self.color_channel)'
        plot_label = '("Linear plot channel", self.linear_plot_channel)'
        self.assertLess(init.index(detection_label), init.index(plot_label))

    def test_interface_setting_selectors_have_distinct_fixed_choices(self):
        init = method_source(
            SETTINGS_DIALOG_CLASS, SETTINGS_DIALOG_SOURCE, "__init__"
        )
        self.assertIn('self.color_channel.addItems(("R", "G", "B"))', init)
        self.assertIn(
            'self.linear_plot_channel.addItems(("R", "G", "B", "RGB"))',
            init,
        )

    def test_interface_dialog_collects_distinct_channel_keys(self):
        source = method_source(
            SETTINGS_DIALOG_CLASS, SETTINGS_DIALOG_SOURCE, "_settings"
        )
        self.assertIn('"color_channel": self.color_channel.currentText()', source)
        self.assertIn(
            '"linear_plot_channel": self.linear_plot_channel.currentText()', source
        )

    def test_setter_redraws_only_with_active_regression(self):
        setter = compile_method(
            "_set_linear_plot_channel_mode",
            {"_LINEAR_PLOT_CHANNEL_MODES": ("R", "G", "B", "RGB")},
        )
        events = []
        harness = types.SimpleNamespace(
            _linear_plot_channel_mode="B",
            _active_linear_regression_set=lambda: object(),
            _plot_regression_result=lambda: events.append(("plot",)),
        )
        setter(harness, "RGB")
        self.assertEqual(events, [("plot",)])
        events.clear()
        harness._active_linear_regression_set = lambda: None
        setter(harness, "G")
        self.assertEqual(events, [])

    def test_plot_mode_never_writes_detection_color_channel_setting(self):
        sources = "\n".join((
            method_source(DETECTMAIN_CLASS, DETECTMAIN_SOURCE, "_set_linear_plot_channel_mode"),
            method_source(DETECTMAIN_CLASS, DETECTMAIN_SOURCE, "_build_linear_plot_projection"),
            method_source(WINDOW_CLASS, WINDOW_SOURCE, "_open_interface_settings"),
        ))
        self.assertNotIn('settings["color_channel"]', sources)
        self.assertNotIn("save_settings", sources)

    def test_interface_color_channel_cannot_write_linear_plot_mode(self):
        assignment_methods = []
        for method in (
            node for node in DETECTMAIN_CLASS.body if isinstance(node, ast.FunctionDef)
        ):
            for node in ast.walk(method):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Attribute) and target.attr == "_linear_plot_channel_mode":
                            assignment_methods.append(method.name)
        self.assertEqual(assignment_methods, ["__init__", "_set_linear_plot_channel_mode"])

    def test_initial_mode_reads_linear_plot_setting_with_b_fallback(self):
        cases = (
            ({"linear_plot_channel": "R"}, "R"),
            ({"linear_plot_channel": "G"}, "G"),
            ({"linear_plot_channel": "B"}, "B"),
            ({"linear_plot_channel": "RGB"}, "RGB"),
            ({}, "B"),
            ({"linear_plot_channel": "invalid"}, "B"),
        )
        for snapshot, expected in cases:
            with self.subTest(snapshot=snapshot):
                restore = compile_startup_mode_restore(
                    lambda apply_to_module=False, value=copy.deepcopy(snapshot): (
                        copy.deepcopy(value),
                        [],
                        None,
                    )
                )
                harness = types.SimpleNamespace()
                self.assertEqual(restore(harness), expected)
                self.assertEqual(harness._linear_plot_channel_mode, expected)

    def test_plot_reads_active_regression_set_directly(self):
        source = method_source(
            DETECTMAIN_CLASS, DETECTMAIN_SOURCE, "_active_linear_regression_set"
        )
        self.assertIn('"_regression_session_state"', source)
        self.assertIn('"active_regression_set"', source)
        self.assertNotIn("_regression_set_from_result", source)

    def test_plot_path_never_reads_regression_result_formulas(self):
        names = (
            "_active_linear_regression_set",
            "_linear_calibration_projection_source",
            "_selected_detection_image_order",
            "_build_linear_plot_projection",
            "_plot_regression_result",
            "_set_linear_plot_channel_mode",
        )
        source = "\n".join(
            method_source(DETECTMAIN_CLASS, DETECTMAIN_SOURCE, name) for name in names
        )
        self.assertNotIn('"formulas"', source)
        self.assertNotIn("_regression_set_from_result", source)

    def test_calibration_revision_gate_rejects_missing_and_stale_samples(self):
        method = compile_method(
            "_linear_calibration_projection_source", {"RegressionSet": RegressionSet}
        )
        active = regression("active")
        stale = regression("stale")
        samples = [{"Con.": 1.0, "Red": 2.0, "Green": 3.0, "Blue": 4.0, "included": True}]
        harness = types.SimpleNamespace(_regression_result={"regression_set": stale, "samples": samples})
        self.assertEqual(method(harness, active), ((), stale.revision))
        harness._regression_result = {"samples": samples}
        self.assertEqual(method(harness, active), ((), None))
        harness._regression_result = {"regression_set": active, "samples": samples}
        self.assertEqual(method(harness, active), (tuple(samples), active.revision))

    def test_projection_builder_receives_only_authoritative_inputs(self):
        captured = {}
        def builder(**kwargs):
            captured.update(kwargs)
            return "projection"
        method = compile_method(
            "_build_linear_plot_projection",
            {"build_linear_plot_projection": builder},
        )
        active = object()
        state = object()
        harness = types.SimpleNamespace(
            _linear_plot_channel_mode="RGB",
            _active_linear_regression_set=lambda: active,
            _linear_calibration_projection_source=lambda value: (("sample",), "revision"),
            _batch_controller=types.SimpleNamespace(state=state),
            _selected_detection_image_order=lambda: 7,
        )
        self.assertEqual(method(harness), "projection")
        self.assertIs(captured["active_regression_set"], active)
        self.assertIs(captured["batch_state"], state)
        self.assertEqual(captured["calibration_samples"], ("sample",))
        self.assertEqual(captured["calibration_revision"], "revision")
        self.assertEqual(captured["plot_mode"], "RGB")
        self.assertEqual(captured["selected_image_order"], 7)

    def test_detection_points_source_is_controller_batch_state(self):
        source = method_source(
            DETECTMAIN_CLASS, DETECTMAIN_SOURCE, "_build_linear_plot_projection"
        )
        self.assertIn("controller.state", source)
        self.assertIn("batch_state=batch_state", source)
        self.assertNotIn("targets", source)

    def test_selected_image_order_comes_only_from_formal_selection_key(self):
        method = compile_method("_selected_detection_image_order")
        controller = types.SimpleNamespace(
            run_token=5,
            state=types.SimpleNamespace(detection_run_token=5),
        )
        harness = types.SimpleNamespace(
            _batch_controller=controller,
            _detection_selected_key=(5, 9, "ignored-path.jpg"),
        )
        self.assertEqual(method(harness), 9)
        harness._detection_selected_key = (4, 9, "same.jpg")
        self.assertIsNone(method(harness))
        source = method_source(
            DETECTMAIN_CLASS, DETECTMAIN_SOURCE, "_selected_detection_image_order"
        )
        self.assertNotIn("current_image", source)
        self.assertNotIn("filename", source)
        self.assertNotIn(".path", source)

    def test_plot_and_setting_redraw_paths_do_not_read_clocks(self):
        names = (
            "_set_linear_plot_channel_mode",
            "_build_linear_plot_projection",
            "_plot_regression_result",
        )
        source = "\n".join(
            method_source(DETECTMAIN_CLASS, DETECTMAIN_SOURCE, name) for name in names
        )
        for forbidden in (
            "time.", "perf_counter", "monotonic", "active_elapsed_ms",
            "_refresh_linear_duration_display", "RegressionTiming",
        ):
            self.assertNotIn(forbidden, source)

    def test_plot_and_setting_redraw_paths_emit_no_worker_signal(self):
        source = "\n".join((
            method_source(DETECTMAIN_CLASS, DETECTMAIN_SOURCE, "_set_linear_plot_channel_mode"),
            method_source(DETECTMAIN_CLASS, DETECTMAIN_SOURCE, "_plot_regression_result"),
        ))
        self.assertNotIn("regression_requested", source)
        self.assertNotIn("detection_requested", source)
        self.assertNotIn("worker", source.lower())

    def test_plot_path_does_not_load_models(self):
        source = "\n".join((
            method_source(DETECTMAIN_CLASS, DETECTMAIN_SOURCE, "_build_linear_plot_projection"),
            method_source(DETECTMAIN_CLASS, DETECTMAIN_SOURCE, "_plot_regression_result"),
        ))
        for forbidden in ("YOLO", "best.pt", "weight", "model", "VideoCapture"):
            self.assertNotIn(forbidden, source)

    def test_plot_path_does_not_call_formula_file_loader(self):
        source = "\n".join((
            method_source(DETECTMAIN_CLASS, DETECTMAIN_SOURCE, "_build_linear_plot_projection"),
            method_source(DETECTMAIN_CLASS, DETECTMAIN_SOURCE, "_plot_regression_result"),
        ))
        self.assertNotIn("_load_saved_regression_set", source)
        self.assertNotIn("_read_excel", source)
        self.assertNotIn("openpyxl", source)

    def test_save_linear_reuses_current_plot_renderer_before_print_png(self):
        source = method_source(
            DETECTMAIN_CLASS, DETECTMAIN_SOURCE, "_build_linear_export_bytes"
        )
        self.assertLess(source.index("self._plot_regression_result()"), source.index("self.canvas.print_png"))
        self.assertEqual(
            method_hash("_build_linear_export_bytes"),
            PROTECTED_METHOD_HASHES["_build_linear_export_bytes"],
        )

    def test_save_transactions_xlsx_callbacks_and_time_functions_are_unchanged(self):
        actual = {name: method_hash(name) for name in PROTECTED_METHOD_HASHES}
        self.assertEqual(actual, PROTECTED_METHOD_HASHES)

    def test_linear_and_detection_xlsx_schema_functions_are_unchanged(self):
        names = (
            "_build_detection_workbook_bytes",
            "_validate_detection_workbook_bytes",
            "_build_linear_workbook_bytes",
            "_validate_linear_workbook_bytes",
        )
        self.assertEqual(
            {name: method_hash(name) for name in names},
            {name: PROTECTED_METHOD_HASHES[name] for name in names},
        )

    def test_both_gui_result_tables_keep_five_columns(self):
        assignment = next(
            node for node in DETECTMAIN_CLASS.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "CALIBRATION_TABLE_HEADERS" for target in node.targets)
        )
        self.assertEqual(
            ast.literal_eval(assignment.value),
            ("No.", "Con.", "Red", "Green", "Blue"),
        )

    def test_shared_time_display_structure_is_unchanged(self):
        names = (
            "_refresh_detection_duration_display",
            "_refresh_linear_duration_display",
            "_set_shared_time_display",
            "_refresh_shared_time_display",
        )
        self.assertEqual(
            {name: method_hash(name) for name in names},
            {name: PROTECTED_METHOD_HASHES[name] for name in names},
        )

    def test_constructor_no_longer_reads_demo_xlsx(self):
        init = method_source(DETECTMAIN_CLASS, DETECTMAIN_SOURCE, "__init__")
        self.assertNotIn("_read_excel", init)
        self.assertNotIn(".xlsx", init)
        self.assertNotIn("linear_regression_table", init)
        self.assertNotIn("detection_table", init)

    def test_constructor_no_longer_builds_demo_regression_set(self):
        init = method_source(DETECTMAIN_CLASS, DETECTMAIN_SOURCE, "__init__")
        self.assertNotIn("RegressionSet.from_formulas", init)
        self.assertNotIn("stats.linregress", init)
        self.assertNotIn("_regression_set_from_result", init)

    def test_constructor_no_longer_calculates_demo_concentrations(self):
        init = method_source(DETECTMAIN_CLASS, DETECTMAIN_SOURCE, "__init__")
        self.assertNotIn("con_pred", init)
        self.assertNotIn("calculate_all", init)
        self.assertNotIn(".calculate(", init)

    def test_ui_and_generated_ui_files_are_unchanged(self):
        actual = {
            name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
            for name in PROTECTED_UI_HASHES
        }
        self.assertEqual(actual, PROTECTED_UI_HASHES)

    def test_phase1c1_core_files_are_unchanged(self):
        actual = {}
        for name in PHASE1C1_IDENTITIES:
            data = Path(name).read_bytes()
            actual[name] = (len(data), hashlib.sha256(data).hexdigest())
        self.assertEqual(actual, PHASE1C1_IDENTITIES)

    def test_new_tests_do_not_import_or_create_qt_gui_objects(self):
        imports = set()
        calls = set()
        strings = set()
        for path in (
            Path("tests/test_phase1c2_linear_plot_renderer.py"),
            Path("tests/test_phase1c2_plot_integration.py"),
            Path("tests/test_phase1c2_repair1_plot_setting_location.py"),
            Path("tests/test_phase1c2_repair5_plot_clarity.py"),
            Path("tests/test_phase1c2_repair6_marker_scale.py"),
        ):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
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
                elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                    strings.add(node.value)
        self.assertNotIn("Py" + "Side6", imports)
        self.assertTrue(calls.isdisjoint({
            "Q" + "Application", "Q" + "Widget", "Q" + "Thread",
            "Detect" + "Main", "YoloDetection" + "Worker",
            "FigureCanvas" + "QTAgg",
        }))
        self.assertNotIn("QT_QPA" + "_PLATFORM", strings)


if __name__ == "__main__":
    unittest.main()
