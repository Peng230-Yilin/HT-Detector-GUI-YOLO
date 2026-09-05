import ast
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GUI_ROOT = PROJECT_ROOT / "Peng1.0_GUI"
DETECTMAIN_SOURCE = GUI_ROOT / "detectmain.py"
DETECTIONWINDOW_SOURCE = GUI_ROOT / "detectionwindow.py"
sys.path.insert(0, str(GUI_ROOT))

import detectmain as detectmain_module  # noqa: E402
from detectmain import DetectMain  # noqa: E402
from linear_series_controller import LinearSeriesController  # noqa: E402
from linear_series_state import LinearSeriesState  # noqa: E402


def _dotted_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _dotted_name(node.value)
        return "{}.{}".format(parent, node.attr) if parent else node.attr
    return None


def _class_node(tree, name):
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError("Class {!r} was not found.".format(name))


def _method_node(class_node, name):
    for node in class_node.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError("Method {!r} was not found.".format(name))


def _called_names(node):
    return {
        _dotted_name(item.func)
        for item in ast.walk(node)
        if isinstance(item, ast.Call) and _dotted_name(item.func) is not None
    }


def _attribute_names(node):
    return {
        _dotted_name(item)
        for item in ast.walk(node)
        if isinstance(item, ast.Attribute) and _dotted_name(item) is not None
    }


class FakeButton:
    def __init__(self, text="", enabled=True):
        self._text = text
        self._enabled = bool(enabled)

    def text(self):
        return self._text

    def setText(self, text):
        self._text = str(text)

    def isEnabled(self):
        return self._enabled

    def setEnabled(self, enabled):
        self._enabled = bool(enabled)


class FakeLabel:
    def __init__(self, text=""):
        self._text = text
        self.clear_count = 0

    def text(self):
        return self._text

    def setText(self, text):
        self._text = str(text)

    def clear(self):
        self._text = ""
        self.clear_count += 1


class FakePixmap:
    def __init__(self, marker, null=False):
        self.marker = marker
        self._null = bool(null)

    def isNull(self):
        return self._null


class FakeSignal:
    def __init__(self):
        self.emissions = []

    def emit(self, *args):
        self.emissions.append(args)


class DetectMainModeHarness:
    _pending_save_type = DetectMain._pending_save_type
    _update_save_button = DetectMain._update_save_button
    _linear_series_has_selection = DetectMain._linear_series_has_selection
    _capture_single_linear_view = DetectMain._capture_single_linear_view
    _restore_single_linear_view = DetectMain._restore_single_linear_view
    _apply_linear_mode_controls = DetectMain._apply_linear_mode_controls
    _set_active_worker_task = DetectMain._set_active_worker_task
    is_worker_task_active = DetectMain.is_worker_task_active
    is_linear_interaction_locked = DetectMain.is_linear_interaction_locked

    def __init__(self):
        self.ui = SimpleNamespace(
            pushButton_4=FakeButton("Linear Regression"),
            pushButton_5=FakeButton("Import Image"),
            pushButton_7=FakeButton("Plot", False),
            pushButton_8=FakeButton("Save", False),
            pushButton=FakeButton("Import Detection"),
            labelOrigImg=FakeLabel("single preview"),
        )
        self._linear_mode = detectmain_module.LINEAR_MODE_SINGLE_IMAGE
        self._single_linear_action_text = "Linear Regression"
        self._single_linear_view_state = None
        self._calibration_source_path = "memory/single.png"
        self._calibration_source_image = object()
        self._active_worker_task = None
        self._close_wait_pending = False
        self._shutdown_requested = False

        self._regression_result = {
            "payload": object(),
            "formulas": {"R": object(), "G": object(), "B": object()},
        }
        self._regression_dirty = True
        self._regression_plot_has_result = True
        self._detection_result = {"detection": object()}
        self._detection_dirty = True
        self._last_completed_result_type = "linear"

        self.origImg = "memory/single-annotated.png"
        self._origPixmap = FakePixmap("single")
        self._linear_series_controller = LinearSeriesController(
            LinearSeriesState(last_confirmed_result=self._regression_result)
        )
        self._linear_series_selection_state = LinearSeriesState(
            last_confirmed_result=self._regression_result
        )
        self._linear_series_weight_path = None

        self._batch_controller = SimpleNamespace(
            state=SimpleNamespace(
                detection_scope="entire_batch",
                numbering_mode="continuous",
                marker=object(),
            )
        )
        self.clear_active_formulas_requested = FakeSignal()
        self.install_saved_formulas_requested = FakeSignal()
        self.restore_active_formulas_requested = FakeSignal()
        self.worker_task_finished = FakeSignal()
        self.busy_changed = FakeSignal()
        self.scaled_labels = []
        self._valid_regression = True
        self._valid_linear_export = True
        self._valid_detection_export = True

        self._set_active_worker_task(None)

    def _scale_label(self, label):
        self.scaled_labels.append(label)

    def _has_valid_regression_result(self):
        return self._valid_regression

    def _has_valid_linear_export(self):
        return self._valid_linear_export

    def _has_valid_detection_export(self):
        return self._valid_detection_export


class LinearSingleImageSourceCompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.detectmain_tree = ast.parse(
            DETECTMAIN_SOURCE.read_text(encoding="utf-8"),
            filename=str(DETECTMAIN_SOURCE),
        )
        cls.detectmain_class = _class_node(cls.detectmain_tree, "DetectMain")
        cls.window_tree = ast.parse(
            DETECTIONWINDOW_SOURCE.read_text(encoding="utf-8"),
            filename=str(DETECTIONWINDOW_SOURCE),
        )
        cls.window_class = _class_node(cls.window_tree, "DetectWindow")

    def test_modes_are_distinct_and_single_is_the_constructor_default(self):
        single = detectmain_module.LINEAR_MODE_SINGLE_IMAGE
        series = detectmain_module.LINEAR_MODE_IMAGE_SERIES
        self.assertIs(type(single), str)
        self.assertIs(type(series), str)
        self.assertNotEqual(single, series)

        initializer = _method_node(self.detectmain_class, "__init__")
        assignments = [
            node
            for node in ast.walk(initializer)
            if isinstance(node, ast.Assign)
            and any(_dotted_name(target) == "self._linear_mode" for target in node.targets)
        ]
        self.assertEqual(len(assignments), 1)
        self.assertEqual(
            _dotted_name(assignments[0].value),
            "LINEAR_MODE_SINGLE_IMAGE",
        )

    def test_single_and_series_buttons_dispatch_to_separate_existing_entries(self):
        select = getattr(DetectMain, "_select_linear_image")
        start = getattr(DetectMain, "_start_linear_action")
        single = detectmain_module.LINEAR_MODE_SINGLE_IMAGE
        series = detectmain_module.LINEAR_MODE_IMAGE_SERIES

        for mode, expected in ((single, "single"), (series, "series")):
            with self.subTest(entry="select", mode=mode):
                harness = SimpleNamespace(
                    _linear_mode=mode,
                    _select_calibration_image=Mock(),
                    _select_linear_series_images=Mock(),
                )
                select(harness)
                self.assertEqual(
                    harness._select_calibration_image.call_count,
                    int(expected == "single"),
                )
                self.assertEqual(
                    harness._select_linear_series_images.call_count,
                    int(expected == "series"),
                )

            with self.subTest(entry="start", mode=mode):
                harness = SimpleNamespace(
                    _linear_mode=mode,
                    _start_linear_regression=Mock(),
                    _start_linear_series=Mock(),
                )
                start(harness)
                self.assertEqual(
                    harness._start_linear_regression.call_count,
                    int(expected == "single"),
                )
                self.assertEqual(
                    harness._start_linear_series.call_count,
                    int(expected == "series"),
                )

    def test_single_keeps_singular_dialog_and_series_owns_plural_dialog(self):
        single_method = _method_node(
            self.detectmain_class, "_select_calibration_image"
        )
        series_method = _method_node(
            self.detectmain_class, "_select_linear_series_images"
        )
        wrapper = _method_node(self.detectmain_class, "_select_linear_image")

        single_calls = _called_names(single_method)
        series_calls = _called_names(series_method)
        wrapper_calls = _called_names(wrapper)
        self.assertIn("QFileDialog.getOpenFileName", single_calls)
        self.assertNotIn("QFileDialog.getOpenFileNames", single_calls)
        self.assertIn("QFileDialog.getOpenFileNames", series_calls)
        self.assertNotIn("QFileDialog.getOpenFileName", series_calls)
        self.assertIn("self._select_calibration_image", wrapper_calls)
        self.assertIn("self._select_linear_series_images", wrapper_calls)

    def test_legacy_regression_signal_worker_and_formula_wiring_stays_intact(self):
        class_assignments = {
            target.id: node.value
            for node in self.detectmain_class.body
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        regression_signal = class_assignments["regression_requested"]
        self.assertIsInstance(regression_signal, ast.Call)
        self.assertEqual(_dotted_name(regression_signal.func), "Signal")
        self.assertEqual(
            tuple(_dotted_name(argument) for argument in regression_signal.args),
            ("str", "str"),
        )
        for name in (
            "clear_active_formulas_requested",
            "install_saved_formulas_requested",
            "restore_active_formulas_requested",
        ):
            self.assertIn(name, class_assignments)

        initializer = _method_node(self.detectmain_class, "__init__")
        connections = {
            (_dotted_name(call.func.value), _dotted_name(call.args[0]))
            for call in ast.walk(initializer)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "connect"
            and len(call.args) == 1
        }
        self.assertIn(
            ("self.regression_requested", "self._detection_worker.regress"),
            connections,
        )
        self.assertIn(
            ("self.ui.pushButton_5.clicked", "self._select_linear_image"),
            connections,
        )
        self.assertIn(
            ("self.ui.pushButton_4.clicked", "self._start_linear_action"),
            connections,
        )
        for signal_name, worker_name in (
            ("clear_active_formulas_requested", "clear_active_formulas"),
            ("install_saved_formulas_requested", "install_saved_formulas"),
            ("restore_active_formulas_requested", "restore_active_formulas"),
        ):
            self.assertIn(
                (
                    "self.{}".format(signal_name),
                    "self._detection_worker.{}".format(worker_name),
                ),
                connections,
            )

        old_start = _method_node(self.detectmain_class, "_start_linear_regression")
        emissions = [
            call
            for call in ast.walk(old_start)
            if isinstance(call, ast.Call)
            and _dotted_name(call.func) == "self.regression_requested.emit"
        ]
        self.assertEqual(len(emissions), 1)
        self.assertEqual(len(emissions[0].args), 2)

    def test_mode_switch_preserves_confirmed_payload_formulas_and_detection(self):
        harness = DetectMainModeHarness()
        regression_result = harness._regression_result
        formulas = regression_result["formulas"]
        detection_result = harness._detection_result
        batch_controller = harness._batch_controller
        batch_state = batch_controller.state
        original_image = harness.origImg
        original_pixmap = harness._origPixmap
        original_label = harness.ui.labelOrigImg.text()

        returned = DetectMain.set_linear_mode(
            harness, detectmain_module.LINEAR_MODE_IMAGE_SERIES
        )

        self.assertEqual(returned, detectmain_module.LINEAR_MODE_IMAGE_SERIES)
        self.assertIs(harness._regression_result, regression_result)
        self.assertIs(harness._regression_result["formulas"], formulas)
        self.assertTrue(harness._regression_dirty)
        self.assertTrue(harness._regression_plot_has_result)
        self.assertEqual(harness._last_completed_result_type, "linear")
        self.assertIs(harness._detection_result, detection_result)
        self.assertTrue(harness._detection_dirty)
        self.assertIs(harness._batch_controller, batch_controller)
        self.assertIs(harness._batch_controller.state, batch_state)
        self.assertEqual(batch_state.detection_scope, "entire_batch")
        self.assertEqual(batch_state.numbering_mode, "continuous")
        self.assertEqual(harness.clear_active_formulas_requested.emissions, [])
        self.assertEqual(harness.install_saved_formulas_requested.emissions, [])
        self.assertEqual(harness.restore_active_formulas_requested.emissions, [])
        self.assertEqual(harness.ui.pushButton_4.text(), "Extract Series")
        self.assertFalse(harness.ui.pushButton_7.isEnabled())
        self.assertFalse(harness.ui.pushButton_8.isEnabled())
        self.assertTrue(harness.ui.pushButton.isEnabled())

        harness.origImg = "memory/series-preview.png"
        harness._origPixmap = FakePixmap("series")
        harness.ui.labelOrigImg.setText("series preview")
        series_controller = harness._linear_series_controller
        returned = DetectMain.set_linear_mode(
            harness, detectmain_module.LINEAR_MODE_SINGLE_IMAGE
        )

        self.assertEqual(returned, detectmain_module.LINEAR_MODE_SINGLE_IMAGE)
        self.assertIs(harness._linear_series_controller, series_controller)
        self.assertIs(
            harness._linear_series_controller.last_confirmed_result,
            regression_result,
        )
        self.assertEqual(harness._linear_series_controller.state.images, ())
        self.assertIs(harness._regression_result, regression_result)
        self.assertIs(harness._detection_result, detection_result)
        self.assertIs(harness._batch_controller, batch_controller)
        self.assertEqual(harness.origImg, original_image)
        self.assertIs(harness._origPixmap, original_pixmap)
        self.assertEqual(harness.ui.labelOrigImg.text(), original_label)
        self.assertEqual(harness.ui.pushButton_4.text(), "Linear Regression")
        self.assertTrue(harness.ui.pushButton_4.isEnabled())
        self.assertTrue(harness.ui.pushButton_7.isEnabled())
        self.assertTrue(harness.ui.pushButton_8.isEnabled())
        self.assertTrue(harness.ui.pushButton.isEnabled())

    def test_detect_main_mode_failure_rolls_back_all_runtime_and_controls(self):
        harness = DetectMainModeHarness()
        DetectMain.set_linear_mode(
            harness, detectmain_module.LINEAR_MODE_IMAGE_SERIES
        )
        harness.origImg = "memory/series-preview.png"
        harness._origPixmap = FakePixmap("series")
        harness.ui.labelOrigImg.setText("series preview")
        harness._linear_series_weight_path = "memory/series-weight.pt"

        controller = harness._linear_series_controller
        single_view = harness._single_linear_view_state
        runtime_snapshot = (
            harness._linear_mode,
            harness.origImg,
            harness._origPixmap,
            harness.ui.labelOrigImg.text(),
            harness._linear_series_weight_path,
        )
        controls = tuple(
            (button.text(), button.isEnabled())
            for button in (
                harness.ui.pushButton_4,
                harness.ui.pushButton_5,
                harness.ui.pushButton_7,
                harness.ui.pushButton_8,
                harness.ui.pushButton,
            )
        )

        def fail_after_mutating_controls():
            harness.ui.pushButton_4.setText("partial mode")
            harness.ui.pushButton_5.setEnabled(False)
            harness.ui.pushButton_7.setEnabled(True)
            harness.ui.pushButton_8.setEnabled(True)
            harness.ui.pushButton.setEnabled(False)
            raise RuntimeError("synthetic mode apply failure")

        harness._apply_linear_mode_controls = fail_after_mutating_controls
        with self.assertRaisesRegex(RuntimeError, "synthetic mode apply failure"):
            DetectMain.set_linear_mode(
                harness, detectmain_module.LINEAR_MODE_SINGLE_IMAGE
            )

        self.assertIs(harness._linear_series_controller, controller)
        self.assertIs(harness._single_linear_view_state, single_view)
        self.assertEqual((
            harness._linear_mode,
            harness.origImg,
            harness._origPixmap,
            harness.ui.labelOrigImg.text(),
            harness._linear_series_weight_path,
        ), runtime_snapshot)
        self.assertEqual(tuple(
            (button.text(), button.isEnabled())
            for button in (
                harness.ui.pushButton_4,
                harness.ui.pushButton_5,
                harness.ui.pushButton_7,
                harness.ui.pushButton_8,
                harness.ui.pushButton,
            )
        ), controls)

    def test_series_busy_and_idle_controls_never_expose_single_plot_or_save(self):
        harness = DetectMainModeHarness()
        DetectMain.set_linear_mode(
            harness, detectmain_module.LINEAR_MODE_IMAGE_SERIES
        )
        harness._linear_series_controller = LinearSeriesController(
            LinearSeriesState.from_paths(
                ["memory/image1.png"],
                last_confirmed_result=harness._regression_result,
            )
        )
        harness._linear_series_selection_state = LinearSeriesState.from_paths(
            ["memory/image1.png"],
            last_confirmed_result=harness._regression_result,
        )

        harness._set_active_worker_task(None)
        self.assertTrue(harness.ui.pushButton_4.isEnabled())
        self.assertFalse(harness.ui.pushButton_7.isEnabled())
        self.assertFalse(harness.ui.pushButton_8.isEnabled())

        harness._set_active_worker_task("linear_series")
        self.assertTrue(harness.is_worker_task_active())
        for button in (
            harness.ui.pushButton_4,
            harness.ui.pushButton_5,
            harness.ui.pushButton_7,
            harness.ui.pushButton_8,
            harness.ui.pushButton,
        ):
            self.assertFalse(button.isEnabled())
        self.assertEqual(harness.busy_changed.emissions[-1], (True,))

        harness._set_active_worker_task(None)
        self.assertFalse(harness.is_worker_task_active())
        self.assertTrue(harness.ui.pushButton_4.isEnabled())
        self.assertFalse(harness.ui.pushButton_7.isEnabled())
        self.assertFalse(harness.ui.pushButton_8.isEnabled())
        self.assertEqual(harness.worker_task_finished.emissions[-1], ())
        self.assertEqual(harness.busy_changed.emissions[-1], (False,))

    def test_mode_setter_rejects_every_interaction_lock_without_mutation(self):
        for lock_state in (
            "detection",
            "regression",
            "linear_series",
            "save_linear",
            "save_detection",
            "pending_close",
            "shutdown",
        ):
            with self.subTest(lock_state=lock_state):
                harness = DetectMainModeHarness()
                if lock_state == "pending_close":
                    harness._close_wait_pending = True
                elif lock_state == "shutdown":
                    harness._shutdown_requested = True
                else:
                    harness._active_worker_task = lock_state
                snapshot = (
                    harness._linear_mode,
                    harness._linear_series_controller,
                    harness._linear_series_selection_state,
                    harness._single_linear_view_state,
                    harness.origImg,
                    harness._origPixmap,
                    harness.ui.labelOrigImg.text(),
                    tuple(
                        (button.text(), button.isEnabled())
                        for button in (
                            harness.ui.pushButton_4,
                            harness.ui.pushButton_5,
                            harness.ui.pushButton_7,
                            harness.ui.pushButton_8,
                            harness.ui.pushButton,
                        )
                    ),
                )
                with self.assertRaisesRegex(RuntimeError, "cannot be changed"):
                    DetectMain.set_linear_mode(
                        harness, detectmain_module.LINEAR_MODE_IMAGE_SERIES
                    )
                self.assertEqual((
                    harness._linear_mode,
                    harness._linear_series_controller,
                    harness._linear_series_selection_state,
                    harness._single_linear_view_state,
                    harness.origImg,
                    harness._origPixmap,
                    harness.ui.labelOrigImg.text(),
                    tuple(
                        (button.text(), button.isEnabled())
                        for button in (
                            harness.ui.pushButton_4,
                            harness.ui.pushButton_5,
                            harness.ui.pushButton_7,
                            harness.ui.pushButton_8,
                            harness.ui.pushButton,
                        )
                    ),
                ), snapshot)

    def test_plot_entry_is_inert_in_series_and_single_save_state_restores(self):
        plot_data = Mock(side_effect=AssertionError("series must not plot"))
        series_harness = SimpleNamespace(
            _linear_mode=detectmain_module.LINEAR_MODE_IMAGE_SERIES,
            _regression_plot_data=plot_data,
        )
        DetectMain._plot_regression_result(series_harness)
        plot_data.assert_not_called()

        harness = DetectMainModeHarness()
        harness._update_save_button()
        self.assertEqual(harness.ui.pushButton_8.text(), "Save Linear")
        self.assertTrue(harness.ui.pushButton_8.isEnabled())
        DetectMain.set_linear_mode(
            harness, detectmain_module.LINEAR_MODE_IMAGE_SERIES
        )
        self.assertEqual(harness.ui.pushButton_8.text(), "Save Linear")
        self.assertFalse(harness.ui.pushButton_8.isEnabled())
        DetectMain.set_linear_mode(
            harness, detectmain_module.LINEAR_MODE_SINGLE_IMAGE
        )
        self.assertEqual(harness.ui.pushButton_8.text(), "Save Linear")
        self.assertTrue(harness.ui.pushButton_8.isEnabled())

    def test_linear_menu_setup_is_isolated_from_detection_groups_and_settings(self):
        initializer = _method_node(self.window_class, "__init__")
        initializer_calls = _called_names(initializer)
        self.assertIn("self._setup_detection_menu", initializer_calls)
        self.assertIn("self._setup_linear_menu", initializer_calls)

        detection_setup = _method_node(self.window_class, "_setup_detection_menu")
        linear_setup = _method_node(self.window_class, "_setup_linear_menu")
        detection_attributes = _attribute_names(detection_setup)
        linear_attributes = _attribute_names(linear_setup)
        linear_calls = _called_names(linear_setup)

        self.assertIn("self._detection_scope_group", detection_attributes)
        self.assertIn("self._numbering_group", detection_attributes)
        self.assertNotIn("self._linear_mode_group", detection_attributes)
        self.assertIn("self._linear_mode_group", linear_attributes)
        self.assertNotIn("self._detection_scope_group", linear_attributes)
        self.assertNotIn("self._numbering_group", linear_attributes)
        for forbidden in (
            "load_detection_preferences",
            "save_detection_preferences",
            "self._save_detection_menu_settings",
            "self._detectMain.set_detection_options",
        ):
            self.assertNotIn(forbidden, linear_calls)


if __name__ == "__main__":
    unittest.main()
