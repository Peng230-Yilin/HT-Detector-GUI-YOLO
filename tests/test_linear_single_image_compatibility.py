import ast
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np
from PySide6.QtGui import QColor, QPixmap, QStandardItem, QStandardItemModel
from PySide6.QtWidgets import (
    QApplication,
    QLabel,
    QLCDNumber,
    QProgressBar,
    QPushButton,
    QTabWidget,
    QTableView,
    QWidget,
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure


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
        self.pixmap = None

    def text(self):
        return self._text

    def setText(self, text):
        self._text = str(text)

    def clear(self):
        self._text = ""
        self.pixmap = None
        self.clear_count += 1

    def setPixmap(self, pixmap):
        self.pixmap = pixmap


class FakeTableView:
    def __init__(self, model=None):
        self._model = model

    def model(self):
        return self._model

    def setModel(self, model):
        self._model = model


class FakeValueWidget:
    def __init__(self, value=0, minimum=0, maximum=100):
        self._value = value
        self._minimum = minimum
        self._maximum = maximum

    def value(self):
        return self._value

    def display(self, value):
        self._value = value

    def setValue(self, value):
        self._value = value

    def setRange(self, minimum, maximum):
        self._minimum = minimum
        self._maximum = maximum

    def minimum(self):
        return self._minimum

    def maximum(self):
        return self._maximum


class FakeTabWidget:
    def __init__(self, text="Detection Image"):
        self._text = text

    def indexOf(self, _widget):
        return 0

    def tabText(self, _index):
        return self._text

    def setTabText(self, _index, text):
        self._text = str(text)


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
    _capture_result_pane_view = DetectMain._capture_result_pane_view
    _apply_result_pane_view = DetectMain._apply_result_pane_view
    _empty_linear_series_result_view = DetectMain._empty_linear_series_result_view
    _publish_detection_result_view = DetectMain._publish_detection_result_view
    _apply_linear_mode_controls = DetectMain._apply_linear_mode_controls
    _set_active_worker_task = DetectMain._set_active_worker_task
    is_worker_task_active = DetectMain.is_worker_task_active
    is_linear_interaction_locked = DetectMain.is_linear_interaction_locked

    def __init__(self):
        self.single_table_model = object()
        self.detection_table_model = object()
        self.ui = SimpleNamespace(
            pushButton_4=FakeButton("Linear Regression"),
            pushButton_5=FakeButton("Import Image"),
            pushButton_7=FakeButton("Plot", False),
            pushButton_8=FakeButton("Save", False),
            pushButton=FakeButton("Import Detection"),
            labelOrigImg=FakeLabel("single preview"),
            labelRecgImg=FakeLabel(""),
            label_4=FakeLabel("Table. Detection"),
            tabviewOrig=FakeTableView(self.single_table_model),
            tabviewRecg=FakeTableView(self.detection_table_model),
            tabWidget=FakeTabWidget(),
            tab_3=object(),
            progressBar=FakeValueWidget(37, 0, 100),
            lcdNumber=FakeValueWidget(246),
        )
        self._linear_mode = detectmain_module.LINEAR_MODE_SINGLE_IMAGE
        self._single_linear_action_text = "Linear Regression"
        self._single_linear_view_state = None
        self._detection_view_state = None
        self._linear_series_result_view_state = None
        self._calibration_source_path = "memory/single.png"
        self._calibration_source_image = object()
        self._last_calibration_directory = "memory"
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
        self._recgPixmap = FakePixmap("detection")
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
        self.plot_restore_calls = 0

        self._set_active_worker_task(None)

    def _scale_label(self, label):
        self.scaled_labels.append(label)

    def _has_valid_regression_result(self):
        return self._valid_regression

    def _has_valid_linear_export(self):
        return self._valid_linear_export

    def _has_valid_detection_export(self):
        return self._valid_detection_export

    def _build_table_model(self, headers, rows):
        return DetectMain._build_table_model(headers, rows)

    def _plot_regression_result(self):
        self.plot_restore_calls += 1
        self._regression_plot_has_result = True

    def _show_calibration_plot_placeholder(self):
        self._regression_plot_has_result = False


class RealQtModeHarness(DetectMainModeHarness):
    REGRESSION_CHANNEL_FIELDS = DetectMain.REGRESSION_CHANNEL_FIELDS
    CHANNEL_COLORS = DetectMain.CHANNEL_COLORS
    CALIBRATION_PLOT_PLACEHOLDER = DetectMain.CALIBRATION_PLOT_PLACEHOLDER
    _regression_plot_data = DetectMain._regression_plot_data
    _plot_regression_result = DetectMain._plot_regression_result
    _show_calibration_plot_placeholder = (
        DetectMain._show_calibration_plot_placeholder
    )

    def __init__(self):
        super().__init__()
        tab_widget = QTabWidget()
        tab_page = QWidget()
        tab_widget.addTab(tab_page, "Detection Image")
        self.ui = SimpleNamespace(
            pushButton_4=QPushButton("Linear Regression"),
            pushButton_5=QPushButton("Import Image"),
            pushButton_7=QPushButton("Plot"),
            pushButton_8=QPushButton("Save"),
            pushButton=QPushButton("Import Detection"),
            labelOrigImg=QLabel("single preview"),
            labelRecgImg=QLabel(),
            label_4=QLabel("Table. Detection"),
            tabviewOrig=QTableView(),
            tabviewRecg=QTableView(),
            tabWidget=tab_widget,
            tab_3=tab_page,
            progressBar=QProgressBar(),
            lcdNumber=QLCDNumber(),
        )
        self.figure = Figure()
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.ax = self.figure.add_subplot(111)
        self._regression_suptitle = None
        self._origPixmap = QPixmap(4, 4)
        self._origPixmap.fill(QColor("red"))
        self._recgPixmap = QPixmap(4, 4)
        self._recgPixmap.fill(QColor("blue"))
        self.ui.progressBar.setRange(7, 89)
        self.ui.progressBar.setValue(37)
        self.ui.lcdNumber.display(246)
        self._set_active_worker_task(None)

    def _scale_label(self, label):
        pixmap = (
            self._origPixmap
            if label is self.ui.labelOrigImg
            else self._recgPixmap
        )
        if pixmap is not None and not pixmap.isNull():
            label.setPixmap(pixmap)


def real_regression_payload(channel):
    formulas = {
        "R": {"slope": 2.0, "intercept": 0.5, "R2": 0.99},
        "G": {"slope": 3.0, "intercept": 0.5, "R2": 0.98},
        "B": {"slope": 4.0, "intercept": 0.5, "R2": 0.97},
    }
    return {
        "samples": [
            {
                "Con.": 1.0,
                "Red": 10.0,
                "Green": 20.0,
                "Blue": 30.0,
                "included": True,
            },
            {
                "Con.": 2.0,
                "Red": 12.0,
                "Green": 23.0,
                "Blue": 34.0,
                "included": True,
            },
        ],
        "formulas": formulas,
        "selected_channel": channel,
    }


def real_table_model(rows):
    model = QStandardItemModel()
    for row in rows:
        model.appendRow([QStandardItem(str(value)) for value in row])
    return model


def model_cells(model):
    return tuple(
        tuple(model.item(row, column).text() for column in range(model.columnCount()))
        for row in range(model.rowCount())
    )


def pixmap_pixel(pixmap):
    return pixmap.toImage().pixelColor(0, 0).getRgb()


class LinearSingleImageSourceCompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
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
        original_table = harness.ui.tabviewOrig.model()
        original_detection_pixmap = harness._recgPixmap
        original_detection_table = harness.ui.tabviewRecg.model()
        original_progress = (
            harness.ui.progressBar.minimum(),
            harness.ui.progressBar.maximum(),
            harness.ui.progressBar.value(),
        )
        original_elapsed = harness.ui.lcdNumber.value()

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
        self.assertIsNone(harness._recgPixmap)
        self.assertEqual(harness.ui.tabviewRecg.model().rowCount(), 0)
        self.assertEqual(harness.ui.tabWidget.tabText(0), "Linear Series Image")
        self.assertEqual(harness.ui.label_4.text(), "Table. Linear Series")

        harness.origImg = "memory/series-preview.png"
        harness._origPixmap = FakePixmap("series")
        harness.ui.labelOrigImg.setText("series preview")
        harness.ui.tabviewOrig.setModel(object())
        harness._regression_plot_has_result = False
        harness.ui.progressBar.setRange(0, 0)
        harness.ui.progressBar.setValue(0)
        harness.ui.lcdNumber.display(999)
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
        self.assertIs(harness.ui.tabviewOrig.model(), original_table)
        self.assertTrue(harness._regression_plot_has_result)
        self.assertEqual(harness.plot_restore_calls, 1)
        self.assertEqual((
            harness.ui.progressBar.minimum(),
            harness.ui.progressBar.maximum(),
            harness.ui.progressBar.value(),
        ), original_progress)
        self.assertEqual(harness.ui.lcdNumber.value(), original_elapsed)
        self.assertIs(harness._recgPixmap, original_detection_pixmap)
        self.assertIs(harness.ui.tabviewRecg.model(), original_detection_table)
        self.assertEqual(harness.ui.tabWidget.tabText(0), "Detection Image")
        self.assertEqual(harness.ui.label_4.text(), "Table. Detection")
        self.assertEqual(harness._last_completed_result_type, "linear")
        self.assertTrue(harness._regression_dirty)
        self.assertIs(harness._regression_result["formulas"], formulas)
        self.assertEqual(harness.clear_active_formulas_requested.emissions, [])
        self.assertEqual(harness.install_saved_formulas_requested.emissions, [])
        self.assertEqual(harness.restore_active_formulas_requested.emissions, [])
        self.assertEqual(harness.ui.pushButton_4.text(), "Linear Regression")
        self.assertTrue(harness.ui.pushButton_4.isEnabled())
        self.assertTrue(harness.ui.pushButton_7.isEnabled())
        self.assertTrue(harness.ui.pushButton_8.isEnabled())
        self.assertTrue(harness.ui.pushButton.isEnabled())

    def test_real_qt_plot_and_single_snapshot_survive_two_round_trips_per_channel(self):
        for channel in ("R", "G", "B"):
            with self.subTest(channel=channel):
                harness = RealQtModeHarness()
                payload = real_regression_payload(channel)
                formulas = payload["formulas"]
                table_model = real_table_model((
                    ("1", "0", "10.00", "20.00", "30.00"),
                    ("2", "1", "12.00", "23.00", "34.00"),
                ))
                detection_model = real_table_model((
                    ("9", "0.5", "90.00", "91.00", "92.00"),
                ))
                source_image = np.full((3, 4, 3), 17, dtype=np.uint8)
                harness._regression_result = payload
                harness._linear_series_controller.remember_confirmed_result(payload)
                harness._linear_series_selection_state = LinearSeriesState(
                    last_confirmed_result=payload
                )
                harness._regression_dirty = True
                harness._last_completed_result_type = "linear"
                harness._calibration_source_path = "memory/{}.png".format(channel)
                harness._calibration_source_image = source_image
                harness._last_calibration_directory = "memory/{}".format(channel)
                harness.origImg = source_image
                harness.ui.tabviewOrig.setModel(table_model)
                harness.ui.tabviewRecg.setModel(detection_model)
                harness._detection_view_state = harness._capture_result_pane_view()
                harness._regression_plot_has_result = False
                harness._plot_regression_result()
                harness._scale_label(harness.ui.labelOrigImg)
                harness._scale_label(harness.ui.labelRecgImg)

                formula = formulas[channel]
                title = harness._regression_suptitle.get_text()
                self.assertIn("{:.4f}".format(formula["slope"]), title)
                self.assertIn("{:.4f}".format(formula["intercept"]), title)
                self.assertIn("{:.4f}".format(formula["R2"]), title)
                baseline_line = np.array(
                    harness.ax.lines[0].get_xydata(), copy=True
                )
                baseline_scatter = np.array(
                    harness.ax.collections[0].get_offsets(), copy=True
                )
                baseline_cells = model_cells(table_model)
                baseline_orig_pixel = pixmap_pixel(harness._origPixmap)
                baseline_label_pixel = pixmap_pixel(
                    harness.ui.labelOrigImg.pixmap()
                )
                baseline_progress = (
                    harness.ui.progressBar.minimum(),
                    harness.ui.progressBar.maximum(),
                    harness.ui.progressBar.value(),
                )
                baseline_elapsed = harness.ui.lcdNumber.value()
                baseline_buttons = tuple(
                    (button.text(), button.isEnabled())
                    for button in (
                        harness.ui.pushButton_4,
                        harness.ui.pushButton_5,
                        harness.ui.pushButton_7,
                        harness.ui.pushButton_8,
                        harness.ui.pushButton,
                    )
                )

                for cycle in range(2):
                    DetectMain.set_linear_mode(
                        harness, detectmain_module.LINEAR_MODE_IMAGE_SERIES
                    )
                    harness.origImg = np.full(
                        (3, 4, 3), cycle + 40, dtype=np.uint8
                    )
                    harness._origPixmap = QPixmap(4, 4)
                    harness._origPixmap.fill(QColor("green"))
                    harness.ui.labelOrigImg.setText("series {}".format(cycle))
                    harness.ui.tabviewOrig.setModel(
                        real_table_model((("series", cycle),))
                    )
                    harness.ui.progressBar.setRange(0, 0)
                    harness.ui.progressBar.setValue(0)
                    harness.ui.lcdNumber.display(999)

                    DetectMain.set_linear_mode(
                        harness, detectmain_module.LINEAR_MODE_SINGLE_IMAGE
                    )

                    self.assertIs(harness._regression_result, payload)
                    self.assertIs(harness._regression_result["formulas"], formulas)
                    self.assertEqual(
                        harness._regression_result["selected_channel"], channel
                    )
                    self.assertTrue(harness._regression_dirty)
                    self.assertEqual(harness._last_completed_result_type, "linear")
                    self.assertEqual(
                        harness._calibration_source_path,
                        "memory/{}.png".format(channel),
                    )
                    self.assertIs(harness._calibration_source_image, source_image)
                    self.assertEqual(
                        harness._last_calibration_directory,
                        "memory/{}".format(channel),
                    )
                    self.assertIs(harness.origImg, source_image)
                    self.assertIs(harness.ui.tabviewOrig.model(), table_model)
                    self.assertEqual(model_cells(table_model), baseline_cells)
                    self.assertEqual(
                        pixmap_pixel(harness._origPixmap), baseline_orig_pixel
                    )
                    self.assertEqual(
                        pixmap_pixel(harness.ui.labelOrigImg.pixmap()),
                        baseline_label_pixel,
                    )
                    self.assertEqual(
                        harness._regression_suptitle.get_text(), title
                    )
                    np.testing.assert_array_equal(
                        harness.ax.lines[0].get_xydata(), baseline_line
                    )
                    np.testing.assert_array_equal(
                        harness.ax.collections[0].get_offsets(),
                        baseline_scatter,
                    )
                    self.assertEqual((
                        harness.ui.progressBar.minimum(),
                        harness.ui.progressBar.maximum(),
                        harness.ui.progressBar.value(),
                    ), baseline_progress)
                    self.assertEqual(harness.ui.lcdNumber.value(), baseline_elapsed)
                    self.assertEqual(tuple(
                        (button.text(), button.isEnabled())
                        for button in (
                            harness.ui.pushButton_4,
                            harness.ui.pushButton_5,
                            harness.ui.pushButton_7,
                            harness.ui.pushButton_8,
                            harness.ui.pushButton,
                        )
                    ), baseline_buttons)

    def test_real_qt_no_plot_snapshot_does_not_create_plot(self):
        harness = RealQtModeHarness()
        harness._show_calibration_plot_placeholder()
        placeholder_text = tuple(text.get_text() for text in harness.ax.texts)
        self.assertFalse(harness._regression_plot_has_result)
        self.assertEqual(len(harness.ax.lines), 0)
        self.assertEqual(len(harness.ax.collections), 0)
        self.assertIsNone(harness._regression_suptitle)

        DetectMain.set_linear_mode(
            harness, detectmain_module.LINEAR_MODE_IMAGE_SERIES
        )
        DetectMain.set_linear_mode(
            harness, detectmain_module.LINEAR_MODE_SINGLE_IMAGE
        )

        self.assertFalse(harness._regression_plot_has_result)
        self.assertEqual(len(harness.ax.lines), 0)
        self.assertEqual(len(harness.ax.collections), 0)
        self.assertIsNone(harness._regression_suptitle)
        self.assertEqual(
            tuple(text.get_text() for text in harness.ax.texts),
            placeholder_text,
        )

    def test_real_qt_detection_completed_in_series_is_visible_on_return(self):
        harness = RealQtModeHarness()
        old_detection_result = harness._detection_result
        old_detection_model = real_table_model((("old", "detection"),))
        harness.ui.tabviewRecg.setModel(old_detection_model)
        harness._detection_view_state = harness._capture_result_pane_view()
        DetectMain.set_linear_mode(
            harness, detectmain_module.LINEAR_MODE_IMAGE_SERIES
        )
        visible_series_model = harness.ui.tabviewRecg.model()
        new_detection_result = {"detection": "new"}
        new_detection_model = real_table_model((("new", "detection"),))
        new_detection_pixmap = QPixmap(4, 4)
        new_detection_pixmap.fill(QColor("green"))
        harness._detection_result = new_detection_result
        harness._detection_dirty = True
        harness._last_completed_result_type = "detection"

        harness._publish_detection_result_view(
            new_detection_pixmap, new_detection_model
        )

        self.assertIsNot(harness._detection_result, old_detection_result)
        self.assertIs(harness.ui.tabviewRecg.model(), visible_series_model)
        self.assertIsNone(harness._recgPixmap)
        DetectMain.set_linear_mode(
            harness, detectmain_module.LINEAR_MODE_SINGLE_IMAGE
        )
        self.assertIs(harness._detection_result, new_detection_result)
        self.assertIs(harness.ui.tabviewRecg.model(), new_detection_model)
        self.assertEqual(
            model_cells(harness.ui.tabviewRecg.model()),
            (("new", "detection"),),
        )
        self.assertEqual(pixmap_pixel(harness._recgPixmap), (0, 128, 0, 255))
        self.assertEqual(
            pixmap_pixel(harness.ui.labelRecgImg.pixmap()),
            (0, 128, 0, 255),
        )
        self.assertEqual(harness._last_completed_result_type, "detection")

    def test_series_mode_keeps_new_detection_view_hidden_until_single_returns(self):
        harness = DetectMainModeHarness()
        DetectMain.set_linear_mode(
            harness, detectmain_module.LINEAR_MODE_IMAGE_SERIES
        )
        visible_series_model = harness.ui.tabviewRecg.model()
        new_pixmap = FakePixmap("new detection")
        new_model = object()
        new_result = {"detection": "new"}
        harness._detection_result = new_result
        harness._detection_dirty = True
        harness._last_completed_result_type = "detection"

        harness._publish_detection_result_view(new_pixmap, new_model)

        self.assertIsNone(harness._recgPixmap)
        self.assertIs(harness.ui.tabviewRecg.model(), visible_series_model)
        self.assertEqual(harness.ui.tabWidget.tabText(0), "Linear Series Image")
        DetectMain.set_linear_mode(
            harness, detectmain_module.LINEAR_MODE_SINGLE_IMAGE
        )
        self.assertIs(harness._detection_result, new_result)
        self.assertIs(harness._recgPixmap, new_pixmap)
        self.assertIs(harness.ui.tabviewRecg.model(), new_model)
        self.assertEqual(harness._last_completed_result_type, "detection")

    def test_detection_completion_in_series_updates_only_hidden_detection_view(self):
        from tests.test_batch_detection_flow import (
            DetectionHandlerHarness,
            runtime_payload,
        )

        harness = DetectionHandlerHarness()
        harness._linear_mode = detectmain_module.LINEAR_MODE_IMAGE_SERIES
        harness._detection_view_state = None
        visible_pixmap = FakePixmap("series result")
        visible_model = object()
        harness._recgPixmap = visible_pixmap
        harness.ui.tabviewRecg.setModel(visible_model)
        harness._batch_controller.set_options("entire_batch", "per_image")
        harness._batch_controller.replace_images(["memory/detection.png"])
        harness._active_worker_task = "detection"
        task = harness._batch_controller.begin()

        DetectMain._on_detection_finished(harness, runtime_payload(task))

        self.assertIs(harness._recgPixmap, visible_pixmap)
        self.assertIs(harness.ui.tabviewRecg.model(), visible_model)
        self.assertIsNotNone(harness._detection_view_state)
        self.assertIsNot(
            harness._detection_view_state.pixmap, visible_pixmap
        )
        self.assertEqual(harness._detection_view_state.model[1][0][0], 1)
        self.assertEqual(harness._detection_view_state.model[1][0][1], 0.5)
        self.assertEqual(
            harness._detection_view_state.tab_text, "Detection Image"
        )
        self.assertEqual(harness._last_completed_result_type, "detection")
        self.assertTrue(harness._detection_dirty)

    def test_single_round_trip_preserves_each_regression_channel(self):
        for channel in ("R", "G", "B"):
            with self.subTest(channel=channel):
                harness = DetectMainModeHarness()
                harness._regression_result["selected_channel"] = channel
                result = harness._regression_result

                DetectMain.set_linear_mode(
                    harness, detectmain_module.LINEAR_MODE_IMAGE_SERIES
                )
                DetectMain.set_linear_mode(
                    harness, detectmain_module.LINEAR_MODE_SINGLE_IMAGE
                )

                self.assertIs(harness._regression_result, result)
                self.assertEqual(
                    harness._regression_result["selected_channel"], channel
                )

    def test_round_trip_without_plot_restores_placeholder_state(self):
        harness = DetectMainModeHarness()
        harness._regression_plot_has_result = False
        DetectMain.set_linear_mode(
            harness, detectmain_module.LINEAR_MODE_IMAGE_SERIES
        )
        harness._regression_plot_has_result = True

        DetectMain.set_linear_mode(
            harness, detectmain_module.LINEAR_MODE_SINGLE_IMAGE
        )

        self.assertFalse(harness._regression_plot_has_result)
        self.assertEqual(harness.plot_restore_calls, 0)

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
            harness._recgPixmap,
            harness.ui.tabviewRecg.model(),
            harness.ui.tabWidget.tabText(0),
            harness.ui.label_4.text(),
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
            harness._recgPixmap,
            harness.ui.tabviewRecg.model(),
            harness.ui.tabWidget.tabText(0),
            harness.ui.label_4.text(),
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
