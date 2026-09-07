import sys
from collections.abc import Mapping
from pathlib import Path
from types import MethodType
import unittest
from unittest import mock

import numpy as np
from PySide6.QtCore import QEventLoop, QObject, QThread, QTimer, Signal, Slot
from PySide6.QtWidgets import QApplication


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GUI_ROOT = PROJECT_ROOT / "Peng1.0_GUI"
sys.path.insert(0, str(GUI_ROOT))

import detectmain  # noqa: E402
import detectionwindow  # noqa: E402
from detectmain import DetectMain  # noqa: E402
from linear_series_controller import (  # noqa: E402
    LINEAR_SERIES_OPERATION,
    LinearSeriesController,
)
from linear_series_state import (  # noqa: E402
    LinearImageStatus,
    LinearSeriesPhase,
    LinearSeriesState,
)


LINEAR_MODE_IMAGE_SERIES = detectmain.LINEAR_MODE_IMAGE_SERIES
IDENTITY_FIELDS = (
    "operation",
    "run_token",
    "job_token",
    "image_order",
    "normalized_path",
    "original_file_name",
)
VIRTUAL_ROOT = str(Path("Z:/codex-linear-series-memory"))
VIRTUAL_WEIGHT = str(Path(VIRTUAL_ROOT) / "weights" / "best.pt")


class SignalSpy:
    def __init__(self):
        self.calls = []

    def emit(self, *arguments):
        self.calls.append(arguments)


class WidgetStub:
    def __init__(self, text=""):
        self.enabled = True
        self.text_value = text
        self.value_value = 0
        self.range_value = (0, 100)
        self.pixmap = None
        self.model_value = None
        self.tab_texts = {0: "Detection Image"}
        self.calls = []

    def setEnabled(self, enabled):
        self.enabled = bool(enabled)
        self.calls.append(("setEnabled", bool(enabled)))

    def isEnabled(self):
        return self.enabled

    def setText(self, text):
        self.text_value = str(text)
        self.calls.append(("setText", str(text)))

    def text(self):
        return self.text_value

    def setValue(self, value):
        self.value_value = value
        self.calls.append(("setValue", value))

    def value(self):
        return self.value_value

    def setRange(self, minimum, maximum):
        self.range_value = (minimum, maximum)
        self.calls.append(("setRange", minimum, maximum))

    def minimum(self):
        return self.range_value[0]

    def maximum(self):
        return self.range_value[1]

    def setPixmap(self, pixmap):
        self.pixmap = pixmap
        self.calls.append(("setPixmap", pixmap))

    def clear(self):
        self.text_value = ""
        self.pixmap = None
        self.calls.append(("clear",))

    def setModel(self, model):
        self.model_value = model
        self.calls.append(("setModel", model))

    def model(self):
        return self.model_value

    def indexOf(self, _widget):
        return 0

    def setTabText(self, index, text):
        self.tab_texts[index] = str(text)
        self.calls.append(("setTabText", index, str(text)))

    def tabText(self, index):
        return self.tab_texts.get(index, "")

    def setAlignment(self, alignment):
        self.calls.append(("setAlignment", alignment))

    def __getattr__(self, name):
        def recorder(*arguments, **keywords):
            self.calls.append((name, arguments, keywords))
            return None

        return recorder


class UiStub:
    def __init__(self):
        self._widgets = {}
        for name in (
            "pushButton_5",
            "pushButton_4",
            "pushButton",
            "pushButton_7",
            "pushButton_8",
            "progressBar",
            "labelOrigImg",
            "labelRecgImg",
            "label_4",
            "tabviewOrig",
            "tabviewRecg",
            "tabWidget",
            "tab_3",
            "lcdNumber",
        ):
            self._widgets[name] = WidgetStub()

    def __getattr__(self, name):
        try:
            return self._widgets[name]
        except KeyError:
            widget = WidgetStub()
            self._widgets[name] = widget
            return widget


class PixmapStub:
    def __init__(self, source):
        self.source = source

    def isNull(self):
        return False


class CameraStub:
    def __init__(self):
        self.pending = False

    def set_close_wait_pending(self, pending):
        self.pending = bool(pending)


class FlowHarness:
    """A non-QObject receiver for DetectMain's pure orchestration methods."""

    def __init__(self, paths=()):
        confirmed = object()
        state = LinearSeriesState.from_paths(
            paths, last_confirmed_result=confirmed
        )
        self._linear_series_controller = LinearSeriesController(state)
        self._linear_series_selection_state = state
        self._linear_series_weight_path = VIRTUAL_WEIGHT
        self._linear_series_ui_warnings = []
        self._linear_mode = LINEAR_MODE_IMAGE_SERIES
        self._single_linear_action_text = "Linear Regression"
        self._single_linear_view_state = None
        self._active_worker_task = None
        self._close_wait_pending = False
        self._shutdown_requested = False
        self._regression_result = confirmed
        self._regression_dirty = True
        self._detection_result = {"preserved": "detection"}
        self._detection_dirty = True
        self._last_completed_result_type = "linear"
        self._regression_plot_has_result = True
        self._calibration_source_path = "single-image-preserved.png"
        self._calibration_source_image = object()
        self._last_calibration_directory = "single-directory-preserved"
        self.origImg = "single-preview-preserved"
        self._origPixmap = PixmapStub("single-preview-preserved")
        self._recgPixmap = None
        self._detection_view_state = None
        self._linear_series_result_view_state = None
        self.ui = UiStub()
        self.mainCamera = CameraStub()

        self.linear_series_extraction_requested = SignalSpy()
        self.detection_status_changed = SignalSpy()
        self.linear_series_status_changed = SignalSpy()
        self.worker_task_finished = SignalSpy()
        self.busy_changed = SignalSpy()
        self.clear_active_formulas_requested = SignalSpy()
        self.install_saved_formulas_requested = SignalSpy()
        self.restore_active_formulas_requested = SignalSpy()

        self.decode_calls = []
        self.decode_error = None
        self.scale_calls = []
        self.messages = []
        self.save_updates = 0
        self.shutdown_request_calls = 0

    def __getattr__(self, name):
        value = DetectMain.__dict__.get(name)
        if value is None:
            raise AttributeError(name)
        if hasattr(value, "__get__"):
            return value.__get__(self, type(self))
        return value

    def _decode_calibration_image(self, path):
        self.decode_calls.append(str(path))
        if self.decode_error is not None:
            raise self.decode_error
        return memory_image()

    def _bgr_image_to_pixmap(self, image):
        return PixmapStub(image)

    def _scale_label(self, label):
        self.scale_calls.append(label)

    def _has_valid_regression_result(self):
        return self._regression_result is not None

    def _update_save_button(self):
        self.save_updates += 1

    def _show_message_safely(self, message_function, *arguments):
        self.messages.append((message_function, arguments))

    def request_shutdown(self):
        self.shutdown_request_calls += 1
        self._shutdown_requested = True

    def _set_active_worker_task(self, task):
        return DetectMain._set_active_worker_task(self, task)


class CloseEventStub:
    def __init__(self):
        self.accepted = 0
        self.ignored = 0

    def accept(self):
        self.accepted += 1

    def ignore(self):
        self.ignored += 1


class MappingCloseMainStub:
    def __init__(self, confirmed, mapping=True, linear_dirty=False,
                 detection_dirty=False):
        self._regression_dirty = bool(linear_dirty)
        self._detection_dirty = bool(detection_dirty)
        self.confirmed = confirmed
        self.mapping_draft = object() if mapping else None
        self.discard_calls = 0

    def has_linear_series_mapping_draft(self):
        return self.mapping_draft is not None

    def discard_linear_series_draft(self):
        self.discard_calls += 1
        self.mapping_draft = None

    def is_worker_task_active(self):
        return False


class MappingCloseHarness:
    closeEvent = detectionwindow.DetectWindow.closeEvent
    _has_unsaved_results = detectionwindow.DetectWindow._has_unsaved_results
    _has_linear_series_mapping_draft = (
        detectionwindow.DetectWindow._has_linear_series_mapping_draft
    )
    _close_discard_categories = (
        detectionwindow.DetectWindow._close_discard_categories
    )

    def __init__(self, mapping=True, linear_dirty=False, detection_dirty=False):
        self.confirmed = object()
        self._detectMain = MappingCloseMainStub(
            self.confirmed,
            mapping=mapping,
            linear_dirty=linear_dirty,
            detection_dirty=detection_dirty,
        )
        self._final_close_allowed = False
        self._shutdown_started = False
        self._close_wait_pending = False
        self.shutdown_calls = 0

    def _activate_close_wait_dialog(self):
        raise AssertionError("mapping close is not a worker wait")

    def _begin_close_wait(self):
        raise AssertionError("mapping close is not a worker wait")

    def _begin_shutdown(self):
        self.shutdown_calls += 1
        self._shutdown_started = True


class ActiveCloseHarness:
    _cancel_close_wait = detectionwindow.DetectWindow._cancel_close_wait
    _on_worker_task_finished = (
        detectionwindow.DetectWindow._on_worker_task_finished
    )

    def __init__(self, detect_main):
        self._detectMain = detect_main
        self._close_wait_pending = True
        self._shutdown_started = False
        self._close_wait_dialog = None
        self.dispose_calls = 0
        self.shutdown_calls = 0
        self.control_states = []

    def _dispose_close_wait_dialog(self):
        self.dispose_calls += 1
        self._close_wait_dialog = None

    def _set_close_controls_disabled(self, disabled):
        self.control_states.append(bool(disabled))

    def _begin_shutdown(self):
        self.shutdown_calls += 1
        return detectionwindow.DetectWindow._begin_shutdown(self)


class IdentityOnlyPayload(Mapping):
    """Raise if a rejected or closing result reads non-identity content."""

    def __init__(self, identity):
        self._identity = dict(identity)

    def __getitem__(self, key):
        if key not in IDENTITY_FIELDS:
            raise AssertionError("payload content read before it was allowed")
        return self._identity[key]

    def __iter__(self):
        return iter(self._identity)

    def __len__(self):
        return len(self._identity)


class PoisonContentPayload(Mapping):
    """Expose identity, but fail if rejected payload content is inspected."""

    _poison_fields = ("image", "samples", "reason")

    def __init__(self, identity):
        self._identity = dict(identity)
        self.poison_reads = []

    def __getitem__(self, key):
        if key in self._identity:
            return self._identity[key]
        self.poison_reads.append(key)
        raise AssertionError("poison payload field was read: {}".format(key))

    def __iter__(self):
        return iter(tuple(self._identity) + self._poison_fields)

    def __len__(self):
        return len(self._identity) + len(self._poison_fields)


class MemoryCallbackWorker(QObject):
    payload_ready = Signal(object)
    work_finished = Signal()

    def __init__(self, payload):
        super().__init__()
        self.payload = payload

    @Slot()
    def run(self):
        self.payload_ready.emit(self.payload)
        self.work_finished.emit()


class MemoryCallbackReceiver(QObject):
    handled = Signal()

    def __init__(self, callback):
        super().__init__()
        self.callback = callback

    @Slot(object)
    def receive(self, payload):
        self.callback(payload)
        self.handled.emit()


def memory_image():
    return np.zeros((3, 4, 3), dtype=np.uint8)


def sample_for(task, red=10.0, green=20.0, blue=30.0):
    return {
        "image_order": task.image_order,
        "normalized_path": task.normalized_path,
        "original_file_name": task.original_file_name,
        "spatial_order": 1,
        "red": red,
        "green": green,
        "blue": blue,
    }


def success_payload(task, samples=None, errors=None):
    if samples is None:
        samples = [sample_for(task)]
    payload = task.context()
    payload.update({
        "image": memory_image(),
        "samples": samples,
        "errors": [] if errors is None else errors,
        "warnings": [],
    })
    return payload


def failure_payload(task, reason="memory image failed"):
    payload = task.context()
    payload.update({
        "error_type": "image_failed",
        "reason": reason,
        "errors": [],
    })
    return payload


def task_snapshot(task):
    return None if task is None else tuple(task.context().items())


def controller_snapshot(controller):
    return (
        controller.run_token,
        task_snapshot(controller.active_job),
        controller.queued_image_orders,
        controller.state.phase,
        tuple(
            (
                image.image_order,
                image.normalized_path,
                image.original_file_name,
                image.status,
                tuple(sample.sample_key for sample in image.samples),
                tuple(error.error_key for error in image.errors),
                image.failure_reason,
            )
            for image in controller.state.images
        ),
    )


def flow_snapshot(harness):
    return (
        controller_snapshot(harness._linear_series_controller),
        tuple(
            (
                image.image_order,
                image.normalized_path,
                image.original_file_name,
                image.status,
            )
            for image in harness._linear_series_selection_state.images
        ),
        harness._active_worker_task,
        tuple(harness.linear_series_extraction_requested.calls),
        tuple(harness.detection_status_changed.calls),
        tuple(harness.linear_series_status_changed.calls),
        harness.ui.progressBar.range_value,
        harness.ui.progressBar.value_value,
        len(harness.messages),
        harness.origImg,
        id(harness._origPixmap),
        harness.ui.labelOrigImg.text(),
        id(harness._recgPixmap),
        harness.ui.labelRecgImg.text(),
        id(harness.ui.tabviewRecg.model()),
        harness.ui.tabWidget.tabText(0),
        harness.ui.label_4.text(),
        id(harness._regression_result),
        harness._regression_dirty,
        id(harness._detection_result),
        harness._detection_dirty,
        harness._last_completed_result_type,
        tuple(harness.clear_active_formulas_requested.calls),
        tuple(harness.install_saved_formulas_requested.calls),
        tuple(harness.restore_active_formulas_requested.calls),
    )


def bind_required_method(name):
    method = getattr(DetectMain, name, None)
    if method is None:
        raise AssertionError("DetectMain is missing required method {}".format(name))
    return method


def invoke(harness, name, *arguments):
    return bind_required_method(name)(harness, *arguments)


class LinearSeriesFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        required = (
            "_select_linear_series_images",
            "_start_linear_series",
            "_dispatch_linear_series_task",
            "_on_linear_series_extraction_finished",
            "_on_linear_series_extraction_failed",
            "_finish_linear_series_run",
        )
        for name in required:
            self.assertTrue(hasattr(DetectMain, name), name)

    def select_images(self, harness, paths):
        guarded_initial = mock.Mock(return_value=VIRTUAL_ROOT)
        with mock.patch.object(
            detectmain, "_dialog_initial_directory", guarded_initial
        ), mock.patch.object(
            detectmain.QFileDialog,
            "getOpenFileNames",
            return_value=(list(paths), "Images"),
        ), mock.patch.object(
            detectmain.QMessageBox, "critical"
        ) as critical:
            invoke(harness, "_select_linear_series_images")
        self.assertEqual(critical.call_count, 0)
        self.assertTrue(guarded_initial.called)

    def start_series(self, harness):
        checked_paths = []

        def virtual_is_file(path):
            rendered = str(path)
            checked_paths.append(rendered)
            portable = rendered.replace("\\", "/")
            self.assertNotIn("HT-Detector_Peng/custom/linear_detection", portable)
            self.assertNotIn("HT-Detector_Peng/weights", portable)
            self.assertEqual(Path(rendered), Path(VIRTUAL_WEIGHT))
            return True

        with mock.patch.object(
            detectmain, "LINEAR_WEIGHT_PATH", Path(VIRTUAL_WEIGHT)
        ), mock.patch.object(
            Path, "is_file", autospec=True, side_effect=virtual_is_file
        ), mock.patch.object(detectmain.QMessageBox, "critical") as critical:
            invoke(harness, "_start_linear_series")
        self.assertEqual(critical.call_count, 0)
        return checked_paths

    def deliver_from_memory_thread(self, callback, payload):
        thread = QThread()
        worker = MemoryCallbackWorker(payload)
        receiver = MemoryCallbackReceiver(callback)
        loop = QEventLoop()
        handled = []
        timed_out = []
        timer = QTimer()
        timer.setSingleShot(True)
        timer.timeout.connect(lambda: (timed_out.append(True), loop.quit()))
        receiver.handled.connect(lambda: (handled.append(True), loop.quit()))
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.payload_ready.connect(receiver.receive)
        worker.work_finished.connect(worker.deleteLater)
        worker.work_finished.connect(thread.quit)

        thread.start()
        timer.start(3000)
        loop.exec()
        timer.stop()
        thread.quit()
        self.assertTrue(thread.wait(3000))
        self.assertEqual(timed_out, [])
        self.assertEqual(handled, [True])
        self.assertFalse(thread.isRunning())

    def test_multi_select_cancel_is_a_complete_no_op(self):
        harness = FlowHarness([VIRTUAL_ROOT + "/old7.PNG"])
        baseline_controller = harness._linear_series_controller
        baseline = flow_snapshot(harness)
        baseline_preview = (harness.origImg, harness._origPixmap)

        with mock.patch.object(
            detectmain, "_dialog_initial_directory", return_value=VIRTUAL_ROOT
        ), mock.patch.object(
            detectmain.QFileDialog,
            "getOpenFileNames",
            return_value=([], "Images"),
        ):
            invoke(harness, "_select_linear_series_images")

        self.assertIs(harness._linear_series_controller, baseline_controller)
        self.assertEqual(flow_snapshot(harness), baseline)
        self.assertEqual((harness.origImg, harness._origPixmap), baseline_preview)
        self.assertEqual(harness.decode_calls, [])

    def test_selection_naturally_sorts_previews_first_and_keeps_duplicates(self):
        harness = FlowHarness([VIRTUAL_ROOT + "/old7.PNG"])
        confirmed = harness._regression_result
        detection = harness._detection_result
        paths = [
            VIRTUAL_ROOT + "/set/image10.PNG",
            VIRTUAL_ROOT + "/b/image2.JPG",
            VIRTUAL_ROOT + "/a/image2.JPG",
            VIRTUAL_ROOT + "/a/image2.JPG",
            VIRTUAL_ROOT + "/set/image1.TiFf",
        ]

        self.select_images(harness, paths)

        controller = harness._linear_series_controller
        images = harness._linear_series_selection_state.images
        self.assertEqual(
            [image.original_file_name for image in images],
            ["image1.TiFf", "image2.JPG", "image2.JPG", "image2.JPG", "image10.PNG"],
        )
        image2_paths = [
            image.normalized_path
            for image in images
            if image.original_file_name == "image2.JPG"
        ]
        self.assertEqual(len(image2_paths), 3)
        self.assertEqual(len(set(image2_paths)), 2)
        self.assertEqual(len({image.image_key for image in images}), 5)
        self.assertEqual(Path(harness.decode_calls[0]).name, "image1.TiFf")
        self.assertEqual(controller.state.phase, LinearSeriesPhase.IDLE)
        self.assertFalse(controller.busy)
        self.assertIsNone(harness._active_worker_task)
        self.assertEqual(harness.linear_series_extraction_requested.calls, [])
        self.assertIs(harness._regression_result, confirmed)
        self.assertIs(harness._detection_result, detection)

    def test_failed_first_preview_preparation_keeps_old_selection_atomically(self):
        harness = FlowHarness([VIRTUAL_ROOT + "/old7.PNG"])
        old_controller = harness._linear_series_controller
        old_preview = (harness.origImg, harness._origPixmap)
        old_result = harness._regression_result
        harness.decode_error = ValueError("synthetic decode failure")

        guarded_initial = mock.Mock(return_value=VIRTUAL_ROOT)
        with mock.patch.object(
            detectmain, "_dialog_initial_directory", guarded_initial
        ), mock.patch.object(
            detectmain.QFileDialog,
            "getOpenFileNames",
            return_value=([VIRTUAL_ROOT + "/new2.png"], "Images"),
        ), mock.patch.object(detectmain.QMessageBox, "critical") as critical:
            invoke(harness, "_select_linear_series_images")

        self.assertIs(harness._linear_series_controller, old_controller)
        self.assertEqual((harness.origImg, harness._origPixmap), old_preview)
        self.assertIs(harness._regression_result, old_result)
        self.assertIsNone(harness._active_worker_task)
        self.assertFalse(old_controller.busy)
        self.assertEqual(critical.call_count + len(harness.messages), 1)

    def test_selection_control_commit_failure_rolls_back_every_visible_state(self):
        harness = FlowHarness([VIRTUAL_ROOT + "/old7.PNG"])
        harness._linear_series_weight_path = "memory/old-weight.pt"
        old_controller = harness._linear_series_controller
        old_preview = (
            harness.origImg,
            harness._origPixmap,
            harness.ui.labelOrigImg.text(),
        )
        old_progress = (
            harness.ui.progressBar.range_value,
            harness.ui.progressBar.value_value,
        )
        buttons = (
            harness.ui.pushButton_4,
            harness.ui.pushButton_5,
            harness.ui.pushButton_7,
            harness.ui.pushButton_8,
            harness.ui.pushButton,
        )
        old_controls = tuple(
            (button.text(), button.isEnabled()) for button in buttons
        )

        def fail_after_mutating_controls():
            for index, button in enumerate(buttons):
                button.setText("partial-{}".format(index))
                button.setEnabled(False)
            raise RuntimeError("synthetic selection commit failure")

        harness._apply_linear_mode_controls = fail_after_mutating_controls
        with mock.patch.object(
            detectmain, "_dialog_initial_directory", return_value=VIRTUAL_ROOT
        ), mock.patch.object(
            detectmain.QFileDialog,
            "getOpenFileNames",
            return_value=([VIRTUAL_ROOT + "/new2.png"], "Images"),
        ):
            invoke(harness, "_select_linear_series_images")

        self.assertIs(harness._linear_series_controller, old_controller)
        self.assertEqual(harness._linear_series_weight_path, "memory/old-weight.pt")
        self.assertEqual((
            harness.origImg,
            harness._origPixmap,
            harness.ui.labelOrigImg.text(),
        ), old_preview)
        self.assertEqual((
            harness.ui.progressBar.range_value,
            harness.ui.progressBar.value_value,
        ), old_progress)
        self.assertEqual(tuple(
            (button.text(), button.isEnabled()) for button in buttons
        ), old_controls)
        self.assertEqual(len(harness.messages), 1)

    def test_busy_selection_reentry_does_not_open_another_dialog(self):
        harness = FlowHarness([VIRTUAL_ROOT + "/image1.png"])
        harness._active_worker_task = "linear_series"
        before = flow_snapshot(harness)
        with mock.patch.object(
            detectmain.QFileDialog, "getOpenFileNames"
        ) as get_names, mock.patch.object(
            detectmain, "_dialog_initial_directory"
        ) as initial_directory:
            invoke(harness, "_select_linear_series_images")

        get_names.assert_not_called()
        initial_directory.assert_not_called()
        self.assertEqual(flow_snapshot(harness), before)

    def test_click_begins_and_dispatches_only_one_exact_context(self):
        harness = FlowHarness([
            VIRTUAL_ROOT + "/image10.png",
            VIRTUAL_ROOT + "/image2.png",
            VIRTUAL_ROOT + "/image1.png",
        ])
        self.assertIsNone(harness._linear_series_controller.run_token)
        self.assertEqual(harness.linear_series_extraction_requested.calls, [])

        self.start_series(harness)

        controller = harness._linear_series_controller
        task = controller.active_job
        self.assertTrue(controller.busy)
        self.assertEqual(harness._active_worker_task, "linear_series")
        self.assertEqual(len(harness.linear_series_extraction_requested.calls), 1)
        path, weight, context = harness.linear_series_extraction_requested.calls[0]
        self.assertEqual(path, task.normalized_path)
        self.assertEqual(weight, VIRTUAL_WEIGHT)
        self.assertEqual(context, task.context())
        self.assertEqual(tuple(context), IDENTITY_FIELDS)
        self.assertEqual(context["operation"], LINEAR_SERIES_OPERATION)
        self.assertEqual(controller.queued_image_orders, (2, 3))

    def test_success_and_failure_handlers_dispatch_strictly_serially(self):
        harness = FlowHarness([
            VIRTUAL_ROOT + "/image1.png",
            VIRTUAL_ROOT + "/image2.png",
            VIRTUAL_ROOT + "/image3.png",
        ])
        self.start_series(harness)
        first = harness._linear_series_controller.active_job
        self.assertEqual(len(harness.linear_series_extraction_requested.calls), 1)

        invoke(
            harness,
            "_on_linear_series_extraction_finished",
            success_payload(first),
        )
        second = harness._linear_series_controller.active_job
        self.assertEqual(second.image_order, 2)
        self.assertEqual(len(harness.linear_series_extraction_requested.calls), 2)
        self.assertEqual(harness._active_worker_task, "linear_series")

        invoke(
            harness,
            "_on_linear_series_extraction_failed",
            failure_payload(second),
        )
        third = harness._linear_series_controller.active_job
        self.assertEqual(third.image_order, 3)
        self.assertEqual(len(harness.linear_series_extraction_requested.calls), 3)
        self.assertEqual(harness.messages, [])
        self.assertEqual(harness._active_worker_task, "linear_series")

        invoke(
            harness,
            "_on_linear_series_extraction_finished",
            success_payload(third, [sample_for(third, 40, 50, 60)]),
        )
        self.assertIsNone(harness._linear_series_controller.active_job)
        self.assertIsNone(harness._linear_series_controller.run_token)
        self.assertEqual(harness._active_worker_task, None)
        self.assertEqual(
            harness._linear_series_controller.state.phase,
            LinearSeriesPhase.MAPPING,
        )

    def test_only_successful_series_images_own_the_visible_result_pane(self):
        harness = FlowHarness([
            VIRTUAL_ROOT + "/image1.png",
            VIRTUAL_ROOT + "/image2.png",
        ])
        self.start_series(harness)
        first = harness._linear_series_controller.active_job
        first_payload = success_payload(first)

        invoke(harness, "_on_linear_series_extraction_finished", first_payload)

        model = harness.ui.tabviewRecg.model()
        self.assertIs(harness._recgPixmap.source, first_payload["image"])
        self.assertIn(first.original_file_name, harness.ui.tabWidget.tabText(0))
        self.assertEqual(model.rowCount(), 1)
        self.assertEqual(model.item(0, 0).text(), "1")
        self.assertEqual(model.item(0, 1).text(), "")
        self.assertEqual(model.item(0, 2).text(), "10.00")
        successful_view = (
            harness._recgPixmap,
            model,
            harness.ui.tabWidget.tabText(0),
            harness.ui.label_4.text(),
        )

        second = harness._linear_series_controller.active_job
        invoke(
            harness,
            "_on_linear_series_extraction_failed",
            failure_payload(second, "second failed"),
        )

        self.assertEqual((
            harness._recgPixmap,
            harness.ui.tabviewRecg.model(),
            harness.ui.tabWidget.tabText(0),
            harness.ui.label_4.text(),
        ), successful_view)
        self.assertEqual(len(harness.messages), 1)
        self.assertIs(harness.messages[0][0], detectmain.QMessageBox.warning)

    def test_partial_sample_error_view_contains_only_valid_numbered_samples(self):
        harness = FlowHarness([VIRTUAL_ROOT + "/partial.png"])
        self.start_series(harness)
        task = harness._linear_series_controller.active_job
        payload = success_payload(task, errors=[{
            "reason": "second sample has an invalid ROI",
            "error_type": "invalid_roi",
            "spatial_order": 2,
        }])

        invoke(harness, "_on_linear_series_extraction_finished", payload)

        image = harness._linear_series_controller.state.images[0]
        model = harness.ui.tabviewRecg.model()
        self.assertEqual(image.status, LinearImageStatus.COMPLETED)
        self.assertEqual(len(image.samples), 1)
        self.assertEqual(len(image.errors), 1)
        self.assertEqual(model.rowCount(), 1)
        self.assertEqual(model.item(0, 0).text(), "1")
        self.assertEqual(model.item(0, 1).text(), "")
        self.assertEqual(
            tuple(model.item(0, column).text() for column in (2, 3, 4)),
            ("10.00", "20.00", "30.00"),
        )
        self.assertEqual(len(harness.messages), 1)
        self.assertIs(harness.messages[0][0], detectmain.QMessageBox.warning)

    def test_zero_sample_success_payload_is_failed_and_never_displayed(self):
        harness = FlowHarness([VIRTUAL_ROOT + "/empty.png"])
        self.start_series(harness)
        task = harness._linear_series_controller.active_job
        payload = success_payload(task, samples=[], errors=[])

        invoke(harness, "_on_linear_series_extraction_finished", payload)

        image = harness._linear_series_controller.state.images[0]
        self.assertEqual(image.status, LinearImageStatus.FAILED)
        self.assertEqual(image.samples, ())
        self.assertIsNone(harness._recgPixmap)
        model = harness.ui.tabviewRecg.model()
        self.assertTrue(model is None or model.rowCount() == 0)
        self.assertEqual(len(harness.messages), 1)
        self.assertIs(harness.messages[0][0], detectmain.QMessageBox.critical)

    def test_rerun_clears_previous_series_view_before_dispatch(self):
        harness = FlowHarness([VIRTUAL_ROOT + "/image1.png"])
        self.start_series(harness)
        first_run = harness._linear_series_controller.active_job
        invoke(
            harness,
            "_on_linear_series_extraction_finished",
            success_payload(first_run),
        )
        self.assertIsNotNone(harness._recgPixmap)
        self.assertEqual(harness.ui.tabviewRecg.model().rowCount(), 1)

        self.start_series(harness)

        second_run = harness._linear_series_controller.active_job
        self.assertGreater(second_run.run_token, first_run.run_token)
        self.assertIsNone(harness._recgPixmap)
        self.assertEqual(harness.ui.tabviewRecg.model().rowCount(), 0)
        self.assertEqual(
            harness.ui.tabWidget.tabText(0), "Linear Series Image"
        )
        self.assertEqual(
            harness.ui.label_4.text(), "Table. Linear Series"
        )
        self.assertEqual(
            len(harness.linear_series_extraction_requested.calls), 2
        )

    def test_duplicate_paths_dispatch_as_distinct_jobs(self):
        repeated = VIRTUAL_ROOT + "/same-name.png"
        harness = FlowHarness([repeated, repeated])
        self.start_series(harness)
        controller = harness._linear_series_controller
        first = controller.active_job
        first_context = harness.linear_series_extraction_requested.calls[0][2]

        invoke(
            harness,
            "_on_linear_series_extraction_finished",
            success_payload(first),
        )

        second = controller.active_job
        second_context = harness.linear_series_extraction_requested.calls[1][2]
        self.assertEqual(first.normalized_path, second.normalized_path)
        self.assertEqual(first.original_file_name, second.original_file_name)
        self.assertEqual(first.run_token, second.run_token)
        self.assertNotEqual(first.job_token, second.job_token)
        self.assertNotEqual(first.image_order, second.image_order)
        self.assertEqual(first_context, first.context())
        self.assertEqual(second_context, second.context())

    def test_controller_tokens_never_repeat_across_reselection_or_runs(self):
        repeated = VIRTUAL_ROOT + "/same.png"
        harness = FlowHarness([repeated])
        controller = harness._linear_series_controller

        self.start_series(harness)
        first = controller.active_job
        first_success = success_payload(first)
        first_failure = failure_payload(first, "late first run")
        invoke(harness, "_on_linear_series_extraction_finished", first_success)

        self.select_images(harness, [repeated])
        self.assertIs(harness._linear_series_controller, controller)
        self.start_series(harness)
        second = controller.active_job
        self.assertNotEqual(
            (second.run_token, second.job_token),
            (first.run_token, first.job_token),
        )
        second_snapshot = flow_snapshot(harness)
        invoke(harness, "_on_linear_series_extraction_finished", first_success)
        invoke(harness, "_on_linear_series_extraction_failed", first_failure)
        self.assertEqual(flow_snapshot(harness), second_snapshot)
        invoke(
            harness,
            "_on_linear_series_extraction_finished",
            success_payload(second),
        )

        self.start_series(harness)
        third = controller.active_job
        self.assertEqual(first.run_token + 1, second.run_token)
        self.assertEqual(second.run_token + 1, third.run_token)
        self.assertEqual(first.job_token + 1, second.job_token)
        self.assertEqual(second.job_token + 1, third.job_token)

    def test_wrong_duplicate_and_late_results_have_no_side_effects(self):
        invalid_builders = (
            lambda task: dict(success_payload(task), operation="detection"),
            lambda task: dict(success_payload(task), run_token=task.run_token + 1),
            lambda task: dict(success_payload(task), job_token=task.job_token + 1),
            lambda task: dict(success_payload(task), run_token=True),
            lambda task: {
                key: value
                for key, value in success_payload(task).items()
                if key != "original_file_name"
            },
        )
        harness = FlowHarness([
            VIRTUAL_ROOT + "/image1.png",
            VIRTUAL_ROOT + "/image2.png",
        ])
        self.start_series(harness)
        first = harness._linear_series_controller.active_job

        for builder in invalid_builders:
            baseline = flow_snapshot(harness)
            invoke(
                harness,
                "_on_linear_series_extraction_finished",
                builder(first),
            )
            self.assertEqual(flow_snapshot(harness), baseline)

        wrong_identity = first.context()
        wrong_identity["operation"] = "not-linear-series"
        baseline = flow_snapshot(harness)
        invoke(
            harness,
            "_on_linear_series_extraction_finished",
            IdentityOnlyPayload(wrong_identity),
        )
        invoke(
            harness,
            "_on_linear_series_extraction_failed",
            IdentityOnlyPayload(wrong_identity),
        )
        self.assertEqual(flow_snapshot(harness), baseline)

        first_payload = success_payload(first)
        invoke(harness, "_on_linear_series_extraction_finished", first_payload)
        second = harness._linear_series_controller.active_job
        after_first = flow_snapshot(harness)
        invoke(harness, "_on_linear_series_extraction_finished", first_payload)
        self.assertEqual(flow_snapshot(harness), after_first)

        second_payload = success_payload(second)
        invoke(harness, "_on_linear_series_extraction_finished", second_payload)
        finished = flow_snapshot(harness)
        invoke(harness, "_on_linear_series_extraction_finished", second_payload)
        invoke(
            harness,
            "_on_linear_series_extraction_failed",
            failure_payload(second, "late"),
        )
        self.assertEqual(flow_snapshot(harness), finished)

    def test_wrong_path_rejects_poison_content_before_any_ui_side_effect(self):
        harness = FlowHarness([
            VIRTUAL_ROOT + "/image1.png",
            VIRTUAL_ROOT + "/image2.png",
        ])
        self.start_series(harness)
        task = harness._linear_series_controller.active_job
        wrong_identity = task.context()
        wrong_identity["normalized_path"] = VIRTUAL_ROOT + "/wrong.png"
        payload = PoisonContentPayload(wrong_identity)
        before = flow_snapshot(harness)
        hidden_views = (
            harness._detection_view_state,
            harness._linear_series_result_view_state,
        )

        invoke(harness, "_on_linear_series_extraction_finished", payload)
        invoke(harness, "_on_linear_series_extraction_failed", payload)

        self.assertEqual(payload.poison_reads, [])
        self.assertEqual(flow_snapshot(harness), before)
        self.assertIs(harness._detection_view_state, hidden_views[0])
        self.assertIs(harness._linear_series_result_view_state, hidden_views[1])

    def test_invalid_legal_payload_becomes_clean_current_image_failure(self):
        harness = FlowHarness([
            VIRTUAL_ROOT + "/image1.png",
            VIRTUAL_ROOT + "/image2.png",
        ])
        self.start_series(harness)
        first = harness._linear_series_controller.active_job
        malformed = first.context()
        malformed.update({
            "image": memory_image(),
            "samples": {"not": "an ordered collection"},
            "errors": [],
            "warnings": [],
            "tainted": "must not be copied",
        })

        invoke(harness, "_on_linear_series_extraction_finished", malformed)

        first_image = harness._linear_series_controller.state.images[0]
        self.assertEqual(first_image.status, LinearImageStatus.FAILED)
        self.assertTrue(first_image.failure_reason)
        self.assertEqual(
            harness._linear_series_controller.active_job.image_order, 2
        )
        self.assertEqual(len(harness.linear_series_extraction_requested.calls), 2)
        self.assertEqual(harness._active_worker_task, "linear_series")
        self.assertEqual(harness.messages, [])

    def test_invalid_preview_dtype_or_layout_becomes_current_image_failure(self):
        invalid_images = (
            np.zeros((3, 4, 3), dtype=np.float64),
            np.zeros((3, 8, 3), dtype=np.uint8)[:, ::2, :],
        )
        for invalid_image in invalid_images:
            with self.subTest(dtype=str(invalid_image.dtype),
                              contiguous=invalid_image.flags.c_contiguous):
                harness = FlowHarness([
                    VIRTUAL_ROOT + "/image1.png",
                    VIRTUAL_ROOT + "/image2.png",
                ])
                self.start_series(harness)
                first = harness._linear_series_controller.active_job
                malformed = success_payload(first)
                malformed["image"] = invalid_image

                invoke(
                    harness,
                    "_on_linear_series_extraction_finished",
                    malformed,
                )

                first_image = harness._linear_series_controller.state.images[0]
                self.assertEqual(first_image.status, LinearImageStatus.FAILED)
                self.assertEqual(
                    harness._linear_series_controller.active_job.image_order,
                    2,
                )
                self.assertEqual(
                    len(harness.linear_series_extraction_requested.calls),
                    2,
                )

    def test_invalid_legal_failed_payload_uses_clean_current_failure(self):
        harness = FlowHarness([
            VIRTUAL_ROOT + "/image1.png",
            VIRTUAL_ROOT + "/image2.png",
        ])
        self.start_series(harness)
        first = harness._linear_series_controller.active_job
        malformed = first.context()
        malformed.update({"errors": object()})

        invoke(
            harness,
            "_on_linear_series_extraction_failed",
            malformed,
        )

        first_image = harness._linear_series_controller.state.images[0]
        self.assertEqual(first_image.status, LinearImageStatus.FAILED)
        self.assertTrue(first_image.failure_reason)
        self.assertEqual(
            harness._linear_series_controller.active_job.image_order,
            2,
        )
        self.assertEqual(len(harness.linear_series_extraction_requested.calls), 2)
        self.assertEqual(harness.messages, [])

    def test_rejected_clean_failed_signal_safely_cancels_and_releases_busy(self):
        harness = FlowHarness([
            VIRTUAL_ROOT + "/image1.png",
            VIRTUAL_ROOT + "/image2.png",
        ])
        self.start_series(harness)
        controller = harness._linear_series_controller
        task = controller.active_job
        malformed = task.context()
        malformed.update({"errors": object()})
        original_accept_failure = controller.accept_failure
        controller.accept_failure = mock.Mock(return_value=False)
        dispatch_count = len(harness.linear_series_extraction_requested.calls)

        try:
            invoke(
                harness,
                "_on_linear_series_extraction_failed",
                malformed,
            )
        finally:
            mocked_accept_failure = controller.accept_failure
            controller.accept_failure = original_accept_failure

        self.assertEqual(mocked_accept_failure.call_count, 2)
        clean_failure = mocked_accept_failure.call_args_list[1].args[0]
        self.assertEqual(
            {field: clean_failure[field] for field in IDENTITY_FIELDS},
            task.context(),
        )
        self.assertIsNone(controller.active_job)
        self.assertFalse(controller.busy)
        self.assertEqual(controller.state.phase, LinearSeriesPhase.CANCELLED)
        self.assertIsNone(harness._active_worker_task)
        self.assertEqual(len(harness.messages), 1)
        self.assertEqual(
            len(harness.linear_series_extraction_requested.calls),
            dispatch_count,
        )

    def test_rejected_clean_failure_safely_cancels_and_releases_gui_busy(self):
        harness = FlowHarness([
            VIRTUAL_ROOT + "/image1.png",
            VIRTUAL_ROOT + "/image2.png",
        ])
        self.start_series(harness)
        controller = harness._linear_series_controller
        task = controller.active_job
        malformed = task.context()
        malformed.update({"samples": object(), "errors": []})
        original_accept_failure = controller.accept_failure
        reject_failure = mock.Mock(return_value=False)
        controller.accept_failure = reject_failure
        baseline_dispatches = tuple(
            harness.linear_series_extraction_requested.calls
        )

        try:
            invoke(
                harness,
                "_on_linear_series_extraction_finished",
                malformed,
            )
        finally:
            controller.accept_failure = original_accept_failure

        reject_failure.assert_called_once()
        clean_failure = reject_failure.call_args.args[0]
        self.assertEqual(
            {field: clean_failure[field] for field in IDENTITY_FIELDS},
            task.context(),
        )
        self.assertEqual(set(clean_failure), set(IDENTITY_FIELDS) | {
            "error_type", "reason", "errors",
        })
        self.assertIsNone(controller.active_job)
        self.assertFalse(controller.busy)
        self.assertEqual(controller.state.phase, LinearSeriesPhase.CANCELLED)
        self.assertIsNone(harness._active_worker_task)
        self.assertEqual(len(harness.messages), 1)
        self.assertEqual(
            tuple(harness.linear_series_extraction_requested.calls),
            baseline_dispatches,
        )

    def test_pending_close_cancels_before_malformed_result_is_processed(self):
        harness = FlowHarness([
            VIRTUAL_ROOT + "/image1.png",
            VIRTUAL_ROOT + "/image2.png",
        ])
        self.start_series(harness)
        controller = harness._linear_series_controller
        task = controller.active_job
        malformed = task.context()
        malformed.update({"samples": object(), "errors": []})
        harness._close_wait_pending = True
        dispatch_count = len(harness.linear_series_extraction_requested.calls)
        status_count = len(harness.detection_status_changed.calls)
        progress = (
            harness.ui.progressBar.range_value,
            harness.ui.progressBar.value_value,
        )

        original_accept_failure = controller.accept_failure
        controller.accept_failure = mock.Mock(return_value=False)
        try:
            invoke(
                harness,
                "_on_linear_series_extraction_finished",
                malformed,
            )
        finally:
            controller.accept_failure = original_accept_failure

        self.assertEqual(controller.state.phase, LinearSeriesPhase.CANCELLED)
        self.assertIsNone(controller.run_token)
        self.assertIsNone(controller.active_job)
        self.assertIsNone(harness._active_worker_task)
        self.assertEqual(len(harness.worker_task_finished.calls), 1)
        self.assertEqual(
            len(harness.linear_series_extraction_requested.calls),
            dispatch_count,
        )
        self.assertEqual(len(harness.detection_status_changed.calls), status_count)
        self.assertEqual(harness.messages, [])
        self.assertEqual((
            harness.ui.progressBar.range_value,
            harness.ui.progressBar.value_value,
        ), progress)

    def test_double_failure_rejection_allows_retry_reselect_and_close(self):
        first_path = VIRTUAL_ROOT + "/image1.png"
        harness = FlowHarness([first_path])
        confirmed = harness._regression_result
        detection = harness._detection_result
        self.start_series(harness)
        controller = harness._linear_series_controller
        task = controller.active_job
        malformed = task.context()
        malformed.update({"errors": object()})

        original_accept_failure = controller.accept_failure
        controller.accept_failure = mock.Mock(return_value=False)
        try:
            invoke(harness, "_on_linear_series_extraction_failed", malformed)
        finally:
            controller.accept_failure = original_accept_failure

        self.assertFalse(controller.busy)
        self.assertIsNone(harness._active_worker_task)
        self.assertIs(harness._regression_result, confirmed)
        self.assertIs(harness._detection_result, detection)
        self.assertEqual(len(harness.messages), 1)

        second_path = VIRTUAL_ROOT + "/image2.png"
        self.select_images(harness, [second_path])
        self.start_series(harness)
        retry = controller.active_job
        self.assertGreater(retry.run_token, task.run_token)
        invoke(
            harness,
            "_on_linear_series_extraction_finished",
            success_payload(retry),
        )
        self.assertEqual(controller.state.phase, LinearSeriesPhase.MAPPING)

        close = MappingCloseHarness()
        close._detectMain = harness
        event = CloseEventStub()
        with mock.patch.object(
            detectionwindow.QMessageBox,
            "question",
            return_value=detectionwindow.QMessageBox.Yes,
        ) as question:
            close.closeEvent(event)
        question.assert_called_once()
        self.assertEqual(close.shutdown_calls, 1)

    def test_preview_and_status_failures_do_not_stall_multi_image_queue(self):
        class FailingPreview(WidgetStub):
            def __init__(self):
                super().__init__()
                self.fail_once = True

            def setPixmap(self, pixmap):
                super().setPixmap(pixmap)
                if self.fail_once:
                    self.fail_once = False
                    raise RuntimeError("synthetic setPixmap failure")

        class FailingStatus(SignalSpy):
            def __init__(self):
                super().__init__()
                self.fail_once = True

            def emit(self, *arguments):
                super().emit(*arguments)
                if self.fail_once:
                    self.fail_once = False
                    raise RuntimeError("synthetic status failure")

        harness = FlowHarness([
            VIRTUAL_ROOT + "/image1.png",
            VIRTUAL_ROOT + "/image2.png",
        ])
        self.start_series(harness)
        label = FailingPreview()
        harness.ui._widgets["labelRecgImg"] = label
        harness._scale_label = lambda target: target.setPixmap(
            harness._recgPixmap
        )
        harness.detection_status_changed = FailingStatus()

        first = harness._linear_series_controller.active_job
        invoke(
            harness,
            "_on_linear_series_extraction_finished",
            success_payload(first),
        )
        second = harness._linear_series_controller.active_job
        self.assertEqual(second.image_order, 2)
        self.assertEqual(len(harness.linear_series_extraction_requested.calls), 2)
        self.assertEqual(harness._active_worker_task, "linear_series")

        invoke(
            harness,
            "_on_linear_series_extraction_finished",
            success_payload(second),
        )
        self.assertEqual(
            harness._linear_series_controller.state.phase,
            LinearSeriesPhase.MAPPING,
        )
        self.assertFalse(harness._linear_series_controller.busy)
        self.assertIsNone(harness._active_worker_task)
        self.assertGreaterEqual(len(harness._linear_series_ui_warnings), 2)
        self.assertIn(
            "UI warnings:",
            harness.detection_status_changed.calls[-1][0],
        )
        self.assertEqual(len(harness.messages), 1)
        self.assertIs(harness.messages[0][0], detectmain.QMessageBox.warning)

    def test_final_progress_warning_is_in_the_single_final_summary(self):
        class FailingFinalProgress(WidgetStub):
            def __init__(self):
                super().__init__()
                self.completed_updates = 0

            def setValue(self, value):
                super().setValue(value)
                if value == 100:
                    self.completed_updates += 1
                    if self.completed_updates == 2:
                        raise RuntimeError("probe final progress")

        harness = FlowHarness([VIRTUAL_ROOT + "/only-image.png"])
        progress = FailingFinalProgress()
        harness.ui._widgets["progressBar"] = progress
        self.start_series(harness)
        controller = harness._linear_series_controller
        task = controller.active_job

        invoke(
            harness,
            "_on_linear_series_extraction_finished",
            success_payload(task),
        )

        warning = "final progress update failed: probe final progress"
        final_summaries = [
            call[0]
            for call in harness.detection_status_changed.calls
            if call[0].startswith("Linear series extraction completed.")
        ]
        self.assertEqual(
            controller.state.images[0].status,
            LinearImageStatus.COMPLETED,
        )
        self.assertEqual(controller.state.phase, LinearSeriesPhase.MAPPING)
        self.assertIsNone(controller.run_token)
        self.assertFalse(controller.busy)
        self.assertIsNone(harness._active_worker_task)
        self.assertEqual(harness._linear_series_ui_warnings, [warning])
        self.assertEqual(len(final_summaries), 1)
        self.assertEqual(final_summaries[0].count(warning), 1)
        self.assertIn("UI warnings: 1", final_summaries[0])
        self.assertIn("Total images: 1", final_summaries[0])
        self.assertIn("Successful images: 1", final_summaries[0])
        self.assertIn("Failed images: 0", final_summaries[0])
        self.assertIn("Valid samples: 1", final_summaries[0])
        self.assertIn("Sample errors: 0", final_summaries[0])
        self.assertEqual(len(harness.linear_series_extraction_requested.calls), 1)
        self.assertEqual(len(harness.messages), 1)
        self.assertIs(harness.messages[0][0], detectmain.QMessageBox.warning)
        self.assertEqual(harness.messages[0][1][2].count(warning), 1)

    def test_preview_then_final_progress_warnings_keep_order_in_summary(self):
        class FailingPreview(WidgetStub):
            def setPixmap(self, pixmap):
                super().setPixmap(pixmap)
                raise RuntimeError("probe preview")

        class FailingFinalProgress(WidgetStub):
            def __init__(self):
                super().__init__()
                self.completed_updates = 0

            def setValue(self, value):
                super().setValue(value)
                if value == 100:
                    self.completed_updates += 1
                    if self.completed_updates == 2:
                        raise RuntimeError("probe final progress")

        harness = FlowHarness([VIRTUAL_ROOT + "/only-image.png"])
        harness.ui._widgets["labelRecgImg"] = FailingPreview()
        harness.ui._widgets["progressBar"] = FailingFinalProgress()
        harness._scale_label = lambda target: target.setPixmap(
            harness._recgPixmap
        )
        self.start_series(harness)
        controller = harness._linear_series_controller
        task = controller.active_job

        invoke(
            harness,
            "_on_linear_series_extraction_finished",
            success_payload(task),
        )

        warnings = [
            "preview update failed: probe preview",
            "final progress update failed: probe final progress",
        ]
        final_summaries = [
            call[0]
            for call in harness.detection_status_changed.calls
            if call[0].startswith("Linear series extraction completed.")
        ]
        self.assertEqual(harness._linear_series_ui_warnings, warnings)
        self.assertEqual(len(final_summaries), 1)
        self.assertIn("UI warnings: 2", final_summaries[0])
        self.assertLess(
            final_summaries[0].index(warnings[0]),
            final_summaries[0].index(warnings[1]),
        )
        for warning in warnings:
            self.assertEqual(final_summaries[0].count(warning), 1)
        self.assertEqual(
            controller.state.images[0].status,
            LinearImageStatus.COMPLETED,
        )
        self.assertEqual(controller.state.phase, LinearSeriesPhase.MAPPING)
        self.assertIsNone(controller.run_token)
        self.assertFalse(controller.busy)
        self.assertIsNone(harness._active_worker_task)
        self.assertEqual(controller.summary(), {
            "total_images": 1,
            "successful_images": 1,
            "failed_images": 0,
            "valid_samples": 1,
            "sample_errors": 0,
        })
        self.assertEqual(len(harness.linear_series_extraction_requested.calls), 1)
        self.assertEqual(len(harness.messages), 1)
        self.assertIs(harness.messages[0][0], detectmain.QMessageBox.warning)
        public_summary = harness.messages[0][1][2]
        for warning in warnings:
            self.assertEqual(public_summary.count(warning), 1)

    def test_public_summary_groups_equal_warning_text_as_occurrences(self):
        class AlwaysFailingPreview(WidgetStub):
            def setPixmap(self, pixmap):
                super().setPixmap(pixmap)
                raise RuntimeError("same preview failure")

        harness = FlowHarness([
            VIRTUAL_ROOT + "/image1.png",
            VIRTUAL_ROOT + "/image2.png",
        ])
        harness.ui._widgets["labelRecgImg"] = AlwaysFailingPreview()
        harness._scale_label = lambda target: target.setPixmap(
            harness._recgPixmap
        )
        self.start_series(harness)
        for _ in range(2):
            task = harness._linear_series_controller.active_job
            invoke(
                harness,
                "_on_linear_series_extraction_finished",
                success_payload(task),
            )

        warning = "preview update failed: same preview failure"
        self.assertEqual(harness._linear_series_ui_warnings, [warning, warning])
        self.assertEqual(len(harness.messages), 1)
        self.assertIs(harness.messages[0][0], detectmain.QMessageBox.warning)
        public_summary = harness.messages[0][1][2]
        self.assertIn("UI warnings: 2", public_summary)
        self.assertEqual(public_summary.count(warning), 1)
        self.assertIn(warning + " (2 occurrences)", public_summary.splitlines())

    def test_public_summary_keeps_first_warning_order_and_single_event_plain(self):
        harness = FlowHarness([VIRTUAL_ROOT + "/only-image.png"])
        self.start_series(harness)
        invoke(
            harness,
            "_record_linear_series_ui_warning",
            "preview update",
            RuntimeError("first"),
        )
        invoke(
            harness,
            "_record_linear_series_ui_warning",
            "final status update",
            RuntimeError("second"),
        )
        task = harness._linear_series_controller.active_job
        invoke(
            harness,
            "_on_linear_series_extraction_finished",
            success_payload(task),
        )

        first = "preview update failed: first"
        second = "final status update failed: second"
        self.assertEqual(harness._linear_series_ui_warnings, [first, second])
        self.assertEqual(len(harness.messages), 1)
        public_summary = harness.messages[0][1][2]
        self.assertIn("UI warnings: 2", public_summary)
        self.assertLess(public_summary.index(first), public_summary.index(second))
        self.assertEqual(public_summary.count(first), 1)
        self.assertEqual(public_summary.count(second), 1)
        self.assertNotIn("occurrences", public_summary)

    def test_final_summary_dialog_type_title_statistics_and_repeat_matrix(self):
        cases = (
            (
                "success",
                ("success",),
                None,
                detectmain.QMessageBox.information,
                "Linear series extraction completed",
                (1, 1, 0, 1, 0),
            ),
            (
                "failed_image",
                ("success", "failure"),
                None,
                detectmain.QMessageBox.warning,
                "Linear series extraction partially completed",
                (2, 1, 1, 1, 1),
            ),
            (
                "sample_error",
                ("sample_error",),
                None,
                detectmain.QMessageBox.warning,
                "Linear series extraction partially completed",
                (1, 1, 0, 1, 1),
            ),
            (
                "ui_warning",
                ("success",),
                "synthetic UI warning",
                detectmain.QMessageBox.warning,
                "Linear series extraction partially completed",
                (1, 1, 0, 1, 0),
            ),
            (
                "all_failed",
                ("failure",),
                None,
                detectmain.QMessageBox.critical,
                "Linear series extraction failed",
                (1, 0, 1, 0, 1),
            ),
        )
        labels = (
            "Total images",
            "Successful images",
            "Failed images",
            "Valid samples",
            "Sample errors",
        )

        for name, outcomes, ui_warning, message_type, title, counts in cases:
            with self.subTest(name=name):
                harness = FlowHarness([
                    VIRTUAL_ROOT + "/{}-{}.png".format(name, index)
                    for index in range(1, len(outcomes) + 1)
                ])
                self.start_series(harness)
                if ui_warning is not None:
                    invoke(
                        harness,
                        "_record_linear_series_ui_warning",
                        "probe",
                        RuntimeError(ui_warning),
                    )
                for outcome in outcomes:
                    task = harness._linear_series_controller.active_job
                    if outcome == "success":
                        invoke(
                            harness,
                            "_on_linear_series_extraction_finished",
                            success_payload(task),
                        )
                    elif outcome == "sample_error":
                        invoke(
                            harness,
                            "_on_linear_series_extraction_finished",
                            success_payload(task, errors=[{
                                "reason": "one sample failed",
                                "error_type": "invalid_roi",
                                "spatial_order": 2,
                            }]),
                        )
                    else:
                        invoke(
                            harness,
                            "_on_linear_series_extraction_failed",
                            failure_payload(task, "image failed"),
                        )

                self.assertEqual(len(harness.messages), 1)
                function, arguments = harness.messages[0]
                self.assertIs(function, message_type)
                self.assertEqual(arguments[1], title)
                body = arguments[2]
                for label, count in zip(labels, counts):
                    self.assertIn("{}: {}".format(label, count), body)
                invoke(harness, "_finish_linear_series_run")
                self.assertEqual(len(harness.messages), 1)

    def test_partial_success_finishes_in_mapping_and_retains_draft(self):
        harness = FlowHarness([
            VIRTUAL_ROOT + "/image1.png",
            VIRTUAL_ROOT + "/image2.png",
        ])
        confirmed = harness._regression_result
        self.start_series(harness)
        first = harness._linear_series_controller.active_job
        invoke(
            harness,
            "_on_linear_series_extraction_finished",
            success_payload(first),
        )
        second = harness._linear_series_controller.active_job
        invoke(
            harness,
            "_on_linear_series_extraction_failed",
            failure_payload(second),
        )

        controller = harness._linear_series_controller
        self.assertEqual(controller.state.phase, LinearSeriesPhase.MAPPING)
        self.assertEqual(controller.summary(), {
            "total_images": 2,
            "successful_images": 1,
            "failed_images": 1,
            "valid_samples": 1,
            "sample_errors": 1,
        })
        self.assertEqual(len(controller.state.all_samples), 1)
        self.assertIs(controller.state.last_confirmed_result, confirmed)
        self.assertIs(harness._regression_result, confirmed)
        self.assertTrue(harness._regression_dirty)
        self.assertIsNone(controller.run_token)
        self.assertIsNone(harness._active_worker_task)
        self.assertEqual(len(harness.messages), 1)
        self.assertIs(harness.messages[0][0], detectmain.QMessageBox.warning)

    def test_all_failed_stays_failed_and_reports_only_one_summary_dialog(self):
        harness = FlowHarness([
            VIRTUAL_ROOT + "/image1.png",
            VIRTUAL_ROOT + "/image2.png",
        ])
        confirmed = harness._regression_result
        self.start_series(harness)
        first = harness._linear_series_controller.active_job
        invoke(
            harness,
            "_on_linear_series_extraction_failed",
            failure_payload(first, "first failed"),
        )
        self.assertEqual(harness.messages, [])
        second = harness._linear_series_controller.active_job
        invoke(
            harness,
            "_on_linear_series_extraction_failed",
            failure_payload(second, "second failed"),
        )

        controller = harness._linear_series_controller
        self.assertEqual(controller.state.phase, LinearSeriesPhase.FAILED)
        self.assertEqual(controller.summary(), {
            "total_images": 2,
            "successful_images": 0,
            "failed_images": 2,
            "valid_samples": 0,
            "sample_errors": 2,
        })
        self.assertIs(controller.state.last_confirmed_result, confirmed)
        self.assertIs(harness._regression_result, confirmed)
        self.assertEqual(len(harness.messages), 1)
        self.assertIsNone(controller.run_token)
        self.assertIsNone(harness._active_worker_task)

    def test_finish_none_keeps_busy_and_does_not_report_completion(self):
        class FinishNoneController:
            def __init__(self):
                self.calls = 0

            def finish_if_done(self):
                self.calls += 1
                return None

        harness = FlowHarness()
        controller = FinishNoneController()
        harness._linear_series_controller = controller
        harness._set_active_worker_task("linear_series")
        baseline = (
            tuple(harness.detection_status_changed.calls),
            tuple(harness.linear_series_status_changed.calls),
            harness.ui.progressBar.range_value,
            harness.ui.progressBar.value_value,
            len(harness.messages),
        )

        invoke(harness, "_finish_linear_series_run")

        self.assertEqual(controller.calls, 1)
        self.assertEqual(harness._active_worker_task, "linear_series")
        self.assertEqual((
            tuple(harness.detection_status_changed.calls),
            tuple(harness.linear_series_status_changed.calls),
            harness.ui.progressBar.range_value,
            harness.ui.progressBar.value_value,
            len(harness.messages),
        ), baseline)

    def test_last_result_reaches_finish_while_gui_is_still_busy(self):
        harness = FlowHarness([VIRTUAL_ROOT + "/only-image.png"])
        self.start_series(harness)
        controller = harness._linear_series_controller
        task = controller.active_job
        original_finish = controller.finish_if_done
        active_tasks_at_finish = []

        def observed_finish():
            active_tasks_at_finish.append(harness._active_worker_task)
            return original_finish()

        controller.finish_if_done = observed_finish
        try:
            invoke(
                harness,
                "_on_linear_series_extraction_finished",
                success_payload(task),
            )
        finally:
            controller.finish_if_done = original_finish

        self.assertEqual(active_tasks_at_finish, ["linear_series"])
        self.assertIsNone(controller.run_token)
        self.assertIsNone(harness._active_worker_task)
        self.assertEqual(len(harness.messages), 1)
        self.assertIs(harness.messages[0][0], detectmain.QMessageBox.information)
        finished = flow_snapshot(harness)
        invoke(
            harness,
            "_on_linear_series_extraction_finished",
            success_payload(task),
        )
        self.assertEqual(flow_snapshot(harness), finished)

    def test_pending_close_cancels_remaining_without_dispatch_or_ui_updates(self):
        harness = FlowHarness([
            VIRTUAL_ROOT + "/image1.png",
            VIRTUAL_ROOT + "/image2.png",
            VIRTUAL_ROOT + "/image3.png",
        ])
        self.start_series(harness)
        controller = harness._linear_series_controller
        first = controller.active_job
        harness._close_wait_pending = True
        dispatch_count = len(harness.linear_series_extraction_requested.calls)
        status_count = (
            len(harness.detection_status_changed.calls),
            len(harness.linear_series_status_changed.calls),
        )
        progress = (
            harness.ui.progressBar.range_value,
            harness.ui.progressBar.value_value,
        )

        invoke(
            harness,
            "_on_linear_series_extraction_finished",
            IdentityOnlyPayload(first.context()),
        )

        self.assertEqual(controller.state.phase, LinearSeriesPhase.CANCELLED)
        self.assertIsNone(controller.run_token)
        self.assertIsNone(controller.active_job)
        self.assertEqual(
            len(harness.linear_series_extraction_requested.calls),
            dispatch_count,
        )
        self.assertEqual((
            len(harness.detection_status_changed.calls),
            len(harness.linear_series_status_changed.calls),
        ), status_count)
        self.assertEqual((
            harness.ui.progressBar.range_value,
            harness.ui.progressBar.value_value,
        ), progress)
        self.assertIsNone(harness._active_worker_task)
        self.assertEqual(len(harness.worker_task_finished.calls), 1)
        self.assertEqual(harness.messages, [])

        closed_snapshot = flow_snapshot(harness)
        invoke(
            harness,
            "_on_linear_series_extraction_failed",
            failure_payload(first, "late after cancel"),
        )
        self.assertEqual(flow_snapshot(harness), closed_snapshot)

    def test_real_qthread_current_close_callbacks_shutdown_exactly_once(self):
        class CloseContinuationSignal(SignalSpy):
            def __init__(self, callback):
                super().__init__()
                self.callback = callback

            def emit(self, *arguments):
                super().emit(*arguments)
                self.callback()

        for result_type in ("success", "failure"):
            with self.subTest(result_type=result_type):
                harness = FlowHarness([
                    VIRTUAL_ROOT + "/current.png",
                    VIRTUAL_ROOT + "/remaining.png",
                ])
                self.start_series(harness)
                task = harness._linear_series_controller.active_job
                close = ActiveCloseHarness(harness)
                harness.worker_task_finished = CloseContinuationSignal(
                    close._on_worker_task_finished
                )
                harness.set_close_wait_pending(True)
                if result_type == "success":
                    callback = lambda payload: invoke(
                        harness,
                        "_on_linear_series_extraction_finished",
                        payload,
                    )
                    payload = success_payload(task)
                else:
                    callback = lambda payload: invoke(
                        harness,
                        "_on_linear_series_extraction_failed",
                        payload,
                    )
                    payload = failure_payload(task, "current failure")

                self.deliver_from_memory_thread(callback, payload)

                controller = harness._linear_series_controller
                self.assertEqual(controller.state.phase, LinearSeriesPhase.CANCELLED)
                self.assertIsNone(controller.run_token)
                self.assertIsNone(controller.active_job)
                self.assertIsNone(harness._active_worker_task)
                self.assertEqual(len(harness.linear_series_extraction_requested.calls), 1)
                self.assertEqual(harness.messages, [])
                self.assertEqual(len(harness.worker_task_finished.calls), 1)
                self.assertEqual(close.dispose_calls, 1)
                self.assertEqual(close.shutdown_calls, 1)
                self.assertEqual(harness.shutdown_request_calls, 1)
                close._on_worker_task_finished()
                self.assertEqual(close.shutdown_calls, 1)
                self.assertEqual(harness.shutdown_request_calls, 1)

    def test_real_qthread_stale_then_cancel_close_allows_one_summary(self):
        class CloseContinuationSignal(SignalSpy):
            def __init__(self, callback):
                super().__init__()
                self.callback = callback

            def emit(self, *arguments):
                super().emit(*arguments)
                self.callback()

        harness = FlowHarness([VIRTUAL_ROOT + "/only-image.png"])
        self.start_series(harness)
        task = harness._linear_series_controller.active_job
        close = ActiveCloseHarness(harness)
        harness.worker_task_finished = CloseContinuationSignal(
            close._on_worker_task_finished
        )
        harness.set_close_wait_pending(True)
        stale = failure_payload(task, "stale failure")
        stale["job_token"] += 1
        before = flow_snapshot(harness)

        self.deliver_from_memory_thread(
            lambda payload: invoke(
                harness, "_on_linear_series_extraction_failed", payload
            ),
            stale,
        )

        self.assertEqual(flow_snapshot(harness), before)
        self.assertTrue(close._close_wait_pending)
        self.assertEqual(close.shutdown_calls, 0)
        self.assertEqual(harness.shutdown_request_calls, 0)

        close._cancel_close_wait()
        self.assertFalse(close._close_wait_pending)
        self.assertFalse(harness._close_wait_pending)
        self.deliver_from_memory_thread(
            lambda payload: invoke(
                harness, "_on_linear_series_extraction_finished", payload
            ),
            success_payload(task),
        )

        self.assertEqual(len(harness.messages), 1)
        self.assertIs(harness.messages[0][0], detectmain.QMessageBox.information)
        self.assertEqual(len(harness.worker_task_finished.calls), 1)
        self.assertEqual(close.shutdown_calls, 0)
        self.assertEqual(harness.shutdown_request_calls, 0)
        finished = flow_snapshot(harness)
        invoke(
            harness,
            "_on_linear_series_extraction_finished",
            success_payload(task),
        )
        self.assertEqual(flow_snapshot(harness), finished)

    def test_shutdown_return_cancels_without_result_view_or_summary(self):
        harness = FlowHarness([VIRTUAL_ROOT + "/only-image.png"])
        self.start_series(harness)
        controller = harness._linear_series_controller
        task = controller.active_job
        payload = success_payload(task)
        harness._shutdown_requested = True
        before_view = (
            harness._recgPixmap,
            harness.ui.tabviewRecg.model(),
            harness.ui.labelRecgImg.text(),
        )

        invoke(harness, "_on_linear_series_extraction_finished", payload)

        self.assertEqual(controller.state.phase, LinearSeriesPhase.CANCELLED)
        self.assertIsNone(controller.run_token)
        self.assertIsNone(harness._active_worker_task)
        self.assertEqual(harness.messages, [])
        self.assertEqual((
            harness._recgPixmap,
            harness.ui.tabviewRecg.model(),
            harness.ui.labelRecgImg.text(),
        ), before_view)

    def test_mapping_close_asks_once_and_reject_is_a_complete_no_op(self):
        harness = MappingCloseHarness()
        event = CloseEventStub()
        draft = harness._detectMain.mapping_draft

        with mock.patch.object(
            detectionwindow.QMessageBox,
            "question",
            return_value=detectionwindow.QMessageBox.No,
        ) as question:
            harness.closeEvent(event)

        question.assert_called_once()
        self.assertEqual(event.ignored, 1)
        self.assertEqual(event.accepted, 0)
        self.assertEqual(harness.shutdown_calls, 0)
        self.assertEqual(harness._detectMain.discard_calls, 0)
        self.assertIs(harness._detectMain.mapping_draft, draft)
        self.assertIs(harness._detectMain.confirmed, harness.confirmed)

    def test_mapping_close_confirmation_discards_only_draft_then_shuts_down(self):
        harness = MappingCloseHarness()
        event = CloseEventStub()

        with mock.patch.object(
            detectionwindow.QMessageBox,
            "question",
            return_value=detectionwindow.QMessageBox.Yes,
        ) as question:
            harness.closeEvent(event)

        question.assert_called_once()
        self.assertEqual(event.ignored, 1)
        self.assertEqual(event.accepted, 0)
        self.assertEqual(harness._detectMain.discard_calls, 1)
        self.assertIsNone(harness._detectMain.mapping_draft)
        self.assertIs(harness._detectMain.confirmed, harness.confirmed)
        self.assertEqual(harness.shutdown_calls, 1)

    def test_close_prompt_discloses_all_seven_unsaved_combinations(self):
        cases = (
            ((True, False, False), ("Unconfirmed Linear Series Mapping draft",)),
            ((False, True, False), ("Unsaved Linear result",)),
            ((False, False, True), ("Unsaved Detection result",)),
            ((True, True, False), (
                "Unconfirmed Linear Series Mapping draft", "Unsaved Linear result"
            )),
            ((True, False, True), (
                "Unconfirmed Linear Series Mapping draft", "Unsaved Detection result"
            )),
            ((False, True, True), (
                "Unsaved Linear result", "Unsaved Detection result"
            )),
            ((True, True, True), (
                "Unconfirmed Linear Series Mapping draft",
                "Unsaved Linear result",
                "Unsaved Detection result",
            )),
        )
        for flags, expected in cases:
            for answer in (detectionwindow.QMessageBox.No,
                           detectionwindow.QMessageBox.Yes):
                with self.subTest(flags=flags, answer=answer):
                    mapping, linear_dirty, detection_dirty = flags
                    harness = MappingCloseHarness(
                        mapping=mapping,
                        linear_dirty=linear_dirty,
                        detection_dirty=detection_dirty,
                    )
                    event = CloseEventStub()
                    draft = harness._detectMain.mapping_draft
                    confirmed = harness._detectMain.confirmed
                    with mock.patch.object(
                        detectionwindow.QMessageBox,
                        "question",
                        return_value=answer,
                    ) as question:
                        harness.closeEvent(event)
                        if answer == detectionwindow.QMessageBox.Yes:
                            harness.closeEvent(event)
                    question.assert_called_once()
                    prompt = question.call_args.args[2]
                    for category in expected:
                        self.assertIn("- " + category, prompt)
                    for absent in {
                        "Unconfirmed Linear Series Mapping draft",
                        "Unsaved Linear result",
                        "Unsaved Detection result",
                    } - set(expected):
                        self.assertNotIn(absent, prompt)
                    self.assertIs(harness._detectMain.confirmed, confirmed)
                    if answer == detectionwindow.QMessageBox.No:
                        self.assertIs(harness._detectMain.mapping_draft, draft)
                        self.assertEqual(harness._detectMain.discard_calls, 0)
                        self.assertEqual(harness.shutdown_calls, 0)
                    else:
                        self.assertEqual(
                            harness._detectMain.discard_calls,
                            int(mapping),
                        )
                        self.assertEqual(harness.shutdown_calls, 1)


if __name__ == "__main__":
    unittest.main()
