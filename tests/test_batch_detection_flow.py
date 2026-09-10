import io
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace

import numpy as np


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Peng1.0_GUI"))

from batch_detection_controller import BatchDetectionController  # noqa: E402
from batch_state import DetectionScope, ImageStatus, NumberingMode  # noqa: E402
from detectmain import DetectMain  # noqa: E402
import detectmain  # noqa: E402
import detectionwindow  # noqa: E402
import camera as camera_module  # noqa: E402
from camera import Camera  # noqa: E402
from PySide6.QtCore import QObject, Qt, Signal  # noqa: E402
from PySide6.QtGui import QPixmap, QColor, QFont  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication,
    QLabel,
    QMainWindow,
    QSplitter,
    QTabWidget,
    QWidget,
)


PROTECTED_DETECTION_DIRECTORY = ROOT / "HT-Detector_Peng" / "custom" / \
    "linear_detection" / "detection"
DETECTION_WEIGHT_PATH = ROOT / "HT-Detector_Peng" / "weights" / \
    "cuvette_Peng" / "yolov8n_train" / "weights" / "best.pt"


def lexical_path(path):
    return os.path.normcase(os.path.normpath(os.fspath(path)))


class DetectionPathAccessGuard:
    def __init__(self, allowed_files=(), allowed_directories=()):
        self.allowed_files = {lexical_path(path) for path in allowed_files}
        self.allowed_directories = {
            lexical_path(path) for path in allowed_directories
        }
        self.protected_directory = lexical_path(PROTECTED_DETECTION_DIRECTORY)
        self.protected_names = {
            "1..10.jpg",
            *("{}.jpg".format(number) for number in range(1, 11)),
        }
        self.default_detection_directory_accesses = 0
        self.protected_image_accesses = 0

    def _reject_protected_path(self, path):
        candidate = lexical_path(path)
        if candidate == self.protected_directory:
            self.default_detection_directory_accesses += 1
            raise AssertionError("The default detection directory was accessed.")
        if (
            os.path.dirname(candidate) == self.protected_directory
            and os.path.basename(candidate).casefold() in self.protected_names
        ):
            self.protected_image_accesses += 1
            raise AssertionError("A protected detection image was accessed.")
        return candidate

    def is_file(self, path):
        return self._reject_protected_path(path) in self.allowed_files

    def is_dir(self, path):
        return self._reject_protected_path(path) in self.allowed_directories


def invoke_detection_selection(harness, selected_names):
    with TemporaryDirectory(prefix="ht-detector-selection-") as temporary_directory:
        initial_directory = Path(temporary_directory)
        selected_paths = [
            str(initial_directory / Path(name).name) for name in selected_names
        ]
        guard = DetectionPathAccessGuard(
            allowed_files=[DETECTION_WEIGHT_PATH, *selected_paths],
            allowed_directories=[initial_directory],
        )
        with patch(
            "detectmain._dialog_initial_directory", return_value=initial_directory
        ), patch(
            "detectmain.QFileDialog.getOpenFileNames",
            return_value=(selected_paths, "Images"),
        ), patch.object(
            Path, "is_file", autospec=True, side_effect=guard.is_file
        ), patch.object(
            Path, "is_dir", autospec=True, side_effect=guard.is_dir
        ):
            DetectMain._select_detection_image(harness)
    return guard


def sample_result(image_order, filename, no_in_image, batch_no):
    return {
        "image_order": image_order,
        "source_file": filename,
        "cuvette_box": (0.0, 0.0, 20.0, 40.0),
        "liquid_box": (4.0, 10.0, 16.0, 35.0),
        "roi_box": (6, 15, 14, 30),
        "red": 1.0,
        "green": 2.0,
        "blue": 3.0,
        "no_in_image": no_in_image,
        "batch_no": batch_no,
        "status": "valid",
        "warnings": [],
    }


def payload(task, count=1, errors=None):
    return {
        "run_token": task.run_token,
        "job_token": task.job_token,
        "sample_results": [
            sample_result(task.image_order, task.source_file, index,
                          task.batch_start_no + index - 1)
            for index in range(1, count + 1)
        ],
        "sample_errors": list(errors or []),
    }


def runtime_payload(task, count=1, errors=None, warnings=None):
    result = payload(task, count=count, errors=errors)
    result.update({
        "source_path": task.path,
        "image": np.zeros((12, 16, 3), dtype=np.uint8),
        "targets": [
            {
                "No.": task.display_start_no + index - 1,
                "Con.": 0.5,
                "Red": 1.0,
                "Green": 2.0,
                "Blue": 3.0,
                "cuvette_box": (0.0, 0.0, 10.0, 10.0),
                "liquid_box": (2.0, 2.0, 8.0, 8.0),
                "rgb_roi": (3, 3, 7, 7),
            }
            for index in range(1, count + 1)
        ],
        "warnings": list(warnings or []),
    })
    return result


def sample_error(task, error_type, reason):
    return {
        "image_order": task.image_order,
        "source_file": task.source_file,
        "error_type": error_type,
        "reason": reason,
        "related_boxes": [],
        "related_cuvette_boxes": [],
        "related_liquid_boxes": [],
        "position": None,
        "no_in_image": None,
        "batch_no": None,
    }


class BatchDetectionControllerTests(unittest.TestCase):
    def setUp(self):
        self.controller = BatchDetectionController()

    def select(self, paths, scope="entire_batch", numbering="per_image"):
        self.controller.set_options(scope, numbering)
        return self.controller.replace_images(paths)

    def test_multiple_images_are_naturally_sorted_and_replace_old_batch(self):
        state = self.select(["image10.png", "image2.png", "image1.png"])
        self.assertEqual([item.original_filename for item in state.images],
                         ["image1.png", "image2.png", "image10.png"])
        replacement = self.controller.replace_images(["new.png"])
        self.assertEqual([item.original_filename for item in replacement.images], ["new.png"])

    def test_current_image_dispatches_only_one_task(self):
        self.select(["b.png", "a.png"], scope="current_image", numbering="continuous")
        task = self.controller.begin()
        self.assertEqual(task.source_file, "a.png")
        self.controller.accept_payload(payload(task))
        self.assertIsNone(self.controller.next_task())
        self.assertEqual(self.controller.finish_if_done()["total_images"], 2)
        self.assertEqual(self.controller.state.images[1].status, ImageStatus.PENDING)

    def test_entire_batch_is_strictly_sequential(self):
        self.select(["image2.png", "image1.png"])
        first = self.controller.begin()
        self.assertIsNone(self.controller.next_task())
        self.controller.accept_payload(payload(first))
        second = self.controller.next_task()
        self.assertEqual([first.source_file, second.source_file], ["image1.png", "image2.png"])

    def test_each_dispatched_job_has_a_unique_identity_even_for_duplicate_paths(self):
        self.select(["same.png", "same.png"])
        first = self.controller.begin()
        self.controller.accept_payload(payload(first))
        second = self.controller.next_task()

        self.assertEqual(first.run_token, second.run_token)
        self.assertNotEqual(first.job_token, second.job_token)
        self.assertEqual(first.path, second.path)
        self.assertEqual(first.context()["job_token"], first.job_token)
        self.assertIs(self.controller.active_job, second)

    def test_late_previous_job_signals_leave_the_active_job_unchanged(self):
        self.select(["a.png", "b.png"])
        first = self.controller.begin()
        first_result = payload(first)
        self.assertTrue(self.controller.accept_payload(first_result))
        self.assertFalse(self.controller.accept_payload(first_result))

        second = self.controller.next_task()
        active_job = self.controller.active_job
        queued_orders = tuple(self.controller._queue)
        self.assertFalse(self.controller.matches_active_result(first_result))
        self.assertFalse(self.controller.accept_payload(first_result))
        self.assertFalse(self.controller.accept_failure(
            first.run_token, first.job_token, "late failure"
        ))

        self.assertIs(self.controller.active_job, active_job)
        self.assertEqual(self.controller.active_job.image_order, 2)
        self.assertEqual(tuple(self.controller._queue), queued_orders)
        self.assertEqual(self.controller.state.images[1].status, ImageStatus.PROCESSING)
        self.assertEqual(self.controller.state.images[1].samples, [])
        self.assertTrue(self.controller.active)
        self.assertIsNone(self.controller.next_task())

    def test_worker_failure_continues_and_summary_distinguishes_counts(self):
        self.select(["a.png", "b.png"])
        first = self.controller.begin()
        self.assertTrue(self.controller.accept_failure(
            first.run_token, first.job_token, "broken image"
        ))
        second = self.controller.next_task()
        partial_error = {
            "image_order": 2, "source_file": "b.png",
            "error_type": "invalid_roi", "reason": "bad roi",
            "related_boxes": [], "related_cuvette_boxes": [],
            "related_liquid_boxes": [], "position": None,
            "no_in_image": None, "batch_no": None,
        }
        self.controller.accept_payload(payload(second, 2, [partial_error]))
        self.assertEqual(self.controller.finish_if_done(), {
            "total_images": 2, "successful_images": 1, "failed_images": 1,
            "valid_samples": 2, "sample_errors": 1,
        })

    def test_payload_image_failed_without_samples_marks_image_failed(self):
        self.select(["a.png"])
        task = self.controller.begin()
        result = payload(task, 0, [
            sample_error(task, "image_failed", "decoder exploded")
        ])
        result["targets"] = []
        self.assertTrue(self.controller.accept_payload(result))
        self.assertEqual(self.controller.state.images[0].status, ImageStatus.FAILED)
        self.assertEqual(
            self.controller.state.images[0].errors[0].reason, "decoder exploded"
        )
        self.assertIsNone(self.controller.state.last_batch_result)

    def test_only_sample_errors_marks_image_failed_and_preserves_every_error(self):
        self.select(["a.png"])
        task = self.controller.begin()
        errors = [
            sample_error(task, "invalid_roi", "ROI outside image"),
            sample_error(task, "measurement_failed", "RGB unavailable"),
        ]
        result = payload(task, 0, errors)
        result["targets"] = []

        self.assertTrue(self.controller.accept_payload(result))
        image = self.controller.state.images[0]
        self.assertEqual(image.status, ImageStatus.FAILED)
        self.assertEqual(
            [error.reason for error in image.errors],
            ["ROI outside image", "RGB unavailable"],
        )
        self.assertEqual(self.controller.summary()["sample_errors"], 2)
        self.assertIsNone(self.controller.state.last_batch_result)

    def test_valid_samples_with_sample_errors_is_completed_and_counted(self):
        self.select(["a.png"])
        task = self.controller.begin()
        result = payload(task, 1, [sample_error(task, "invalid_roi", "bad ROI")])

        self.assertTrue(self.controller.accept_payload(result))
        image = self.controller.state.images[0]
        self.assertEqual(image.status, ImageStatus.COMPLETED)
        self.assertEqual(len(image.samples), 1)
        self.assertEqual(len(image.errors), 1)
        self.assertIs(self.controller.state.last_batch_result, result)
        self.assertEqual(self.controller.summary()["sample_errors"], 1)

    def test_continuous_and_per_image_display_starts(self):
        self.select(["a.png", "b.png"], numbering="continuous")
        first = self.controller.begin()
        self.controller.accept_payload(payload(first, 2))
        second = self.controller.next_task()
        self.assertEqual((first.display_start_no, second.display_start_no), (1, 3))
        self.controller.accept_failure(second.run_token, second.job_token, "fail")
        self.controller.finish_if_done()
        self.select(["a.png", "b.png"], numbering="per_image")
        first = self.controller.begin()
        self.controller.accept_payload(payload(first, 2))
        second = self.controller.next_task()
        self.assertEqual((first.display_start_no, second.display_start_no), (1, 1))
        self.assertEqual(second.batch_start_no, 3)

    def test_error_samples_do_not_advance_next_batch_start(self):
        self.select(["a.png", "b.png"], numbering="continuous")
        first = self.controller.begin()
        self.controller.accept_payload(payload(first, 1, [{
            "image_order": 1, "source_file": "a.png", "error_type": "invalid_roi",
            "reason": "bad", "related_boxes": [], "related_cuvette_boxes": [],
            "related_liquid_boxes": [], "position": None,
            "no_in_image": None, "batch_no": None,
        }]))
        self.assertEqual(self.controller.next_task().batch_start_no, 2)

    def test_busy_reentry_and_replacement_are_rejected(self):
        self.select(["a.png"])
        first = self.controller.begin()
        self.assertIsNone(self.controller.begin())
        with self.assertRaises(RuntimeError):
            self.controller.replace_images(["b.png"])
        with self.assertRaises(RuntimeError):
            self.controller.set_options("current_image", "per_image")
        self.assertEqual(first.source_file, "a.png")

    def test_stale_tokens_cannot_update_current_run(self):
        self.select(["a.png"])
        task = self.controller.begin()
        active_job = self.controller.active_job
        wrong_job = payload(task)
        wrong_job["job_token"] += 100
        self.assertFalse(self.controller.accept_payload(wrong_job))
        self.assertFalse(self.controller.accept_failure(
            wrong_job["run_token"], wrong_job["job_token"], "wrong job"
        ))
        stale = payload(task)
        stale["run_token"] += 100
        self.assertFalse(self.controller.accept_payload(stale))
        self.assertFalse(self.controller.accept_failure(
            stale["run_token"], stale["job_token"], "old"
        ))
        self.assertIs(self.controller.active_job, active_job)
        self.assertEqual(self.controller.state.images[0].status, ImageStatus.PROCESSING)

    def test_new_run_clears_old_result_and_failed_run_cannot_restore_it(self):
        self.select(["a.png"])
        first = self.controller.begin()
        first_result = payload(first, 2)
        self.controller.accept_payload(first_result)
        self.assertIs(self.controller.state.last_batch_result, first_result)
        self.controller.finish_if_done()

        second = self.controller.begin()
        self.assertIsNone(self.controller.state.last_batch_result)
        self.assertEqual(self.controller.state.images[0].samples, [])
        self.assertFalse(self.controller.accept_payload(first_result))
        self.assertIsNone(self.controller.state.last_batch_result)
        self.assertTrue(self.controller.accept_failure(
            second.run_token, second.job_token, "second run failed"
        ))
        self.controller.finish_if_done()
        self.assertIsNone(self.controller.state.last_batch_result)
        self.assertFalse(self.controller.accept_payload(first_result))
        self.assertIsNone(self.controller.state.last_batch_result)

    def test_rejected_begin_does_not_clear_existing_result(self):
        marker = {"old": True}
        self.controller.state.last_batch_result = marker
        self.assertIsNone(self.controller.begin())
        self.assertIs(self.controller.state.last_batch_result, marker)

        self.select(["a.png"])
        task = self.controller.begin()
        accepted = payload(task)
        self.controller.accept_payload(accepted)
        self.assertIs(self.controller.state.last_batch_result, accepted)
        self.assertIsNone(self.controller.begin())
        self.assertIs(self.controller.state.last_batch_result, accepted)


class FakeProgressBar:
    def __init__(self):
        self.value = None

    def setRange(self, *_args):
        pass

    def setValue(self, value):
        self.value = value


class FakePixmap:
    def isNull(self):
        return False


class FakeButton:
    def __init__(self):
        self.text = None
        self.enabled = None

    def setText(self, text):
        self.text = text

    def setEnabled(self, enabled):
        self.enabled = bool(enabled)

    def isEnabled(self):
        return self.enabled


class FakeLabel:
    def __init__(self):
        self.cleared = False
        self._text = ""

    def setText(self, text):
        self._text = text

    def text(self):
        return self._text

    def clear(self):
        self.cleared = True
        self._text = ""


class FakeHeader:
    def hide(self):
        pass

    def setSectionResizeMode(self, *_args):
        pass


class FakeTableView:
    def __init__(self):
        self._model = None
        self._vertical_header = FakeHeader()
        self._horizontal_header = FakeHeader()

    def setModel(self, model):
        self._model = model

    def model(self):
        return self._model

    def verticalHeader(self):
        return self._vertical_header

    def horizontalHeader(self):
        return self._horizontal_header


class FakeSignal:
    def __init__(self):
        self.values = []

    def emit(self, *values):
        self.values.append(values)


class SelectionHarness:
    def __init__(self):
        self._close_wait_pending = False
        self._shutdown_requested = False
        self._active_worker_task = None
        self._detection_dirty = False
        self._detection_result = None
        self._last_completed_result_type = None
        self._batch_controller = BatchDetectionController()
        self._detection_weight_path = None
        self.ui = type("Ui", (), {"progressBar": FakeProgressBar()})()
        self.dispatched = []

    def _update_save_button(self):
        pass

    def _clear_detection_results_for_new_run(self):
        self._detection_result = None
        self._detection_dirty = False
        if self._last_completed_result_type == "detection":
            self._last_completed_result_type = None
        self._update_save_button()

    def _set_active_worker_task(self, task):
        self._active_worker_task = task

    def _dispatch_detection_task(self, task):
        self.dispatched.append(task)


class DetectionSelectionCancelTests(unittest.TestCase):
    def invoke(self, harness, selected):
        guard = invoke_detection_selection(harness, selected)
        self.assertEqual(guard.default_detection_directory_accesses, 0)
        self.assertEqual(guard.protected_image_accesses, 0)

    def test_cancel_preserves_state_and_does_not_become_busy(self):
        harness = SelectionHarness()
        original = harness._batch_controller.state
        previous_result = {"previous": True}
        original.last_batch_result = previous_result
        harness._detection_result = previous_result
        self.invoke(harness, [])
        self.assertIs(harness._batch_controller.state, original)
        self.assertIs(harness._batch_controller.state.last_batch_result, previous_result)
        self.assertIs(harness._detection_result, previous_result)
        self.assertIsNone(harness._active_worker_task)
        self.assertEqual(harness.dispatched, [])


class RunUiHarness(SelectionHarness):
    def __init__(self):
        super().__init__()
        self.detection_status_changed = FakeSignal()
        self.detection_requested = FakeSignal()
        self.messages = []

    @property
    def batch_state(self):
        return self._batch_controller.state

    def _show_message_safely(self, _function, _parent, title, message):
        self.messages.append((title, message))


class DetectionHandlerHarness(RunUiHarness):
    _normalized_detection_targets = staticmethod(
        DetectMain._normalized_detection_targets
    )
    _validated_source_path = staticmethod(DetectMain._validated_source_path)
    _normalized_detection_runtime_payload = (
        DetectMain._normalized_detection_runtime_payload
    )
    _validated_detection_export_payload = (
        DetectMain._validated_detection_export_payload
    )
    _has_valid_detection_export = DetectMain._has_valid_detection_export
    _pending_save_type = DetectMain._pending_save_type
    _update_save_button = DetectMain._update_save_button
    _detection_failure_message = staticmethod(DetectMain._detection_failure_message)
    _clear_detection_results_for_new_run = (
        DetectMain._clear_detection_results_for_new_run
    )
    _advance_detection_run = DetectMain._advance_detection_run
    _finish_detection_run = DetectMain._finish_detection_run

    def __init__(self):
        super().__init__()
        self.ui.tabviewRecg = FakeTableView()
        self.ui.labelRecgImg = FakeLabel()
        self.ui.pushButton_8 = FakeButton()
        self._recgPixmap = None
        self._regression_dirty = False

    def _bgr_image_to_pixmap(self, _image):
        return FakePixmap()

    def _build_table_model(self, headers, rows):
        return tuple(headers), tuple(rows)

    def _populate_tableview(self, table, headers, rows):
        table.setModel((tuple(headers), tuple(rows)))

    def _scale_label(self, _label):
        pass

    def _safe_detection_stem(self, source_path):
        return DetectMain._safe_detection_stem(source_path)

    def _set_active_worker_task(self, task):
        self._active_worker_task = task
        self._update_save_button()


class DetectionHandlerTests(unittest.TestCase):
    def create_run(self, scope="entire_batch", image_names=("image.png",)):
        harness = DetectionHandlerHarness()
        harness._batch_controller.set_options(scope, "per_image")
        harness._batch_controller.replace_images(image_names)
        harness._active_worker_task = "detection"
        task = harness._batch_controller.begin()
        harness._dispatch_detection_task(task)
        return harness, task

    def test_real_handler_quietly_rejects_late_success_from_previous_job(self):
        harness, first = self.create_run(image_names=("image1.png", "image2.png"))
        first_result = runtime_payload(first)
        DetectMain._on_detection_finished(harness, first_result)
        second = harness._batch_controller.active_job
        self.assertEqual(second.image_order, 2)

        active_job = harness._batch_controller.active_job
        queued_orders = tuple(harness._batch_controller._queue)
        current_result = harness._detection_result
        second_samples = list(harness.batch_state.images[1].samples)
        second_errors = list(harness.batch_state.images[1].errors)
        busy = harness._active_worker_task
        dispatch_count = len(harness.dispatched)
        message_count = len(harness.messages)

        self.assertFalse(harness._batch_controller.matches_active_result(first_result))
        with patch.object(
            harness._batch_controller,
            "accept_payload",
            wraps=harness._batch_controller.accept_payload,
        ) as accept_payload:
            DetectMain._on_detection_finished(harness, first_result)
        accept_payload.assert_not_called()

        self.assertIs(harness._batch_controller.active_job, active_job)
        self.assertEqual(harness._batch_controller.active_job.image_order, 2)
        self.assertEqual(tuple(harness._batch_controller._queue), queued_orders)
        self.assertIs(harness._detection_result, current_result)
        self.assertEqual(harness.batch_state.images[1].samples, second_samples)
        self.assertEqual(harness.batch_state.images[1].errors, second_errors)
        self.assertEqual(harness._active_worker_task, busy)
        self.assertEqual(len(harness.dispatched), dispatch_count)
        self.assertEqual(len(harness.messages), message_count)

        for guard_name in ("_close_wait_pending", "_shutdown_requested"):
            with self.subTest(guard_name=guard_name):
                setattr(harness, guard_name, True)
                DetectMain._on_detection_finished(harness, first_result)
                DetectMain._on_detection_failed(harness, {
                    "run_token": first.run_token,
                    "job_token": first.job_token,
                    "message": "late failure while closing",
                })
                setattr(harness, guard_name, False)
                self.assertIs(harness._batch_controller.active_job, active_job)
                self.assertEqual(harness._active_worker_task, busy)
                self.assertEqual(len(harness.dispatched), dispatch_count)
                self.assertEqual(len(harness.messages), message_count)

    def test_entire_batch_structured_failure_advances_to_second_image(self):
        harness, first = self.create_run(
            scope="entire_batch", image_names=("image1.png", "image2.png")
        )
        reason = "first image could not be decoded"
        DetectMain._on_detection_finished(harness, runtime_payload(
            first,
            count=0,
            errors=[sample_error(first, "image_failed", reason)],
        ))

        self.assertEqual(harness.batch_state.images[0].status, ImageStatus.FAILED)
        self.assertEqual(harness.batch_state.images[0].errors[0].reason, reason)
        self.assertEqual(harness._batch_controller.active_job.image_order, 2)
        self.assertEqual(harness._active_worker_task, "detection")
        self.assertEqual(harness.messages, [])

        second = harness._batch_controller.active_job
        DetectMain._on_detection_finished(harness, runtime_payload(second))
        self.assertEqual(harness.batch_state.images[1].status, ImageStatus.COMPLETED)
        self.assertEqual(
            sum(title == "Detection" for title, _ in harness.messages), 1
        )
        self.assertEqual(
            sum(title == "Detection warning" for title, _ in harness.messages), 0
        )
        self.assertIn("Failed images: 1", harness.messages[0][1])

    def test_image_failed_empty_targets_preserves_reason_and_dialog_contract(self):
        reason = "decoder returned the original worker failure"
        for scope, expected_warning_count, expected_summary_count in (
            ("entire_batch", 0, 1),
            ("current_image", 1, 0),
        ):
            with self.subTest(scope=scope):
                harness, task = self.create_run(scope=scope)
                result = runtime_payload(
                    task, count=0,
                    errors=[sample_error(task, "image_failed", reason)],
                )
                DetectMain._on_detection_finished(harness, result)

                image = harness.batch_state.images[0]
                self.assertEqual(image.status, ImageStatus.FAILED)
                self.assertEqual(image.errors[0].reason, reason)
                self.assertIsNone(harness.batch_state.last_batch_result)
                self.assertIsNone(harness._detection_result)
                self.assertEqual(
                    sum(title == "Detection warning" for title, _ in harness.messages),
                    expected_warning_count,
                )
                self.assertEqual(
                    sum(title == "Detection" for title, _ in harness.messages),
                    expected_summary_count,
                )
                if scope == "current_image":
                    self.assertEqual(harness.messages[0][1], reason)

    def test_only_sample_errors_and_partial_success_use_runtime_validation(self):
        harness, task = self.create_run()
        errors = [
            sample_error(task, "invalid_roi", "ROI outside image"),
            sample_error(task, "measurement_failed", "RGB unavailable"),
        ]
        DetectMain._on_detection_finished(
            harness, runtime_payload(task, count=0, errors=errors)
        )
        image = harness.batch_state.images[0]
        self.assertEqual(image.status, ImageStatus.FAILED)
        self.assertEqual([error.reason for error in image.errors], [
            "ROI outside image", "RGB unavailable",
        ])
        self.assertIn("Sample errors: 2", harness.messages[0][1])
        self.assertIsNone(harness.batch_state.last_batch_result)

        harness, task = self.create_run()
        partial_error = sample_error(task, "invalid_roi", "one bad sample")
        DetectMain._on_detection_finished(
            harness, runtime_payload(task, count=1, errors=[partial_error])
        )
        image = harness.batch_state.images[0]
        self.assertEqual(image.status, ImageStatus.COMPLETED)
        self.assertEqual(len(image.samples), 1)
        self.assertEqual(len(image.errors), 1)
        self.assertIsNone(image.errors[0].no_in_image)
        self.assertIsNone(image.errors[0].batch_no)
        self.assertEqual(harness._detection_result["targets"][0]["No."], 1)
        self.assertIsNotNone(harness.batch_state.last_batch_result)
        self.assertIn("Valid samples: 1", harness.messages[0][1])
        self.assertIn("Sample errors: 1", harness.messages[0][1])

    def test_save_validation_alone_rejects_zero_target_runtime_payload(self):
        harness, task = self.create_run(scope="current_image")
        result = runtime_payload(
            task, count=0,
            errors=[sample_error(task, "image_failed", "read failed")],
        )
        normalized = DetectMain._normalized_detection_runtime_payload(harness, result)
        self.assertEqual(normalized["targets"], [])
        self.assertEqual(normalized["sample_errors"][0]["reason"], "read failed")
        with self.assertRaisesRegex(
            ValueError, "At least one valid detection target is required"
        ):
            DetectMain._validated_detection_export_payload(harness, result)

    def test_new_confirmed_run_clears_gui_and_controller_result_permanently(self):
        harness, first = self.create_run()
        first_result = runtime_payload(first)
        DetectMain._on_detection_finished(harness, first_result)
        self.assertIsNotNone(harness._detection_result)
        self.assertIsNotNone(harness.batch_state.last_batch_result)
        self.assertTrue(harness.ui.pushButton_8.isEnabled())

        harness._detection_dirty = False
        old_pixmap = harness._recgPixmap
        old_model = harness.ui.tabviewRecg.model()
        self.assertIsNotNone(old_pixmap)
        self.assertIsNotNone(old_model)
        guard = invoke_detection_selection(harness, ["replacement.png"])
        second = harness._batch_controller.active_job
        self.assertEqual(guard.default_detection_directory_accesses, 0)
        self.assertEqual(guard.protected_image_accesses, 0)
        self.assertIsNone(harness._detection_result)
        self.assertIsNone(harness.batch_state.last_batch_result)
        self.assertFalse(harness.ui.pushButton_8.isEnabled())
        self.assertIsNone(harness._recgPixmap)
        self.assertNotEqual(harness.ui.tabviewRecg.model(), old_model)
        self.assertEqual(harness.ui.tabviewRecg.model()[1], ())
        self.assertTrue(harness.ui.labelRecgImg.cleared)

        DetectMain._on_detection_failed(harness, {
            "run_token": second.run_token,
            "job_token": second.job_token,
            "message": "second run failed",
        })
        self.assertIsNone(harness._detection_result)
        self.assertIsNone(harness.batch_state.last_batch_result)
        self.assertFalse(harness.ui.pushButton_8.isEnabled())

        DetectMain._on_detection_finished(harness, first_result)
        self.assertIsNone(harness._detection_result)
        self.assertIsNone(harness.batch_state.last_batch_result)
        self.assertFalse(harness.ui.pushButton_8.isEnabled())


class DetectionProgressTests(unittest.TestCase):
    def test_progress_and_entire_batch_summary(self):
        harness = RunUiHarness()
        harness._batch_controller.set_options("entire_batch", "per_image")
        harness._batch_controller.replace_images(["image2.png", "image1.png"])
        harness._detection_weight_path = "unused.pt"
        harness._active_worker_task = "detection"
        first = harness._batch_controller.begin()
        DetectMain._dispatch_detection_task(harness, first)
        self.assertTrue(harness.detection_status_changed.values[-1][0].startswith(
            "Detecting 1/2: image1.png | Imported images: 2 | Planned this run: 2"
        ))
        harness._batch_controller.accept_payload(payload(first, 1))
        second = harness._batch_controller.next_task()
        DetectMain._dispatch_detection_task(harness, second)
        self.assertTrue(harness.detection_status_changed.values[-1][0].startswith(
            "Detecting 2/2: image2.png | Imported images: 2 | Planned this run: 2"
        ))
        harness._batch_controller.accept_failure(
            second.run_token, second.job_token, "broken"
        )
        DetectMain._finish_detection_run(harness)
        self.assertIn("Imported images: 2", harness.messages[0][1])
        self.assertIn("Completed this run: 2", harness.messages[0][1])
        self.assertIn("Successful images: 1", harness.messages[0][1])
        self.assertIn("Failed images: 1", harness.messages[0][1])
        self.assertIsNone(harness._active_worker_task)


class DetectionSelectionTests(unittest.TestCase):
    def invoke(self, harness, selected):
        guard = invoke_detection_selection(harness, selected)
        self.assertEqual(guard.default_detection_directory_accesses, 0)
        self.assertEqual(guard.protected_image_accesses, 0)

    def test_single_file_selection_remains_compatible(self):
        harness = SelectionHarness()
        self.invoke(harness, ["sample.png"])
        self.assertEqual(len(harness._batch_controller.state.images), 1)
        self.assertEqual(harness.dispatched[0].source_file, "sample.png")
        self.assertEqual(harness._active_worker_task, "detection")

    def test_multiple_selection_is_sorted_and_new_batch_replaces_old(self):
        harness = SelectionHarness()
        harness._detection_result = {"old": True}
        harness._last_completed_result_type = "detection"
        self.invoke(harness, ["image10.png", "image2.png"])
        self.assertEqual(
            [item.original_filename for item in harness._batch_controller.state.images],
            ["image2.png", "image10.png"],
        )
        self.assertIsNone(harness._detection_result)
        self.assertIsNone(harness._last_completed_result_type)
        harness._batch_controller.accept_failure(
            harness.dispatched[-1].run_token,
            harness.dispatched[-1].job_token,
            "done",
        )
        harness._batch_controller.finish_if_done()
        harness._active_worker_task = None
        self.invoke(harness, ["replacement.png"])
        self.assertEqual(
            [item.original_filename for item in harness._batch_controller.state.images],
            ["replacement.png"],
        )

    def test_busy_click_does_not_open_dialog_or_start_second_queue(self):
        harness = SelectionHarness()
        harness._active_worker_task = "detection"
        previous_result = {"previous": True}
        harness._batch_controller.state.last_batch_result = previous_result
        harness._detection_result = previous_result
        with TemporaryDirectory(prefix="ht-detector-busy-") as temporary_directory:
            initial_directory = Path(temporary_directory)
            guard = DetectionPathAccessGuard(
                allowed_files=[DETECTION_WEIGHT_PATH],
                allowed_directories=[initial_directory],
            )
            with patch(
                "detectmain._dialog_initial_directory", return_value=initial_directory
            ), patch(
                "detectmain.QFileDialog.getOpenFileNames"
            ) as dialog, patch(
                "detectmain.QMessageBox.warning"
            ), patch.object(
                Path, "is_file", autospec=True, side_effect=guard.is_file
            ), patch.object(
                Path, "is_dir", autospec=True, side_effect=guard.is_dir
            ):
                DetectMain._select_detection_image(harness)
        dialog.assert_not_called()
        self.assertEqual(harness.dispatched, [])
        self.assertIs(harness._batch_controller.state.last_batch_result, previous_result)
        self.assertIs(harness._detection_result, previous_result)
        self.assertEqual(guard.default_detection_directory_accesses, 0)
        self.assertEqual(guard.protected_image_accesses, 0)


class MemoryDetectionMain(DetectMain):
    """Real Qt controls and production handlers, without application startup/I/O."""

    def __init__(self):
        QWidget.__init__(self)
        self.ui = detectmain.ui_detectmain.Ui_Form()
        self.ui.setupUi(self)
        self._batch_controller = BatchDetectionController()
        self._detection_weight_path = "memory-only.pt"
        self._detection_result = None
        self._detection_dirty = False
        self._regression_result = None
        self._regression_dirty = False
        self._regression_plot_has_result = False
        self._last_completed_result_type = None
        self._active_worker_task = None
        self._close_wait_pending = False
        self._shutdown_requested = False
        self._linear_mode = detectmain.LINEAR_MODE_SINGLE_IMAGE
        self._single_linear_action_text = self.ui.pushButton_4.text()
        self._single_linear_view_state = None
        self._detection_view_state = None
        self._linear_series_result_view_state = None
        self._linear_series_controller = detectmain.LinearSeriesController()
        self._linear_series_selection_state = detectmain.LinearSeriesState()
        self._linear_series_weight_path = None
        self._calibration_source_path = None
        self._calibration_source_image = None
        self._last_calibration_directory = None
        self.origImg = None
        self._origPixmap = None
        self._recgPixmap = None
        self.mainCamera = SimpleNamespace(set_close_wait_pending=Mock())
        self.requests = []
        self.statuses = []
        self.messages = []
        self.worker_returns = []
        self.detection_requested.connect(lambda *args: self.requests.append(args))
        self.detection_status_changed.connect(self.statuses.append)
        self.worker_task_finished.connect(lambda: self.worker_returns.append(True))
        installer = getattr(self, "_install_detection_browser", None)
        if installer is not None:
            installer()
        self.ui.pushButton_8.clicked.connect(self._save_pending_result)

    def _scale_label(self, label):
        pixmap = self._origPixmap if label is self.ui.labelOrigImg else self._recgPixmap
        if pixmap is not None and not pixmap.isNull():
            label.setPixmap(pixmap)

    def _has_valid_regression_result(self):
        return False

    def _has_valid_linear_export(self):
        return self._regression_dirty

    def _show_message_safely(self, function, _parent, title, message):
        self.messages.append((function, title, message))


class DetectionBrowserContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        # All image paths are lexical identities. No model/config/source reads.
        for target in (
            "detectmain.load_effective_settings",
            "interface_config.load_effective_settings",
            "interface_config.load_detection_preferences",
            "yolo_detection_worker.load_effective_settings",
            "yolo_detection_worker.YoloDetectionWorker._get_model",
            "yolo_detection_worker.YoloDetectionWorker.detect",
            "numpy.fromfile",
        ):
            guard = patch(target, side_effect=AssertionError("Unexpected resource access"))
            guard.start()
            self.addCleanup(guard.stop)
        guard = patch.object(Path, "is_dir", return_value=False)
        guard.start()
        self.addCleanup(guard.stop)
        self.main = MemoryDetectionMain()
        self.addCleanup(self.main.deleteLater)

    def start(self, scope="entire_batch", paths=None):
        main = self.main
        paths = paths or ["Z:/qbatch-memory/image{}.png".format(n) for n in range(1, 5)]
        main.set_detection_options(scope, "per_image")
        main._batch_controller.replace_images(paths)
        task = main._batch_controller.begin()
        main._clear_detection_results_for_new_run()
        main._set_active_worker_task("detection")
        main._dispatch_detection_task(task)
        return task

    def result(self, task, count=1):
        result = runtime_payload(task, count)
        result["image"][:] = (task.image_order * 10, 20, 30)
        for index, (target, sample) in enumerate(zip(result["targets"], result["sample_results"])):
            target["Red"] = sample["red"] = task.image_order + 0.123456789 + index
            target["Con."] = task.image_order + 0.7654321 + index
        return result

    def finish_all(self):
        delivered = []
        while self.main._batch_controller.active_job is not None:
            result = self.result(self.main._batch_controller.active_job)
            delivered.append(result)
            self.main._on_detection_finished(result)
        return delivered

    def select(self, index):
        selector = self.main.ui.detectionImageSelector
        selector.setCurrentIndex(index)
        return self.main._detection_result

    def test_current_image_four_imports_plan_and_complete_only_one(self):
        task = self.start("current_image")
        self.assertIn("Imported images: 4", self.main.statuses[-1])
        self.assertIn("Planned this run: 1", self.main.statuses[-1])
        self.assertIn("Scope: Current Image", self.main.statuses[-1])
        self.assertIn("Target: image1.png", self.main.statuses[-1])
        self.main._on_detection_finished(self.result(task))
        self.assertEqual(len(self.main.requests), 1)
        summary = self.main.statuses[-1]
        for text in ("Completed this run: 1", "Successful images: 1", "Failed images: 0",
                     "Not included this run: 3", "Unfinished this run: 0"):
            self.assertIn(text, summary)
        self.assertEqual([i.status for i in self.main.batch_state.images],
                         [ImageStatus.COMPLETED] + [ImageStatus.PENDING] * 3)

    def test_entire_batch_four_distinct_terminal_results_and_truthful_summary(self):
        self.start()
        for number in range(1, 5):
            task = self.main._batch_controller.active_job
            if number == 2:
                self.main._on_detection_failed(dict(task.context(), message="memory failure"))
            else:
                self.main._on_detection_finished(self.result(task))
        self.assertEqual(len(self.main.requests), 4)
        summary = self.main.statuses[-1]
        for text in ("Completed this run: 4", "Successful images: 3", "Failed images: 1",
                     "Not included this run: 0", "Unfinished this run: 0"):
            self.assertIn(text, summary)
        self.assertFalse(self.main._batch_controller.active)

    def test_four_results_browse_without_inference_or_execution_state_changes(self):
        self.start()
        results = self.finish_all()
        controller = self.main._batch_controller
        before = (controller.state.current_image_index, tuple(controller._queue),
                  controller.summary(), controller.state.detection_scope, len(self.main.requests))
        self.assertEqual(self.main.ui.detectionImageSelector.count(), 4)
        for index in (2, 0, 3, 1, 0):
            chosen = self.select(index)
            self.assertEqual(chosen["source_path"], results[index]["source_path"])
            np.testing.assert_array_equal(chosen["image"], results[index]["image"])
            pixel = self.main._recgPixmap.toImage().pixelColor(0, 0)
            self.assertEqual(pixel.blue(), (index + 1) * 10)
            table = self.main.ui.tabviewRecg.model()
            self.assertEqual(table.rowCount(), 1)
            self.assertEqual(table.item(0, 2).text(), "{:.2f}".format(index + 1.123456789))
            self.assertEqual(chosen["targets"][0]["Red"], results[index]["targets"][0]["Red"])
            self.assertEqual(self.main.batch_state.images[index].samples[0].red,
                             results[index]["sample_results"][0]["red"])
            self.assertIn("image{}.png".format(index + 1), self.main.ui.detectionImageSelector.toolTip())
        self.assertEqual(before, (controller.state.current_image_index, tuple(controller._queue),
                                 controller.summary(), controller.state.detection_scope, len(self.main.requests)))

    def test_same_names_and_duplicate_paths_have_independent_results(self):
        self.start(paths=["Z:/qbatch-memory/a/same.png", "Z:/qbatch-memory/b/same.png",
                          "Z:/qbatch-memory/a/same.png", "Z:/qbatch-memory/c/same.png"])
        results = self.finish_all()
        for index, result in enumerate(results):
            chosen = self.select(index)
            self.assertEqual(chosen["source_path"], result["source_path"])
            self.assertEqual(chosen["targets"][0]["Red"], result["targets"][0]["Red"])
        self.assertEqual(len(self.main._detection_image_results), 4)

    def test_failed_and_pending_selection_cannot_show_or_save_another_success(self):
        task = self.start("current_image")
        self.main._on_detection_finished(self.result(task))
        self.select(1)
        self.assertIsNone(self.main._detection_result)
        self.assertIsNone(self.main._recgPixmap)
        self.assertEqual(self.main.ui.tabviewRecg.model().rowCount(), 0)
        self.assertIn("pending", self.main.ui.detectionImageSelector.toolTip())
        self.assertFalse(self.main.ui.pushButton_8.isEnabled())
        with patch.object(self.main, "_commit_detection_export") as commit, patch.object(
            detectmain.QMessageBox, "critical"
        ):
            self.main._save_detection_result()
        commit.assert_not_called()
        self.assertTrue(self.main._detection_dirty)

        self.start()
        first = self.main._batch_controller.active_job
        self.main._on_detection_finished(self.result(first))
        failed = self.main._batch_controller.active_job
        self.main._on_detection_failed(dict(failed.context(), message="decode failed"))
        self.finish_all()
        self.select(1)
        self.assertIsNone(self.main._detection_result)
        self.assertIsNone(self.main._recgPixmap)
        self.assertEqual(self.main.ui.tabviewRecg.model().rowCount(), 0)
        self.assertIn("decode failed", self.main.ui.detectionImageSelector.toolTip())
        self.assertFalse(self.main.ui.pushButton_8.isEnabled())

    def test_cache_owns_annotation_and_minimal_export_data_without_full_payload(self):
        task = self.start("current_image")
        result = self.result(task, count=2)
        result["original_rgb"] = np.ones((50, 50, 3), dtype=np.uint8)
        self.main._on_detection_finished(result)
        result["image"][:] = 255
        result["targets"][0]["Red"] = 999
        self.select(1)
        chosen = self.select(0)
        self.assertEqual(set(chosen), {"source_path", "image", "targets"})
        self.assertEqual(int(chosen["image"][0, 0, 0]), 10)
        self.assertEqual(chosen["targets"][0]["Red"], 1.123456789)
        self.assertEqual(self.main.ui.tabviewRecg.model().rowCount(), 2)

    def test_save_follows_selection_and_only_clears_its_own_dirty_flag(self):
        self.start()
        results = self.finish_all()
        saved = []

        def commit(_directory, stem, workbook, png):
            saved.append((stem, workbook, png))
            return detectmain._SaveTransactionResult(True, ("memory.xlsx", "memory.png"))

        with patch.object(self.main, "_commit_detection_export", side_effect=commit):
            for index in (1, 3, 0, 2):
                chosen = self.select(index)
                self.assertIn("image{}.png".format(index + 1), self.main.ui.pushButton_8.toolTip())
                self.assertTrue(self.main.ui.pushButton_8.isEnabled())
                self.main.ui.pushButton_8.click()
                self.assertFalse(self.main.ui.pushButton_8.isEnabled())
                self.assertEqual(saved[-1][0], "image{}".format(index + 1))
                export = self.main._validated_detection_export_payload(chosen)
                self.main._validate_detection_workbook_bytes(saved[-1][1], export)
                import cv2
                decoded = cv2.imdecode(np.frombuffer(saved[-1][2], dtype=np.uint8), cv2.IMREAD_COLOR)
                np.testing.assert_array_equal(decoded, results[index]["image"])
                self.assertEqual(self.main._detection_dirty, len(saved) != 4)

    def test_save_failure_keeps_all_successes_dirty(self):
        self.start()
        self.finish_all()
        self.select(0)
        with patch.object(self.main, "_commit_detection_export", side_effect=RuntimeError("disk stub")):
            self.main._save_detection_result()
        self.assertTrue(self.main._detection_dirty)
        for index in range(4):
            self.select(index)
            self.assertTrue(self.main.ui.pushButton_8.isEnabled())

    def test_browser_is_disabled_during_run_and_programmatic_selection_is_ignored(self):
        task = self.start()
        selector = self.main.ui.detectionImageSelector
        self.assertFalse(selector.isEnabled())
        before = self.main._detection_selected_key
        selector.setCurrentIndex(3)
        self.assertEqual(self.main._detection_selected_key, before)
        self.assertEqual(selector.currentIndex(), 0)
        self.assertIs(self.main._batch_controller.active_job, task)
        self.finish_all()
        self.assertTrue(selector.isEnabled())

    def test_wrong_identity_duplicate_and_late_callbacks_do_not_poison_cache(self):
        task = self.start()
        result = self.result(task)
        for field, value in (("source_path", "Z:/wrong.png"), ("job_token", True),
                             ("run_token", 999), ("image_order", 999)):
            wrong = dict(result, **{field: value})
            self.main._on_detection_finished(wrong)
            self.assertIs(self.main._batch_controller.active_job, task)
            self.assertEqual(self.main._detection_image_results, {})
        wrong = self.result(task)
        wrong["sample_results"][0]["image_order"] = 99
        self.main._on_detection_finished(wrong)
        self.assertEqual(self.main._detection_image_results, {})
        self.main._on_detection_finished(result)
        before = dict(self.main._detection_image_results)
        self.main._on_detection_finished(result)
        self.main._on_detection_failed(dict(task.context(), message="duplicate"))
        self.assertEqual(self.main._detection_image_results, before)
        self.finish_all()
        self.start()
        self.main._on_detection_finished(result)
        self.assertEqual(self.main._detection_image_results, {})
        self.assertIsNone(self.main._detection_result)

    def test_cancel_file_selection_keeps_cache_selection_and_unsaved_results(self):
        self.start()
        self.finish_all()
        self.select(0)
        before = (self.main.batch_state, self.main._detection_image_results,
                  self.main._detection_selected_key, self.main._detection_result)
        with patch.object(detectmain.QMessageBox, "question", return_value=detectmain.QMessageBox.Yes), patch.object(
            detectmain, "_dialog_initial_directory", return_value="Z:/qbatch-memory"
        ), patch.object(detectmain.QFileDialog, "getOpenFileNames", return_value=([], "Images")):
            self.main._select_detection_image()
        self.assertEqual(before, (self.main.batch_state, self.main._detection_image_results,
                                 self.main._detection_selected_key, self.main._detection_result))
        self.assertTrue(self.main._detection_dirty)

    def test_series_hides_browser_and_detection_does_not_replace_series_view(self):
        self.start()
        self.finish_all()
        self.select(0)
        selected = self.main._detection_selected_key
        self.main.set_linear_mode(detectmain.LINEAR_MODE_IMAGE_SERIES)
        self.assertTrue(self.main.ui.detectionBrowser.isHidden())
        series_model = self.main.ui.tabviewRecg.model()
        series_text = self.main.ui.labelRecgImg.text()
        self.main.ui.detectionImageSelector.setCurrentIndex(2)
        self.assertEqual(self.main._detection_selected_key, selected)
        self.assertIs(self.main.ui.tabviewRecg.model(), series_model)
        self.assertEqual(self.main.ui.labelRecgImg.text(), series_text)
        self.main.set_linear_mode(detectmain.LINEAR_MODE_SINGLE_IMAGE)
        self.assertFalse(self.main.ui.detectionBrowser.isHidden())
        self.assertEqual(self.main._detection_selected_key, selected)
        self.assertEqual(self.main._detection_result["targets"][0]["Red"], 1.123456789)

    def test_wait_close_ignores_payload_and_cancel_close_before_return_resumes(self):
        for cancel, failed in ((False, False), (False, True), (True, False), (True, True)):
            with self.subTest(cancel=cancel, failed=failed):
                self.main._close_wait_pending = False
                self.main._active_worker_task = None
                self.main._batch_controller = BatchDetectionController()
                task = self.start()
                self.main.set_close_wait_pending(True)
                self.assertFalse(self.main.ui.detectionImageSelector.isEnabled())
                if cancel:
                    self.main.set_close_wait_pending(False)
                if failed:
                    self.main._on_detection_failed(dict(task.context(), message="controlled failure"))
                else:
                    self.main._on_detection_finished(self.result(task))
                if cancel:
                    self.assertEqual(self.main._batch_controller.active_job.image_order, 2)
                    self.finish_all()
                    self.assertTrue(self.main.ui.detectionImageSelector.isEnabled())
                else:
                    self.assertEqual(self.main._detection_image_results, {})
                    self.assertIsNone(self.main._active_worker_task)
                    self.assertFalse(self.main.ui.detectionImageSelector.isEnabled())

    def test_file_dialog_entry_preserves_scope_and_reselection_clears_every_result(self):
        paths = ["Z:/qbatch-memory/import{}.png".format(n) for n in (4, 2, 1, 3)]
        for scope, planned in (("current_image", 1), ("entire_batch", 4)):
            with self.subTest(scope=scope):
                self.main.set_detection_options(scope, "per_image")
                with patch.object(detectmain, "_dialog_initial_directory", return_value="Z:/qbatch-memory"), patch.object(
                    detectmain.QFileDialog, "getOpenFileNames", return_value=(paths, "Images")
                ), patch.object(Path, "is_file", return_value=True), patch.object(
                    detectmain.QMessageBox, "question", return_value=detectmain.QMessageBox.Yes
                ):
                    self.main._select_detection_image()
                self.assertEqual(self.main.batch_state.detection_scope, scope)
                self.assertEqual(self.main._detection_image_results, {})
                self.assertIsNone(self.main._detection_result)
                self.assertFalse(self.main._detection_dirty)
                self.assertIn("Planned this run: {}".format(planned), self.main.statuses[-1])
                first = self.main._batch_controller.active_job
                self.assertEqual(first.source_file, "import1.png")
                self.finish_all()

    def test_failed_payload_keeps_its_own_annotation_with_empty_table_and_errors(self):
        self.start()
        first = self.main._batch_controller.active_job
        self.main._on_detection_finished(self.result(first))
        failed = self.main._batch_controller.active_job
        result = self.result(failed, count=0)
        result["sample_errors"] = [sample_error(failed, "invalid_roi", "no valid ROI")]
        self.main._on_detection_finished(result)
        self.finish_all()
        self.select(1)
        self.assertEqual(self.main._recgPixmap.toImage().pixelColor(0, 0).blue(), 20)
        self.assertIn("no valid ROI", self.main.ui.detectionImageSelector.toolTip())
        self.assertEqual(self.main.ui.tabviewRecg.model().rowCount(), 0)
        self.assertIsNone(self.main._detection_result)
        self.assertFalse(self.main.ui.pushButton_8.isEnabled())

    def test_detection_during_series_preserves_series_and_restores_single_regression(self):
        linear_result = {"memory": "confirmed single regression"}
        original = QPixmap(3, 3)
        original.fill(QColor("red"))
        self.main._origPixmap = original
        self.main.origImg = "memory-single"
        self.main._calibration_source_path = "Z:/qbatch-memory/single.png"
        self.main._calibration_source_image = np.ones((3, 3, 3), dtype=np.uint8)
        self.main._regression_result = linear_result
        self.main._regression_dirty = True
        self.main._last_completed_result_type = "linear"
        single_table = self.main._build_table_model(["No.", "Red"], [(1, 17.12345)])
        self.main.ui.tabviewOrig.setModel(single_table)
        self.main.set_linear_mode(detectmain.LINEAR_MODE_IMAGE_SERIES)
        series_pixmap = QPixmap(3, 3)
        series_pixmap.fill(QColor("yellow"))
        series_table = self.main._build_table_model(["Red"], [(92.5,)])
        view = detectmain._ResultPaneViewState(series_pixmap, "", series_table, "Series", "Series table")
        self.main._linear_series_result_view_state = view
        self.main._apply_result_pane_view(view)
        self.start()
        self.finish_all()
        self.assertTrue(self.main.ui.detectionBrowser.isHidden())
        self.assertIs(self.main._recgPixmap, series_pixmap)
        self.assertIs(self.main.ui.tabviewRecg.model(), series_table)
        self.main.set_linear_mode(detectmain.LINEAR_MODE_SINGLE_IMAGE)
        self.assertFalse(self.main.ui.detectionBrowser.isHidden())
        self.assertEqual(self.main._recgPixmap.toImage().pixelColor(0, 0).blue(), 40)
        self.assertIs(self.main.ui.tabviewOrig.model(), single_table)
        self.assertIs(self.main._regression_result, linear_result)
        self.assertTrue(self.main._regression_dirty)
        self.assertIs(self.main._origPixmap, original)
        self.select(0)
        self.assertEqual(self.main._recgPixmap.toImage().pixelColor(0, 0).blue(), 10)

    def test_mode_switch_failure_restores_browser_visibility_and_selection(self):
        self.start()
        self.finish_all()
        self.select(0)
        selected = self.main._detection_selected_key
        with patch.object(self.main, "_update_save_button", side_effect=RuntimeError("control stub")):
            with self.assertRaisesRegex(RuntimeError, "control stub"):
                self.main.set_linear_mode(detectmain.LINEAR_MODE_IMAGE_SERIES)
        self.assertEqual(self.main.linear_mode, detectmain.LINEAR_MODE_SINGLE_IMAGE)
        self.assertFalse(self.main.ui.detectionBrowser.isHidden())
        self.assertTrue(self.main.ui.detectionImageSelector.isEnabled())
        self.assertEqual(self.main._detection_selected_key, selected)

    def test_close_warning_uses_all_dirty_results_even_when_selected_image_is_saved(self):
        self.start()
        self.finish_all()
        self.select(0)
        with patch.object(self.main, "_commit_detection_export", return_value=detectmain._SaveTransactionResult(
            True, ("memory.xlsx", "memory.png")
        )):
            self.main._save_detection_result()
        window = SimpleNamespace(_detectMain=self.main, _has_linear_series_mapping_draft=lambda: False)
        self.assertEqual(detectionwindow.DetectWindow._close_discard_categories(window),
                         ("Unsaved Detection result",))

    def test_wrong_failure_identity_is_ignored_before_message_or_state_changes(self):
        task = self.start()
        before = (len(self.main.requests), tuple(self.main.statuses), self.main._detection_selected_key)
        self.main._on_detection_failed(dict(task.context(), source_path="Z:/qbatch-memory/wrong.png",
                                           message="wrong identity"))
        self.assertEqual(before, (len(self.main.requests), tuple(self.main.statuses),
                                  self.main._detection_selected_key))
        self.assertIs(self.main._batch_controller.active_job, task)
        self.assertEqual(self.main._detection_image_results, {})

    def test_incomplete_run_does_not_report_finished_or_unlock_browsing(self):
        self.start()
        before = tuple(self.main.statuses)
        self.main._finish_detection_run()
        self.assertEqual(tuple(self.main.statuses), before)
        self.assertEqual(self.main._active_worker_task, "detection")
        self.assertFalse(self.main.ui.detectionImageSelector.isEnabled())

    def test_current_malformed_return_releases_close_wait_without_reading_result_content(self):
        task = self.start()
        self.main.set_close_wait_pending(True)
        with patch.object(self.main, "_normalized_detection_runtime_payload",
                          side_effect=AssertionError("Closing must not decode a result")):
            self.main._on_detection_finished({"run_token": task.run_token,
                                              "job_token": task.job_token,
                                              "sample_results": object()})
        self.assertEqual(self.main.worker_returns, [True])
        self.assertIsNone(self.main._active_worker_task)
        self.assertEqual(self.main._detection_image_results, {})
        self.assertEqual(len(self.main.requests), 1)

    def complete_linear_for_save(self):
        # These cases use the production validators, including their rejection
        # of invalid payloads; the older browser fixture's shortcuts do not apply.
        main = self.main
        main._has_valid_linear_export = DetectMain._has_valid_linear_export.__get__(main)
        main._has_valid_regression_result = DetectMain._has_valid_regression_result.__get__(main)
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

        main.figure = Figure()
        main.canvas = FigureCanvasQTAgg(main.figure)
        self.addCleanup(main.canvas.deleteLater)
        main.ax = main.figure.add_subplot(111)
        main._regression_suptitle = None
        payload = {
            "source_path": "Z:/qbatch-memory/linear.png",
            "image": np.full((12, 16, 3), (71, 82, 93), dtype=np.uint8),
            "samples": [
                {"No.": n, "Con.": float(n), "Red": 2.0 * n + 1,
                 "Green": 3.0 * n + 1, "Blue": 4.0 * n + 1, "included": True}
                for n in (1, 2)
            ],
            "formulas": {
                channel: {"slope": slope, "intercept": 1.0, "r": 1.0,
                          "R2": 1.0, "p": 0.01, "std_err": 0.0}
                for channel, slope in (("R", 2.0), ("G", 3.0), ("B", 4.0))
            },
            "selected_channel": "R", "elapsed_ms": 1.0, "warnings": [],
        }
        main._set_active_worker_task("regression")
        main._on_regression_finished(payload)
        self.assertIs(main._regression_result, payload)
        self.assertTrue(main._has_valid_linear_export())
        self.assert_save_button("Save Linear", True)
        return payload

    def assert_save_button(self, text, enabled):
        self.assertEqual(self.main.ui.pushButton_8.text(), text)
        self.assertEqual(self.main.ui.pushButton_8.isEnabled(), enabled)
        if text == "Save Linear":
            self.assertNotIn("Detection", self.main.ui.pushButton_8.toolTip())

    def save_detection_selection(self, index):
        chosen = self.select(index)
        self.assert_save_button("Save Detection", True)
        self.assertIn(Path(chosen["source_path"]).name, self.main.ui.pushButton_8.toolTip())
        with patch.object(self.main, "_commit_detection_export", return_value=
                          detectmain._SaveTransactionResult(True, ("memory.xlsx", "memory.png"))) as commit:
            self.main.ui.pushButton_8.click()
        commit.assert_called_once()
        self.assertEqual(commit.call_args.args[1], Path(chosen["source_path"]).stem)
        self.assertFalse(self.main._selected_detection_record().dirty)

    def assert_linear_files(self, files, payload):
        import io
        import cv2

        self.assertEqual(set(files), set(self.main.LINEAR_EXPORT_FILENAMES))
        workbook = detectmain.openpyxl.load_workbook(
            io.BytesIO(files["linear_con_rgb.xlsx"]), read_only=True, data_only=True
        )
        try:
            sheet = workbook["Sheet1"]
            for row, sample in enumerate(payload["samples"], start=2):
                self.assertEqual(
                    tuple(sheet.cell(row, col).value for col in range(1, 7)),
                    tuple(sample[field] for field in ("No.", "Con.", "Red", "Green", "Blue", "included")),
                )
            for row, channel in enumerate(("R", "G", "B"), start=2):
                self.assertEqual(sheet.cell(row, 8).value, channel)
                self.assertEqual(sheet.cell(row, 9).value, payload["formulas"][channel]["slope"])
        finally:
            workbook.close()
        annotated = cv2.imdecode(np.frombuffer(files["calibration_annotated.png"], dtype=np.uint8),
                                 cv2.IMREAD_COLOR)
        np.testing.assert_array_equal(annotated, payload["image"])
        curve = cv2.imdecode(np.frombuffer(files["standard_curve.png"], dtype=np.uint8),
                            cv2.IMREAD_COLOR)
        self.assertIsNotNone(curve)
        self.assertGreater(curve.size, 0)

    def save_linear_selection(self, payload):
        self.assert_save_button("Save Linear", True)
        # Existence/commit are simulated; serialization, plotting, validation,
        # target selection, and the Qt button's save handler remain production code.
        with patch.object(Path, "exists", return_value=False), patch.object(
            self.main, "_commit_linear_export", return_value=detectmain._SaveTransactionResult(True)
        ) as commit, patch.object(self.main, "_commit_detection_export") as detection_commit:
            self.main.ui.pushButton_8.click()
        commit.assert_called_once()
        detection_commit.assert_not_called()
        self.assertEqual(commit.call_args.args[0], ROOT / "HT-Detector_Peng" / "runs" /
                         "detect" / "results" / "linear")
        self.assert_linear_files(commit.call_args.args[1], payload)
        self.assertFalse(self.main._regression_dirty)

    def test_linear_save_recovers_immediately_after_four_detection_saves(self):
        payload = self.complete_linear_for_save()
        self.start()
        self.finish_all()
        for count, index in enumerate((1, 3, 0, 2), start=1):
            self.save_detection_selection(index)
            self.assertTrue(self.main._regression_dirty)
            self.assertEqual(sum(r.dirty for r in self.main._detection_image_results.values()), 4 - count)
            self.assertEqual(self.main._detection_dirty, count < 4)
        # No Plot, mode switch, or new inference between the last save and this click.
        self.save_linear_selection(payload)
        self.assertFalse(self.main._detection_dirty)
        self.assertFalse(self.main.ui.pushButton_8.isEnabled())
        self.assertEqual(len(self.main.requests), 4)

    def test_linear_save_after_plot_preserves_unselected_detection_dirty(self):
        payload = self.complete_linear_for_save()
        self.start()
        self.finish_all()
        self.save_detection_selection(1)
        selected_key = self.main._detection_selected_key
        dirty = {key: record.dirty for key, record in self.main._detection_image_results.items()}
        self.assert_save_button("Save Linear", True)
        self.main._plot_regression_result()
        self.assert_save_button("Save Linear", True)
        self.save_linear_selection(payload)
        self.assertEqual(self.main._detection_selected_key, selected_key)
        self.assertEqual({key: r.dirty for key, r in self.main._detection_image_results.items()}, dirty)
        self.assertTrue(self.main._detection_dirty)
        self.assertFalse(self.main.ui.pushButton_8.isEnabled())
        self.select(0)
        self.assert_save_button("Save Detection", True)

    def test_linear_save_unavailable_for_missing_saved_or_invalid_linear(self):
        payload = self.complete_linear_for_save()
        self.start()
        self.finish_all()
        self.save_detection_selection(0)
        for name, result, dirty in (("missing", None, False), ("saved", payload, False),
                                    ("invalid", dict(payload, samples=[]), True)):
            with self.subTest(linear=name):
                self.main._regression_result = result
                self.main._regression_dirty = dirty
                self.main._update_save_button()
                self.assertEqual(self.main._has_valid_linear_export(), name == "saved")
                self.assertFalse(self.main.ui.pushButton_8.isEnabled())
                with patch.object(self.main, "_commit_detection_export") as detection_commit, patch.object(
                    self.main, "_commit_linear_export"
                ) as linear_commit:
                    self.main.ui.pushButton_8.click()
                detection_commit.assert_not_called()
                linear_commit.assert_not_called()
                self.assertEqual(sum(r.dirty for r in self.main._detection_image_results.values()), 3)

    def test_linear_save_cancel_keeps_both_result_types_dirty(self):
        self.complete_linear_for_save()
        self.start()
        self.finish_all()
        self.save_detection_selection(0)
        dirty = {key: record.dirty for key, record in self.main._detection_image_results.items()}
        self.assert_save_button("Save Linear", True)
        with patch.object(Path, "exists", return_value=True), patch.object(
            detectmain.QMessageBox, "question", return_value=detectmain.QMessageBox.No
        ) as question, patch.object(self.main, "_commit_linear_export") as linear_commit, patch.object(
            self.main, "_commit_detection_export"
        ) as detection_commit:
            self.main.ui.pushButton_8.click()
        question.assert_called_once()
        self.assertEqual(question.call_args.args[1], "Overwrite linear regression files")
        linear_commit.assert_not_called()
        detection_commit.assert_not_called()
        self.assertTrue(self.main._regression_dirty)
        self.assertTrue(self.main._detection_dirty)
        self.assertEqual({key: r.dirty for key, r in self.main._detection_image_results.items()}, dirty)
        self.assert_save_button("Save Linear", True)

    def test_linear_save_commit_failure_keeps_both_result_types_dirty(self):
        payload = self.complete_linear_for_save()
        self.start()
        self.finish_all()
        self.save_detection_selection(0)
        dirty = {key: record.dirty for key, record in self.main._detection_image_results.items()}
        self.assert_save_button("Save Linear", True)
        with patch.object(Path, "exists", return_value=False), patch.object(
            self.main, "_commit_linear_export", side_effect=RuntimeError("linear disk stub")
        ) as commit, patch.object(self.main, "_commit_detection_export") as detection_commit:
            self.main.ui.pushButton_8.click()
        commit.assert_called_once()
        detection_commit.assert_not_called()
        self.assert_linear_files(commit.call_args.args[1], payload)
        self.assertTrue(self.main._regression_dirty)
        self.assertTrue(self.main._detection_dirty)
        self.assertEqual({key: r.dirty for key, r in self.main._detection_image_results.items()}, dirty)
        self.assertIn("linear disk stub", self.main.messages[-1][2])
        self.assert_save_button("Save Linear", True)

    def test_linear_save_from_pending_or_failed_preserves_other_detection_results(self):
        for status in ("pending", "failed"):
            with self.subTest(status=status):
                payload = self.complete_linear_for_save()
                first = self.start("current_image" if status == "pending" else "entire_batch")
                self.main._on_detection_finished(self.result(first))
                if status == "failed":
                    failed = self.main._batch_controller.active_job
                    self.main._on_detection_failed(dict(failed.context(), message="synthetic failure"))
                    self.finish_all()
                self.select(1)
                self.assertIsNone(self.main._detection_result)
                self.assertIsNone(self.main._recgPixmap)
                self.assertEqual(self.main.ui.tabviewRecg.model().rowCount(), 0)
                self.assertIn(status, self.main.ui.detectionImageSelector.toolTip())
                dirty = {key: record.dirty for key, record in self.main._detection_image_results.items()}
                self.save_linear_selection(payload)
                self.assertTrue(self.main._detection_dirty)
                self.assertEqual({key: r.dirty for key, r in self.main._detection_image_results.items()}, dirty)
                self.assertFalse(self.main.ui.pushButton_8.isEnabled())
                self.select(0)
                self.assert_save_button("Save Detection", True)

    def test_linear_save_retains_head_priority_for_new_linear_result(self):
        self.start()
        self.finish_all()
        selected_result = self.select(1)
        payload = self.complete_linear_for_save()
        self.save_linear_selection(payload)
        self.assertEqual(sum(r.dirty for r in self.main._detection_image_results.values()), 4)
        self.assertTrue(self.main._detection_dirty)
        self.assertIs(self.main._detection_result, selected_result)
        self.assert_save_button("Save Detection", True)


class ObservedConstructionMain(DetectMain):
    """Execute the production constructor; observe, never replace, snapshots."""

    def _capture_result_pane_view(self):
        view = super()._capture_result_pane_view()
        if not hasattr(self, "initial_snapshots"):
            self.initial_snapshots = []
        self.initial_snapshots.append((
            view, (self.ui.progressBar.minimum(), self.ui.progressBar.maximum(),
                   self.ui.progressBar.value()), self.ui.lcdNumber.value(),
        ))
        return view


class DetectionStartupAndLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.resource_requests = []
        self.statuses = []
        self.messages = []
        self.requests = []
        self.figures_before = set(detectmain.plt.get_fignums())
        was_interactive = detectmain.plt.isinteractive()
        detectmain.plt.ioff()
        self.addCleanup(detectmain.plt.interactive, was_interactive)
        self.addCleanup(self.close_figures)

        def memory_workbook(path, **_kwargs):
            self.resource_requests.append(str(path).replace("\\", "/"))
            workbook = detectmain.openpyxl.Workbook()
            for row in (("No.", "Con.", "Red", "Green", "Blue"),
                        (1, 1, 3, 4, 5), (2, 2, 5, 7, 9), (3, 3, 7, 10, 13)):
                workbook.active.append(row)
            return workbook

        def memory_image_reference(path):
            reference = str(path).replace("\\", "/")
            self.resource_requests.append(reference)
            return "memory-image:" + reference

        def memory_pixmap(*args):
            if args and isinstance(args[0], str):
                if not args[0].startswith("memory-image:"):
                    raise AssertionError("Unexpected image load: " + args[0])
                pixmap = QPixmap(16, 12)
                pixmap.fill(QColor("magenta"))
                return pixmap
            return QPixmap(*args)

        memory_pixmap.fromImage = QPixmap.fromImage

        def memory_open(path, *_args, **_kwargs):
            reference = str(path).replace("\\", "/")
            self.resource_requests.append(reference)
            if not reference.endswith("/interface/detect/time.txt"):
                raise AssertionError("Unexpected startup file read: " + reference)
            return io.StringIO("377 ms")

        camera = SimpleNamespace(
            _ui=SimpleNamespace(cameraWidget=QWidget()), shutdown_ready=Mock(),
            set_close_wait_pending=Mock(), request_shutdown=Mock(),
        )
        # Install all external boundaries before construction. The real Qt UI,
        # splitters, plotting, snapshots, mode controls and validators execute.
        for boundary in (
            patch.object(detectmain, "Camera", return_value=camera),
            patch.object(detectmain.QThread, "start", autospec=True),
            patch.object(detectmain.openpyxl, "load_workbook", side_effect=memory_workbook),
            patch.object(DetectMain, "_resolve_image", side_effect=memory_image_reference),
            patch.object(detectmain, "QPixmap", new=memory_pixmap),
            patch.object(detectmain, "open", create=True, side_effect=memory_open),
            patch.object(detectmain, "load_effective_settings",
                         side_effect=AssertionError("Unexpected configuration read")),
            patch("interface_config.load_effective_settings",
                  side_effect=AssertionError("Unexpected configuration read")),
            patch("interface_config.load_detection_preferences",
                  side_effect=AssertionError("Unexpected preferences read")),
            patch("yolo_detection_worker.load_effective_settings",
                  side_effect=AssertionError("Unexpected worker configuration read")),
            patch.object(detectmain.YoloDetectionWorker, "_get_model",
                         side_effect=AssertionError("Unexpected model load")),
            patch("numpy.fromfile", side_effect=AssertionError("Unexpected image read")),
            patch.object(Path, "is_dir", return_value=False),
        ):
            boundary.start()
            self.addCleanup(boundary.stop)
        self.main = ObservedConstructionMain(camera_enabled=False)
        self.addCleanup(self.main.deleteLater)
        self.main.detection_status_changed.connect(self.statuses.append)
        self.main.detection_requested.connect(lambda *args: self.requests.append(args))
        dialog = patch.object(DetectMain, "_show_message_safely", side_effect=
                              lambda function, _parent, title, message:
                              self.messages.append((function, title, message)))
        dialog.start()
        self.addCleanup(dialog.stop)

    def close_figures(self):
        for number in set(detectmain.plt.get_fignums()) - self.figures_before:
            detectmain.plt.close(number)

    def assert_empty_view(self, view):
        with self.subTest(field="pixmap"):
            self.assertTrue(view.pixmap is None or view.pixmap.isNull())
        with self.subTest(field="placeholder"):
            self.assertTrue(view.label_text.strip())
            self.assertNotIn("\n", view.label_text)
        with self.subTest(field="table"):
            self.assertEqual(view.model.rowCount(), 0)
            self.assertEqual(tuple(view.model.headerData(c, Qt.Horizontal)
                                   for c in range(view.model.columnCount())),
                             ("No.", "Con.", "Red", "Green", "Blue"))

    def assert_initial_controls(self):
        with self.subTest(field="progress"):
            self.assertEqual((self.main.ui.progressBar.minimum(),
                              self.main.ui.progressBar.maximum(),
                              self.main.ui.progressBar.value()), (0, 100, 0))
        with self.subTest(field="elapsed"):
            self.assertEqual(self.main.ui.lcdNumber.value(), 0)
        self.assertEqual(self.main.ui.detectionImageSelector.count(), 0)
        self.assertFalse(self.main.ui.detectionImageSelector.isEnabled())
        self.assertFalse(self.main.ui.pushButton_8.isEnabled())
        self.assertIsNone(self.main._detection_result)
        self.assertFalse(self.main._detection_dirty)

    def test_constructor_and_first_snapshot_are_empty_without_detection_demo_loads(self):
        self.assert_empty_view(self.main._capture_result_pane_view())
        first, progress, elapsed = self.main.initial_snapshots[0]
        self.assert_empty_view(first)
        with self.subTest(field="first snapshot progress"):
            self.assertEqual(progress, (0, 100, 0))
        with self.subTest(field="first snapshot elapsed"):
            self.assertEqual(elapsed, 0)
        self.assert_initial_controls()
        self.assertFalse(self.main.ui.labelRecgImg.pixmap() and
                         not self.main.ui.labelRecgImg.pixmap().isNull())
        self.assertEqual(self.main.ui.tabviewOrig.model().rowCount(), 0)
        self.assertEqual(self.main.ui.labelOrigImg.text(), "Import a calibration image")
        self.assertFalse(self.main._regression_plot_has_result)
        with self.subTest(field="no Detection display resource reads"):
            self.assertEqual([path for path in self.resource_requests if any(
                part in path for part in ("/detect/only_table/", "/detect/image/",
                                          "/detect/time.txt"))], [])

    def test_empty_single_series_round_trips_do_not_revive_startup_content(self):
        for _ in range(2):
            self.main.set_linear_mode(detectmain.LINEAR_MODE_IMAGE_SERIES)
            self.main.set_linear_mode(detectmain.LINEAR_MODE_SINGLE_IMAGE)
            self.assert_empty_view(self.main._capture_result_pane_view())
            self.assert_initial_controls()

    def show_layout(self, point_size, width, height):
        font = QFont(self.main.font())
        font.setPointSize(point_size)
        self.main.setFont(font)
        self.main.resize(width, height)
        self.main.show()  # QT_QPA_PLATFORM=offscreen; no interactive application.
        self.app.processEvents()
        return self.main.findChild(QSplitter, "detectionSplitter")

    def test_production_browser_contains_only_caption_and_selector(self):
        browser = self.main.ui.detectionBrowser
        self.assertEqual(browser.layout().count(), 2)
        self.assertEqual([label.text() for label in browser.findChildren(QLabel)],
                         ["Detection results:"])
        self.assertIs(browser.layout().itemAt(1).widget(), self.main.ui.detectionImageSelector)
        self.assertIsNone(getattr(self.main.ui, "detectionRunSummary", None))
        self.assertIsNone(getattr(self.main.ui, "detectionImageStatus", None))

    def test_production_layout_assigns_extra_height_to_splitter_at_different_fonts(self):
        browser = self.main.ui.detectionBrowser
        for point_size, width in ((9, 1400), (15, 1800)):
            with self.subTest(point_size=point_size, width=width):
                splitter = self.show_layout(point_size, width, 1000)
                self.assertIsNotNone(splitter)
                before = (browser.height(), splitter.height(), self.main.ui.groupBox_3.height())
                self.main.resize(self.main.width(), self.main.height() + 600)
                self.app.processEvents()
                self.assertEqual(browser.height(), before[0])
                self.assertGreater(splitter.height(), before[1])
                self.assertEqual(splitter.height() - before[1],
                                 self.main.ui.groupBox_3.height() - before[2])
                caption = browser.findChildren(QLabel)[0]
                margins = browser.layout().contentsMargins()
                natural = max(caption.sizeHint().height(),
                              self.main.ui.detectionImageSelector.sizeHint().height())
                self.assertEqual(browser.height(), natural + margins.top() + margins.bottom())
                old_sizes = splitter.sizes()
                splitter.setSizes(list(reversed(old_sizes)))
                self.app.processEvents()
                self.assertNotEqual(splitter.sizes(), old_sizes)

    def start(self, paths, scope="current_image"):
        self.main.set_detection_options(scope, "per_image")
        self.main._batch_controller.replace_images(paths)
        task = self.main._batch_controller.begin()
        self.main._clear_detection_results_for_new_run()
        self.main._set_active_worker_task("detection")
        self.main._dispatch_detection_task(task)
        return task

    def test_long_information_and_pending_tooltips_follow_selection_without_height_growth(self):
        self.show_layout(12, 1500, 1100)
        browser = self.main.ui.detectionBrowser
        initial_height = browser.height()
        first_path = "Z:/qbatch-memory/" + "long_name_" * 30 + ".png"
        pending_path = "Z:/qbatch-memory/pending.png"
        warning = "UNIQUE_FIRST_WARNING " * 40
        task = self.start([first_path, pending_path])
        self.main._on_detection_finished(runtime_payload(task, warnings=[warning]))
        pending_path = self.main.batch_state.images[1].path
        selector = self.main.ui.detectionImageSelector
        for text in (task.path, "completed", "unsaved", warning):
            self.assertIn(text, selector.toolTip())
            self.assertIn(text, selector.itemData(0, Qt.ToolTipRole))
        self.assertEqual(selector.toolTip(), self.main.ui.labelRecgImg.toolTip())
        self.assertIn("Completed this run: 1", self.statuses[-1])
        self.assertEqual(self.main.ui.pushButton_8.text(), "Save Detection")
        self.assertIn(task.path, self.main.ui.pushButton_8.toolTip())
        selector.setCurrentIndex(1)
        self.app.processEvents()
        self.assertIn(pending_path, selector.toolTip())
        self.assertIn("pending", selector.toolTip())
        self.assertIn("not been processed", selector.toolTip())
        self.assertNotIn(warning, selector.toolTip())
        self.assertNotIn(task.path, selector.toolTip())
        self.assertEqual(selector.toolTip(), self.main.ui.labelRecgImg.toolTip())
        self.assertIn("pending", self.main.ui.labelRecgImg.text())
        self.assertNotIn("\n", self.main.ui.labelRecgImg.text())
        self.assertNotIn(pending_path, self.main.ui.labelRecgImg.text())
        self.assertIsNone(self.main._detection_result)
        self.assertEqual(self.main.ui.tabviewRecg.model().rowCount(), 0)
        self.assertFalse(self.main.ui.pushButton_8.isEnabled())
        self.assertEqual(browser.height(), initial_height)
        self.start(["Z:/qbatch-memory/replacement.png"])
        self.assertNotIn(pending_path, selector.toolTip())
        self.assertNotIn(warning, self.main.ui.labelRecgImg.toolTip())

    def test_failed_background_and_diagnostics_restore_with_real_results(self):
        first = self.start(["Z:/qbatch-memory/success.png", "Z:/qbatch-memory/failed.png"],
                           scope="entire_batch")
        self.main._on_detection_finished(runtime_payload(first, warnings=["SUCCESS_ONLY"]))
        failed = self.main._batch_controller.active_job
        data = runtime_payload(failed, count=0, warnings=["FAILED_ONLY"], errors=[
            sample_error(failed, "invalid_roi", "FAILED_ROI")])
        data["image"][:] = (23, 34, 45)
        self.main._on_detection_finished(data)
        selector = self.main.ui.detectionImageSelector
        for index in (0, 1, 0, 1):
            selector.setCurrentIndex(index)
            expected = "SUCCESS_ONLY" if index == 0 else "FAILED_ONLY"
            obsolete = "FAILED_ONLY" if index == 0 else "SUCCESS_ONLY"
            self.assertIn(expected, selector.toolTip())
            self.assertNotIn(obsolete, selector.toolTip())
            self.assertEqual(selector.toolTip(), self.main.ui.labelRecgImg.toolTip())
            before = self.main._capture_result_pane_view()
            self.main.set_linear_mode(detectmain.LINEAR_MODE_IMAGE_SERIES)
            self.assertNotIn(expected, self.main.ui.labelRecgImg.toolTip())
            self.main.set_linear_mode(detectmain.LINEAR_MODE_SINGLE_IMAGE)
            self.assertIs(self.main._recgPixmap, before.pixmap)
            self.assertIs(self.main.ui.tabviewRecg.model(), before.model)
            self.assertEqual(selector.toolTip(), self.main.ui.labelRecgImg.toolTip())
            if index == 1:
                self.assertIn("FAILED_ROI", selector.toolTip())
                self.assertEqual(self.main._recgPixmap.toImage().pixelColor(0, 0).blue(), 23)
                self.assertEqual(self.main.ui.tabviewRecg.model().rowCount(), 0)
                self.assertIsNone(self.main._detection_result)
                self.assertFalse(self.main.ui.pushButton_8.isEnabled())
            else:
                self.assertTrue(self.main._has_valid_detection_export())
                self.assertTrue(self.main.ui.pushButton_8.isEnabled())
        self.assertEqual(len(self.requests), 2)

    def test_valid_detection_without_browser_entries_survives_modes_and_refresh(self):
        # Legacy Single result: no batch rows, but a valid payload and dirty flag.
        image = np.full((12, 16, 3), (17, 28, 39), dtype=np.uint8)
        result = {"source_path": "Z:/qbatch-memory/legacy.png", "image": image,
                  "targets": [{"No.": 1, "Con.": 0.5, "Red": 1., "Green": 2.,
                               "Blue": 3., "rgb_roi": (3, 3, 7, 7)}]}
        self.main._detection_result = result
        self.main._detection_dirty = True
        self.main._last_completed_result_type = "detection"
        model = self.main._build_table_model(("No.", "Con.", "Red", "Green", "Blue"),
                                             [(1, .5, 1., 2., 3.)])
        pixmap = self.main._bgr_image_to_pixmap(image)
        self.main._publish_detection_result_view(pixmap, model)
        self.main.ui.progressBar.setValue(73)
        self.main.ui.lcdNumber.display(246)
        for _ in range(2):
            self.main._refresh_detection_browser()
            self.main._update_save_button()
            self.assertEqual(self.main.ui.detectionImageSelector.count(), 0)
            self.assertTrue(self.main._has_valid_detection_export())
            self.assertTrue(self.main.ui.pushButton_8.isEnabled())
            self.main.set_linear_mode(detectmain.LINEAR_MODE_IMAGE_SERIES)
            self.main.ui.progressBar.setRange(0, 0)
            self.main.ui.lcdNumber.display(999)
            self.main.set_linear_mode(detectmain.LINEAR_MODE_SINGLE_IMAGE)
            self.assertIs(self.main._detection_result, result)
            self.assertIs(self.main._recgPixmap, pixmap)
            self.assertIs(self.main.ui.tabviewRecg.model(), model)
            self.assertTrue(self.main._detection_dirty)
            self.assertEqual((self.main.ui.progressBar.minimum(),
                              self.main.ui.progressBar.maximum(),
                              self.main.ui.progressBar.value()), (0, 100, 73))
            self.assertEqual(self.main.ui.lcdNumber.value(), 246)


class _CallbackSignal:
    def __init__(self):
        self._slots = []
        self.connect_calls = 0
        self.disconnect_calls = 0

    def connect(self, slot):
        self.connect_calls += 1
        self._slots.append(slot)

    def disconnect(self, slot):
        self.disconnect_calls += 1
        try:
            self._slots.remove(slot)
        except ValueError as error:
            raise TypeError("slot is not connected") from error

    def emit(self, *args):
        for slot in tuple(self._slots):
            slot(*args)


class _SimulatedVideoDevice:
    def __init__(self, is_null=False):
        self.is_null = is_null

    def isNull(self):
        return self.is_null


class _SimulatedCaptureSession:
    def __init__(self, _parent):
        self.camera = None
        self.video_output = None
        self.deleted = False

    def setCamera(self, camera):
        self.camera = camera

    def setVideoOutput(self, output):
        self.video_output = output

    def deleteLater(self):
        self.deleted = True


class _SimulatedCameraBackend:
    def __init__(self, _device, _parent):
        self.activeChanged = _CallbackSignal()
        self.errorOccurred = _CallbackSignal()
        self.start_calls = 0
        self.stop_calls = 0
        self.active = False
        self.deleted = False
        self.diagnostic = "simulated camera error"

    def start(self):
        self.start_calls += 1

    def stop(self):
        self.stop_calls += 1
        was_active = self.active
        self.active = False
        if was_active:
            self.activeChanged.emit(False)

    def isActive(self):
        return self.active

    def activate(self):
        self.active = True
        self.activeChanged.emit(True)

    def fail_to_activate(self):
        self.active = False
        self.activeChanged.emit(False)

    def fail_while_running(self):
        self.errorOccurred.emit(object())

    def error(self):
        return 1

    def errorString(self):
        return self.diagnostic

    def deleteLater(self):
        self.deleted = True


class CameraMenuStateContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.device = _SimulatedVideoDevice()
        self.backends = []
        self.sessions = []

        def make_backend(device, parent):
            backend = _SimulatedCameraBackend(device, parent)
            self.backends.append(backend)
            return backend

        def make_session(parent):
            session = _SimulatedCaptureSession(parent)
            self.sessions.append(session)
            return session

        devices = SimpleNamespace(defaultVideoInput=lambda: self.device)
        for boundary in (
            patch.object(camera_module, "QMediaDevices", devices),
            patch.object(camera_module, "QCamera", side_effect=make_backend),
            patch.object(camera_module, "QMediaCaptureSession", side_effect=make_session),
            patch.object(camera_module.QMessageBox, "warning"),
        ):
            boundary.start()
            self.addCleanup(boundary.stop)
        self.camera = Camera(camera_enabled=True)
        self.addCleanup(self._shutdown_camera)

    def _shutdown_camera(self):
        self.camera.request_shutdown()
        self.camera.deleteLater()

    def test_async_success_manual_stop_and_restart_notify_once_per_transition(self):
        states = []
        self.camera.camera_state_changed.connect(states.append)

        self.camera.start_camera()
        self.camera.start_camera()
        self.assertEqual(self.camera.camera_state, Camera.STATE_STARTING)
        self.assertEqual(len(self.backends), 1)
        self.assertEqual(self.backends[0].start_calls, 1)
        first_backend = self.backends[0]
        self.assertEqual(first_backend.activeChanged.connect_calls, 1)
        self.assertEqual(first_backend.errorOccurred.connect_calls, 1)
        self.assertEqual(len(first_backend.activeChanged._slots), 1)
        self.assertEqual(len(first_backend.errorOccurred._slots), 1)
        first_backend.activate()
        self.assertEqual(self.camera.camera_state, Camera.STATE_ON)

        self.camera.stop_camera()
        self.camera.stop_camera()
        self.assertEqual(self.camera.camera_state, Camera.STATE_OFF)
        self.assertEqual(first_backend.stop_calls, 1)
        self.assertEqual(first_backend.activeChanged.disconnect_calls, 1)
        self.assertEqual(first_backend.errorOccurred.disconnect_calls, 1)
        self.assertEqual(first_backend.activeChanged._slots, [])
        self.assertEqual(first_backend.errorOccurred._slots, [])
        first_backend.activeChanged.emit(True)
        first_backend.errorOccurred.emit(object())
        self.assertEqual(self.camera.camera_state, Camera.STATE_OFF)

        self.camera.start_camera()
        self.assertEqual(len(self.backends), 2)
        self.backends[1].activate()
        self.assertEqual(self.camera.camera_state, Camera.STATE_ON)
        self.assertEqual(states, [
            Camera.STATE_STARTING,
            Camera.STATE_ON,
            Camera.STATE_STOPPING,
            Camera.STATE_OFF,
            Camera.STATE_STARTING,
            Camera.STATE_ON,
        ])

    def test_start_failure_and_runtime_error_return_to_off_and_keep_retry_available(self):
        states = []
        self.camera.camera_state_changed.connect(states.append)
        self.device.is_null = True
        self.camera.start_camera()
        self.assertEqual(self.camera.camera_state, Camera.STATE_OFF)
        self.assertEqual(states, [Camera.STATE_STARTING, Camera.STATE_OFF])
        camera_module.QMessageBox.warning.assert_called_once()

        self.device.is_null = False
        self.camera.start_camera()
        backend = self.backends[-1]
        backend.activate()
        backend.fail_while_running()
        self.assertEqual(self.camera.camera_state, Camera.STATE_OFF)
        self.assertIsNone(self.camera.m_camera)
        self.assertEqual(camera_module.QMessageBox.warning.call_count, 2)
        self.camera.start_camera()
        self.assertEqual(self.camera.camera_state, Camera.STATE_STARTING)

    def test_asynchronous_start_failure_returns_to_off_and_notifies_once(self):
        states = []
        self.camera.camera_state_changed.connect(states.append)

        self.camera.start_camera()
        backend = self.backends[-1]
        self.assertEqual(self.camera.camera_state, Camera.STATE_STARTING)
        backend.fail_to_activate()

        self.assertEqual(self.camera.camera_state, Camera.STATE_OFF)
        self.assertIsNone(self.camera.m_camera)
        self.assertEqual(states, [Camera.STATE_STARTING, Camera.STATE_OFF])
        camera_module.QMessageBox.warning.assert_called_once()

    def test_camera_state_property_is_read_only(self):
        self.assertEqual(self.camera.camera_state, Camera.STATE_OFF)
        with self.assertRaises(AttributeError):
            self.camera.camera_state = Camera.STATE_ON
        self.assertEqual(self.camera.camera_state, Camera.STATE_OFF)

    def test_public_visibility_control_hides_legacy_toggle_without_disabling_preview(self):
        button = self.camera._camera_toggle_button
        controls_layout = button.parentWidget().layout()
        button_item = next(
            (
                controls_layout.itemAt(index)
                for index in range(controls_layout.count())
                if controls_layout.itemAt(index).widget() is button
            ),
            None,
        )
        capture_widget = self.camera._ui.captureWidget
        tab_widgets = (
            [capture_widget]
            if isinstance(capture_widget, QTabWidget)
            else capture_widget.findChildren(QTabWidget)
        )
        self.assertIsNotNone(button_item)
        self.assertTrue(tab_widgets)
        tab_snapshot = tuple(
            (tabs, tuple(tabs.tabText(index) for index in range(tabs.count())))
            for tabs in tab_widgets
        )
        tab_titles = {
            title.strip().casefold()
            for _tabs, titles in tab_snapshot
            for title in titles
        }
        self.assertTrue({"image", "video"}.issubset(tab_titles))
        retained_controls = (
            self.camera._ui.takeImageButton,
            self.camera._ui.recordButton,
            self.camera._ui.pauseButton,
            self.camera._ui.stopButton,
            self.camera._ui.muteButton,
            self.camera._ui.metaDataButton,
        )

        self.camera.set_camera_toggle_visible(False)
        self.camera.set_camera_toggle_visible(False)
        self.assertTrue(button.isHidden())
        self.assertTrue(button_item.isEmpty())
        self.assertFalse(self.camera._ui.stackedWidget.isHidden())
        self.camera.set_camera_toggle_visible(True)
        self.assertFalse(button.isHidden())
        self.camera.set_camera_toggle_visible(False)
        self.assertTrue(button.isHidden())
        self.assertEqual(
            tuple(
                (tabs, tuple(tabs.tabText(index) for index in range(tabs.count())))
                for tabs in tab_widgets
            ),
            tab_snapshot,
        )
        self.assertTrue(all(control.parent() is not None for control in retained_controls))


class _WindowCamera(QObject):
    STATE_OFF = "OFF"
    STATE_STARTING = "STARTING"
    STATE_ON = "ON"
    STATE_STOPPING = "STOPPING"
    STATE_DISABLED = "DISABLED"
    STATE_SHUTTING_DOWN = "SHUTTING_DOWN"

    def __init__(self, enabled=True, start_outcome="pending"):
        super().__init__()
        self.camera_state_changed = _CallbackSignal()
        self.camera_enabled = enabled
        self._state = self.STATE_OFF if enabled else self.STATE_DISABLED
        self.start_outcome = start_outcome
        self.start_attempts = 0
        self.stop_attempts = 0
        self.start_calls = 0
        self.stop_calls = 0
        self.toggle_visibility = []
        self._ui = SimpleNamespace(cameraWidget=QWidget())

    @property
    def camera_state(self):
        return self._state

    def _set_state(self, state):
        if state == self._state:
            return
        self._state = state
        self.camera_state_changed.emit(state)

    def set_camera_toggle_visible(self, visible):
        self.toggle_visibility.append(bool(visible))

    def start_camera(self):
        self.start_attempts += 1
        if not self.camera_enabled or self._state != self.STATE_OFF:
            return
        self.start_calls += 1
        self._set_state(self.STATE_STARTING)
        if self.start_outcome == "success":
            self._set_state(self.STATE_ON)
        elif self.start_outcome == "failure":
            self._set_state(self.STATE_OFF)

    def stop_camera(self):
        self.stop_attempts += 1
        if self._state not in (self.STATE_STARTING, self.STATE_ON):
            return
        self.stop_calls += 1
        self._set_state(self.STATE_STOPPING)
        self._set_state(self.STATE_OFF)

    def complete_start(self):
        self._set_state(self.STATE_ON)

    def fail_start(self):
        self._set_state(self.STATE_OFF)

    def runtime_error(self):
        self._set_state(self.STATE_OFF)

    def set_close_wait_pending(self, _pending):
        pass

    def request_shutdown(self):
        self._set_state(self.STATE_SHUTTING_DOWN)


class _WindowMain(QObject):
    worker_task_finished = Signal()
    shutdown_ready = Signal()
    busy_changed = Signal(bool)
    detection_status_changed = Signal(str)

    def __init__(self, camera):
        super().__init__()
        self.mainCamera = camera
        self.ui = SimpleNamespace(detectmainWidget=QWidget())
        self.detection_options = []
        self.linear_modes = []
        self.worker_active = False
        self.linear_locked = False
        self._detection_result = object()
        self._detection_dirty = True
        self._regression_result = object()
        self._regression_dirty = True
        self.save_state = object()
        self.shutdown_calls = 0

    def set_detection_options(self, scope, numbering):
        self.detection_options.append((scope, numbering))

    def set_linear_mode(self, mode):
        self.linear_modes.append(mode)
        return mode

    def is_worker_task_active(self):
        return self.worker_active

    def is_linear_interaction_locked(self):
        return self.linear_locked

    def request_shutdown(self):
        self.shutdown_calls += 1
        self.mainCamera.request_shutdown()


class _WindowControl:
    def __init__(self, text=""):
        self.clicked = _CallbackSignal()
        self.textActivated = _CallbackSignal()
        self._text = text

    def currentText(self):
        return self._text


class _WindowDetectFile:
    def __init__(self):
        self.fileModel = SimpleNamespace(setNameFilters=Mock(), setRootPath=Mock())
        self.dirModel = SimpleNamespace(fileInfo=Mock())
        self.ui = SimpleNamespace(
            filemanagerTabWidget=QWidget(),
            dirTreeView=_WindowControl(),
            fileListView=_WindowControl(),
            extComboBox=_WindowControl("*.png"),
        )


class DetectionWindowMenuContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def make_window(self, start_outcome="pending", camera_enabled=True, camera=None):
        if camera is None:
            camera = _WindowCamera(camera_enabled, start_outcome)
        main = _WindowMain(camera)
        detect_file = _WindowDetectFile()
        preferences = {
            "detection_scope": "current_image",
            "numbering_mode": "per_image",
        }
        self.saved_preferences = Mock()
        for boundary in (
            patch.object(detectionwindow, "DetectMain", return_value=main),
            patch.object(detectionwindow, "DetectFile", return_value=detect_file),
            patch.object(detectionwindow, "load_detection_preferences",
                         return_value=(preferences, [])),
            patch.object(detectionwindow, "save_detection_preferences",
                         self.saved_preferences),
        ):
            boundary.start()
            self.addCleanup(boundary.stop)
        window = detectionwindow.DetectWindow(object(), camera_enabled=camera_enabled)
        self.addCleanup(window.deleteLater)
        self.addCleanup(main.deleteLater)
        self.addCleanup(camera.request_shutdown)
        self.addCleanup(camera.deleteLater)
        return window, main, camera

    def make_real_camera_window(self):
        device = _SimulatedVideoDevice()
        backends = []

        def make_backend(video_device, parent):
            backend = _SimulatedCameraBackend(video_device, parent)
            backends.append(backend)
            return backend

        devices = SimpleNamespace(defaultVideoInput=lambda: device)
        for boundary in (
            patch.object(camera_module, "QMediaDevices", devices),
            patch.object(camera_module, "QCamera", side_effect=make_backend),
            patch.object(
                camera_module,
                "QMediaCaptureSession",
                side_effect=_SimulatedCaptureSession,
            ),
            patch.object(camera_module.QMessageBox, "warning"),
        ):
            boundary.start()
            self.addCleanup(boundary.stop)
        camera = Camera(camera_enabled=True)
        window, main, camera = self.make_window(camera=camera)
        return window, main, camera, backends

    @staticmethod
    def result_snapshot(main):
        return (
            main._detection_result,
            main._detection_dirty,
            main._regression_result,
            main._regression_dirty,
            main.save_state,
        )

    def test_default_start_async_success_manual_stop_restart_and_runtime_error_sync_action(self):
        window, main, camera = self.make_window()
        action = window._uiWindow.actionEnableCamera
        original_results = self.result_snapshot(main)

        self.assertTrue(action.isCheckable())
        self.assertEqual(camera.toggle_visibility, [False])
        self.assertEqual(camera.start_calls, 1)
        self.assertEqual(camera.camera_state, camera.STATE_STARTING)
        self.assertFalse(action.isChecked())

        action.trigger()
        self.assertEqual(camera.start_attempts, 2)
        self.assertEqual(camera.start_calls, 1)
        self.assertEqual(camera.stop_attempts, 0)
        self.assertFalse(action.isChecked())

        camera.complete_start()
        self.assertTrue(action.isChecked())
        self.assertEqual(camera.start_calls, 1)
        action.trigger()
        self.assertEqual(camera.stop_calls, 1)
        self.assertEqual(camera.camera_state, camera.STATE_OFF)
        self.assertFalse(action.isChecked())

        action.trigger()
        self.assertEqual(camera.start_calls, 2)
        camera.complete_start()
        camera.runtime_error()
        self.assertFalse(action.isChecked())
        self.assertEqual((camera.start_calls, camera.stop_calls), (2, 1))
        self.assertEqual(self.result_snapshot(main), original_results)

    def test_default_start_failure_unchecks_and_allows_manual_retry(self):
        window, main, camera = self.make_window(start_outcome="failure")
        action = window._uiWindow.actionEnableCamera
        original_results = self.result_snapshot(main)
        self.assertEqual(camera.start_calls, 1)
        self.assertFalse(action.isChecked())
        self.assertTrue(action.isEnabled())
        action.trigger()
        self.assertEqual(camera.start_calls, 2)
        self.assertFalse(action.isChecked())
        self.assertEqual(self.result_snapshot(main), original_results)

    def test_async_default_start_failure_unchecks_and_allows_manual_retry(self):
        window, main, camera = self.make_window(start_outcome="pending")
        action = window._uiWindow.actionEnableCamera
        original_results = self.result_snapshot(main)

        self.assertFalse(action.isChecked())
        camera.fail_start()
        self.assertFalse(action.isChecked())
        self.assertTrue(action.isEnabled())

        camera.start_outcome = "failure"
        action.trigger()
        self.assertEqual(camera.start_calls, 2)
        self.assertFalse(action.isChecked())
        self.assertEqual(self.result_snapshot(main), original_results)

    def test_programmatic_sync_and_duplicate_notifications_do_not_control_camera(self):
        window, main, camera = self.make_window(start_outcome="pending")
        action = window._uiWindow.actionEnableCamera
        self.assertEqual(camera.camera_state_changed.connect_calls, 1)
        self.assertEqual(len(camera.camera_state_changed._slots), 1)
        self.assertFalse(action.isChecked())
        attempts = (camera.start_attempts, camera.stop_attempts)

        action.setChecked(True)
        self.assertEqual((camera.start_attempts, camera.stop_attempts), attempts)
        window._sync_camera_action(camera.STATE_STARTING)
        self.assertFalse(action.isChecked())
        camera.camera_state_changed.emit(camera.STATE_STARTING)
        camera.camera_state_changed.emit(camera.STATE_STARTING)
        self.assertEqual((camera.start_attempts, camera.stop_attempts), attempts)
        self.assertFalse(action.isChecked())

    def test_shutdown_and_disabled_camera_sync_never_start_camera(self):
        window, main, camera = self.make_window(start_outcome="success")
        action = window._uiWindow.actionEnableCamera
        self.assertTrue(action.isChecked())
        start_attempts = camera.start_attempts

        window._begin_shutdown()
        self.assertTrue(window._shutdown_started)
        self.assertEqual(main.shutdown_calls, 1)
        self.assertEqual(camera.camera_state, camera.STATE_SHUTTING_DOWN)
        self.assertFalse(action.isChecked())
        self.assertFalse(action.isEnabled())
        camera.camera_state_changed.emit(camera.STATE_SHUTTING_DOWN)
        action.setChecked(True)
        window._sync_camera_action(camera.camera_state)
        self.assertFalse(action.isChecked())
        self.assertEqual(camera.start_attempts, start_attempts)

        disabled_window, _disabled_main, disabled_camera = self.make_window(
            camera_enabled=False
        )
        disabled_action = disabled_window._uiWindow.actionEnableCamera
        self.assertEqual(disabled_camera.camera_state, disabled_camera.STATE_DISABLED)
        self.assertEqual(disabled_camera.start_attempts, 0)
        self.assertFalse(disabled_action.isChecked())
        self.assertFalse(disabled_action.isEnabled())

    def test_real_camera_state_signal_drives_real_window_action(self):
        window, main, camera, backends = self.make_real_camera_window()
        action = window._uiWindow.actionEnableCamera
        original_results = self.result_snapshot(main)

        self.assertEqual(camera.camera_state, Camera.STATE_STARTING)
        self.assertFalse(action.isChecked())
        self.assertEqual(len(backends), 1)
        self.assertEqual(backends[0].start_calls, 1)
        backends[0].activate()
        self.assertEqual(camera.camera_state, Camera.STATE_ON)
        self.assertTrue(action.isChecked())

        action.trigger()
        self.assertEqual(camera.camera_state, Camera.STATE_OFF)
        self.assertFalse(action.isChecked())
        self.assertEqual(backends[0].stop_calls, 1)

        action.trigger()
        self.assertEqual(camera.camera_state, Camera.STATE_STARTING)
        self.assertFalse(action.isChecked())
        self.assertEqual(len(backends), 2)
        self.assertEqual(backends[1].start_calls, 1)
        backends[1].activate()
        self.assertEqual(camera.camera_state, Camera.STATE_ON)
        self.assertTrue(action.isChecked())
        self.assertEqual(self.result_snapshot(main), original_results)

    def test_detection_and_linear_are_existing_edit_submenus_with_live_actions(self):
        reference_window = QMainWindow()
        reference_ui = detectionwindow.ui_detectwindow.Ui_detectWindow()
        reference_ui.setupUi(reference_window)
        self.addCleanup(reference_window.deleteLater)
        window, main, camera = self.make_window(start_outcome="success")
        ui = window._uiWindow
        top_level_menus = [action.menu() for action in ui.menubar.actions()]
        self.assertNotIn(ui.menuDetection, top_level_menus)
        self.assertNotIn(ui.menuLinear, top_level_menus)
        self.assertIn(ui.menuEdit, top_level_menus)

        edit_actions = ui.menuEdit.actions()
        original_edit_names = (
            "actionUndo",
            "actionRedo",
            "actionCut",
            "actionCopy",
            "actionPaste",
            "actionRemove",
            "actionDelete",
            "actionAdvanced",
            "actionFind_Replace",
            "actionPreferences",
        )
        original_edit_actions = [getattr(ui, name) for name in original_edit_names]
        self.assertEqual(
            [action for action in edit_actions if not action.isSeparator()],
            original_edit_actions + [
                ui.actionEnableCamera,
                ui.menuDetection.menuAction(),
                ui.menuLinear.menuAction(),
            ],
        )
        self.assertTrue(ui.actionEnableCamera.isCheckable())
        self.assertEqual(edit_actions.count(ui.menuDetection.menuAction()), 1)
        self.assertEqual(edit_actions.count(ui.menuLinear.menuAction()), 1)
        self.assertLess(
            edit_actions.index(ui.actionEnableCamera),
            edit_actions.index(ui.menuDetection.menuAction()),
        )
        self.assertLess(
            edit_actions.index(ui.menuDetection.menuAction()),
            edit_actions.index(ui.menuLinear.menuAction()),
        )
        self.assertIs(ui.menuDetection.menuAction().menu(), ui.menuDetection)
        self.assertIs(ui.menuLinear.menuAction().menu(), ui.menuLinear)
        self.assertEqual(ui.menuDetection.actions(), [
            ui.menuDetection_Scope.menuAction(),
            ui.menuDetection_Numbering.menuAction(),
        ])
        self.assertEqual(ui.menuDetection_Scope.actions(), [
            ui.actionDetection_Current_Image,
            ui.actionDetection_Entire_Batch,
        ])
        self.assertEqual(ui.menuDetection_Numbering.actions(), [
            ui.actionDetection_Per_Image,
            ui.actionDetection_Continuous,
        ])
        self.assertEqual(ui.menuLinear.actions(), [
            ui.actionLinear_Single_Image,
            ui.actionLinear_Image_Series,
        ])
        self.assertTrue(window._detection_scope_group.isExclusive())
        self.assertTrue(window._numbering_group.isExclusive())
        self.assertTrue(window._linear_mode_group.isExclusive())
        self.assertEqual(window._detection_scope_group.actions(), [
            ui.actionDetection_Current_Image,
            ui.actionDetection_Entire_Batch,
        ])
        self.assertEqual(window._numbering_group.actions(), [
            ui.actionDetection_Per_Image,
            ui.actionDetection_Continuous,
        ])
        self.assertEqual(window._linear_mode_group.actions(), [
            ui.actionLinear_Single_Image,
            ui.actionLinear_Image_Series,
        ])
        self.assertTrue(ui.actionDetection_Current_Image.isChecked())
        self.assertFalse(ui.actionDetection_Entire_Batch.isChecked())
        self.assertTrue(ui.actionDetection_Per_Image.isChecked())
        self.assertFalse(ui.actionDetection_Continuous.isChecked())
        self.assertTrue(ui.actionLinear_Single_Image.isChecked())
        self.assertFalse(ui.actionLinear_Image_Series.isChecked())
        for name in original_edit_names + (
            "actionDetection_Current_Image",
            "actionDetection_Entire_Batch",
            "actionDetection_Per_Image",
            "actionDetection_Continuous",
            "actionLinear_Single_Image",
            "actionLinear_Image_Series",
        ):
            self.assertEqual(
                getattr(ui, name).shortcuts(),
                getattr(reference_ui, name).shortcuts(),
            )

        ui.actionDetection_Entire_Batch.trigger()
        self.assertEqual(main.detection_options[-1], ("entire_batch", "per_image"))
        ui.actionDetection_Continuous.trigger()
        self.assertEqual(main.detection_options[-1], ("entire_batch", "continuous"))
        self.assertEqual(self.saved_preferences.call_args_list[-1].args[0], {
            "detection_scope": "entire_batch",
            "numbering_mode": "continuous",
        })
        ui.actionDetection_Current_Image.trigger()
        ui.actionDetection_Per_Image.trigger()
        self.assertEqual(main.detection_options[-1], ("current_image", "per_image"))
        starts_before_mode_change = camera.start_calls
        ui.actionLinear_Image_Series.trigger()
        self.assertEqual(main.linear_modes[-1], detectmain.LINEAR_MODE_IMAGE_SERIES)
        ui.actionLinear_Single_Image.trigger()
        self.assertEqual(main.linear_modes[-1], detectmain.LINEAR_MODE_SINGLE_IMAGE)
        self.assertEqual(camera.start_calls, starts_before_mode_change)

    def test_busy_rules_and_camera_sync_preserve_existing_results(self):
        window, main, camera = self.make_window(start_outcome="success")
        ui = window._uiWindow
        original_results = self.result_snapshot(main)
        main.busy_changed.emit(True)
        self.assertFalse(ui.menuDetection_Scope.isEnabled())
        self.assertFalse(ui.menuDetection_Numbering.isEnabled())
        self.assertFalse(ui.menuLinear.isEnabled())
        camera.runtime_error()
        self.assertFalse(ui.actionEnableCamera.isChecked())
        self.assertEqual(self.result_snapshot(main), original_results)
        main.busy_changed.emit(False)
        self.assertTrue(ui.menuDetection_Scope.isEnabled())
        self.assertTrue(ui.menuDetection_Numbering.isEnabled())
        self.assertTrue(ui.menuLinear.isEnabled())


if __name__ == "__main__":
    unittest.main()
