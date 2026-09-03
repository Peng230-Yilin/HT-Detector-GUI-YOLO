import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Peng1.0_GUI"))

from batch_detection_controller import BatchDetectionController  # noqa: E402
from batch_state import DetectionScope, ImageStatus, NumberingMode  # noqa: E402
from detectmain import DetectMain  # noqa: E402


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

    def clear(self):
        self.cleared = True


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
        self.assertEqual(
            harness.detection_status_changed.values[-1][0],
            "Detecting 1/2: image1.png",
        )
        harness._batch_controller.accept_payload(payload(first, 1))
        second = harness._batch_controller.next_task()
        DetectMain._dispatch_detection_task(harness, second)
        self.assertEqual(
            harness.detection_status_changed.values[-1][0],
            "Detecting 2/2: image2.png",
        )
        harness._batch_controller.accept_failure(
            second.run_token, second.job_token, "broken"
        )
        DetectMain._finish_detection_run(harness)
        self.assertIn("Total images: 2", harness.messages[0][1])
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


if __name__ == "__main__":
    unittest.main()
