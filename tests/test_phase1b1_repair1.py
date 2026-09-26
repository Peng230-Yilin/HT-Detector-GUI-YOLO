import copy
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from batch_detection_controller import BatchDetectionController  # noqa: E402
from batch_state import (  # noqa: E402
    CalibrationAttemptState,
    ConcentrationStatus,
    DetectionScope,
    ImageItem,
    ImageStatus,
    NumberingMode,
    RegressionSessionState,
    RegressionSet,
    SampleResult,
    detection_table_rows,
)
from detectmain import DetectMain  # noqa: E402
from yolo_detection_worker import YoloDetectionWorker  # noqa: E402


def formula(slope, intercept):
    return {
        "slope": slope,
        "intercept": intercept,
        "r": 0.9,
        "R2": 0.81,
        "p": 0.01,
        "std_err": 0.1,
    }


def regression(source_id, offset=0.0):
    return RegressionSet.from_formulas(
        {
            "R": formula(2.0, 10.0 + offset),
            "G": formula(3.0, 20.0 + offset),
            "B": formula(4.0, 30.0 + offset),
        },
        source_id=source_id,
        valid_ranges=(0.0, 100.0),
    )


def callback_identity(task, **extra):
    value = {
        "run_token": task.run_token,
        "job_token": task.job_token,
        "regression_revision": task.regression_revision,
        "image_order": task.image_order,
        "source_path": task.path,
        "source_file": task.source_file,
    }
    value.update(extra)
    return value


def success_payload(task):
    return callback_identity(task, sample_results=[], sample_errors=[])


def controller_snapshot(controller):
    return {
        "active_job": controller.active_job,
        "queue": tuple(controller._queue),
        "next_batch_no": controller._next_batch_no,
        "current_image_index": controller.state.current_image_index,
        "last_batch_result": controller.state.last_batch_result,
        "images": copy.deepcopy(controller.state.images),
    }


class TestRunRegressionAuthority(unittest.TestCase):
    def test_run_snapshot_survives_batch_state_replacement_and_new_run_uses_new_set(self):
        old = regression("old")
        new = regression("new", offset=1.0)
        controller = BatchDetectionController()
        controller.set_options(DetectionScope.ALL_IMPORTED_IMAGES, NumberingMode.PER_IMAGE)
        controller.replace_images(["two.png", "one.png"])

        first = controller.begin(old)
        run_context = controller.run_context
        controller.state.regression_set = new
        self.assertTrue(controller.accept_payload(success_payload(first)))
        second = controller.next_task()

        self.assertIs(controller.run_context, run_context)
        self.assertIs(first.regression_set, old)
        self.assertIs(second.regression_set, old)
        self.assertEqual(first.regression_revision, old.revision)
        self.assertEqual(second.regression_revision, old.revision)
        self.assertIs(controller.state.regression_set, new)

        self.assertTrue(controller.accept_payload(success_payload(second)))
        controller.finish_if_done()
        next_run = controller.begin()
        self.assertIs(next_run.regression_set, new)
        self.assertEqual(next_run.regression_revision, new.revision)
        self.assertNotEqual(next_run.run_token, first.run_token)


class TestStrictCallbackIdentity(unittest.TestCase):
    def test_failure_missing_any_identity_field_is_rejected_without_state_change(self):
        for missing_field in (
            "run_token",
            "job_token",
            "regression_revision",
            "image_order",
            "source_path",
            "source_file",
        ):
            with self.subTest(missing_field=missing_field):
                controller = BatchDetectionController()
                controller.replace_images(["one.png", "two.png"])
                task = controller.begin(regression("identity"))
                failure = callback_identity(task, message="must be rejected")
                failure.pop(missing_field)
                before = controller_snapshot(controller)

                self.assertFalse(controller.accept_failure(failure))
                self.assertEqual(controller_snapshot(controller), before)

    def test_handlers_reject_incomplete_success_and_failure_before_any_state_change(self):
        for callback_name in ("_on_detection_finished", "_on_detection_failed"):
            for missing_field in (
                "run_token",
                "job_token",
                "regression_revision",
                "image_order",
                "source_path",
                "source_file",
            ):
                with self.subTest(callback=callback_name, missing_field=missing_field):
                    controller = BatchDetectionController()
                    controller.replace_images(["one.png"])
                    task = controller.begin(regression("handler-identity"))
                    callback = callback_identity(
                        task,
                        message="rejected",
                        sample_results=[],
                        sample_errors=[],
                    )
                    callback.pop(missing_field)
                    harness = SimpleNamespace(_batch_controller=controller)
                    before = controller_snapshot(controller)

                    getattr(DetectMain, callback_name)(harness, callback)

                    self.assertEqual(controller_snapshot(controller), before)

    def test_failure_source_or_order_mismatch_is_rejected_without_state_change(self):
        for field, wrong_value in (
            ("source_path", "different/location.png"),
            ("source_file", "different.png"),
            ("image_order", 99),
        ):
            with self.subTest(field=field):
                controller = BatchDetectionController()
                controller.replace_images(["one.png", "two.png"])
                task = controller.begin(regression("identity"))
                failure = callback_identity(task, message="must be rejected")
                failure[field] = wrong_value
                before = controller_snapshot(controller)

                self.assertFalse(controller.accept_failure(failure))
                self.assertEqual(controller_snapshot(controller), before)

    def test_valid_failure_changes_only_the_expected_image(self):
        controller = BatchDetectionController()
        controller.set_options(DetectionScope.ALL_IMPORTED_IMAGES, NumberingMode.PER_IMAGE)
        controller.replace_images(["one.png", "two.png"])
        task = controller.begin(regression("identity"))
        untouched_before = copy.deepcopy(controller.state.images[1])

        self.assertTrue(controller.accept_failure(
            callback_identity(task, message="controlled failure")
        ))

        self.assertEqual(controller.state.images[0].status, ImageStatus.FAILED)
        self.assertEqual(
            controller.state.images[0].errors[0].reason, "controlled failure"
        )
        self.assertEqual(controller.state.images[1], untouched_before)

    def test_worker_outer_failure_returns_the_complete_original_identity(self):
        controller = BatchDetectionController()
        controller.replace_images(["source.png"])
        task = controller.begin(regression("worker"))
        worker = YoloDetectionWorker()
        failures = []
        worker.failed.connect(failures.append)

        with patch(
            "yolo_detection_worker.load_effective_settings",
            side_effect=RuntimeError("settings unavailable"),
        ):
            worker.detect(task.path, "unused.pt", task.context())

        self.assertEqual(len(failures), 1)
        self.assertEqual({
            name: failures[0][name]
            for name in (
                "run_token",
                "job_token",
                "regression_revision",
                "image_order",
                "source_path",
                "source_file",
            )
        }, callback_identity(task))


def authoritative_sample():
    return SampleResult(
        image_order=1,
        source_file="sample.png",
        cuvette_box=(0.0, 0.0, 20.0, 40.0),
        liquid_box=(4.0, 10.0, 16.0, 35.0),
        roi_box=(6.0, 15.0, 14.0, 30.0),
        red=11.0,
        green=22.0,
        blue=33.0,
        con_r=1.25,
        con_g=2.5,
        con_b=3.75,
        status_r=ConcentrationStatus.IN_RANGE,
        status_g=ConcentrationStatus.IN_RANGE,
        status_b=ConcentrationStatus.IN_RANGE,
        no_in_image=4,
        batch_no=9,
    )


class TestDetectionTableAuthority(unittest.TestCase):
    def test_mutating_legacy_target_concentration_cannot_change_table_rows(self):
        image = ImageItem(
            "sample.png",
            "sample.png",
            1,
            status=ImageStatus.COMPLETED,
            samples=[authoritative_sample()],
        )
        cached_targets = [{"Con.": -999.0}]
        before = detection_table_rows(image, NumberingMode.PER_IMAGE, "R")
        cached_targets[0]["Con."] = 999999.0
        after = detection_table_rows(image, NumberingMode.PER_IMAGE, "R")

        self.assertEqual(before, [(4, 1.25, 11.0, 22.0, 33.0)])
        self.assertEqual(after, before)

    def test_each_display_channel_reads_its_sample_concentration(self):
        for channel, expected in (("R", 1.25), ("G", 2.5), ("B", 3.75)):
            with self.subTest(channel=channel):
                image = ImageItem(
                    "sample.png",
                    "sample.png",
                    1,
                    status=ImageStatus.COMPLETED,
                    samples=[authoritative_sample()],
                )
                row = detection_table_rows(
                    image, NumberingMode.CONTINUOUS_BATCH, channel
                )[0]
                self.assertEqual(row, (9, expected, 11.0, 22.0, 33.0))


class TestRegressionSessionAuthority(unittest.TestCase):
    def test_failed_calibration_without_active_regression_blocks_legacy_loader(self):
        session = RegressionSessionState()
        operation = session.begin_attempt("failed-calibration")
        session.record_failure(
            operation.source_id,
            operation.operation_token,
        )
        legacy_loader = Mock(side_effect=AssertionError("legacy disk read is forbidden"))
        harness = SimpleNamespace(
            _regression_session_state=session,
            _load_saved_regression_set=legacy_loader,
        )

        with self.assertRaisesRegex(RuntimeError, "Calibration failed"):
            DetectMain._regression_set_for_detection(harness)
        legacy_loader.assert_not_called()
        self.assertEqual(
            session.calibration_state, CalibrationAttemptState.FAILED_NO_ACTIVE
        )

    def test_failed_calibration_with_active_regression_preserves_old_revision(self):
        old = regression("active-old")
        session = RegressionSessionState()
        session.activate(old)
        operation = session.begin_attempt(old.source_id)
        self.assertIs(
            session.record_failure(
                operation.source_id,
                operation.operation_token,
            ),
            old,
        )
        legacy_loader = Mock(side_effect=AssertionError("legacy disk read is forbidden"))

        self.assertIs(session.regression_for_detection(legacy_loader), old)
        legacy_loader.assert_not_called()
        self.assertEqual(session.active_regression_set.revision, old.revision)
        self.assertEqual(
            session.calibration_state, CalibrationAttemptState.FAILED_WITH_ACTIVE
        )

    def test_regression_failure_handler_reports_the_preserved_revision(self):
        class ProgressBar:
            def setRange(self, *_args):
                pass

            def setValue(self, _value):
                pass

        old = regression("handler-active-old")
        session = RegressionSessionState(active_regression_set=old)
        source_path = "C:/calibration/handler-failure.png"
        operation = session.begin_attempt(
            DetectMain._regression_source_id(source_path)
        )
        messages = []
        harness = SimpleNamespace(
            _close_wait_pending=False,
            _shutdown_requested=False,
            _active_worker_task="regression",
            _calibration_source_path=source_path,
            _regression_session_state=session,
            _regression_result=None,
            _regression_dirty=False,
            ui=SimpleNamespace(progressBar=ProgressBar()),
            _restore_previous_regression_formulas=lambda *_args: None,
            _show_message_safely=lambda _function, _parent, _title, message: messages.append(message),
            _set_active_worker_task=lambda _task: None,
        )

        DetectMain._on_regression_failed(
            harness,
            {
                "source_path": source_path,
                "operation_token": operation.operation_token,
                "message": "synthetic calibration failure",
            },
        )

        self.assertIs(session.active_regression_set, old)
        self.assertEqual(
            session.calibration_state, CalibrationAttemptState.FAILED_WITH_ACTIVE
        )
        self.assertIn(old.revision, messages[0])
        self.assertIn("remains active", messages[0])

    def test_success_after_failure_unblocks_detection_and_activates_new_revision(self):
        new = regression("active-new", offset=7.0)
        session = RegressionSessionState()
        operation = session.begin_attempt("failed-calibration")
        session.record_failure(
            operation.source_id,
            operation.operation_token,
        )
        session.activate(new)
        legacy_loader = Mock(side_effect=AssertionError("legacy disk read is forbidden"))

        self.assertIs(session.regression_for_detection(legacy_loader), new)
        legacy_loader.assert_not_called()
        self.assertEqual(session.calibration_state, CalibrationAttemptState.SUCCEEDED)
        self.assertEqual(session.active_regression_set.revision, new.revision)


if __name__ == "__main__":
    unittest.main()
