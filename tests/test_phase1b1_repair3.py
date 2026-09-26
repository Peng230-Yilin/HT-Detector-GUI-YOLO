import builtins
import sys
import unittest
from pathlib import Path
from types import MappingProxyType, MethodType, SimpleNamespace
from unittest.mock import Mock, patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from batch_detection_controller import (  # noqa: E402
    BatchDetectionController,
    DetectionTask,
)
from batch_state import (  # noqa: E402
    DetectionScope,
    ImageStatus,
    NumberingMode,
    RegressionSet,
    normalize_internal_path,
)
from detectmain import DetectMain  # noqa: E402
from yolo_detection_worker import YoloDetectionWorker  # noqa: E402


IDENTITY_FIELDS = (
    "run_token",
    "job_token",
    "regression_revision",
    "image_order",
    "source_path",
    "source_file",
)


def formula(slope, intercept):
    return {
        "slope": slope,
        "intercept": intercept,
        "r": 0.9,
        "R2": 0.81,
        "p": 0.01,
        "std_err": 0.1,
    }


def regression_set(source_id="repair3-regression"):
    return RegressionSet.from_formulas(
        {
            "R": formula(2.0, 10.0),
            "G": formula(3.0, 20.0),
            "B": formula(4.0, 30.0),
        },
        source_id=source_id,
        valid_ranges=(0.0, 100.0),
    )


def authoritative_context():
    regressions = regression_set("repair3-worker-regression")
    source_path = PROJECT_ROOT / "authoritative-input" / "sample-12.png"
    controller = BatchDetectionController()
    controller.set_options(
        DetectionScope.CURRENT_IMAGE, NumberingMode.CONTINUOUS_BATCH
    )
    controller.replace_images([source_path])
    task = controller.begin(regressions)
    context, rejection = controller.prepare_dispatch_context(
        task, YoloDetectionWorker.lock_detection_context
    )
    if rejection is not None:
        raise AssertionError("The formal worker context was unexpectedly rejected.")
    return context


def identity_from_context(context):
    return {name: context[name] for name in IDENTITY_FIELDS}


def identity_from_task(task):
    return {
        "run_token": task.run_token,
        "job_token": task.job_token,
        "regression_revision": task.regression_revision,
        "image_order": task.image_order,
        "source_path": task.path,
        "source_file": task.source_file,
    }


class FakeSignal:
    def __init__(self):
        self.values = []

    def emit(self, *values):
        self.values.append(values)


class FakeProgressBar:
    def __init__(self):
        self.ranges = []
        self.values = []

    def setRange(self, minimum, maximum):
        self.ranges.append((minimum, maximum))

    def setValue(self, value):
        self.values.append(value)


def detection_controller(paths, scope=DetectionScope.CURRENT_IMAGE):
    controller = BatchDetectionController()
    controller.set_options(scope, NumberingMode.CONTINUOUS_BATCH)
    controller.replace_images(paths)
    return controller


def dispatch_harness(controller):
    harness = SimpleNamespace(
        _batch_controller=controller,
        batch_state=controller.state,
        _detection_weight_path="unused.pt",
        _detection_planned_orders=tuple(
            image.image_order for image in controller.state.images
        ),
        _detection_run_scope=controller.state.detection_scope,
        _detection_cache_state=controller.state,
        _detection_cache_run_token=controller.run_token,
        _active_worker_task=None,
        _active_worker_task_history=[],
        detection_requested=FakeSignal(),
        detection_status_changed=FakeSignal(),
        _show_message_safely=Mock(),
        ui=SimpleNamespace(
            progressBar=FakeProgressBar(),
            detectionImageSelector=None,
        ),
    )

    def set_active_worker_task(value):
        harness._active_worker_task = value
        harness._active_worker_task_history.append(value)

    harness._set_active_worker_task = set_active_worker_task
    harness._finish_detection_run = MethodType(
        DetectMain._finish_detection_run, harness
    )
    harness._advance_detection_run = MethodType(
        DetectMain._advance_detection_run, harness
    )
    harness._dispatch_detection_task = MethodType(
        DetectMain._dispatch_detection_task, harness
    )
    return harness


class TestDispatchContextRejection(unittest.TestCase):
    def _rejected_context(self, mutate):
        controller = detection_controller(["one.png"])
        task = controller.begin(regression_set())
        original = task.context()
        broken = dict(original)
        mutate(broken)

        with patch.object(DetectionTask, "context", return_value=broken):
            locked, failure = controller.prepare_dispatch_context(
                task, YoloDetectionWorker.lock_detection_context
            )

        self.assertIsNone(locked)
        self.assertEqual(
            {name: failure[name] for name in IDENTITY_FIELDS},
            identity_from_task(task),
        )
        self.assertIsNone(controller.active_job)
        self.assertEqual(controller.state.images[0].status, ImageStatus.FAILED)
        self.assertEqual(controller._next_batch_no, 1)
        return controller, task, failure

    def test_each_missing_identity_field_uses_authoritative_failure_and_clears_task(self):
        for field in IDENTITY_FIELDS:
            with self.subTest(field=field):
                controller, _task, failure = self._rejected_context(
                    lambda context, name=field: context.pop(name)
                )
                self.assertIn(field, failure)
                self.assertIn(field, controller.state.images[0].errors[0].reason)

    def test_missing_batch_start_no_closes_processing_state(self):
        controller, _task, failure = self._rejected_context(
            lambda context: context.pop("batch_start_no")
        )
        self.assertIn("batch_start_no", failure["message"])
        self.assertNotEqual(
            controller.state.images[0].status, ImageStatus.PROCESSING
        )

    def test_missing_display_start_no_closes_processing_state(self):
        controller, _task, failure = self._rejected_context(
            lambda context: context.pop("display_start_no")
        )
        self.assertIn("display_start_no", failure["message"])
        self.assertNotEqual(
            controller.state.images[0].status, ImageStatus.PROCESSING
        )

    def test_regression_revision_mismatch_closes_processing_state(self):
        controller, task, failure = self._rejected_context(
            lambda context: context.__setitem__(
                "regression_revision", "not-" + context["regression_revision"]
            )
        )
        self.assertIn("revision", failure["message"])
        self.assertEqual(failure["regression_revision"], task.regression_revision)
        self.assertNotEqual(
            controller.state.images[0].status, ImageStatus.PROCESSING
        )

    def test_invalid_dispatch_finishes_without_worker_or_dependencies(self):
        controller = detection_controller(["one.png"])
        task = controller.begin(regression_set())
        harness = dispatch_harness(controller)
        broken = task.context()
        broken.pop("run_token")
        imported_dependencies = []
        original_import = builtins.__import__

        def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name in {"cv2", "numpy"}:
                imported_dependencies.append(name)
                raise AssertionError("third-party import reached during rejection")
            return original_import(name, globals, locals, fromlist, level)

        with patch.object(DetectionTask, "context", return_value=broken), patch(
            "builtins.__import__", side_effect=guarded_import
        ), patch(
            "yolo_detection_worker.load_effective_settings"
        ) as load_settings, patch.object(
            YoloDetectionWorker, "_get_model"
        ) as get_model:
            harness._dispatch_detection_task(task)

        self.assertEqual(imported_dependencies, [])
        load_settings.assert_not_called()
        get_model.assert_not_called()
        self.assertEqual(harness.detection_requested.values, [])
        self.assertIsNone(controller.active_job)
        self.assertFalse(controller.active)
        self.assertEqual(controller.state.images[0].status, ImageStatus.FAILED)
        self.assertIsNone(harness._active_worker_task)


class TestDispatchProgression(unittest.TestCase):
    def test_rejected_first_task_advances_once_without_consuming_number(self):
        controller = detection_controller(
            ["one.png", "two.png"], DetectionScope.ALL_IMPORTED_IMAGES
        )
        first = controller.begin(regression_set())
        harness = dispatch_harness(controller)
        original_context = DetectionTask.context
        advance_calls = []
        real_advance = harness._advance_detection_run

        def context_with_first_rejected(task):
            context = original_context(task)
            if task.image_order == first.image_order:
                context.pop("source_file")
            return context

        def counted_advance():
            advance_calls.append(controller.active_job)
            real_advance()

        harness._advance_detection_run = counted_advance
        with patch.object(
            DetectionTask, "context", new=context_with_first_rejected
        ):
            harness._dispatch_detection_task(first)

        self.assertEqual(
            advance_calls, [],
            "Rejected contexts must advance inside the bounded dispatch loop.",
        )
        self.assertEqual(len(harness.detection_requested.values), 1)
        second = controller.active_job
        self.assertIsNotNone(second)
        self.assertEqual(second.image_order, 2)
        self.assertEqual(second.batch_start_no, 1)
        self.assertEqual(second.display_start_no, 1)
        self.assertNotEqual(first.job_token, second.job_token)
        self.assertEqual(controller.state.images[0].status, ImageStatus.FAILED)
        self.assertEqual(controller.state.images[1].status, ImageStatus.PROCESSING)
        self.assertEqual(harness._active_worker_task, "detection")
        dispatched_context = harness.detection_requested.values[0][2]
        self.assertEqual(
            {name: dispatched_context[name] for name in IDENTITY_FIELDS},
            identity_from_task(second),
        )

    def test_valid_dispatch_emits_once_with_one_immutable_authoritative_context(self):
        controller = detection_controller(["one.png"])
        task = controller.begin(regression_set())
        harness = dispatch_harness(controller)

        harness._dispatch_detection_task(task)

        self.assertEqual(len(harness.detection_requested.values), 1)
        image_path, weight_path, context = harness.detection_requested.values[0]
        self.assertEqual(image_path, task.path)
        self.assertEqual(weight_path, "unused.pt")
        self.assertEqual(
            {name: context[name] for name in IDENTITY_FIELDS},
            identity_from_task(task),
        )
        self.assertIs(context["regression_set"], task.regression_set)
        self.assertEqual(context["regression_revision"], task.regression_revision)
        self.assertEqual(context["batch_start_no"], task.batch_start_no)
        self.assertEqual(context["display_start_no"], task.display_start_no)
        with self.assertRaises(TypeError):
            context["image_order"] = 999


class TestRepair4DispatchLoop(unittest.TestCase):
    def _assert_all_rejected(self, image_count):
        paths = [
            PROJECT_ROOT / "repair4-invalid" / "image-{:05d}.png".format(index)
            for index in range(1, image_count + 1)
        ]
        controller = detection_controller(
            paths, DetectionScope.ALL_IMPORTED_IMAGES
        )
        first = controller.begin(regression_set("repair4-all-rejected"))
        harness = dispatch_harness(controller)
        original_context = DetectionTask.context

        def rejected_context(task):
            context = original_context(task)
            context.pop("source_file")
            return context

        with patch.object(
            DetectionTask, "context", new=rejected_context
        ):
            harness._dispatch_detection_task(first)

        counts = {
            status: sum(image.status == status for image in controller.state.images)
            for status in ImageStatus
        }
        self.assertEqual(counts[ImageStatus.PENDING], 0)
        self.assertEqual(counts[ImageStatus.PROCESSING], 0)
        self.assertEqual(counts[ImageStatus.COMPLETED], 0)
        self.assertEqual(counts[ImageStatus.FAILED], image_count)
        self.assertEqual(harness.detection_requested.values, [])
        self.assertIsNone(controller.active_job)
        self.assertFalse(controller.active)
        self.assertIsNone(harness._active_worker_task)
        self.assertNotIn("detection", harness._active_worker_task_history)
        self.assertEqual(len(harness.detection_status_changed.values), 1)
        self.assertEqual(controller._next_batch_no, 1)
        self.assertTrue(all(not image.samples for image in controller.state.images))

    def test_1500_consecutive_rejections_finish_with_bounded_stack(self):
        self._assert_all_rejected(1500)

    def test_5000_consecutive_rejections_finish_with_bounded_stack(self):
        self._assert_all_rejected(5000)

    def test_1500_rejections_then_one_valid_task_emits_once_and_stops(self):
        paths = [
            PROJECT_ROOT / "repair4-mixed" / "image-{:05d}.png".format(index)
            for index in range(1, 1503)
        ]
        controller = detection_controller(
            paths, DetectionScope.ALL_IMPORTED_IMAGES
        )
        first = controller.begin(regression_set("repair4-mixed"))
        harness = dispatch_harness(controller)
        original_context = DetectionTask.context

        def first_1500_rejected(task):
            context = original_context(task)
            if task.image_order <= 1500:
                context.pop("source_file")
            return context

        with patch.object(
            DetectionTask, "context", new=first_1500_rejected
        ):
            harness._dispatch_detection_task(first)

        self.assertEqual(
            sum(image.status == ImageStatus.FAILED for image in controller.state.images),
            1500,
        )
        self.assertEqual(
            sum(image.status == ImageStatus.PROCESSING for image in controller.state.images),
            1,
        )
        self.assertEqual(
            sum(image.status == ImageStatus.PENDING for image in controller.state.images),
            1,
        )
        self.assertEqual(len(harness.detection_requested.values), 1)
        emitted = harness.detection_requested.values[0][2]
        active = controller.active_job
        self.assertIsNotNone(active)
        self.assertEqual(active.image_order, 1501)
        self.assertEqual(emitted["image_order"], 1501)
        self.assertEqual(emitted["job_token"], active.job_token)
        self.assertEqual(active.batch_start_no, 1)
        self.assertEqual(active.display_start_no, 1)
        self.assertEqual(controller._next_batch_no, 1)
        self.assertEqual(harness._active_worker_task, "detection")


class TestRepair4ContextContract(unittest.TestCase):
    def test_one_locked_context_object_reaches_signal_and_worker_validation(self):
        controller = detection_controller([
            PROJECT_ROOT / "repair4-context" / "identity.png"
        ])
        task = controller.begin(regression_set("repair4-context"))
        harness = dispatch_harness(controller)
        raw_values = []
        locked_values = []
        controller_returned_values = []
        real_lock = YoloDetectionWorker.lock_detection_context
        real_prepare = controller.prepare_dispatch_context

        def recording_lock(image_path, context):
            raw_values.append(context)
            locked = real_lock(image_path, context)
            locked_values.append(locked)
            return locked

        def recording_prepare(task, validator):
            result = real_prepare(task, validator)
            controller_returned_values.append(result[0])
            return result

        with patch.object(
            controller,
            "prepare_dispatch_context",
            side_effect=recording_prepare,
        ), patch.object(
            YoloDetectionWorker, "lock_detection_context", side_effect=recording_lock
        ):
            harness._dispatch_detection_task(task)

        self.assertEqual(len(locked_values), 1)
        self.assertEqual(len(raw_values), 1)
        self.assertEqual(len(controller_returned_values), 1)
        self.assertEqual(len(harness.detection_requested.values), 1)
        locked_context = locked_values[0]
        controller_returned_context = controller_returned_values[0]
        emitted_context = harness.detection_requested.values[0][2]
        worker_validated_context = YoloDetectionWorker._validated_detection_context(
            task.path, emitted_context
        )
        self.assertIsInstance(locked_context, MappingProxyType)
        self.assertIs(controller_returned_context, locked_context)
        self.assertIs(emitted_context, controller_returned_context)
        self.assertIs(worker_validated_context, emitted_context)
        self.assertIs(
            emitted_context["regression_set"], task.regression_set
        )
        self.assertEqual(
            emitted_context["regression_revision"], task.regression_revision
        )
        raw_values[0]["job_token"] = 999
        self.assertEqual(emitted_context["job_token"], task.job_token)
        with self.assertRaises(TypeError):
            emitted_context["job_token"] = 999

    def test_stale_task_run_and_job_do_not_change_new_active_job(self):
        controller = detection_controller([
            PROJECT_ROOT / "repair4-stale" / "one.png"
        ])
        old_task = controller.begin(regression_set("repair4-stale-old"))
        old_failure = identity_from_task(old_task)
        old_failure["message"] = "finish old run"
        self.assertTrue(controller.accept_failure(old_failure))
        self.assertIsNotNone(controller.finish_if_done())

        new_task = controller.begin(regression_set("repair4-stale-new"))
        self.assertIsNotNone(new_task)
        self.assertIs(controller.active_job, new_task)
        self.assertEqual(
            controller.state.images[0].status, ImageStatus.PROCESSING
        )

        with self.assertRaises(ValueError):
            controller.prepare_dispatch_context(
                old_task, YoloDetectionWorker.lock_detection_context
            )
        self.assertIsNone(controller.reject_dispatch(old_task, ValueError("stale")))

        for field, value in (
            ("run_token", old_task.run_token),
            ("job_token", old_task.job_token),
        ):
            with self.subTest(field=field):
                stale_failure = identity_from_task(new_task)
                stale_failure[field] = value
                stale_failure["message"] = "stale callback"
                self.assertFalse(controller.accept_failure(stale_failure))
                self.assertIs(controller.active_job, new_task)
                self.assertEqual(
                    controller.state.images[0].status, ImageStatus.PROCESSING
                )
                self.assertEqual(controller._next_batch_no, 1)

    def test_invalid_job_token_values_never_change_active_job(self):
        controller = detection_controller([
            PROJECT_ROOT / "repair4-token" / "one.png"
        ])
        task = controller.begin(regression_set("repair4-token"))

        for invalid in ("1", 0, -1, True):
            with self.subTest(job_token=invalid):
                invalid_context = task.context()
                invalid_context["job_token"] = invalid
                with self.assertRaises(ValueError):
                    YoloDetectionWorker.lock_detection_context(
                        task.path, invalid_context
                    )

                invalid_failure = identity_from_task(task)
                invalid_failure["job_token"] = invalid
                invalid_failure["message"] = "invalid job token"
                self.assertFalse(controller.accept_failure(invalid_failure))
                self.assertIs(controller.active_job, task)
                self.assertEqual(
                    controller.state.images[0].status, ImageStatus.PROCESSING
                )
                self.assertEqual(controller._next_batch_no, 1)

    def test_formal_context_uses_absolute_normalized_path_and_exact_filename(self):
        raw_path = (
            PROJECT_ROOT
            / "repair4-relative"
            / "source files"
            / "样本 =1.png"
        )
        controller = detection_controller([raw_path])
        task = controller.begin(regression_set("repair4-path"))
        context, rejection = controller.prepare_dispatch_context(
            task, YoloDetectionWorker.lock_detection_context
        )

        self.assertIsNone(rejection)
        self.assertEqual(task.path, normalize_internal_path(raw_path))
        self.assertEqual(context["source_path"], task.path)
        self.assertEqual(context["source_file"], "样本 =1.png")
        self.assertEqual(context["source_file"], task.source_file)
        self.assertTrue(Path(context["source_path"]).is_absolute())
        self.assertIs(
            YoloDetectionWorker._validated_detection_context(task.path, context),
            context,
        )


class TestWorkerTerminalPaths(unittest.TestCase):
    def _worker_callbacks(self):
        worker = YoloDetectionWorker()
        successes = []
        failures = []
        worker.finished.connect(successes.append)
        worker.failed.connect(failures.append)
        return worker, successes, failures

    def _assert_one_failure(self, context, successes, failures, text):
        self.assertEqual(successes, [])
        self.assertEqual(len(failures), 1)
        self.assertEqual(
            {name: failures[0][name] for name in IDENTITY_FIELDS},
            identity_from_context(context),
        )
        self.assertIn(text, failures[0]["message"])

    def test_cv2_import_failure_keeps_identity_and_is_failure_only(self):
        context = authoritative_context()
        worker, successes, failures = self._worker_callbacks()
        original_import = builtins.__import__

        def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "cv2":
                raise ImportError("repair3 cv2 failure")
            return original_import(name, globals, locals, fromlist, level)

        with patch("builtins.__import__", side_effect=guarded_import):
            worker.detect(context["source_path"], "unused.pt", context)

        self._assert_one_failure(
            context, successes, failures, "repair3 cv2 failure"
        )

    def test_numpy_import_failure_keeps_identity_and_is_failure_only(self):
        context = authoritative_context()
        worker, successes, failures = self._worker_callbacks()
        original_import = builtins.__import__

        def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "numpy":
                raise ImportError("repair3 numpy failure")
            return original_import(name, globals, locals, fromlist, level)

        with patch.dict(sys.modules, {"cv2": SimpleNamespace()}), patch(
            "builtins.__import__", side_effect=guarded_import
        ):
            worker.detect(context["source_path"], "unused.pt", context)

        self._assert_one_failure(
            context, successes, failures, "repair3 numpy failure"
        )

    def test_settings_failure_keeps_identity_and_is_failure_only(self):
        context = authoritative_context()
        worker, successes, failures = self._worker_callbacks()

        with patch.dict(
            sys.modules, {"cv2": SimpleNamespace(), "numpy": SimpleNamespace()}
        ), patch(
            "yolo_detection_worker.load_effective_settings",
            side_effect=RuntimeError("repair3 settings failure"),
        ):
            worker.detect(context["source_path"], "unused.pt", context)

        self._assert_one_failure(
            context, successes, failures, "repair3 settings failure"
        )

    def test_success_keeps_identity_and_never_emits_failure(self):
        context = authoritative_context()
        worker, successes, failures = self._worker_callbacks()
        encoded_path = object()
        source_image = object()
        fake_numpy = SimpleNamespace(
            uint8=object(), fromfile=Mock(return_value=encoded_path)
        )
        fake_cv2 = SimpleNamespace(
            IMREAD_COLOR=object(), imdecode=Mock(return_value=source_image)
        )
        model = Mock()
        model.predict.return_value = [object()]

        with patch.dict(
            sys.modules, {"cv2": fake_cv2, "numpy": fake_numpy}
        ), patch(
            "yolo_detection_worker.load_effective_settings",
            return_value=({"detect_confidence": 0.5}, [], None),
        ), patch.object(
            worker, "_get_model", return_value=model
        ), patch.object(
            worker,
            "_build_payload",
            return_value={"sample_results": [], "sample_errors": []},
        ):
            worker.detect(context["source_path"], "unused.pt", context)

        self.assertEqual(failures, [])
        self.assertEqual(len(successes), 1)
        self.assertEqual(
            {name: successes[0][name] for name in IDENTITY_FIELDS},
            identity_from_context(context),
        )


if __name__ == "__main__":
    unittest.main()
