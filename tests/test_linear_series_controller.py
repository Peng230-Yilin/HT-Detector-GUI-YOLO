import sys
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GUI_ROOT = PROJECT_ROOT / "Peng1.0_GUI"
sys.path.insert(0, str(GUI_ROOT))

from linear_series_controller import (  # noqa: E402
    LINEAR_SERIES_OPERATION,
    LinearSeriesController,
    LinearSeriesTask,
)
from linear_series_state import (  # noqa: E402
    LinearImageStatus,
    LinearSeriesPhase,
    LinearSeriesState,
)


def rgb_sample(red=1.0, green=2.0, blue=3.0):
    return {"red": red, "green": green, "blue": blue}


def success_payload(task, samples=(), errors=()):
    payload = task.context()
    payload.update({"samples": list(samples), "errors": list(errors)})
    return payload


def failure_payload(task, reason="image failed", errors=()):
    payload = task.context()
    payload.update({"reason": reason, "errors": list(errors)})
    return payload


def task_snapshot(task):
    if task is None:
        return None
    return (
        id(task),
        task.operation,
        task.run_token,
        task.job_token,
        task.image_order,
        task.normalized_path,
        task.original_file_name,
    )


def sample_snapshot(sample):
    return (
        sample.sample_key,
        sample.image_order,
        sample.normalized_path,
        sample.original_file_name,
        sample.spatial_order,
        sample.red,
        sample.green,
        sample.blue,
        sample.series_number,
        sample.included,
        sample.concentration_text,
        sample.status.value,
        sample.cuvette_box,
        sample.liquid_box,
        sample.roi_box,
    )


def error_snapshot(error):
    return (
        error.error_key,
        error.image_order,
        error.normalized_path,
        error.original_file_name,
        error.spatial_order,
        error.error_type,
        error.reason,
        error.series_number,
    )


def image_snapshot(image):
    return (
        image.image_key,
        image.image_order,
        image.selection_order,
        image.normalized_path,
        image.original_file_name,
        image.status.value,
        tuple(sample_snapshot(sample) for sample in image.samples),
        tuple(error_snapshot(error) for error in image.errors),
        image.failure_reason,
    )


def mutable_snapshot(controller):
    """Return a recursively immutable, value-only controller snapshot."""

    return (
        controller.run_token,
        controller._run_token_counter,
        controller._job_token_counter,
        task_snapshot(controller.active_job),
        tuple(task_snapshot(task) for task in controller._run_tasks),
        tuple(task_snapshot(task) for task in controller._queue),
        controller.queued_image_orders,
        id(controller.state),
        controller.state.phase.value,
        controller.busy,
        tuple(image_snapshot(image) for image in controller.state.images),
        id(controller.last_confirmed_result),
    )


class LinearSeriesTaskTests(unittest.TestCase):
    def test_task_has_complete_immutable_worker_identity(self):
        controller = LinearSeriesController()
        task = controller.begin([str(Path("folder") / "Image01.PNG")])

        self.assertIsInstance(task, LinearSeriesTask)
        self.assertEqual(task.operation, LINEAR_SERIES_OPERATION)
        self.assertEqual(set(task.context()), {
            "operation", "run_token", "job_token", "image_order",
            "normalized_path", "original_file_name",
        })
        self.assertEqual(task.original_file_name, "Image01.PNG")
        self.assertEqual(task.path, task.normalized_path)
        self.assertEqual(task.source_file, task.original_file_name)
        with self.assertRaises((AttributeError, TypeError)):
            task.job_token = 100

    def test_task_constructor_rejects_bool_tokens_and_invalid_identity_fields(self):
        valid = {
            "operation": LINEAR_SERIES_OPERATION,
            "run_token": 1,
            "job_token": 2,
            "image_order": 3,
            "normalized_path": "C:/images/sample.png",
            "original_file_name": "sample.png",
        }
        invalid_values = {
            "operation": (None, "other_operation"),
            "run_token": (True, "1", 0, -1),
            "job_token": (False, "2", 0, -1),
            "image_order": (True, "3", 0, -1),
            "normalized_path": (Path("sample.png"), ""),
            "original_file_name": (b"sample.png", ""),
        }
        for field, values in invalid_values.items():
            for value in values:
                with self.subTest(field=field, value=value):
                    arguments = dict(valid)
                    arguments[field] = value
                    with self.assertRaises(ValueError):
                        LinearSeriesTask(**arguments)


class LinearSeriesControllerTests(unittest.TestCase):
    def setUp(self):
        self.controller = LinearSeriesController()

    def test_begin_dispatches_natural_order_strictly_one_at_a_time(self):
        first = self.controller.begin(["image10.png", "image2.png", "image1.png"])
        self.assertEqual(first.original_file_name, "image1.png")
        self.assertEqual(self.controller.queued_image_orders, (2, 3))
        active_snapshot = mutable_snapshot(self.controller)
        self.assertIsNone(self.controller.next_task())
        self.assertEqual(mutable_snapshot(self.controller), active_snapshot)
        self.assertIs(self.controller.active_job, first)

        self.assertTrue(self.controller.accept_success(success_payload(first)))
        second = self.controller.next_task()
        self.assertEqual(second.original_file_name, "image2.png")
        self.assertEqual(second.run_token, first.run_token)
        self.assertNotEqual(second.job_token, first.job_token)
        self.assertIsNone(self.controller.next_task())

        self.assertTrue(self.controller.accept_success(success_payload(second)))
        third = self.controller.next_task()
        self.assertEqual(third.original_file_name, "image10.png")

    def test_duplicate_path_is_two_jobs_with_distinct_job_tokens(self):
        first = self.controller.begin(["same.png", "same.png"])
        self.assertTrue(self.controller.accept_success(success_payload(first)))
        second = self.controller.next_task()

        self.assertEqual(first.normalized_path, second.normalized_path)
        self.assertEqual(first.original_file_name, second.original_file_name)
        self.assertEqual(first.run_token, second.run_token)
        self.assertNotEqual(first.job_token, second.job_token)
        self.assertNotEqual(first.image_order, second.image_order)

    def test_wrong_operation_run_job_and_source_identity_are_side_effect_free(self):
        task = self.controller.begin(["sample.png"])
        baseline = mutable_snapshot(self.controller)
        changes = {
            "operation": "detection",
            "run_token": task.run_token + 1,
            "job_token": task.job_token + 1,
            "image_order": task.image_order + 1,
            "normalized_path": task.normalized_path + ".other",
            "original_file_name": "renamed.png",
        }
        for field, wrong_value in changes.items():
            with self.subTest(field=field):
                success = success_payload(task)
                success[field] = wrong_value
                failure = failure_payload(task)
                failure[field] = wrong_value
                self.assertFalse(self.controller.matches_active_result(success))
                self.assertFalse(self.controller.matches_active_result(failure))
                self.assertFalse(self.controller.accept_success(success))
                self.assertFalse(self.controller.accept_failure(failure))
                self.assertEqual(mutable_snapshot(self.controller), baseline)

    def test_wrong_identity_types_and_bool_tokens_are_side_effect_free(self):
        task = self.controller.begin(["sample.png"])
        baseline = mutable_snapshot(self.controller)
        changes = {
            "operation": (b"linear_series_extract",),
            "run_token": (True, str(task.run_token)),
            "job_token": (True, str(task.job_token)),
            "image_order": (True, str(task.image_order)),
            "normalized_path": (Path(task.normalized_path),),
            "original_file_name": (task.original_file_name.encode("utf-8"),),
        }
        for field, wrong_values in changes.items():
            for wrong_value in wrong_values:
                with self.subTest(field=field, wrong_value=wrong_value):
                    success = success_payload(task)
                    success[field] = wrong_value
                    failure = failure_payload(task)
                    failure[field] = wrong_value
                    self.assertFalse(
                        self.controller.matches_active_result(success)
                    )
                    self.assertFalse(
                        self.controller.matches_active_result(failure)
                    )
                    self.assertFalse(self.controller.accept_success(success))
                    self.assertFalse(self.controller.accept_failure(failure))
                    self.assertEqual(
                        mutable_snapshot(self.controller), baseline
                    )

        self.assertTrue(
            self.controller.accept_success(
                success_payload(task, [rgb_sample()])
            )
        )

    def test_missing_identity_is_never_accepted_by_success_or_failure(self):
        task = self.controller.begin(["sample.png"])
        for field in task.context():
            with self.subTest(field=field):
                success = success_payload(task)
                del success[field]
                failure = failure_payload(task)
                del failure[field]
                baseline = mutable_snapshot(self.controller)
                self.assertFalse(self.controller.accept_success(success))
                self.assertFalse(self.controller.accept_failure(failure))
                self.assertEqual(mutable_snapshot(self.controller), baseline)

    def test_same_run_previous_job_late_signals_do_not_touch_active_job(self):
        first = self.controller.begin(["a.png", "b.png"])
        first_success = success_payload(first, [rgb_sample()])
        self.assertTrue(self.controller.accept_success(first_success))
        second = self.controller.next_task()
        baseline = mutable_snapshot(self.controller)

        self.assertFalse(self.controller.accept_success(first_success))
        self.assertFalse(self.controller.accept_failure(failure_payload(first, "late")))
        self.assertEqual(mutable_snapshot(self.controller), baseline)
        self.assertIs(self.controller.active_job, second)
        self.assertEqual(
            self.controller.state.image_for_order(2).status,
            LinearImageStatus.PROCESSING,
        )

    def test_previous_job_and_previous_run_result_matrix_is_side_effect_free(self):
        first = self.controller.begin(["a.png", "b.png"])
        first_success = success_payload(first, [rgb_sample()])
        self.assertTrue(self.controller.accept_success(first_success))
        second = self.controller.next_task()
        second_baseline = mutable_snapshot(self.controller)

        self.assertFalse(self.controller.accept_success(first_success))
        self.assertFalse(
            self.controller.accept_failure(failure_payload(first, "late"))
        )
        self.assertEqual(mutable_snapshot(self.controller), second_baseline)

        replacement = self.controller.begin(["replacement.png"])
        replacement_baseline = mutable_snapshot(self.controller)
        self.assertFalse(
            self.controller.accept_success(
                success_payload(second, [rgb_sample()])
            )
        )
        self.assertFalse(
            self.controller.accept_failure(
                failure_payload(second, "late previous run")
            )
        )
        self.assertEqual(
            mutable_snapshot(self.controller), replacement_baseline
        )
        self.assertIs(self.controller.active_job, replacement)

    def test_duplicate_success_and_success_then_failure_are_rejected(self):
        task = self.controller.begin(["sample.png"])
        accepted = success_payload(task, [rgb_sample()])
        self.assertTrue(self.controller.accept_success(accepted))
        baseline = mutable_snapshot(self.controller)

        self.assertFalse(self.controller.accept_success(accepted))
        self.assertFalse(self.controller.accept_failure(failure_payload(task, "late")))
        self.assertEqual(mutable_snapshot(self.controller), baseline)
        self.assertEqual(len(self.controller.state.images[0].samples), 1)

    def test_duplicate_failure_and_failure_then_success_are_rejected(self):
        task = self.controller.begin(["sample.png"])
        failed = failure_payload(task, "decoder failed")
        self.assertTrue(self.controller.accept_failure(failed))
        baseline = mutable_snapshot(self.controller)

        self.assertFalse(self.controller.accept_failure(failed))
        self.assertFalse(self.controller.accept_success(success_payload(task)))
        self.assertEqual(mutable_snapshot(self.controller), baseline)
        self.assertEqual(
            self.controller.state.images[0].status, LinearImageStatus.FAILED
        )

    def test_malformed_current_payload_does_not_consume_active_job(self):
        task = self.controller.begin(["sample.png"])
        baseline = mutable_snapshot(self.controller)

        no_samples = task.context()
        non_collection = success_payload(task)
        non_collection["samples"] = "not a collection"
        no_reason = task.context()
        for payload, accept in (
            (no_samples, self.controller.accept_success),
            (non_collection, self.controller.accept_success),
            (no_reason, self.controller.accept_failure),
        ):
            with self.subTest(payload=payload):
                self.assertFalse(accept(payload))
                self.assertEqual(mutable_snapshot(self.controller), baseline)

    def test_unordered_and_one_shot_payload_collections_are_rejected(self):
        task = self.controller.begin(["sample.png"])
        baseline = mutable_snapshot(self.controller)
        invalid_collections = (
            {"unordered"},
            frozenset(("unordered",)),
            {"red": 1.0, "green": 2.0, "blue": 3.0},
        )
        for field in ("samples", "errors"):
            for values in invalid_collections:
                with self.subTest(field=field, collection=type(values).__name__):
                    payload = success_payload(task, [rgb_sample()])
                    payload[field] = values
                    self.assertFalse(self.controller.accept_success(payload))
                    self.assertEqual(
                        mutable_snapshot(self.controller), baseline
                    )

            iterations = []

            def one_shot():
                iterations.append(field)
                yield rgb_sample()

            payload = success_payload(task, [rgb_sample()])
            payload[field] = one_shot()
            self.assertFalse(self.controller.accept_success(payload))
            self.assertEqual(iterations, [])
            self.assertEqual(mutable_snapshot(self.controller), baseline)

        tuple_payload = task.context()
        tuple_payload.update({"samples": (rgb_sample(),), "errors": ()})
        self.assertTrue(self.controller.accept_success(tuple_payload))

    def test_success_validation_exception_is_atomic_and_retryable(self):
        class ExplodingSample:
            red = 4.0
            blue = 6.0

            @property
            def green(self):
                raise RuntimeError("sample validation exploded")

        task = self.controller.begin(["sample.png"])
        baseline = mutable_snapshot(self.controller)
        payload = success_payload(
            task,
            [rgb_sample(), ExplodingSample()],
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "sample validation exploded",
        ):
            self.controller.accept_success(payload)

        self.assertEqual(mutable_snapshot(self.controller), baseline)
        self.assertIs(self.controller.active_job, task)
        self.assertTrue(
            self.controller.accept_success(
                success_payload(task, [rgb_sample()])
            )
        )

    def test_failure_validation_exception_is_atomic_and_retryable(self):
        class ExplodingError:
            @property
            def reason(self):
                raise RuntimeError("error validation exploded")

        task = self.controller.begin(["sample.png"])
        baseline = mutable_snapshot(self.controller)
        payload = failure_payload(
            task,
            reason="decoder failed",
            errors=[ExplodingError()],
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "error validation exploded",
        ):
            self.controller.accept_failure(payload)

        self.assertEqual(mutable_snapshot(self.controller), baseline)
        self.assertIs(self.controller.active_job, task)
        self.assertTrue(
            self.controller.accept_failure(
                failure_payload(task, "decoder failed")
            )
        )

    def test_cancel_invalidates_run_without_needing_worker_termination(self):
        marker = {"confirmed": True}
        self.controller = LinearSeriesController(
            LinearSeriesState(last_confirmed_result=marker)
        )
        task = self.controller.begin(["sample.png"])

        self.assertTrue(self.controller.cancel())
        self.assertIsNone(self.controller.run_token)
        self.assertIsNone(self.controller.active_job)
        self.assertEqual(self.controller.queued_image_orders, ())
        self.assertFalse(self.controller.busy)
        self.assertEqual(self.controller.state.phase, LinearSeriesPhase.CANCELLED)
        self.assertIs(self.controller.last_confirmed_result, marker)

        baseline = mutable_snapshot(self.controller)
        self.assertFalse(self.controller.accept_success(success_payload(task)))
        self.assertFalse(self.controller.accept_failure(failure_payload(task)))
        self.assertEqual(mutable_snapshot(self.controller), baseline)

    def test_cancel_after_extraction_finish_discards_mapping_draft(self):
        marker = object()
        self.controller = LinearSeriesController(
            LinearSeriesState(last_confirmed_result=marker)
        )
        task = self.controller.begin(["sample.png"])
        self.assertTrue(self.controller.accept_success(
            success_payload(task, [rgb_sample()])
        ))
        self.assertEqual(self.controller.state.phase, LinearSeriesPhase.MAPPING)
        self.assertIsNotNone(self.controller.finish_if_done())
        self.assertIsNone(self.controller.run_token)

        self.assertTrue(self.controller.cancel())
        self.assertEqual(self.controller.state.phase, LinearSeriesPhase.CANCELLED)
        self.assertEqual(self.controller.state.images[0].samples, ())
        self.assertEqual(self.controller.state.images[0].errors, ())
        self.assertIsNone(self.controller.state.images[0].failure_reason)
        self.assertEqual(
            self.controller.state.images[0].status,
            LinearImageStatus.CANCELLED,
        )
        self.assertIs(self.controller.last_confirmed_result, marker)
        with self.assertRaisesRegex(RuntimeError, "(?i)cancelled|stale"):
            self.controller.state.confirm(object())
        self.assertFalse(self.controller.cancel())

    def test_failed_series_has_failed_phase_and_preserves_confirmed_reference(self):
        marker = object()
        self.controller = LinearSeriesController(
            LinearSeriesState(last_confirmed_result=marker)
        )
        task = self.controller.begin(["broken.png"])

        self.assertTrue(self.controller.accept_failure(
            failure_payload(task, "decode failed")
        ))
        self.assertEqual(self.controller.state.phase, LinearSeriesPhase.FAILED)
        self.assertIs(self.controller.last_confirmed_result, marker)
        self.assertTrue(self.controller.busy)
        self.assertIsNotNone(self.controller.finish_if_done())
        self.assertFalse(self.controller.busy)
        self.assertIsNone(self.controller.run_token)

    def test_repeated_begin_invalidates_old_run_and_replaces_all_draft_data(self):
        marker = object()
        self.controller = LinearSeriesController(
            LinearSeriesState(last_confirmed_result=marker)
        )
        old_task = self.controller.begin(["old2.png", "old1.png"])
        old_run = old_task.run_token
        old_state = self.controller.state

        new_task = self.controller.begin(["new.png"])
        self.assertNotEqual(new_task.run_token, old_run)
        self.assertEqual([image.original_file_name for image in self.controller.state.images], [
            "new.png"
        ])
        self.assertTrue(all(not image.samples and not image.errors
                            for image in self.controller.state.images))
        self.assertIs(self.controller.last_confirmed_result, marker)
        self.assertEqual(old_state.phase, LinearSeriesPhase.CANCELLED)

        baseline = mutable_snapshot(self.controller)
        self.assertFalse(self.controller.accept_success(success_payload(old_task)))
        self.assertFalse(self.controller.accept_failure(failure_payload(old_task)))
        self.assertEqual(mutable_snapshot(self.controller), baseline)
        self.assertIs(self.controller.active_job, new_task)

    def test_begin_failure_is_atomic_and_old_active_job_remains_retryable(self):
        class ExplodingPath:
            def __fspath__(self):
                raise RuntimeError("path conversion exploded")

        old_task = self.controller.begin(["old.png"])
        old_state = self.controller.state
        baseline = mutable_snapshot(self.controller)

        with self.assertRaisesRegex(RuntimeError, "path conversion exploded"):
            self.controller.begin([ExplodingPath()])

        self.assertIs(self.controller.state, old_state)
        self.assertEqual(mutable_snapshot(self.controller), baseline)
        self.assertIs(self.controller.active_job, old_task)
        self.assertTrue(
            self.controller.accept_success(
                success_payload(old_task, [rgb_sample()])
            )
        )

    def test_finish_then_begin_cancels_old_mapping_reference_as_stale(self):
        marker = object()
        self.controller = LinearSeriesController(
            LinearSeriesState(last_confirmed_result=marker)
        )
        old_task = self.controller.begin(["old.png"])
        self.assertTrue(
            self.controller.accept_success(
                success_payload(old_task, [rgb_sample()])
            )
        )
        old_mapping_state = self.controller.state
        self.assertIsNotNone(self.controller.finish_if_done())
        self.assertIsNone(self.controller.run_token)
        self.assertEqual(
            old_mapping_state.phase,
            LinearSeriesPhase.MAPPING,
        )

        new_task = self.controller.begin(["new.png"])

        self.assertIsNotNone(new_task)
        self.assertEqual(
            old_mapping_state.phase,
            LinearSeriesPhase.CANCELLED,
        )
        self.assertTrue(
            all(
                image.status == LinearImageStatus.CANCELLED
                for image in old_mapping_state.images
            )
        )
        with self.assertRaisesRegex(RuntimeError, "(?i)cancelled|stale"):
            old_mapping_state.confirm(object())
        self.assertIs(self.controller.last_confirmed_result, marker)

    def test_task_queue_is_an_immutable_snapshot_of_selected_paths(self):
        selected_paths = ["image10.png", "image2.png", "image1.png"]
        first = self.controller.begin(selected_paths)
        queue_view = self.controller.queued_image_orders

        self.assertIsInstance(self.controller._run_tasks, tuple)
        self.assertIsInstance(self.controller._queue, tuple)
        self.assertTrue(
            all(
                isinstance(task, LinearSeriesTask)
                for task in self.controller._run_tasks
            )
        )
        self.assertEqual(queue_view, (2, 3))
        with self.assertRaises(AttributeError):
            queue_view.append(4)
        with self.assertRaises(TypeError):
            self.controller._queue[0] = first

        selected_paths.clear()
        queue_view += (999,)
        self.assertEqual(self.controller.queued_image_orders, (2, 3))
        self.assertEqual(
            tuple(
                image.original_file_name
                for image in self.controller.state.images
            ),
            ("image1.png", "image2.png", "image10.png"),
        )

    def test_repeated_begin_same_paths_clears_completed_draft_samples(self):
        first = self.controller.begin(["sample.png"])
        self.assertTrue(self.controller.accept_success(
            success_payload(first, [rgb_sample()])
        ))
        self.assertEqual(len(self.controller.state.images[0].samples), 1)

        second = self.controller.begin()
        self.assertNotEqual(second.run_token, first.run_token)
        self.assertEqual(self.controller.state.images[0].samples, ())
        self.assertEqual(self.controller.state.images[0].errors, ())
        self.assertEqual(
            self.controller.state.images[0].status, LinearImageStatus.PROCESSING
        )

    def test_empty_begin_invalidates_active_run_and_preserves_confirmed_reference(self):
        marker = object()
        self.controller = LinearSeriesController(
            LinearSeriesState(last_confirmed_result=marker)
        )
        old_task = self.controller.begin(["old.png"])

        self.assertIsNone(self.controller.begin([]))
        self.assertEqual(self.controller.state.images, ())
        self.assertFalse(self.controller.busy)
        self.assertIsNone(self.controller.run_token)
        self.assertIs(self.controller.last_confirmed_result, marker)
        self.assertFalse(self.controller.accept_success(success_payload(old_task)))

    def test_busy_and_summary_follow_queue_and_terminal_states(self):
        first = self.controller.begin(["a.png", "b.png"])
        self.assertTrue(self.controller.busy)
        self.assertTrue(self.controller.accept_failure(
            failure_payload(first, "bad image")
        ))
        self.assertTrue(self.controller.busy)

        second = self.controller.next_task()
        self.assertTrue(self.controller.busy)
        self.assertTrue(self.controller.accept_success(
            success_payload(second, [rgb_sample(), rgb_sample(4, 5, 6)])
        ))
        self.assertTrue(self.controller.busy)
        self.assertEqual(self.controller.state.phase, LinearSeriesPhase.MAPPING)
        self.assertEqual(self.controller.finish_if_done(), {
            "total_images": 2,
            "successful_images": 1,
            "failed_images": 1,
            "valid_samples": 2,
            "sample_errors": 1,
        })
        self.assertIsNone(self.controller.run_token)
        self.assertFalse(self.controller.busy)
        self.assertIsNone(self.controller.finish_if_done())

    def test_finish_rejects_pending_image_desynchronization_without_retiring_run(self):
        marker = object()
        self.controller = LinearSeriesController(
            LinearSeriesState(last_confirmed_result=marker)
        )
        first = self.controller.begin(["a.png", "b.png"])
        run_token = self.controller.run_token

        # Simulate an internal queue bookkeeping defect.  The immutable state
        # still records the second original task as pending.
        self.controller._queue = ()
        self.assertTrue(
            self.controller.accept_success(
                success_payload(first, [rgb_sample()])
            )
        )
        self.assertTrue(self.controller.busy)
        self.assertTrue(self.controller.state.busy)
        self.assertEqual(self.controller.run_token, run_token)
        self.assertEqual(
            self.controller.state.phase,
            LinearSeriesPhase.EXTRACTING,
        )
        self.assertEqual(
            self.controller.state.image_for_order(2).status,
            LinearImageStatus.PENDING,
        )
        active_before = self.controller.active_job
        queue_before = self.controller._queue
        state_before = self.controller.state
        confirmed_before = self.controller.last_confirmed_result
        baseline = mutable_snapshot(self.controller)

        self.assertIsNone(self.controller.finish_if_done())
        self.assertTrue(self.controller.busy)
        self.assertTrue(self.controller.state.busy)
        self.assertEqual(self.controller.run_token, run_token)
        self.assertEqual(
            self.controller.state.phase,
            LinearSeriesPhase.EXTRACTING,
        )
        self.assertEqual(
            self.controller.state.image_for_order(2).status,
            LinearImageStatus.PENDING,
        )
        self.assertIs(self.controller.active_job, active_before)
        self.assertIs(self.controller._queue, queue_before)
        self.assertIs(self.controller.state, state_before)
        self.assertIs(self.controller.last_confirmed_result, confirmed_before)
        self.assertEqual(mutable_snapshot(self.controller), baseline)

    def test_payload_aliases_still_require_full_identity(self):
        task = self.controller.begin(["sample.png"])
        payload = task.context()
        payload.update({"sample_results": [rgb_sample()], "sample_errors": []})
        missing_operation = dict(payload)
        missing_operation.pop("operation")
        baseline = mutable_snapshot(self.controller)

        self.assertFalse(self.controller.accept_payload(missing_operation))
        self.assertEqual(mutable_snapshot(self.controller), baseline)
        self.assertIs(self.controller.active_job, task)
        self.assertTrue(self.controller.accept_payload(payload))


if __name__ == "__main__":
    unittest.main()
