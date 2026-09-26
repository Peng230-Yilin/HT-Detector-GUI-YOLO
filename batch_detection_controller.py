"""Qt-independent state machine for sequential detection runs."""

from dataclasses import dataclass
import time
from types import MappingProxyType
from typing import Optional

from batch_state import (
    BatchState,
    DetectionScope,
    ImageStatus,
    NumberingMode,
    ConcentrationStatus,
    RegressionSet,
    SampleError,
    SampleErrorType,
    SampleResult,
    SampleStatus,
    normalize_internal_path,
)


@dataclass(frozen=True)
class DetectionRunContext:
    run_token: int
    regression_set: RegressionSet
    regression_revision: str

    def __post_init__(self):
        if self.regression_revision != self.regression_set.revision:
            raise ValueError("Run regression revision does not match its RegressionSet.")


@dataclass(frozen=True)
class DetectionTask:
    run_token: int
    job_token: int
    regression_revision: str
    path: str
    source_file: str
    image_order: int
    batch_start_no: int
    display_start_no: int
    regression_set: RegressionSet

    def __post_init__(self):
        if self.regression_revision != self.regression_set.revision:
            raise ValueError("Task regression revision does not match its RegressionSet.")

    def context(self):
        return {
            "run_token": self.run_token,
            "job_token": self.job_token,
            "regression_revision": self.regression_revision,
            "image_order": self.image_order,
            "source_path": self.path,
            "source_file": self.source_file,
            "batch_start_no": self.batch_start_no,
            "display_start_no": self.display_start_no,
            "regression_set": self.regression_set,
        }


class BatchDetectionController:
    def __init__(self, state=None, clock_ns=None):
        self.state = state or BatchState()
        self._clock_ns = time.perf_counter_ns if clock_ns is None else clock_ns
        if not callable(self._clock_ns):
            raise TypeError("clock_ns must be callable.")
        self._token_counter = 0
        self._job_token_counter = 0
        self._run_context = None
        self._queue = []
        self._active_job = None
        self._prepared_job = None
        self._active_started_ns = None
        self._next_batch_no = 1

    @property
    def active(self):
        return self._run_context is not None

    @property
    def run_token(self):
        return None if self._run_context is None else self._run_context.run_token

    @property
    def run_context(self):
        return self._run_context

    @property
    def active_job(self):
        return self._active_job

    @property
    def active_elapsed_ms(self):
        """Return elapsed whole milliseconds without changing timer state."""
        if self._active_job is None or self._active_started_ns is None:
            return None
        elapsed_ns = self._clock_ns() - self._active_started_ns
        return max(0, elapsed_ns // 1_000_000)

    def replace_images(self, paths):
        if self.active:
            raise RuntimeError("Cannot replace images during detection.")
        self._prepared_job = None
        self._active_started_ns = None
        scope = self.state.detection_scope
        numbering_mode = self.state.numbering_mode
        regression_set = self.state.regression_set
        self.state = BatchState.from_paths(paths)
        self.state.detection_scope = scope
        self.state.numbering_mode = numbering_mode
        self.state.regression_set = regression_set
        return self.state

    def set_options(self, scope, numbering_mode):
        if self.active:
            raise RuntimeError("Cannot change detection options during detection.")
        self.state.detection_scope = DetectionScope(scope)
        self.state.numbering_mode = NumberingMode(numbering_mode)

    def begin(self, regression_set=None):
        if self.active:
            return None
        if not self.state.images:
            return None
        if regression_set is None:
            regression_set = self.state.regression_set
        if not isinstance(regression_set, RegressionSet):
            raise TypeError("A complete RegressionSet is required for detection.")
        self.state.regression_set = regression_set
        self._token_counter += 1
        self._run_context = DetectionRunContext(
            self._token_counter, regression_set, regression_set.revision
        )
        self._active_job = None
        self._prepared_job = None
        self._active_started_ns = None
        self._next_batch_no = 1
        self.state.last_batch_result = None
        if self.state.detection_scope == DetectionScope.CURRENT_IMAGE:
            self._queue = [self.state.current_image.image_order]
        else:
            self._queue = [image.image_order for image in self.state.images]
        planned_orders = set(self._queue)
        for image in self.state.images:
            image.status = ImageStatus.PENDING
            image.samples.clear()
            image.errors.clear()
            if image.image_order in planned_orders:
                image.detection_duration_ms = None
        return self.next_task()

    def next_task(self) -> Optional[DetectionTask]:
        if not self.active or self._active_job is not None or not self._queue:
            return None
        image_order = self._queue.pop(0)
        image = self._image(image_order)
        self.state.current_image_index = self.state.images.index(image)
        image.status = ImageStatus.PROCESSING
        self._job_token_counter += 1
        continuous = (
            self.state.detection_scope == DetectionScope.ALL_IMPORTED_IMAGES
            and self.state.numbering_mode == NumberingMode.CONTINUOUS_BATCH
        )
        task = DetectionTask(
            run_token=self._run_context.run_token,
            job_token=self._job_token_counter,
            regression_revision=self._run_context.regression_revision,
            path=image.path,
            source_file=image.original_filename,
            image_order=image.image_order,
            batch_start_no=self._next_batch_no,
            display_start_no=self._next_batch_no if continuous else 1,
            regression_set=self._run_context.regression_set,
        )
        self._active_job = task
        self._prepared_job = None
        self._active_started_ns = None
        return task

    def prepare_dispatch_context(self, task, validator):
        """Validate and freeze the active task before any asynchronous emit.

        A rejected context is converted through the normal controller failure
        path using the active task's authoritative identity.  The caller can
        then advance the run exactly once without emitting a worker signal.
        """
        if task is not self._active_job:
            raise ValueError("Only the current active detection task may be dispatched.")
        try:
            locked_context = validator(task.path, task.context())
            if not isinstance(locked_context, MappingProxyType):
                raise ValueError(
                    "Detection context validator must return an immutable mapping proxy."
                )
        except Exception as error:
            failure = self.reject_dispatch(task, error)
            if failure is None:
                raise RuntimeError(
                    "The rejected detection task could not be completed."
                ) from error
            return None, failure
        self._prepared_job = task
        return locked_context, None

    def start_dispatch_timer(self, task):
        """Start timing a prepared task immediately before its worker emit."""
        if task is not self._active_job or task is not self._prepared_job:
            raise ValueError("Only the prepared active detection task may be dispatched.")
        if self._active_started_ns is not None:
            raise RuntimeError("The active detection task timer is already running.")
        self._active_started_ns = self._clock_ns()
        self._prepared_job = None

    def reject_dispatch(self, task, error):
        """Fail the active task without trusting data from a rejected context."""
        authoritative = self._active_job
        if task is not authoritative:
            return None
        failure = {
            "run_token": authoritative.run_token,
            "job_token": authoritative.job_token,
            "regression_revision": authoritative.regression_revision,
            "image_order": authoritative.image_order,
            "source_path": authoritative.path,
            "source_file": authoritative.source_file,
            "message": "Detection dispatch rejected: {}: {}".format(
                type(error).__name__, error
            ),
        }
        if not self.accept_failure(failure):
            raise RuntimeError("The authoritative detection task rejection failed.")
        return failure

    def matches_active_result(self, result):
        if not isinstance(result, dict) or self._active_job is None:
            return False
        task = self._active_job
        required = {
            "run_token": task.run_token,
            "job_token": task.job_token,
            "regression_revision": task.regression_revision,
            "image_order": task.image_order,
            "source_path": task.path,
            "source_file": task.source_file,
        }
        if any(name not in result for name in required):
            return False
        for name in ("run_token", "job_token", "image_order"):
            if type(result[name]) is not int or result[name] != required[name]:
                return False
        if (
            type(result["regression_revision"]) is not str
            or result["regression_revision"] != task.regression_revision
            or type(result["source_path"]) is not str
            or type(result["source_file"]) is not str
            or result["source_file"] != task.source_file
        ):
            return False
        try:
            return normalize_internal_path(result["source_path"]) == task.path
        except (TypeError, ValueError, OSError):
            return False

    def accept_payload(self, payload):
        if not self.matches_active_result(payload):
            return False
        image = self._image(self._active_job.image_order)
        image.samples = [self._sample_from_dict(value) for value in payload.get("sample_results", [])]
        image.errors = [self._error_from_dict(value) for value in payload.get("sample_errors", [])]
        image.status = ImageStatus.COMPLETED if image.samples else ImageStatus.FAILED
        if image.status == ImageStatus.COMPLETED:
            self._next_batch_no += len(image.samples)
            self.state.last_batch_result = payload
        self._store_active_duration(image)
        self._active_job = None
        self._prepared_job = None
        return True

    def accept_failure(self, failure):
        if not self.matches_active_result(failure):
            return False
        image = self._image(self._active_job.image_order)
        image.status = ImageStatus.FAILED
        image.samples.clear()
        image.errors = [SampleError(
            image.image_order, image.original_filename,
            SampleErrorType.IMAGE_FAILED,
            str(failure.get("message", "Detection failed.")),
        )]
        self._store_active_duration(image)
        self._active_job = None
        self._prepared_job = None
        return True

    def finish_if_done(self):
        if not self.active or self._active_job is not None or self._queue:
            return None
        summary = self.summary()
        self._run_context = None
        self._prepared_job = None
        self._active_started_ns = None
        return summary

    def summary(self):
        return {
            "total_images": len(self.state.images),
            "successful_images": sum(i.status == ImageStatus.COMPLETED for i in self.state.images),
            "failed_images": sum(i.status == ImageStatus.FAILED for i in self.state.images),
            "valid_samples": sum(len(i.samples) for i in self.state.images if i.status == ImageStatus.COMPLETED),
            "sample_errors": sum(
                error.error_type != SampleErrorType.IMAGE_FAILED
                for image in self.state.images for error in image.errors
            ),
        }

    def _image(self, image_order):
        return next(image for image in self.state.images if image.image_order == image_order)

    def _store_active_duration(self, image):
        if self._active_started_ns is None:
            return
        elapsed_ns = self._clock_ns() - self._active_started_ns
        image.detection_duration_ms = max(0, elapsed_ns // 1_000_000)
        self._active_started_ns = None

    @staticmethod
    def _sample_from_dict(value):
        fields = dict(value)
        fields["status"] = SampleStatus(fields.get("status", "valid"))
        for channel in ("r", "g", "b"):
            name = "status_{}".format(channel)
            fields[name] = ConcentrationStatus(fields[name])
        return SampleResult(**fields)

    @staticmethod
    def _error_from_dict(value):
        fields = dict(value)
        fields.pop("no_in_image", None)
        fields.pop("batch_no", None)
        return SampleError(**fields)
