"""Qt-independent controller for strictly sequential Linear Image Series runs.

The controller deliberately knows nothing about threads or signals.  A caller
dispatches the immutable task returned by :meth:`begin` or :meth:`next_task`,
then offers worker payloads back to :meth:`accept_success` or
:meth:`accept_failure`.  Every result must echo the complete task identity.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple

from linear_series_state import (
    LinearImageStatus,
    LinearSeriesInvariantError,
    LinearSeriesPhase,
    LinearSeriesState,
)


LINEAR_SERIES_OPERATION = "linear_series_extract"
_IDENTITY_FIELDS = (
    "operation",
    "run_token",
    "job_token",
    "image_order",
    "normalized_path",
    "original_file_name",
)


@dataclass(frozen=True)
class LinearSeriesTask:
    """Immutable identity and source information for one image job."""

    operation: str
    run_token: int
    job_token: int
    image_order: int
    normalized_path: str
    original_file_name: str

    def __post_init__(self) -> None:
        if (
            type(self.operation) is not str
            or self.operation != LINEAR_SERIES_OPERATION
        ):
            raise ValueError("Invalid Linear series operation.")
        for name in ("run_token", "job_token", "image_order"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError("{} must be a positive exact integer.".format(name))
        for name in ("normalized_path", "original_file_name"):
            value = getattr(self, name)
            if type(value) is not str or not value:
                raise ValueError("{} must be a non-empty string.".format(name))

    def context(self) -> Dict[str, Any]:
        """Return the exact identity dictionary a worker must echo."""

        return {name: getattr(self, name) for name in _IDENTITY_FIELDS}

    def as_payload(self) -> Dict[str, Any]:
        """Alias useful at thread/worker boundaries."""

        return self.context()

    @property
    def path(self) -> str:
        """Compatibility alias; the canonical field is ``normalized_path``."""

        return self.normalized_path

    @property
    def source_file(self) -> str:
        """Compatibility alias; the canonical field preserves the file name."""

        return self.original_file_name


class LinearSeriesController:
    """Coordinate one Linear Image Series image job at a time.

    Run tokens identify a complete series attempt.  Job tokens identify one
    dispatch within that run and are monotonic across runs.  ``begin`` always
    creates a fresh draft, so calling it again invalidates the old run while
    retaining the opaque last-confirmed Linear result reference held by state.
    """

    operation = LINEAR_SERIES_OPERATION

    def __init__(self, state: Optional[LinearSeriesState] = None) -> None:
        initial_state = state if state is not None else LinearSeriesState()
        if not isinstance(initial_state, LinearSeriesState):
            raise TypeError("state must be a LinearSeriesState")
        initial_state.validate()
        self.state = initial_state
        self._run_token_counter = 0
        self._job_token_counter = 0
        self._run_token: Optional[int] = None
        self._run_tasks: Tuple[LinearSeriesTask, ...] = ()
        self._queue: Tuple[LinearSeriesTask, ...] = ()
        self._active_job: Optional[LinearSeriesTask] = None

    @property
    def run_token(self) -> Optional[int]:
        return self._run_token

    @property
    def active_job(self) -> Optional[LinearSeriesTask]:
        return self._active_job

    @property
    def queued_image_orders(self) -> Tuple[int, ...]:
        return tuple(task.image_order for task in self._queue)

    @property
    def busy(self) -> bool:
        """Whether an extraction run still awaits formal retirement."""

        return self._run_token is not None

    @property
    def active(self) -> bool:
        """Compatibility alias for the extraction busy state."""

        return self.busy

    @property
    def last_confirmed_result(self) -> Any:
        """Return the opaque confirmed Linear result preserved by state."""

        return self.state.last_confirmed_result

    def remember_confirmed_result(self, result: Any) -> Any:
        """Record an opaque confirmed result without coupling to GUI/plot data."""

        return self.state.remember_confirmed_result(result)

    def begin(
        self, paths: Optional[Iterable[object]] = None
    ) -> Optional[LinearSeriesTask]:
        """Start a fresh run and return its first task.

        If ``paths`` is omitted, the current state's paths are reused.  A
        repeated call intentionally abandons the old draft; its task can still
        finish in a worker, but its payload can no longer match this controller.
        """

        old_state = self.state
        if paths is None:
            selected_paths = tuple(
                image.normalized_path for image in old_state.images
            )
        else:
            selected_paths = tuple(paths)

        # Build the complete replacement, including its immutable task
        # identities and initial PROCESSING transition, before touching the
        # current run or its draft.
        confirmed_result = old_state.last_confirmed_result
        candidate_state = LinearSeriesState.from_paths(
            selected_paths,
            last_confirmed_result=confirmed_result,
        )
        candidate_state.validate()

        candidate_run_token = self._run_token_counter + 1
        candidate_job_counter = self._job_token_counter
        candidate_tasks = []
        for image in candidate_state.images:
            candidate_job_counter += 1
            candidate_tasks.append(
                LinearSeriesTask(
                    operation=self.operation,
                    run_token=candidate_run_token,
                    job_token=candidate_job_counter,
                    image_order=image.image_order,
                    normalized_path=image.normalized_path,
                    original_file_name=image.original_file_name,
                )
            )
        run_tasks = tuple(candidate_tasks)

        candidate_active = run_tasks[0] if run_tasks else None
        candidate_queue = run_tasks[1:] if run_tasks else ()
        if candidate_active is not None:
            self._require_task_image(
                candidate_state,
                candidate_active,
                candidate_run_token,
                LinearImageStatus.PENDING,
            )
            candidate_state.mark_image_processing(
                candidate_active.image_order
            )
            candidate_state.validate()

        # Only a fully prepared replacement may invalidate an old draft.
        # This phase-based check deliberately covers a finished extraction
        # whose run token was already retired while its Mapping draft remains.
        old_phase = old_state.phase
        if old_phase not in (
            LinearSeriesPhase.READY,
            LinearSeriesPhase.CANCELLED,
        ):
            old_state.cancel()

        self.state = candidate_state
        self._run_token_counter = candidate_run_token
        self._job_token_counter = candidate_job_counter
        self._run_token = (
            candidate_run_token if candidate_active is not None else None
        )
        self._run_tasks = run_tasks
        self._queue = candidate_queue
        self._active_job = candidate_active
        return candidate_active

    def next_task(self) -> Optional[LinearSeriesTask]:
        """Dispatch the next queued image, or reject an out-of-order request.

        Returning ``None`` is side-effect free when a job is already active,
        the run was cancelled/finished, or the queue is exhausted.
        """

        if self._run_token is None or self._active_job is not None or not self._queue:
            return None

        if type(self._queue) is not tuple:
            raise RuntimeError("Linear series task queue is not immutable.")
        task = self._queue[0]
        candidate_state = self.state.clone()
        self._require_active_run_task(
            candidate_state,
            task,
            LinearImageStatus.PENDING,
        )
        candidate_state.mark_image_processing(task.image_order)
        candidate_state.validate()

        # Adoption cannot fail after candidate validation and changes the
        # complete immutable state graph in one assignment.
        self.state._adopt_validated(candidate_state)
        self._queue = self._queue[1:]
        self._active_job = task
        return task

    request_next_task = next_task

    def matches_active_result(self, payload: object) -> bool:
        """Return whether ``payload`` exactly identifies the active job."""

        active_job = self._active_job
        if (
            not isinstance(payload, Mapping)
            or type(active_job) is not LinearSeriesTask
        ):
            return False
        try:
            for name in _IDENTITY_FIELDS:
                expected = getattr(active_job, name)
                actual = payload[name]
                if type(actual) is not type(expected) or actual != expected:
                    return False
        except (KeyError, TypeError):
            return False
        if (
            type(self._run_token) is not int
            or active_job.run_token != self._run_token
        ):
            return False
        if not any(active_job is task for task in self._run_tasks):
            return False
        return True

    def accept_success(self, payload: object) -> bool:
        """Accept an active job's spatially ordered sample/error collections."""

        if not self.matches_active_result(payload):
            return False
        assert isinstance(payload, Mapping)  # narrowed by matches_active_result

        samples = self._payload_collection(payload, "samples", "sample_results")
        if samples is None:
            return False
        errors = self._payload_collection(
            payload,
            "errors",
            "sample_errors",
            default=(),
        )
        if errors is None:
            return False

        active_job = self._active_job
        assert active_job is not None
        candidate_state = self.state.clone()
        self._require_active_run_task(
            candidate_state,
            active_job,
            LinearImageStatus.PROCESSING,
        )
        candidate_state.accept_image_result(
            active_job.image_order, samples, errors=errors
        )
        candidate_state.validate()
        terminal_image = candidate_state.image_for_order(active_job.image_order)
        if terminal_image.status not in (
            LinearImageStatus.COMPLETED,
            LinearImageStatus.FAILED,
        ):
            raise RuntimeError("Successful payload did not terminate its image.")

        self.state._adopt_validated(candidate_state)
        self._active_job = None
        return True

    accept_payload = accept_success

    def accept_failure(self, payload: object) -> bool:
        """Accept an active job failure after the same full identity check."""

        if not self.matches_active_result(payload):
            return False
        assert isinstance(payload, Mapping)  # narrowed by matches_active_result
        if "reason" not in payload:
            return False
        errors = self._payload_collection(
            payload,
            "errors",
            "sample_errors",
            default=(),
        )
        if errors is None:
            return False

        active_job = self._active_job
        assert active_job is not None
        candidate_state = self.state.clone()
        self._require_active_run_task(
            candidate_state,
            active_job,
            LinearImageStatus.PROCESSING,
        )
        candidate_state.fail_image(
            active_job.image_order, str(payload["reason"]), errors=errors
        )
        candidate_state.validate()
        terminal_image = candidate_state.image_for_order(active_job.image_order)
        if terminal_image.status != LinearImageStatus.FAILED:
            raise RuntimeError("Failure payload did not fail its image.")

        self.state._adopt_validated(candidate_state)
        self._active_job = None
        return True

    def finish_if_done(self) -> Optional[Dict[str, int]]:
        """Retire a fully processed run and return its non-UI summary."""

        if self._run_token is None or self._active_job is not None or self._queue:
            return None
        if not self._finish_state_is_consistent():
            return None
        result = self.summary()
        self._run_token = None
        self._run_tasks = ()
        return result

    def cancel(self) -> bool:
        """Invalidate this run without attempting to terminate its worker."""

        cancellable = self._run_token is not None or self.state.phase in (
            LinearSeriesPhase.EXTRACTING,
            LinearSeriesPhase.MAPPING,
            LinearSeriesPhase.FAILED,
        )
        if not cancellable:
            return False
        # State cancellation is independently atomic.  Do it before retiring
        # controller identity so an exception cannot leave a half-cancelled run.
        self.state.cancel()
        self._run_token = None
        self._active_job = None
        self._run_tasks = ()
        self._queue = ()
        return True

    def summary(self) -> Dict[str, int]:
        """Return extraction counts without importing GUI or numerical stacks."""

        images = self.state.images
        successful_images = 0
        failed_images = 0
        valid_samples = 0
        sample_errors = 0
        for image in images:
            if image.status == LinearImageStatus.COMPLETED:
                successful_images += 1
                valid_samples += len(image.samples)
            elif image.status == LinearImageStatus.FAILED:
                failed_images += 1
            sample_errors += len(image.errors)
        return {
            "total_images": len(images),
            "successful_images": successful_images,
            "failed_images": failed_images,
            "valid_samples": valid_samples,
            "sample_errors": sample_errors,
        }

    @staticmethod
    def _payload_collection(
        payload: Mapping, primary: str, alias: str, default: object = None
    ) -> Optional[Tuple[Any, ...]]:
        if primary in payload:
            values = payload[primary]
        elif alias in payload:
            values = payload[alias]
        else:
            values = default
        if type(values) not in (list, tuple):
            return None
        return tuple(values)

    @staticmethod
    def _task_matches_image(task, image, run_token, expected_status=None):
        if type(task) is not LinearSeriesTask:
            return False
        if type(run_token) is not int or run_token <= 0:
            return False
        if (
            type(task.operation) is not str
            or task.operation != LINEAR_SERIES_OPERATION
            or type(task.run_token) is not int
            or task.run_token != run_token
            or type(task.job_token) is not int
            or task.job_token <= 0
            or type(task.image_order) is not int
            or task.image_order <= 0
            or type(task.normalized_path) is not str
            or not task.normalized_path
            or type(task.original_file_name) is not str
            or not task.original_file_name
        ):
            return False
        if (
            task.image_order != image.image_order
            or task.normalized_path != image.normalized_path
            or task.original_file_name != image.original_file_name
        ):
            return False
        return expected_status is None or image.status == expected_status

    @classmethod
    def _require_task_image(
        cls, state, task, run_token, expected_status
    ) -> None:
        image = state.image_for_order(task.image_order)
        if not cls._task_matches_image(
            task, image, run_token, expected_status
        ):
            raise RuntimeError(
                "Linear series task and image identity are inconsistent."
            )

    def _require_active_run_task(
        self, state, task, expected_status
    ) -> None:
        if type(self._run_tasks) is not tuple or not any(
            task is original for original in self._run_tasks
        ):
            raise RuntimeError(
                "Active Linear series task is not part of this run."
            )
        self._require_task_image(
            state,
            task,
            self._run_token,
            expected_status,
        )

    def _finish_state_is_consistent(self) -> bool:
        if (
            type(self._run_token) is not int
            or self._run_token <= 0
            or type(self._queue) is not tuple
            or self._queue
            or type(self._run_tasks) is not tuple
            or not self._run_tasks
        ):
            return False
        try:
            self.state.validate()
            images = self.state.images
            phase = self.state.phase
        except LinearSeriesInvariantError:
            return False
        if len(images) != len(self._run_tasks):
            return False

        job_tokens = []
        for task, image in zip(self._run_tasks, images):
            if not self._task_matches_image(
                task, image, self._run_token
            ):
                return False
            if image.status not in (
                LinearImageStatus.COMPLETED,
                LinearImageStatus.FAILED,
            ):
                return False
            job_tokens.append(task.job_token)
        if len(job_tokens) != len(set(job_tokens)):
            return False

        has_samples = any(image.samples for image in images)
        expected_phase = (
            LinearSeriesPhase.MAPPING
            if has_samples
            else LinearSeriesPhase.FAILED
        )
        return phase == expected_phase


# Concise compatibility name for callers that prefer a generic task term.
LinearImageTask = LinearSeriesTask


__all__ = [
    "LINEAR_SERIES_OPERATION",
    "LinearImageTask",
    "LinearSeriesController",
    "LinearSeriesTask",
]
