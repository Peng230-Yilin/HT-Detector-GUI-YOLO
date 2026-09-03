"""Qt-independent state machine for sequential detection runs."""

from dataclasses import dataclass
from typing import Optional

from batch_state import (
    BatchState,
    DetectionScope,
    ImageStatus,
    NumberingMode,
    SampleError,
    SampleErrorType,
    SampleResult,
)


@dataclass(frozen=True)
class DetectionTask:
    run_token: int
    job_token: int
    path: str
    source_file: str
    image_order: int
    batch_start_no: int
    display_start_no: int

    def context(self):
        return {
            "run_token": self.run_token,
            "job_token": self.job_token,
            "image_order": self.image_order,
            "batch_start_no": self.batch_start_no,
            "display_start_no": self.display_start_no,
        }


class BatchDetectionController:
    def __init__(self, state=None):
        self.state = state or BatchState()
        self._token_counter = 0
        self._job_token_counter = 0
        self._run_token = None
        self._queue = []
        self._active_job = None
        self._next_batch_no = 1

    @property
    def active(self):
        return self._run_token is not None

    @property
    def run_token(self):
        return self._run_token

    @property
    def active_job(self):
        return self._active_job

    def replace_images(self, paths):
        if self.active:
            raise RuntimeError("Cannot replace images during detection.")
        scope = self.state.detection_scope
        numbering_mode = self.state.numbering_mode
        self.state = BatchState.from_paths(paths)
        self.state.detection_scope = scope
        self.state.numbering_mode = numbering_mode
        return self.state

    def set_options(self, scope, numbering_mode):
        if self.active:
            raise RuntimeError("Cannot change detection options during detection.")
        self.state.detection_scope = DetectionScope(scope)
        self.state.numbering_mode = NumberingMode(numbering_mode)

    def begin(self):
        if self.active:
            return None
        if not self.state.images:
            return None
        self._token_counter += 1
        self._run_token = self._token_counter
        self._active_job = None
        self._next_batch_no = 1
        self.state.last_batch_result = None
        for image in self.state.images:
            image.status = ImageStatus.PENDING
            image.samples.clear()
            image.errors.clear()
        if self.state.detection_scope == DetectionScope.CURRENT_IMAGE:
            self._queue = [self.state.current_image.image_order]
        else:
            self._queue = [image.image_order for image in self.state.images]
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
            self._run_token, self._job_token_counter, image.path,
            image.original_filename, image.image_order, self._next_batch_no,
            self._next_batch_no if continuous else 1,
        )
        self._active_job = task
        return task

    def matches_active_result(self, result):
        if not isinstance(result, dict):
            return False
        return self._matches(result.get("run_token"), result.get("job_token"))

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
        self._active_job = None
        return True

    def accept_failure(self, run_token, job_token, reason):
        if not self._matches(run_token, job_token):
            return False
        image = self._image(self._active_job.image_order)
        image.status = ImageStatus.FAILED
        image.samples.clear()
        image.errors = [SampleError(
            image.image_order, image.original_filename,
            SampleErrorType.IMAGE_FAILED, str(reason),
        )]
        self._active_job = None
        return True

    def finish_if_done(self):
        if not self.active or self._active_job is not None or self._queue:
            return None
        summary = self.summary()
        self._run_token = None
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

    def _matches(self, run_token, job_token):
        return (
            self.active
            and self._active_job is not None
            and run_token == self._active_job.run_token
            and job_token == self._active_job.job_token
        )

    def _image(self, image_order):
        return next(image for image in self.state.images if image.image_order == image_order)

    @staticmethod
    def _sample_from_dict(value):
        fields = dict(value)
        fields["status"] = fields.get("status", "valid")
        return SampleResult(**fields)

    @staticmethod
    def _error_from_dict(value):
        fields = dict(value)
        fields.pop("no_in_image", None)
        fields.pop("batch_no", None)
        return SampleError(**fields)
