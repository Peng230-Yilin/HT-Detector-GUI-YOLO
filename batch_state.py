"""Pure batch, pairing, spatial ordering, and numbering primitives."""

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from enum import Enum
import hashlib
import json
import math
import numbers
import os
from pathlib import Path
import re
from statistics import median
import time
from typing import Iterable, List, Optional, Sequence, Tuple


Box = Tuple[float, float, float, float]
_NATURAL_PART = re.compile(r"(\d+)")


class DetectionScope(str, Enum):
    CURRENT_IMAGE = "current_image"
    ALL_IMPORTED_IMAGES = "entire_batch"


class NumberingMode(str, Enum):
    PER_IMAGE = "per_image"
    CONTINUOUS_BATCH = "continuous"


class CalibrationAttemptState(str, Enum):
    NOT_ATTEMPTED = "not_attempted"
    FAILED_NO_ACTIVE = "failed_no_active"
    FAILED_WITH_ACTIVE = "failed_with_active"
    SUCCEEDED = "succeeded"


class ImageStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class SampleStatus(str, Enum):
    VALID = "valid"


class ConcentrationStatus(str, Enum):
    IN_RANGE = "IN_RANGE"
    BELOW_RANGE = "BELOW_RANGE"
    ABOVE_RANGE = "ABOVE_RANGE"
    MISSING_REGRESSION = "MISSING_REGRESSION"
    INVALID_SLOPE = "INVALID_SLOPE"
    INVALID_REGRESSION = "INVALID_REGRESSION"
    RANGE_UNAVAILABLE = "RANGE_UNAVAILABLE"
    CALCULATION_FAILED = "CALCULATION_FAILED"


_NUMERIC_CONCENTRATION_STATUSES = frozenset({
    ConcentrationStatus.IN_RANGE,
    ConcentrationStatus.BELOW_RANGE,
    ConcentrationStatus.ABOVE_RANGE,
    ConcentrationStatus.RANGE_UNAVAILABLE,
})
_FORMULA_FIELDS = ("slope", "intercept", "r", "R2", "p", "std_err")
_REGRESSION_CHANNELS = ("R", "G", "B")
_ZERO_SLOPE_TOLERANCE = 1e-12


def _finite_real(value):
    return (
        not isinstance(value, bool)
        and isinstance(value, numbers.Real)
        and math.isfinite(float(value))
    )


@dataclass(frozen=True)
class RegressionChannelModel:
    """One immutable channel slot from one confirmed regression source."""

    source_id: str
    channel: str
    slope: Optional[float] = None
    intercept: Optional[float] = None
    r: Optional[float] = None
    r_squared: Optional[float] = None
    p: Optional[float] = None
    std_err: Optional[float] = None
    valid_min: Optional[float] = None
    valid_max: Optional[float] = None
    unavailable_status: Optional[ConcentrationStatus] = None

    def __post_init__(self):
        if not isinstance(self.source_id, str) or not self.source_id:
            raise ValueError("Regression source_id must be a non-empty string.")
        if self.channel not in _REGRESSION_CHANNELS:
            raise ValueError("Regression channel must be R, G, or B.")
        if self.unavailable_status is not None and not isinstance(
            self.unavailable_status, ConcentrationStatus
        ):
            raise ValueError("Regression unavailable_status is invalid.")

        if self.unavailable_status in (None, ConcentrationStatus.RANGE_UNAVAILABLE):
            for name in ("slope", "intercept", "r", "r_squared", "p", "std_err"):
                if not _finite_real(getattr(self, name)):
                    raise ValueError("Regression {} must be finite.".format(name))
            if math.isclose(
                float(self.slope), 0.0, rel_tol=0.0,
                abs_tol=_ZERO_SLOPE_TOLERANCE,
            ):
                raise ValueError("A calculable regression slope must not be zero or near zero.")

        if self.unavailable_status is None:
            if not _finite_real(self.valid_min) or not _finite_real(self.valid_max):
                raise ValueError("A valid regression range must be finite.")
            if float(self.valid_min) > float(self.valid_max):
                raise ValueError("Regression valid_min must not exceed valid_max.")
        elif self.unavailable_status == ConcentrationStatus.RANGE_UNAVAILABLE:
            if self.valid_min is not None or self.valid_max is not None:
                raise ValueError("An unavailable regression range must be empty.")

    @classmethod
    def from_formula(cls, source_id, channel, formula, valid_range=None):
        if formula is None:
            return cls(
                source_id, channel,
                unavailable_status=ConcentrationStatus.MISSING_REGRESSION,
            )
        if not isinstance(formula, Mapping):
            return cls(
                source_id, channel,
                unavailable_status=ConcentrationStatus.INVALID_REGRESSION,
            )

        raw_slope = formula.get("slope")
        if not _finite_real(raw_slope):
            return cls(
                source_id, channel,
                unavailable_status=ConcentrationStatus.INVALID_REGRESSION,
            )
        slope = float(raw_slope)
        if math.isclose(
            slope, 0.0, rel_tol=0.0, abs_tol=_ZERO_SLOPE_TOLERANCE
        ):
            return cls(
                source_id, channel, slope=slope,
                unavailable_status=ConcentrationStatus.INVALID_SLOPE,
            )

        raw_values = {name: formula.get(name) for name in _FORMULA_FIELDS}
        if not all(_finite_real(value) for value in raw_values.values()):
            return cls(
                source_id, channel, slope=slope,
                unavailable_status=ConcentrationStatus.INVALID_REGRESSION,
            )
        values = {name: float(value) for name, value in raw_values.items()}
        model_values = {
            "slope": values["slope"],
            "intercept": values["intercept"],
            "r": values["r"],
            "r_squared": values["R2"],
            "p": values["p"],
            "std_err": values["std_err"],
        }

        if valid_range is None:
            return cls(
                source_id, channel, **model_values,
                unavailable_status=ConcentrationStatus.RANGE_UNAVAILABLE,
            )
        if (
            not isinstance(valid_range, (tuple, list))
            or len(valid_range) != 2
            or not all(_finite_real(value) for value in valid_range)
            or float(valid_range[0]) > float(valid_range[1])
        ):
            return cls(
                source_id, channel,
                unavailable_status=ConcentrationStatus.INVALID_REGRESSION,
            )
        return cls(
            source_id, channel, **model_values,
            valid_min=float(valid_range[0]),
            valid_max=float(valid_range[1]),
        )

    def calculate(self, intensity):
        if self.unavailable_status not in (
            None, ConcentrationStatus.RANGE_UNAVAILABLE
        ):
            return None, self.unavailable_status
        if not _finite_real(intensity):
            return None, ConcentrationStatus.CALCULATION_FAILED
        try:
            concentration = (float(intensity) - self.intercept) / self.slope
        except (ArithmeticError, TypeError, ValueError):
            return None, ConcentrationStatus.CALCULATION_FAILED
        if not math.isfinite(concentration):
            return None, ConcentrationStatus.CALCULATION_FAILED
        if self.unavailable_status == ConcentrationStatus.RANGE_UNAVAILABLE:
            return concentration, ConcentrationStatus.RANGE_UNAVAILABLE
        if concentration < self.valid_min:
            return concentration, ConcentrationStatus.BELOW_RANGE
        if concentration > self.valid_max:
            return concentration, ConcentrationStatus.ABOVE_RANGE
        return concentration, ConcentrationStatus.IN_RANGE

    def revision_values(self):
        return (
            self.channel, self.slope, self.intercept, self.r, self.r_squared,
            self.p, self.std_err, self.valid_min, self.valid_max,
            self.unavailable_status.value if self.unavailable_status else None,
        )


@dataclass(frozen=True)
class RegressionSet:
    """A cohesive R/G/B snapshot that can never mix confirmed sources."""

    source_id: str
    red: RegressionChannelModel
    green: RegressionChannelModel
    blue: RegressionChannelModel
    revision: str

    def __post_init__(self):
        if not isinstance(self.source_id, str) or not self.source_id:
            raise ValueError("RegressionSet source_id must be a non-empty string.")
        models = (self.red, self.green, self.blue)
        if tuple(model.channel for model in models) != _REGRESSION_CHANNELS:
            raise ValueError("RegressionSet must contain ordered R, G, and B slots.")
        if any(model.source_id != self.source_id for model in models):
            raise ValueError("RegressionSet channels must share one confirmed source.")
        expected = self._revision_for(self.source_id, models)
        if self.revision != expected:
            raise ValueError("RegressionSet revision does not match its channel snapshot.")

    @staticmethod
    def _revision_for(source_id, models):
        document = {
            "source_id": source_id,
            "channels": [model.revision_values() for model in models],
        }
        encoded = json.dumps(
            document, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @classmethod
    def from_formulas(cls, formulas, source_id, valid_ranges=None):
        if not isinstance(source_id, str) or not source_id:
            raise ValueError("Regression source_id must be a non-empty string.")
        if not isinstance(formulas, Mapping):
            formulas = {}
        unknown = set(formulas) - set(_REGRESSION_CHANNELS)
        if unknown:
            raise ValueError(
                "Regression formulas contain unknown channel(s): {}.".format(
                    ", ".join(sorted(map(str, unknown)))
                )
            )
        if valid_ranges is None:
            ranges = {}
        elif isinstance(valid_ranges, Mapping):
            ranges = valid_ranges
        else:
            ranges = {channel: valid_ranges for channel in _REGRESSION_CHANNELS}
        models = tuple(
            RegressionChannelModel.from_formula(
                source_id, channel, formulas.get(channel), ranges.get(channel)
            )
            for channel in _REGRESSION_CHANNELS
        )
        revision = cls._revision_for(source_id, models)
        return cls(source_id, models[0], models[1], models[2], revision)

    @classmethod
    def missing(cls, source_id="unconfigured"):
        return cls.from_formulas({}, source_id)

    def model(self, channel):
        try:
            return {"R": self.red, "G": self.green, "B": self.blue}[channel]
        except KeyError as error:
            raise ValueError("Regression channel must be R, G, or B.") from error

    def calculate_all(self, red, green, blue):
        values = {"R": red, "G": green, "B": blue}
        return {
            channel: self.model(channel).calculate(values[channel])
            for channel in _REGRESSION_CHANNELS
        }


@dataclass(frozen=True)
class RegressionTiming:
    """Authoritative elapsed time bound to one confirmed RegressionSet."""

    source_id: str
    regression_revision: str
    duration_ms: Optional[int]

    def __post_init__(self):
        if not isinstance(self.source_id, str) or not self.source_id:
            raise ValueError("Regression timing source_id must be a non-empty string.")
        if not isinstance(self.regression_revision, str) or not self.regression_revision:
            raise ValueError(
                "Regression timing revision must be a non-empty string."
            )
        _validate_duration_ms(self.duration_ms, "Regression duration")

    def matches(self, regression_set):
        return (
            isinstance(regression_set, RegressionSet)
            and self.source_id == regression_set.source_id
            and self.regression_revision == regression_set.revision
        )


@dataclass(frozen=True)
class RegressionOperationIdentity:
    """Immutable identity for one dispatched Linear Regression operation."""

    source_id: str
    operation_token: int

    def __post_init__(self):
        if not isinstance(self.source_id, str) or not self.source_id:
            raise ValueError(
                "Regression operation source_id must be a non-empty string."
            )
        if type(self.operation_token) is not int or self.operation_token <= 0:
            raise ValueError(
                "Regression operation token must be an exact positive integer."
            )


@dataclass
class RegressionSessionState:
    """Explicitly controls runtime regression activation and legacy fallback."""

    active_regression_set: Optional[RegressionSet] = None
    calibration_state: CalibrationAttemptState = CalibrationAttemptState.NOT_ATTEMPTED
    active_regression_timing: Optional[RegressionTiming] = None
    clock_ns: object = field(default=time.perf_counter_ns, repr=False, compare=False)
    _active_operation: Optional[RegressionOperationIdentity] = field(
        default=None, init=False, repr=False
    )
    _active_started_ns: Optional[int] = field(default=None, init=False, repr=False)
    _next_operation_token: int = field(default=1, init=False, repr=False)

    def __post_init__(self):
        if (
            self.active_regression_set is not None
            and not isinstance(self.active_regression_set, RegressionSet)
        ):
            raise TypeError("active_regression_set must be a RegressionSet or None.")
        if not isinstance(self.calibration_state, CalibrationAttemptState):
            self.calibration_state = CalibrationAttemptState(self.calibration_state)
        if (
            self.active_regression_timing is not None
            and not isinstance(self.active_regression_timing, RegressionTiming)
        ):
            raise TypeError(
                "active_regression_timing must be a RegressionTiming or None."
            )
        if (
            self.active_regression_timing is not None
            and (
                self.active_regression_set is None
                or not self.active_regression_timing.matches(
                    self.active_regression_set
                )
            )
        ):
            raise ValueError(
                "Regression timing must match the active RegressionSet."
            )
        if not callable(self.clock_ns):
            raise TypeError("clock_ns must be callable.")

    def _read_clock_ns(self):
        value = self.clock_ns()
        if type(value) is not int:
            raise TypeError("clock_ns must return an exact integer.")
        return value

    @property
    def active_source_id(self):
        operation = self._active_operation
        return None if operation is None else operation.source_id

    @property
    def active_operation(self):
        return self._active_operation

    @property
    def active_operation_token(self):
        operation = self._active_operation
        return None if operation is None else operation.operation_token

    @property
    def attempt_active(self):
        return (
            self._active_operation is not None
            and self._active_started_ns is not None
        )

    @property
    def active_elapsed_ms(self):
        if not self.attempt_active:
            return None
        return max(0, (self._read_clock_ns() - self._active_started_ns) // 1_000_000)

    def begin_attempt(self, source_id):
        if not isinstance(source_id, str) or not source_id:
            raise ValueError("Regression attempt source_id must be a non-empty string.")
        if self.attempt_active:
            raise RuntimeError("A Linear Regression attempt is already active.")
        operation = RegressionOperationIdentity(
            source_id,
            self._next_operation_token,
        )
        self._next_operation_token += 1
        started_ns = self._read_clock_ns()
        self._active_operation = operation
        self._active_started_ns = started_ns
        return operation

    def matches_active_operation(self, source_id, operation_token):
        if not isinstance(source_id, str) or not source_id:
            return False
        if type(operation_token) is not int or operation_token <= 0:
            return False
        operation = self._active_operation
        return (
            self.attempt_active
            and operation.source_id == source_id
            and operation.operation_token == operation_token
        )

    def cancel_attempt(self):
        was_active = self.attempt_active
        self._active_operation = None
        self._active_started_ns = None
        return was_active

    def accept_success(self, regression_set, operation_token):
        if not isinstance(regression_set, RegressionSet):
            raise TypeError("Only a complete RegressionSet can be accepted.")
        if not self.matches_active_operation(
            regression_set.source_id,
            operation_token,
        ):
            return None
        ended_ns = self._read_clock_ns()
        duration_ms = max(
            0, (ended_ns - self._active_started_ns) // 1_000_000
        )
        timing = RegressionTiming(
            regression_set.source_id,
            regression_set.revision,
            duration_ms,
        )
        self.active_regression_set = regression_set
        self.active_regression_timing = timing
        self.calibration_state = CalibrationAttemptState.SUCCEEDED
        self.cancel_attempt()
        return timing

    def timing_for(self, regression_set):
        if not isinstance(regression_set, RegressionSet):
            return None
        timing = self.active_regression_timing
        if (
            self.active_regression_set != regression_set
            or timing is None
            or not timing.matches(regression_set)
        ):
            return None
        return timing

    def activate(self, regression_set):
        if not isinstance(regression_set, RegressionSet):
            raise TypeError("Only a complete RegressionSet can be activated.")
        self.active_regression_set = regression_set
        if (
            self.active_regression_timing is not None
            and not self.active_regression_timing.matches(regression_set)
        ):
            self.active_regression_timing = None
        self.calibration_state = CalibrationAttemptState.SUCCEEDED
        return regression_set

    def accept_failure(self, source_id, operation_token):
        if not self.matches_active_operation(source_id, operation_token):
            return False
        self.cancel_attempt()
        self.calibration_state = (
            CalibrationAttemptState.FAILED_WITH_ACTIVE
            if self.active_regression_set is not None
            else CalibrationAttemptState.FAILED_NO_ACTIVE
        )
        return True

    def record_failure(self, source_id, operation_token):
        if not self.accept_failure(source_id, operation_token):
            return None
        return self.active_regression_set

    def regression_for_detection(self, legacy_loader):
        if self.active_regression_set is not None:
            return self.active_regression_set
        if self.calibration_state == CalibrationAttemptState.FAILED_NO_ACTIVE:
            raise RuntimeError(
                "Calibration failed and no active regression is available. "
                "Complete a valid Linear Regression before Detection."
            )
        if self.calibration_state != CalibrationAttemptState.NOT_ATTEMPTED:
            raise RuntimeError("No active regression is available for Detection.")
        regression_set = legacy_loader()
        if not isinstance(regression_set, RegressionSet):
            raise TypeError("The legacy regression loader did not return a RegressionSet.")
        self.active_regression_set = regression_set
        return regression_set


class SampleErrorType(str, Enum):
    UNMATCHED_CUVETTE = "unmatched_cuvette"
    UNMATCHED_LIQUID = "unmatched_liquid"
    AMBIGUOUS_CUVETTE = "ambiguous_cuvette"
    AMBIGUOUS_LIQUID = "ambiguous_liquid"
    INVALID_ROI = "invalid_roi"
    MEASUREMENT_FAILED = "measurement_failed"
    IMAGE_FAILED = "image_failed"


@dataclass
class SampleResult:
    image_order: int
    source_file: str
    cuvette_box: Box
    liquid_box: Box
    roi_box: Box
    red: float
    green: float
    blue: float
    con_r: Optional[float] = None
    con_g: Optional[float] = None
    con_b: Optional[float] = None
    status_r: ConcentrationStatus = ConcentrationStatus.MISSING_REGRESSION
    status_g: ConcentrationStatus = ConcentrationStatus.MISSING_REGRESSION
    status_b: ConcentrationStatus = ConcentrationStatus.MISSING_REGRESSION
    no_in_image: Optional[int] = None
    batch_no: Optional[int] = None
    detection_run_token: Optional[int] = None
    status: SampleStatus = SampleStatus.VALID
    warnings: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.detection_run_token is not None and (
            isinstance(self.detection_run_token, bool)
            or not isinstance(self.detection_run_token, int)
            or self.detection_run_token < 1
        ):
            raise ValueError("detection_run_token must be a positive integer or None.")
        for channel in ("r", "g", "b"):
            concentration = getattr(self, "con_{}".format(channel))
            status = getattr(self, "status_{}".format(channel))
            if not isinstance(status, ConcentrationStatus):
                raise ValueError("Status.{} is invalid.".format(channel.upper()))
            if status in _NUMERIC_CONCENTRATION_STATUSES:
                if not _finite_real(concentration):
                    raise ValueError(
                        "Con.{} must be finite for status {}.".format(
                            channel.upper(), status.value
                        )
                    )
            elif concentration is not None:
                raise ValueError(
                    "Con.{} must be empty for status {}.".format(
                        channel.upper(), status.value
                    )
                )

    def as_dict(self):
        value = asdict(self)
        value.pop("detection_run_token", None)
        value["status"] = self.status.value
        for channel in ("r", "g", "b"):
            value["status_{}".format(channel)] = getattr(
                self, "status_{}".format(channel)
            ).value
        return value

    def concentration_for(self, channel):
        if channel not in _REGRESSION_CHANNELS:
            raise ValueError("Concentration channel must be R, G, or B.")
        return getattr(self, "con_{}".format(channel.lower()))

    def concentration_status_for(self, channel):
        if channel not in _REGRESSION_CHANNELS:
            raise ValueError("Concentration channel must be R, G, or B.")
        return getattr(self, "status_{}".format(channel.lower()))

    def legacy_target(self, concentration, display_number=None):
        return {
            "No.": self.no_in_image if display_number is None else display_number,
            "Con.": concentration,
            "Red": self.red,
            "Green": self.green,
            "Blue": self.blue,
            "cuvette_box": self.cuvette_box,
            "liquid_box": self.liquid_box,
            "rgb_roi": self.roi_box,
        }


@dataclass
class SampleError:
    image_order: int
    source_file: str
    error_type: SampleErrorType
    reason: str
    related_boxes: List[Box] = field(default_factory=list)
    related_cuvette_boxes: List[Box] = field(default_factory=list)
    related_liquid_boxes: List[Box] = field(default_factory=list)
    position: Optional[Tuple[float, float]] = None
    no_in_image: None = field(default=None, init=False)
    batch_no: None = field(default=None, init=False)

    def as_dict(self):
        return asdict(self)


@dataclass
class ImageItem:
    path: str
    original_filename: str
    image_order: int
    status: ImageStatus = ImageStatus.PENDING
    samples: List[SampleResult] = field(default_factory=list)
    errors: List[SampleError] = field(default_factory=list)
    detection_duration_ms: Optional[int] = None

    def __post_init__(self):
        _validate_detection_duration_ms(self.detection_duration_ms)


def _validate_duration_ms(duration_ms, description):
    if duration_ms is not None and (
        isinstance(duration_ms, bool)
        or not isinstance(duration_ms, int)
        or duration_ms < 0
    ):
        raise ValueError(
            "{} must be a non-negative integer or None.".format(description)
        )


def _validate_detection_duration_ms(duration_ms):
    _validate_duration_ms(duration_ms, "Detection duration")


def format_detection_duration(duration_ms, empty=""):
    """Format an authoritative image duration as integer milliseconds."""

    _validate_detection_duration_ms(duration_ms)
    return empty if duration_ms is None else str(duration_ms)


def detection_table_rows(image: ImageItem, numbering_mode, channel):
    """Build display rows exclusively from an image's authoritative samples."""

    if not isinstance(image, ImageItem):
        raise TypeError("A valid ImageItem is required for the Detection table.")
    numbering_mode = NumberingMode(numbering_mode)
    if channel not in _REGRESSION_CHANNELS:
        raise ValueError("Detection table channel must be R, G, or B.")
    number_field = (
        "batch_no"
        if numbering_mode == NumberingMode.CONTINUOUS_BATCH
        else "no_in_image"
    )
    return [
        (
            getattr(sample, number_field),
            sample.concentration_for(channel),
            sample.red,
            sample.green,
            sample.blue,
        )
        for sample in image.samples
    ]


@dataclass(frozen=True)
class DetectionExportIdentity:
    """Read-only Scheme B identity copied from authoritative batch state."""

    image_order: int
    original_filename: str
    no_in_image: int
    batch_no: int


def detection_export_identities(image: ImageItem) -> Tuple[DetectionExportIdentity, ...]:
    """Return export identities without deriving numbers from display targets."""

    if not isinstance(image, ImageItem):
        raise ValueError("A valid batch image is required for Detection export.")
    if image.status != ImageStatus.COMPLETED or not image.samples:
        raise ValueError("Only a completed image with valid samples can be exported.")
    if isinstance(image.image_order, bool) or not isinstance(image.image_order, int):
        raise ValueError("Detection export Image Order must be an integer.")
    if image.image_order < 1:
        raise ValueError("Detection export Image Order must start at 1.")
    if not isinstance(image.original_filename, str) or not image.original_filename:
        raise ValueError("Detection export Original Filename is missing.")

    identities = []
    for sample in image.samples:
        if sample.image_order != image.image_order:
            raise ValueError("Detection sample Image Order does not match its image.")
        if sample.source_file != image.original_filename:
            raise ValueError("Detection sample Original Filename does not match its image.")
        for field_name, value in (
            ("No. in Image", sample.no_in_image),
            ("Batch No.", sample.batch_no),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError("Detection export {} must be a positive integer.".format(field_name))
        identities.append(DetectionExportIdentity(
            image_order=image.image_order,
            original_filename=image.original_filename,
            no_in_image=sample.no_in_image,
            batch_no=sample.batch_no,
        ))
    return tuple(identities)


_DEFAULT_INDEX = object()


@dataclass(init=False)
class BatchState:
    images: List[ImageItem]
    _current_image_index: Optional[int]
    last_batch_result: Optional[object]
    detection_scope: DetectionScope
    numbering_mode: NumberingMode
    regression_set: RegressionSet
    detection_run_token: Optional[int]

    def __init__(
        self,
        images=None,
        current_image_index=_DEFAULT_INDEX,
        last_batch_result=None,
        detection_scope=DetectionScope.CURRENT_IMAGE,
        numbering_mode=NumberingMode.PER_IMAGE,
        regression_set=None,
        detection_run_token=None,
    ):
        self.images = list(images) if images is not None else []
        self._current_image_index = None
        self.last_batch_result = last_batch_result
        self.detection_scope = detection_scope
        self.numbering_mode = numbering_mode
        self.regression_set = (
            regression_set
            if regression_set is not None
            else RegressionSet.missing()
        )
        if not isinstance(self.regression_set, RegressionSet):
            raise TypeError("regression_set must be a RegressionSet.")
        if detection_run_token is not None and (
            isinstance(detection_run_token, bool)
            or not isinstance(detection_run_token, int)
            or detection_run_token < 1
        ):
            raise ValueError("detection_run_token must be a positive integer or None.")
        self.detection_run_token = detection_run_token
        if current_image_index is _DEFAULT_INDEX:
            current_image_index = 0 if self.images else None
        self.current_image_index = current_image_index

    @property
    def current_image_index(self) -> Optional[int]:
        self._validate_current_image_index(self._current_image_index)
        return self._current_image_index

    @current_image_index.setter
    def current_image_index(self, value):
        self._validate_current_image_index(value)
        self._current_image_index = value

    def _validate_current_image_index(self, value):
        if not self.images:
            if value is not None:
                raise IndexError("An empty batch has no current image index.")
            return
        if (isinstance(value, bool) or not isinstance(value, int)
                or value < 0 or value >= len(self.images)):
            raise IndexError("Current image index is outside the batch.")

    @property
    def current_image(self) -> Optional[ImageItem]:
        index = self.current_image_index
        return None if index is None else self.images[index]

    @classmethod
    def from_paths(cls, paths: Iterable[os.PathLike]):
        return cls(images=build_image_items(paths))


@dataclass
class PairingResult:
    pairs: List[Tuple[Box, Box]] = field(default_factory=list)
    errors: List[SampleError] = field(default_factory=list)


def normalize_internal_path(path: os.PathLike) -> str:
    """Normalize a path lexically without requiring it to exist."""
    return os.path.normpath(os.path.abspath(os.fspath(path)))


def _natural_text_key(text: str):
    return tuple(
        (0, int(part), len(part)) if part.isdigit() else (1, part.casefold())
        for part in _NATURAL_PART.split(text)
        if part
    )


def natural_sort_paths(paths: Iterable[os.PathLike]) -> List[os.PathLike]:
    """Return inputs in deterministic natural filename order, preserving values."""
    indexed = list(enumerate(paths))

    def key(item):
        index, original = item
        raw = os.fspath(original)
        normalized = normalize_internal_path(raw)
        return (
            _natural_text_key(Path(raw).name),
            normalized.casefold(),
            normalized,
            index,
        )

    return [original for _, original in sorted(indexed, key=key)]


def build_image_items(paths: Iterable[os.PathLike]) -> List[ImageItem]:
    return [
        ImageItem(
            path=normalize_internal_path(path),
            original_filename=Path(os.fspath(path)).name,
            image_order=image_order,
        )
        for image_order, path in enumerate(natural_sort_paths(paths), start=1)
    ]


def _center(box: Box) -> Tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def _intersection_width(first: Box, second: Box) -> float:
    return max(0.0, min(first[2], second[2]) - max(first[0], second[0]))


def _candidate_indexes(liquid: Box, cuvettes: Sequence[Box]) -> List[int]:
    center_x, center_y = _center(liquid)
    contained = [
        index
        for index, box in enumerate(cuvettes)
        if box[0] <= center_x <= box[2] and box[1] <= center_y <= box[3]
    ]
    if contained:
        return contained
    return [
        index for index, box in enumerate(cuvettes)
        if _intersection_width(liquid, box) > 0.0
    ]


def pair_cuvettes_and_liquids(
    cuvettes: Sequence[Box], liquids: Sequence[Box], image_order=1, source_file=""
) -> PairingResult:
    """Pair only mutually unique geometry candidates and record every rejection."""
    cuvettes = sorted(tuple(map(float, box)) for box in cuvettes)
    liquids = sorted(tuple(map(float, box)) for box in liquids)
    liquid_candidates = [_candidate_indexes(liquid, cuvettes) for liquid in liquids]
    cuvette_candidates = [
        [liquid_index for liquid_index, candidates in enumerate(liquid_candidates)
         if cuvette_index in candidates]
        for cuvette_index in range(len(cuvettes))
    ]
    result = PairingResult()
    paired_cuvettes = set()
    paired_liquids = set()

    for liquid_index, candidates in enumerate(liquid_candidates):
        if len(candidates) != 1:
            continue
        cuvette_index = candidates[0]
        if len(cuvette_candidates[cuvette_index]) == 1:
            result.pairs.append((cuvettes[cuvette_index], liquids[liquid_index]))
            paired_cuvettes.add(cuvette_index)
            paired_liquids.add(liquid_index)

    for index, liquid in enumerate(liquids):
        if index in paired_liquids:
            continue
        candidates = liquid_candidates[index]
        error_type = (SampleErrorType.UNMATCHED_LIQUID if not candidates
                      else SampleErrorType.AMBIGUOUS_LIQUID)
        reason = ("No cuvette candidate for liquid." if not candidates
                  else "Liquid has multiple candidates or competes for a cuvette.")
        result.errors.append(SampleError(
            image_order, source_file, error_type, reason,
            [liquid] + sorted(cuvettes[i] for i in candidates),
            related_cuvette_boxes=sorted(cuvettes[i] for i in candidates),
            related_liquid_boxes=[liquid],
            position=_center(liquid),
        ))

    for index, cuvette in enumerate(cuvettes):
        if index in paired_cuvettes:
            continue
        candidates = cuvette_candidates[index]
        error_type = (SampleErrorType.UNMATCHED_CUVETTE if not candidates
                      else SampleErrorType.AMBIGUOUS_CUVETTE)
        reason = ("No liquid candidate for cuvette." if not candidates
                  else "Cuvette has multiple candidates or is part of an ambiguity.")
        result.errors.append(SampleError(
            image_order, source_file, error_type, reason,
            [cuvette] + sorted(liquids[i] for i in candidates),
            related_cuvette_boxes=[cuvette],
            related_liquid_boxes=sorted(liquids[i] for i in candidates),
            position=_center(cuvette),
        ))

    result.pairs = sort_spatially(result.pairs, box_getter=lambda pair: pair[0])
    result.errors.sort(key=lambda error: (
        error.error_type.value,
        error.position[1] if error.position else float("inf"),
        error.position[0] if error.position else float("inf"),
        tuple(error.related_cuvette_boxes),
        tuple(error.related_liquid_boxes),
        error.reason,
    ))
    return result


def sort_spatially(items: Sequence, box_getter=lambda item: item.cuvette_box) -> List:
    """Sort adaptive rows top-to-bottom and each row left-to-right."""
    if not items:
        return []
    decorated = [(item, tuple(map(float, box_getter(item)))) for item in items]
    typical_height = median(max(0.0, box[3] - box[1]) for _, box in decorated)
    row_tolerance = typical_height * 0.5
    decorated.sort(key=lambda pair: (_center(pair[1])[1], _center(pair[1])[0], pair[1]))
    rows = []
    for item, box in decorated:
        center_x, center_y = _center(box)
        if not rows or center_y - rows[-1]["anchor_y"] > row_tolerance:
            rows.append({"anchor_y": center_y, "values": [(center_x, center_y, box, item)]})
        else:
            rows[-1]["values"].append((center_x, center_y, box, item))
    ordered = []
    for row in rows:
        row["values"].sort(key=lambda value: (value[0], value[1], value[2]))
        ordered.extend(value[-1] for value in row["values"])
    return ordered


def assign_image_numbers(samples: Sequence[SampleResult]) -> List[SampleResult]:
    ordered = sort_spatially(samples)
    for number, sample in enumerate(ordered, start=1):
        sample.no_in_image = number
    return ordered


def assign_batch_numbers(images: Sequence[ImageItem]) -> int:
    for image in images:
        for sample in image.samples:
            sample.batch_no = None
    batch_no = 0
    for image in sorted(images, key=lambda item: item.image_order):
        if image.status != ImageStatus.COMPLETED:
            continue
        image.samples = assign_image_numbers(image.samples)
        for sample in image.samples:
            batch_no += 1
            sample.batch_no = batch_no
    return batch_no
