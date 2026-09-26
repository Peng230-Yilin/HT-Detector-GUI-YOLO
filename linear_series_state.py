"""Pure, transactional state primitives for Linear Image Series.

The module deliberately imports no GUI, worker, imaging, or numerical stack.
Externally visible domain records are immutable. LinearSeriesState owns the
only mutable pointer and commits fully validated immutable state snapshots.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from decimal import Decimal, DecimalException
from enum import Enum
import math
import os
from pathlib import Path
import re
from typing import Any, ClassVar, Optional, Tuple


Box = Tuple[float, float, float, float]
_MISSING = object()
_ABSENT = object()
_NATURAL_PART = re.compile(r"(\d+)")
_DECIMAL_TEXT = re.compile(
    r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$",
    re.ASCII,
)


class LinearSeriesPhase(str, Enum):
    """Lifecycle of an unconfirmed Linear Image Series draft."""

    IDLE = "idle"
    EXTRACTING = "extracting"
    RUNNING = "extracting"
    MAPPING = "mapping"
    READY = "ready"
    CONFIRMED = "ready"
    CANCELLED = "cancelled"
    FAILED = "failed"


class LinearImageStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class LinearSampleStatus(str, Enum):
    VALID = "valid"


SeriesPhase = LinearSeriesPhase
ImageStatus = LinearImageStatus
SampleStatus = LinearSampleStatus


class LinearSeriesInvariantError(ValueError):
    """A malformed state graph or illegal state transition."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = str(code)


@dataclass(frozen=True)
class RegressionSampleIdentity:
    sample_key: str
    original_file_name: str
    series_number: int

    def __post_init__(self):
        _require_nonempty_text(self.sample_key, "sample_key")
        _require_nonempty_text(
            self.original_file_name,
            "original_file_name",
        )
        _require_positive_int(self.series_number, "series_number")

    @property
    def image_name(self):
        return self.original_file_name

    @property
    def no(self):
        return self.series_number


class RegressionValidationError(ValueError):
    """A regression validation failure with stable structured details."""

    def __init__(
        self,
        code,
        message,
        sample_keys=(),
        conflict_groups=(),
        sample_details=(),
    ):
        super().__init__(message)
        self.code = str(code)
        self.sample_keys = tuple(sample_keys)
        self.conflict_groups = tuple(tuple(group) for group in conflict_groups)
        details = []
        for detail in sample_details:
            if not isinstance(detail, RegressionSampleIdentity):
                detail = RegressionSampleIdentity(*detail)
            details.append(detail)
        self.sample_details = tuple(details)


def _invariant(code, message):
    return LinearSeriesInvariantError(code, message)


def _require_positive_int(value, label):
    if type(value) is not int or value <= 0:
        raise _invariant(
            "invalid_{}".format(label),
            "{} must be a positive integer and bool is not accepted.".format(label),
        )
    return value


def _require_nonempty_text(value, label):
    if type(value) is not str or not value:
        raise _invariant(
            "invalid_{}".format(label),
            "{} must be a non-empty string.".format(label),
        )
    return value


def _finite_numeric_value(value, label):
    if isinstance(value, (bool, str, bytes, bytearray)):
        raise ValueError("{} must be a finite numeric value".format(label))
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("{} must be a finite numeric value".format(label)) from error
    if not math.isfinite(converted):
        raise ValueError("{} must be a finite numeric value".format(label))
    return converted


def _validated_rgb(red, green, blue):
    try:
        return (
            _finite_numeric_value(red, "Red"),
            _finite_numeric_value(green, "Green"),
            _finite_numeric_value(blue, "Blue"),
        )
    except ValueError as error:
        raise _invariant("invalid_rgb", str(error)) from error


def _freeze_box(value, label):
    if value is None:
        return None
    if type(value) not in (list, tuple) or len(value) != 4:
        raise _invariant(
            "invalid_geometry",
            "{} must be an ordered numeric sequence of length 4.".format(label),
        )
    try:
        return tuple(
            _finite_numeric_value(channel, "{} coordinate".format(label))
            for channel in value
        )
    except ValueError as error:
        raise _invariant("invalid_geometry", str(error)) from error


@dataclass(frozen=True)
class LinearSampleError:
    """An immutable extraction error that can never receive a No.x."""

    image_order: int
    original_file_name: str
    reason: str
    normalized_path: str = ""
    error_type: str = "sample_error"
    spatial_order: Optional[int] = None
    error_key: str = ""
    series_number: ClassVar[None] = None

    def __post_init__(self):
        _require_positive_int(self.image_order, "image_order")
        _require_nonempty_text(self.original_file_name, "original_file_name")
        if type(self.normalized_path) is not str:
            raise _invariant(
                "invalid_normalized_path", "normalized_path must be a string."
            )
        _require_nonempty_text(self.reason, "error_reason")
        _require_nonempty_text(self.error_type, "error_type")
        if self.spatial_order is not None:
            _require_positive_int(self.spatial_order, "spatial_order")
        if type(self.error_key) is not str:
            raise _invariant("invalid_error_key", "error_key must be a string.")

    @property
    def image_name(self):
        return self.original_file_name

    @property
    def original_filename(self):
        return self.original_file_name

    @property
    def source_file(self):
        return self.original_file_name

    @property
    def path(self):
        return self.normalized_path

    @property
    def number(self):
        return None

    @property
    def no(self):
        return None

    @property
    def sample_key(self):
        return None


@dataclass(frozen=True)
class LinearSampleItem:
    """One immutable valid RGB calibration sample."""

    sample_key: str
    image_order: int
    normalized_path: str
    original_file_name: str
    spatial_order: int
    red: float
    green: float
    blue: float
    series_number: Optional[int] = None
    included: bool = True
    concentration_text: str = ""
    status: LinearSampleStatus = LinearSampleStatus.VALID
    cuvette_box: Optional[Box] = None
    liquid_box: Optional[Box] = None
    roi_box: Optional[Box] = None

    def __post_init__(self):
        _require_nonempty_text(self.sample_key, "sample_key")
        _require_positive_int(self.image_order, "image_order")
        _require_nonempty_text(self.normalized_path, "normalized_path")
        _require_nonempty_text(self.original_file_name, "original_file_name")
        _require_positive_int(self.spatial_order, "spatial_order")
        red, green, blue = _validated_rgb(self.red, self.green, self.blue)
        object.__setattr__(self, "red", red)
        object.__setattr__(self, "green", green)
        object.__setattr__(self, "blue", blue)
        if self.series_number is not None:
            _require_positive_int(self.series_number, "series_number")
        if type(self.included) is not bool:
            raise _invariant("invalid_included", "included must be an exact bool.")
        if type(self.concentration_text) is not str:
            raise _invariant(
                "invalid_concentration_text",
                "concentration_text must be a string.",
            )
        if type(self.status) is not LinearSampleStatus:
            raise _invariant(
                "invalid_sample_status",
                "Sample status must be LinearSampleStatus.VALID.",
            )
        object.__setattr__(
            self, "cuvette_box", _freeze_box(self.cuvette_box, "cuvette_box")
        )
        object.__setattr__(
            self, "liquid_box", _freeze_box(self.liquid_box, "liquid_box")
        )
        object.__setattr__(self, "roi_box", _freeze_box(self.roi_box, "roi_box"))

    @property
    def image_name(self):
        return self.original_file_name

    @property
    def original_filename(self):
        return self.original_file_name

    @property
    def source_file(self):
        return self.original_file_name

    @property
    def path(self):
        return self.normalized_path

    @property
    def number(self):
        return self.series_number

    @property
    def no(self):
        return self.series_number

    @property
    def no_x(self):
        return self.series_number

    @property
    def include(self):
        return self.included

    @property
    def concentration(self):
        return self.concentration_text


@dataclass(frozen=True, init=False)
class LinearImageItem:
    """One immutable image task, including repeated selections of one path."""

    normalized_path: str
    original_file_name: str
    image_order: int
    selection_order: int
    image_key: str
    status: LinearImageStatus
    _samples: Tuple[LinearSampleItem, ...] = field(repr=False)
    _errors: Tuple[LinearSampleError, ...] = field(repr=False)
    failure_reason: Optional[str]

    def __init__(
        self,
        normalized_path,
        original_file_name,
        image_order,
        selection_order,
        image_key,
        status=LinearImageStatus.PENDING,
        samples=(),
        errors=(),
        failure_reason=None,
    ):
        _require_nonempty_text(normalized_path, "normalized_path")
        _require_nonempty_text(original_file_name, "original_file_name")
        _require_positive_int(image_order, "image_order")
        _require_positive_int(selection_order, "selection_order")
        _require_nonempty_text(image_key, "image_key")
        if type(status) is not LinearImageStatus:
            raise _invariant(
                "invalid_image_status", "status must be a LinearImageStatus value."
            )
        if type(samples) not in (list, tuple):
            raise _invariant(
                "invalid_samples_collection", "samples must be a list or tuple."
            )
        if type(errors) not in (list, tuple):
            raise _invariant(
                "invalid_errors_collection", "errors must be a list or tuple."
            )
        sample_tuple = tuple(samples)
        error_tuple = tuple(errors)
        if any(type(sample) is not LinearSampleItem for sample in sample_tuple):
            raise _invariant(
                "invalid_sample_item", "Every sample must be a LinearSampleItem."
            )
        if any(type(error) is not LinearSampleError for error in error_tuple):
            raise _invariant(
                "invalid_error_item", "Every error must be a LinearSampleError."
            )
        if failure_reason is not None and type(failure_reason) is not str:
            raise _invariant(
                "invalid_failure_reason",
                "failure_reason must be a string or None.",
            )
        object.__setattr__(self, "normalized_path", normalized_path)
        object.__setattr__(self, "original_file_name", original_file_name)
        object.__setattr__(self, "image_order", image_order)
        object.__setattr__(self, "selection_order", selection_order)
        object.__setattr__(self, "image_key", image_key)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "_samples", sample_tuple)
        object.__setattr__(self, "_errors", error_tuple)
        object.__setattr__(self, "failure_reason", failure_reason)

    @property
    def samples(self):
        return self._samples

    @property
    def errors(self):
        return self._errors

    @property
    def path(self):
        return self.normalized_path

    @property
    def original_filename(self):
        return self.original_file_name

    @property
    def image_name(self):
        return self.original_file_name

    def _updated(
        self,
        *,
        status=_MISSING,
        samples=_MISSING,
        errors=_MISSING,
        failure_reason=_MISSING,
    ):
        return LinearImageItem(
            normalized_path=self.normalized_path,
            original_file_name=self.original_file_name,
            image_order=self.image_order,
            selection_order=self.selection_order,
            image_key=self.image_key,
            status=self.status if status is _MISSING else status,
            samples=self._samples if samples is _MISSING else samples,
            errors=self._errors if errors is _MISSING else errors,
            failure_reason=(
                self.failure_reason if failure_reason is _MISSING else failure_reason
            ),
        )


@dataclass(frozen=True)
class RegressionInput:
    """Deeply immutable, validated values ready for numerical regression."""

    sample_keys: Tuple[str, ...]
    image_names: Tuple[str, ...]
    normalized_paths: Tuple[str, ...]
    numbers: Tuple[int, ...]
    concentration_texts: Tuple[str, ...]
    normalized_concentration_texts: Tuple[str, ...]
    decimal_concentrations: Tuple[Decimal, ...]
    concentrations: Tuple[float, ...]
    red_values: Tuple[float, ...]
    green_values: Tuple[float, ...]
    blue_values: Tuple[float, ...]

    def __post_init__(self):
        tuple_fields = (
            "sample_keys",
            "image_names",
            "normalized_paths",
            "numbers",
            "concentration_texts",
            "normalized_concentration_texts",
            "decimal_concentrations",
            "concentrations",
            "red_values",
            "green_values",
            "blue_values",
        )
        for name in tuple_fields:
            if type(getattr(self, name)) is not tuple:
                raise ValueError("RegressionInput.{} must be a tuple.".format(name))
        lengths = {len(getattr(self, name)) for name in tuple_fields}
        if len(lengths) != 1:
            raise ValueError("RegressionInput fields must have equal lengths.")
        if len(self.sample_keys) < 2:
            raise ValueError("RegressionInput requires at least two rows.")

        for name in (
            "sample_keys",
            "image_names",
            "normalized_paths",
            "concentration_texts",
            "normalized_concentration_texts",
        ):
            if any(type(value) is not str for value in getattr(self, name)):
                raise ValueError(
                    "RegressionInput.{} must contain strings.".format(name)
                )
        for name in ("sample_keys", "image_names", "normalized_paths"):
            if any(not value for value in getattr(self, name)):
                raise ValueError(
                    "RegressionInput.{} must not contain empty strings.".format(
                        name
                    )
                )
        if len(set(self.sample_keys)) != len(self.sample_keys):
            raise ValueError("RegressionInput sample keys must be unique.")
        if any(type(value) is not int or value <= 0 for value in self.numbers):
            raise ValueError(
                "RegressionInput numbers must be positive exact integers."
            )
        if (
            tuple(sorted(self.numbers)) != self.numbers
            or len(set(self.numbers)) != len(self.numbers)
        ):
            raise ValueError(
                "RegressionInput numbers must be unique and increasing."
            )
        if any(type(value) is not Decimal for value in self.decimal_concentrations):
            raise ValueError(
                "RegressionInput exact concentrations must be Decimals."
            )
        for name in ("concentrations", "red_values", "green_values", "blue_values"):
            values = getattr(self, name)
            if any(
                type(value) is not float or not math.isfinite(value)
                for value in values
            ):
                raise ValueError(
                    "RegressionInput.{} must contain finite Python floats.".format(
                        name
                    )
                )

        for index, decimal_value in enumerate(self.decimal_concentrations):
            if not decimal_value.is_finite() or decimal_value < 0:
                raise ValueError(
                    "RegressionInput Decimal concentrations must be finite."
                )
            parsed = parse_concentration(self.concentration_texts[index])
            if parsed != decimal_value:
                raise ValueError(
                    "Raw concentration text does not match its Decimal."
                )
            canonical_decimal = _canonical_decimal(decimal_value)
            if self.normalized_concentration_texts[index] != str(
                canonical_decimal
            ):
                raise ValueError(
                    "Normalized concentration text does not match its Decimal."
                )
            converted = float(decimal_value)
            if not math.isfinite(converted) or (
                decimal_value != 0 and converted == 0.0
            ):
                raise ValueError(
                    "RegressionInput Decimal is not safely representable."
                )
            if self.concentrations[index] != converted:
                raise ValueError(
                    "Float concentration does not match its Decimal."
                )
        if len(set(self.decimal_concentrations)) != len(
            self.decimal_concentrations
        ):
            raise ValueError(
                "RegressionInput Decimal concentrations must be unique."
            )
        if len(set(self.concentrations)) != len(self.concentrations):
            raise ValueError(
                "RegressionInput float concentrations must be unique."
            )

    @property
    def included_count(self):
        return len(self.sample_keys)

    @property
    def raw_concentration_texts(self):
        return self.concentration_texts

    @property
    def canonical_concentration_texts(self):
        return self.normalized_concentration_texts

    @property
    def concentration_decimals(self):
        return self.decimal_concentrations

    @property
    def concentration_floats(self):
        return self.concentrations

    @property
    def red(self):
        return self.red_values

    @property
    def green(self):
        return self.green_values

    @property
    def blue(self):
        return self.blue_values


def normalize_internal_path(path):
    """Normalize a path lexically without requiring filesystem access."""

    raw = os.fspath(path)
    if not isinstance(raw, str):
        raw = os.fsdecode(raw)
    if not raw.strip():
        raise ValueError("Image path must not be empty.")
    return os.path.normpath(os.path.abspath(raw))


def _original_file_name(path):
    raw = os.fspath(path)
    if not isinstance(raw, str):
        raw = os.fsdecode(raw)
    native_name = Path(raw).name
    fallback_name = raw.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
    return native_name if native_name == fallback_name else fallback_name


def _natural_text_key(text):
    return tuple(
        (0, int(part), len(part)) if part.isdigit() else (1, part.casefold())
        for part in _NATURAL_PART.split(text)
        if part
    )


def _naturally_sorted_indexed_paths(paths):
    indexed = list(enumerate(paths, start=1))

    def key(item):
        selection_order, original = item
        normalized = normalize_internal_path(original)
        name = _original_file_name(original)
        return (
            _natural_text_key(name),
            normalized.casefold(),
            normalized,
            selection_order,
        )

    return sorted(indexed, key=key)


def natural_sort_paths(paths):
    return [path for _, path in _naturally_sorted_indexed_paths(paths)]


def make_image_key(image_order):
    _require_positive_int(image_order, "image_order")
    return "linear-image:{:06d}".format(image_order)


def make_sample_key(image_order, spatial_order):
    _require_positive_int(image_order, "image_order")
    _require_positive_int(spatial_order, "spatial_order")
    return "linear-sample:{:06d}:{:06d}".format(image_order, spatial_order)


def build_image_items(paths):
    items = []
    for image_order, (selection_order, path) in enumerate(
        _naturally_sorted_indexed_paths(paths), start=1
    ):
        items.append(
            LinearImageItem(
                normalized_path=normalize_internal_path(path),
                original_file_name=_original_file_name(path),
                image_order=image_order,
                selection_order=selection_order,
                image_key=make_image_key(image_order),
            )
        )
    return items


def _canonical_decimal(value):
    if value.is_zero():
        return Decimal((0, (0,), 0))
    sign, digits, exponent = value.as_tuple()
    digits = list(digits)
    while digits and digits[-1] == 0:
        digits.pop()
        exponent += 1
    return Decimal((sign, tuple(digits), exponent))


def parse_concentration(value):
    """Parse one finite non-negative concentration as an exact Decimal."""

    if isinstance(value, bool) or value is None:
        raise RegressionValidationError(
            "invalid_concentration", "Concentration must be a number."
        )
    if isinstance(value, Decimal):
        parsed = value
    else:
        try:
            text = str(value)
        except Exception as error:
            raise RegressionValidationError(
                "invalid_concentration",
                "Concentration could not be converted to text.",
            ) from error
        text = text.strip()
        if not text:
            raise RegressionValidationError(
                "empty_concentration", "Concentration must not be empty."
            )
        if _DECIMAL_TEXT.fullmatch(text) is None:
            raise RegressionValidationError(
                "invalid_concentration",
                "Concentration must be an integer, decimal, or "
                "scientific-notation number.",
            )
        try:
            parsed = Decimal(text)
        except (DecimalException, ValueError, TypeError, OverflowError) as error:
            raise RegressionValidationError(
                "invalid_concentration",
                "Concentration must be a valid number.",
            ) from error
    if not parsed.is_finite():
        raise RegressionValidationError(
            "non_finite_concentration", "Concentration must be finite."
        )
    if parsed < 0:
        raise RegressionValidationError(
            "negative_concentration",
            "Concentration must be greater than or equal to 0.",
        )
    try:
        return _canonical_decimal(parsed)
    except (DecimalException, ValueError, TypeError, OverflowError) as error:
        raise RegressionValidationError(
            "invalid_concentration",
            "Concentration could not be normalized.",
        ) from error


def normalize_concentration(value):
    return str(parse_concentration(value))


def _editable_concentration_text(value):
    if value is None:
        return ""
    if type(value) is str:
        return value
    try:
        return str(value)
    except Exception as error:
        raise RegressionValidationError(
            "invalid_concentration",
            "Concentration could not be converted to text.",
        ) from error


def _value_from(source, names, default=_MISSING):
    if isinstance(source, Mapping):
        for name in names:
            if name in source:
                return source[name]
    else:
        for name in names:
            try:
                return getattr(source, name)
            except AttributeError:
                continue
    if default is not _MISSING:
        return default
    raise KeyError(names[0])


def _rgb_from(source):
    rgb = _value_from(source, ("rgb", "RGB"), default=_ABSENT)
    if rgb is not _ABSENT:
        if type(rgb) not in (list, tuple) or len(rgb) != 3:
            raise ValueError(
                "RGB must be an ordered numeric list or tuple of length 3"
            )
        red, green, blue = rgb
    else:
        try:
            red = _value_from(source, ("red", "Red", "R"))
            green = _value_from(source, ("green", "Green", "G"))
            blue = _value_from(source, ("blue", "Blue", "B"))
        except KeyError as error:
            raise ValueError(
                "RGB sample is missing {}".format(error.args[0])
            ) from error
    return (
        _finite_numeric_value(red, "Red"),
        _finite_numeric_value(green, "Green"),
        _finite_numeric_value(blue, "Blue"),
    )


def _optional_box(source, names):
    value = _value_from(source, names, default=None)
    return _freeze_box(value, names[0])


def _sample_detail(sample):
    return "sample_key={}, Image Name={}, No.{}".format(
        sample.sample_key,
        sample.original_file_name,
        sample.series_number,
    )


def _sample_identity(sample):
    return RegressionSampleIdentity(
        sample.sample_key,
        sample.original_file_name,
        sample.series_number,
    )


def _format_conflicts(title, groups):
    rendered_groups = []
    for value, samples in groups:
        rendered_groups.append(
            "{} -> [{}]".format(
                value,
                "; ".join(_sample_detail(sample) for sample in samples),
            )
        )
    return "{}: {}".format(title, " | ".join(rendered_groups))


@dataclass(frozen=True)
class _LinearSeriesData:
    images: Tuple[LinearImageItem, ...]
    phase: LinearSeriesPhase
    last_confirmed_result: Any = field(compare=False)


class LinearSeriesState:
    """Owner of one immutable and fully validated state snapshot."""

    __slots__ = ("_data",)

    def __init__(
        self,
        images=None,
        phase=LinearSeriesPhase.IDLE,
        last_confirmed_result=None,
    ):
        if images is None:
            images = ()
        if type(images) not in (list, tuple):
            raise _invariant(
                "invalid_images_collection",
                "images must be a list or tuple.",
            )
        incoming = tuple(images)
        provisional = _LinearSeriesData(incoming, phase, last_confirmed_result)
        self._validate_data(
            provisional,
            check_numbering=False,
            allow_unsorted=True,
        )
        data = _LinearSeriesData(
            tuple(sorted(incoming, key=lambda item: item.image_order)),
            phase,
            last_confirmed_result,
        )
        self._validate_data(data)
        self._data = data

    @classmethod
    def from_paths(cls, paths, last_confirmed_result=None):
        return cls(
            images=build_image_items(paths),
            last_confirmed_result=last_confirmed_result,
        )

    def clone(self):
        self._assert_invariants()
        clone = object.__new__(LinearSeriesState)
        clone._data = self._data
        return clone

    def _adopt_validated(self, candidate):
        if type(candidate) is not LinearSeriesState:
            raise TypeError("candidate must be a LinearSeriesState")
        candidate._assert_invariants()
        self._data = candidate._data

    def validate(self):
        self._assert_invariants()
        return True

    @property
    def images(self):
        self._assert_invariants()
        return self._data.images

    @property
    def phase(self):
        self._assert_invariants()
        return self._data.phase

    @property
    def last_confirmed_result(self):
        self._assert_invariants()
        return self._data.last_confirmed_result

    @property
    def all_samples(self):
        self._assert_invariants()
        return self._all_samples_unchecked(self._data.images)

    @property
    def included_samples(self):
        self._assert_invariants()
        return tuple(
            sample
            for sample in self._all_samples_unchecked(self._data.images)
            if sample.included
        )

    @property
    def busy(self):
        self._assert_invariants()
        return self._data.phase == LinearSeriesPhase.EXTRACTING

    @property
    def all_terminal(self):
        self._assert_invariants()
        return bool(self._data.images) and all(
            image.status
            in (
                LinearImageStatus.COMPLETED,
                LinearImageStatus.FAILED,
                LinearImageStatus.CANCELLED,
            )
            for image in self._data.images
        )

    def remember_confirmed_result(self, result):
        self._assert_invariants()
        candidate = _LinearSeriesData(
            self._data.images,
            self._data.phase,
            result,
        )
        self._validate_data(candidate)
        self._data = candidate
        return result

    def image_for_order(self, image_order):
        _require_positive_int(image_order, "image_order")
        self._assert_invariants()
        for image in self._data.images:
            if image.image_order == image_order:
                return image
        raise KeyError("Unknown linear image_order: {}".format(image_order))

    def sample_for_key(self, sample_key):
        _require_nonempty_text(sample_key, "sample_key")
        self._assert_invariants()
        for sample in self._all_samples_unchecked(self._data.images):
            if sample.sample_key == sample_key:
                return sample
        raise KeyError("Unknown linear sample key: {}".format(sample_key))

    def begin_extraction(self):
        self._assert_invariants()
        if self._data.phase != LinearSeriesPhase.IDLE:
            raise RuntimeError("Linear series extraction has already begun.")
        if not self._data.images:
            return False
        self._commit(
            self._data.images,
            LinearSeriesPhase.EXTRACTING,
        )
        return True

    begin = begin_extraction

    def mark_image_processing(self, image_order):
        _require_positive_int(image_order, "image_order")
        self._assert_invariants()
        index, image = self._image_index(image_order)
        if image.status != LinearImageStatus.PENDING:
            raise RuntimeError("Linear image task is not pending.")
        if any(
            item.status == LinearImageStatus.PROCESSING
            for item in self._data.images
        ):
            raise RuntimeError("A Linear image task is already processing.")
        if self._data.phase not in (
            LinearSeriesPhase.IDLE,
            LinearSeriesPhase.EXTRACTING,
        ):
            raise RuntimeError("Linear series is not accepting image work.")
        candidate = list(self._data.images)
        candidate[index] = image._updated(
            status=LinearImageStatus.PROCESSING
        )
        self._commit(candidate, LinearSeriesPhase.EXTRACTING)
        return self._data.images[index]

    def accept_image_result(self, image_order, samples, errors=()):
        _require_positive_int(image_order, "image_order")
        self._assert_invariants()
        index, image = self._image_index(image_order)
        self._ensure_accepting_result(image)
        if type(samples) not in (list, tuple):
            raise TypeError("samples must be an ordered list or tuple")
        if type(errors) not in (list, tuple):
            raise TypeError("errors must be an ordered list or tuple")

        accepted = []
        accepted_errors = []
        for spatial_order, source in enumerate(samples, start=1):
            expected_key = make_sample_key(
                image.image_order,
                spatial_order,
            )
            self._validate_source_identity(
                image,
                source,
                sample_key=expected_key,
            )
            try:
                red, green, blue = _rgb_from(source)
            except ValueError as error:
                accepted_errors.append(
                    self._new_error(
                        image,
                        reason=str(error),
                        error_type="invalid_rgb",
                        spatial_order=spatial_order,
                        error_index=len(accepted_errors) + 1,
                    )
                )
                continue
            accepted.append(
                LinearSampleItem(
                    sample_key=expected_key,
                    image_order=image.image_order,
                    normalized_path=image.normalized_path,
                    original_file_name=image.original_file_name,
                    spatial_order=spatial_order,
                    red=red,
                    green=green,
                    blue=blue,
                    included=True,
                    concentration_text="",
                    cuvette_box=_optional_box(
                        source,
                        ("cuvette_box",),
                    ),
                    liquid_box=_optional_box(
                        source,
                        ("liquid_box",),
                    ),
                    roi_box=_optional_box(
                        source,
                        ("roi_box", "rgb_roi"),
                    ),
                )
            )

        for source in errors:
            accepted_errors.append(
                self._coerce_error(
                    image,
                    source,
                    len(accepted_errors) + 1,
                )
            )
        if not accepted and not accepted_errors:
            accepted_errors.append(
                self._new_error(
                    image,
                    reason="No valid RGB samples were received.",
                    error_type="image_failed",
                    error_index=1,
                )
            )

        replacement = image._updated(
            status=(
                LinearImageStatus.COMPLETED
                if accepted
                else LinearImageStatus.FAILED
            ),
            samples=tuple(accepted),
            errors=tuple(accepted_errors),
            failure_reason=(
                None if accepted else accepted_errors[0].reason
            ),
        )
        candidate_images = list(self._data.images)
        candidate_images[index] = replacement
        phase = self._phase_for_images(candidate_images)
        candidate_data = self._numbered_data(
            candidate_images,
            phase,
            self._data.last_confirmed_result,
        )
        self._data = candidate_data
        return self._data.images[index]

    accept_spatial_result = accept_image_result
    complete_image = accept_image_result

    def fail_image(self, image_order, reason, errors=()):
        _require_positive_int(image_order, "image_order")
        self._assert_invariants()
        index, image = self._image_index(image_order)
        self._ensure_accepting_result(image)
        if type(errors) not in (list, tuple):
            raise TypeError("errors must be an ordered list or tuple")
        normalized_reason = str(reason).strip()
        if not normalized_reason:
            normalized_reason = "Linear image extraction failed."
        accepted_errors = [
            self._new_error(
                image,
                reason=normalized_reason,
                error_type="image_failed",
                error_index=1,
            )
        ]
        for source in errors:
            accepted_errors.append(
                self._coerce_error(
                    image,
                    source,
                    len(accepted_errors) + 1,
                )
            )
        replacement = image._updated(
            status=LinearImageStatus.FAILED,
            samples=(),
            errors=tuple(accepted_errors),
            failure_reason=normalized_reason,
        )
        candidate_images = list(self._data.images)
        candidate_images[index] = replacement
        phase = self._phase_for_images(candidate_images)
        candidate_data = self._numbered_data(
            candidate_images,
            phase,
            self._data.last_confirmed_result,
        )
        self._data = candidate_data
        return self._data.images[index]

    def cancel(self):
        self._assert_invariants()
        if self._data.phase == LinearSeriesPhase.READY:
            raise RuntimeError(
                "A confirmed Linear series cannot be cancelled."
            )
        if self._data.phase == LinearSeriesPhase.CANCELLED:
            return False
        candidate_images = tuple(
            image._updated(
                status=LinearImageStatus.CANCELLED,
                samples=(),
                errors=(),
                failure_reason=None,
            )
            for image in self._data.images
        )
        self._commit(candidate_images, LinearSeriesPhase.CANCELLED)
        return True

    def confirm(self, result):
        self._assert_invariants()
        if self._data.phase == LinearSeriesPhase.CANCELLED:
            raise RuntimeError(
                "Cancelled or stale Linear series cannot be confirmed."
            )
        if self._data.phase != LinearSeriesPhase.MAPPING:
            raise RuntimeError(
                "Linear series can only be confirmed from mapping."
            )
        candidate = _LinearSeriesData(
            self._data.images,
            LinearSeriesPhase.READY,
            result,
        )
        self._validate_data(candidate)
        self._data = candidate
        return result

    def set_included(self, sample_key, included):
        _require_nonempty_text(sample_key, "sample_key")
        if type(included) is not bool:
            raise _invariant(
                "invalid_included",
                "included must be an exact bool.",
            )
        self._assert_invariants()
        self._ensure_mapping_edit()
        return self._replace_sample(
            sample_key,
            included=included,
        )

    def set_concentration(self, sample_key, value):
        _require_nonempty_text(sample_key, "sample_key")
        self._assert_invariants()
        self._ensure_mapping_edit()
        return self._replace_sample(
            sample_key,
            concentration_text=_editable_concentration_text(value),
        )

    def assign_series_numbers(self):
        """Atomically assign global numbers after full validation."""

        self._validate_data(self._data, check_numbering=False)
        candidate = self._numbered_data(
            self._data.images,
            self._data.phase,
            self._data.last_confirmed_result,
        )
        self._data = candidate
        return len(self._all_samples_unchecked(self._data.images))

    def build_regression_input(self):
        self._assert_invariants()
        if self._data.phase not in (
            LinearSeriesPhase.MAPPING,
            LinearSeriesPhase.READY,
        ):
            raise RegressionValidationError(
                "invalid_regression_phase",
                "Regression input is available only in mapping or ready phase.",
            )
        included = [
            sample
            for sample in self._all_samples_unchecked(self._data.images)
            if sample.included
        ]
        if len(included) < 2:
            details = tuple(
                _sample_identity(sample) for sample in included
            )
            message = (
                "At least 2 included samples are required for regression."
            )
            if included:
                message += " Relevant rows: [{}]".format(
                    "; ".join(
                        _sample_detail(sample) for sample in included
                    )
                )
            else:
                message += " Included rows: none."
            raise RegressionValidationError(
                "insufficient_included_samples",
                message,
                [sample.sample_key for sample in included],
                sample_details=details,
            )

        parsed_rows = []
        for sample in included:
            try:
                decimal_value = parse_concentration(
                    sample.concentration_text
                )
            except RegressionValidationError as error:
                raise RegressionValidationError(
                    error.code,
                    "{} ({})".format(
                        error,
                        _sample_detail(sample),
                    ),
                    [sample.sample_key],
                    sample_details=[_sample_identity(sample)],
                ) from error
            parsed_rows.append((sample, decimal_value))

        decimal_groups = {}
        for sample, value in parsed_rows:
            decimal_groups.setdefault(value, []).append(sample)
        duplicate_groups = [
            (value, samples)
            for value, samples in decimal_groups.items()
            if len(samples) > 1
        ]
        duplicate_groups.sort(
            key=lambda group: group[1][0].series_number
        )
        if duplicate_groups:
            message = _format_conflicts(
                "Duplicate included concentrations",
                duplicate_groups,
            )
            if len(decimal_groups) < 2:
                message += (
                    "; at least 2 different concentrations are required."
                )
            conflict_samples = [
                sample
                for _, samples in duplicate_groups
                for sample in samples
            ]
            raise RegressionValidationError(
                "duplicate_concentration",
                message,
                [sample.sample_key for sample in conflict_samples],
                [
                    [sample.sample_key for sample in samples]
                    for _, samples in duplicate_groups
                ],
                [
                    _sample_identity(sample)
                    for sample in conflict_samples
                ],
            )

        float_rows = []
        for sample, decimal_value in parsed_rows:
            try:
                float_value = float(decimal_value)
            except (OverflowError, ValueError) as error:
                raise RegressionValidationError(
                    "float_overflow",
                    "Concentration overflows finite float ({})".format(
                        _sample_detail(sample)
                    ),
                    [sample.sample_key],
                    sample_details=[_sample_identity(sample)],
                ) from error
            if not math.isfinite(float_value):
                raise RegressionValidationError(
                    "float_overflow",
                    "Concentration overflows finite float ({})".format(
                        _sample_detail(sample)
                    ),
                    [sample.sample_key],
                    sample_details=[_sample_identity(sample)],
                )
            if decimal_value != 0 and float_value == 0.0:
                raise RegressionValidationError(
                    "float_underflow",
                    "Non-zero concentration underflows to float zero "
                    "({})".format(_sample_detail(sample)),
                    [sample.sample_key],
                    sample_details=[_sample_identity(sample)],
                )
            float_rows.append(
                (sample, decimal_value, float_value)
            )

        float_groups = {}
        for sample, decimal_value, float_value in float_rows:
            float_groups.setdefault(float_value, []).append(
                (sample, decimal_value)
            )
        folded_groups = []
        for float_value, rows in float_groups.items():
            if len(
                {decimal_value for _, decimal_value in rows}
            ) > 1:
                folded_groups.append(
                    (
                        float_value,
                        [sample for sample, _ in rows],
                    )
                )
        folded_groups.sort(
            key=lambda group: group[1][0].series_number
        )
        if folded_groups:
            conflict_samples = [
                sample
                for _, samples in folded_groups
                for sample in samples
            ]
            raise RegressionValidationError(
                "float_collision",
                _format_conflicts(
                    "Different Decimal concentrations collapse "
                    "to the same float",
                    folded_groups,
                ),
                [sample.sample_key for sample in conflict_samples],
                [
                    [sample.sample_key for sample in samples]
                    for _, samples in folded_groups
                ],
                [
                    _sample_identity(sample)
                    for sample in conflict_samples
                ],
            )

        return RegressionInput(
            sample_keys=tuple(
                sample.sample_key
                for sample, _, _ in float_rows
            ),
            image_names=tuple(
                sample.original_file_name
                for sample, _, _ in float_rows
            ),
            normalized_paths=tuple(
                sample.normalized_path
                for sample, _, _ in float_rows
            ),
            numbers=tuple(
                sample.series_number
                for sample, _, _ in float_rows
            ),
            concentration_texts=tuple(
                sample.concentration_text
                for sample, _, _ in float_rows
            ),
            normalized_concentration_texts=tuple(
                str(decimal_value)
                for _, decimal_value, _ in float_rows
            ),
            decimal_concentrations=tuple(
                decimal_value
                for _, decimal_value, _ in float_rows
            ),
            concentrations=tuple(
                float_value
                for _, _, float_value in float_rows
            ),
            red_values=tuple(
                sample.red for sample, _, _ in float_rows
            ),
            green_values=tuple(
                sample.green for sample, _, _ in float_rows
            ),
            blue_values=tuple(
                sample.blue for sample, _, _ in float_rows
            ),
        )

    regression_input = build_regression_input
    validate_regression_input = build_regression_input

    def _ensure_mapping_edit(self):
        if self._data.phase != LinearSeriesPhase.MAPPING:
            raise RuntimeError(
                "Include and concentration can be edited only in mapping."
            )

    def _replace_sample(self, sample_key, **changes):
        candidate_images = list(self._data.images)
        for image_index, image in enumerate(self._data.images):
            for sample_index, sample in enumerate(image.samples):
                if sample.sample_key != sample_key:
                    continue
                samples = list(image.samples)
                samples[sample_index] = replace(sample, **changes)
                candidate_images[image_index] = image._updated(
                    samples=tuple(samples)
                )
                self._commit(
                    candidate_images,
                    self._data.phase,
                )
                return self._data.images[
                    image_index
                ].samples[sample_index]
        raise KeyError(
            "Unknown linear sample key: {}".format(sample_key)
        )

    def _image_index(self, image_order):
        for index, image in enumerate(self._data.images):
            if image.image_order == image_order:
                return index, image
        raise KeyError(
            "Unknown linear image_order: {}".format(image_order)
        )

    def _ensure_accepting_result(self, image):
        if self._data.phase == LinearSeriesPhase.CANCELLED:
            raise RuntimeError(
                "Cancelled Linear series rejects image results."
            )
        if self._data.phase != LinearSeriesPhase.EXTRACTING:
            raise RuntimeError(
                "Linear series is not accepting image results."
            )
        if image.status != LinearImageStatus.PROCESSING:
            raise RuntimeError(
                "Linear image task is not processing."
            )

    @staticmethod
    def _new_error(
        image,
        reason,
        error_type="sample_error",
        spatial_order=None,
        error_index=1,
    ):
        return LinearSampleError(
            image_order=image.image_order,
            original_file_name=image.original_file_name,
            reason=str(reason),
            normalized_path=image.normalized_path,
            error_type=str(error_type),
            spatial_order=spatial_order,
            error_key="linear-error:{:06d}:{:06d}".format(
                image.image_order,
                error_index,
            ),
        )

    def _coerce_error(self, image, source, error_index):
        expected_key = "linear-error:{:06d}:{:06d}".format(
            image.image_order,
            error_index,
        )
        self._validate_source_identity(
            image,
            source,
            error_key=expected_key,
        )
        try:
            reason = _value_from(
                source,
                ("reason", "message"),
            )
        except KeyError:
            if type(source) is str:
                reason = source
            else:
                raise _invariant(
                    "missing_error_reason",
                    "Worker error is missing a reason.",
                )
        error_type = _value_from(
            source,
            ("error_type", "type"),
            default="sample_error",
        )
        if isinstance(error_type, Enum):
            error_type = error_type.value
        spatial_order = _value_from(
            source,
            ("spatial_order",),
            default=None,
        )
        return self._new_error(
            image,
            reason=str(reason),
            error_type=str(error_type),
            spatial_order=spatial_order,
            error_index=error_index,
        )

    @staticmethod
    def _validate_source_identity(
        image,
        source,
        sample_key=None,
        error_key=None,
    ):
        checks = (
            (
                ("image_order",),
                image.image_order,
                "image_order",
            ),
            (
                ("normalized_path", "path"),
                image.normalized_path,
                "normalized_path",
            ),
            (
                (
                    "original_file_name",
                    "original_filename",
                    "image_name",
                    "source_file",
                ),
                image.original_file_name,
                "original_file_name",
            ),
        )
        for names, expected, label in checks:
            for name in names:
                actual = _value_from(
                    source,
                    (name,),
                    default=_ABSENT,
                )
                if actual is _ABSENT:
                    continue
                if (
                    type(actual) is not type(expected)
                    or actual != expected
                ):
                    raise _invariant(
                        "mismatched_{}".format(label),
                        "Worker result {} does not match the "
                        "active image.".format(label),
                    )
        if sample_key is not None:
            actual = _value_from(
                source,
                ("sample_key",),
                default=_ABSENT,
            )
            if (
                actual is not _ABSENT
                and actual not in (None, "")
                and (
                    type(actual) is not str
                    or actual != sample_key
                )
            ):
                raise _invariant(
                    "mismatched_sample_key",
                    "Worker sample key does not match the "
                    "confirmed spatial order.",
                )
        if error_key is not None:
            actual = _value_from(
                source,
                ("error_key",),
                default=_ABSENT,
            )
            if (
                actual is not _ABSENT
                and actual not in (None, "")
                and (
                    type(actual) is not str
                    or actual != error_key
                )
            ):
                raise _invariant(
                    "mismatched_error_key",
                    "Worker error key does not match its row.",
                )

    @staticmethod
    def _all_samples_unchecked(images):
        return tuple(
            sample
            for image in images
            for sample in image.samples
        )

    @staticmethod
    def _phase_for_images(images):
        images = tuple(images)
        if not images:
            return LinearSeriesPhase.IDLE
        if any(
            image.status
            in (
                LinearImageStatus.PENDING,
                LinearImageStatus.PROCESSING,
            )
            for image in images
        ):
            return LinearSeriesPhase.EXTRACTING
        if all(
            image.status == LinearImageStatus.CANCELLED
            for image in images
        ):
            return LinearSeriesPhase.CANCELLED
        if any(image.samples for image in images):
            return LinearSeriesPhase.MAPPING
        return LinearSeriesPhase.FAILED

    @classmethod
    def _numbered_data(
        cls,
        images,
        phase,
        last_confirmed_result,
    ):
        incoming = tuple(images)
        provisional = _LinearSeriesData(
            incoming,
            phase,
            last_confirmed_result,
        )
        cls._validate_data(
            provisional,
            check_numbering=False,
            allow_unsorted=True,
        )
        number = 0
        numbered_images = []
        for image in sorted(
            incoming,
            key=lambda item: item.image_order,
        ):
            if image.status != LinearImageStatus.COMPLETED:
                numbered_images.append(image)
                continue
            numbered_samples = []
            for sample in sorted(
                image.samples,
                key=lambda item: item.spatial_order,
            ):
                number += 1
                numbered_samples.append(
                    replace(
                        sample,
                        series_number=number,
                    )
                )
            numbered_images.append(
                image._updated(
                    samples=tuple(numbered_samples)
                )
            )
        candidate = _LinearSeriesData(
            tuple(numbered_images),
            phase,
            last_confirmed_result,
        )
        cls._validate_data(candidate)
        return candidate

    def _commit(
        self,
        images,
        phase,
        last_confirmed_result=_MISSING,
    ):
        if last_confirmed_result is _MISSING:
            last_confirmed_result = (
                self._data.last_confirmed_result
            )
        candidate = _LinearSeriesData(
            tuple(images),
            phase,
            last_confirmed_result,
        )
        self._validate_data(candidate)
        self._data = candidate

    def _assert_invariants(self):
        self._validate_data(self._data)

    @classmethod
    def _validate_data(
        cls,
        data,
        check_numbering=True,
        allow_unsorted=False,
    ):
        if type(data) is not _LinearSeriesData:
            raise _invariant(
                "invalid_state_data",
                "Linear series data snapshot is invalid.",
            )
        images = data.images
        phase = data.phase
        if type(images) is not tuple:
            raise _invariant(
                "invalid_images_collection",
                "Internal images must be a tuple.",
            )
        if type(phase) is not LinearSeriesPhase:
            raise _invariant(
                "invalid_phase",
                "phase must be a LinearSeriesPhase value.",
            )
        if any(
            type(image) is not LinearImageItem
            for image in images
        ):
            raise _invariant(
                "invalid_image_item",
                "Every image must be a LinearImageItem.",
            )

        orders = []
        image_keys = []
        selection_orders = []
        sample_keys = []
        error_keys = []
        processing_count = 0
        expected_number = 0

        for image in images:
            _require_positive_int(
                image.image_order,
                "image_order",
            )
            _require_positive_int(
                image.selection_order,
                "selection_order",
            )
            _require_nonempty_text(
                image.image_key,
                "image_key",
            )
            _require_nonempty_text(
                image.normalized_path,
                "normalized_path",
            )
            _require_nonempty_text(
                image.original_file_name,
                "original_file_name",
            )
            if type(image.status) is not LinearImageStatus:
                raise _invariant(
                    "invalid_image_status",
                    "Every image status must be valid.",
                )
            if type(image.samples) is not tuple:
                raise _invariant(
                    "invalid_samples_collection",
                    "Image samples must be a tuple.",
                )
            if type(image.errors) is not tuple:
                raise _invariant(
                    "invalid_errors_collection",
                    "Image errors must be a tuple.",
                )
            if (
                image.failure_reason is not None
                and type(image.failure_reason) is not str
            ):
                raise _invariant(
                    "invalid_failure_reason",
                    "failure_reason must be a string or None.",
                )

            orders.append(image.image_order)
            image_keys.append(image.image_key)
            selection_orders.append(image.selection_order)
            if image.status == LinearImageStatus.PROCESSING:
                processing_count += 1

            if image.status in (
                LinearImageStatus.PENDING,
                LinearImageStatus.PROCESSING,
                LinearImageStatus.CANCELLED,
            ):
                if (
                    image.samples
                    or image.errors
                    or image.failure_reason is not None
                ):
                    raise _invariant(
                        "nonterminal_image_has_results",
                        "Pending, processing, or cancelled images "
                        "cannot carry results.",
                    )
            elif image.status == LinearImageStatus.COMPLETED:
                if not image.samples:
                    raise _invariant(
                        "completed_without_samples",
                        "A completed image must contain at least "
                        "one valid sample.",
                    )
                if image.failure_reason is not None:
                    raise _invariant(
                        "completed_with_failure",
                        "A completed image cannot carry a "
                        "failure reason.",
                    )
            elif image.status == LinearImageStatus.FAILED:
                if image.samples:
                    raise _invariant(
                        "failed_with_samples",
                        "A failed image cannot contain samples.",
                    )
                if (
                    not image.errors
                    or not image.failure_reason
                ):
                    raise _invariant(
                        "failed_without_error",
                        "A failed image must contain an error "
                        "and failure reason.",
                    )

            spatial_orders = []
            for sample in image.samples:
                if type(sample) is not LinearSampleItem:
                    raise _invariant(
                        "invalid_sample_item",
                        "Every sample must be a LinearSampleItem.",
                    )
                _require_nonempty_text(
                    sample.sample_key,
                    "sample_key",
                )
                _require_positive_int(
                    sample.image_order,
                    "image_order",
                )
                _require_positive_int(
                    sample.spatial_order,
                    "spatial_order",
                )
                if (
                    sample.series_number is not None
                    and (
                        type(sample.series_number) is not int
                        or sample.series_number <= 0
                    )
                ):
                    raise _invariant(
                        "invalid_series_number",
                        "series_number must be a positive "
                        "exact integer.",
                    )
                if type(sample.included) is not bool:
                    raise _invariant(
                        "invalid_included",
                        "included must be an exact bool.",
                    )
                if type(sample.concentration_text) is not str:
                    raise _invariant(
                        "invalid_concentration_text",
                        "concentration_text must be a string.",
                    )
                if type(sample.status) is not LinearSampleStatus:
                    raise _invariant(
                        "invalid_sample_status",
                        "Every sample status must be valid.",
                    )
                _require_nonempty_text(
                    sample.normalized_path,
                    "normalized_path",
                )
                _require_nonempty_text(
                    sample.original_file_name,
                    "original_file_name",
                )
                if sample.image_order != image.image_order:
                    raise _invariant(
                        "mismatched_sample_image",
                        "Sample image_order is inconsistent.",
                    )
                if (
                    sample.normalized_path
                    != image.normalized_path
                    or sample.original_file_name
                    != image.original_file_name
                ):
                    raise _invariant(
                        "mismatched_sample_identity",
                        "Sample image identity is inconsistent.",
                    )
                rgb_values = (
                    sample.red,
                    sample.green,
                    sample.blue,
                )
                if any(
                    type(value) is not float
                    or not math.isfinite(value)
                    for value in rgb_values
                ):
                    raise _invariant(
                        "invalid_rgb",
                        "Stored RGB channels must be finite Python floats.",
                    )
                for box_name in (
                    "cuvette_box",
                    "liquid_box",
                    "roi_box",
                ):
                    box = getattr(sample, box_name)
                    if (
                        box is not None
                        and (
                            type(box) is not tuple
                            or len(box) != 4
                            or any(
                                type(value) is not float
                                or not math.isfinite(value)
                                for value in box
                            )
                        )
                    ):
                        raise _invariant(
                            "invalid_geometry",
                            "{} is not an immutable box.".format(
                                box_name
                            ),
                        )
                spatial_orders.append(
                    sample.spatial_order
                )
                sample_keys.append(sample.sample_key)
                expected_number += 1
                if (
                    check_numbering
                    and sample.series_number
                    != expected_number
                ):
                    raise _invariant(
                        "invalid_series_number",
                        "Sample numbers must be global, "
                        "increasing, and gap-free.",
                    )
            if (
                spatial_orders != sorted(spatial_orders)
                or len(spatial_orders)
                != len(set(spatial_orders))
            ):
                raise _invariant(
                    "invalid_spatial_order",
                    "Sample spatial orders must be unique "
                    "and increasing.",
                )

            for error in image.errors:
                if type(error) is not LinearSampleError:
                    raise _invariant(
                        "invalid_error_item",
                        "Every error must be a LinearSampleError.",
                    )
                _require_positive_int(
                    error.image_order,
                    "image_order",
                )
                _require_nonempty_text(
                    error.original_file_name,
                    "original_file_name",
                )
                if type(error.normalized_path) is not str:
                    raise _invariant(
                        "invalid_normalized_path",
                        "normalized_path must be a string.",
                    )
                _require_nonempty_text(
                    error.reason,
                    "error_reason",
                )
                _require_nonempty_text(
                    error.error_type,
                    "error_type",
                )
                _require_nonempty_text(
                    error.error_key,
                    "error_key",
                )
                if error.spatial_order is not None:
                    _require_positive_int(
                        error.spatial_order,
                        "spatial_order",
                    )
                if (
                    error.image_order != image.image_order
                    or error.normalized_path
                    != image.normalized_path
                    or error.original_file_name
                    != image.original_file_name
                ):
                    raise _invariant(
                        "mismatched_error_identity",
                        "Error image identity is inconsistent.",
                    )
                if error.series_number is not None:
                    raise _invariant(
                        "numbered_error",
                        "Error samples can never receive No.x.",
                    )
                error_keys.append(error.error_key)

        if len(orders) != len(set(orders)):
            raise _invariant(
                "duplicate_image_order",
                "Linear image_order values must be unique.",
            )
        if not allow_unsorted and orders != sorted(orders):
            raise _invariant(
                "unordered_images",
                "Images must be stored in image_order sequence.",
            )
        if len(image_keys) != len(set(image_keys)):
            raise _invariant(
                "duplicate_image_key",
                "Linear image keys must be unique.",
            )
        if len(selection_orders) != len(
            set(selection_orders)
        ):
            raise _invariant(
                "duplicate_selection_order",
                "Selection order values must be unique.",
            )
        if len(sample_keys) != len(set(sample_keys)):
            raise _invariant(
                "duplicate_sample_key",
                "Linear sample keys must be unique.",
            )
        if len(error_keys) != len(set(error_keys)):
            raise _invariant(
                "duplicate_error_key",
                "Linear error keys must be unique.",
            )
        if processing_count > 1:
            raise _invariant(
                "multiple_processing_images",
                "At most one image may be processing.",
            )

        statuses = tuple(
            image.status for image in images
        )
        if not images:
            if phase not in (
                LinearSeriesPhase.IDLE,
                LinearSeriesPhase.CANCELLED,
            ):
                raise _invariant(
                    "invalid_empty_phase",
                    "An empty series must be idle or cancelled.",
                )
            return
        if phase == LinearSeriesPhase.IDLE:
            if any(
                status != LinearImageStatus.PENDING
                for status in statuses
            ):
                raise _invariant(
                    "invalid_idle_state",
                    "An idle series may contain only "
                    "pending images.",
                )
        elif phase == LinearSeriesPhase.EXTRACTING:
            if not any(
                status
                in (
                    LinearImageStatus.PENDING,
                    LinearImageStatus.PROCESSING,
                )
                for status in statuses
            ):
                raise _invariant(
                    "invalid_extracting_state",
                    "An extracting series must contain "
                    "unfinished image work.",
                )
            if any(
                status == LinearImageStatus.CANCELLED
                for status in statuses
            ):
                raise _invariant(
                    "invalid_extracting_state",
                    "Cancelled images cannot be extracting.",
                )
        elif phase in (
            LinearSeriesPhase.MAPPING,
            LinearSeriesPhase.READY,
        ):
            if any(
                status
                not in (
                    LinearImageStatus.COMPLETED,
                    LinearImageStatus.FAILED,
                )
                for status in statuses
            ) or not sample_keys:
                raise _invariant(
                    "invalid_mapping_state",
                    "Mapping or ready state requires "
                    "terminal images and valid samples.",
                )
        elif phase == LinearSeriesPhase.FAILED:
            if any(
                status != LinearImageStatus.FAILED
                for status in statuses
            ) or sample_keys:
                raise _invariant(
                    "invalid_failed_state",
                    "Failed state requires every image to "
                    "be failed and no samples.",
                )
        elif phase == LinearSeriesPhase.CANCELLED:
            if any(
                status != LinearImageStatus.CANCELLED
                for status in statuses
            ):
                raise _invariant(
                    "invalid_cancelled_state",
                    "Cancelled state requires every image "
                    "to be cancelled.",
                )


def assign_series_numbers(state: LinearSeriesState) -> int:
    if not isinstance(state, LinearSeriesState):
        raise TypeError("assign_series_numbers requires LinearSeriesState")
    return state.assign_series_numbers()


__all__ = [
    "Box",
    "ImageStatus",
    "LinearImageItem",
    "LinearImageStatus",
    "LinearSampleError",
    "LinearSampleItem",
    "LinearSampleStatus",
    "LinearSeriesInvariantError",
    "LinearSeriesPhase",
    "LinearSeriesState",
    "RegressionInput",
    "RegressionSampleIdentity",
    "RegressionValidationError",
    "SampleStatus",
    "SeriesPhase",
    "assign_series_numbers",
    "build_image_items",
    "make_image_key",
    "make_sample_key",
    "natural_sort_paths",
    "normalize_concentration",
    "normalize_internal_path",
    "parse_concentration",
]
