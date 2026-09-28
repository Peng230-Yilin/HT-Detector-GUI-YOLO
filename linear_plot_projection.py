"""Immutable, Qt-independent data projection for future Linear plots."""

from collections.abc import Mapping
from dataclasses import dataclass
import math
import numbers
from typing import Optional, Tuple

from batch_state import (
    BatchState,
    ConcentrationStatus,
    ImageStatus,
    NumberingMode,
    RegressionSet,
)


_CHANNELS = ("R", "G", "B")
_PLOT_MODES = ("R", "G", "B", "RGB")
_PLOTTABLE_STATUSES = frozenset({
    ConcentrationStatus.IN_RANGE,
    ConcentrationStatus.BELOW_RANGE,
    ConcentrationStatus.ABOVE_RANGE,
    ConcentrationStatus.RANGE_UNAVAILABLE,
})
_CALIBRATION_KEYS = {
    "concentration": ("concentration", "Con."),
    "red": ("red", "Red"),
    "green": ("green", "Green"),
    "blue": ("blue", "Blue"),
    "included": ("included",),
}


def _is_finite_real(value):
    return (
        not isinstance(value, bool)
        and isinstance(value, numbers.Real)
        and math.isfinite(float(value))
    )


def _finite_float(value, name):
    if not _is_finite_real(value):
        raise ValueError("{} must be a finite real number.".format(name))
    return float(value)


def _optional_finite_float(value, name):
    return None if value is None else _finite_float(value, name)


def _optional_positive_int(value, name):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("{} must be a positive integer or None.".format(name))
    return value


def _mapping_field(mapping, field):
    for key in _CALIBRATION_KEYS[field]:
        if key in mapping:
            return mapping[key]
    raise ValueError("Calibration sample field {} is missing.".format(field))


@dataclass(frozen=True)
class CalibrationSample:
    concentration: float
    red: float
    green: float
    blue: float
    included: bool

    def __post_init__(self):
        object.__setattr__(
            self,
            "concentration",
            _finite_float(self.concentration, "Calibration concentration"),
        )
        for field in ("red", "green", "blue"):
            object.__setattr__(
                self,
                field,
                _finite_float(
                    getattr(self, field),
                    "Calibration {} intensity".format(field.upper()),
                ),
            )
        if not isinstance(self.included, bool):
            raise ValueError("Calibration included must be bool.")


@dataclass(frozen=True)
class CalibrationPoint:
    concentration: float
    intensity: float

    def __post_init__(self):
        object.__setattr__(
            self,
            "concentration",
            _finite_float(self.concentration, "Calibration point concentration"),
        )
        object.__setattr__(
            self,
            "intensity",
            _finite_float(self.intensity, "Calibration point intensity"),
        )


@dataclass(frozen=True)
class CalibrationCurveView:
    channel: str
    slope: Optional[float]
    intercept: Optional[float]
    r_squared: Optional[float]
    range_min: Optional[float]
    range_max: Optional[float]
    calibration_points: Tuple[CalibrationPoint, ...]
    unavailable_status: Optional[ConcentrationStatus]

    def __post_init__(self):
        if self.channel not in _CHANNELS:
            raise ValueError("Calibration curve channel must be R, G, or B.")
        for field in (
            "slope",
            "intercept",
            "r_squared",
            "range_min",
            "range_max",
        ):
            object.__setattr__(
                self,
                field,
                _optional_finite_float(
                    getattr(self, field),
                    "Calibration curve {}".format(field),
                ),
            )
        if not isinstance(self.calibration_points, tuple):
            raise TypeError("calibration_points must be a tuple.")
        if not all(
            isinstance(point, CalibrationPoint)
            for point in self.calibration_points
        ):
            raise TypeError(
                "calibration_points must contain only CalibrationPoint values."
            )
        if self.unavailable_status is not None and not isinstance(
            self.unavailable_status, ConcentrationStatus
        ):
            raise TypeError(
                "Calibration curve unavailable_status is invalid."
            )


@dataclass(frozen=True)
class DetectionPlotPoint:
    image_order: int
    original_filename: str
    no_in_image: Optional[int]
    batch_no: Optional[int]
    channel: str
    concentration: float
    intensity: float
    status: ConcentrationStatus
    is_current_image: bool
    run_token: Optional[int] = None
    alpha: float = 1.0

    def __post_init__(self):
        if (
            isinstance(self.image_order, bool)
            or not isinstance(self.image_order, int)
            or self.image_order < 1
        ):
            raise ValueError("Detection point image_order must be positive.")
        if not isinstance(self.original_filename, str):
            raise TypeError("Detection point original_filename must be a string.")
        object.__setattr__(
            self,
            "no_in_image",
            _optional_positive_int(self.no_in_image, "Detection point no_in_image"),
        )
        object.__setattr__(
            self,
            "batch_no",
            _optional_positive_int(self.batch_no, "Detection point batch_no"),
        )
        if self.channel not in _CHANNELS:
            raise ValueError("Detection point channel must be R, G, or B.")
        object.__setattr__(
            self,
            "concentration",
            _finite_float(self.concentration, "Detection point concentration"),
        )
        object.__setattr__(
            self,
            "intensity",
            _finite_float(self.intensity, "Detection point intensity"),
        )
        if self.status not in _PLOTTABLE_STATUSES:
            raise ValueError("Detection point status is not plottable.")
        if not isinstance(self.is_current_image, bool):
            raise TypeError("Detection point is_current_image must be bool.")
        object.__setattr__(
            self,
            "run_token",
            _optional_positive_int(self.run_token, "Detection point run_token"),
        )
        object.__setattr__(
            self,
            "alpha",
            _finite_float(self.alpha, "Detection point alpha"),
        )
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("Detection point alpha must be between 0 and 1.")


@dataclass(frozen=True)
class LinearPlotProjection:
    regression_revision: str
    plot_mode: str
    curves: Tuple[CalibrationCurveView, ...]
    detection_points: Tuple[DetectionPlotPoint, ...]
    numbering_mode: Optional[NumberingMode]
    selected_image_order: Optional[int]
    calibration_revision: Optional[str]
    detection_regression_revision: Optional[str]
    calibration_points_stale: bool
    detection_points_stale: bool
    detection_run_token: Optional[int] = None

    def __post_init__(self):
        if not isinstance(self.regression_revision, str) or not self.regression_revision:
            raise ValueError("regression_revision must be a non-empty string.")
        if self.plot_mode not in _PLOT_MODES:
            raise ValueError("plot_mode must be R, G, B, or RGB.")
        if not isinstance(self.curves, tuple) or not all(
            isinstance(curve, CalibrationCurveView) for curve in self.curves
        ):
            raise TypeError("curves must be a tuple of CalibrationCurveView values.")
        if not isinstance(self.detection_points, tuple) or not all(
            isinstance(point, DetectionPlotPoint)
            for point in self.detection_points
        ):
            raise TypeError(
                "detection_points must be a tuple of DetectionPlotPoint values."
            )
        if self.numbering_mode is not None and not isinstance(
            self.numbering_mode, NumberingMode
        ):
            raise TypeError("numbering_mode must be NumberingMode or None.")
        _optional_positive_int(self.selected_image_order, "selected_image_order")
        for field in ("calibration_revision", "detection_regression_revision"):
            value = getattr(self, field)
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError("{} must be a non-empty string or None.".format(field))
        if not isinstance(self.calibration_points_stale, bool):
            raise TypeError("calibration_points_stale must be bool.")
        if not isinstance(self.detection_points_stale, bool):
            raise TypeError("detection_points_stale must be bool.")
        _optional_positive_int(self.detection_run_token, "detection_run_token")


def _snapshot_calibration_samples(calibration_samples):
    if calibration_samples is None:
        return ()
    snapshot = []
    for value in calibration_samples:
        if isinstance(value, CalibrationSample):
            sample = CalibrationSample(
                value.concentration,
                value.red,
                value.green,
                value.blue,
                value.included,
            )
        elif isinstance(value, Mapping):
            sample = CalibrationSample(
                concentration=_mapping_field(value, "concentration"),
                red=_mapping_field(value, "red"),
                green=_mapping_field(value, "green"),
                blue=_mapping_field(value, "blue"),
                included=_mapping_field(value, "included"),
            )
        else:
            raise TypeError(
                "calibration_samples must contain CalibrationSample or mapping values."
            )
        snapshot.append(sample)
    return tuple(snapshot)


def _channels_for_mode(plot_mode):
    if plot_mode not in _PLOT_MODES:
        raise ValueError("plot_mode must be R, G, B, or RGB.")
    return _CHANNELS if plot_mode == "RGB" else (plot_mode,)


def _calibration_points(samples, channel):
    intensity_field = {"R": "red", "G": "green", "B": "blue"}[channel]
    return tuple(
        CalibrationPoint(sample.concentration, getattr(sample, intensity_field))
        for sample in samples
        if sample.included
    )


def _curve_view(regression_set, calibration_samples, channel, points_stale):
    model = regression_set.model(channel)
    return CalibrationCurveView(
        channel=channel,
        slope=model.slope,
        intercept=model.intercept,
        r_squared=model.r_squared,
        range_min=model.valid_min,
        range_max=model.valid_max,
        calibration_points=(
            ()
            if points_stale
            else _calibration_points(calibration_samples, channel)
        ),
        unavailable_status=model.unavailable_status,
    )


def _point_values(sample, channel):
    suffix = channel.lower()
    return (
        getattr(sample, "con_{}".format(suffix)),
        getattr(sample, {"R": "red", "G": "green", "B": "blue"}[channel]),
        getattr(sample, "status_{}".format(suffix)),
    )


def _detection_points(batch_state, channels, selected_image_order):
    points = []
    run_token = getattr(batch_state, "detection_run_token", None)
    numbering_mode = NumberingMode(batch_state.numbering_mode)
    for image in sorted(batch_state.images, key=lambda item: item.image_order):
        if image.status != ImageStatus.COMPLETED or not image.samples:
            continue
        if (
            numbering_mode == NumberingMode.PER_IMAGE
            and image.image_order != selected_image_order
        ):
            continue
        for sample in image.samples:
            sample_run_token = getattr(sample, "detection_run_token", None)
            if run_token is not None and sample_run_token != run_token:
                continue
            for channel in channels:
                concentration, intensity, status = _point_values(sample, channel)
                if (
                    status not in _PLOTTABLE_STATUSES
                    or not _is_finite_real(concentration)
                    or not _is_finite_real(intensity)
                ):
                    continue
                points.append(DetectionPlotPoint(
                    image_order=image.image_order,
                    original_filename=str(image.original_filename),
                    no_in_image=sample.no_in_image,
                    batch_no=sample.batch_no,
                    channel=channel,
                    concentration=concentration,
                    intensity=intensity,
                    status=status,
                    is_current_image=image.image_order == selected_image_order,
                    run_token=sample_run_token,
                    alpha=(
                        1.0
                        if image.image_order == selected_image_order
                        or numbering_mode == NumberingMode.PER_IMAGE
                        else 0.30
                    ),
                ))
    return tuple(points)


def build_linear_plot_projection(
    *,
    active_regression_set,
    calibration_samples,
    calibration_revision,
    batch_state,
    plot_mode,
    selected_image_order,
):
    """Create an immutable plotting snapshot without inference or I/O."""

    if not isinstance(active_regression_set, RegressionSet):
        raise TypeError("active_regression_set must be a RegressionSet.")
    channels = _channels_for_mode(plot_mode)
    selected_image_order = _optional_positive_int(
        selected_image_order, "selected_image_order"
    )
    samples = _snapshot_calibration_samples(calibration_samples)
    calibration_points_stale = (
        calibration_revision != active_regression_set.revision
    )
    curves = tuple(
        _curve_view(
            active_regression_set,
            samples,
            channel,
            calibration_points_stale,
        )
        for channel in channels
    )

    numbering_mode = None
    detection_revision = None
    detection_points_stale = False
    detection_points = ()
    detection_run_token = None
    if batch_state is not None:
        if not isinstance(batch_state, BatchState):
            raise TypeError("batch_state must be BatchState or None.")
        if not isinstance(batch_state.regression_set, RegressionSet):
            raise TypeError("batch_state.regression_set must be a RegressionSet.")
        numbering_mode = NumberingMode(batch_state.numbering_mode)
        detection_run_token = getattr(batch_state, "detection_run_token", None)
        _optional_positive_int(detection_run_token, "detection_run_token")
        detection_revision = batch_state.regression_set.revision
        detection_points_stale = (
            detection_revision != active_regression_set.revision
        )
        if not detection_points_stale:
            detection_points = _detection_points(
                batch_state, channels, selected_image_order
            )

    return LinearPlotProjection(
        regression_revision=active_regression_set.revision,
        plot_mode=plot_mode,
        curves=curves,
        detection_points=detection_points,
        numbering_mode=numbering_mode,
        selected_image_order=selected_image_order,
        calibration_revision=calibration_revision,
        detection_regression_revision=detection_revision,
        calibration_points_stale=calibration_points_stale,
        detection_points_stale=detection_points_stale,
        detection_run_token=detection_run_token,
    )


__all__ = (
    "CalibrationSample",
    "CalibrationPoint",
    "CalibrationCurveView",
    "DetectionPlotPoint",
    "LinearPlotProjection",
    "build_linear_plot_projection",
)
