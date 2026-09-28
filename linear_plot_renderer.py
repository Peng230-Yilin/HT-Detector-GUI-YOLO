"""Qt-independent renderer for immutable Linear plot projections."""

from linear_plot_projection import LinearPlotProjection


_CHANNEL_ORDER = ("R", "G", "B")
_CHANNEL_COLORS = {"R": "red", "G": "green", "B": "blue"}
_OUT_OF_RANGE_COLORS = {"R": "#8B0000", "G": "#006400", "B": "#00008B"}
_REGRESSION_COLORS = {"R": "#E57373", "G": "#66BB6A", "B": "#64B5F6"}
RGB_SUMMARY_COLOR = "#666666"
DETECTED_SUMMARY_MARKER = "D"
OUT_OF_RANGE_SUMMARY_MARKER = "x"
DETECTED_MARKER_SIZE = 50
OUT_OF_RANGE_MARKER_SIZE = 65
STANDARD_MARKER_SIZE = 36
DETECTED_EDGE_LINEWIDTH = 1.5
TITLE_FONT_SIZE = 13
SUBTITLE_FONT_SIZE = 9
AXIS_LABEL_FONT_SIZE = 10
TICK_FONT_SIZE = 9
LEGEND_FONT_SIZE = 9


def _compact_number(value, significant_digits=4):
    value = float(value)
    if value == 0.0:
        return "0"
    text = format(value, ".{}g".format(significant_digits))
    if "e" in text:
        mantissa, exponent = text.split("e")
        text = "{}e{:+d}".format(mantissa, int(exponent))
    return text


def _equation(slope, intercept):
    intercept = float(intercept)
    sign = "+" if intercept >= 0.0 else "-"
    return "y = {}x {} {}".format(
        _compact_number(slope),
        sign,
        _compact_number(abs(intercept)),
    )


def _line_domain(curve):
    if curve.range_min is not None and curve.range_max is not None:
        lower = curve.range_min
        upper = curve.range_max
        if lower == upper:
            margin = max(abs(lower) * 0.05, 0.5)
            return (lower - margin, upper + margin)
        return (lower, upper)
    concentrations = tuple(
        point.concentration for point in curve.calibration_points
    )
    if concentrations:
        lower = min(concentrations)
        upper = max(concentrations)
    elif curve.range_min is not None and curve.range_max is not None:
        lower = curve.range_min
        upper = curve.range_max
    else:
        return None
    if lower == upper:
        margin = max(abs(lower) * 0.05, 0.5)
        lower -= margin
        upper += margin
    return (lower, upper)


def _curve_label(curve):
    r_squared = 0.0 if float(curve.r_squared) == 0.0 else curve.r_squared
    return "{}; R² = {:.3f}".format(
        _equation(curve.slope, curve.intercept),
        r_squared,
    )


def _status_name(point):
    return getattr(point.status, "value", str(point.status))


def _is_out_of_range(point):
    return _status_name(point) in ("BELOW_RANGE", "ABOVE_RANGE")


def _sample_key(point):
    return (
        point.run_token,
        point.image_order,
        point.original_filename,
        point.no_in_image,
        point.batch_no,
    )


def _unique_sample_count(points):
    return len({_sample_key(point) for point in points})


def _number_for_point(point, numbering_mode):
    mode = getattr(numbering_mode, "value", numbering_mode)
    return point.batch_no if mode == "continuous" else point.no_in_image


def _draw_detection_points(axes, points, numbering_mode):
    for channel in _CHANNEL_ORDER:
        channel_points = tuple(point for point in points if point.channel == channel)
        for out_of_range in (False, True):
            styled = tuple(
                point for point in channel_points
                if _is_out_of_range(point) is out_of_range
            )
            for alpha in dict.fromkeys(point.alpha for point in styled):
                group = tuple(point for point in styled if point.alpha == alpha)
                if not group:
                    continue
                marker_style = {}
                if not out_of_range:
                    marker_style = {
                        "edgecolors": _OUT_OF_RANGE_COLORS[channel],
                        "linewidths": DETECTED_EDGE_LINEWIDTH,
                    }
                axes.scatter(
                    tuple(point.concentration for point in group),
                    tuple(point.intensity for point in group),
                    color=(
                        _OUT_OF_RANGE_COLORS[channel]
                        if out_of_range else _CHANNEL_COLORS[channel]
                    ),
                    marker=(
                        OUT_OF_RANGE_SUMMARY_MARKER
                        if out_of_range else DETECTED_SUMMARY_MARKER
                    ),
                    s=(
                        OUT_OF_RANGE_MARKER_SIZE
                        if out_of_range else DETECTED_MARKER_SIZE
                    ),
                    alpha=alpha,
                    label="_nolegend_",
                    zorder=4,
                    **marker_style
                )

    samples = {}
    for point in points:
        samples.setdefault(_sample_key(point), []).append(point)
    channel_order = {channel: index for index, channel in enumerate(_CHANNEL_ORDER)}
    for sample_points in samples.values():
        ordered = sorted(sample_points, key=lambda point: channel_order[point.channel])
        out_of_range = [point for point in ordered if _is_out_of_range(point)]
        anchor = out_of_range[0] if out_of_range else ordered[0]
        number = _number_for_point(anchor, numbering_mode)
        if number is None:
            continue
        axes.annotate(
            str(number),
            (anchor.concentration, anchor.intensity),
            xytext=(5, 5),
            textcoords="offset points",
            color=(
                _OUT_OF_RANGE_COLORS[anchor.channel]
                if _is_out_of_range(anchor) else _CHANNEL_COLORS[anchor.channel]
            ),
            fontsize=LEGEND_FONT_SIZE,
            alpha=anchor.alpha,
            zorder=5,
        )


def _draw_extrapolation(axes, curve, points, color):
    if (
        curve.slope is None
        or curve.intercept is None
        or curve.range_min is None
        or curve.range_max is None
    ):
        return
    for status, boundary, select in (
        ("BELOW_RANGE", curve.range_min, min),
        ("ABOVE_RANGE", curve.range_max, max),
    ):
        concentrations = tuple(
            point.concentration for point in points if _status_name(point) == status
        )
        if not concentrations:
            continue
        endpoint = select(concentrations)
        domain = ((endpoint, boundary) if status == "BELOW_RANGE" else (boundary, endpoint))
        axes.plot(
            domain,
            tuple(curve.slope * value + curve.intercept for value in domain),
            color=color,
            linewidth=2.2,
            linestyle="--",
            label="_nolegend_",
            zorder=2,
        )


def render_linear_plot(axes, projection):
    """Render one immutable projection without drawing or saving its canvas."""

    if not isinstance(projection, LinearPlotProjection):
        raise TypeError("projection must be a LinearPlotProjection.")

    channels = _CHANNEL_ORDER if projection.plot_mode == "RGB" else (
        projection.plot_mode,
    )
    curves = {curve.channel: curve for curve in projection.curves}
    missing = tuple(channel for channel in channels if channel not in curves)
    if missing:
        raise ValueError(
            "Projection is missing curve channel(s): {}.".format(
                ", ".join(missing)
            )
        )

    axes.clear()
    rgb_mode = projection.plot_mode == "RGB"
    detection_points = tuple(
        point for point in projection.detection_points if point.channel in channels
    )
    points_by_channel = {
        channel: tuple(point for point in detection_points if point.channel == channel)
        for channel in channels
    }
    for channel in channels:
        curve = curves[channel]
        color = _CHANNEL_COLORS[channel]
        regression_color = _REGRESSION_COLORS[channel]
        calibration_x = tuple(
            point.concentration for point in curve.calibration_points
        )
        calibration_y = tuple(
            point.intensity for point in curve.calibration_points
        )
        if calibration_x:
            axes.scatter(
                calibration_x,
                calibration_y,
                color=regression_color,
                edgecolors=regression_color,
                marker="o",
                s=STANDARD_MARKER_SIZE,
                label="_nolegend_",
                zorder=3,
            )

        domain = _line_domain(curve)
        if (
            curve.slope is not None
            and curve.intercept is not None
            and curve.r_squared is not None
            and domain is not None
        ):
            line_y = tuple(
                curve.slope * concentration + curve.intercept
                for concentration in domain
            )
            axes.plot(
                domain,
                line_y,
                color=regression_color,
                linewidth=2.2,
                linestyle="-",
                label="_nolegend_",
                zorder=2,
            )
            _draw_extrapolation(
                axes,
                curve,
                points_by_channel[channel],
                regression_color,
            )

    _draw_detection_points(axes, detection_points, projection.numbering_mode)
    normal_points = tuple(
        point for point in detection_points if not _is_out_of_range(point)
    )
    detected_count = _unique_sample_count(
        detection_points if rgb_mode else normal_points
    )
    out_of_range_count = _unique_sample_count(tuple(
        point for point in detection_points if _is_out_of_range(point)
    ))
    if detected_count:
        detected_style = {}
        if not rgb_mode:
            detected_style = {
                "edgecolors": _OUT_OF_RANGE_COLORS[projection.plot_mode],
                "linewidths": DETECTED_EDGE_LINEWIDTH,
            }
        axes.scatter(
            (), (),
            color=(RGB_SUMMARY_COLOR if rgb_mode else _CHANNEL_COLORS[projection.plot_mode]),
            marker=DETECTED_SUMMARY_MARKER,
            s=DETECTED_MARKER_SIZE,
            label="Detected (n={})".format(detected_count),
            zorder=4,
            **detected_style
        )
    if out_of_range_count:
        axes.scatter(
            (), (),
            color=(
                RGB_SUMMARY_COLOR if rgb_mode
                else _OUT_OF_RANGE_COLORS[projection.plot_mode]
            ),
            marker=OUT_OF_RANGE_SUMMARY_MARKER,
            s=OUT_OF_RANGE_MARKER_SIZE,
            label="Out of range (n={})".format(out_of_range_count),
            zorder=4,
        )

    if rgb_mode:
        title = "Linear Regression – RGB"
        ylabel = "Intensity"
    else:
        title = "Linear Regression – {}".format(projection.plot_mode)
        ylabel = "{} Intensity".format(projection.plot_mode)
    if rgb_mode:
        axes.set_title(title, fontsize=TITLE_FONT_SIZE, y=1.18, pad=0)
    else:
        axes.set_title(title, fontsize=TITLE_FONT_SIZE, y=1.10, pad=0)
        curve = curves[projection.plot_mode]
        if (
            curve.slope is not None
            and curve.intercept is not None
            and curve.r_squared is not None
        ):
            axes.text(
                0.5,
                1.02,
                _curve_label(curve),
                transform=axes.transAxes,
                ha="center",
                va="bottom",
                fontsize=SUBTITLE_FONT_SIZE,
            )
    axes.set_xlabel("Concentration", fontsize=AXIS_LABEL_FONT_SIZE)
    axes.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONT_SIZE)
    axes.tick_params(axis="both", labelsize=TICK_FONT_SIZE)
    axes.grid(True, alpha=0.3)
    if rgb_mode and detected_count:
        axes.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.08),
            ncol=2 if out_of_range_count else 1,
            frameon=False,
            fontsize=LEGEND_FONT_SIZE,
            handlelength=1.2,
            columnspacing=0.8,
            handletextpad=0.4,
            borderaxespad=0.0,
        )
    elif not rgb_mode and (detected_count or out_of_range_count):
        axes.legend(
            loc="best",
            frameon=True,
            framealpha=0.85,
            fontsize=LEGEND_FONT_SIZE,
            handlelength=1.4,
            labelspacing=0.35,
            borderpad=0.4,
            handletextpad=0.5,
            borderaxespad=0.5,
        )


__all__ = (
    "RGB_SUMMARY_COLOR",
    "DETECTED_SUMMARY_MARKER",
    "OUT_OF_RANGE_SUMMARY_MARKER",
    "render_linear_plot",
)
