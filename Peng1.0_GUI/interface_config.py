import copy
import importlib.util
import json
import math
import numbers
import sys
from pathlib import Path

from PySide6.QtCore import QIODevice, QSaveFile, QStandardPaths


CONFIG_VERSION = 1
CONFIG_DIRECTORY_NAME = "HT-Detector-GUI-YOLO"
CONFIG_FILE_NAME = "interface_settings.json"
ALLOWED_SETTINGS = (
    "detect_confidence",
    "show_confidence",
    "x0_ratio",
    "y0_ratio",
    "x1_ratio",
    "y1_ratio",
    "color_channel",
    "rgb_calculate_accuracy",
    "rgb_display_accuracy",
    "con_display_accuracy",
    "Order_Con_R_G_B",
    "con_list",
    "linear_formula_point_matrix",
)
TEXT_ORDERS = (
    "ConRGB",
    "ConRBG",
    "ConGRB",
    "ConGBR",
    "ConBRG",
    "ConBGR",
)

_DEFAULT_SETTINGS = None


def load_interface_module():
    repository_root = Path(__file__).resolve().parent.parent
    interface_path = repository_root / "HT-Detector_Peng" / "interface.py"
    if not interface_path.is_file():
        raise RuntimeError(
            "The HT-Detector_Peng interface module was not found: {}".format(
                interface_path
            )
        )

    existing_module = sys.modules.get("interface")
    existing_file = getattr(existing_module, "__file__", None)
    if existing_file:
        try:
            if Path(existing_file).resolve() == interface_path.resolve():
                return existing_module
        except (OSError, RuntimeError):
            pass

    spec = importlib.util.spec_from_file_location("interface", interface_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(
            "Could not create an import specification for: {}".format(interface_path)
        )

    module = importlib.util.module_from_spec(spec)
    had_existing_module = "interface" in sys.modules
    sys.modules["interface"] = module
    try:
        spec.loader.exec_module(module)
    except Exception as error:
        if had_existing_module:
            sys.modules["interface"] = existing_module
        else:
            sys.modules.pop("interface", None)
        raise RuntimeError(
            "Failed to load HT-Detector_Peng interface module from {}: {}".format(
                interface_path, error
            )
        ) from error
    return module


def configuration_path():
    config_root = QStandardPaths.writableLocation(
        QStandardPaths.StandardLocation.AppConfigLocation
    )
    if not config_root:
        raise RuntimeError("Qt did not provide an application configuration directory.")
    return Path(config_root) / CONFIG_DIRECTORY_NAME / CONFIG_FILE_NAME


def default_settings(interface_module=None):
    global _DEFAULT_SETTINGS
    if _DEFAULT_SETTINGS is None:
        module = interface_module or load_interface_module()
        raw_defaults = {
            name: copy.deepcopy(getattr(module, name)) for name in ALLOWED_SETTINGS
        }
        _DEFAULT_SETTINGS = validate_settings(raw_defaults)
    return copy.deepcopy(_DEFAULT_SETTINGS)


def _require_number(settings, name, minimum, maximum):
    value = settings[name]
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError("{} must be numeric and must not be bool.".format(name))
    value = float(value)
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(
            "{} must be a finite number from {} to {}.".format(
                name, minimum, maximum
            )
        )


def _require_integer(settings, name, minimum, maximum):
    value = settings[name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("{} must be an integer and must not be bool.".format(name))
    if not minimum <= value <= maximum:
        raise ValueError("{} must be from {} to {}.".format(name, minimum, maximum))


def validate_settings(settings):
    if not isinstance(settings, dict):
        raise ValueError("Settings must be an object.")
    missing = set(ALLOWED_SETTINGS) - set(settings)
    unknown = set(settings) - set(ALLOWED_SETTINGS)
    if missing:
        raise ValueError("Missing setting(s): {}.".format(", ".join(sorted(missing))))
    if unknown:
        raise ValueError("Unknown setting(s): {}.".format(", ".join(sorted(unknown))))

    validated = copy.deepcopy(settings)

    _require_number(settings, "detect_confidence", 0.001, 1.0)
    if not isinstance(settings["show_confidence"], bool):
        raise ValueError("show_confidence must be bool.")
    for name in ("x0_ratio", "y0_ratio", "x1_ratio", "y1_ratio"):
        _require_number(settings, name, 0.0, 1.0)
    if settings["x1_ratio"] - settings["x0_ratio"] < 0.0001:
        raise ValueError("x1_ratio - x0_ratio must be at least 0.0001.")
    if settings["y1_ratio"] - settings["y0_ratio"] < 0.0001:
        raise ValueError("y1_ratio - y0_ratio must be at least 0.0001.")
    if settings["color_channel"] not in ("R", "G", "B"):
        raise ValueError("color_channel must be R, G, or B.")
    _require_integer(settings, "rgb_calculate_accuracy", 0, 16)
    _require_integer(settings, "rgb_display_accuracy", 0, 6)
    _require_integer(settings, "con_display_accuracy", 0, 6)
    if settings["Order_Con_R_G_B"] not in TEXT_ORDERS:
        raise ValueError("Order_Con_R_G_B is invalid.")

    concentrations = settings["con_list"]
    if not isinstance(concentrations, list):
        raise ValueError("con_list must be a list.")
    if not 2 <= len(concentrations) <= 100:
        raise ValueError("con_list must contain from 2 to 100 concentrations.")
    normalized_concentrations = []
    for index, value in enumerate(concentrations, start=1):
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise ValueError(
                "con_list item {} must be numeric and must not be bool.".format(index)
            )
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("con_list item {} must be finite.".format(index))
        if value < 0.0:
            raise ValueError("con_list item {} must be at least 0.".format(index))
        if normalized_concentrations and value <= normalized_concentrations[-1]:
            raise ValueError("con_list must be strictly increasing from left to right.")
        normalized_concentrations.append(value)
    validated["con_list"] = normalized_concentrations

    point_matrix = settings["linear_formula_point_matrix"]
    if not isinstance(point_matrix, list):
        raise ValueError("linear_formula_point_matrix must be a list.")
    if len(point_matrix) != len(normalized_concentrations):
        raise ValueError(
            "linear_formula_point_matrix length must match con_list length."
        )
    normalized_matrix = []
    for index, value in enumerate(point_matrix, start=1):
        if isinstance(value, bool):
            normalized_matrix.append(value)
        elif isinstance(value, int) and value in (0, 1):
            normalized_matrix.append(bool(value))
        else:
            raise ValueError(
                "linear_formula_point_matrix item {} must be bool or integer 0/1.".format(
                    index
                )
            )
    if sum(normalized_matrix) < 2:
        raise ValueError("At least two calibration points must be used in regression.")
    validated["linear_formula_point_matrix"] = normalized_matrix
    return validated


def apply_settings(interface_module, settings):
    validated = validate_settings(settings)
    for name in ALLOWED_SETTINGS:
        setattr(interface_module, name, copy.deepcopy(validated[name]))
    return validated


def load_effective_settings(config_file=None, apply_to_module=True):
    interface_module = load_interface_module()
    defaults = default_settings(interface_module)
    path = Path(config_file) if config_file is not None else configuration_path()
    if not path.is_file():
        settings = apply_settings(interface_module, defaults) if apply_to_module else defaults
        return settings, [], interface_module

    try:
        with path.open("r", encoding="utf-8") as stream:
            document = json.load(stream)
        if not isinstance(document, dict):
            raise ValueError("The configuration root must be an object.")
        if set(document) - {"version", "overrides"}:
            raise ValueError("The configuration contains unknown top-level fields.")
        if document.get("version") != CONFIG_VERSION:
            raise ValueError("Unsupported configuration version.")
        overrides = document.get("overrides")
        if not isinstance(overrides, dict):
            raise ValueError("overrides must be an object.")
        unknown = set(overrides) - set(ALLOWED_SETTINGS)
        if unknown:
            raise ValueError(
                "Unknown override setting(s): {}.".format(", ".join(sorted(unknown)))
            )
        merged = copy.deepcopy(defaults)
        merged.update(overrides)
        validated = validate_settings(merged)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        warning = "Invalid interface settings; using interface.py defaults: {}".format(
            error
        )
        settings = apply_settings(interface_module, defaults) if apply_to_module else defaults
        return settings, [warning], interface_module
    settings = (
        apply_settings(interface_module, validated) if apply_to_module else validated
    )
    return settings, [], interface_module


def save_settings(settings, config_file=None):
    validated = validate_settings(settings)
    defaults = default_settings()
    overrides = {
        name: copy.deepcopy(validated[name])
        for name in ALLOWED_SETTINGS
        if validated[name] != defaults[name]
    }
    document = {"version": CONFIG_VERSION, "overrides": overrides}
    encoded = (json.dumps(document, ensure_ascii=False, indent=2) + "\n").encode(
        "utf-8"
    )
    path = Path(config_file) if config_file is not None else configuration_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise RuntimeError(
            "Could not create the configuration directory {}: {}".format(
                path.parent, error
            )
        ) from error

    save_file = QSaveFile(str(path))
    if not save_file.open(QIODevice.OpenModeFlag.WriteOnly):
        raise RuntimeError(
            "Could not open the configuration file for writing: {}".format(path)
        )
    if save_file.write(encoded) != len(encoded):
        save_file.cancelWriting()
        raise RuntimeError("Could not write the complete configuration file: {}".format(path))
    if not save_file.commit():
        raise RuntimeError("Could not atomically save the configuration file: {}".format(path))
    return path, document
