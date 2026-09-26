import ast
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


class _FakeSignal:
    def connect(self, _callback):
        pass

    def emit(self, _value):
        pass


fake_qtcore = types.ModuleType("PySide6.QtCore")
fake_qtcore.QObject = type("QObject", (), {})
fake_qtcore.Signal = lambda *_args, **_kwargs: _FakeSignal()
fake_qtcore.Slot = lambda *_args, **_kwargs: (lambda function: function)
fake_qtcore.QIODevice = type("QIODevice", (), {})
fake_qtcore.QSaveFile = type("QSaveFile", (), {})
fake_qtcore.QStandardPaths = type("QStandardPaths", (), {})
fake_pyside6 = types.ModuleType("PySide6")
fake_pyside6.QtCore = fake_qtcore
sys.modules["PySide6"] = fake_pyside6
sys.modules["PySide6.QtCore"] = fake_qtcore

import yolo_detection_worker as worker_module  # noqa: E402
from batch_state import ConcentrationStatus, RegressionSet, SampleResult  # noqa: E402
from yolo_detection_worker import YoloDetectionWorker  # noqa: E402


def _formula(slope, intercept=0.0):
    return {
        "slope": slope,
        "intercept": intercept,
        "r": 0.9,
        "R2": 0.81,
        "p": 0.01,
        "std_err": 0.1,
    }


def _regression_set(formulas=None, valid_range=(0.0, 10.0)):
    return RegressionSet.from_formulas(
        formulas
        or {
            "R": _formula(1.0),
            "G": _formula(2.0),
            "B": _formula(4.0),
        },
        source_id="repair5-fake-regression",
        valid_ranges=valid_range,
    )


def _settings(channel="R"):
    return {
        "color_channel": channel,
        "Order_Con_R_G_B": "ConRGB",
        "show_confidence": False,
        "x0_ratio": 0.2,
        "y0_ratio": 0.2,
        "x1_ratio": 0.8,
        "y1_ratio": 0.8,
        "rgb_calculate_accuracy": 3,
        "rgb_display_accuracy": 2,
        "con_display_accuracy": 2,
    }


class _FakeTensor:
    def __init__(self, values):
        self._values = np.asarray(values)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._values


class _FakeBoxes:
    def __init__(self, coordinates, classes):
        self.xyxy = _FakeTensor(coordinates)
        self.cls = _FakeTensor(classes)
        self.conf = _FakeTensor([0.9] * len(classes))

    def __len__(self):
        return len(self.cls._values)


class _FakeResult:
    names = {0: "cuvette", 1: "liquid"}

    def __init__(self, coordinates, classes):
        self.orig_img = np.zeros((140, 240, 3), dtype=np.uint8)
        self.boxes = _FakeBoxes(coordinates, classes)


class _FakeCv2:
    FONT_HERSHEY_SIMPLEX = 0

    @staticmethod
    def rectangle(*_args, **_kwargs):
        pass

    @staticmethod
    def getTextSize(text, _font, _scale, _thickness):
        return (max(len(text), 1) * 8, 12), 3


class _WorkerHarness:
    def __init__(self):
        self.text_blocks = []

    @staticmethod
    def _class_ids(_names):
        return 0, 1

    @staticmethod
    def _draw_detection_label(*_args, **_kwargs):
        pass

    def _draw_text_block(self, _cv2, _image, text_lines, _anchor_x, _preferred_top):
        self.text_blocks.append(tuple(text for text, _color in text_lines))

    @staticmethod
    def _regression_formula(_concentrations, _values, channel):
        return _formula({"R": 1.0, "G": 2.0, "B": 3.0}[channel])


def _one_pair_result():
    return _FakeResult(
        [(10, 10, 50, 110), (15, 35, 45, 95)],
        [0, 1],
    )


def _two_pair_result():
    return _FakeResult(
        [
            (10, 10, 50, 110),
            (90, 10, 130, 110),
            (15, 35, 45, 95),
            (95, 35, 125, 95),
        ],
        [0, 0, 1, 1],
    )


def _concentration_label(harness):
    labels = [
        text
        for block in harness.text_blocks
        for text in block
        if text.startswith("Con.")
    ]
    if len(labels) != 1:
        raise AssertionError("Expected exactly one concentration label.")
    return labels[0]


class DetectionImageChannelLabelTests(unittest.TestCase):
    def _build_detection(self, channel, averages, regressions=None):
        harness = _WorkerHarness()
        with patch.object(
            worker_module, "_calculate_rgb_averages", return_value=averages
        ):
            payload = YoloDetectionWorker._build_payload(
                harness,
                _one_pair_result(),
                _settings(channel),
                _FakeCv2,
                source_path="fake/sample.png",
                regression_set=regressions or _regression_set(),
            )
        return harness, payload

    def test_r_g_b_labels_use_the_matching_authoritative_sample_fields(self):
        expected = {"R": 1.11, "G": 2.22, "B": 3.33}
        averages = (1.11, 4.44, 13.32)

        for channel in ("R", "G", "B"):
            with self.subTest(channel=channel):
                harness, payload = self._build_detection(channel, averages)
                sample = payload["sample_results"][0]
                selected_field = "con_{}".format(channel.lower())

                self.assertEqual(
                    (sample["con_r"], sample["con_g"], sample["con_b"]),
                    (1.11, 2.22, 3.33),
                )
                self.assertEqual(sample[selected_field], expected[channel])
                self.assertEqual(
                    _concentration_label(harness),
                    "Con.{}:{}".format(channel, expected[channel]),
                )

    def test_label_does_not_read_the_legacy_target_concentration_copy(self):
        original_legacy_target = SampleResult.legacy_target

        def poisoned_target(sample, concentration, display_number=None):
            target = original_legacy_target(sample, concentration, display_number)
            target["Con."] = 987654.0
            return target

        with patch.object(SampleResult, "legacy_target", new=poisoned_target):
            harness, payload = self._build_detection("R", (1.11, 4.44, 13.32))

        self.assertEqual(payload["targets"][0]["Con."], 987654.0)
        self.assertEqual(payload["sample_results"][0]["con_r"], 1.11)
        self.assertEqual(_concentration_label(harness), "Con.R:1.11")

    def test_negative_out_of_range_and_unavailable_display_semantics_are_preserved(self):
        scenarios = (
            (
                "negative",
                "R",
                (-2.345, 2.0, 3.0),
                _regression_set(),
                -2.345,
                ConcentrationStatus.BELOW_RANGE.value,
                "Con.R:-2.35",
            ),
            (
                "above range",
                "B",
                (1.0, 2.0, 49.38),
                _regression_set(),
                12.345,
                ConcentrationStatus.ABOVE_RANGE.value,
                "Con.B:12.35",
            ),
            (
                "not calculable",
                "G",
                (1.0, 2.0, 3.0),
                _regression_set(
                    {
                        "R": _formula(1.0),
                        "G": _formula(0.0),
                        "B": _formula(1.0),
                    }
                ),
                None,
                ConcentrationStatus.INVALID_SLOPE.value,
                "Con.G:N/A (INVALID_SLOPE)",
            ),
        )

        for name, channel, averages, regressions, value, status, label in scenarios:
            with self.subTest(name=name):
                harness, payload = self._build_detection(
                    channel, averages, regressions
                )
                sample = payload["sample_results"][0]
                suffix = channel.lower()
                self.assertEqual(sample["con_{}".format(suffix)], value)
                self.assertEqual(sample["status_{}".format(suffix)], status)
                self.assertEqual(_concentration_label(harness), label)


class UnchangedContractTests(unittest.TestCase):
    def test_linear_regression_image_keeps_generic_con_label(self):
        harness = _WorkerHarness()
        settings = _settings("G")
        settings.update(
            {
                "con_list": [-1.25, 12.5],
                "linear_formula_point_matrix": [True, True],
            }
        )
        with patch.object(
            worker_module,
            "_calculate_rgb_averages",
            side_effect=((1.0, 2.0, 3.0), (2.0, 4.0, 6.0)),
        ):
            YoloDetectionWorker._build_regression_payload(
                harness,
                _two_pair_result(),
                settings,
                _FakeCv2,
                source_path="fake/calibration.png",
            )

        labels = [
            text
            for block in harness.text_blocks
            for text in block
            if text.startswith("Con.")
        ]
        self.assertEqual(labels, ["Con.:-1.25", "Con.:12.5"])

    def test_detection_table_and_a_through_t_xlsx_headers_are_unchanged(self):
        source = (PROJECT_ROOT / "detectmain.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        detect_main = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "DetectMain"
        )

        class_values = {}
        methods = {}
        for node in detect_main.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name) and target.id in {
                    "CALIBRATION_TABLE_HEADERS",
                    "DETECTION_IDENTITY_HEADERS",
                    "DETECTION_RESULT_HEADERS",
                }:
                    class_values[target.id] = ast.literal_eval(node.value)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                methods[node.name] = node

        self.assertEqual(
            class_values["CALIBRATION_TABLE_HEADERS"],
            ("No.", "Con.", "Red", "Green", "Blue"),
        )
        xlsx_headers = (
            class_values["DETECTION_IDENTITY_HEADERS"]
            + class_values["DETECTION_RESULT_HEADERS"]
        )
        self.assertEqual(
            xlsx_headers,
            (
                "Image Order",
                "Source File",
                "No. in Image",
                "Batch No.",
                "R",
                "G",
                "B",
                "Con.R",
                "Con.G",
                "Con.B",
                "Status.R",
                "Status.G",
                "Status.B",
                None,
                "x0_con",
                "y0_con",
                "x1_con",
                "y1_con",
                "w_con",
                "h_con",
            ),
        )
        self.assertEqual(len(xlsx_headers), 20)
        self.assertIsNone(xlsx_headers[13])

        for method_name in (
            "_clear_detection_results_for_new_run",
            "_show_detection_image",
        ):
            header_values = [
                ast.literal_eval(node.value)
                for node in ast.walk(methods[method_name])
                if isinstance(node, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id == "headers"
                    for target in node.targets
                )
            ]
            self.assertIn(
                ["No.", "Con.", "Red", "Green", "Blue"], header_values
            )


if __name__ == "__main__":
    unittest.main()
