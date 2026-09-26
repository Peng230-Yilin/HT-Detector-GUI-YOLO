import sys
from pathlib import Path
import unittest
from unittest.mock import patch

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GUI_ROOT = PROJECT_ROOT
sys.path.insert(0, str(GUI_ROOT))

import yolo_detection_worker as worker_module  # noqa: E402
from batch_state import (  # noqa: E402
    ConcentrationStatus,
    RegressionSet,
    normalize_internal_path,
)
from yolo_detection_worker import YoloDetectionWorker  # noqa: E402


class FakeTensor:
    def __init__(self, values):
        self._values = np.asarray(values)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._values


class FakeBoxes:
    def __init__(self, coordinates, classes, confidences=None):
        self.xyxy = FakeTensor(coordinates)
        self.cls = FakeTensor(classes)
        self.conf = FakeTensor(confidences or [0.9] * len(classes))

    def __len__(self):
        return len(self.cls._values)


class FakeResult:
    names = {0: "cuvette", 1: "liquid"}

    def __init__(self, boxes):
        self.orig_img = np.zeros((120, 180, 3), dtype=np.uint8)
        self.boxes = boxes


def settings():
    return {
        "color_channel": "R",
        "Order_Con_R_G_B": "ConRGB",
        "show_confidence": False,
        "x0_ratio": 0.2,
        "y0_ratio": 0.2,
        "x1_ratio": 0.8,
        "y1_ratio": 0.8,
        "rgb_calculate_accuracy": 2,
        "rgb_display_accuracy": 2,
        "con_display_accuracy": 2,
        "detect_confidence": 0.5,
    }


def paired_boxes(cuvettes, liquids):
    coordinates = []
    classes = []
    for box in cuvettes:
        coordinates.append(box)
        classes.append(0)
    for box in liquids:
        coordinates.append(box)
        classes.append(1)
    return FakeBoxes(coordinates, classes)


def formula(slope, intercept):
    return {
        "slope": slope,
        "intercept": intercept,
        "r": 0.9,
        "R2": 0.81,
        "p": 0.01,
        "std_err": 0.1,
    }


def regression_set(formulas=None, source="worker-test"):
    formulas = formulas or {
        "R": formula(2.0, 10.0),
        "G": formula(4.0, 20.0),
        "B": formula(3.0, 40.0),
    }
    return RegressionSet.from_formulas(
        formulas, source_id=source, valid_ranges=(0.0, 100.0)
    )


def formal_detection_context(
    source_path, run_token, job_token, image_order,
    batch_start_no, display_start_no,
):
    source_path = normalize_internal_path(source_path)
    regressions = regression_set()
    values = {
        "run_token": run_token,
        "job_token": job_token,
        "image_order": image_order,
        "source_path": source_path,
        "source_file": Path(source_path).name,
        "batch_start_no": batch_start_no,
        "display_start_no": display_start_no,
        "regression_set": regressions,
        "regression_revision": regressions.revision,
    }
    return source_path, YoloDetectionWorker.lock_detection_context(
        source_path, values
    )


class WorkerPayloadTests(unittest.TestCase):
    def setUp(self):
        self.worker = YoloDetectionWorker()

    def build(self, result, averages=None, source_path="folder/source.PNG",
              regressions=None):
        kwargs = {
            "source_path": source_path,
            "regression_set": regressions or regression_set(),
        }
        if averages is None:
            return self.worker._build_payload(result, settings(), cv2, **kwargs)
        patch_kwargs = (
            {"side_effect": averages}
            if callable(averages)
            else {"return_value": averages}
        )
        with patch.object(
            worker_module, "_calculate_rgb_averages", **patch_kwargs
        ):
            return self.worker._build_payload(result, settings(), cv2, **kwargs)

    def test_success_payload_keeps_legacy_and_new_fields(self):
        result = FakeResult(paired_boxes(
            [(10, 10, 40, 90)], [(15, 30, 35, 80)]
        ))
        payload = self.build(result)
        self.assertTrue({"image", "targets", "warnings",
                         "sample_results", "sample_errors"} <= set(payload))
        self.assertEqual(len(payload["targets"]), 1)
        self.assertEqual(set(payload["targets"][0]), {
            "No.", "Con.", "Red", "Green", "Blue",
            "cuvette_box", "liquid_box", "rgb_roi",
        })
        self.assertEqual(payload["targets"][0]["No."], 1)
        sample = payload["sample_results"][0]
        self.assertIsNotNone(sample["con_r"])
        self.assertIsNotNone(sample["con_g"])
        self.assertIsNotNone(sample["con_b"])
        self.assertEqual(
            {sample["status_r"], sample["status_g"], sample["status_b"]},
            {ConcentrationStatus.BELOW_RANGE.value},
        )

    def test_three_distinct_equations_never_cross_channels(self):
        result = FakeResult(paired_boxes(
            [(10, 10, 40, 90)], [(15, 30, 35, 80)]
        ))
        payload = self.build(result, averages=(14.0, 28.0, 49.0))
        sample = payload["sample_results"][0]
        self.assertEqual(
            (sample["con_r"], sample["con_g"], sample["con_b"]),
            (2.0, 2.0, 3.0),
        )
        self.assertEqual(payload["targets"][0]["Con."], sample["con_r"])

    def test_one_invalid_channel_does_not_affect_other_channels_or_numbering(self):
        result = FakeResult(paired_boxes(
            [(10, 10, 40, 90)], [(15, 30, 35, 80)]
        ))
        regressions = regression_set({
            "R": formula(2.0, 10.0),
            "G": formula(0.0, 20.0),
            "B": formula(3.0, 40.0),
        }, source="partial")
        payload = self.build(
            result, averages=(14.0, 28.0, 49.0), regressions=regressions
        )
        sample = payload["sample_results"][0]
        self.assertEqual(sample["no_in_image"], 1)
        self.assertEqual(sample["con_r"], 2.0)
        self.assertIsNone(sample["con_g"])
        self.assertEqual(sample["status_g"], ConcentrationStatus.INVALID_SLOPE.value)
        self.assertEqual(sample["con_b"], 3.0)

    def test_display_number_matches_legacy_target_and_real_drawing_calls(self):
        result = FakeResult(paired_boxes(
            [(10, 10, 40, 90)], [(15, 30, 35, 80)]
        ))
        scenarios = (
            ("continuous", 2, 3, 3),
            ("per_image", 2, 3, 1),
            ("current", 1, 1, 1),
        )
        for name, image_order, batch_start_no, display_start_no in scenarios:
            with self.subTest(name=name), \
                    patch.object(cv2, "putText", wraps=cv2.putText) as put_text:
                payload = self.worker._build_payload(
                    result, settings(), cv2, source_path="second.png",
                    image_order=image_order, batch_start_no=batch_start_no,
                    display_start_no=display_start_no,
                    regression_set=regression_set(),
                )

            expected_display_no = display_start_no
            drawn_numbers = {
                call.args[1] for call in put_text.call_args_list
                if call.args[1].startswith("No.")
            }
            self.assertEqual(payload["targets"][0]["No."], expected_display_no)
            self.assertEqual(drawn_numbers, {"No.{}".format(expected_display_no)})
            self.assertEqual(payload["sample_results"][0]["no_in_image"], 1)
            self.assertEqual(
                payload["sample_results"][0]["batch_no"], batch_start_no
            )

    def test_invalid_roi_is_structured_and_other_sample_survives(self):
        result = FakeResult(paired_boxes(
            [(10, 10, 40, 90), (200, 10, 230, 90)],
            [(15, 30, 35, 80), (205, 30, 225, 80)],
        ))
        payload = self.build(result)
        self.assertEqual([target["No."] for target in payload["targets"]], [1])
        self.assertEqual(
            [error["error_type"] for error in payload["sample_errors"]],
            ["invalid_roi"],
        )
        self.assertEqual(payload["sample_errors"][0]["source_file"], "source.PNG")
        self.assertIsNone(payload["sample_errors"][0]["no_in_image"])
        self.assertIsNone(payload["sample_errors"][0]["batch_no"])

    def test_measurement_failure_does_not_create_number_gap(self):
        result = FakeResult(paired_boxes(
            [(10, 10, 40, 90), (70, 10, 100, 90), (130, 10, 160, 90)],
            [(15, 30, 35, 80), (75, 30, 95, 80), (135, 30, 155, 80)],
        ))

        def averages(_image, roi, _accuracy):
            if 40 <= roi[0] < 100:
                raise ValueError("The calculated RGB values are not finite.")
            return 10.0, 20.0, 30.0

        payload = self.build(result, averages=averages)
        self.assertEqual([target["No."] for target in payload["targets"]], [1, 2])
        self.assertEqual(
            [error["error_type"] for error in payload["sample_errors"]],
            ["measurement_failed"],
        )

    def test_no_detections_has_source_image_error(self):
        result = FakeResult(FakeBoxes([], []))
        payload = self.build(result, source_path="folder/empty.JPG")
        self.assertEqual(payload["targets"], [])
        self.assertEqual(payload["sample_results"], [])
        self.assertEqual(payload["sample_errors"][0]["error_type"], "image_failed")
        self.assertEqual(payload["sample_errors"][0]["source_file"], "empty.JPG")

    def test_detect_boundary_adds_source_path_without_loading_model(self):
        result = FakeResult(paired_boxes(
            [(10, 10, 40, 90)], [(15, 30, 35, 80)]
        ))

        class FakeModel:
            def predict(self, **_kwargs):
                return [result]

        completed = []
        failures = []
        self.worker.finished.connect(completed.append)
        self.worker.failed.connect(failures.append)
        source_path, context = formal_detection_context(
            "virtual/source.png", 5, 8, 1, 1, 1
        )
        regressions = context["regression_set"]
        with patch.object(worker_module, "load_effective_settings",
                          return_value=(settings(), [], None)), \
                patch.object(np, "fromfile", return_value=np.array([1], dtype=np.uint8)), \
                patch.object(cv2, "imdecode", return_value=result.orig_img), \
                patch.object(self.worker, "_get_model", return_value=FakeModel()), \
                patch.object(self.worker, "_load_formula",
                             side_effect=AssertionError("formula files must not be read")):
            self.worker.detect(source_path, "unused.pt", context)
        self.assertEqual(failures, [])
        self.assertEqual(len(completed), 1)
        self.assertEqual(completed[0]["source_path"], source_path)
        self.assertEqual(completed[0]["sample_results"][0]["source_file"], "source.png")
        self.assertEqual(completed[0]["run_token"], 5)
        self.assertEqual(completed[0]["job_token"], 8)
        self.assertEqual(completed[0]["regression_revision"], regressions.revision)

    def test_detect_propagates_run_and_job_tokens_on_success(self):
        result = FakeResult(paired_boxes(
            [(10, 10, 40, 90)], [(15, 30, 35, 80)]
        ))

        class FakeModel:
            def predict(self, **_kwargs):
                return [result]

        completed = []
        self.worker.finished.connect(completed.append)
        source_path, context = formal_detection_context(
            "virtual/source.png", 17, 19, 3, 31, 7
        )
        regressions = context["regression_set"]
        with patch.object(worker_module, "load_effective_settings",
                          return_value=(settings(), [], None)), \
                patch.object(np, "fromfile", return_value=np.array([1], dtype=np.uint8)), \
                patch.object(cv2, "imdecode", return_value=result.orig_img), \
                patch.object(self.worker, "_get_model", return_value=FakeModel()):
            self.worker.detect(source_path, "unused.pt", context)

        self.assertEqual(completed[0]["run_token"], 17)
        self.assertEqual(completed[0]["job_token"], 19)
        self.assertEqual(completed[0]["regression_revision"], regressions.revision)
        self.assertEqual(completed[0]["image_order"], 3)
        self.assertEqual(completed[0]["source_path"], source_path)
        self.assertEqual(completed[0]["source_file"], "source.png")

    def test_detect_propagates_run_and_job_tokens_on_failure(self):
        failures = []
        self.worker.failed.connect(failures.append)
        source_path, context = formal_detection_context(
            "virtual/source.png", 23, 29, 5, 41, 11
        )
        regressions = context["regression_set"]
        with patch.object(worker_module, "load_effective_settings",
                          side_effect=RuntimeError("settings unavailable")):
            self.worker.detect(source_path, "unused.pt", context)

        self.assertEqual(len(failures), 1)
        self.assertIn("settings unavailable", failures[0]["message"])
        self.assertEqual(failures[0]["run_token"], 23)
        self.assertEqual(failures[0]["job_token"], 29)
        self.assertEqual(failures[0]["regression_revision"], regressions.revision)
        self.assertEqual(failures[0]["image_order"], 5)
        self.assertEqual(failures[0]["source_path"], source_path)
        self.assertEqual(failures[0]["source_file"], "source.png")


if __name__ == "__main__":
    unittest.main()
