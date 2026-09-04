import sys
import types
from collections.abc import Mapping
from pathlib import Path
import unittest
from unittest.mock import patch

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GUI_ROOT = PROJECT_ROOT / "Peng1.0_GUI"
sys.path.insert(0, str(GUI_ROOT))

import yolo_detection_worker as worker_module  # noqa: E402
from linear_series_controller import (  # noqa: E402
    LINEAR_SERIES_OPERATION,
    LinearSeriesController,
)
from linear_series_state import LinearImageStatus  # noqa: E402
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
    def __init__(self, records):
        self.xyxy = FakeTensor([box for _, box, _ in records])
        self.cls = FakeTensor([class_id for class_id, _, _ in records])
        self.conf = FakeTensor([confidence for _, _, confidence in records])

    def __len__(self):
        return len(self.cls._values)


class FakeResult:
    names = {0: "cuvette", 1: "liquid"}

    def __init__(self, records, image=None):
        self.orig_img = (
            np.zeros((240, 320, 3), dtype=np.uint8)
            if image is None
            else image
        )
        self.boxes = FakeBoxes(records)


class FakeModel:
    def __init__(self, results=None, error=None):
        self.results = results
        self.error = error
        self.calls = []

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.results


class GuardSettings(Mapping):
    def __init__(self, values, forbidden):
        self._values = dict(values)
        self.forbidden = set(forbidden)
        self.accessed = []

    def __getitem__(self, key):
        self.accessed.append(key)
        if key in self.forbidden:
            raise AssertionError("forbidden setting accessed: {}".format(key))
        return self._values[key]

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)


def series_settings():
    return {
        "detect_confidence": 0.55,
        "show_confidence": False,
        "x0_ratio": 0.2,
        "y0_ratio": 0.2,
        "x1_ratio": 0.8,
        "y1_ratio": 0.8,
        "rgb_calculate_accuracy": 3,
    }


def identity(path="virtual/series-样品.PNG", run_token=11, job_token=17,
             image_order=1):
    return {
        "operation": LINEAR_SERIES_OPERATION,
        "run_token": run_token,
        "job_token": job_token,
        "image_order": image_order,
        "normalized_path": path,
        "original_file_name": Path(path).name,
    }


def paired_records(cuvettes, liquids):
    records = [(0, box, 0.91) for box in cuvettes]
    records.extend((1, box, 0.87) for box in liquids)
    return records


SINGLE_RECORDS = paired_records(
    [(10, 10, 50, 90)],
    [(15, 30, 45, 80)],
)


class LinearSeriesWorkerPayloadTests(unittest.TestCase):
    def invoke(self, worker, result, context=None, *, image_path=None,
               settings_value=None, model=None, encoded=None, decoded_marker=True):
        context = identity() if context is None else context
        if image_path is None:
            image_path = context.get("normalized_path", "virtual/invalid.png")
        if settings_value is None:
            settings_value = series_settings()
        if encoded is None:
            encoded = np.array([1, 2, 3], dtype=np.uint8)
        if model is None:
            model = FakeModel([result])
        if decoded_marker is True:
            decoded = result.orig_img
        else:
            decoded = decoded_marker

        completed = []
        failures = []
        worker.linear_series_extraction_finished.connect(completed.append)
        worker.linear_series_extraction_failed.connect(failures.append)
        with patch.object(
            worker_module,
            "load_effective_settings",
            return_value=(settings_value, ["config warning"], None),
        ), patch.object(
            np, "fromfile", return_value=encoded
        ), patch.object(
            cv2, "imdecode", return_value=decoded
        ), patch.object(
            worker, "_get_model", return_value=model
        ):
            worker.extract_linear_series_image(image_path, "virtual/model.pt", context)
        return completed, failures, model

    def build(self, records, context=None, image=None, settings_value=None):
        if context is None:
            context = identity()
        if settings_value is None:
            settings_value = series_settings()
        return YoloDetectionWorker._build_linear_series_payload(
            FakeResult(records, image=image),
            settings_value,
            cv2,
            context,
            config_warnings=["config warning"],
        )

    def test_success_uses_only_dedicated_series_signal(self):
        worker = YoloDetectionWorker()
        legacy_finished = []
        legacy_failed = []
        regression_finished = []
        regression_failed = []
        worker.finished.connect(legacy_finished.append)
        worker.failed.connect(legacy_failed.append)
        worker.regression_finished.connect(regression_finished.append)
        worker.regression_failed.connect(regression_failed.append)

        completed, failures, _ = self.invoke(
            worker, FakeResult(SINGLE_RECORDS)
        )

        self.assertEqual(len(completed), 1)
        self.assertEqual(failures, [])
        self.assertEqual(legacy_finished, [])
        self.assertEqual(legacy_failed, [])
        self.assertEqual(regression_finished, [])
        self.assertEqual(regression_failed, [])

    def test_single_sample_payload_has_exact_identity_and_raw_fields(self):
        context = identity()
        source = np.full((240, 320, 3), (3, 5, 7), dtype=np.uint8)
        payload = self.build(SINGLE_RECORDS, context=context, image=source)

        for key, value in context.items():
            self.assertEqual(payload[key], value)
        self.assertEqual(set(payload), set(context) | {
            "image", "samples", "errors", "warnings",
        })
        self.assertIsNot(payload["image"], source)
        self.assertEqual(payload["image"].shape, source.shape)
        self.assertEqual(payload["warnings"], ["config warning"])
        self.assertEqual(payload["errors"], [])
        self.assertEqual(len(payload["samples"]), 1)
        sample = payload["samples"][0]
        self.assertEqual(set(sample), {
            "image_order", "normalized_path", "original_file_name",
            "spatial_order", "red", "green", "blue", "cuvette_box",
            "liquid_box", "roi_box",
        })
        for forbidden in (
            "No.", "series_number", "Con.", "concentration", "included",
            "sample_key", "con_r", "con_g", "con_b",
        ):
            self.assertNotIn(forbidden, sample)

    def test_payload_values_have_transport_safe_python_types(self):
        payload = self.build(SINGLE_RECORDS)
        sample = payload["samples"][0]

        self.assertIs(type(sample["spatial_order"]), int)
        for channel in ("red", "green", "blue"):
            self.assertIs(type(sample[channel]), float)
            self.assertTrue(np.isfinite(sample[channel]))
        for name in ("cuvette_box", "liquid_box"):
            self.assertIs(type(sample[name]), tuple)
            self.assertTrue(all(type(value) is float for value in sample[name]))
        self.assertIs(type(sample["roi_box"]), tuple)
        self.assertTrue(all(type(value) is int for value in sample["roi_box"]))

    def test_spatial_order_is_multirow_and_independent_of_detection_order(self):
        records = paired_records(
            [(10, 10, 50, 90), (120, 12, 160, 92), (25, 120, 65, 210)],
            [(15, 30, 45, 80), (125, 32, 155, 82), (30, 140, 60, 200)],
        )
        orderings = (
            records,
            tuple(reversed(records)),
            records[2:] + records[:2],
            [records[index] for index in (5, 2, 4, 1, 3, 0)],
        )
        expected_boxes = [
            (10.0, 10.0, 50.0, 90.0),
            (120.0, 12.0, 160.0, 92.0),
            (25.0, 120.0, 65.0, 210.0),
        ]

        for ordering in orderings:
            with self.subTest(ordering=ordering):
                payload = self.build(ordering)
                self.assertEqual(
                    [sample["spatial_order"] for sample in payload["samples"]],
                    [1, 2, 3],
                )
                self.assertEqual(
                    [sample["cuvette_box"] for sample in payload["samples"]],
                    expected_boxes,
                )

    def test_same_detections_produce_identical_payload_and_annotation(self):
        records = paired_records(
            [(10, 10, 50, 90), (120, 12, 160, 92)],
            [(15, 30, 45, 80), (125, 32, 155, 82)],
        )
        forward = self.build(records)
        reverse = self.build(tuple(reversed(records)))

        self.assertEqual(forward["samples"], reverse["samples"])
        self.assertEqual(forward["errors"], reverse["errors"])
        np.testing.assert_array_equal(forward["image"], reverse["image"])

    def test_invalid_roi_is_structured_while_other_sample_survives(self):
        records = paired_records(
            [(10, 10, 50, 90), (120, 10, 160, 90)],
            [(15, 30, 45, 80), (125, 30, 155, 80)],
        )
        calculate_roi = worker_module._calculate_rgb_roi

        def selective_roi(box, image_shape, ratios):
            if box[0] > 100:
                raise ValueError("deliberately invalid ROI")
            return calculate_roi(box, image_shape, ratios)

        with patch.object(
            worker_module, "_calculate_rgb_roi", side_effect=selective_roi
        ):
            payload = self.build(records)

        self.assertEqual(len(payload["samples"]), 1)
        self.assertEqual(payload["samples"][0]["spatial_order"], 1)
        self.assertEqual(
            [error["error_type"] for error in payload["errors"]],
            ["invalid_roi"],
        )
        self.assertIn("deliberately invalid ROI", payload["errors"][0]["reason"])

    def test_measurement_failure_is_structured_while_other_sample_survives(self):
        records = paired_records(
            [(10, 10, 50, 90), (120, 10, 160, 90)],
            [(15, 30, 45, 80), (125, 30, 155, 80)],
        )

        def selective_measurement(_image, roi, _accuracy):
            if roi[0] > 100:
                raise ArithmeticError("deliberate measurement failure")
            return 1.25, 2.5, 3.75

        with patch.object(
            worker_module,
            "_calculate_rgb_averages",
            side_effect=selective_measurement,
        ):
            payload = self.build(records)

        self.assertEqual(len(payload["samples"]), 1)
        self.assertEqual(
            [error["error_type"] for error in payload["errors"]],
            ["measurement_failed"],
        )
        self.assertIn("deliberate measurement failure", payload["errors"][0]["reason"])

    def test_zero_valid_samples_with_structured_errors_remains_success_payload(self):
        records = paired_records(
            [(10, 10, 50, 90), (120, 10, 160, 90)],
            [(15, 30, 45, 80), (125, 30, 155, 80)],
        )
        worker = YoloDetectionWorker()
        with patch.object(
            worker_module,
            "_calculate_rgb_roi",
            side_effect=ValueError("all ROIs invalid"),
        ):
            completed, failures, _ = self.invoke(worker, FakeResult(records))

        self.assertEqual(failures, [])
        self.assertEqual(len(completed), 1)
        self.assertEqual(completed[0]["samples"], [])
        self.assertEqual(
            [error["error_type"] for error in completed[0]["errors"]],
            ["invalid_roi", "invalid_roi"],
        )

    def test_pairing_reports_all_unmatched_and_ambiguous_types(self):
        records = paired_records(
            [(0, 0, 40, 80), (20, 0, 60, 80), (150, 0, 190, 80)],
            [(25, 20, 35, 60), (250, 20, 270, 60)],
        )
        payload = self.build(records)

        self.assertEqual(payload["samples"], [])
        self.assertEqual(
            {error["error_type"] for error in payload["errors"]},
            {
                "ambiguous_cuvette",
                "ambiguous_liquid",
                "unmatched_cuvette",
                "unmatched_liquid",
            },
        )
        for error in payload["errors"]:
            self.assertIs(type(error["error_type"]), str)
            self.assertIs(type(error["related_boxes"]), tuple)
            self.assertTrue(
                all(type(box) is tuple for box in error["related_boxes"])
            )

    def test_pairing_error_order_is_deterministic_for_input_permutations(self):
        records = paired_records(
            [(0, 0, 40, 80), (20, 0, 60, 80), (150, 0, 190, 80)],
            [(25, 20, 35, 60), (250, 20, 270, 60)],
        )
        expected = self.build(records)["errors"]

        for ordering in (
            tuple(reversed(records)),
            records[1:] + records[:1],
            [records[index] for index in (3, 1, 4, 0, 2)],
        ):
            with self.subTest(ordering=ordering):
                self.assertEqual(self.build(ordering)["errors"], expected)

    def test_no_detections_is_structured_success_not_fatal_failure(self):
        worker = YoloDetectionWorker()
        completed, failures, _ = self.invoke(worker, FakeResult([]))

        self.assertEqual(failures, [])
        self.assertEqual(len(completed), 1)
        self.assertEqual(completed[0]["samples"], [])
        self.assertEqual(
            [error["error_type"] for error in completed[0]["errors"]],
            ["image_failed"],
        )

    def test_valid_context_fatal_boundaries_echo_identity(self):
        context = identity()
        scenarios = (
            ("empty", np.array([], dtype=np.uint8), True,
             FakeModel([FakeResult(SINGLE_RECORDS)])),
            ("decode", np.array([1], dtype=np.uint8), None,
             FakeModel([FakeResult(SINGLE_RECORDS)])),
            ("inference", np.array([1], dtype=np.uint8), True,
             FakeModel(error=RuntimeError("inference exploded"))),
            ("no-results", np.array([1], dtype=np.uint8), True,
             FakeModel(results=[])),
        )

        for name, encoded, decoded, model in scenarios:
            with self.subTest(name=name):
                worker = YoloDetectionWorker()
                completed, failures, _ = self.invoke(
                    worker,
                    FakeResult(SINGLE_RECORDS),
                    context=context,
                    encoded=encoded,
                    decoded_marker=decoded,
                    model=model,
                )
                self.assertEqual(completed, [])
                self.assertEqual(len(failures), 1)
                for key, value in context.items():
                    self.assertEqual(failures[0][key], value)
                self.assertEqual(failures[0]["error_type"], "image_failed")
                self.assertIs(type(failures[0]["reason"]), str)
                self.assertEqual(failures[0]["errors"], [])

    def test_invalid_context_is_rejected_before_settings_io_and_model(self):
        base = identity()
        variants = []
        missing = dict(base)
        missing.pop("job_token")
        variants.append(("missing", base["normalized_path"], missing))
        extra = dict(base, extra_field="not allowed")
        variants.append(("extra", base["normalized_path"], extra))
        variants.append(("operation", base["normalized_path"],
                         dict(base, operation="detection")))
        variants.append(("bool-token", base["normalized_path"],
                         dict(base, run_token=True)))
        variants.append(("wrong-path", "virtual/other.png", base))

        for name, image_path, bad_context in variants:
            with self.subTest(name=name):
                worker = YoloDetectionWorker()
                completed = []
                failures = []
                worker.linear_series_extraction_finished.connect(completed.append)
                worker.linear_series_extraction_failed.connect(failures.append)
                with patch.object(
                    worker_module,
                    "load_effective_settings",
                    side_effect=AssertionError("settings must not be loaded"),
                ), patch.object(
                    np,
                    "fromfile",
                    side_effect=AssertionError("image must not be read"),
                ), patch.object(
                    worker,
                    "_get_model",
                    side_effect=AssertionError("model must not be loaded"),
                ):
                    worker.extract_linear_series_image(
                        image_path, "virtual/model.pt", bad_context
                    )

                self.assertEqual(completed, [])
                self.assertEqual(len(failures), 1)
                self.assertEqual(failures[0]["error_type"], "invalid_context")
                self.assertIs(failures[0]["context_valid"], False)
                self.assertNotIn("must not be", failures[0]["reason"])

    def test_invalid_context_diagnostic_cannot_be_consumed_by_controller(self):
        controller = LinearSeriesController()
        task = controller.begin(["virtual/controller.png"])
        self.assertIsNotNone(task)
        before_state = controller.state
        before_image = controller.state.images[0]
        bad_context = task.context()
        bad_context["job_token"] = True
        diagnostic = YoloDetectionWorker._linear_series_context_diagnostic(
            bad_context, ValueError("invalid context")
        )

        self.assertFalse(controller.accept_failure(diagnostic))
        self.assertIs(controller.state, before_state)
        self.assertIs(controller.state.images[0], before_image)
        self.assertEqual(before_image.status, LinearImageStatus.PROCESSING)
        self.assertEqual(before_image.samples, ())
        self.assertEqual(before_image.errors, ())
        self.assertEqual(controller.active_job, task)

    def test_unicode_path_and_duplicate_path_jobs_echo_exact_identity(self):
        path = "virtual/目录/样品.Aa.PNG"
        contexts = (
            identity(path, run_token=31, job_token=41, image_order=1),
            identity(path, run_token=31, job_token=42, image_order=2),
        )
        observed = []
        for context in contexts:
            worker = YoloDetectionWorker()
            completed, failures, _ = self.invoke(
                worker, FakeResult(SINGLE_RECORDS), context=context
            )
            self.assertEqual(failures, [])
            observed.append({key: completed[0][key] for key in context})

        self.assertEqual(observed, list(contexts))
        self.assertNotEqual(observed[0]["job_token"], observed[1]["job_token"])
        self.assertNotEqual(observed[0]["image_order"], observed[1]["image_order"])

    def test_series_entry_reads_only_allowlisted_settings(self):
        allowed = series_settings()
        forbidden = {
            "con_list", "linear_formula_point_matrix", "color_channel",
            "Order_Con_R_G_B", "con_display_accuracy",
            "rgb_display_accuracy", "detection_scope", "numbering_mode",
        }
        guard = GuardSettings({**allowed, **{key: object() for key in forbidden}},
                              forbidden)
        worker = YoloDetectionWorker()

        completed, failures, _ = self.invoke(
            worker, FakeResult(SINGLE_RECORDS), settings_value=guard
        )

        self.assertEqual(failures, [])
        self.assertEqual(len(completed), 1)
        self.assertEqual(set(guard.accessed), set(allowed))
        self.assertTrue(set(guard.accessed).isdisjoint(forbidden))

    def test_series_entry_never_uses_formula_numbering_or_regression_helpers(self):
        worker = YoloDetectionWorker()
        with patch.object(
            worker,
            "_load_formula",
            side_effect=AssertionError("formula access is forbidden"),
        ), patch.object(
            worker_module,
            "_calculate_measurement",
            side_effect=AssertionError("concentration is forbidden"),
        ), patch.object(
            worker_module,
            "assign_image_numbers",
            side_effect=AssertionError("numbering is forbidden"),
        ), patch.object(
            worker,
            "_build_regression_payload",
            side_effect=AssertionError("regression is forbidden"),
        ):
            completed, failures, _ = self.invoke(
                worker, FakeResult(SINGLE_RECORDS)
            )

        self.assertEqual(failures, [])
        self.assertEqual(len(completed), 1)

    def test_series_entry_does_not_mutate_formula_caches(self):
        worker = YoloDetectionWorker()
        active = {"R": (1.0, 2.0)}
        cache = {"virtual/formula.xlsx": (3.0, 4.0)}
        worker._active_formulas = active
        worker._formula_cache = cache

        completed, failures, _ = self.invoke(
            worker, FakeResult(SINGLE_RECORDS)
        )

        self.assertEqual(failures, [])
        self.assertEqual(len(completed), 1)
        self.assertIs(worker._active_formulas, active)
        self.assertIs(worker._formula_cache, cache)
        self.assertEqual(active, {"R": (1.0, 2.0)})
        self.assertEqual(cache, {"virtual/formula.xlsx": (3.0, 4.0)})

    def test_annotation_uses_local_sample_label_without_no_or_concentration(self):
        with patch.object(cv2, "putText", wraps=cv2.putText) as put_text:
            self.build(SINGLE_RECORDS)

        texts = [call.args[1] for call in put_text.call_args_list]
        self.assertIn("Sample 1", texts)
        self.assertFalse(any(text.startswith("No.") for text in texts))
        self.assertFalse(any(text.startswith("Con.") for text in texts))

    def test_worker_payload_is_accepted_by_controller(self):
        controller = LinearSeriesController()
        task = controller.begin(["virtual/controller.png"])
        self.assertIsNotNone(task)
        payload = self.build(SINGLE_RECORDS, context=task.context())

        self.assertTrue(controller.accept_success(payload))
        image = controller.state.image_for_order(task.image_order)
        self.assertEqual(image.status, LinearImageStatus.COMPLETED)
        self.assertEqual(len(image.samples), 1)
        self.assertEqual(image.errors, ())

    def test_partial_worker_payload_is_accepted_by_controller(self):
        controller = LinearSeriesController()
        task = controller.begin(["virtual/controller.png"])
        records = paired_records(
            [(10, 10, 50, 90), (120, 10, 160, 90)],
            [(15, 30, 45, 80), (125, 30, 155, 80)],
        )

        def selective_measurement(_image, roi, _accuracy):
            if roi[0] > 100:
                raise ValueError("one sample failed")
            return 1.0, 2.0, 3.0

        with patch.object(
            worker_module,
            "_calculate_rgb_averages",
            side_effect=selective_measurement,
        ):
            payload = self.build(records, context=task.context())

        self.assertTrue(controller.accept_success(payload))
        image = controller.state.image_for_order(task.image_order)
        self.assertEqual(image.status, LinearImageStatus.COMPLETED)
        self.assertEqual(len(image.samples), 1)
        self.assertEqual(len(image.errors), 1)
        self.assertEqual(image.errors[0].error_type, "measurement_failed")

    def test_fatal_worker_payload_is_accepted_once_by_controller(self):
        controller = LinearSeriesController()
        task = controller.begin(["virtual/controller.png"])
        worker = YoloDetectionWorker()
        completed, failures, _ = self.invoke(
            worker,
            FakeResult(SINGLE_RECORDS),
            context=task.context(),
            model=FakeModel(error=RuntimeError("fatal inference")),
        )

        self.assertEqual(completed, [])
        self.assertEqual(len(failures), 1)
        self.assertTrue(controller.accept_failure(failures[0]))
        self.assertFalse(controller.accept_failure(failures[0]))
        image = controller.state.image_for_order(task.image_order)
        self.assertEqual(image.status, LinearImageStatus.FAILED)
        self.assertEqual(len(image.errors), 1)
        self.assertEqual(image.errors[0].error_type, "image_failed")

    def test_model_cache_reuses_same_weight_and_reloads_changed_weight(self):
        created = []

        def yolo_factory(weight_path):
            model = object()
            created.append((weight_path, model))
            return model

        fake_ultralytics = types.ModuleType("ultralytics")
        fake_ultralytics.YOLO = yolo_factory
        worker = YoloDetectionWorker()
        with patch.dict(sys.modules, {"ultralytics": fake_ultralytics}):
            first = worker._get_model("virtual/one.pt")
            repeated = worker._get_model("virtual/one.pt")
            changed = worker._get_model("virtual/two.pt")

        self.assertIs(first, repeated)
        self.assertIsNot(first, changed)
        self.assertEqual(
            [weight_path for weight_path, _ in created],
            ["virtual/one.pt", "virtual/two.pt"],
        )

    def test_malformed_detection_arrays_fail_without_success_signal(self):
        malformed_records = [(0, (10, 10, float("nan"), 90), 0.9)]
        worker = YoloDetectionWorker()
        completed, failures, _ = self.invoke(
            worker, FakeResult(malformed_records)
        )

        self.assertEqual(completed, [])
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0]["error_type"], "image_failed")
        self.assertIn("finite coordinates", failures[0]["reason"])


if __name__ == "__main__":
    unittest.main()
