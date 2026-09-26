import builtins
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from batch_state import RegressionSet  # noqa: E402
from yolo_detection_worker import YoloDetectionWorker  # noqa: E402


IDENTITY_FIELDS = (
    "run_token",
    "job_token",
    "regression_revision",
    "image_order",
    "source_path",
    "source_file",
)


def formula(slope, intercept):
    return {
        "slope": slope,
        "intercept": intercept,
        "r": 0.9,
        "R2": 0.81,
        "p": 0.01,
        "std_err": 0.1,
    }


def authoritative_context():
    regression_set = RegressionSet.from_formulas(
        {
            "R": formula(2.0, 10.0),
            "G": formula(3.0, 20.0),
            "B": formula(4.0, 30.0),
        },
        source_id="repair2-authoritative-regression",
        valid_ranges=(0.0, 100.0),
    )
    source_path = str(PROJECT_ROOT / "authoritative-input" / "sample-09.png")
    return {
        "run_token": 17,
        "job_token": 23,
        "regression_revision": regression_set.revision,
        "image_order": 9,
        "source_path": source_path,
        "source_file": "sample-09.png",
        "batch_start_no": 31,
        "display_start_no": 7,
        "regression_set": regression_set,
    }


def identity_from(context):
    return {name: context[name] for name in IDENTITY_FIELDS}


class TestDetectionImportFailureIdentity(unittest.TestCase):
    def _worker_callbacks(self):
        worker = YoloDetectionWorker()
        successes = []
        failures = []
        worker.finished.connect(successes.append)
        worker.failed.connect(failures.append)
        return worker, successes, failures

    def _assert_single_authoritative_failure(self, context, successes, failures):
        self.assertEqual(successes, [])
        self.assertEqual(len(failures), 1)
        actual = {name: failures[0][name] for name in IDENTITY_FIELDS}
        self.assertEqual(actual, identity_from(context))
        self.assertNotIn(None, actual.values())
        self.assertNotEqual(actual["image_order"], 1)

    def test_cv2_import_failure_uses_complete_original_identity_once(self):
        context = authoritative_context()
        worker, successes, failures = self._worker_callbacks()
        original_import = builtins.__import__

        def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "cv2":
                raise ImportError("synthetic cv2 import failure")
            return original_import(name, globals, locals, fromlist, level)

        with patch("builtins.__import__", side_effect=guarded_import):
            worker.detect(context["source_path"], "unused.pt", context)

        self._assert_single_authoritative_failure(context, successes, failures)
        self.assertIn("synthetic cv2 import failure", failures[0]["message"])

    def test_numpy_import_failure_uses_complete_original_identity_once(self):
        context = authoritative_context()
        worker, successes, failures = self._worker_callbacks()
        original_import = builtins.__import__

        def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "numpy":
                raise ImportError("synthetic numpy import failure")
            return original_import(name, globals, locals, fromlist, level)

        with patch.dict(sys.modules, {"cv2": SimpleNamespace()}):
            with patch("builtins.__import__", side_effect=guarded_import):
                worker.detect(context["source_path"], "unused.pt", context)

        self._assert_single_authoritative_failure(context, successes, failures)
        self.assertIn("synthetic numpy import failure", failures[0]["message"])


class TestDetectionContextRejection(unittest.TestCase):
    def test_each_missing_identity_field_is_rejected_before_import_without_emit(self):
        original_import = builtins.__import__

        def reject_dependency_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name in {"cv2", "numpy"}:
                raise AssertionError("third-party import reached for invalid context")
            return original_import(name, globals, locals, fromlist, level)

        for missing_field in IDENTITY_FIELDS:
            with self.subTest(missing_field=missing_field):
                context = authoritative_context()
                context.pop(missing_field)
                worker = YoloDetectionWorker()
                successes = []
                failures = []
                worker.finished.connect(successes.append)
                worker.failed.connect(failures.append)

                with patch("builtins.__import__", side_effect=reject_dependency_import):
                    with self.assertRaisesRegex(ValueError, missing_field):
                        worker.detect(
                            authoritative_context()["source_path"],
                            "unused.pt",
                            context,
                        )

                self.assertEqual(successes, [])
                self.assertEqual(failures, [])


class TestDetectionPostImportPaths(unittest.TestCase):
    def _worker_callbacks(self):
        worker = YoloDetectionWorker()
        successes = []
        failures = []
        worker.finished.connect(successes.append)
        worker.failed.connect(failures.append)
        return worker, successes, failures

    def test_settings_failure_uses_complete_original_identity_once(self):
        context = authoritative_context()
        worker, successes, failures = self._worker_callbacks()

        with patch.dict(
            sys.modules,
            {"cv2": SimpleNamespace(), "numpy": SimpleNamespace()},
        ):
            with patch(
                "yolo_detection_worker.load_effective_settings",
                side_effect=RuntimeError("synthetic settings failure"),
            ):
                worker.detect(context["source_path"], "unused.pt", context)

        self.assertEqual(successes, [])
        self.assertEqual(len(failures), 1)
        self.assertEqual(
            {name: failures[0][name] for name in IDENTITY_FIELDS},
            identity_from(context),
        )
        self.assertIn("synthetic settings failure", failures[0]["message"])

    def test_success_payload_keeps_complete_original_identity_without_failure(self):
        context = authoritative_context()
        worker, successes, failures = self._worker_callbacks()
        encoded_path = object()
        source_image = object()
        fake_numpy = SimpleNamespace(
            uint8=object(),
            fromfile=Mock(return_value=encoded_path),
        )
        fake_cv2 = SimpleNamespace(
            IMREAD_COLOR=object(),
            imdecode=Mock(return_value=source_image),
        )
        model = Mock()
        model.predict.return_value = [object()]
        success_body = {
            "sample_results": [],
            "sample_errors": [],
            "sentinel": "successful detection",
        }

        with patch.dict(sys.modules, {"cv2": fake_cv2, "numpy": fake_numpy}):
            with patch(
                "yolo_detection_worker.load_effective_settings",
                return_value=({"detect_confidence": 0.5}, [], None),
            ):
                with patch.object(worker, "_get_model", return_value=model):
                    with patch.object(
                        worker,
                        "_build_payload",
                        return_value=dict(success_body),
                    ) as build_payload:
                        worker.detect(
                            context["source_path"], "unused.pt", context
                        )

        self.assertEqual(failures, [])
        self.assertEqual(len(successes), 1)
        self.assertEqual(
            {name: successes[0][name] for name in IDENTITY_FIELDS},
            identity_from(context),
        )
        self.assertEqual(successes[0]["sentinel"], "successful detection")
        fake_numpy.fromfile.assert_called_once_with(
            context["source_path"], dtype=fake_numpy.uint8
        )
        fake_cv2.imdecode.assert_called_once_with(
            encoded_path, fake_cv2.IMREAD_COLOR
        )
        model.predict.assert_called_once_with(
            source=source_image,
            device="cpu",
            conf=0.5,
            save=False,
            verbose=False,
        )
        build_payload.assert_called_once()
        self.assertEqual(
            build_payload.call_args.kwargs["image_order"],
            context["image_order"],
        )
        self.assertEqual(
            build_payload.call_args.kwargs["batch_start_no"],
            context["batch_start_no"],
        )
        self.assertEqual(
            build_payload.call_args.kwargs["display_start_no"],
            context["display_start_no"],
        )
        self.assertIs(
            build_payload.call_args.kwargs["regression_set"],
            context["regression_set"],
        )


if __name__ == "__main__":
    unittest.main()
