import ast
import inspect
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

from batch_detection_controller import BatchDetectionController
from batch_state import (
    BatchState,
    ImageItem,
    ImageStatus,
    RegressionSessionState,
    RegressionSet,
)
from detectmain import (
    DetectMain,
    LINEAR_MODE_SINGLE_IMAGE,
    TIME_DISPLAY_DETECTION,
    TIME_DISPLAY_LINEAR,
    TIME_DISPLAY_NONE,
)


ROOT = Path(__file__).resolve().parents[1]


class FakeClock:
    def __init__(self, initial_ns=10_000_000_000):
        self.value_ns = initial_ns
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return self.value_ns

    def advance_ms(self, milliseconds):
        self.value_ns += milliseconds * 1_000_000


class FakeDisplay:
    def __init__(self):
        self.values = []

    def display(self, value):
        self.values.append(value)


class FakeLabel:
    def __init__(self):
        self.values = []

    def setText(self, value):
        self.values.append(value)


class FakeTimer:
    def __init__(self, active=False):
        self.active = active
        self.stop_calls = 0

    def isActive(self):
        return self.active

    def stop(self):
        self.active = False
        self.stop_calls += 1


def formulas():
    return {
        channel: {
            "slope": float(index + 1),
            "intercept": 0.0,
            "r": 1.0,
            "R2": 1.0,
            "p": 0.0,
            "std_err": 0.0,
        }
        for index, channel in enumerate(("R", "G", "B"))
    }


def regression_payload(source_path="C:/calibration/current.png"):
    return {
        "source_path": source_path,
        "formulas": formulas(),
        "samples": [
            {"Con.": 1.0, "included": True},
            {"Con.": 2.0, "included": True},
        ],
    }


def regression_set(source_id="test-regression"):
    return RegressionSet.from_formulas(
        formulas(), source_id=source_id, valid_ranges=(1.0, 2.0)
    )


def shared_harness(**values):
    defaults = {
        "_time_display_mode": TIME_DISPLAY_NONE,
        "ui": SimpleNamespace(label_9=FakeLabel(), lcdNumber=FakeDisplay()),
    }
    defaults.update(values)
    return SimpleNamespace(**defaults)


def prepared_detection_controller(paths, clock):
    controller = BatchDetectionController(
        BatchState.from_paths(paths), clock_ns=clock
    )
    task = controller.begin(regression_set())
    context, rejection = controller.prepare_dispatch_context(
        task, lambda _path, value: MappingProxyType(dict(value))
    )
    if rejection is not None or context is None:
        raise AssertionError("The valid detection task was not prepared.")
    controller.start_dispatch_timer(task)
    return controller, task


class SharedTimeUiStructureTests(unittest.TestCase):
    def test_formal_ui_has_exactly_one_time_label_and_value_control(self):
        root = ET.parse(ROOT / "ui" / "detectmain.ui").getroot()
        lcd_names = [
            widget.get("name")
            for widget in root.findall(".//widget[@class='QLCDNumber']")
        ]
        time_labels = []
        for widget in root.findall(".//widget[@class='QLabel']"):
            text = widget.find("./property[@name='text']/string")
            if text is not None and "Time (ms):" in (text.text or ""):
                time_labels.append((widget.get("name"), text.text))

        self.assertEqual(lcd_names, ["lcdNumber"])
        self.assertEqual(time_labels, [("label_9", "Time (ms):")])

    def test_removed_linear_time_row_reclaims_the_button_layout_row(self):
        root = ET.parse(ROOT / "ui" / "detectmain.ui").getroot()
        grid = root.find(".//layout[@name='gridLayout']")
        self.assertIsNotNone(grid)
        self.assertEqual({item.get("row") for item in grid.findall("./item")}, {"0"})
        xml_text = (ROOT / "ui" / "detectmain.ui").read_text(encoding="utf-8")
        generated = (ROOT / "ui" / "ui_detectmain.py").read_text(encoding="utf-8")
        for legacy_name in ("labelLinearTime", "lcdLinearTime"):
            self.assertNotIn(legacy_name, xml_text)
            self.assertNotIn(legacy_name, generated)

    def test_initial_label_is_generic_and_value_is_blank(self):
        harness = shared_harness()
        DetectMain._set_shared_time_display(harness, TIME_DISPLAY_NONE, None)

        self.assertEqual(harness.ui.label_9.values, ["Time (ms):"])
        self.assertEqual(harness.ui.lcdNumber.values, [""])
        self.assertEqual(harness._time_display_mode, TIME_DISPLAY_NONE)


class SharedTimeAuthorityTests(unittest.TestCase):
    def make_linear_harness(self, clock, payload=None):
        payload = regression_payload() if payload is None else payload
        session = RegressionSessionState(clock_ns=clock)
        return shared_harness(
            _linear_mode=LINEAR_MODE_SINGLE_IMAGE,
            _calibration_source_path=payload["source_path"],
            _regression_result=payload,
            _regression_session_state=session,
            _linear_duration_timer=FakeTimer(active=True),
        ), session, DetectMain._regression_set_from_result(payload)

    def test_legal_linear_start_and_completion_use_regression_timing(self):
        clock = FakeClock()
        harness, session, current = self.make_linear_harness(clock)
        operation = session.begin_attempt(current.source_id)
        clock.advance_ms(4921)

        DetectMain._refresh_linear_duration_display(harness)
        timing = session.accept_success(current, operation.operation_token)
        DetectMain._refresh_linear_duration_display(harness)

        self.assertEqual(timing.duration_ms, 4921)
        self.assertEqual(harness.ui.lcdNumber.values, ["4921", "4921"])
        self.assertEqual(
            harness.ui.label_9.values,
            ["Linear Time (ms):", "Linear Time (ms):"],
        )

    def test_plot_refresh_reads_fixed_regression_timing_without_new_clock_read(self):
        clock = FakeClock()
        harness, session, current = self.make_linear_harness(clock)
        operation = session.begin_attempt(current.source_id)
        clock.advance_ms(83)
        session.accept_success(current, operation.operation_token)
        calls_before = clock.calls

        DetectMain._refresh_linear_duration_display(harness)

        self.assertEqual(clock.calls, calls_before)
        self.assertEqual(harness.ui.label_9.values[-1], "Linear Time (ms):")
        self.assertEqual(harness.ui.lcdNumber.values[-1], "83")

    def test_legal_detection_start_and_completion_use_image_item_duration(self):
        clock = FakeClock()
        controller, task = prepared_detection_controller(["C:/input/a.png"], clock)
        harness = shared_harness(
            _batch_controller=controller,
            _detection_duration_timer=FakeTimer(active=True),
            _detection_selected_key=None,
        )
        clock.advance_ms(764)

        DetectMain._refresh_detection_duration_display(
            harness, controller.state.current_image
        )
        payload = dict(task.context(), sample_results=[], sample_errors=[])
        self.assertTrue(controller.accept_payload(payload))
        DetectMain._refresh_detection_duration_display(
            harness, controller.state.current_image
        )

        self.assertEqual(
            controller.state.current_image.detection_duration_ms, 764
        )
        self.assertEqual(harness.ui.lcdNumber.values, ["764", "764"])
        self.assertEqual(
            harness.ui.label_9.values,
            ["Detection Time (ms):", "Detection Time (ms):"],
        )

    def test_detection_selection_reads_each_image_and_clears_missing_duration(self):
        first = ImageItem(
            "C:/input/a.png", "a.png", 1,
            status=ImageStatus.COMPLETED, detection_duration_ms=111,
        )
        second = ImageItem(
            "C:/input/b.png", "b.png", 2,
            status=ImageStatus.PENDING, detection_duration_ms=None,
        )
        controller = BatchDetectionController(BatchState(images=[first, second]))
        harness = shared_harness(
            _batch_controller=controller,
            _detection_duration_timer=None,
            _detection_selected_key=None,
        )

        DetectMain._refresh_detection_duration_display(harness, first)
        DetectMain._refresh_detection_duration_display(harness, second)

        self.assertEqual(harness.ui.lcdNumber.values, ["111", ""])
        self.assertEqual(harness._time_display_mode, TIME_DISPLAY_DETECTION)

    def test_linear_detection_linear_switch_does_not_cross_values(self):
        clock = FakeClock()
        payload = regression_payload()
        linear_harness, session, current = self.make_linear_harness(clock, payload)
        operation = session.begin_attempt(current.source_id)
        clock.advance_ms(456)
        session.accept_success(current, operation.operation_token)
        image = ImageItem(
            "C:/input/a.png", "a.png", 1,
            status=ImageStatus.COMPLETED, detection_duration_ms=789,
        )
        linear_harness._batch_controller = BatchDetectionController(
            BatchState(images=[image])
        )
        linear_harness._detection_duration_timer = None
        linear_harness._detection_selected_key = None

        DetectMain._refresh_linear_duration_display(linear_harness)
        DetectMain._refresh_detection_duration_display(linear_harness, image)
        DetectMain._refresh_linear_duration_display(linear_harness)

        self.assertEqual(linear_harness.ui.lcdNumber.values, ["456", "789", "456"])
        self.assertEqual(
            linear_harness.ui.label_9.values,
            ["Linear Time (ms):", "Detection Time (ms):", "Linear Time (ms):"],
        )


class SharedTimeCallbackGateTests(unittest.TestCase):
    def test_same_source_stale_linear_callbacks_cannot_change_shared_display(self):
        source_path = "C:/calibration/repeated.png"
        source_id = DetectMain._regression_source_id(source_path)
        clock = FakeClock()
        session = RegressionSessionState(clock_ns=clock)
        old_set = regression_set(source_id)
        old_operation = session.begin_attempt(source_id)
        clock.advance_ms(10)
        session.accept_success(old_set, old_operation.operation_token)
        current_operation = session.begin_attempt(source_id)
        harness = shared_harness(
            _active_worker_task="regression",
            _calibration_source_path=source_path,
            _regression_session_state=session,
        )
        calls_before = clock.calls

        DetectMain._on_regression_finished(
            harness,
            {"source_path": source_path, "operation_token": old_operation.operation_token},
        )
        DetectMain._on_regression_failed(
            harness,
            {
                "source_path": source_path,
                "operation_token": old_operation.operation_token,
                "message": "late failure",
            },
        )

        self.assertEqual(harness.ui.label_9.values, [])
        self.assertEqual(harness.ui.lcdNumber.values, [])
        self.assertEqual(clock.calls, calls_before)
        self.assertIs(session.active_operation, current_operation)
        self.assertEqual(harness._active_worker_task, "regression")

    def test_wrong_detection_identity_cannot_change_shared_display_or_timer(self):
        clock = FakeClock()
        controller, task = prepared_detection_controller(["C:/input/a.png"], clock)
        harness = shared_harness(_batch_controller=controller)
        wrong = dict(
            task.context(),
            job_token=task.job_token + 1,
            sample_results=[],
            sample_errors=[],
        )
        calls_before = clock.calls

        DetectMain._on_detection_finished(harness, wrong)
        DetectMain._on_detection_failed(harness, dict(wrong, message="wrong"))

        self.assertEqual(harness.ui.label_9.values, [])
        self.assertEqual(harness.ui.lcdNumber.values, [])
        self.assertEqual(clock.calls, calls_before)
        self.assertIs(controller.active_job, task)
        self.assertTrue(controller.active)

    def test_callback_identity_gate_is_first_statement_before_ui_work(self):
        tree = ast.parse((ROOT / "detectmain.py").read_text(encoding="utf-8"))
        class_node = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "DetectMain"
        )
        methods = {
            node.name: node
            for node in class_node.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        expected_gate = {
            "_on_regression_finished": "_linear_regression_result_matches",
            "_on_regression_failed": "_linear_regression_failure_matches",
            "_on_detection_finished": "_detection_result_matches",
            "_on_detection_failed": "_detection_result_matches",
        }
        for method_name, gate_name in expected_gate.items():
            with self.subTest(method=method_name):
                first = methods[method_name].body[0]
                self.assertIsInstance(first, ast.If)
                self.assertIn(gate_name, ast.unparse(first.test))
                self.assertTrue(any(isinstance(node, ast.Return) for node in first.body))


class SharedTimePreservationTests(unittest.TestCase):
    def test_excel_headers_gui_tables_and_png_paths_remain_separate(self):
        self.assertEqual(
            DetectMain.CALIBRATION_TABLE_HEADERS,
            ("No.", "Con.", "Red", "Green", "Blue"),
        )
        self.assertEqual(len(DetectMain.DETECTION_TABLE_HEADERS), 21)
        self.assertIsNone(DetectMain.DETECTION_TABLE_HEADERS[13])
        self.assertEqual(
            DetectMain.DETECTION_TABLE_HEADERS[20], "Detection Time (ms)"
        )
        main_source = (ROOT / "detectmain.py").read_text(encoding="utf-8")
        self.assertIn('worksheet.cell(1, 15, "Linear Time (ms)")', main_source)
        self.assertIn(
            'worksheet.cell(row_index, 15).number_format = "0"', main_source
        )
        self.assertIn(
            ').number_format = "0"',
            inspect.getsource(DetectMain._build_detection_workbook_bytes),
        )

        for method in (
            DetectMain._build_detection_export_bytes,
            DetectMain._build_linear_export_bytes,
        ):
            source = inspect.getsource(method)
            self.assertNotIn("Linear Time (ms)", source)
            self.assertNotIn("Detection Time (ms)", source)

    def test_presentation_mode_is_not_a_duration_cache_or_label_inference(self):
        source = inspect.getsource(DetectMain._set_shared_time_display)
        self.assertIn("self._time_display_mode = mode", source)
        self.assertNotIn(".text()", source)
        self.assertNotIn(".value()", source)
        self.assertNotIn("linear_duration_ms", source)
        self.assertNotIn("detection_duration_ms", source)


if __name__ == "__main__":
    unittest.main()
