import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[1]
GUI_ROOT = ROOT / "Peng1.0_GUI"
UI_ROOT = GUI_ROOT / "ui"
sys.path.insert(0, str(GUI_ROOT))

from PySide6.QtGui import QActionGroup  # noqa: E402
from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton  # noqa: E402
import detectionwindow  # noqa: E402
from ui.ui_detectwindow import Ui_detectWindow  # noqa: E402


SINGLE_IMAGE = "single_image"
IMAGE_SERIES = "image_series"


class FakeDetectMain:
    def __init__(self):
        self.busy = False
        self._linear_mode = SINGLE_IMAGE
        self.mode_calls = []
        self.fail_mode = None

    @property
    def linear_mode(self):
        return self._linear_mode

    def is_worker_task_active(self):
        return self.busy

    def is_linear_interaction_locked(self):
        return self.busy

    def set_linear_mode(self, mode):
        self.mode_calls.append(mode)
        # Deliberately mutate before raising.  The window-level transaction must
        # restore both the QAction state and its collaborator's runtime state.
        self._linear_mode = mode
        if mode == self.fail_mode:
            raise RuntimeError("linear mode application failed")


class LinearMenuHarness(QMainWindow):
    _setup_linear_menu = detectionwindow.DetectWindow._setup_linear_menu
    _set_linear_menu_checked_state = (
        detectionwindow.DetectWindow._set_linear_menu_checked_state
    )
    _apply_linear_mode = detectionwindow.DetectWindow._apply_linear_mode
    _set_detection_menu_enabled = (
        detectionwindow.DetectWindow._set_detection_menu_enabled
    )

    def __init__(self):
        super().__init__()
        self._uiWindow = Ui_detectWindow()
        self._uiWindow.setupUi(self)
        self._detectMain = FakeDetectMain()

        # Build the existing groups without invoking their configuration-backed
        # setup.  This lets the isolation assertions guard all config access.
        self._detection_scope_group = QActionGroup(self)
        self._detection_scope_group.setExclusive(True)
        self._detection_scope_group.addAction(
            self._uiWindow.actionDetection_Current_Image
        )
        self._detection_scope_group.addAction(
            self._uiWindow.actionDetection_Entire_Batch
        )
        self._numbering_group = QActionGroup(self)
        self._numbering_group.setExclusive(True)
        self._numbering_group.addAction(
            self._uiWindow.actionDetection_Per_Image
        )
        self._numbering_group.addAction(
            self._uiWindow.actionDetection_Continuous
        )


class LinearControlsHarness:
    _apply_linear_mode_controls = (
        detectionwindow.DetectMain._apply_linear_mode_controls
    )

    def __init__(self):
        self.ui = SimpleNamespace(pushButton_4=QPushButton("Original Linear Action"))
        self._single_linear_action_text = self.ui.pushButton_4.text()
        self._linear_mode = SINGLE_IMAGE
        self._active_worker_task = None
        self.refresh_calls = []

    def _set_active_worker_task(self, task):
        self.refresh_calls.append(task)


class LinearMenuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def make_window(self):
        window = LinearMenuHarness()
        self.addCleanup(window.close)
        window._setup_linear_menu()
        return window

    def test_ui_contains_top_level_linear_menu_and_exact_actions(self):
        window = QMainWindow()
        self.addCleanup(window.close)
        ui = Ui_detectWindow()
        ui.setupUi(window)

        self.assertEqual(ui.menuLinear.objectName(), "menuLinear")
        self.assertEqual(ui.menuLinear.title(), "Linear")
        self.assertIn(ui.menuLinear.menuAction(), ui.menubar.actions())
        self.assertEqual(
            ui.menuLinear.actions(),
            [ui.actionLinear_Single_Image, ui.actionLinear_Image_Series],
        )
        self.assertEqual(
            ui.actionLinear_Single_Image.objectName(),
            "actionLinear_Single_Image",
        )
        self.assertEqual(
            ui.actionLinear_Image_Series.objectName(),
            "actionLinear_Image_Series",
        )
        self.assertEqual(ui.actionLinear_Single_Image.text(), "Single Image")
        self.assertEqual(ui.actionLinear_Image_Series.text(), "Image Series")
        self.assertTrue(ui.actionLinear_Single_Image.isCheckable())
        self.assertTrue(ui.actionLinear_Image_Series.isCheckable())

    def test_group_is_exclusive_and_isolated_from_detection_groups(self):
        window = self.make_window()
        linear_actions = set(window._linear_mode_group.actions())
        detection_actions = set(window._detection_scope_group.actions())
        numbering_actions = set(window._numbering_group.actions())

        self.assertTrue(window._linear_mode_group.isExclusive())
        self.assertEqual(
            linear_actions,
            {
                window._uiWindow.actionLinear_Single_Image,
                window._uiWindow.actionLinear_Image_Series,
            },
        )
        self.assertTrue(linear_actions.isdisjoint(detection_actions))
        self.assertTrue(linear_actions.isdisjoint(numbering_actions))
        self.assertIs(
            window._uiWindow.actionLinear_Single_Image.actionGroup(),
            window._linear_mode_group,
        )
        self.assertIs(
            window._uiWindow.actionLinear_Image_Series.actionGroup(),
            window._linear_mode_group,
        )

        detection_state = tuple(
            action.isChecked()
            for action in (*detection_actions, *numbering_actions)
        )
        window._uiWindow.actionLinear_Image_Series.trigger()
        self.app.processEvents()
        self.assertEqual(
            tuple(
                action.isChecked()
                for action in (*detection_actions, *numbering_actions)
            ),
            detection_state,
        )

    def test_every_window_defaults_to_single_image_without_persistence(self):
        first = self.make_window()
        self.assertTrue(first._uiWindow.actionLinear_Single_Image.isChecked())
        self.assertFalse(first._uiWindow.actionLinear_Image_Series.isChecked())
        self.assertEqual(first._detectMain.linear_mode, SINGLE_IMAGE)

        first._uiWindow.actionLinear_Image_Series.trigger()
        self.app.processEvents()
        self.assertEqual(first._detectMain.linear_mode, IMAGE_SERIES)

        second = self.make_window()
        self.assertTrue(second._uiWindow.actionLinear_Single_Image.isChecked())
        self.assertFalse(second._uiWindow.actionLinear_Image_Series.isChecked())
        self.assertEqual(second._detectMain.linear_mode, SINGLE_IMAGE)

    def test_linear_setup_never_reads_or_writes_detection_config(self):
        with patch.object(
            detectionwindow,
            "load_detection_preferences",
            side_effect=AssertionError("Linear setup read detection config"),
        ) as load_config, patch.object(
            detectionwindow,
            "save_detection_preferences",
            side_effect=AssertionError("Linear setup wrote detection config"),
        ) as save_config:
            window = self.make_window()
            self.assertTrue(window._uiWindow.actionLinear_Single_Image.isChecked())

        load_config.assert_not_called()
        save_config.assert_not_called()

    def test_apply_failure_rolls_back_action_and_runtime_atomically(self):
        window = self.make_window()
        window._detectMain.mode_calls.clear()
        window._detectMain.fail_mode = IMAGE_SERIES

        with patch.object(
            detectionwindow.QMessageBox, "warning", return_value=None
        ), patch.object(
            detectionwindow.QMessageBox, "critical", return_value=None
        ):
            window._uiWindow.actionLinear_Image_Series.trigger()
            self.app.processEvents()

        self.assertTrue(window._uiWindow.actionLinear_Single_Image.isChecked())
        self.assertFalse(window._uiWindow.actionLinear_Image_Series.isChecked())
        self.assertEqual(window._detectMain.linear_mode, SINGLE_IMAGE)
        self.assertIn(IMAGE_SERIES, window._detectMain.mode_calls)

    def test_busy_disables_whole_menu_rejects_switch_and_then_restores(self):
        window = self.make_window()
        window._detectMain.mode_calls.clear()
        window._detectMain.busy = True
        window._set_detection_menu_enabled(True)

        self.assertFalse(window._uiWindow.menuLinear.isEnabled())
        self.assertFalse(window._uiWindow.menuLinear.menuAction().isEnabled())
        window._uiWindow.actionLinear_Image_Series.trigger()
        self.app.processEvents()
        self.assertTrue(window._uiWindow.actionLinear_Single_Image.isChecked())
        self.assertFalse(window._uiWindow.actionLinear_Image_Series.isChecked())
        self.assertEqual(window._detectMain.linear_mode, SINGLE_IMAGE)
        self.assertEqual(window._detectMain.mode_calls, [])

        window._detectMain.busy = False
        window._set_detection_menu_enabled(False)
        self.assertTrue(window._uiWindow.menuLinear.isEnabled())
        self.assertTrue(window._uiWindow.menuLinear.menuAction().isEnabled())

    def test_direct_handler_rejects_every_authoritative_lock_state(self):
        window = self.make_window()
        for lock_state in (
            "detection",
            "regression",
            "linear_series",
            "save_linear",
            "save_detection",
            "pending_close",
            "shutdown",
        ):
            with self.subTest(lock_state=lock_state):
                window._detectMain.busy = True
                window._detectMain.mode_calls.clear()
                before = (
                    window._linear_mode,
                    window._detectMain.linear_mode,
                    window._uiWindow.actionLinear_Single_Image.isChecked(),
                    window._uiWindow.actionLinear_Image_Series.isChecked(),
                )
                window._uiWindow.actionLinear_Image_Series.setChecked(True)
                self.assertFalse(
                    window._apply_linear_mode(
                        window._uiWindow.actionLinear_Image_Series
                    )
                )
                self.assertEqual(window._detectMain.mode_calls, [])
                self.assertEqual((
                    window._linear_mode,
                    window._detectMain.linear_mode,
                    window._uiWindow.actionLinear_Single_Image.isChecked(),
                    window._uiWindow.actionLinear_Image_Series.isChecked(),
                ), before)

    def test_detect_main_uses_series_text_and_restores_original_single_text(self):
        controls = LinearControlsHarness()
        controls._linear_mode = IMAGE_SERIES
        controls._apply_linear_mode_controls()
        self.assertEqual(controls.ui.pushButton_4.text(), "Extract Series")

        controls._linear_mode = SINGLE_IMAGE
        controls._apply_linear_mode_controls()
        self.assertEqual(controls.ui.pushButton_4.text(), "Original Linear Action")
        self.assertEqual(controls.refresh_calls, [None, None])

    def test_generated_python_is_exactly_in_sync_with_ui_source(self):
        uic = ROOT / ".venv" / "Scripts" / "pyside6-uic.exe"
        self.assertTrue(uic.is_file(), "Project pyside6-uic executable is missing")
        committed = UI_ROOT / "ui_detectwindow.py"

        environment = os.environ.copy()
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        result = subprocess.run(
            [str(uic), "detectwindow.ui"],
            cwd=str(UI_ROOT),
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr or result.stdout)
        generated_text = result.stdout

        generated_resource_import = "import img_rc"
        project_resource_import = "import rc_img"
        self.assertEqual(generated_text.count(generated_resource_import), 1)
        self.assertNotIn(project_resource_import, generated_text)
        convention_adjusted = generated_text.replace(
            generated_resource_import,
            project_resource_import,
        )
        committed_text = committed.read_text(encoding="utf-8")
        self.assertEqual(committed_text.count(project_resource_import), 1)
        self.assertEqual(
            convention_adjusted.splitlines(),
            committed_text.splitlines(),
        )


if __name__ == "__main__":
    unittest.main()
