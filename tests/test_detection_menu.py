import json
import os
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Peng1.0_GUI"))

from PySide6.QtWidgets import QApplication, QMainWindow  # noqa: E402
import detectionwindow  # noqa: E402
import interface_config  # noqa: E402
from ui.ui_detectwindow import Ui_detectWindow  # noqa: E402


class FakeDetectMain:
    def __init__(self):
        self.options = []
        self.detection_scope = None
        self.numbering_mode = None
        self.busy = False

    def set_detection_options(self, scope, numbering):
        self.options.append((scope, numbering))
        self.detection_scope = scope
        self.numbering_mode = numbering

    def is_worker_task_active(self):
        return self.busy


class MenuHarness(QMainWindow):
    _setup_detection_menu = detectionwindow.DetectWindow._setup_detection_menu
    _set_detection_menu_checked_state = (
        detectionwindow.DetectWindow._set_detection_menu_checked_state
    )
    _save_detection_menu_settings = detectionwindow.DetectWindow._save_detection_menu_settings
    _set_detection_menu_enabled = detectionwindow.DetectWindow._set_detection_menu_enabled

    def __init__(self):
        super().__init__()
        self._uiWindow = Ui_detectWindow()
        self._uiWindow.setupUi(self)
        self._detectMain = FakeDetectMain()


class DetectionMenuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def defaults(self):
        return dict(interface_config.DETECTION_PREFERENCES_DEFAULTS)

    def assert_menu_state(self, window, scope, numbering):
        self.assertEqual(
            window._uiWindow.actionDetection_Current_Image.isChecked(),
            scope == "current_image",
        )
        self.assertEqual(
            window._uiWindow.actionDetection_Entire_Batch.isChecked(),
            scope == "entire_batch",
        )
        self.assertEqual(
            window._uiWindow.actionDetection_Per_Image.isChecked(),
            numbering == "per_image",
        )
        self.assertEqual(
            window._uiWindow.actionDetection_Continuous.isChecked(),
            numbering == "continuous",
        )

    def assert_failed_commit_is_atomic(self, action_name, attempted_settings):
        previous_settings = self.defaults()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "detection_preferences.json"
            interface_config.save_detection_preferences(
                previous_settings, config_file=path
            )
            original_file = path.read_bytes()
            real_save_file = interface_config.QSaveFile
            commit_calls = []
            save_calls = []

            class CommitFailingSaveFile:
                def __init__(self, file_name):
                    self._delegate = real_save_file(file_name)

                def open(self, mode):
                    return self._delegate.open(mode)

                def write(self, data):
                    return self._delegate.write(data)

                def cancelWriting(self):
                    return self._delegate.cancelWriting()

                def commit(self):
                    commit_calls.append(True)
                    self._delegate.cancelWriting()
                    return False

            def save_to_temporary_file(settings):
                save_calls.append(dict(settings))
                return interface_config.save_detection_preferences(
                    settings, config_file=path
                )

            with patch.object(
                detectionwindow,
                "load_detection_preferences",
                return_value=(dict(previous_settings), []),
            ), patch.object(
                detectionwindow,
                "save_detection_preferences",
                side_effect=save_to_temporary_file,
            ), patch.object(
                interface_config, "QSaveFile", CommitFailingSaveFile
            ), patch.object(
                detectionwindow.QMessageBox, "warning", return_value=None
            ) as warning:
                window = MenuHarness()
                window._setup_detection_menu()
                getattr(window._uiWindow, action_name).trigger()
                self.app.processEvents()

                self.assert_menu_state(
                    window,
                    previous_settings["detection_scope"],
                    previous_settings["numbering_mode"],
                )
                self.assertEqual(
                    window._detectMain.detection_scope,
                    previous_settings["detection_scope"],
                )
                self.assertEqual(
                    window._detectMain.numbering_mode,
                    previous_settings["numbering_mode"],
                )
                self.assertEqual(
                    window._detectMain.options,
                    [("current_image", "per_image")],
                )
                self.assertEqual(window._detection_menu_settings, previous_settings)
                self.assertEqual(save_calls, [attempted_settings])
                self.assertEqual(len(commit_calls), 1)
                warning.assert_called_once()
                self.assertEqual(path.read_bytes(), original_file)
                loaded, warnings = interface_config.load_detection_preferences(
                    config_file=path
                )
                self.assertEqual(warnings, [])
                self.assertEqual(loaded, previous_settings)
                window.close()

    def test_ui_contains_expected_detection_menu(self):
        window = QMainWindow()
        ui = Ui_detectWindow()
        ui.setupUi(window)
        self.assertEqual(ui.menuDetection.title(), "Detection")
        self.assertEqual(ui.menuDetection_Scope.title(), "Detection Scope")
        self.assertEqual(ui.menuDetection_Numbering.title(), "Numbering")
        self.assertTrue(ui.actionDetection_Current_Image.isCheckable())
        self.assertTrue(ui.actionDetection_Per_Image.isCheckable())
        window.close()

    def test_defaults_groups_exclusivity_persistence_and_busy_disable(self):
        stored = []
        with patch.object(detectionwindow, "load_detection_preferences",
                          return_value=(self.defaults(), [])), \
                patch.object(detectionwindow, "save_detection_preferences",
                             side_effect=lambda value: stored.append(dict(value))):
            window = MenuHarness()
            window._setup_detection_menu()
            self.assertTrue(window._detection_scope_group.isExclusive())
            self.assertTrue(window._numbering_group.isExclusive())
            self.assertTrue(window._uiWindow.actionDetection_Current_Image.isChecked())
            self.assertTrue(window._uiWindow.actionDetection_Per_Image.isChecked())
            window._uiWindow.actionDetection_Entire_Batch.trigger()
            window._uiWindow.actionDetection_Continuous.trigger()
            self.assertTrue(window._uiWindow.actionDetection_Entire_Batch.isChecked())
            self.assertFalse(window._uiWindow.actionDetection_Current_Image.isChecked())
            self.assertEqual(stored[-1]["detection_scope"], "entire_batch")
            self.assertEqual(stored[-1]["numbering_mode"], "continuous")
            self.assertEqual(
                window._detectMain.options[-1], ("entire_batch", "continuous")
            )
            window._set_detection_menu_enabled(True)
            self.assertFalse(window._uiWindow.menuDetection_Scope.isEnabled())
            self.assertFalse(window._uiWindow.menuDetection_Numbering.isEnabled())
            window._set_detection_menu_enabled(False)
            self.assertTrue(window._uiWindow.menuDetection_Scope.isEnabled())
            window.close()

    def test_scope_commit_failure_rolls_back_ui_runtime_and_file_once(self):
        self.assert_failed_commit_is_atomic(
            "actionDetection_Entire_Batch",
            {"detection_scope": "entire_batch", "numbering_mode": "per_image"},
        )

    def test_numbering_commit_failure_rolls_back_ui_runtime_and_file_once(self):
        self.assert_failed_commit_is_atomic(
            "actionDetection_Continuous",
            {"detection_scope": "current_image", "numbering_mode": "continuous"},
        )

    def test_config_round_trip_uses_temporary_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            settings = {"detection_scope": "entire_batch", "numbering_mode": "continuous"}
            interface_config.save_detection_preferences(settings, config_file=path)
            loaded, warnings = interface_config.load_detection_preferences(config_file=path)
            self.assertEqual(warnings, [])
            self.assertEqual(loaded["detection_scope"], "entire_batch")
            self.assertEqual(loaded["numbering_mode"], "continuous")

    def test_invalid_config_falls_back_to_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            path.write_text(json.dumps({
                "detection_scope": "invalid", "numbering_mode": "per_image",
            }), encoding="utf-8")
            loaded, warnings = interface_config.load_detection_preferences(config_file=path)
            self.assertEqual(loaded["detection_scope"], "current_image")
            self.assertEqual(loaded["numbering_mode"], "per_image")
            self.assertEqual(len(warnings), 1)


if __name__ == "__main__":
    unittest.main()
