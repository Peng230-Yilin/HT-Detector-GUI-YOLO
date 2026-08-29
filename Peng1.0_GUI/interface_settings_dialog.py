from PySide6.QtCore import Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from interface_config import (
    TEXT_ORDERS,
    default_settings,
    load_effective_settings,
    save_settings,
    validate_settings,
)


class InterfaceSettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Interface Settings")
        self.setModal(True)
        self._defaults = default_settings()
        current, warnings, _ = load_effective_settings(apply_to_module=False)

        self.detect_confidence = self._double_spin(0.001, 1.0, 0.01, 3)
        self.show_confidence = QCheckBox(self)
        detection_group = self._group(
            "Detection",
            (
                ("Detection confidence", self.detect_confidence),
                ("Show confidence", self.show_confidence),
            ),
        )

        self.x0_ratio = self._double_spin(0.0, 1.0, 0.01, 4)
        self.y0_ratio = self._double_spin(0.0, 1.0, 0.01, 4)
        self.x1_ratio = self._double_spin(0.0, 1.0, 0.01, 4)
        self.y1_ratio = self._double_spin(0.0, 1.0, 0.01, 4)
        roi_group = self._group(
            "RGB ROI",
            (
                ("Left ratio", self.x0_ratio),
                ("Top ratio", self.y0_ratio),
                ("Right ratio", self.x1_ratio),
                ("Bottom ratio", self.y1_ratio),
            ),
        )

        self.color_channel = QComboBox(self)
        self.color_channel.addItems(("R", "G", "B"))
        self.rgb_calculate_accuracy = self._integer_spin(0, 16)
        self.rgb_display_accuracy = self._integer_spin(0, 6)
        self.con_display_accuracy = self._integer_spin(0, 6)
        concentration_group = self._group(
            "Concentration",
            (
                ("Color channel", self.color_channel),
                ("RGB calculation precision", self.rgb_calculate_accuracy),
                ("RGB display decimals", self.rgb_display_accuracy),
                ("Concentration display decimals", self.con_display_accuracy),
            ),
        )

        self.text_order = QComboBox(self)
        self.text_order.addItems(TEXT_ORDERS)
        display_group = self._group("Display", (("Text order", self.text_order),))

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
            | QDialogButtonBox.StandardButton.RestoreDefaults,
            parent=self,
        )
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        restore_button = self.buttons.button(
            QDialogButtonBox.StandardButton.RestoreDefaults
        )
        restore_button.clicked.connect(self._restore_defaults)

        layout = QVBoxLayout(self)
        layout.addWidget(detection_group)
        layout.addWidget(roi_group)
        layout.addWidget(concentration_group)
        layout.addWidget(display_group)
        layout.addWidget(self.buttons)
        self._set_values(current)

        if warnings:
            QMessageBox.warning(self, "Interface Settings", "\n".join(warnings))

    @staticmethod
    def _double_spin(minimum, maximum, step, decimals):
        control = QDoubleSpinBox()
        control.setRange(minimum, maximum)
        control.setSingleStep(step)
        control.setDecimals(decimals)
        return control

    @staticmethod
    def _integer_spin(minimum, maximum):
        control = QSpinBox()
        control.setRange(minimum, maximum)
        return control

    @staticmethod
    def _group(title, rows):
        group = QGroupBox(title)
        layout = QFormLayout(group)
        for label, control in rows:
            layout.addRow(label, control)
        return group

    def _set_values(self, settings):
        self.detect_confidence.setValue(settings["detect_confidence"])
        self.show_confidence.setChecked(settings["show_confidence"])
        self.x0_ratio.setValue(settings["x0_ratio"])
        self.y0_ratio.setValue(settings["y0_ratio"])
        self.x1_ratio.setValue(settings["x1_ratio"])
        self.y1_ratio.setValue(settings["y1_ratio"])
        self.color_channel.setCurrentText(settings["color_channel"])
        self.rgb_calculate_accuracy.setValue(settings["rgb_calculate_accuracy"])
        self.rgb_display_accuracy.setValue(settings["rgb_display_accuracy"])
        self.con_display_accuracy.setValue(settings["con_display_accuracy"])
        self.text_order.setCurrentText(settings["Order_Con_R_G_B"])

    def _settings(self):
        return {
            "detect_confidence": self.detect_confidence.value(),
            "show_confidence": self.show_confidence.isChecked(),
            "x0_ratio": self.x0_ratio.value(),
            "y0_ratio": self.y0_ratio.value(),
            "x1_ratio": self.x1_ratio.value(),
            "y1_ratio": self.y1_ratio.value(),
            "color_channel": self.color_channel.currentText(),
            "rgb_calculate_accuracy": self.rgb_calculate_accuracy.value(),
            "rgb_display_accuracy": self.rgb_display_accuracy.value(),
            "con_display_accuracy": self.con_display_accuracy.value(),
            "Order_Con_R_G_B": self.text_order.currentText(),
        }

    @Slot()
    def _restore_defaults(self):
        self._set_values(self._defaults)

    def accept(self):
        try:
            settings = validate_settings(self._settings())
            save_settings(settings)
        except (RuntimeError, ValueError) as error:
            QMessageBox.critical(self, "Interface Settings", str(error))
            return
        QMessageBox.information(
            self,
            "Interface Settings",
            "Settings saved. Changes will take effect on the next detection.",
        )
        super().accept()
