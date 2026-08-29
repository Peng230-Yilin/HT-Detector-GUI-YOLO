from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QVBoxLayout,
    QWidget,
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
        self.resize(700, 650)
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

        detection_tab = QWidget(self)
        detection_layout = QVBoxLayout(detection_tab)
        detection_layout.addWidget(detection_group)
        detection_layout.addWidget(roi_group)
        detection_layout.addWidget(concentration_group)
        detection_layout.addWidget(display_group)
        detection_layout.addStretch()

        calibration_tab = QWidget(self)
        calibration_layout = QVBoxLayout(calibration_tab)
        calibration_layout.addWidget(
            QLabel(
                "Rows correspond to calibration samples from left to right.",
                calibration_tab,
            )
        )
        self.calibration_table = QTableWidget(0, 2, calibration_tab)
        self.calibration_table.setHorizontalHeaderLabels(
            ("Concentration", "Use in regression")
        )
        self.calibration_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        self.calibration_table.setSelectionMode(
            QTableWidget.SelectionMode.ExtendedSelection
        )
        header = self.calibration_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        calibration_layout.addWidget(self.calibration_table)

        self.add_row_button = QPushButton("Add Row", calibration_tab)
        self.remove_row_button = QPushButton("Remove Selected Row", calibration_tab)
        self.add_row_button.clicked.connect(self._add_calibration_row)
        self.remove_row_button.clicked.connect(self._remove_selected_calibration_rows)
        calibration_buttons = QHBoxLayout()
        calibration_buttons.addWidget(self.add_row_button)
        calibration_buttons.addWidget(self.remove_row_button)
        calibration_buttons.addStretch()
        calibration_layout.addLayout(calibration_buttons)

        self.tabs = QTabWidget(self)
        self.tabs.addTab(detection_tab, "Detection")
        self.tabs.addTab(calibration_tab, "Calibration")

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
        layout.addWidget(self.tabs)
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
        self._set_calibration_rows(
            settings["con_list"], settings["linear_formula_point_matrix"]
        )

    @staticmethod
    def _calibration_spin(value):
        control = QDoubleSpinBox()
        control.setRange(0.0, 1e12)
        control.setDecimals(8)
        control.setSingleStep(1.0)
        control.setValue(value)
        return control

    @staticmethod
    def _centered_checkbox(checked):
        container = QWidget()
        checkbox = QCheckBox(container)
        checkbox.setChecked(checked)
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(checkbox)
        return container, checkbox

    def _append_calibration_row(self, concentration, included=True):
        row = self.calibration_table.rowCount()
        self.calibration_table.insertRow(row)
        self.calibration_table.setCellWidget(
            row, 0, self._calibration_spin(concentration)
        )
        container, checkbox = self._centered_checkbox(included)
        container.setProperty("regression_checkbox", checkbox)
        self.calibration_table.setCellWidget(row, 1, container)

    def _set_calibration_rows(self, concentrations, point_matrix):
        self.calibration_table.setRowCount(0)
        for concentration, included in zip(concentrations, point_matrix):
            self._append_calibration_row(concentration, included)

    @Slot()
    def _add_calibration_row(self):
        if self.calibration_table.rowCount() >= 100:
            QMessageBox.warning(
                self, "Interface Settings", "A maximum of 100 calibration rows is allowed."
            )
            return
        if self.calibration_table.rowCount():
            previous = self.calibration_table.cellWidget(
                self.calibration_table.rowCount() - 1, 0
            ).value()
            concentration = min(1e12, previous + 1.0)
        else:
            concentration = 0.0
        self._append_calibration_row(concentration, True)

    @Slot()
    def _remove_selected_calibration_rows(self):
        selected_rows = sorted(
            {index.row() for index in self.calibration_table.selectedIndexes()},
            reverse=True,
        )
        removable_count = max(0, self.calibration_table.rowCount() - 2)
        for row in selected_rows[:removable_count]:
            self.calibration_table.removeRow(row)

    def _calibration_values(self):
        concentrations = []
        point_matrix = []
        for row in range(self.calibration_table.rowCount()):
            concentrations.append(self.calibration_table.cellWidget(row, 0).value())
            container = self.calibration_table.cellWidget(row, 1)
            checkbox = container.property("regression_checkbox")
            point_matrix.append(checkbox.isChecked())
        return concentrations, point_matrix

    def _settings(self):
        concentrations, point_matrix = self._calibration_values()
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
            "con_list": concentrations,
            "linear_formula_point_matrix": point_matrix,
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
