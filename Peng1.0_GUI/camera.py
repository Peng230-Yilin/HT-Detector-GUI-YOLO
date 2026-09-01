# This Python file uses the following encoding: utf-8

# Copyright (C) 2023 The Qt Company Ltd.
# SPDX-License-Identifier: LicenseRef-Qt-Commercial OR BSD-3-Clause

import weakref

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtMultimedia import QCamera, QMediaCaptureSession, QMediaDevices
from PySide6.QtWidgets import (
    QLabel,
    QMessageBox,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ui.ui_camera import Ui_Camera


class Camera(QWidget):
    shutdown_ready = Signal()

    STATE_OFF = "OFF"
    STATE_STARTING = "STARTING"
    STATE_ON = "ON"
    STATE_STOPPING = "STOPPING"
    STATE_DISABLED = "DISABLED"
    STATE_SHUTTING_DOWN = "SHUTTING_DOWN"

    def __init__(self, camera_enabled):
        super().__init__()
        self.camera_enabled = bool(camera_enabled)
        self.m_devices = None
        self.m_camera = None
        self.m_captureSession = None
        self.m_audioInput = None
        self.m_mediaRecorder = None
        self.m_imageCapture = None
        self._camera_generation = 0
        self._camera_connections = []
        self._camera_state = (
            self.STATE_OFF if self.camera_enabled else self.STATE_DISABLED
        )
        self._close_wait_pending = False
        self._shutdown_requested = False
        self._shutdown_ready_emitted = False
        self._last_camera_diagnostic = None
        self._last_cleanup_errors = ()

        self._ui = Ui_Camera()
        self._ui.setupUi(self)
        self._install_camera_controls()
        self._disable_deferred_camera_features()
        self._set_camera_state(self._camera_state)

    def _install_camera_controls(self):
        self._ui.horizontalLayout_8.removeWidget(self._ui.groupBox)
        self._ui.horizontalLayout_8.removeWidget(self._ui.captureWidget)

        splitter = QSplitter(Qt.Horizontal, self._ui.cameraWidget)
        splitter.setObjectName("cameraSettingsSplitter")
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(6)
        splitter.addWidget(self._ui.groupBox)

        controls = QWidget(splitter)
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        self._camera_toggle_button = QPushButton(controls)
        self._camera_toggle_button.setObjectName("cameraToggleButton")
        self._camera_toggle_button.clicked.connect(self._toggle_camera)
        controls_layout.addWidget(self._camera_toggle_button)
        controls_layout.addWidget(self._ui.captureWidget)
        splitter.addWidget(controls)
        splitter.setStretchFactor(0, 5)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([500, 100])

        self._status_page = QWidget(self._ui.stackedWidget)
        status_layout = QVBoxLayout(self._status_page)
        self._camera_status_message = QLabel(self._status_page)
        self._camera_status_message.setObjectName("cameraStatusMessage")
        self._camera_status_message.setAlignment(Qt.AlignCenter)
        self._camera_status_message.setWordWrap(True)
        status_layout.addWidget(self._camera_status_message)
        self._status_page_index = self._ui.stackedWidget.addWidget(
            self._status_page
        )

        self._ui.verticalLayout_2.removeItem(self._ui.horizontalLayout_8)
        self._ui.verticalLayout_2.addWidget(splitter)

    def _disable_deferred_camera_features(self):
        tooltip = "Available in v1.3"
        controls = (
            self._ui.takeImageButton,
            self._ui.recordButton,
            self._ui.pauseButton,
            self._ui.stopButton,
            self._ui.muteButton,
            self._ui.metaDataButton,
            self._ui.imageResolutionBox,
            self._ui.imageCodecBox,
            self._ui.imageQualitySlider,
            self._ui.exposureCompensation,
        )
        for control in controls:
            control.setEnabled(False)
            control.setToolTip(tooltip)
        self._ui.captureWidget.setEnabled(False)
        self._ui.captureWidget.setToolTip(tooltip)

    def _set_camera_state(self, state):
        self._camera_state = state
        button_text = {
            self.STATE_OFF: "Start Camera",
            self.STATE_STARTING: "Starting Camera...",
            self.STATE_ON: "Stop Camera",
            self.STATE_STOPPING: "Stopping Camera...",
            self.STATE_DISABLED: "Camera Unavailable",
            self.STATE_SHUTTING_DOWN: "Shutting Down...",
        }[state]
        self._camera_toggle_button.setText(button_text)
        self._camera_toggle_button.setEnabled(
            self.camera_enabled
            and not self._shutdown_requested
            and state in (self.STATE_OFF, self.STATE_STARTING, self.STATE_ON)
        )

        if state == self.STATE_ON:
            self._ui.stackedWidget.setCurrentWidget(self._ui.viewfinderPage)
            return

        messages = {
            self.STATE_OFF: "Camera is off.\n摄像头已关闭。",
            self.STATE_STARTING: "Starting camera...\n正在启动摄像头……",
            self.STATE_STOPPING: "Stopping camera...\n正在关闭摄像头……",
            self.STATE_DISABLED: (
                "Camera is available only in the main window.\n"
                "摄像头仅在主窗口中可用。"
            ),
            self.STATE_SHUTTING_DOWN: (
                "Shutting down camera...\n正在关闭摄像头……"
            ),
        }
        self._camera_status_message.setText(messages[state])
        self._ui.stackedWidget.setCurrentIndex(self._status_page_index)

    @Slot()
    def _toggle_camera(self):
        if self._camera_state == self.STATE_OFF:
            self.start_camera()
        elif self._camera_state in (self.STATE_STARTING, self.STATE_ON):
            self.stop_camera()

    @Slot()
    def start_camera(self):
        if (
            not self.camera_enabled
            or self._shutdown_requested
            or self._camera_state != self.STATE_OFF
        ):
            return

        self._set_camera_state(self.STATE_STARTING)
        try:
            camera_device = QMediaDevices.defaultVideoInput()
            if camera_device.isNull():
                raise RuntimeError("No default camera is available.")
            self._create_preview_objects(camera_device)
            camera = self.m_camera
            if camera is None:
                raise RuntimeError("Camera initialization did not complete.")
            camera.start()
        except Exception as error:
            self._last_camera_diagnostic = (None, repr(error))
            self._finish_camera_stop(show_error=True)

    def _create_preview_objects(self, camera_device):
        session = None
        camera = None
        try:
            session = QMediaCaptureSession(self)
            camera = QCamera(camera_device, self)
            session.setVideoOutput(self._ui.viewfinder)
            session.setCamera(camera)

            self._camera_generation += 1
            generation = self._camera_generation
            self.m_captureSession = session
            self.m_camera = camera
            self._connect_camera_signals(camera, generation)
        except Exception:
            cleanup_errors = []
            self._camera_generation += 1
            connections = self._camera_connections
            self._camera_connections = []
            for signal, handler in connections:
                try:
                    signal.disconnect(handler)
                except (RuntimeError, TypeError) as error:
                    cleanup_errors.append(repr(error))
            self.m_camera = None
            self.m_captureSession = None
            if session is not None:
                try:
                    session.setCamera(None)
                    session.setVideoOutput(None)
                except Exception as error:
                    cleanup_errors.append(repr(error))
            for obj in (camera, session):
                if obj is not None:
                    try:
                        obj.deleteLater()
                    except Exception as error:
                        cleanup_errors.append(repr(error))
            self._last_cleanup_errors = tuple(cleanup_errors)
            raise

    def _connect_camera_signals(self, camera, generation):
        owner_ref = weakref.ref(self)
        camera_ref = weakref.ref(camera)

        def active_changed(active):
            owner = owner_ref()
            source = camera_ref()
            if owner is None or source is None:
                return
            if generation != owner._camera_generation:
                return
            if source is not owner.m_camera:
                return
            owner._on_camera_active_changed(bool(active))

        def error_occurred(*_args):
            owner = owner_ref()
            source = camera_ref()
            if owner is None or source is None:
                return
            if generation != owner._camera_generation:
                return
            if source is not owner.m_camera:
                return
            owner._on_camera_error(source)

        camera.activeChanged.connect(active_changed)
        try:
            camera.errorOccurred.connect(error_occurred)
        except Exception:
            camera.activeChanged.disconnect(active_changed)
            raise
        self._camera_connections = (
            (camera.activeChanged, active_changed),
            (camera.errorOccurred, error_occurred),
        )

    @Slot(bool)
    def _on_camera_active_changed(self, active):
        if active:
            if self._shutdown_requested or self._camera_state in (
                self.STATE_STOPPING,
                self.STATE_SHUTTING_DOWN,
            ):
                camera = self.m_camera
                if camera is not None:
                    camera.stop()
                return
            if self._camera_state == self.STATE_STARTING:
                self._set_camera_state(self.STATE_ON)
            return

        if self._camera_state == self.STATE_STARTING:
            self._finish_camera_stop(show_error=True)
        elif self._camera_state in (
            self.STATE_ON,
            self.STATE_STOPPING,
            self.STATE_SHUTTING_DOWN,
        ):
            self._finish_camera_stop(show_error=False)

    @Slot()
    def stop_camera(self):
        if self._shutdown_requested or self._camera_state not in (
            self.STATE_STARTING,
            self.STATE_ON,
        ):
            return
        self._set_camera_state(self.STATE_STOPPING)
        self._request_camera_stop()

    def _request_camera_stop(self):
        camera = self.m_camera
        if camera is None:
            self._finish_camera_stop(show_error=False)
            return
        try:
            camera.stop()
            if camera is self.m_camera and not camera.isActive():
                self._finish_camera_stop(show_error=False)
        except Exception as error:
            self._last_camera_diagnostic = (None, repr(error))
            self._finish_camera_stop(show_error=False)

    def _on_camera_error(self, camera):
        try:
            diagnostic = (camera.error(), camera.errorString())
        except RuntimeError as error:
            diagnostic = (None, repr(error))
        self._last_camera_diagnostic = diagnostic
        self._finish_camera_stop(show_error=True)

    def _finish_camera_stop(self, show_error):
        self._release_preview_objects()
        if self._shutdown_requested:
            self._set_camera_state(self.STATE_SHUTTING_DOWN)
            self._emit_shutdown_ready_once()
            return
        self._set_camera_state(self.STATE_OFF)
        if show_error:
            self._camera_status_message.setText(
                "Camera is unavailable or already in use.\n"
                "摄像头不可用或正在被其他程序占用。"
            )
            if not self._modal_errors_suppressed():
                QMessageBox.warning(
                    self,
                    "Camera Error / 摄像头错误",
                    "Camera is unavailable or already in use.\n"
                    "摄像头不可用或正在被其他程序占用。",
                )

    def _release_preview_objects(self):
        self._camera_generation += 1
        cleanup_errors = []
        connections = self._camera_connections
        self._camera_connections = []
        for signal, handler in connections:
            try:
                signal.disconnect(handler)
            except (RuntimeError, TypeError) as error:
                cleanup_errors.append(repr(error))

        camera = self.m_camera
        session = self.m_captureSession
        self.m_camera = None
        self.m_captureSession = None
        if session is not None:
            for setter in (session.setCamera, session.setVideoOutput):
                try:
                    setter(None)
                except Exception as error:
                    cleanup_errors.append(repr(error))
        if camera is not None:
            try:
                if camera.isActive():
                    camera.stop()
            except Exception as error:
                cleanup_errors.append(repr(error))
        for obj in (camera, session):
            if obj is not None:
                try:
                    obj.deleteLater()
                except Exception as error:
                    cleanup_errors.append(repr(error))
        self._last_cleanup_errors = tuple(cleanup_errors)

    @Slot()
    def startCamera(self):
        self.start_camera()

    @Slot()
    def stopCamera(self):
        self.stop_camera()

    @Slot(bool)
    def set_close_wait_pending(self, pending):
        self._close_wait_pending = bool(pending)

    def _modal_errors_suppressed(self):
        return self._close_wait_pending or self._shutdown_requested

    @Slot()
    def request_shutdown(self):
        if self._shutdown_requested:
            if self.m_camera is None:
                self._emit_shutdown_ready_once()
            return
        self._shutdown_requested = True
        self._set_camera_state(self.STATE_SHUTTING_DOWN)
        if self.m_camera is None:
            self._emit_shutdown_ready_once()
            return
        self._request_camera_stop()

    def _emit_shutdown_ready_once(self):
        if self._shutdown_ready_emitted:
            return
        self._shutdown_ready_emitted = True
        self.shutdown_ready.emit()
