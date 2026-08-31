# This Python file uses the following encoding: utf-8

# Copyright (C) 2023 The Qt Company Ltd.
# SPDX-License-Identifier: LicenseRef-Qt-Commercial OR BSD-3-Clause

import os
import sys
import weakref
from pathlib import Path

from PySide6.QtMultimedia import (QAudioInput, QCamera, QCameraDevice,
                                  QImageCapture, QMediaCaptureSession,
                                  QMediaDevices, QMediaMetaData,
                                  QMediaRecorder)
from PySide6.QtWidgets import (QDialog, QLabel, QMainWindow, QMessageBox,
                               QPushButton, QSplitter, QVBoxLayout, QWidget)
from PySide6.QtGui import QAction, QActionGroup, QIcon, QImage, QPixmap
from PySide6.QtCore import (QDateTime, QDir, QObject, QSignalBlocker,
                            QTimer, Qt, Signal, Slot, qWarning)

#from metadatadialog import MetaDataDialog
#from imagesettings import ImageSettings
#from videosettings import VideoSettings, is_android

#if is_android or sys.platform == "darwin":
#    from PySide6.QtCore import QMicrophonePermission, QCameraPermission

#if is_android:
#    from ui_camera_mobile import Ui_Camera
#else:
#    from ui.ui_camera import Ui_Camera

from ui.ui_camera import Ui_Camera



from PySide6.QtMultimedia import QImageCapture
from PySide6.QtWidgets import QDialog
from PySide6.QtCore import QSize

#from ui_imagesettings import Ui_ImageSettingsUi



class Camera(QWidget):
    shutdown_ready = Signal()
    STATE_OFF = "OFF"
    STATE_STARTING = "STARTING"
    STATE_ON = "ON"
    STATE_STOPPING = "STOPPING"
    STATE_DISABLED = "DISABLED"
    STATE_SHUTTING_DOWN = "SHUTTING_DOWN"
    INIT_NOT_INITIALIZED = "NOT_INITIALIZED"
    INIT_INITIALIZING = "INITIALIZING"
    INIT_INITIALIZED = "INITIALIZED"

    def __init__(self, camera_enabled):
        super().__init__()

        print("init_start_________--------")
        self.camera_enabled = bool(camera_enabled)
        self._video_devices_group = None
        self.m_devices = None
        self.m_imageCapture = None
        self.m_captureSession = None
        self.m_camera = None
        self.m_mediaRecorder = None
        self.m_audioInput = None

        self.m_isCapturingImage = False
        self.m_applicationExiting = False
        self.m_doImageCapture = True
        self._close_wait_pending = False
        self._shutdown_requested = False
        self._shutdown_ready_emitted = False
        self._media_initialization_state = self.INIT_NOT_INITIALIZED
        self._start_pending = False
        self._start_cancel_requested = False
        self._cancelled_start_camera = None
        self._recorder_terminal_error = False
        self._last_camera_diagnostic = None
        self._last_recorder_diagnostic = None
        self._last_capture_diagnostic = None
        self._last_initialization_error = None
        self._last_cleanup_errors = ()
        self._generation_counter = 0
        self._media_generation = None
        self._camera_generation_counter = 0
        self._camera_generation = None
        self._signal_connections = []
        self._camera_device_actions = []
        self._device_action_connections = []
        self._device_actions_generation = None
        self._cancel_token_counter = 0
        self._active_cancel_token = None
        self._preview_timer_counter = 0
        self._active_preview_timer_token = None
        self._camera_candidate_counter = 0
        self._active_camera_candidate_token = None
        self._camera_state = (
            self.STATE_OFF if self.camera_enabled else self.STATE_DISABLED
        )

        self.m_metaDataDialog = None

        self._ui = Ui_Camera()
        self._ui.setupUi(self)
        self._install_resizable_splitter()
        self._ui.captureWidget.currentChanged.connect(self.updateCaptureMode)
        self._ui.metaDataButton.clicked.connect(self.showMetaDataDialog)
        self._ui.exposureCompensation.valueChanged.connect(
            self.setExposureCompensation
        )

        image = Path(__file__).parent / "shutter.svg"
        self._ui.takeImageButton.setIcon(QIcon(os.fspath(image)))
#        if not is_android:
#            self._ui.actionAbout_Qt.triggered.connect(qApp.aboutQt)  # noqa: F821

        # disable all buttons by default
        self.updateCameraActive(False)
        self.readyForCapture(False)
        self._ui.recordButton.setEnabled(False)
        self._ui.pauseButton.setEnabled(False)
        self._ui.stopButton.setEnabled(False)
        self._ui.metaDataButton.setEnabled(False)

        if self.camera_enabled:
            self._set_camera_state(self.STATE_OFF)
        else:
            self._configure_disabled_camera_ui()

    def _configure_disabled_camera_ui(self):
        self._camera_toggle_button.setEnabled(False)
        self._set_camera_state(self.STATE_DISABLED)

    def _install_resizable_splitter(self):
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

        self._camera_status_message = QLabel(self._ui.viewfinderPage)
        self._camera_status_message.setObjectName("cameraStatusMessage")
        self._camera_status_message.setAlignment(Qt.AlignCenter)
        self._camera_status_message.setWordWrap(True)
        self._ui.verticalLayout_3.addWidget(self._camera_status_message)

        self._ui.verticalLayout_2.removeItem(self._ui.horizontalLayout_8)
        self._ui.verticalLayout_2.addWidget(splitter)

    def _next_media_generation(self):
        self._generation_counter += 1
        return self._generation_counter

    def _next_camera_generation(self):
        self._camera_generation_counter += 1
        return self._camera_generation_counter

    def _invalidate_preview_restore(self):
        self._preview_timer_counter += 1
        self._active_preview_timer_token = None

    def _schedule_preview_restore(self):
        self._preview_timer_counter += 1
        timer_token = self._preview_timer_counter
        self._active_preview_timer_token = timer_token
        generation = self._media_generation
        camera_generation = self._camera_generation
        capture_ref = weakref.ref(self.m_imageCapture)
        owner_ref = weakref.ref(self)

        def restore_if_current():
            owner = owner_ref()
            capture = capture_ref()
            if owner is None or capture is None:
                return
            if timer_token != owner._active_preview_timer_token:
                return
            if generation != owner._media_generation:
                return
            if camera_generation != owner._camera_generation:
                return
            if capture is not owner.m_imageCapture:
                return
            if owner._camera_state != owner.STATE_ON:
                return
            owner._active_preview_timer_token = None
            owner.displayViewfinder()

        QTimer.singleShot(4000, restore_if_current)

    def _connect_media_signal(
        self,
        signal,
        sender,
        member_name,
        handler_name,
        generation,
        camera_generation=None,
        pass_arguments=True,
    ):
        owner_ref = weakref.ref(self)
        sender_ref = weakref.ref(sender)

        def guarded_handler(*args):
            owner = owner_ref()
            source = sender_ref()
            if owner is None or source is None:
                return
            if generation != owner._media_generation:
                return
            if getattr(owner, member_name, None) is not source:
                return
            if (
                camera_generation is not None
                and camera_generation != owner._camera_generation
            ):
                return
            handler = getattr(owner, handler_name)
            if pass_arguments:
                handler(*args)
            else:
                handler()

        signal.connect(guarded_handler)
        self._signal_connections.append(
            (signal, guarded_handler, sender_ref, generation)
        )

    def _disconnect_media_signals(self, sender=None):
        remaining = []
        errors = []
        for signal, handler, source_ref, generation in self._signal_connections:
            source = source_ref()
            if sender is not None and source is not sender:
                remaining.append((signal, handler, source_ref, generation))
                continue
            try:
                signal.disconnect(handler)
            except Exception as error:
                errors.append(repr(error))
        self._signal_connections = remaining
        return errors

    def _set_camera_state(self, state):
        self._camera_state = state
        active = state == self.STATE_ON
        self._ui.captureWidget.setEnabled(active)
        self._ui.viewfinder.setVisible(active)
        self._camera_status_message.setVisible(not active)
        if state == self.STATE_DISABLED:
            self._camera_toggle_button.setText("Camera Unavailable")
            self._camera_status_message.setText(
                "Camera is available only in the main window.\n"
                "摄像头仅在主窗口中可用。"
            )
        elif state == self.STATE_STARTING:
            self._camera_toggle_button.setText("Starting Camera...")
            self._camera_status_message.setText(
                "Starting camera...\n正在启动摄像头……"
            )
        elif state == self.STATE_STOPPING:
            self._camera_toggle_button.setText("Stopping Camera...")
        elif state == self.STATE_SHUTTING_DOWN:
            self._camera_toggle_button.setText("Shutting Down...")
        else:
            self._camera_toggle_button.setText(
                "Stop Camera"
                if active
                else "Start Camera"
            )
            if not active:
                self._camera_status_message.setText(
                    "Camera is off.\n摄像头已关闭。"
                )
        self._camera_toggle_button.setEnabled(
            self.camera_enabled
            and state not in (
                self.STATE_DISABLED,
                self.STATE_STOPPING,
                self.STATE_SHUTTING_DOWN,
            )
        )

    @Slot()
    def _toggle_camera(self):
        if self._camera_state in (self.STATE_OFF,):
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
            self.initialize()
            if self.m_camera is None:
                raise RuntimeError("No camera is available.")
            self._start_pending = True
            self._start_cancel_requested = False
            self.m_camera.start()
        except Exception as error:
            self._last_initialization_error = repr(error)
            self._start_pending = False
            self._set_camera_unavailable()
            if not self._modal_errors_suppressed():
                self._show_camera_error()

    @Slot()
    def stop_camera(self):
        if self._shutdown_requested or self._camera_state in (
            self.STATE_DISABLED,
            self.STATE_OFF,
            self.STATE_STOPPING,
            self.STATE_SHUTTING_DOWN,
        ):
            return
        self._set_camera_state(self.STATE_STOPPING)
        if self._start_pending:
            self._cancel_pending_start()
            return
        self._continue_camera_stop()

    def _continue_camera_stop(self):
        if self._start_pending:
            return
        if self.m_isCapturingImage:
            return
        if (
            self.m_mediaRecorder is not None
            and not self._recorder_terminal_error
            and self.m_mediaRecorder.recorderState() != QMediaRecorder.StoppedState
        ):
            self.m_mediaRecorder.stop()
            return
        if self.m_camera is not None and self.m_camera.isActive():
            self.m_camera.stop()
            return
        if not self._shutdown_requested:
            self._set_camera_state(self.STATE_OFF)

    def _user_media_operation_allowed(self):
        return bool(
            self.camera_enabled
            and not self._shutdown_requested
            and self._camera_state == self.STATE_ON
            and self.m_camera is not None
            and self.m_camera.isActive()
        )

    def _cancel_pending_start(self):
        if not self._start_pending or self._start_cancel_requested:
            return
        self._start_cancel_requested = True
        self._cancel_token_counter += 1
        cancel_token = self._cancel_token_counter
        self._active_cancel_token = cancel_token
        camera = self.m_camera
        self._cancelled_start_camera = camera
        if camera is not None:
            owner_ref = weakref.ref(self)

            def on_destroyed():
                owner = owner_ref()
                if owner is not None:
                    owner._on_cancelled_start_camera_destroyed(cancel_token)

            camera.destroyed.connect(on_destroyed)
        self._cleanup_media_objects()
        if camera is None:
            self._queue_pending_start_cancel_finish(cancel_token)

    def _on_cancelled_start_camera_destroyed(self, cancel_token):
        self._queue_pending_start_cancel_finish(cancel_token)

    def _queue_pending_start_cancel_finish(self, cancel_token):
        owner_ref = weakref.ref(self)

        def finish_if_current():
            owner = owner_ref()
            if owner is not None:
                owner._finish_pending_start_cancel(cancel_token)

        QTimer.singleShot(0, finish_if_current)

    def _finish_pending_start_cancel(self, cancel_token):
        if (
            not self._start_cancel_requested
            or cancel_token != self._active_cancel_token
        ):
            return
        self._active_cancel_token = None
        self._cancelled_start_camera = None
        self._start_pending = False
        self._start_cancel_requested = False
        if self._shutdown_requested:
            self._maybe_finish_shutdown()
        else:
            self._set_camera_state(self.STATE_OFF)

    def _cleanup_media_objects(self):
        cleanup_errors = []
        self._invalidate_preview_restore()
        self._active_camera_candidate_token = None
        self._media_generation = None
        self._camera_generation = None
        cleanup_errors.extend(self._disconnect_media_signals())
        cleanup_errors.extend(self._clear_camera_device_actions())
        session = self.m_captureSession
        camera = self.m_camera
        recorder = self.m_mediaRecorder
        objects = (
            self.m_imageCapture,
            recorder,
            camera,
            self.m_audioInput,
            session,
            self.m_devices,
            self._video_devices_group,
        )
        if session is not None:
            for setter in (
                session.setImageCapture,
                session.setRecorder,
                session.setCamera,
                session.setAudioInput,
                session.setVideoOutput,
            ):
                try:
                    setter(None)
                except Exception as error:
                    cleanup_errors.append(repr(error))
        if recorder is not None:
            try:
                recorder.stop()
            except Exception as error:
                cleanup_errors.append(repr(error))
        if camera is not None:
            try:
                camera.stop()
            except Exception as error:
                cleanup_errors.append(repr(error))
        seen = set()
        for obj in objects:
            if obj is None or id(obj) in seen:
                continue
            seen.add(id(obj))
            try:
                obj.deleteLater()
            except Exception as error:
                cleanup_errors.append(repr(error))
        self.m_imageCapture = None
        self.m_mediaRecorder = None
        self.m_camera = None
        self.m_audioInput = None
        self.m_captureSession = None
        self.m_devices = None
        self._video_devices_group = None
        self.m_isCapturingImage = False
        self._recorder_terminal_error = False
        self._media_initialization_state = self.INIT_NOT_INITIALIZED
        self._last_cleanup_errors = tuple(cleanup_errors)

    @staticmethod
    def _snapshot_button(button):
        return {
            "enabled": not button.testAttribute(Qt.WA_ForceDisabled),
            "checked": button.isChecked() if button.isCheckable() else None,
            "text": button.text(),
        }

    @staticmethod
    def _restore_button(button, snapshot):
        blocker = QSignalBlocker(button)
        button.setText(snapshot["text"])
        if snapshot["checked"] is not None:
            button.setChecked(snapshot["checked"])
        button.setEnabled(snapshot["enabled"])
        del blocker

    @staticmethod
    def _snapshot_slider(slider):
        return {
            "minimum": slider.minimum(),
            "maximum": slider.maximum(),
            "value": slider.value(),
            "enabled": not slider.testAttribute(Qt.WA_ForceDisabled),
        }

    @staticmethod
    def _restore_slider(slider, snapshot):
        blocker = QSignalBlocker(slider)
        slider.setRange(snapshot["minimum"], snapshot["maximum"])
        slider.setValue(snapshot["value"])
        slider.setEnabled(snapshot["enabled"])
        del blocker

    def _snapshot_initialization_state(self):
        buttons = {
            name: self._snapshot_button(getattr(self._ui, name))
            for name in (
                "recordButton",
                "pauseButton",
                "stopButton",
                "metaDataButton",
                "takeImageButton",
            )
        }
        return {
            "buttons": buttons,
            "toggle": self._snapshot_button(self._camera_toggle_button),
            "codec": self._snapshot_combo_box(self._ui.imageCodecBox),
            "resolution": self._snapshot_combo_box(self._ui.imageResolutionBox),
            "quality": self._snapshot_slider(self._ui.imageQualitySlider),
            "exposure": self._snapshot_slider(self._ui.exposureCompensation),
            "capture_enabled": not self._ui.captureWidget.testAttribute(
                Qt.WA_ForceDisabled
            ),
            "capture_index": self._ui.captureWidget.currentIndex(),
            "stacked_index": self._ui.stackedWidget.currentIndex(),
            "viewfinder_hidden": self._ui.viewfinder.isHidden(),
            "status_hidden": self._camera_status_message.isHidden(),
            "status_text": self._camera_status_message.text(),
            "camera_state": self._camera_state,
            "do_image_capture": self.m_doImageCapture,
            "is_capturing_image": self.m_isCapturingImage,
            "recorder_terminal_error": self._recorder_terminal_error,
            "start_pending": self._start_pending,
            "start_cancel_requested": self._start_cancel_requested,
        }

    def _restore_initialization_state(self, snapshot):
        for name, state in snapshot["buttons"].items():
            self._restore_button(getattr(self._ui, name), state)
        self._restore_button(self._camera_toggle_button, snapshot["toggle"])
        self._restore_combo_box(self._ui.imageCodecBox, snapshot["codec"])
        self._restore_combo_box(
            self._ui.imageResolutionBox, snapshot["resolution"]
        )
        self._restore_slider(self._ui.imageQualitySlider, snapshot["quality"])
        self._restore_slider(self._ui.exposureCompensation, snapshot["exposure"])
        blockers = (
            QSignalBlocker(self._ui.captureWidget),
            QSignalBlocker(self._ui.stackedWidget),
        )
        self._ui.captureWidget.setCurrentIndex(snapshot["capture_index"])
        self._ui.captureWidget.setEnabled(snapshot["capture_enabled"])
        self._ui.stackedWidget.setCurrentIndex(snapshot["stacked_index"])
        self._ui.viewfinder.setVisible(not snapshot["viewfinder_hidden"])
        self._camera_status_message.setText(snapshot["status_text"])
        self._camera_status_message.setVisible(not snapshot["status_hidden"])
        self._camera_state = snapshot["camera_state"]
        self.m_doImageCapture = snapshot["do_image_capture"]
        self.m_isCapturingImage = snapshot["is_capturing_image"]
        self._recorder_terminal_error = snapshot["recorder_terminal_error"]
        self._start_pending = snapshot["start_pending"]
        self._start_cancel_requested = snapshot["start_cancel_requested"]
        del blockers

    @Slot()
    def initialize(self):
        if not self.camera_enabled:
            return
        if self._media_initialization_state == self.INIT_INITIALIZED:
            return
        if self._media_initialization_state == self.INIT_INITIALIZING:
            raise RuntimeError("Camera media initialization is already in progress.")
        initialization_snapshot = self._snapshot_initialization_state()
        self._media_initialization_state = self.INIT_INITIALIZING
        generation = self._next_media_generation()
        camera_generation = self._next_camera_generation()
        self._media_generation = generation
        self._camera_generation = camera_generation
        try:
            self.m_devices = QMediaDevices(self)
            self.m_captureSession = QMediaCaptureSession(self)
            self.m_audioInput = QAudioInput(self)
            self.m_camera = QCamera(QMediaDevices.defaultVideoInput(), self)
            self.m_mediaRecorder = QMediaRecorder(self)
            self.m_imageCapture = QImageCapture(self)
            self._video_devices_group = QActionGroup(self)
            self._video_devices_group.setExclusive(True)

            self.m_captureSession.setAudioInput(self.m_audioInput)
            self.m_captureSession.setCamera(self.m_camera)
            self.m_captureSession.setRecorder(self.m_mediaRecorder)
            self.m_captureSession.setImageCapture(self.m_imageCapture)
            self.m_captureSession.setVideoOutput(self._ui.viewfinder)

            self._connect_media_signal(
                self.m_devices.videoInputsChanged,
                self.m_devices,
                "m_devices",
                "_handle_devices_changed",
                generation,
                pass_arguments=False,
            )
            self._connect_media_signal(
                self.m_camera.activeChanged,
                self.m_camera,
                "m_camera",
                "updateCameraActive",
                generation,
                camera_generation,
            )
            self._connect_media_signal(
                self.m_camera.errorOccurred,
                self.m_camera,
                "m_camera",
                "displayCameraError",
                generation,
                camera_generation,
                pass_arguments=False,
            )
            self._connect_media_signal(
                self.m_mediaRecorder.recorderStateChanged,
                self.m_mediaRecorder,
                "m_mediaRecorder",
                "updateRecorderState",
                generation,
            )
            self._connect_media_signal(
                self.m_mediaRecorder.durationChanged,
                self.m_mediaRecorder,
                "m_mediaRecorder",
                "updateRecordTime",
                generation,
                pass_arguments=False,
            )
            self._connect_media_signal(
                self.m_mediaRecorder.errorChanged,
                self.m_mediaRecorder,
                "m_mediaRecorder",
                "displayRecorderError",
                generation,
                pass_arguments=False,
            )
            self._connect_media_signal(
                self.m_imageCapture.readyForCaptureChanged,
                self.m_imageCapture,
                "m_imageCapture",
                "readyForCapture",
                generation,
            )
            self._connect_media_signal(
                self.m_imageCapture.imageCaptured,
                self.m_imageCapture,
                "m_imageCapture",
                "processCapturedImage",
                generation,
            )
            self._connect_media_signal(
                self.m_imageCapture.imageSaved,
                self.m_imageCapture,
                "m_imageCapture",
                "imageSaved",
                generation,
            )
            self._connect_media_signal(
                self.m_imageCapture.errorOccurred,
                self.m_imageCapture,
                "m_imageCapture",
                "displayCaptureError",
                generation,
            )

            self._update_cameras(generation)
            self.updateRecorderState(self.m_mediaRecorder.recorderState())
            self.readyForCapture(self.m_imageCapture.isReadyForCapture())
            self.m_doImageCapture = self._ui.captureWidget.currentIndex() == 0
            image_settings = self._collect_image_settings()
            self._commit_image_settings(image_settings)
        except Exception as error:
            self._last_initialization_error = repr(error)
            self._cleanup_media_objects()
            self._restore_initialization_state(initialization_snapshot)
            self._media_initialization_state = self.INIT_NOT_INITIALIZED
            raise
        self._recorder_terminal_error = False
        self._media_initialization_state = self.INIT_INITIALIZED

    @Slot(QCameraDevice)
    def setCamera(self, cameraDevice):
        if (
            not self._user_media_operation_allowed()
            or self.m_captureSession is None
        ):
            return
        old_camera = self.m_camera
        old_camera_generation = self._camera_generation
        old_state = self._camera_state
        old_start_pending = self._start_pending
        old_session_camera = self.m_captureSession.camera()
        old_ui = {
            "codec": self._snapshot_combo_box(self._ui.imageCodecBox),
            "resolution": self._snapshot_combo_box(
                self._ui.imageResolutionBox
            ),
            "quality": self._snapshot_slider(self._ui.imageQualitySlider),
        }
        self._camera_candidate_counter += 1
        candidate_token = self._camera_candidate_counter
        self._active_camera_candidate_token = candidate_token
        candidate_generation = self._next_camera_generation()
        candidate_events = {
            "active_seen": False,
            "active": False,
            "error": False,
        }
        candidate_connections = []
        new_camera = None

        def restore_old_ui():
            self._restore_combo_box(self._ui.imageCodecBox, old_ui["codec"])
            self._restore_combo_box(
                self._ui.imageResolutionBox, old_ui["resolution"]
            )
            self._restore_slider(self._ui.imageQualitySlider, old_ui["quality"])

        def disconnect_candidate_connections():
            errors = []
            for signal, handler in candidate_connections:
                try:
                    signal.disconnect(handler)
                except Exception as error:
                    errors.append(repr(error))
            candidate_connections.clear()
            return errors

        try:
            new_camera = QCamera(cameraDevice, self)
            owner_ref = weakref.ref(self)
            candidate_ref = weakref.ref(new_camera)

            def candidate_active_changed(active):
                owner = owner_ref()
                candidate = candidate_ref()
                if owner is None or candidate is None:
                    return
                if candidate_token != owner._active_camera_candidate_token:
                    return
                candidate_events["active_seen"] = True
                candidate_events["active"] = bool(active)

            def candidate_error_occurred(*_args):
                owner = owner_ref()
                candidate = candidate_ref()
                if owner is None or candidate is None:
                    return
                if candidate_token != owner._active_camera_candidate_token:
                    return
                candidate_events["error"] = True

            new_camera.activeChanged.connect(candidate_active_changed)
            candidate_connections.append(
                (new_camera.activeChanged, candidate_active_changed)
            )
            new_camera.errorOccurred.connect(candidate_error_occurred)
            candidate_connections.append(
                (new_camera.errorOccurred, candidate_error_occurred)
            )

            # These normal wrappers remain dormant until the member and camera
            # generation are committed together below.
            self._connect_media_signal(
                new_camera.activeChanged,
                new_camera,
                "m_camera",
                "updateCameraActive",
                self._media_generation,
                candidate_generation,
            )
            self._connect_media_signal(
                new_camera.errorOccurred,
                new_camera,
                "m_camera",
                "displayCameraError",
                self._media_generation,
                candidate_generation,
                pass_arguments=False,
            )

            self.m_captureSession.setCamera(new_camera)
            new_settings = self._collect_image_settings()
            self._commit_image_settings(new_settings)
            new_camera.start()
            if candidate_events["error"] or new_camera.error() != QCamera.NoError:
                raise RuntimeError("Candidate camera reported a synchronous error.")
            if candidate_events["active_seen"] and not candidate_events["active"]:
                raise RuntimeError("Candidate camera synchronously became inactive.")
        except Exception:
            self._active_camera_candidate_token = None
            cleanup_errors = disconnect_candidate_connections()
            rollback_failed = False
            if new_camera is not None:
                cleanup_errors.extend(self._disconnect_media_signals(new_camera))
            try:
                self.m_captureSession.setCamera(old_session_camera)
            except Exception as error:
                cleanup_errors.append(repr(error))
                rollback_failed = True
            try:
                restore_old_ui()
            except Exception as error:
                cleanup_errors.append(repr(error))
                rollback_failed = True
            self.m_camera = old_camera
            self._camera_generation = old_camera_generation
            self._start_pending = old_start_pending
            if new_camera is not None:
                try:
                    new_camera.stop()
                except Exception as error:
                    cleanup_errors.append(repr(error))
                    rollback_failed = True
                try:
                    new_camera.deleteLater()
                except Exception as error:
                    cleanup_errors.append(repr(error))
                    rollback_failed = True
            if rollback_failed:
                self._start_pending = False
                self._set_camera_state(self.STATE_OFF)
                if old_camera is not None:
                    try:
                        old_camera.stop()
                    except Exception as error:
                        cleanup_errors.append(repr(error))
            else:
                self._camera_state = old_state
            self._last_cleanup_errors = tuple(cleanup_errors)
            raise

        # Formal commit boundary: after this pair changes, candidate wrappers
        # become the only current Camera wrappers and old wrappers become stale.
        self._camera_generation = candidate_generation
        self.m_camera = new_camera
        self._active_camera_candidate_token = None
        cleanup_errors = disconnect_candidate_connections()
        cleanup_errors.extend(self._disconnect_media_signals(old_camera))
        self._invalidate_preview_restore()
        if old_camera is not None:
            try:
                old_camera.stop()
            except Exception as error:
                cleanup_errors.append(repr(error))
            try:
                old_camera.deleteLater()
            except Exception as error:
                cleanup_errors.append(repr(error))
        self._last_cleanup_errors = tuple(cleanup_errors)
        if candidate_events["active"] or new_camera.isActive():
            self._start_pending = False
            self._set_camera_state(self.STATE_ON)
            self._update_cameras(self._media_generation)
        else:
            self._start_pending = True
            self._set_camera_state(self.STATE_STARTING)


    def keyPressEvent(self, event):
        if not self._user_media_operation_allowed():
            super().keyPressEvent(event)
            return
        if event.isAutoRepeat():
            return

        key = event.key()
        if key == Qt.Key_CameraFocus:
            self.displayViewfinder()
            event.accept()
        elif key == Qt.Key_Camera:
            if self.m_doImageCapture:
                self.takeImage()
            else:
                if self.m_mediaRecorder.recorderState() == QMediaRecorder.RecordingState:
                    self.stop()
                else:
                    self.record()

            event.accept()
        else:
            super().keyPressEvent(event)

    @Slot()
    def updateRecordTime(self):
        if (
            not self._user_media_operation_allowed()
            or self.m_mediaRecorder is None
        ):
            return
        d = self.m_mediaRecorder.duration() / 1000
        self._show_status_message(f"Recorded {d} sec")

    def _show_status_message(self, message):
        statusbar = getattr(self._ui, "statusbar", None)
        if statusbar is not None:
            statusbar.showMessage(message)

    @Slot(int, QImage)
    def processCapturedImage(self, requestId, img):
        scaled_image = img.scaled(self._ui.viewfinder.size(), Qt.KeepAspectRatio,
                                  Qt.SmoothTransformation)

        self._ui.lastImagePreviewLabel.setPixmap(QPixmap.fromImage(scaled_image))

        # Display captured image for 4 seconds. The guarded callback cannot
        # mutate a later media graph or a later capture in this generation.
        self.displayCapturedImage()
        self._schedule_preview_restore()

    @Slot()
    def configureCaptureSettings(self):
        if not self._user_media_operation_allowed():
            return
        if self.m_doImageCapture:
            self.configureImageSettings()
        else:
            self.configureVideoSettings()

    @Slot()
    def configureVideoSettings(self):
        if (
            not self._user_media_operation_allowed()
            or self.m_mediaRecorder is None
        ):
            return
        settings_dialog = VideoSettings(self.m_mediaRecorder)

        if settings_dialog.exec():
            settings_dialog.apply_settings()

    @Slot()
    def configureImageSettings(self):
        if (
            not self._user_media_operation_allowed()
            or self.m_imageCapture is None
        ):
            return
        self._configure_image_settings()

    def _configure_image_settings(self):
        if self.m_imageCapture is None:
            raise RuntimeError("Image capture is not initialized.")
        self._commit_image_settings(self._collect_image_settings())

    def _collect_image_settings(self):
        image_capture = self.m_imageCapture
        if image_capture is None:
            raise RuntimeError("Image capture is not initialized.")
        formats = [("Default image format", QImageCapture.UnspecifiedFormat)]
        for file_format in QImageCapture.supportedFormats():
            description = QImageCapture.fileFormatDescription(file_format)
            name = QImageCapture.fileFormatName(file_format)
            formats.append((f"{name} : {description}", file_format))
        camera = image_capture.captureSession().camera()
        if camera is None:
            raise RuntimeError("Camera is not attached to image capture.")
        resolutions = [("Default Resolution", QSize())]
        for resolution in camera.cameraDevice().photoResolutions():
            resolutions.append(
                (f"{resolution.width()}x{resolution.height()}", resolution)
            )
        return {
            "formats": tuple(formats),
            "resolutions": tuple(resolutions),
            "format": image_capture.fileFormat(),
            "resolution": image_capture.resolution(),
            "quality": image_capture.quality().value,
            "quality_minimum": 0,
            "quality_maximum": QImageCapture.VeryHighQuality.value,
        }

    @staticmethod
    def _snapshot_combo_box(combo_box):
        model = combo_box.model()
        items = []
        for index in range(combo_box.count()):
            model_index = model.index(index, combo_box.modelColumn())
            items.append(
                (
                    combo_box.itemText(index),
                    combo_box.itemIcon(index),
                    dict(model.itemData(model_index)),
                )
            )
        return {
            "items": tuple(items),
            "current_index": combo_box.currentIndex(),
            "enabled": not combo_box.testAttribute(Qt.WA_ForceDisabled),
        }

    @staticmethod
    def _restore_combo_box(combo_box, snapshot):
        blocker = QSignalBlocker(combo_box)
        combo_box.clear()
        model = combo_box.model()
        for text, icon, roles in snapshot["items"]:
            combo_box.addItem(icon, text)
            model_index = model.index(combo_box.count() - 1, combo_box.modelColumn())
            for role, value in roles.items():
                model.setData(model_index, value, role)
        combo_box.setCurrentIndex(snapshot["current_index"])
        combo_box.setEnabled(snapshot["enabled"])
        del blocker

    def _commit_image_settings(self, settings):
        codec_box = self._ui.imageCodecBox
        resolution_box = self._ui.imageResolutionBox
        quality_slider = self._ui.imageQualitySlider
        codec_snapshot = self._snapshot_combo_box(codec_box)
        resolution_snapshot = self._snapshot_combo_box(resolution_box)
        slider_snapshot = self._snapshot_slider(quality_slider)
        blockers = (
            QSignalBlocker(codec_box),
            QSignalBlocker(resolution_box),
            QSignalBlocker(quality_slider),
        )
        try:
            codec_box.clear()
            for text, value in settings["formats"]:
                codec_box.addItem(text, value)
            resolution_box.clear()
            for text, value in settings["resolutions"]:
                resolution_box.addItem(text, value)
            quality_slider.setRange(
                settings["quality_minimum"], settings["quality_maximum"]
            )
            self.select_combo_box_item(codec_box, settings["format"])
            self.select_combo_box_item(
                resolution_box, settings["resolution"]
            )
            quality_slider.setValue(settings["quality"])
            self.m_imageCapture.setFileFormat(self.box_value(codec_box))
            self.m_imageCapture.setQuality(
                QImageCapture.Quality(quality_slider.value())
            )
            self.m_imageCapture.setResolution(
                self.box_value(resolution_box)
            )
        except Exception:
            self._restore_combo_box(codec_box, codec_snapshot)
            self._restore_combo_box(resolution_box, resolution_snapshot)
            self._restore_slider(quality_slider, slider_snapshot)
            raise
        finally:
            del blockers

#        settings_dialog = ImageSettings(self.m_imageCapture)

#        if settings_dialog.exec():
#            settings_dialog.apply_image_settings()

    @Slot()
    def record(self):
        if (
            not self._user_media_operation_allowed()
            or self.m_mediaRecorder is None
        ):
            return
        self._recorder_terminal_error = False
        self.m_mediaRecorder.record()
        self.updateRecordTime()

    @Slot()
    def pause(self):
        if (
            not self._user_media_operation_allowed()
            or self.m_mediaRecorder is None
        ):
            return
        self.m_mediaRecorder.pause()

    @Slot()
    def stop(self):
        if (
            not self._user_media_operation_allowed()
            or self.m_mediaRecorder is None
        ):
            return
        self.m_mediaRecorder.stop()

    @Slot(bool)
    def setMuted(self, muted):
        if self._user_media_operation_allowed() and self.m_audioInput is not None:
            self.m_audioInput.setMuted(muted)

    @Slot()
    def takeImage(self):
        if (
            not self._user_media_operation_allowed()
            or self.m_imageCapture is None
        ):
            return
        self._invalidate_preview_restore()
        self.m_isCapturingImage = True
        self.m_imageCapture.captureToFile()

    @Slot(int, QImageCapture.Error, str)
    def displayCaptureError(self, id, error, errorString):
        self._last_capture_diagnostic = (error, errorString)
        if not self._modal_errors_suppressed():
            QMessageBox.warning(
                self,
                "Capture Error / 拍照错误",
                "Photo capture failed.\n照片拍摄失败。",
            )
        self.m_isCapturingImage = False
        if self._camera_state in (self.STATE_STOPPING, self.STATE_SHUTTING_DOWN):
            self._continue_camera_stop()
        self._maybe_finish_shutdown()
        if self.m_applicationExiting and not self._shutdown_requested:
            self.close()

    @Slot()
    def startCamera(self):
        self.start_camera()

    @Slot()
    def stopCamera(self):
        self.stop_camera()

    @Slot()
    def updateCaptureMode(self):
        if not self._user_media_operation_allowed():
            return
        tab_index = self._ui.captureWidget.currentIndex()
        self.m_doImageCapture = (tab_index == 0)

    @Slot(bool)
    def updateCameraActive(self, active):
        if active:
            if (
                self._shutdown_requested
                or self._start_cancel_requested
                or self._camera_state not in (
                self.STATE_STARTING,
                self.STATE_ON,
                )
            ):
                if self.m_camera is not None:
                    self.m_camera.stop()
            else:
                self._start_pending = False
                self._set_camera_state(self.STATE_ON)
                self._update_cameras(self._media_generation)
                if self.m_imageCapture is not None:
                    self.readyForCapture(
                        self.m_imageCapture.isReadyForCapture()
                    )
        else:
            if self._start_cancel_requested:
                pass
            elif self._start_pending:
                self._start_pending = False
                if self._shutdown_requested:
                    self._maybe_finish_shutdown()
                else:
                    self._set_camera_unavailable()
            elif self._camera_state not in (
                self.STATE_STARTING,
                self.STATE_DISABLED,
            ) and not self._shutdown_requested:
                self._set_camera_state(self.STATE_OFF)
        self._maybe_finish_shutdown()

    @Slot(QMediaRecorder.RecorderState)
    def updateRecorderState(self, state):
        if state == QMediaRecorder.StoppedState:
            self._recorder_terminal_error = False
            self._ui.recordButton.setEnabled(True)
            self._ui.pauseButton.setEnabled(False)
            self._ui.stopButton.setEnabled(False)
            self._ui.metaDataButton.setEnabled(True)
        elif state == QMediaRecorder.PausedState:
            self._ui.recordButton.setEnabled(True)
            self._ui.pauseButton.setEnabled(False)
            self._ui.stopButton.setEnabled(True)
            self._ui.metaDataButton.setEnabled(False)
        elif state == QMediaRecorder.RecordingState:
            self._ui.recordButton.setEnabled(False)
            self._ui.pauseButton.setEnabled(True)
            self._ui.stopButton.setEnabled(True)
            self._ui.metaDataButton.setEnabled(False)
        if (
            state == QMediaRecorder.StoppedState
            and self._camera_state in (self.STATE_STOPPING, self.STATE_SHUTTING_DOWN)
        ):
            self._continue_camera_stop()
        self._maybe_finish_shutdown()

    @Slot(int)
    def setExposureCompensation(self, index):
        if self._user_media_operation_allowed():
            self.m_camera.setExposureCompensation(index * 0.5)

    @Slot()
    def displayRecorderError(self):
        if (
            self.m_mediaRecorder is not None
            and self.m_mediaRecorder.error() != QMediaRecorder.NoError
        ):
            self._last_recorder_diagnostic = (
                self.m_mediaRecorder.error(),
                self.m_mediaRecorder.errorString(),
            )
            self._recorder_terminal_error = True
            if not self._modal_errors_suppressed():
                QMessageBox.warning(
                    self,
                    "Recorder Error / 录像错误",
                    "Recording could not be completed.\n无法完成录像。",
                )
            self._ui.recordButton.setEnabled(
                self._user_media_operation_allowed()
            )
            self._ui.pauseButton.setEnabled(False)
            self._ui.stopButton.setEnabled(False)
            if self._camera_state in (
                self.STATE_STOPPING,
                self.STATE_SHUTTING_DOWN,
            ):
                self._continue_camera_stop()
            self._maybe_finish_shutdown()

    @Slot()
    def displayCameraError(self):
        if self.m_camera is not None and self.m_camera.error() != QCamera.NoError:
            self._last_camera_diagnostic = (
                self.m_camera.error(),
                self.m_camera.errorString(),
            )
            if not self._modal_errors_suppressed():
                self._show_camera_error()
            if self._start_pending and not self._start_cancel_requested:
                self._start_pending = False
            if not self._shutdown_requested:
                self._set_camera_unavailable()
            elif self._start_cancel_requested:
                pass
            elif self.m_camera.isActive():
                self.m_camera.stop()
            self._maybe_finish_shutdown()

    def _set_camera_unavailable(self):
        self._set_camera_state(self.STATE_OFF)
        self._camera_status_message.setText(
            "Camera is unavailable or already in use.\n"
            "摄像头不可用或正在被其他程序占用。"
        )

    def _show_camera_error(self):
        QMessageBox.warning(
            self,
            "Camera Error / 摄像头错误",
            "Camera is unavailable or already in use.\n"
            "摄像头不可用或正在被其他程序占用。",
        )

    @Slot(QAction)
    def updateCameraDevice(self, action):
        if (
            self._user_media_operation_allowed()
            and action in self._camera_device_actions
            and self._device_actions_generation == self._media_generation
        ):
            self.setCamera(action.data())

    @Slot()
    def displayViewfinder(self):
        if not self._user_media_operation_allowed():
            return
        self._ui.stackedWidget.setCurrentIndex(0)

    @Slot()
    def displayCapturedImage(self):
        if not self._user_media_operation_allowed():
            return
        self._ui.stackedWidget.setCurrentIndex(1)

    @Slot(bool)
    def readyForCapture(self, ready):
        self._ui.takeImageButton.setEnabled(
            bool(ready) and self._user_media_operation_allowed()
        )

    @Slot(int, str)
    def imageSaved(self, id, fileName):
        f = QDir.toNativeSeparators(fileName)
        self._show_status_message(f"Captured \"{f}\"")

        self.m_isCapturingImage = False
        if self._camera_state in (self.STATE_STOPPING, self.STATE_SHUTTING_DOWN):
            self._continue_camera_stop()
        self._maybe_finish_shutdown()
        if self.m_applicationExiting and not self._shutdown_requested:
            self.close()

    @Slot()
    def request_shutdown(self):
        if self._shutdown_requested:
            self._maybe_finish_shutdown()
            return
        self._shutdown_requested = True
        self._invalidate_preview_restore()
        self.m_applicationExiting = True
        self._set_camera_state(self.STATE_SHUTTING_DOWN)
        if self._start_pending:
            self._cancel_pending_start()
            return
        self._continue_camera_stop()
        self._maybe_finish_shutdown()

    @Slot(bool)
    def set_close_wait_pending(self, pending):
        self._close_wait_pending = bool(pending)

    def _modal_errors_suppressed(self):
        return self._close_wait_pending or self._shutdown_requested

    def _maybe_finish_shutdown(self):
        if not getattr(self, "_shutdown_requested", False):
            return
        recorder_stopped = (
            self.m_mediaRecorder is None
            or self._recorder_terminal_error
            or self.m_mediaRecorder.recorderState() == QMediaRecorder.StoppedState
        )
        camera_stopped = self.m_camera is None or not self.m_camera.isActive()
        if (
            recorder_stopped
            and camera_stopped
            and not self.m_isCapturingImage
            and not self._start_pending
            and not self._start_cancel_requested
            and self._cancelled_start_camera is None
            and not self._shutdown_ready_emitted
        ):
            if self.m_captureSession is not None:
                self.m_captureSession.setImageCapture(None)
                self.m_captureSession.setRecorder(None)
                self.m_captureSession.setCamera(None)
                self.m_captureSession.setAudioInput(None)
                self.m_captureSession.setVideoOutput(None)
            self._shutdown_ready_emitted = True
            self.shutdown_ready.emit()

    def closeEvent(self, event):
        if self.m_isCapturingImage:
            self.setEnabled(False)
            self.m_applicationExiting = True
            event.ignore()
        else:
            event.accept()

    @Slot()
    def updateCameras(self):
        if not self._user_media_operation_allowed():
            return
        self._update_cameras(self._media_generation)

    def _handle_devices_changed(self):
        if self._user_media_operation_allowed():
            self._update_cameras(self._media_generation)

    def _clear_camera_device_actions(self):
        errors = []
        group = self._video_devices_group
        for action, handler in self._device_action_connections:
            try:
                action.triggered.disconnect(handler)
            except Exception as error:
                errors.append(repr(error))
        for action in self._camera_device_actions:
            if group is not None:
                group.removeAction(action)
            action.deleteLater()
        self._device_action_connections = []
        self._camera_device_actions = []
        self._device_actions_generation = None
        return errors

    def _update_cameras(self, generation):
        if (
            not self.camera_enabled
            or self._video_devices_group is None
            or generation != self._media_generation
        ):
            return
        available_cameras = QMediaDevices.videoInputs()
        current_device = None
        if self.m_camera is not None:
            current_device = self.m_camera.cameraDevice()
        if current_device is None or current_device.isNull():
            current_device = QMediaDevices.defaultVideoInput()
        new_actions = []
        new_connections = []
        owner_ref = weakref.ref(self)
        try:
            for camera_device in available_cameras:
                action = QAction(
                    camera_device.description(), self._video_devices_group
                )
                new_actions.append(action)
                action.setCheckable(True)
                action.setData(camera_device)
                action.setChecked(camera_device == current_device)
                action_ref = weakref.ref(action)

                def guarded_trigger(
                    _checked=False,
                    *,
                    action_ref=action_ref,
                    action_generation=generation,
                    owner_ref=owner_ref,
                ):
                    owner = owner_ref()
                    current_action = action_ref()
                    if owner is None or current_action is None:
                        return
                    if action_generation != owner._media_generation:
                        return
                    if action_generation != owner._device_actions_generation:
                        return
                    if current_action not in owner._camera_device_actions:
                        return
                    owner.updateCameraDevice(current_action)

                action.triggered.connect(guarded_trigger)
                new_connections.append((action, guarded_trigger))
        except Exception:
            for action, handler in new_connections:
                action.triggered.disconnect(handler)
            for action in new_actions:
                self._video_devices_group.removeAction(action)
                action.deleteLater()
            raise
        self._clear_camera_device_actions()
        self._camera_device_actions = new_actions
        self._device_action_connections = new_connections
        self._device_actions_generation = generation

    @Slot()
    def showMetaDataDialog(self):
        if (
            not self._user_media_operation_allowed()
            or self.m_mediaRecorder is None
        ):
            return
        if not self.m_metaDataDialog:
            self.m_metaDataDialog = MetaDataDialog(self)
        self.m_metaDataDialog.setAttribute(Qt.WA_DeleteOnClose, False)
        if self.m_metaDataDialog.exec() == QDialog.Accepted:
            self.saveMetaData()

    @Slot()
    def saveMetaData(self):
        if (
            not self._user_media_operation_allowed()
            or self.m_mediaRecorder is None
        ):
            return
        data = QMediaMetaData()
        for i in range(0, QMediaMetaData.NumMetaData):
            val = self.m_metaDataDialog.m_metaDataFields[i].text()
            if val:
                key = QMediaMetaData.Key(i)
                if key == QMediaMetaData.CoverArtImage:
                    cover_art = QImage(val)
                    data.insert(key, cover_art)
                elif key == QMediaMetaData.ThumbnailImage:
                    thumbnail = QImage(val)
                    data.insert(key, thumbnail)
                elif key == QMediaMetaData.Date:
                    date = QDateTime.fromString(val)
                    data.insert(key, date)
                else:
                    data.insert(key, val)

        self.m_mediaRecorder.setMetaData(data)

    def box_value(self, box):
        idx = box.currentIndex()
        return None if idx == -1 else box.itemData(idx)


    def select_combo_box_item(self, box, value):
        idx = box.findData(value)
        if idx != -1:
            box.setCurrentIndex(idx)

