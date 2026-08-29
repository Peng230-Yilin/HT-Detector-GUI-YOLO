# This Python file uses the following encoding: utf-8

# Copyright (C) 2023 The Qt Company Ltd.
# SPDX-License-Identifier: LicenseRef-Qt-Commercial OR BSD-3-Clause

import os
import sys
from pathlib import Path

from PySide6.QtMultimedia import (QAudioInput, QCamera, QCameraDevice,
                                  QImageCapture, QMediaCaptureSession,
                                  QMediaDevices, QMediaMetaData,
                                  QMediaRecorder)
from PySide6.QtWidgets import QDialog, QMainWindow, QMessageBox, QSplitter, QWidget
from PySide6.QtGui import QAction, QActionGroup, QIcon, QImage, QPixmap
from PySide6.QtCore import QDateTime, QDir, QTimer, Qt, Slot, qWarning, QObject

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
    def __init__(self):
        super().__init__()

        print("init_start_________--------")
        self._video_devices_group = None
        self.m_devices = QMediaDevices()
        self.m_imageCapture = None
        self.m_captureSession = QMediaCaptureSession()
        self.m_camera = None
        self.m_mediaRecorder = None

        self.m_isCapturingImage = False
        self.m_applicationExiting = False
        self.m_doImageCapture = True

        self.m_metaDataDialog = None

        self._ui = Ui_Camera()
        self._ui.setupUi(self)
        self._install_resizable_splitter()

        available_cameras = QMediaDevices.videoInputs()
        print("type-available_cameras = ", type(available_cameras))
        print("len(type-available_cameras) = ", len(available_cameras))
        print("teeeeeeeeeeet")

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

        # try to actually initialize camera & mic
        self.initialize()

    def _install_resizable_splitter(self):
        self._ui.horizontalLayout_8.removeWidget(self._ui.groupBox)
        self._ui.horizontalLayout_8.removeWidget(self._ui.captureWidget)

        splitter = QSplitter(Qt.Horizontal, self._ui.cameraWidget)
        splitter.setObjectName("cameraSettingsSplitter")
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(6)
        splitter.addWidget(self._ui.groupBox)
        splitter.addWidget(self._ui.captureWidget)
        splitter.setStretchFactor(0, 5)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([500, 100])

        self._ui.verticalLayout_2.removeItem(self._ui.horizontalLayout_8)
        self._ui.verticalLayout_2.addWidget(splitter)

    @Slot()
    def initialize(self):
#        if is_android or sys.platform == "darwin":
#            is_nuitka = "__compiled__" in globals()
#            if not is_nuitka and sys.platform == "darwin":
#                print("This example does not work on macOS when Python is run in interpreted mode."
#                      "For this example to work on macOS, package the example using pyside6-deploy"
#                      "For more information, read `Notes for Developer` in the documentation")
#                sys.exit(0)
        if False:
            # camera
            cam_permission = QCameraPermission()
            cam_permission_status = qApp.checkPermission(cam_permission)  # noqa: F821
            if cam_permission_status == Qt.PermissionStatus.Undetermined:
                qApp.requestPermission(cam_permission, self, self.initialize)  # noqa: F821
                return
            if cam_permission_status == Qt.PermissionStatus.Denied:
                qWarning("Camera permission is not granted!")
                return
            elif cam_permission_status == Qt.PermissionStatus.Granted:
                print("[AudioSource] Camera permission granted")

            # microphone
            microphone_permission = QMicrophonePermission()
            microphone_permission_status = qApp.checkPermission(microphone_permission)  # noqa: F821
            if microphone_permission_status == Qt.PermissionStatus.Undetermined:
                qApp.requestPermission(microphone_permission, self, self.initialize)  # noqa: F821
                return
            if microphone_permission_status == Qt.PermissionStatus.Denied:
                qWarning("Microphone permission is not granted!")
                self.initializeErrorWindow()
                return
            elif microphone_permission_status == Qt.PermissionStatus.Granted:
                print("[AudioSource] Microphone permission granted")

        print("start")
        self.m_audioInput = QAudioInput()
        self.m_captureSession.setAudioInput(self.m_audioInput)

        # Camera devices

        self._video_devices_group = QActionGroup(self)
        self._video_devices_group.setExclusive(True)
        self.updateCameras()
        self.m_devices.videoInputsChanged.connect(self.updateCameras)

        print("middle")

        self._video_devices_group.triggered.connect(self.updateCameraDevice)
        self._ui.captureWidget.currentChanged.connect(self.updateCaptureMode)

        self._ui.metaDataButton.clicked.connect(self.showMetaDataDialog)
        self._ui.exposureCompensation.valueChanged.connect(self.setExposureCompensation)

        self.setCamera(QMediaDevices.defaultVideoInput())

        # image settings
        self.configureImageSettings()
        print("end")

    @Slot(QCameraDevice)
    def setCamera(self, cameraDevice):
        self.m_camera = QCamera(cameraDevice)
        self.m_captureSession.setCamera(self.m_camera)

        self.m_camera.activeChanged.connect(self.updateCameraActive)
        self.m_camera.errorOccurred.connect(self.displayCameraError)

        if not self.m_mediaRecorder:
            self.m_mediaRecorder = QMediaRecorder()
            self.m_captureSession.setRecorder(self.m_mediaRecorder)
            self.m_mediaRecorder.recorderStateChanged.connect(self.updateRecorderState)
            self.m_mediaRecorder.durationChanged.connect(self.updateRecordTime)
            self.m_mediaRecorder.errorChanged.connect(self.displayRecorderError)

        if not self.m_imageCapture:
            self.m_imageCapture = QImageCapture()
            self.m_captureSession.setImageCapture(self.m_imageCapture)
            self.m_imageCapture.readyForCaptureChanged.connect(self.readyForCapture)
            self.m_imageCapture.imageCaptured.connect(self.processCapturedImage)
            self.m_imageCapture.imageSaved.connect(self.imageSaved)
            self.m_imageCapture.errorOccurred.connect(self.displayCaptureError)

        self.m_captureSession.setVideoOutput(self._ui.viewfinder)

        self.updateCameraActive(self.m_camera.isActive())
        print("isActive = ", self.m_camera.isActive())
        self.updateRecorderState(self.m_mediaRecorder.recorderState())
        self.readyForCapture(self.m_imageCapture.isReadyForCapture())

        self.updateCaptureMode()

        self.m_camera.start()

    def keyPressEvent(self, event):
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
        d = self.m_mediaRecorder.duration() / 1000
        self._ui.statusbar.showMessage(f"Recorded {d} sec")

    @Slot(int, QImage)
    def processCapturedImage(self, requestId, img):
        scaled_image = img.scaled(self._ui.viewfinder.size(), Qt.KeepAspectRatio,
                                  Qt.SmoothTransformation)

        self._ui.lastImagePreviewLabel.setPixmap(QPixmap.fromImage(scaled_image))

        # Display captured image for 4 seconds.
        self.displayCapturedImage()
        QTimer.singleShot(4000, self.displayViewfinder)

    @Slot()
    def configureCaptureSettings(self):
        if self.m_doImageCapture:
            self.configureImageSettings()
        else:
            self.configureVideoSettings()

    @Slot()
    def configureVideoSettings(self):
        settings_dialog = VideoSettings(self.m_mediaRecorder)

        if settings_dialog.exec():
            settings_dialog.apply_settings()

    @Slot()
    def configureImageSettings(self):
        #image settings
        # image codecs
        self._ui.imageCodecBox.addItem("Default image format",
                                       QImageCapture.UnspecifiedFormat)
        supported_image_formats = QImageCapture.supportedFormats()
        for f in supported_image_formats:
            description = QImageCapture.fileFormatDescription(f)
            name = QImageCapture.fileFormatName(f)
            self._ui.imageCodecBox.addItem(f"{name} : {description}", f)

        self._ui.imageQualitySlider.setRange(0, QImageCapture.VeryHighQuality.value)

        self._ui.imageResolutionBox.addItem("Default Resolution", QSize())
        camera = self.m_imageCapture.captureSession().camera()
        supported_resolutions = camera.cameraDevice().photoResolutions()
        for resolution in supported_resolutions:
            w, h = resolution.width(), resolution.height()
            self._ui.imageResolutionBox.addItem(f"{w}x{h}", resolution)

        self.select_combo_box_item(self._ui.imageCodecBox, self.m_imageCapture.fileFormat())
        self.select_combo_box_item(self._ui.imageResolutionBox, self.m_imageCapture.resolution())
        self._ui.imageQualitySlider.setValue(self.m_imageCapture.quality().value)

        self.m_imageCapture.setFileFormat(self.box_value(self._ui.imageCodecBox))
        q = self._ui.imageQualitySlider.value()
        self.m_imageCapture.setQuality(QImageCapture.Quality(q))
        self.m_imageCapture.setResolution(self.box_value(self._ui.imageResolutionBox))

#        settings_dialog = ImageSettings(self.m_imageCapture)

#        if settings_dialog.exec():
#            settings_dialog.apply_image_settings()

    @Slot()
    def record(self):
        self.m_mediaRecorder.record()
        self.updateRecordTime()

    @Slot()
    def pause(self):
        self.m_mediaRecorder.pause()

    @Slot()
    def stop(self):
        self.m_mediaRecorder.stop()

    @Slot(bool)
    def setMuted(self, muted):
        self.m_captureSession.audioInput().setMuted(muted)

    @Slot()
    def takeImage(self):
        self.m_isCapturingImage = True
        self.m_imageCapture.captureToFile()

    @Slot(int, QImageCapture.Error, str)
    def displayCaptureError(self, id, error, errorString):
        QMessageBox.warning(self, "Image Capture Error", errorString)
        self.m_isCapturingImage = False

    @Slot()
    def startCamera(self):
        self.m_camera.start()

    @Slot()
    def stopCamera(self):
        self.m_camera.stop()

    @Slot()
    def updateCaptureMode(self):
        tab_index = self._ui.captureWidget.currentIndex()
        self.m_doImageCapture = (tab_index == 0)

    @Slot(bool)
    def updateCameraActive(self, active):
        if active:
#            self._ui.actionStartCamera.setEnabled(False)
#            self._ui.actionStopCamera.setEnabled(True)
            self._ui.captureWidget.setEnabled(True)
#            self._ui.actionSettings.setEnabled(True)
        else:
#            self._ui.actionStartCamera.setEnabled(True)
#            self._ui.actionStopCamera.setEnabled(False)
            self._ui.captureWidget.setEnabled(False)
#            self._ui.actionSettings.setEnabled(False)

    @Slot(QMediaRecorder.RecorderState)
    def updateRecorderState(self, state):
        if state == QMediaRecorder.StoppedState:
            self._ui.recordButton.setEnabled(True)
            self._ui.pauseButton.setEnabled(True)
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

    @Slot(int)
    def setExposureCompensation(self, index):
        self.m_camera.setExposureCompensation(index * 0.5)

    @Slot()
    def displayRecorderError(self):
        if self.m_mediaRecorder.error() != QMediaRecorder.NoError:
            QMessageBox.warning(self, "Capture Error",
                                self.m_mediaRecorder.errorString())

    @Slot()
    def displayCameraError(self):
        if self.m_camera.error() != QCamera.NoError:
            QMessageBox.warning(self, "Camera Error",
                                self.m_camera.errorString())

    @Slot(QAction)
    def updateCameraDevice(self, action):
        self.setCamera(QCameraDevice(action))

    @Slot()
    def displayViewfinder(self):
        self._ui.stackedWidget.setCurrentIndex(0)

    @Slot()
    def displayCapturedImage(self):
        self._ui.stackedWidget.setCurrentIndex(1)

    @Slot(bool)
    def readyForCapture(self, ready):
        self._ui.takeImageButton.setEnabled(ready)

    @Slot(int, str)
    def imageSaved(self, id, fileName):
        f = QDir.toNativeSeparators(fileName)
        self._ui.statusbar.showMessage(f"Captured \"{f}\"")

        self.m_isCapturingImage = False
        if self.m_applicationExiting:
            self.close()

    def closeEvent(self, event):
        if self.m_isCapturingImage:
            self.setEnabled(False)
            self.m_applicationExiting = True
            event.ignore()
        else:
            event.accept()

    @Slot()
    def updateCameras(self):
#        self._ui.menuDevices.clear()
        available_cameras = QMediaDevices.videoInputs()
        print("type-available_cameras = ", type(available_cameras))
        print("len(type-available_cameras) = ", len(available_cameras))
        for cameraDevice in available_cameras:
            video_device_action = QAction(cameraDevice.description(),
                                          self._video_devices_group)
            video_device_action.setCheckable(True)
            video_device_action.setData(cameraDevice)
            if cameraDevice == QMediaDevices.defaultVideoInput():
                video_device_action.setChecked(True)

#            self._ui.menuDevices.addAction(video_device_action)

    @Slot()
    def showMetaDataDialog(self):
        if not self.m_metaDataDialog:
            self.m_metaDataDialog = MetaDataDialog(self)
        self.m_metaDataDialog.setAttribute(Qt.WA_DeleteOnClose, False)
        if self.m_metaDataDialog.exec() == QDialog.Accepted:
            self.saveMetaData()

    @Slot()
    def saveMetaData(self):
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

