# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'camera.ui'
##
## Created by: Qt User Interface Compiler version 6.5.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (QAbstractButton, QApplication, QComboBox, QDialogButtonBox,
    QFrame, QGridLayout, QGroupBox, QHBoxLayout,
    QLabel, QPushButton, QSizePolicy, QSlider,
    QSpacerItem, QStackedWidget, QTabWidget, QVBoxLayout,
    QWidget)

class Ui_Camera(object):
    def setupUi(self, Camera):
        if not Camera.objectName():
            Camera.setObjectName(u"Camera")
        Camera.resize(665, 470)
        self.horizontalLayout_2 = QHBoxLayout(Camera)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.cameraWidget = QWidget(Camera)
        self.cameraWidget.setObjectName(u"cameraWidget")
        self.verticalLayout_2 = QVBoxLayout(self.cameraWidget)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.horizontalLayout_8 = QHBoxLayout()
        self.horizontalLayout_8.setObjectName(u"horizontalLayout_8")
        self.groupBox = QGroupBox(self.cameraWidget)
        self.groupBox.setObjectName(u"groupBox")
        self.verticalLayout = QVBoxLayout(self.groupBox)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.stackedWidget = QStackedWidget(self.groupBox)
        self.stackedWidget.setObjectName(u"stackedWidget")
        sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.stackedWidget.sizePolicy().hasHeightForWidth())
        self.stackedWidget.setSizePolicy(sizePolicy)
        palette = QPalette()
        brush = QBrush(QColor(255, 255, 255, 255))
        brush.setStyle(Qt.SolidPattern)
        palette.setBrush(QPalette.Active, QPalette.Base, brush)
        brush1 = QBrush(QColor(145, 145, 145, 255))
        brush1.setStyle(Qt.SolidPattern)
        palette.setBrush(QPalette.Active, QPalette.Window, brush1)
        palette.setBrush(QPalette.Inactive, QPalette.Base, brush)
        palette.setBrush(QPalette.Inactive, QPalette.Window, brush1)
        palette.setBrush(QPalette.Disabled, QPalette.Base, brush1)
        palette.setBrush(QPalette.Disabled, QPalette.Window, brush1)
        self.stackedWidget.setPalette(palette)
        self.viewfinderPage = QWidget()
        self.viewfinderPage.setObjectName(u"viewfinderPage")
        self.verticalLayout_3 = QVBoxLayout(self.viewfinderPage)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.viewfinder = QVideoWidget(self.viewfinderPage)
        self.viewfinder.setObjectName(u"viewfinder")

        self.verticalLayout_3.addWidget(self.viewfinder)

        self.stackedWidget.addWidget(self.viewfinderPage)
        self.previewPage = QWidget()
        self.previewPage.setObjectName(u"previewPage")
        self.gridLayout_4 = QGridLayout(self.previewPage)
        self.gridLayout_4.setObjectName(u"gridLayout_4")
        self.lastImagePreviewLabel = QLabel(self.previewPage)
        self.lastImagePreviewLabel.setObjectName(u"lastImagePreviewLabel")
        sizePolicy1 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.MinimumExpanding)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.lastImagePreviewLabel.sizePolicy().hasHeightForWidth())
        self.lastImagePreviewLabel.setSizePolicy(sizePolicy1)
        self.lastImagePreviewLabel.setFrameShape(QFrame.Box)

        self.gridLayout_4.addWidget(self.lastImagePreviewLabel, 0, 0, 1, 1)

        self.stackedWidget.addWidget(self.previewPage)

        self.verticalLayout.addWidget(self.stackedWidget)


        self.horizontalLayout_8.addWidget(self.groupBox)

        self.captureWidget = QTabWidget(self.cameraWidget)
        self.captureWidget.setObjectName(u"captureWidget")
        self.tab_2 = QWidget()
        self.tab_2.setObjectName(u"tab_2")
        self.verticalLayout_7 = QVBoxLayout(self.tab_2)
        self.verticalLayout_7.setObjectName(u"verticalLayout_7")
        self.verticalLayout_6 = QVBoxLayout()
        self.verticalLayout_6.setObjectName(u"verticalLayout_6")
        self.verticalLayout_5 = QVBoxLayout()
        self.verticalLayout_5.setObjectName(u"verticalLayout_5")
        self.takeImageButton = QPushButton(self.tab_2)
        self.takeImageButton.setObjectName(u"takeImageButton")
        self.takeImageButton.setEnabled(False)

        self.verticalLayout_5.addWidget(self.takeImageButton)

        self.groupBox_2 = QGroupBox(self.tab_2)
        self.groupBox_2.setObjectName(u"groupBox_2")
        self.horizontalLayout = QHBoxLayout(self.groupBox_2)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.verticalLayout_4 = QVBoxLayout()
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.label_8 = QLabel(self.groupBox_2)
        self.label_8.setObjectName(u"label_8")

        self.verticalLayout_4.addWidget(self.label_8)

        self.imageResolutionBox = QComboBox(self.groupBox_2)
        self.imageResolutionBox.setObjectName(u"imageResolutionBox")

        self.verticalLayout_4.addWidget(self.imageResolutionBox)

        self.label_6 = QLabel(self.groupBox_2)
        self.label_6.setObjectName(u"label_6")

        self.verticalLayout_4.addWidget(self.label_6)

        self.imageCodecBox = QComboBox(self.groupBox_2)
        self.imageCodecBox.setObjectName(u"imageCodecBox")

        self.verticalLayout_4.addWidget(self.imageCodecBox)

        self.label_7 = QLabel(self.groupBox_2)
        self.label_7.setObjectName(u"label_7")

        self.verticalLayout_4.addWidget(self.label_7)

        self.imageQualitySlider = QSlider(self.groupBox_2)
        self.imageQualitySlider.setObjectName(u"imageQualitySlider")
        self.imageQualitySlider.setMaximum(4)
        self.imageQualitySlider.setOrientation(Qt.Horizontal)

        self.verticalLayout_4.addWidget(self.imageQualitySlider)

        self.label = QLabel(self.groupBox_2)
        self.label.setObjectName(u"label")

        self.verticalLayout_4.addWidget(self.label)

        self.exposureCompensation = QSlider(self.groupBox_2)
        self.exposureCompensation.setObjectName(u"exposureCompensation")
        self.exposureCompensation.setMinimum(-4)
        self.exposureCompensation.setMaximum(4)
        self.exposureCompensation.setPageStep(2)
        self.exposureCompensation.setOrientation(Qt.Horizontal)
        self.exposureCompensation.setTickPosition(QSlider.TicksAbove)

        self.verticalLayout_4.addWidget(self.exposureCompensation)


        self.horizontalLayout.addLayout(self.verticalLayout_4)


        self.verticalLayout_5.addWidget(self.groupBox_2)


        self.verticalLayout_6.addLayout(self.verticalLayout_5)

        self.buttonBox = QDialogButtonBox(self.tab_2)
        self.buttonBox.setObjectName(u"buttonBox")
        self.buttonBox.setOrientation(Qt.Horizontal)
        self.buttonBox.setStandardButtons(QDialogButtonBox.Cancel|QDialogButtonBox.Ok)

        self.verticalLayout_6.addWidget(self.buttonBox)


        self.verticalLayout_7.addLayout(self.verticalLayout_6)

        self.captureWidget.addTab(self.tab_2, "")
        self.tab = QWidget()
        self.tab.setObjectName(u"tab")
        self.gridLayout_2 = QGridLayout(self.tab)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.recordButton = QPushButton(self.tab)
        self.recordButton.setObjectName(u"recordButton")

        self.gridLayout_2.addWidget(self.recordButton, 0, 0, 1, 1)

        self.pauseButton = QPushButton(self.tab)
        self.pauseButton.setObjectName(u"pauseButton")

        self.gridLayout_2.addWidget(self.pauseButton, 1, 0, 1, 1)

        self.stopButton = QPushButton(self.tab)
        self.stopButton.setObjectName(u"stopButton")

        self.gridLayout_2.addWidget(self.stopButton, 2, 0, 1, 1)

        self.verticalSpacer = QSpacerItem(20, 76, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.gridLayout_2.addItem(self.verticalSpacer, 3, 0, 1, 1)

        self.muteButton = QPushButton(self.tab)
        self.muteButton.setObjectName(u"muteButton")
        self.muteButton.setCheckable(True)

        self.gridLayout_2.addWidget(self.muteButton, 4, 0, 1, 1)

        self.metaDataButton = QPushButton(self.tab)
        self.metaDataButton.setObjectName(u"metaDataButton")
        self.metaDataButton.setCheckable(True)

        self.gridLayout_2.addWidget(self.metaDataButton, 5, 0, 1, 1)

        self.captureWidget.addTab(self.tab, "")

        self.horizontalLayout_8.addWidget(self.captureWidget)

        self.horizontalLayout_8.setStretch(0, 5)
        self.horizontalLayout_8.setStretch(1, 1)

        self.verticalLayout_2.addLayout(self.horizontalLayout_8)


        self.horizontalLayout_2.addWidget(self.cameraWidget)


        self.retranslateUi(Camera)

        self.stackedWidget.setCurrentIndex(0)
        self.captureWidget.setCurrentIndex(0)


        QMetaObject.connectSlotsByName(Camera)
    # setupUi

    def retranslateUi(self, Camera):
        Camera.setWindowTitle(QCoreApplication.translate("Camera", u"Form", None))
        self.groupBox.setTitle("")
        self.lastImagePreviewLabel.setText("")
        self.takeImageButton.setText(QCoreApplication.translate("Camera", u"Capture Photo", None))
        self.groupBox_2.setTitle(QCoreApplication.translate("Camera", u"Image", None))
        self.label_8.setText(QCoreApplication.translate("Camera", u"Resolution:", None))
        self.label_6.setText(QCoreApplication.translate("Camera", u"Image Format:", None))
        self.label_7.setText(QCoreApplication.translate("Camera", u"Quality:", None))
        self.label.setText(QCoreApplication.translate("Camera", u"Exposure Compensation:", None))
        self.captureWidget.setTabText(self.captureWidget.indexOf(self.tab_2), QCoreApplication.translate("Camera", u"Image", None))
        self.recordButton.setText(QCoreApplication.translate("Camera", u"Record", None))
        self.pauseButton.setText(QCoreApplication.translate("Camera", u"Pause", None))
        self.stopButton.setText(QCoreApplication.translate("Camera", u"Stop", None))
        self.muteButton.setText(QCoreApplication.translate("Camera", u"Mute", None))
        self.metaDataButton.setText(QCoreApplication.translate("Camera", u"Set metadata", None))
        self.captureWidget.setTabText(self.captureWidget.indexOf(self.tab), QCoreApplication.translate("Camera", u"Video", None))
    # retranslateUi

