# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'detectmain.ui'
##
## Created by: Qt User Interface Compiler version 6.7.2
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
from PySide6.QtWidgets import (QApplication, QFrame, QGridLayout, QGroupBox,
    QHBoxLayout, QHeaderView, QLCDNumber, QLabel,
    QLayout, QProgressBar, QPushButton, QSizePolicy,
    QSpacerItem, QTabWidget, QTableView, QVBoxLayout,
    QWidget)

class Ui_Form(object):
    def setupUi(self, Form):
        if not Form.objectName():
            Form.setObjectName(u"Form")
        Form.resize(1261, 802)
        self.verticalLayout_2 = QVBoxLayout(Form)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.detectmainWidget = QWidget(Form)
        self.detectmainWidget.setObjectName(u"detectmainWidget")
        self.verticalLayout_13 = QVBoxLayout(self.detectmainWidget)
        self.verticalLayout_13.setObjectName(u"verticalLayout_13")
        self.gridLayout_3 = QGridLayout()
        self.gridLayout_3.setObjectName(u"gridLayout_3")
        self.cameraMainGroupBox = QGroupBox(self.detectmainWidget)
        self.cameraMainGroupBox.setObjectName(u"cameraMainGroupBox")
        self.horizontalLayout_9 = QHBoxLayout(self.cameraMainGroupBox)
        self.horizontalLayout_9.setObjectName(u"horizontalLayout_9")
        self.horizontalLayout_6 = QHBoxLayout()
        self.horizontalLayout_6.setSpacing(3)
        self.horizontalLayout_6.setObjectName(u"horizontalLayout_6")
        self.horizontalLayout_6.setSizeConstraint(QLayout.SizeConstraint.SetNoConstraint)
        self.tabWidget_2 = QTabWidget(self.cameraMainGroupBox)
        self.tabWidget_2.setObjectName(u"tabWidget_2")
        font = QFont()
        font.setPointSize(10)
        font.setBold(False)
        self.tabWidget_2.setFont(font)
        self.tab_4 = QWidget()
        self.tab_4.setObjectName(u"tab_4")
        self.horizontalLayout_5 = QHBoxLayout(self.tab_4)
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.labelOrigImg = QLabel(self.tab_4)
        self.labelOrigImg.setObjectName(u"labelOrigImg")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.labelOrigImg.sizePolicy().hasHeightForWidth())
        self.labelOrigImg.setSizePolicy(sizePolicy)

        self.horizontalLayout_5.addWidget(self.labelOrigImg)

        self.tabWidget_2.addTab(self.tab_4, "")

        self.horizontalLayout_6.addWidget(self.tabWidget_2)

        self.verticalLayout_8 = QVBoxLayout()
        self.verticalLayout_8.setObjectName(u"verticalLayout_8")
        self.label_2 = QLabel(self.cameraMainGroupBox)
        self.label_2.setObjectName(u"label_2")
        font1 = QFont()
        font1.setPointSize(9)
        self.label_2.setFont(font1)

        self.verticalLayout_8.addWidget(self.label_2)

        self.tabviewOrig = QTableView(self.cameraMainGroupBox)
        self.tabviewOrig.setObjectName(u"tabviewOrig")
        font2 = QFont()
        font2.setPointSize(11)
        font2.setBold(True)
        self.tabviewOrig.setFont(font2)

        self.verticalLayout_8.addWidget(self.tabviewOrig)


        self.horizontalLayout_6.addLayout(self.verticalLayout_8)

        self.horizontalLayout_6.setStretch(0, 5)
        self.horizontalLayout_6.setStretch(1, 2)

        self.horizontalLayout_9.addLayout(self.horizontalLayout_6)


        self.gridLayout_3.addWidget(self.cameraMainGroupBox, 0, 0, 1, 1)

        self.groupBox_4 = QGroupBox(self.detectmainWidget)
        self.groupBox_4.setObjectName(u"groupBox_4")
        self.horizontalLayout_7 = QHBoxLayout(self.groupBox_4)
        self.horizontalLayout_7.setObjectName(u"horizontalLayout_7")
        self.verticalLayout_12 = QVBoxLayout()
        self.verticalLayout_12.setObjectName(u"verticalLayout_12")
        self.widgetChart = QWidget(self.groupBox_4)
        self.widgetChart.setObjectName(u"widgetChart")
        sizePolicy.setHeightForWidth(self.widgetChart.sizePolicy().hasHeightForWidth())
        self.widgetChart.setSizePolicy(sizePolicy)

        self.verticalLayout_12.addWidget(self.widgetChart)

        self.gridLayout = QGridLayout()
        self.gridLayout.setObjectName(u"gridLayout")
        self.horizontalSpacer_5 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.gridLayout.addItem(self.horizontalSpacer_5, 1, 0, 1, 1)

        self.pushButton_5 = QPushButton(self.groupBox_4)
        self.pushButton_5.setObjectName(u"pushButton_5")

        self.gridLayout.addWidget(self.pushButton_5, 1, 1, 1, 1)

        self.pushButton_8 = QPushButton(self.groupBox_4)
        self.pushButton_8.setObjectName(u"pushButton_8")

        self.gridLayout.addWidget(self.pushButton_8, 1, 9, 1, 1)

        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.gridLayout.addItem(self.horizontalSpacer, 1, 2, 1, 1)

        self.pushButton_4 = QPushButton(self.groupBox_4)
        self.pushButton_4.setObjectName(u"pushButton_4")

        self.gridLayout.addWidget(self.pushButton_4, 1, 3, 1, 1)

        self.horizontalSpacer_4 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.gridLayout.addItem(self.horizontalSpacer_4, 1, 8, 1, 1)

        self.pushButton_7 = QPushButton(self.groupBox_4)
        self.pushButton_7.setObjectName(u"pushButton_7")

        self.gridLayout.addWidget(self.pushButton_7, 1, 5, 1, 1)

        self.horizontalSpacer_2 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.gridLayout.addItem(self.horizontalSpacer_2, 1, 4, 1, 1)

        self.horizontalSpacer_3 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.gridLayout.addItem(self.horizontalSpacer_3, 1, 6, 1, 1)

        self.pushButton = QPushButton(self.groupBox_4)
        self.pushButton.setObjectName(u"pushButton")

        self.gridLayout.addWidget(self.pushButton, 1, 7, 1, 1)

        self.horizontalSpacer_6 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.gridLayout.addItem(self.horizontalSpacer_6, 1, 10, 1, 1)


        self.verticalLayout_12.addLayout(self.gridLayout)

        self.verticalLayout_12.setStretch(0, 5)

        self.horizontalLayout_7.addLayout(self.verticalLayout_12)


        self.gridLayout_3.addWidget(self.groupBox_4, 0, 1, 1, 1)

        self.groupBox_6 = QGroupBox(self.detectmainWidget)
        self.groupBox_6.setObjectName(u"groupBox_6")
        self.horizontalLayout_10 = QHBoxLayout(self.groupBox_6)
        self.horizontalLayout_10.setObjectName(u"horizontalLayout_10")
        self.cameraMainlVLayout = QVBoxLayout()
        self.cameraMainlVLayout.setObjectName(u"cameraMainlVLayout")
        self.cameraMainlVLayout.setSizeConstraint(QLayout.SizeConstraint.SetDefaultConstraint)

        self.horizontalLayout_10.addLayout(self.cameraMainlVLayout)


        self.gridLayout_3.addWidget(self.groupBox_6, 1, 0, 1, 1)

        self.groupBox_3 = QGroupBox(self.detectmainWidget)
        self.groupBox_3.setObjectName(u"groupBox_3")
        self.verticalLayout_11 = QVBoxLayout(self.groupBox_3)
        self.verticalLayout_11.setObjectName(u"verticalLayout_11")
        self.horizontalLayout_4 = QHBoxLayout()
        self.horizontalLayout_4.setSpacing(3)
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.verticalLayout_10 = QVBoxLayout()
        self.verticalLayout_10.setObjectName(u"verticalLayout_10")
        self.tabWidget = QTabWidget(self.groupBox_3)
        self.tabWidget.setObjectName(u"tabWidget")
        self.tabWidget.setFont(font)
        self.tabWidget.setTabPosition(QTabWidget.TabPosition.North)
        self.tabWidget.setTabShape(QTabWidget.TabShape.Rounded)
        self.tab_3 = QWidget()
        self.tab_3.setObjectName(u"tab_3")
        self.horizontalLayout_3 = QHBoxLayout(self.tab_3)
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.labelRecgImg = QLabel(self.tab_3)
        self.labelRecgImg.setObjectName(u"labelRecgImg")
        sizePolicy.setHeightForWidth(self.labelRecgImg.sizePolicy().hasHeightForWidth())
        self.labelRecgImg.setSizePolicy(sizePolicy)

        self.horizontalLayout_3.addWidget(self.labelRecgImg)

        self.tabWidget.addTab(self.tab_3, "")

        self.verticalLayout_10.addWidget(self.tabWidget)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.label_5 = QLabel(self.groupBox_3)
        self.label_5.setObjectName(u"label_5")
        font3 = QFont()
        font3.setPointSize(10)
        self.label_5.setFont(font3)
        self.label_5.setFrameShape(QFrame.Shape.NoFrame)
        self.label_5.setFrameShadow(QFrame.Shadow.Raised)

        self.horizontalLayout_2.addWidget(self.label_5)

        self.progressBar = QProgressBar(self.groupBox_3)
        self.progressBar.setObjectName(u"progressBar")
        self.progressBar.setValue(100)
        self.progressBar.setTextVisible(False)
        self.progressBar.setOrientation(Qt.Orientation.Horizontal)
        self.progressBar.setInvertedAppearance(False)
        self.progressBar.setTextDirection(QProgressBar.Direction.BottomToTop)

        self.horizontalLayout_2.addWidget(self.progressBar)

        self.label_9 = QLabel(self.groupBox_3)
        self.label_9.setObjectName(u"label_9")
        self.label_9.setFont(font3)
        self.label_9.setFrameShape(QFrame.Shape.NoFrame)
        self.label_9.setFrameShadow(QFrame.Shadow.Raised)

        self.horizontalLayout_2.addWidget(self.label_9)

        self.lcdNumber = QLCDNumber(self.groupBox_3)
        self.lcdNumber.setObjectName(u"lcdNumber")
        self.lcdNumber.setAcceptDrops(False)
        self.lcdNumber.setFrameShape(QFrame.Shape.Panel)
        self.lcdNumber.setProperty("intValue", 377)

        self.horizontalLayout_2.addWidget(self.lcdNumber)


        self.verticalLayout_10.addLayout(self.horizontalLayout_2)


        self.horizontalLayout_4.addLayout(self.verticalLayout_10)

        self.verticalLayout_9 = QVBoxLayout()
        self.verticalLayout_9.setObjectName(u"verticalLayout_9")
        self.label_4 = QLabel(self.groupBox_3)
        self.label_4.setObjectName(u"label_4")
        self.label_4.setFont(font1)

        self.verticalLayout_9.addWidget(self.label_4)

        self.tabviewRecg = QTableView(self.groupBox_3)
        self.tabviewRecg.setObjectName(u"tabviewRecg")
        self.tabviewRecg.setFont(font2)

        self.verticalLayout_9.addWidget(self.tabviewRecg)


        self.horizontalLayout_4.addLayout(self.verticalLayout_9)

        self.horizontalLayout_4.setStretch(0, 5)
        self.horizontalLayout_4.setStretch(1, 2)

        self.verticalLayout_11.addLayout(self.horizontalLayout_4)


        self.gridLayout_3.addWidget(self.groupBox_3, 1, 1, 1, 1)

        self.gridLayout_3.setRowStretch(0, 1)
        self.gridLayout_3.setRowStretch(1, 1)
        self.gridLayout_3.setColumnStretch(0, 1)
        self.gridLayout_3.setColumnStretch(1, 1)

        self.verticalLayout_13.addLayout(self.gridLayout_3)


        self.verticalLayout_2.addWidget(self.detectmainWidget)


        self.retranslateUi(Form)

        self.tabWidget_2.setCurrentIndex(0)
        self.tabWidget.setCurrentIndex(0)


        QMetaObject.connectSlotsByName(Form)
    # setupUi

    def retranslateUi(self, Form):
        Form.setWindowTitle(QCoreApplication.translate("Form", u"Form", None))
        self.cameraMainGroupBox.setTitle("")
        self.labelOrigImg.setText("")
        self.tabWidget_2.setTabText(self.tabWidget_2.indexOf(self.tab_4), QCoreApplication.translate("Form", u"Linear Regression Image", None))
        self.label_2.setText(QCoreApplication.translate("Form", u"Table. Linear Regression", None))
        self.groupBox_4.setTitle("")
        self.pushButton_5.setText(QCoreApplication.translate("Form", u"  Import Image  ", None))
        self.pushButton_8.setText(QCoreApplication.translate("Form", u"Save", None))
        self.pushButton_4.setText(QCoreApplication.translate("Form", u"  Linear Regression  ", None))
        self.pushButton_7.setText(QCoreApplication.translate("Form", u"Plot", None))
        self.pushButton.setText(QCoreApplication.translate("Form", u"Detect", None))
        self.groupBox_6.setTitle("")
        self.groupBox_3.setTitle("")
        self.labelRecgImg.setText("")
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_3), QCoreApplication.translate("Form", u"Detection Image", None))
        self.label_5.setText(QCoreApplication.translate("Form", u"Progress:", None))
        self.label_9.setText(QCoreApplication.translate("Form", u"Period:", None))
        self.label_4.setText(QCoreApplication.translate("Form", u"Table. Detection", None))
    # retranslateUi

