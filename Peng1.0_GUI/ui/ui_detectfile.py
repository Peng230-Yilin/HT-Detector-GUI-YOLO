# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'detectfile.ui'
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
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QComboBox, QFrame,
    QHeaderView, QListView, QSizePolicy, QTabWidget,
    QTreeView, QVBoxLayout, QWidget)

class Ui_filemanagerWidget(object):
    def setupUi(self, filemanagerWidget):
        if not filemanagerWidget.objectName():
            filemanagerWidget.setObjectName(u"filemanagerWidget")
        filemanagerWidget.resize(606, 573)
        self.filemanagerTabWidget = QTabWidget(filemanagerWidget)
        self.filemanagerTabWidget.setObjectName(u"filemanagerTabWidget")
        self.filemanagerTabWidget.setGeometry(QRect(160, 30, 272, 477))
        self.tab = QWidget()
        self.tab.setObjectName(u"tab")
        self.verticalLayout = QVBoxLayout(self.tab)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.dirComboBox = QComboBox(self.tab)
        self.dirComboBox.setObjectName(u"dirComboBox")

        self.verticalLayout.addWidget(self.dirComboBox)

        self.dirTreeView = QTreeView(self.tab)
        self.dirTreeView.setObjectName(u"dirTreeView")
        self.dirTreeView.setFrameShape(QFrame.Panel)
        self.dirTreeView.setLineWidth(3)

        self.verticalLayout.addWidget(self.dirTreeView)

        self.fileListView = QListView(self.tab)
        self.fileListView.setObjectName(u"fileListView")
        self.fileListView.setFrameShape(QFrame.Panel)
        self.fileListView.setLineWidth(3)
        self.fileListView.setEditTriggers(QAbstractItemView.DoubleClicked|QAbstractItemView.EditKeyPressed)
        self.fileListView.setSelectionMode(QAbstractItemView.ExtendedSelection)

        self.verticalLayout.addWidget(self.fileListView)

        self.extComboBox = QComboBox(self.tab)
        self.extComboBox.addItem("")
        self.extComboBox.addItem("")
        self.extComboBox.addItem("")
        self.extComboBox.setObjectName(u"extComboBox")

        self.verticalLayout.addWidget(self.extComboBox)

        self.filemanagerTabWidget.addTab(self.tab, "")
        self.tab_2 = QWidget()
        self.tab_2.setObjectName(u"tab_2")
        self.filemanagerTabWidget.addTab(self.tab_2, "")
        self.tab_3 = QWidget()
        self.tab_3.setObjectName(u"tab_3")
        self.filemanagerTabWidget.addTab(self.tab_3, "")

        self.retranslateUi(filemanagerWidget)

        self.filemanagerTabWidget.setCurrentIndex(0)


        QMetaObject.connectSlotsByName(filemanagerWidget)
    # setupUi

    def retranslateUi(self, filemanagerWidget):
        filemanagerWidget.setWindowTitle(QCoreApplication.translate("filemanagerWidget", u"Form", None))
        self.extComboBox.setItemText(0, QCoreApplication.translate("filemanagerWidget", u"All Files (*.*)", None))
        self.extComboBox.setItemText(1, QCoreApplication.translate("filemanagerWidget", u"Text (*.txt)", None))
        self.extComboBox.setItemText(2, QCoreApplication.translate("filemanagerWidget", u"Jpeg (*.jpg)", None))

        self.filemanagerTabWidget.setTabText(self.filemanagerTabWidget.indexOf(self.tab), QCoreApplication.translate("filemanagerWidget", u"Directory", None))
        self.filemanagerTabWidget.setTabText(self.filemanagerTabWidget.indexOf(self.tab_2), QCoreApplication.translate("filemanagerWidget", u"Functions", None))
        self.filemanagerTabWidget.setTabText(self.filemanagerTabWidget.indexOf(self.tab_3), QCoreApplication.translate("filemanagerWidget", u"Property", None))
    # retranslateUi

