# This Python file uses the following encoding: utf-8

# if __name__ == "__main__":
#     pass
import sys
from PySide6.QtWidgets import (QMainWindow, QFileDialog,QColorDialog, QComboBox,
                                QDialog, QFontComboBox,QTextEdit, QInputDialog,
                                QLineEdit, QMenu, QMessageBox,QProgressBar, QToolBar,
                                QVBoxLayout, QWidget, QTreeView, QFileSystemModel)
from PySide6.QtGui import QAction, QGuiApplication, QIcon, QKeySequence
from PySide6.QtCore import QUrl, Qt, Slot, Signal, QDir

from PySide6.QtPrintSupport import (QAbstractPrintDialog, QPrinter,
                                    QPrintDialog, QPrintPreviewDialog)
from PySide6.QtGui import QIcon

from ui import ui_detectfile

class DetectFile(QWidget):
    def __init__(self):
        super().__init__()
        self.ui = ui_detectfile.Ui_filemanagerWidget()
        self.ui.setupUi(self)

        self.dirModel = QFileSystemModel()
        self.dirModel.setFilter(QDir.NoDotAndDotDot | QDir.AllDirs)
        self.dirModel.setRootPath("D:/")
        self.ui.dirTreeView.setModel(self.dirModel)

        self.fileModel = QFileSystemModel()
        self.fileModel.setFilter(QDir.NoDotAndDotDot | QDir.Files)
        self.fileModel.setNameFilterDisables(False)
#        self.filemodel.setRootPath(self.sPath)
        self.ui.fileListView.setModel(self.fileModel)








#    def _init_ui_detectmain(self):
#        self.dir_model = QFileSystemModel()
#        self.dir_model.setFilter(QDir.NoDotAndDotDot | QDir.AllDirs)
#        self.dir_model.setRootPath("D:/")
#        self._ui_detectmain._dir_treeview.setModel(self.dir_model)

#        self.file_model = QFileSystemModel()
#        self.file_model.setFilter(QDir.NoDotAndDotDot | QDir.Files)
#        self.file_model.setNameFilterDisables(False)
##        self.filemodel.setRootPath(self.sPath)
#        self._ui_detectmain._file_listview.setModel(self.file_model)
#        self._ui_detectmain._dir_treeview.clicked.connect(self.clicked_dir)
#        self._ui_detectmain._file_listview.clicked.connect(self.clicked_file)

#        self._ui_detectmain._ext_combobox.textActivated.connect(self.clicked_ext)
