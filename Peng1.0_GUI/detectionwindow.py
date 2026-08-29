# This Python file uses the following encoding: utf-8

import sys
from PySide6.QtWidgets import (QMainWindow, QFileDialog,QColorDialog, QComboBox,
                                QDialog, QFontComboBox,QTextEdit, QInputDialog,
                                QLineEdit, QMenu, QMessageBox,QProgressBar, QToolBar,
                                QVBoxLayout, QHBoxLayout, QGridLayout, QWidget,
                                QTreeView, QFileSystemModel)
from PySide6.QtGui import QAction, QGuiApplication, QIcon, QKeySequence
from PySide6.QtCore import QUrl, Qt, Slot, Signal, QDir, QSize

from PySide6.QtPrintSupport import (QAbstractPrintDialog, QPrinter,
                                    QPrintDialog, QPrintPreviewDialog)
from PySide6.QtGui import QIcon
from ui import ui_detectmain
from ui import ui_detectwindow
from ui import ui_detectfile
from detectfile import DetectFile
from detectmain import DetectMain


#RSRC_PATH = ":/images/mac" if sys.platform == 'darwin' else ":/images/win"
RSRC_PATH = ":/win"


class DetectWindow(QMainWindow):
    about_to_close = Signal()
    def __init__(self, detection):
        super().__init__()
        self._uiWindow = ui_detectwindow.Ui_detectWindow()
        self._uiWindow.setupUi(self)
        self._detection = detection

        self._detectFile = DetectFile()
        self._detectMain = DetectMain()

        #set up the dockWidget in mainwindow -> filemanager
        self._uiWindow.dockWidget.setWidget(self._detectFile.ui.filemanagerTabWidget)
        #These can be created by constructing a widget with the required visual properties - a QFrame ,
        #for example - and adding child widgets to it, usually managed by a layout.
        #Composite widgets can also be created by subclassing a standard widget, such as QWidget or QFrame ,
        #and adding the necessary layout and child widgets in the constructor of the subclass.
        #Many of the examples provided with Qt use this approach, and it is also covered in the Qt Widgets Tutorial .
        self._uiWindow.mainwindowVLayout.addWidget(self._detectMain.ui.detectmainWidget)
        self._uiWindow.mainwindowVLayout.setContentsMargins(0,0,0,0)
        self._uiWindow.centralwidget.setLayout(self._uiWindow.mainwindowVLayout)

        # camera
#        self._detectMain.mainCamera.show()



        self._detectFile.ui.dirTreeView.clicked.connect(self.onClickDir)
        self._detectFile.ui.fileListView.clicked.connect(self.onClickFile)
        self._detectFile.ui.extComboBox.textActivated.connect(self.onClickExt)

    @Slot(int)
    def onClickDir(self, index):
#        pass
        path = self._detectFile.dirModel.fileInfo(index).absoluteFilePath()
        self._detectFile.ui.fileListView.setRootIndex(self._detectFile.fileModel.setRootPath(path))
#        self._dir_line_edit.setText(path)

    @Slot(int)
    def onClickFile(self, index):
#        pass
        path = self._detectFile.fileModel.fileInfo(index).absoluteFilePath()
        self._show_status_message(path)

    @Slot(str)
    def onClickExt(self, text):
#        pass
        self._show_status_message(text)
        self._detectFile.fileModel.setNameFilters(text)






        #Action in File menu
        self._uiWindow.actionNew_Window.triggered.connect(self.handle_new_window_triggered)
        self._uiWindow.actionOpen_File.triggered.connect(self.handle_file_open_triggered)
        self._uiWindow.actionSave.triggered.connect(self.file_save)
        self._uiWindow.actionSave_As.triggered.connect(self.file_save_as)
        self._uiWindow.actionPrint.triggered.connect(self.file_print)
        self._uiWindow.actionPrint_Preview.triggered.connect(self.file_print_preview)
        self._uiWindow.actionExport_Pdf.triggered.connect(self.file_print_pdf)
        self._uiWindow.actionExit.triggered.connect(self.close)

        #Action in Edit menu
#        self._uiWindow.actionUndo.triggered.connect(self.)
#        self._uiWindow.actionRedo.connect(self.)
#        self._uiWindow.actionCut.triggered.connect(self.)
#        self._uiWindow.actionCopy.triggered.connect(self.)
#        self._uiWindow.actionPaste.triggered.connect(self.)
#        self._uiWindow.actionRemove.triggered.connect(self.)
#        self._uiWindow.actionDelete.triggered.connect(self.)
#        self._uiWindow.actionAdvanced.triggered.connect(self.)
#        self._uiWindow.actionFind_Replace.triggered.connect(self.)
#        self._uiWindow.actionPreferences.triggered.connect(self.)
        #Action in View menu
#        self._uiWindow.actionRefresh.triggered.connect(self.)
#        self._uiWindow.actionFullscreen.triggered.connect(self.)
#        self._uiWindow.actionRestore.triggered.connect(self.)
#        self._uiWindow.actionZoom_In.triggered.connect(self.)
#        self._uiWindow.actionZoom_Out.triggered.connect(self.)
#        self._uiWindow.actionZoom_Original.triggered.connect(self.)
#        self._uiWindow.actionZoom_Fit_Best.triggered.connect(self.)
#        self._uiWindow.actionGoNext.triggered.connect(self.)
#        self._uiWindow.actionGoPrevious.triggered.connect(self.)


#        self._uiWindow.actionAbout_Qt_Creator.triggered.connect(self.)
#        self._uiWindow..triggered.connect(self.)
#        self._uiWindow..triggered.connect(self.)
#        self._uiWindow..triggered.connect(self.)





#def remove_backspace(keys):
#    result = keys.copy()
#    # Chromium already handles navigate on backspace when appropriate.
#    for i, key in enumerate(result):
#        if (key[0].key() & Qt.Key_unknown) == Qt.Key_Backspace:
#            del result[i]
#            break
#    return result

#class DetectionWindow(QMainWindow):
#    about_to_close = Signal()

#    def __init__(self, detection):
#        super().__init__()
##        print("begin detectionwindow init")
#        self._detection = detection
#        self._toolbar_tool = None
#        self._toolbar_navigation = None


##        self._file_tree_view = QTreeView()
##        self.init_file_tree_view()

#        #menuBar
#        mb = self.menuBar()
#        print("begin detectionwindow init")
#        mb.addMenu(self.create_file_menu())
#        mb.addMenu(self.create_edit_menu())
#        mb.addMenu(self.create_view_menu())
#        mb.addMenu(self.create_tools_menu())
#        mb.addMenu(self.create_window_menu())
#        mb.addMenu(self.create_help_menu())
#        print("finish detectionwindow init")

#        self._toolbar_tool = self.create_tool_bar_tool()
#        self._toolbar_navigation = self.create_tool_bar_navigation()
#        self.addToolBar(self._toolbar_tool)
##        self._toolbar_navigation.setAllowedAreas(Qt.TopToolBarArea | Qt.BottomToolBarArea)
#        self.addToolBarBreak(Qt.TopToolBarArea)
#        self.addToolBar(self._toolbar_navigation)

#        self._ui_detectmain = ui_detectmain.Ui_Form()
#        central_widget = QWidget(self)
#        self._ui_detectmain.setupUi(central_widget)
#        self.setCentralWidget(central_widget)
#        self._init_ui_detectmain()


#    @Slot(int)
#    def clicked_dir(self, index):
#        path = self.dir_model.fileInfo(index).absoluteFilePath()
#        self._ui_detectmain._file_listview.setRootIndex(self.file_model.setRootPath(path))
#        self._dir_line_edit.setText(path)
##        print("dir")
##        print(path)
##        print(index)

#    @Slot(int)
#    def clicked_file(self, index):
#        path = self.file_model.fileInfo(index).absoluteFilePath()
#        self._show_status_message(path)

#    @Slot(str)
#    def clicked_ext(self, text):
#        self._show_status_message(text)
#        self.file_model.setNameFilters(text)

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


#    def create_tool_bar_tool(self):
#        tb = QToolBar("File self.actions")
##        tb.setMovable(True)
##        tb.toggleViewAction().setEnabled(True)
#        tb.addAction(self._new_window_action)
#        tb.addAction(self._open_file_action)
#        tb.addAction(self._save_action)
#        tb.addAction(self._print_action)
#        tb.addAction(self._export_pdf_action)
#        tb.addSeparator()
#        tb.addAction(self._stop_action)
#        tb.addAction(self._reload_action)
#        tb.addAction(self._zoom_in_action)
#        tb.addAction(self._zoom_out_action)
#        tb.addAction(self._reset_zoom_action)
#        tb.addSeparator()
#        return tb

#    @Slot()
#    def _back(self):
#        pass

#    @Slot()
#    def _forward(self):
#        pass

#    @Slot()
#    def _stop_reload(self):
#        pass

#    def create_tool_bar_navigation(self):
#        navigation_bar = QToolBar("Navigation")
#        navigation_bar.setMovable(False)
#        navigation_bar.toggleViewAction().setEnabled(False)

#        self._history_back_action = QAction(self)
#        back_shortcuts = remove_backspace(QKeySequence.keyBindings(QKeySequence.Back))

#        # For some reason Qt doesn't bind the dedicated Back key to Back.
#        back_shortcuts.append(QKeySequence(Qt.Key_Back))
#        self._history_back_action.setShortcuts(back_shortcuts)
#        self._history_back_action.setIconVisibleInMenu(False)
#        self._history_back_action.setIcon(QIcon.fromTheme(QIcon.ThemeIcon.GoPrevious))#QIcon.ThemeIcon.GoPrevious)#.ThemeIcon.GoPrevious) #(QIcon(":/3rdparty/go-previous.png"))
##        print('hello', QIcon.themeName)
#        self._history_back_action.setToolTip("Go back in history")
#        self._history_back_action.triggered.connect(self._back)
#        navigation_bar.addAction(self._history_back_action)

#        self._history_forward_action = QAction(self)
#        fwd_shortcuts = remove_backspace(QKeySequence.keyBindings(QKeySequence.Forward))
#        fwd_shortcuts.append(QKeySequence(Qt.Key_Forward))
#        self._history_forward_action.setShortcuts(fwd_shortcuts)
#        self._history_forward_action.setIconVisibleInMenu(False)
#        self._history_forward_action.setIcon(QIcon(":/3rdparty/go-next.png"))
#        self._history_forward_action.setToolTip("Go forward in history")
#        self._history_forward_action.triggered.connect(self._forward)
#        navigation_bar.addAction(self._history_forward_action)

##        self._stop_reload_action = QAction(self)
##        self._stop_reload_action.triggered.connect(self._stop_reload)
##        navigation_bar.addAction(self._stop_reload_action)

#        self._dir_line_edit = QLineEdit(self)
#        self._fav_action = QAction(self)
#        self._dir_line_edit.addAction(self._fav_action, QLineEdit.LeadingPosition)
#        self._dir_line_edit.setClearButtonEnabled(True)
#        navigation_bar.addWidget(self._dir_line_edit)

#        self._search_line_edit = QLineEdit(self)
#        self._search_line_edit.setClearButtonEnabled(True)
#        navigation_bar.addWidget(self._search_line_edit)



#        downloads_action = QAction(self)
#        downloads_action.setIcon(QIcon(":/3rdparty/go-bottom.png"))
#        downloads_action.setToolTip("Show downloads")
#        navigation_bar.addAction(downloads_action)
##        dw = self._browser.download_manager_widget()
##        downloads_action.triggered.connect(dw.show)
#        return navigation_bar



##    def create_tool_bar_bottom(self):
##        tb = QToolBar("File self.actions")
##        tb.addAction(self._save_action)
##        return tb

    @Slot(str)
    def _show_status_message(self, m):
        self.statusBar().showMessage(m)

    @Slot()
    def file_save(self):
        pass
#        if not self._file_name or self._file_name.startswith(":/"):
#            return self.file_save_as()
    @Slot()
    def file_save_as(self):
        file_dialog = QFileDialog(self, "Save as...")
        file_dialog.setAcceptMode(QFileDialog.AcceptSave)
        pass

    @Slot()
    def file_print(self):
        printer = QPrinter(QPrinter.HighResolution)
        dlg = QPrintDialog(printer, self)
        if self._text_edit.textCursor().hasSelection():
            dlg.setOption(QAbstractPrintDialog.PrintSelection)
        dlg.setWindowTitle("Print Document")
        if dlg.exec() == QDialog.Accepted:
            self._text_edit.print_(printer)
    @Slot()
    def file_print_preview(self):
        printer = QPrinter(QPrinter.HighResolution)
        preview = QPrintPreviewDialog(printer, self)
        preview.paintRequested.connect(self._text_edit.print_)
        preview.exec()
    @Slot()
    def file_print_pdf(self):
        file_dialog = QFileDialog(self, "Export PDF")
        file_dialog.setAcceptMode(QFileDialog.AcceptSave)
#        file_dialog.setMimeTypeFilters(["application/pdf"])
#        file_dialog.setDefaultSuffix("pdf")
#        if file_dialog.exec() != QDialog.Accepted:
#            return
#        pdf_file_name = file_dialog.selectedFiles()[0]
#        printer = QPrinter(QPrinter.HighResolution)
#        printer.setOutputFormat(QPrinter.PdfFormat)
#        printer.setOutputFileName(pdf_file_name)
#        self._text_edit.document().print_(printer)
#        native_fn = QDir.toNativeSeparators(pdf_file_name)
#        self.statusBar().showMessage(f'Exported "{native_fn}"')
#    def create_file_menu(self):
#        file_menu = QMenu("&File")
#        icon = QIcon.fromTheme(QIcon.ThemeIcon.DocumentNew)  #"document-new", QIcon(RSRC_PATH + "/filenew.png"))
#        self._new_window_action = file_menu.addAction(icon, "&New Window", QKeySequence.New,self.handle_new_window_triggered)

#        icon = QIcon.fromTheme(QIcon.ThemeIcon.FolderOpen)  #"document-open")#, QIcon.ThemeIcon.FolderOpen)#QIcon(RSRC_PATH + "/fileopen.png"))
#        self._open_file_action = file_menu.addAction(icon, "&Open File...", QKeySequence.Open,self.handle_file_open_triggered)
#        file_menu.addSeparator()
#        icon = QIcon.fromTheme(QIcon.ThemeIcon.DocumentSave)  #"document-save", QIcon(RSRC_PATH + "/filesave.png"))
#        self._save_action = file_menu.addAction(icon, "&Save", self.file_save)
#        self._save_action.setShortcut(QKeySequence.Save)
#        self._save_action.setEnabled(False)
#        self._save_as_action = file_menu.addAction("Save &As...", self.file_save_as)
#        self._save_as_action.setPriority(QAction.LowPriority)
#        file_menu.addSeparator()

#        icon = QIcon.fromTheme(QIcon.ThemeIcon.DocumentPrint)  #"document-print", QIcon(RSRC_PATH + "/fileprint.png"))
#        self._print_action = file_menu.addAction(icon, "&Print...", self.file_print)
#        self._print_action.setPriority(QAction.LowPriority)
#        self._print_action.setShortcut(QKeySequence.Print)

#        icon = QIcon.fromTheme(QIcon.ThemeIcon.DocumentPrintPreview)  #"fileprint", QIcon(RSRC_PATH + "/fileprint.png"))
#        self._print_preview_action = file_menu.addAction(icon, "Print Preview...", self.file_print_preview)

#        icon = QIcon.fromTheme("exportpdf", QIcon(RSRC_PATH + "/exportpdf.png"))
#        self._export_pdf_action = file_menu.addAction(icon, "&Export PDF...", self.file_print_pdf)
#        self._export_pdf_action.setPriority(QAction.LowPriority)
#        self._export_pdf_action.setShortcut(Qt.CTRL | Qt.Key_D)

#        file_menu.addSeparator()

#        self._close_action = QAction("Quit", self)
#        self._close_action.setShortcut(QKeySequence(Qt.CTRL | Qt.Key_Q))
#        self._close_action.triggered.connect(self.close)
#        file_menu.addAction(self._close_action)
#        return file_menu

#    def create_edit_menu(self):
#        edit_menu = QMenu("Edit")
#        return edit_menu


#    @Slot()
#    def _stop(self):
#        pass

#    @Slot()
#    def _reload(self):
#        pass

#    @Slot()
#    def _zoom_in(self):
#        pass

#    @Slot()
#    def _zoom_out(self):
#        pass

#    @Slot()
#    def _reset_zoom(self):
#        pass

#    @Slot()
#    def _toggle_toolbar(self):
#        if self._toolbar_tool.isVisible():
#            self._view_toolbar_action.setText("Show Toolbar")
#            self._toolbar_tool.close()
#            self._toolbar_navigation.close()
#        else:
#            self._view_toolbar_action.setText("Hide Toolbar")
#            self._toolbar_tool.show()
#            self._toolbar_navigation.show()
#    @Slot()
#    def _toggle_statusbar(self):
#        sb = self.statusBar()
#        if sb.isVisible():
#            self._view_statusbar_action.setText("Show Status Bar")
#            sb.close()
#        else:
#            self._view_statusbar_action.setText("Hide Status Bar")
#            sb.show()
#    def create_view_menu(self):
#        view_menu = QMenu("View")
#        self._stop_action = view_menu.addAction(QIcon.fromTheme(QIcon.ThemeIcon.MediaPlaybackStart),"Stop")
#        shortcuts = []
#        shortcuts.append(QKeySequence(Qt.CTRL | Qt.Key_Period))
#        shortcuts.append(QKeySequence(Qt.Key_Escape))
#        self._stop_action.setShortcuts(shortcuts)
#        self._stop_action.triggered.connect(self._stop)

#        self._reload_action = view_menu.addAction("Reload Page")
#        self._reload_action.setShortcuts(QKeySequence.Refresh)
#        self._reload_action.triggered.connect(self._reload)

#        self._zoom_in_action = view_menu.addAction(QIcon.fromTheme(QIcon.ThemeIcon.ZoomIn),"Zoom In")
#        self._zoom_in_action.setShortcut(QKeySequence(Qt.CTRL | Qt.Key_Plus))
#        self._zoom_in_action.triggered.connect(self._zoom_in)

#        self._zoom_out_action = view_menu.addAction(QIcon.fromTheme(QIcon.ThemeIcon.ZoomOut),"Zoom Out")
#        self._zoom_out_action.setShortcut(QKeySequence(Qt.CTRL | Qt.Key_Minus))
#        self._zoom_out_action.triggered.connect(self._zoom_out)

#        self._reset_zoom_action = view_menu.addAction(QIcon.fromTheme(QIcon.ThemeIcon.ZoomFitBest),"Reset Zoom")
#        self._reset_zoom_action.setShortcut(QKeySequence(Qt.CTRL | Qt.Key_0))
#        self._reset_zoom_action.triggered.connect(self._reset_zoom)

#        view_menu.addSeparator()
#        self._view_toolbar_action = QAction("Hide Toolbar", self)
#        self._view_toolbar_action.setShortcut("Ctrl+|")
#        self._view_toolbar_action.triggered.connect(self._toggle_toolbar)
#        view_menu.addAction(self._view_toolbar_action)

#        self._view_statusbar_action = QAction("Hide Status Bar", self)
#        self._view_statusbar_action.setShortcut("Ctrl+/")
#        self._view_statusbar_action.triggered.connect(self._toggle_statusbar)
#        view_menu.addAction(self._view_statusbar_action)
#        return view_menu

#    def create_tools_menu(self):
#        tools_menu = QMenu("Tools")
#        return tools_menu
#    def create_window_menu(self):
#        window_menu = QMenu("Window")
#        return window_menu
#    def create_help_menu(self):
#        help_menu = QMenu("Help")
#        help_menu.addAction("About Qt", qApp.aboutQt)  # noqa: F821
#        return help_menu


    def handle_new_window_triggered(self):
        window = self._detection.create_window()

    def handle_file_open_triggered(self):
        filter = "Web Resources (*.html *.htm *.svg *.png *.jpg *.gif *.svgz);;All files (*.*)"
        url, _ = QFileDialog.getOpenFileUrl(self, "Open Web Resource", "", filter)
#        if url:
#            self.current_tab().setUrl(url)


