# This Python file uses the following encoding: utf-8

import sys
from PySide6.QtWidgets import (QMainWindow, QFileDialog,QColorDialog, QComboBox,
                                QDialog, QFontComboBox,QTextEdit, QInputDialog,
                                QLineEdit, QMenu, QMessageBox,QProgressBar, QToolBar,
                                QVBoxLayout, QHBoxLayout, QGridLayout, QWidget,
                                QTreeView, QFileSystemModel)
from PySide6.QtGui import QAction, QActionGroup, QGuiApplication, QIcon, QKeySequence
from PySide6.QtCore import (
    QUrl, Qt, Slot, Signal, QDir, QSize, QTimer, QSignalBlocker
)

from PySide6.QtPrintSupport import (QAbstractPrintDialog, QPrinter,
                                    QPrintDialog, QPrintPreviewDialog)
from PySide6.QtGui import QIcon
from ui import ui_detectmain
from ui import ui_detectwindow
from ui import ui_detectfile
from detectfile import DetectFile
from detectmain import (
    DetectMain,
    LINEAR_MODE_IMAGE_SERIES,
    LINEAR_MODE_SINGLE_IMAGE,
)
from interface_config import load_detection_preferences, save_detection_preferences
from interface_settings_dialog import InterfaceSettingsDialog


#RSRC_PATH = ":/images/mac" if sys.platform == 'darwin' else ":/images/win"
RSRC_PATH = ":/win"


class DetectWindow(QMainWindow):
    about_to_close = Signal(object)
    def __init__(self, detection, camera_enabled):
        super().__init__()
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        self._close_wait_pending = False
        self._shutdown_started = False
        self._final_close_allowed = False
        self._about_to_close_emitted = False
        self._close_wait_dialog = None
        self._close_control_states = None
        self._uiWindow = ui_detectwindow.Ui_detectWindow()
        self._uiWindow.setupUi(self)
        self._detection = detection

        self._detectFile = DetectFile()
        self._detectMain = DetectMain(camera_enabled=camera_enabled)
        self._detectMain.worker_task_finished.connect(
            self._on_worker_task_finished
        )
        self._detectMain.shutdown_ready.connect(self._on_shutdown_ready)
        self._detectMain.busy_changed.connect(self._set_detection_menu_enabled)
        self._detectMain.detection_status_changed.connect(
            self._uiWindow.statusbar.showMessage
        )

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

        # Actions in the File menu are connected once per window instance.
        self._uiWindow.actionNew_Window.triggered.connect(self.handle_new_window_triggered)
        self._uiWindow.actionOpen_File.triggered.connect(self.handle_file_open_triggered)
        self._uiWindow.actionSave.triggered.connect(self.file_save)
        self._uiWindow.actionSave_As.triggered.connect(self.file_save_as)
        self._uiWindow.actionPrint.triggered.connect(self.file_print)
        self._uiWindow.actionPrint_Preview.triggered.connect(self.file_print_preview)
        self._uiWindow.actionExport_Pdf.triggered.connect(self.file_print_pdf)
        self._uiWindow.actionExit.triggered.connect(self.close)
        self.onClickExt(self._detectFile.ui.extComboBox.currentText())

        self._interface_settings_dialog = None
        self._uiWindow.actionPreferences.setText("Interface")
        self._uiWindow.actionPreferences.triggered.connect(
            self._open_interface_settings
        )
        self._setup_detection_menu()
        self._setup_linear_menu()

    def _setup_detection_menu(self):
        self._detection_scope_group = QActionGroup(self)
        self._detection_scope_group.setExclusive(True)
        self._detection_scope_group.addAction(self._uiWindow.actionDetection_Current_Image)
        self._detection_scope_group.addAction(self._uiWindow.actionDetection_Entire_Batch)
        self._numbering_group = QActionGroup(self)
        self._numbering_group.setExclusive(True)
        self._numbering_group.addAction(self._uiWindow.actionDetection_Per_Image)
        self._numbering_group.addAction(self._uiWindow.actionDetection_Continuous)
        settings, warnings = load_detection_preferences()
        self._set_detection_menu_checked_state(settings)
        self._detectMain.set_detection_options(
            settings["detection_scope"], settings["numbering_mode"]
        )
        self._detection_menu_settings = dict(settings)
        self._detection_scope_group.triggered.connect(self._save_detection_menu_settings)
        self._numbering_group.triggered.connect(self._save_detection_menu_settings)
        if warnings:
            self.statusBar().showMessage(warnings[0])

    def _set_detection_menu_checked_state(self, settings):
        actions = (
            self._uiWindow.actionDetection_Current_Image,
            self._uiWindow.actionDetection_Entire_Batch,
            self._uiWindow.actionDetection_Per_Image,
            self._uiWindow.actionDetection_Continuous,
        )
        blockers = [QSignalBlocker(action) for action in actions]
        try:
            self._uiWindow.actionDetection_Current_Image.setChecked(
                settings["detection_scope"] == "current_image"
            )
            self._uiWindow.actionDetection_Entire_Batch.setChecked(
                settings["detection_scope"] == "entire_batch"
            )
            self._uiWindow.actionDetection_Per_Image.setChecked(
                settings["numbering_mode"] == "per_image"
            )
            self._uiWindow.actionDetection_Continuous.setChecked(
                settings["numbering_mode"] == "continuous"
            )
        finally:
            blockers.clear()

    def _save_detection_menu_settings(self, _action=None):
        previous_settings = dict(self._detection_menu_settings)
        if self._detectMain.is_worker_task_active():
            self._set_detection_menu_checked_state(previous_settings)
            return
        settings = {"detection_scope": (
            "entire_batch" if self._uiWindow.actionDetection_Entire_Batch.isChecked()
            else "current_image"
        ), "numbering_mode": (
            "continuous" if self._uiWindow.actionDetection_Continuous.isChecked()
            else "per_image"
        )}
        if settings == previous_settings:
            return
        try:
            save_detection_preferences(settings)
        except Exception as error:
            self._set_detection_menu_checked_state(previous_settings)
            QMessageBox.warning(self, "Detection settings", str(error))
            return
        self._detectMain.set_detection_options(
            settings["detection_scope"], settings["numbering_mode"]
        )
        self._detection_menu_settings = dict(settings)

    def _set_detection_menu_enabled(self, busy):
        self._uiWindow.menuDetection_Scope.setEnabled(not busy)
        self._uiWindow.menuDetection_Numbering.setEnabled(not busy)
        self._uiWindow.menuLinear.setEnabled(not busy)

    def _setup_linear_menu(self):
        self._linear_mode_group = QActionGroup(self)
        self._linear_mode_group.setExclusive(True)
        self._linear_mode_group.addAction(
            self._uiWindow.actionLinear_Single_Image
        )
        self._linear_mode_group.addAction(
            self._uiWindow.actionLinear_Image_Series
        )
        self._linear_mode = LINEAR_MODE_SINGLE_IMAGE
        self._set_linear_menu_checked_state(self._linear_mode)
        try:
            self._detectMain.set_linear_mode(self._linear_mode)
        except Exception as error:
            QMessageBox.warning(self, "Linear mode", str(error))
        self._linear_mode_group.triggered.connect(self._apply_linear_mode)

    def _set_linear_menu_checked_state(self, mode):
        actions = (
            self._uiWindow.actionLinear_Single_Image,
            self._uiWindow.actionLinear_Image_Series,
        )
        blockers = [QSignalBlocker(action) for action in actions]
        try:
            self._uiWindow.actionLinear_Single_Image.setChecked(
                mode == LINEAR_MODE_SINGLE_IMAGE
            )
            self._uiWindow.actionLinear_Image_Series.setChecked(
                mode == LINEAR_MODE_IMAGE_SERIES
            )
        finally:
            blockers.clear()

    @Slot(QAction)
    def _apply_linear_mode(self, action):
        previous_mode = self._linear_mode
        if action is self._uiWindow.actionLinear_Single_Image:
            requested_mode = LINEAR_MODE_SINGLE_IMAGE
        elif action is self._uiWindow.actionLinear_Image_Series:
            requested_mode = LINEAR_MODE_IMAGE_SERIES
        else:
            self._set_linear_menu_checked_state(previous_mode)
            return False

        if self._detectMain.is_linear_interaction_locked():
            self._set_linear_menu_checked_state(previous_mode)
            return False
        if requested_mode == previous_mode:
            self._set_linear_menu_checked_state(previous_mode)
            return True

        try:
            self._detectMain.set_linear_mode(requested_mode)
        except Exception as error:
            try:
                self._detectMain.set_linear_mode(previous_mode)
            except Exception:
                pass
            self._set_linear_menu_checked_state(previous_mode)
            QMessageBox.warning(self, "Linear mode", str(error))
            return False

        self._linear_mode = requested_mode
        self._set_linear_menu_checked_state(requested_mode)
        return True

    def closeEvent(self, event):
        if self._final_close_allowed:
            if not self._about_to_close_emitted:
                self._about_to_close_emitted = True
                self.about_to_close.emit(self)
            event.accept()
            return
        if self._shutdown_started:
            event.ignore()
            return
        if self._close_wait_pending:
            event.ignore()
            self._activate_close_wait_dialog()
            return
        discard_categories = self._close_discard_categories()
        has_mapping_draft = bool(
            "Unconfirmed Linear Series Mapping draft" in discard_categories
        )
        if discard_categories:
            prompt = (
                "Closing will discard:\n- "
                + "\n- ".join(discard_categories)
                + "\n\nDiscard these items and close the window?"
            )
            answer = QMessageBox.question(
                self,
                "Unsaved results",
                prompt,
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                event.ignore()
                return
            if has_mapping_draft:
                try:
                    self._detectMain.discard_linear_series_draft()
                except Exception as error:
                    QMessageBox.warning(self, "Linear series draft", str(error))
                    event.ignore()
                    return
        if self._detectMain.is_worker_task_active():
            self._begin_close_wait()
            event.ignore()
            return
        self._begin_shutdown()
        event.ignore()

    def _has_unsaved_results(self):
        return bool(self._close_discard_categories())

    def _close_discard_categories(self):
        categories = []
        if self._has_linear_series_mapping_draft():
            categories.append("Unconfirmed Linear Series Mapping draft")
        if self._detectMain._regression_dirty:
            categories.append("Unsaved Linear result")
        if self._detectMain._detection_dirty:
            categories.append("Unsaved Detection result")
        return tuple(categories)

    def _has_linear_series_mapping_draft(self):
        checker = getattr(
            self._detectMain, "has_linear_series_mapping_draft", None
        )
        return bool(checker()) if callable(checker) else False

    def _begin_close_wait(self):
        if self._close_wait_pending or self._shutdown_started:
            return
        self._close_wait_pending = True
        self._detectMain.set_close_wait_pending(True)
        self._set_close_controls_disabled(True)
        dialog = QMessageBox(self)
        dialog.setWindowTitle("Waiting to close")
        dialog.setText(
            "Current task is still running. The window will close safely when it finishes."
        )
        dialog.setIcon(QMessageBox.Information)
        dialog.setStandardButtons(QMessageBox.NoButton)
        cancel_button = dialog.addButton("Cancel Close", QMessageBox.RejectRole)
        cancel_button.clicked.connect(self._cancel_close_wait)
        dialog.finished.connect(self._on_close_wait_dialog_finished)
        dialog.setModal(False)
        self._close_wait_dialog = dialog
        dialog.show()

    def _activate_close_wait_dialog(self):
        if self._close_wait_dialog is not None:
            self._close_wait_dialog.raise_()
            self._close_wait_dialog.activateWindow()

    @Slot()
    def _cancel_close_wait(self):
        if not self._close_wait_pending or self._shutdown_started:
            return
        self._close_wait_pending = False
        self._dispose_close_wait_dialog()
        self._detectMain.set_close_wait_pending(False)
        self._set_close_controls_disabled(False)

    @Slot(int)
    def _on_close_wait_dialog_finished(self, _result):
        if self._close_wait_dialog is not None:
            self._cancel_close_wait()

    @Slot()
    def _on_worker_task_finished(self):
        if not self._close_wait_pending or self._shutdown_started:
            return
        self._close_wait_pending = False
        self._dispose_close_wait_dialog()
        self._begin_shutdown()

    def _begin_shutdown(self):
        if self._shutdown_started:
            return
        self._shutdown_started = True
        self._set_close_controls_disabled(True)
        self._detectMain.request_shutdown()

    @Slot()
    def _on_shutdown_ready(self):
        if not self._shutdown_started or self._final_close_allowed:
            return
        self._final_close_allowed = True
        QTimer.singleShot(0, self.close)

    def _dispose_close_wait_dialog(self):
        if self._close_wait_dialog is None:
            return
        dialog = self._close_wait_dialog
        self._close_wait_dialog = None
        dialog.hide()
        dialog.deleteLater()

    def _set_close_controls_disabled(self, disabled):
        actions = (
            self._uiWindow.actionNew_Window,
            self._uiWindow.actionOpen_File,
            self._uiWindow.actionSave,
            self._uiWindow.actionSave_As,
            self._uiWindow.actionPrint,
            self._uiWindow.actionPrint_Preview,
            self._uiWindow.actionExport_Pdf,
            self._uiWindow.actionPreferences,
        )
        if disabled:
            if self._close_control_states is None:
                camera_widget = self._detectMain.mainCamera._ui.cameraWidget
                self._close_control_states = (
                    tuple(action.isEnabled() for action in actions),
                    camera_widget.isEnabled(),
                )
            for action in actions:
                action.setEnabled(False)
            self._detectMain.mainCamera._ui.cameraWidget.setEnabled(False)
        elif self._close_control_states is not None:
            action_states, camera_enabled = self._close_control_states
            for action, enabled in zip(actions, action_states):
                action.setEnabled(enabled)
            self._detectMain.mainCamera._ui.cameraWidget.setEnabled(
                camera_enabled
            )
            self._close_control_states = None

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

    @Slot()
    def _open_interface_settings(self):
        if self._interface_settings_dialog is not None:
            self._interface_settings_dialog.raise_()
            self._interface_settings_dialog.activateWindow()
            return
        dialog = InterfaceSettingsDialog(self)
        self._interface_settings_dialog = dialog
        try:
            dialog.exec()
        finally:
            self._interface_settings_dialog = None
            dialog.deleteLater()
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


