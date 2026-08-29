# This Python file uses the following encoding: utf-8

from PySide6.QtCore import QObject, Qt, Slot
from detectionwindow import DetectWindow

class Detection(QObject):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._windows = []


    def create_hidden_window(self):
        print("begin")
        main_window = DetectWindow(self)
        self._windows.append(main_window)
        main_window.about_to_close.connect(self._remove_window)
        print("Detection->create_hidden_window")
        return main_window


    def create_window(self):
        main_window = self.create_hidden_window()
        main_window.show()
        return main_window

    @Slot()
    def _remove_window(self):
        w = self.sender()
        if w in self._windows:
            del self._windows[self._windows.index(w)]
