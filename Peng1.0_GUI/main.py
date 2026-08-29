# This Python file uses the following encoding: utf-8
import sys
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import (QCoreApplication, QLoggingCategory, QUrl, qVersion)
from PySide6.QtGui import QIcon
from detection import Detection
import rc_img



if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon(":AppLogoColor.png"))
    QCoreApplication.setOrganizationName("QtProject")
    QCoreApplication.setApplicationName("Fast Detection")
    QCoreApplication.setApplicationVersion(qVersion())

    detection = Detection()
    print("in main")
    window = detection.create_hidden_window()


    available_geometry = window.screen().availableGeometry()
    window.resize((available_geometry.width() * 2) / 3,
              (available_geometry.height() * 2) / 3)
    window.move((available_geometry.width() - window.width()) / 2,
            (available_geometry.height() - window.height()) / 2)

    window.show()
    sys.exit(app.exec())
