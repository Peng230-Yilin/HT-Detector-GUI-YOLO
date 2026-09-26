# This Python file uses the following encoding: utf-8
import sys
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QCoreApplication
from PySide6.QtGui import QIcon
from detection import Detection
from product_version import PRODUCT_NAME, PRODUCT_VERSION
import rc_img



if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon(":AppLogoColor.png"))
    # Preserve these historical internal identifiers so existing Qt settings
    # remain in the same per-user location. PRODUCT_NAME is the display name.
    QCoreApplication.setOrganizationName("QtProject")
    QCoreApplication.setApplicationName("Fast Detection")
    QCoreApplication.setApplicationVersion(PRODUCT_VERSION)
    app.setApplicationDisplayName(PRODUCT_NAME)

    detection = Detection()
    print("in main")
    window = detection.create_hidden_window()
    window.setWindowTitle("{} {}".format(PRODUCT_NAME, PRODUCT_VERSION))


    available_geometry = window.screen().availableGeometry()
    window.resize((available_geometry.width() * 2) / 3,
              (available_geometry.height() * 2) / 3)
    window.move((available_geometry.width() - window.width()) / 2,
            (available_geometry.height() - window.height()) / 2)

    window.show()
    sys.exit(app.exec())
