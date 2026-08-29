# This Python file uses the following encoding: utf-8

import sys
import os
from PySide6.QtWidgets import (QMainWindow, QFileDialog,QColorDialog, QComboBox,
                                QDialog, QFontComboBox,QTextEdit, QInputDialog,
                                QLineEdit, QMenu, QMessageBox,QProgressBar, QToolBar,
                                QVBoxLayout, QWidget, QTreeView, QTableView, QFileSystemModel,
                                QHeaderView, QHBoxLayout, QSplitter)
from PySide6.QtGui import (QAction, QGuiApplication, QIcon, QKeySequence, QStandardItemModel,
                            QStandardItem, QImage, QPixmap)
from PySide6.QtCore import (QUrl, Qt, Slot, Signal, QDir, QEvent)

from PySide6.QtPrintSupport import (QAbstractPrintDialog, QPrinter,
                                    QPrintDialog, QPrintPreviewDialog)
from PySide6.QtSql import QSqlTableModel
from PySide6.QtGui import QIcon
#from tr import tr
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from scipy import stats
import openpyxl

from ui import ui_detectmain
from camera import Camera


class DetectMain(QWidget):
    def __init__(self):
        super().__init__()
        self.ui = ui_detectmain.Ui_Form()
        self.ui.setupUi(self)
        self._install_resizable_splitters()

#        #put the cameraWidget in cameraMainlVLayout
        self.mainCamera = Camera()
#        self.ui.cameraMainGroupBox.setWidget(self.mainCamera._ui.cameraWidget)
        self.ui.cameraMainlVLayout.addWidget(self.mainCamera._ui.cameraWidget)
        self.ui.cameraMainlVLayout.setContentsMargins(0,0,0,0)


        #
        self.layout = QVBoxLayout()
        self.ui.widgetChart.setLayout(self.layout)

        # canvas
#        self.figure = plt.figure()
        self.figure, self.ax = plt.subplots(figsize=(15,15), dpi=100)#nrows=1, ncols=1,figsize=(15,10))#figsize=(20,15))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.layout.addWidget(self.canvas)
#        self.layout.addWidget(self.toolbar)




        # Linear regression: read source data (full precision for the fit)
        lin_headers, lin_rows = self._read_excel('./interface/linear/table/linear_regression_table.xlsx')
        ycol = self._first_color_index(lin_headers)
        con_lin = [r[lin_headers.index("Con.")] for r in lin_rows]
        y_channel = [r[ycol] for r in lin_rows]
        y_label = lin_headers[ycol]
        channel_color = self.CHANNEL_COLORS.get(y_label, '#2ca02c')
        detect_color = self.CONTRAST_COLORS.get(y_label, '#ff7f0e')

        # Linear regression display table: read from only_table/
        disp_headers, disp_rows = self._read_excel(
            './interface/linear/only_table/linear_regression_table.xlsx')
        self._populate_tableview(self.ui.tabviewOrig, disp_headers, disp_rows)



        # plot the Chart
        font = {'family': 'serif',
                'color':  'black',
                'weight': 'normal',
                'size': 13,
                }

        x = con_lin
        y = y_channel
        slope, intercept, r, p, std_err = stats.linregress(x, y)
        R2 = pow(r, 2)
        def myfunc(x):
            return slope*x+intercept
        mymodel = list(map(myfunc, x))

        text = "Y = {:.4f}*X+{:.2f}".format(slope, intercept)
#        print('text=', text)
#        print("y = {:.2f}*x+{:.2f}".format(slope, intercept))

        # Detection: read detection-source data, compute predicted concentrations
        det_headers, det_rows = self._read_excel('./interface/detect/table/detection_table.xlsx')
        dcol = self._first_color_index(det_headers)
        det_channel = [r[dcol] for r in det_rows]
        con_pred = [(v - intercept) / slope for v in det_channel]
        print('con: ', con_pred)

#        plt.plot(x, y, 'k')
        plt.title('Linear Regression and Detection of Real Samples', fontdict=font, fontsize=15) # 'Linear Regression and ML-assisted HT-Detection'

        # Formula overlay: keep clear of the regression line by docking to the
        # corner opposite the slope direction; axes-fraction coords keep it
        # stable across data scales.
        if slope >= 0:
            tx, ha = 0.02, 'left'
        else:
            tx, ha = 0.98, 'right'
        self.ax.text(tx, 0.96,
                "Y = {:.4f} * X+{:.2f} ,  R$^2$ = {:.4f}".format(slope, intercept, R2),
                transform=self.ax.transAxes,
                ha=ha, va='top',
                fontsize=16, fontstyle='italic', fontfamily='times new roman',
                color=(0, 0, 0, 1),
                bbox=dict(facecolor=channel_color, alpha=0.5, edgecolor='none',
                          boxstyle='round,pad=0.3'))
        # plt.text(61, 143, r'$\cos(2 \pi t) \exp(-t)$', fontdict=font)
        plt.xlabel('Concentration of AA (μM)', fontdict=font, fontsize=13) #'Concentration of Hg$^{2+}$ (μM)'
        plt.ylabel('{} Value'.format(y_label), fontdict=font, fontsize=13)



        plt.scatter(x, y, color=channel_color, linewidths=4, zorder=1)
        plt.plot(x, mymodel, color=channel_color, linewidth=3, linestyle='--', alpha=0.7, zorder=2)
        plt.scatter(con_pred, det_channel, color=detect_color, linewidths=4, zorder=3)




        plt.legend(('experimental data', 'linear regression', 'detection result'),
                   loc='lower right', shadow=True)

        marker_color = detect_color
        marker_size = 9   # one size smaller than before
        # 28 px between bands comfortably exceeds the rendered label-box height
        # (~19 px at fontsize 9 with pad=1) so adjacent-band labels cannot overlap
        # vertically. Anchored at each point's x preserves left-to-right order; x
        # collisions bump the label to the next band. label_half_w stays pessimistic
        # so tight clusters still fan out across distinct bands.
        dx_global = -3
        order = sorted(range(len(con_pred)), key=lambda i: con_pred[i])
        x_span = (max(con_pred) - min(con_pred)) if len(con_pred) > 1 else 1.0
        label_half_w = max(x_span * 0.11, 1e-6)
        dy_cycle = [-26, -54, -82, -110, 26, 54, 82, 110]
        band_intervals = {dy: [] for dy in dy_cycle}
        assigned_dy = [dy_cycle[0]] * len(con_pred)
        for idx in order:
            xL = con_pred[idx] - label_half_w
            xR = con_pred[idx] + label_half_w
            for dy in dy_cycle:
                collides = any(max(xL, l) < min(xR, r) for l, r in band_intervals[dy])
                if not collides:
                    band_intervals[dy].append((xL, xR))
                    assigned_dy[idx] = dy
                    break
        for i, (cx, cy) in enumerate(zip(con_pred, det_channel)):
            plt.annotate('({:.2f},{:.2f})'.format(cx, cy),
                         xy=(cx, cy), xytext=(dx_global, assigned_dy[i]),
                         textcoords='offset points',
                         ha='center', va='center',
                         fontsize=marker_size, color=marker_color,
                         bbox=dict(facecolor='white', alpha=0.75,
                                   edgecolor='none', pad=1.0),
                         arrowprops=dict(arrowstyle='-', color='lightgray',
                                         lw=0.8, linestyle='--', alpha=0.9))

        # Modest axis padding so labels stay inside the plot frame at any window size.
        y0, y1 = self.ax.get_ylim()
        yr = y1 - y0
        self.ax.set_ylim(y0 - yr * 0.12, y1 + yr * 0.06)


        # Render into the embedded Qt canvas only — no standalone pyplot window.
        self.canvas.draw()



        # Detection display table: read from only_table/
        det_disp_headers, det_disp_rows = self._read_excel(
            './interface/detect/only_table/detection_table.xlsx')
        self._populate_tableview(self.ui.tabviewRecg, det_disp_headers, det_disp_rows)


        self.origImg = self._resolve_image('./interface/linear/image/linear_regression_image')
        self.recgImg = self._resolve_image('./interface/detect/image/detection_image')
        self._origPixmap = QPixmap(self.origImg)
        self._recgPixmap = QPixmap(self.recgImg)
        for lbl in (self.ui.labelOrigImg, self.ui.labelRecgImg):
            lbl.setMinimumSize(1, 1)
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setScaledContents(False)
            # Event filter catches the label's own resize, which fires reliably
            # even when DetectMain itself isn't in a layout (the UI widget is).
            lbl.installEventFilter(self)
        self._update_image_scales()

        self.ui.progressBar.setValue(100)
        try:
            with open('./interface/detect/time.txt', 'r', encoding='utf-8') as f:
                raw = f.read().strip()
            digits = ''.join(ch for ch in raw if ch.isdigit())
            if digits:
                self.ui.lcdNumber.display(int(digits))
        except OSError:
            pass

    CHANNEL_COLORS = {
        'Red':   '#d62728',
        'Green': '#2ca02c',
        'Blue':  '#1f77b4',
    }

    # High-contrast partners for each regression channel: complementary-ish hues
    # that stay clearly separated from the corresponding CHANNEL_COLORS entry.
    CONTRAST_COLORS = {
        'Red':   '#17becf',   # cyan vs red
        'Green': '#e377c2',   # magenta vs green
        'Blue':  '#ff7f0e',   # orange vs blue
    }

    IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png')

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_image_scales()

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.Resize and obj in (
                getattr(self.ui, 'labelOrigImg', None),
                getattr(self.ui, 'labelRecgImg', None)):
            self._scale_label(obj)
        return super().eventFilter(obj, event)

    def _update_image_scales(self):
        for label in (getattr(self.ui, 'labelOrigImg', None),
                      getattr(self.ui, 'labelRecgImg', None)):
            if label is not None:
                self._scale_label(label)

    def _install_resizable_splitters(self):
        self._install_linear_regression_splitter()
        self._install_detection_splitter()

    @staticmethod
    def _configure_splitter(splitter, sizes):
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(6)
        for index, stretch in enumerate(sizes):
            splitter.setStretchFactor(index, stretch)
        splitter.setSizes([size * 100 for size in sizes])

    def _install_linear_regression_splitter(self):
        self.ui.horizontalLayout_6.removeWidget(self.ui.tabWidget_2)
        self.ui.verticalLayout_8.removeWidget(self.ui.label_2)
        self.ui.verticalLayout_8.removeWidget(self.ui.tabviewOrig)

        table_panel = QWidget(self.ui.cameraMainGroupBox)
        table_panel.setObjectName("linearRegressionTablePanel")
        table_layout = QVBoxLayout(table_panel)
        table_layout.setContentsMargins(0, 0, 0, 0)
        table_layout.addWidget(self.ui.label_2)
        table_layout.addWidget(self.ui.tabviewOrig)

        splitter = QSplitter(Qt.Horizontal, self.ui.cameraMainGroupBox)
        splitter.setObjectName("linearRegressionSplitter")
        splitter.addWidget(self.ui.tabWidget_2)
        splitter.addWidget(table_panel)
        self._configure_splitter(splitter, [5, 2])

        self.ui.horizontalLayout_9.removeItem(self.ui.horizontalLayout_6)
        self.ui.horizontalLayout_9.addWidget(splitter)

    def _install_detection_splitter(self):
        self.ui.verticalLayout_10.removeWidget(self.ui.tabWidget)
        for widget in (self.ui.label_5, self.ui.progressBar,
                       self.ui.label_9, self.ui.lcdNumber):
            self.ui.horizontalLayout_2.removeWidget(widget)
        self.ui.verticalLayout_9.removeWidget(self.ui.label_4)
        self.ui.verticalLayout_9.removeWidget(self.ui.tabviewRecg)

        image_panel = QWidget(self.ui.groupBox_3)
        image_panel.setObjectName("detectionImagePanel")
        image_layout = QVBoxLayout(image_panel)
        image_layout.setContentsMargins(0, 0, 0, 0)
        image_layout.addWidget(self.ui.tabWidget)
        progress_layout = QHBoxLayout()
        progress_layout.addWidget(self.ui.label_5)
        progress_layout.addWidget(self.ui.progressBar)
        progress_layout.addWidget(self.ui.label_9)
        progress_layout.addWidget(self.ui.lcdNumber)
        image_layout.addLayout(progress_layout)

        table_panel = QWidget(self.ui.groupBox_3)
        table_panel.setObjectName("detectionTablePanel")
        table_layout = QVBoxLayout(table_panel)
        table_layout.setContentsMargins(0, 0, 0, 0)
        table_layout.addWidget(self.ui.label_4)
        table_layout.addWidget(self.ui.tabviewRecg)

        splitter = QSplitter(Qt.Horizontal, self.ui.groupBox_3)
        splitter.setObjectName("detectionSplitter")
        splitter.addWidget(image_panel)
        splitter.addWidget(table_panel)
        self._configure_splitter(splitter, [5, 2])

        self.ui.verticalLayout_11.removeItem(self.ui.horizontalLayout_4)
        self.ui.verticalLayout_11.addWidget(splitter)

    def _scale_label(self, label):
        if label is getattr(self.ui, 'labelOrigImg', None):
            pix = getattr(self, '_origPixmap', None)
        elif label is getattr(self.ui, 'labelRecgImg', None):
            pix = getattr(self, '_recgPixmap', None)
        else:
            return
        if pix is None or pix.isNull():
            return
        lw, lh = label.width(), label.height()
        if lw <= 0 or lh <= 0:
            return
        # KeepAspectRatioByExpanding fills the target rectangle completely without
        # distortion; we then crop-center so the label never exceeds its own bounds.
        scaled = pix.scaled(lw, lh, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
        sw, sh = scaled.width(), scaled.height()
        if sw > lw or sh > lh:
            x = max(0, (sw - lw) // 2)
            y = max(0, (sh - lh) // 2)
            scaled = scaled.copy(x, y, min(lw, sw), min(lh, sh))
        label.setPixmap(scaled)

    @classmethod
    def _resolve_image(cls, base_no_ext):
        for ext in cls.IMAGE_EXTENSIONS:
            candidate = base_no_ext + ext
            if os.path.exists(candidate):
                return candidate
        return base_no_ext + cls.IMAGE_EXTENSIONS[0]

    @staticmethod
    def _read_excel(path):
        wb = openpyxl.load_workbook(path, data_only=True)
        ws = wb.active
        it = ws.iter_rows(values_only=True)
        headers = [str(h) for h in next(it)]
        rows = [r for r in it if any(c is not None for c in r)]
        return headers, rows

    @staticmethod
    def _first_color_index(headers):
        return headers.index("Con.") + 1

    @staticmethod
    def _populate_tableview(view, headers, rows):
        model = QStandardItemModel()
        model.setColumnCount(len(headers))
        for c, h in enumerate(headers):
            model.setHeaderData(c, Qt.Horizontal, h)
        for r, row in enumerate(rows):
            for c, val in enumerate(row):
                if headers[c] == "No.":
                    text = str(int(val))
                elif isinstance(val, (int, float)):
                    text = "{:.2f}".format(round(float(val), 2))
                else:
                    text = "" if val is None else str(val)
                model.setItem(r, c, QStandardItem(text))
        view.setModel(model)
        view.verticalHeader().hide()
        for c in range(len(headers)):
            view.setColumnWidth(c, view.width() // 2)


# self.ui.labelOrigImg.width(),

#class Camera(QMainWindow):
#class ImageSettings(QDialog):
