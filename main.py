import sys
from random import randint
from time import sleep

import numpy as np
from PIL import Image
from PIL.ImageQt import toqpixmap
from PyQt5.QtCore import QRect, QPoint, Qt, QSize, pyqtSignal, QThread, QTimer
from PyQt5.QtGui import QPainter, QImage, QPixmap

import main_new
from small_video_widget import SmallVideo
from utils.video import FrameGrabber
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QFileDialog, QDialog, QVBoxLayout, QPushButton, QLabel, \
    QRubberBand

import pyqtgraph as pg

pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')



class SelectBox(QDialog):

    def __init__(self, parent=None):
        """
        Select box widget
        Lets the user interactively select bounding boxes
        for the mice.

        :param parent:
        """
        super().__init__()

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        self.setWindowTitle('Box selektieren')

        self.imageLabel = QLabel()
        layout.addWidget(self.imageLabel)

        self.setLayout(layout)

        self.rubberBand = QRubberBand(QRubberBand.Rectangle, self)
        self.origin = QPoint()
        self.endpoint = QPoint()

    def setPicture(self, frame):
        """
        Pass the current frame to the widget

        :param frame: The current frame
        """
        data = frame.to_image().rotate(-90, expand=True)
        data.thumbnail((960, 960),Image.ANTIALIAS)
        self.imageLabel.setPixmap(toqpixmap(data))

    def mousePressEvent(self, event):
        """
        Triggers when left mouse button is pressed
        Saves the origin (current mouse position) and initializes
        the (draggable) rubber band.

        :param event: Event
        """
        if event.button() == Qt.LeftButton:
            self.origin = QPoint(event.pos())
            self.rubberBand.setGeometry(QRect(self.origin, QSize()))
            self.rubberBand.show()

    def mouseMoveEvent(self, event):
        """
        Updates the rubber band

        :param event: Event
        """
        if not self.origin.isNull():
            self.rubberBand.setGeometry(QRect(self.origin, event.pos()).normalized())

    def mouseReleaseEvent(self, event):
        """
        Triggers when left mouse button is released
        Saves the end point (current mouse position) of the
        box and closes the widget window.

        :param event: Event
        """
        if event.button() == Qt.LeftButton:
            self.endpoint = QPoint(event.pos())
            self.accept()

    def getPoints(self):
        """
        Returns the start and end point of the box
        This corresponds to the upper left and lower right
        coordinates of the box in the frame.

        :return: QPoint, QPoint
        """
        return self.origin, self.endpoint


class MGSApp(QMainWindow, main_new.Ui_MainWindow):
    """
    Signal to request a specific frame by frame index
    """
    request_frame = pyqtSignal(object)

    """
    Signal to request the next frame
    """
    request_next_frame = pyqtSignal(object)


    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.frame_grabber = FrameGrabber()
        self.action_open.triggered.connect(self.showDialog)
        self.actionBoxen_festlegen.triggered.connect(self.showPopup)

        self.frame_grabber.update_frame_range.connect(self.set_frame_range)
        self.request_frame.connect(self.frame_grabber.request_frame)
        self.request_next_frame.connect(self.frame_grabber.request_next_frame)

        self.frame_slider.valueChanged.connect(self.frame_changed)
        self.frame_selector.valueChanged.connect(self.frame_changed)

        self.frame_grabber.frame_ready.connect(self.updateWidets)

        self.mouse_widgets = []

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.next_frame)

        self.play_button.clicked.connect(self.onPushPlayButton)

        self.init_plot_widget()
        self.plot_curves = []

    def init_plot_widget(self):
        """
        Sets some defaults for the plot widget
        Draws a vertical line to indicate the current frame

        """
        self.plot_widget.setDownsampling(mode='peak')
        self.plot_widget.setClipToView(True)
        self.plot_widget.setYRange(min=0, max=9)
        self.plot_widget.plotItem.showGrid(x=True, y=True)

        self.line_indicator = pg.InfiniteLine(0, 90, label="Akt. Frame", pen=pg.mkPen((100, 100, 100), width=3),
                                              labelOpts={'position': 0.9, 'color': (100, 100, 100)})
        self.plot_widget.addItem(self.line_indicator)



    def update_plot(self, value, widget_index=0):
        frame = self.frame_grabber.active_frame
        self.plot_data[widget_index,frame] = value

        x = np.arange(frame-200, frame)
        y = self.plot_data[widget_index, x]
        nonzero = np.nonzero(y)

        self.line_indicator.setPos(frame)

        if self.mouse_widgets[widget_index].should_plot:
            self.plot_curves[widget_index].setData(x=x[nonzero], y=y[nonzero], connect="finite")
        else:
            self.plot_curves[widget_index].setData(x=[], y=[])




    def showDialog(self):
        fname = QFileDialog.getOpenFileName(self, 'Datei öffnen', 'V:\MGS\Festplatte\Gruppe i.p\Woche 1\MGS\W1', '*.mov')

        if fname[0]:
            self.frame_grabber.set_file(fname[0])
            self.enableVideoControls()
            self.statusbar.showMessage('Video erfolgreich geladen', 3000)


    def next_frame(self):
        value = self.frame_slider.value() + 1

        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(value)
        self.frame_slider.blockSignals(False)

        self.frame_selector.blockSignals(True)
        self.frame_selector.setValue(value)
        self.frame_selector.blockSignals(False)

        self.frame_grabber.active_frame = value

        self.request_next_frame.emit(value)

    def onPushPlayButton(self):

        if self.timer.isActive():
            self.timer.stop()
            self.play_button.setText("Start")
        else:
            self.timer.start(50)
            self.play_button.setText("Pause")


    def onPushPauseButton(self):
        self.timer.stop()
        self.play_button.setEnabled(True)
        self.pause_button.setEnabled(False)

    def enableVideoControls(self):
        self.play_button.setEnabled(True)
        self.frame_slider.setEnabled(True)
        self.actionBoxen_festlegen.setEnabled(True)
        self.frame_selector.setEnabled(True)


    def showPopup(self):
        self.w = SelectBox()
        index, frame = next(self.frame_grabber.next_frame())

        self.w.setPicture(frame)
        self.w.show()

        if self.w.exec_() == QDialog.Accepted:
            startpoint, endpoint = self.w.getPoints()

            self.addMouseWidget(startpoint, endpoint)
            self.frame_changed(self.frame_slider.value())

    def set_frame_range(self, maximum):
        print("frame range =", maximum)
        self.frame_slider.setMaximum(maximum)
        self.frame_selector.setMaximum(maximum)
        self.plot_data = np.zeros((4, maximum))

    def frame_changed(self, value):
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(value)
        self.frame_slider.blockSignals(False)

        self.frame_selector.blockSignals(True)
        self.frame_selector.setValue(value)
        self.frame_selector.blockSignals(False)

        self.frame_grabber.active_frame = value

        self.request_frame.emit(value)

    def addMouseWidget(self, startpoint, endpoint):

        widgets_count = len(self.mouse_widgets)

        pens = [
            pg.mkPen((255, 0, 0), width=2), # red
            pg.mkPen((0, 255, 0), width=2), # green
            pg.mkPen((0, 0, 255), width=2), # blue
            pg.mkPen((255, 255, 0), width=2) # yellow
        ]

        rgbs = [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0)
        ]

        if widgets_count >= 4:
            return

        server_address = self.ip_input.text()
        server_port = self.port_input.text()

        # Create a new widget for the mouse
        widget = SmallVideo(self.centralwidget, startpoint, endpoint, widgets_count, server_address, server_port)
        widget.setObjectName("mouse_widget_" + str(widgets_count))
        widget.display.setStyleSheet("border: 4px solid rgb" + str(rgbs[widgets_count])) # draw a colored border

        self.plot_curves.append(self.plot_widget.plot(pen=pens[widgets_count]))
        widget.worker_thread.result_ready.connect(self.update_plot)

        row = 1 if widgets_count > 1 else 0
        column = widgets_count % 2
        self.gridLayout.addWidget(widget, row, column, 1, 1)
        widget.show()

        self.mouse_widgets.append(widget)

    def updateWidets(self, frame, index):
        for widget in self.mouse_widgets:
            widget.updateFrame(frame)



if __name__ == '__main__':
    app = QApplication(sys.argv)

    form = MGSApp()
    form.show()

    app.exec_()