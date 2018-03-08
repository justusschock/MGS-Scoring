from queue import Queue
from random import randint

import numpy as np
import time

from PIL import ImageDraw
from PIL.ImageQt import toqpixmap
from PyQt5 import Qt, QtCore
from PyQt5.QtCore import QRectF, QTimer, QThread

from PyQt5.QtWidgets import QWidget

import video_small
import zmq


class AsyncBackendCall(QThread):
    result_ready = QtCore.pyqtSignal(object, object)
    coordinates_ready = QtCore.pyqtSignal(object)

    def __init__(self, widget_index = None, server_address='kepler', server_port='5555'):
        QThread.__init__(self)
        self.current_frame = None
        self.widget_index = widget_index

        context = zmq.Context()
        print("Connecting to server...")
        self.socket = context.socket(zmq.REQ)
        self.socket.connect("tcp://%s:%s" % (server_address, server_port))

    def __del__(self):
        self.wait()

    def run(self):
        # Main program loop
        # Runs infinitely and send the current frame to the server
        while True:
            if(self.current_frame):
                self.make_request(self.current_frame)


    def make_request(self, frame):
        # Send the frame object via zmq socket
        self.socket.send_pyobj(frame)

         # Get the reply
        reply = self.socket.recv_pyobj()

        # Fire the events (ready)
        self.result_ready.emit(reply['value'], self.widget_index)
        self.coordinates_ready.emit(reply['coords'])

    def set_frame(self, item):
        self.current_frame = item

class SmallVideo(QWidget, video_small.Ui_Form):
    plot_value_ready = QtCore.pyqtSignal(object, object)

    def __init__(self, parent, startpoint, endpoint, widget_index, server_address='kepler', server_port='5555'):
        super().__init__(parent=parent)
        self.setupUi(self)

        self.startpoint = startpoint
        self.endpoint = endpoint

        self.worker_thread = AsyncBackendCall(widget_index, server_address, server_port)
        self.worker_thread.start()

        self.should_plot = True
        self.show_plot.stateChanged.connect(self.show_plot_state_changed)

        self.should_show_coordinates = False
        self.show_coordinates.stateChanged.connect(self.show_coordinates_state_changed)

        self.eye_coordinates = []

        self.worker_thread.coordinates_ready.connect(self.update_coordinates)


    def updateFrame(self, frame):
        # The frame has been re-sized (by 1/2) for the box selection, thus
        # all coordinates need to be doubled again here.
        cropped = frame.crop((self.startpoint.x() * 2, self.startpoint.y() * 2, self.endpoint.x() * 2, self.endpoint.y() * 2))

        self.worker_thread.set_frame(cropped)

        if self.should_show_coordinates:
            temp = cropped.copy()

            draw = ImageDraw.Draw(temp)
            r = 3

            for coordinate in self.eye_coordinates:
                draw.ellipse((coordinate[0] - r, coordinate[1] - r, coordinate[0] + r, coordinate[1] + r), fill=(255, 255, 0, 255))

            self.display.setPixmap(temp)
        else:
            self.display.setPixmap(cropped)

    def show_plot_state_changed(self):
        if self.show_plot.isChecked():
            self.should_plot = True
        else:
            self.should_plot = False

    def update_coordinates(self, coordinates):
        self.eye_coordinates = coordinates

    def show_coordinates_state_changed(self):
        if self.show_coordinates.isChecked():
            self.should_show_coordinates = True
        else:
            self.should_show_coordinates = False
