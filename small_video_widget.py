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
        while True:
            if(self.current_frame):
                self.make_request(self.current_frame)
            # if(self.queue.empty()):
            #     continue
            #
            # item = self.queue.get()
            # self.make_request(item)
            # self.queue.task_done()

    def make_request(self, frame):
        # time.sleep(1)
        # value = randint(1, 10)

        self.socket.send_pyobj(frame)

         # Get the reply.
        reply = self.socket.recv_pyobj()


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
        # self.widget_index = widget_index

        self.worker_thread = AsyncBackendCall(widget_index, server_address, server_port)
        self.worker_thread.start()

        self.should_plot = True
        self.show_plot.stateChanged.connect(self.show_plot_state_changed)

        self.should_show_coordinates = False
        self.show_coordinates.stateChanged.connect(self.show_coordinates_state_changed)

        self.eye_coordinates = []

        self.worker_thread.coordinates_ready.connect(self.update_coordinates)

        # data1 = np.random.normal(size=300)
        # self.plot.setDownsampling(mode='peak')
        # self.plot.setClipToView(True)
        # self.plot.setYRange(min=0, max=10)
        # # self.plot.setRange(xRange=[-100, 0])
        # self.curve = self.plot.plot()


    def updateFrame(self, frame):
        # data = frame.to_image()
        # # data = data.resize((1024, 768))
        # data = data.rotate(-90, expand=True)
        cropped = frame.crop((self.startpoint.x() * 2, self.startpoint.y() * 2, self.endpoint.x() * 2, self.endpoint.y() * 2))
        # self.display.setPixmap(cropped)
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
