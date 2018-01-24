from PyQt5 import QtCore

from PIL.ImageQt import toqpixmap
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLabel, QSizePolicy, QWidget


class DisplayWidget(QLabel):
    def __init__(self, parent=None):
        super(DisplayWidget, self).__init__(parent)
        self.setMinimumSize(1080 / 10, 1920 / 10)

        size_policy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        size_policy.setHeightForWidth(True)
        # size_policy.setWidthForHeight(True)

        self.setSizePolicy(size_policy)

        self.setAlignment(Qt.AlignHCenter | Qt.AlignBottom)

        self.pixmap = None
        self.startpoint = None
        self.endpoint = None

        self.setMargin(10)

    def heightForWidth(self, width):
        return width * 16 / 9.0

    def widthForHeight(self, height):
        return height * 9. / 16.

    def sizeHint(self):
        width = self.width()
        height = self.height()
        return QtCore.QSize(self.widthForHeight(height), height)

    def resizeEvent(self, event):
        if self.pixmap:
            super(DisplayWidget, self).setPixmap(
                self.pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def setStartpoint(self, point):
        self.startpoint = point

    def setEndpoint(self, point):
        self.endpoint = point

    def getData(self, frame):
        if self.startpoint and self.endpoint:
            data = frame.crop((self.startpoint.x() * 2, self.startpoint.y() * 2, self.endpoint.x() * 2, self.endpoint.y() * 2))
        else:
            data = frame

        return data

    def setPixmap(self, frame):
        data = self.getData(frame)
        self.pixmap = toqpixmap(data)

        super().setPixmap(self.pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

