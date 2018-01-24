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
            super(DisplayWidget, self).setPixmap(self.pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def setPixmap(self, frame):
        # data = frame.to_image()
        # # data = data.resize((1024, 768))
        # data = data.rotate(-90, expand=True)
        data = frame

        self.pixmap = toqpixmap(data)

        super().setPixmap(self.pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))



class MainDisplayWidget(DisplayWidget):
    def __init__(self, parent=None):
        super(MainDisplayWidget, self).__init__(parent)

    def setPixmap(self, frame):
        # data = frame.to_image()
        # # data = data.resize((1024, 768))
        # data = data.rotate(-90, expand=True)
        data = frame

        self.pixmap = toqpixmap(data)

        super().setPixmap(self.pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

class SmallDisplayWidget(DisplayWidget):
    def __init__(self, parent=None):
        super(SmallDisplayWidget, self).__init__(parent)

        self.x1 = 0
        self.y1 = 0
        self.x2 = 500
        self.y2 = 500

    def setX1(self, value):
        self.x1 = value if value < self.x2 else self.x1

    def setY1(self, value):
        self.y1 = value if value < self.y2 else self.y1

    def setX2(self, value):
        self.x2 = value if value > self.x1 else self.x2

    def setY2(self, value):
        self.y2 = value if value > self.y1 else self.y2

    def setPixmap(self, frame):
        # data = frame.to_image()
        # # data = data.resize((1024, 768))
        # data = data.rotate(-90, expand=True)
        cropped = frame.crop((self.x1, self.y1, self.x2, self.y2))

        self.pixmap = toqpixmap(cropped)

        super().setPixmap(self.pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))