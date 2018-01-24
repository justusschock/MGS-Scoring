from PIL import Image
from PIL.ImageQt import toqpixmap
from PyQt5.QtCore import QRect, QPoint, Qt, QSize

from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, \
    QRubberBand


class SelectBox(QDialog):
    def __init__(self, parent=None):
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
        data = frame.to_image().rotate(-90, expand=True)
        data.thumbnail((960, 960), Image.ANTIALIAS)
        self.imageLabel.setPixmap(toqpixmap(data))

    def mousePressEvent(self, event):

        if event.button() == Qt.LeftButton:
            self.origin = QPoint(event.pos())
            self.rubberBand.setGeometry(QRect(self.origin, QSize()))
            self.rubberBand.show()

    def mouseMoveEvent(self, event):

        if not self.origin.isNull():
            self.rubberBand.setGeometry(QRect(self.origin, event.pos()).normalized())

    def mouseReleaseEvent(self, event):

        if event.button() == Qt.LeftButton:
            self.endpoint = QPoint(event.pos())
            self.accept()
            # self.origin.x()
            # self.rubberBand.hide()

    def getPoints(self):
        return self.origin, self.endpoint
