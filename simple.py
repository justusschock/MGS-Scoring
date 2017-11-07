import sys
import av
from PyQt5 import QtCore

from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QTextEdit, QAction, QFileDialog, QMainWindow
from PyQt5.QtGui import QIcon, QPixmap, QImage

from PIL.ImageQt import toqpixmap



class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.title = 'MGS Projekt'
        self.left = 100
        self.top = 100
        self.width = 640
        self.height = 480
        self.i = 1

        self.initUI()

        # timer = QtCore.QTimer(self)
        # timer.timeout.connect(self.timerEvent)
        # timer.start(100)

    def initUI(self):
        self.initMenu()

        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        # Create widget
        self.label = QLabel(self)
        # pixmap = QPixmap('sandbox/0000.jpg')
        # self.label.setPixmap(pixmap)
        self.setCentralWidget(self.label)
        # # self.resize(pixmap.width(), pixmap.height())
        # # self.pixmap = pixmap

        self.show()

    def initMenu(self):
        self.statusBar()

        openFile = QAction(QIcon('icons/actions/32/document-open.png'), 'Öffnen', self)
        openFile.setShortcut('Ctrl+O')
        openFile.setStatusTip('Datei öffnen')
        openFile.triggered.connect(self.showDialog)

        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&Datei')
        fileMenu.addAction(openFile)

    def showDialog(self):
        fname = QFileDialog.getOpenFileName(self, 'Datei öffnen', 'F:\Projects\Projekt_Maus', '*.mov')

        if fname[0]:
            self.container = av.open(fname[0])

            timer = QtCore.QTimer(self)
            timer.timeout.connect(self.timerEvent)
            timer.start(100)


    def timerEvent(self, **kwargs):
        frame = next(self.container.decode(video=0))
        # frame.to_image().save('sandbox/test.jpg')
        # pixmap = QPixmap('sandbox/000' + str(self.i % 7) + '.jpg')
        data = frame.to_image()
        # data = data.rotate(-90)
        pixmap = toqpixmap(data)
        pixmap = pixmap.scaledToWidth(640)
        self.label.setPixmap(pixmap)

        self.i += 1

        print('test')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())