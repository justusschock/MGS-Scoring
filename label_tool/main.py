import sys
import os

import h5py
import numpy as np

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QDialog

import label_tool.main_layout as main_layout
from label_tool.video import FrameGrabber
from label_tool.select_box_widget import SelectBox


class MGSApp(QMainWindow, main_layout.Ui_MainWindow):
    request_frame = pyqtSignal(object)
    request_next_frame = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.frame_grabber = FrameGrabber()

        self.action_open.triggered.connect(self.showDialog)
        self.action_save.triggered.connect(self.showSaveDialog)
        self.actionBoxen_festlegen.triggered.connect(self.showPopup)

        self.frame_grabber.update_frame_range.connect(self.set_frame_range)
        self.request_frame.connect(self.frame_grabber.request_frame)
        self.request_next_frame.connect(self.frame_grabber.request_next_frame)

        self.frame_slider.valueChanged.connect(self.frame_changed)
        self.frame_selector.valueChanged.connect(self.frame_changed)

        self.frame_grabber.frame_ready.connect(self.display_widget.setPixmap)

        self.pushButton.clicked.connect(self.buttonClicked)
        self.pushButton_2.clicked.connect(self.buttonClicked)
        self.pushButton_3.clicked.connect(self.buttonClicked)
        self.pushButton_4.clicked.connect(self.buttonClicked)
        self.pushButton_5.clicked.connect(self.buttonClicked)
        self.pushButton_6.clicked.connect(self.buttonClicked)
        self.pushButton_7.clicked.connect(self.buttonClicked)
        self.pushButton_8.clicked.connect(self.buttonClicked)
        self.pushButton_9.clicked.connect(self.buttonClicked)
        self.pushButton_10.clicked.connect(self.buttonClicked)
        self.pushButton_11.clicked.connect(self.buttonClicked)

        self.data_images = []
        self.data_labels = []

    def showDialog(self):
        fname = QFileDialog.getOpenFileName(self, 'Datei öffnen', 'V:\MGS\Festplatte\Gruppe i.p\Woche 1\MGS\W1', '*.mov')

        if fname[0]:
            self.frame_grabber.set_file(fname[0])
            self.enableVideoControls()
            self.enablePushButtons()

            self.statusbar.showMessage('Video erfolgreich geladen', 3000)

    def showSaveDialog(self):
        fname = QFileDialog.getSaveFileName(self, 'Datei speichern', 'mouse_dataset', filter='*.hdf5')

        if fname[0]:
            # try:
            #     os.remove(fname[0])
            # except OSError:
            #     pass

            with h5py.File(fname[0], "w") as file:
                file.create_dataset('labels', data=np.array(self.data_labels, dtype=np.float))
                file.create_dataset('images', data=np.array(self.data_images))

    def buttonClicked(self):

        sender = self.sender()

        index = int(self.frame_selector.text())
        frame = self.frame_grabber.get_frame(index)
        img = frame.to_image().rotate(-90, expand=True)
        data = self.display_widget.getData(img)

        self.data_images.append(np.array(data))

        if sender.text() == 'X':
            self.data_labels.append(None)
        else:
            self.data_labels.append(int(sender.text()))

        self.frame_selector.setValue(index + self.skip_frames_selector.value())

        self.statusbar.showMessage(sender.text() + ' wurde gedrückt')


    def set_frame_range(self, maximum):
        print("frame range =", maximum)
        self.frame_slider.setMaximum(maximum)
        self.frame_selector.setMaximum(maximum)

    def frame_changed(self, value):
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(value)
        self.frame_slider.blockSignals(False)

        self.frame_selector.blockSignals(True)
        self.frame_selector.setValue(value)
        self.frame_selector.blockSignals(False)

        #self.display.current_index = value
        self.frame_grabber.active_frame = value

        self.request_frame.emit(value)

    def enableVideoControls(self):
        # self.play_button.setEnabled(True)
        self.frame_slider.setEnabled(True)
        self.actionBoxen_festlegen.setEnabled(True)
        self.frame_selector.setEnabled(True)

    def enablePushButtons(self):
        self.pushButton.setEnabled(True)
        self.pushButton_2.setEnabled(True)
        self.pushButton_3.setEnabled(True)
        self.pushButton_4.setEnabled(True)
        self.pushButton_5.setEnabled(True)
        self.pushButton_6.setEnabled(True)
        self.pushButton_7.setEnabled(True)
        self.pushButton_8.setEnabled(True)
        self.pushButton_9.setEnabled(True)
        self.pushButton_10.setEnabled(True)
        self.pushButton_11.setEnabled(True)

    def showPopup(self):
        self.w = SelectBox()
        index, frame = next(self.frame_grabber.next_frame())
        self.w.setPicture(frame)
        self.w.show()

        if self.w.exec_() == QDialog.Accepted:
            startpoint, endpoint = self.w.getPoints()

            self.display_widget.setStartpoint(startpoint)
            self.display_widget.setEndpoint(endpoint)

            self.frame_selector.setValue(1)


if __name__ == '__main__':  # if we're running file directly and not importing it
    app = QApplication(sys.argv)  # A new instance of QApplication
    form = MGSApp()  # We set the form to be our ExampleApp (design)
    form.show()  # Show the form
    app.exec_()
