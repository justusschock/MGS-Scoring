# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui/main_new.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 595)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setSizeConstraint(QtWidgets.QLayout.SetMinimumSize)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.play_button = QtWidgets.QPushButton(self.centralwidget)
        self.play_button.setEnabled(False)
        self.play_button.setObjectName("play_button")
        self.horizontalLayout.addWidget(self.play_button)
        self.frame_selector = QtWidgets.QSpinBox(self.centralwidget)
        self.frame_selector.setEnabled(False)
        self.frame_selector.setMinimumSize(QtCore.QSize(50, 0))
        self.frame_selector.setObjectName("frame_selector")
        self.horizontalLayout.addWidget(self.frame_selector)
        self.frame_slider = QtWidgets.QSlider(self.centralwidget)
        self.frame_slider.setEnabled(False)
        self.frame_slider.setOrientation(QtCore.Qt.Horizontal)
        self.frame_slider.setObjectName("frame_slider")
        self.horizontalLayout.addWidget(self.frame_slider)
        self.gridLayout_2.addLayout(self.horizontalLayout, 2, 0, 1, 1)
        self.plot_widget = PlotWidget(self.centralwidget)
        self.plot_widget.setEnabled(True)
        self.plot_widget.setMinimumSize(QtCore.QSize(0, 200))
        self.plot_widget.setObjectName("plot_widget")
        self.gridLayout_2.addWidget(self.plot_widget, 1, 0, 1, 1)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.verticalLayout.addLayout(self.gridLayout)
        self.gridLayout_2.addLayout(self.verticalLayout, 0, 0, 1, 1)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setObjectName("label")
        self.horizontalLayout_2.addWidget(self.label)
        self.ip_input = QtWidgets.QLineEdit(self.centralwidget)
        self.ip_input.setObjectName("ip_input")
        self.horizontalLayout_2.addWidget(self.ip_input)
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_2.addWidget(self.label_2)
        self.port_input = QtWidgets.QSpinBox(self.centralwidget)
        self.port_input.setMinimumSize(QtCore.QSize(70, 0))
        self.port_input.setMaximum(65535)
        self.port_input.setProperty("value", 5555)
        self.port_input.setObjectName("port_input")
        self.horizontalLayout_2.addWidget(self.port_input)
        self.gridLayout_2.addLayout(self.horizontalLayout_2, 3, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 21))
        self.menubar.setObjectName("menubar")
        self.menuDatei = QtWidgets.QMenu(self.menubar)
        self.menuDatei.setObjectName("menuDatei")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.action_open = QtWidgets.QAction(MainWindow)
        self.action_open.setObjectName("action_open")
        self.actionBoxen_festlegen = QtWidgets.QAction(MainWindow)
        self.actionBoxen_festlegen.setEnabled(False)
        self.actionBoxen_festlegen.setObjectName("actionBoxen_festlegen")
        self.menuDatei.addAction(self.action_open)
        self.menuDatei.addAction(self.actionBoxen_festlegen)
        self.menubar.addAction(self.menuDatei.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MGS Projekt"))
        self.play_button.setText(_translate("MainWindow", "Start"))
        self.label.setText(_translate("MainWindow", "Server Adresse: "))
        self.ip_input.setText(_translate("MainWindow", "kepler"))
        self.label_2.setText(_translate("MainWindow", "Server Port:"))
        self.menuDatei.setTitle(_translate("MainWindow", "Datei"))
        self.action_open.setText(_translate("MainWindow", "Öffnen..."))
        self.action_open.setShortcut(_translate("MainWindow", "Ctrl+O"))
        self.actionBoxen_festlegen.setText(_translate("MainWindow", "Boxen festlegen"))

from pyqtgraph import PlotWidget