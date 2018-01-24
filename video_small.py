# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui/video_small.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(400, 300)
        self.gridLayout = QtWidgets.QGridLayout(Form)
        self.gridLayout.setObjectName("gridLayout")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.display = DisplayWidget(Form)
        self.display.setStyleSheet("border: 4px solid red;")
        self.display.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.display.setObjectName("display")
        self.verticalLayout.addWidget(self.display)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.show_plot = QtWidgets.QCheckBox(Form)
        self.show_plot.setEnabled(True)
        self.show_plot.setAutoFillBackground(False)
        self.show_plot.setStyleSheet("")
        self.show_plot.setChecked(True)
        self.show_plot.setTristate(False)
        self.show_plot.setObjectName("show_plot")
        self.horizontalLayout.addWidget(self.show_plot)
        self.show_coordinates = QtWidgets.QCheckBox(Form)
        self.show_coordinates.setObjectName("show_coordinates")
        self.horizontalLayout.addWidget(self.show_coordinates)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.gridLayout.addLayout(self.verticalLayout, 0, 0, 1, 1)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.display.setText(_translate("Form", "Video hier"))
        self.show_plot.setText(_translate("Form", "In Plot anzeigen"))
        self.show_coordinates.setText(_translate("Form", "Koordinaten anzeigen"))

from customwidgets import DisplayWidget
