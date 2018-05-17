# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mainwindow.ui'
#
# Created by: PyQt4 UI code generator 4.12.1
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName(_fromUtf8("MainWindow"))
        MainWindow.resize(800, 600)
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.verticalLayout_3 = QtGui.QVBoxLayout(self.centralwidget)
        self.verticalLayout_3.setObjectName(_fromUtf8("verticalLayout_3"))
        self.horizontalLayout_3 = QtGui.QHBoxLayout()
        self.horizontalLayout_3.setObjectName(_fromUtf8("horizontalLayout_3"))
        self.DataSetChooser = QtGui.QComboBox(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setItalic(True)
        self.DataSetChooser.setFont(font)
        self.DataSetChooser.setObjectName(_fromUtf8("DataSetChooser"))
        self.DataSetChooser.addItem(_fromUtf8(""))
        self.DataSetChooser.addItem(_fromUtf8(""))
        self.horizontalLayout_3.addWidget(self.DataSetChooser)
        spacerItem = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem)
        spacerItem1 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem1)
        self.verticalLayout_3.addLayout(self.horizontalLayout_3)
        self.horizontalLayout_4 = QtGui.QHBoxLayout()
        self.horizontalLayout_4.setObjectName(_fromUtf8("horizontalLayout_4"))
        self.verticalLayout = QtGui.QVBoxLayout()
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.NormalPic = QtGui.QLabel(self.centralwidget)
        self.NormalPic.setObjectName(_fromUtf8("NormalPic"))
        self.verticalLayout.addWidget(self.NormalPic)
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.NormalDetect = QtGui.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setItalic(True)
        self.NormalDetect.setFont(font)
        self.NormalDetect.setObjectName(_fromUtf8("NormalDetect"))
        self.horizontalLayout.addWidget(self.NormalDetect)
        spacerItem2 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem2)
        self.uploadButton = QtGui.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setItalic(True)
        self.uploadButton.setFont(font)
        self.uploadButton.setObjectName(_fromUtf8("uploadButton"))
        self.horizontalLayout.addWidget(self.uploadButton)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.horizontalLayout_4.addLayout(self.verticalLayout)
        self.verticalLayout_2 = QtGui.QVBoxLayout()
        self.verticalLayout_2.setObjectName(_fromUtf8("verticalLayout_2"))
        self.AdvPic = QtGui.QLabel(self.centralwidget)
        self.AdvPic.setObjectName(_fromUtf8("AdvPic"))
        self.verticalLayout_2.addWidget(self.AdvPic)
        self.horizontalLayout_2 = QtGui.QHBoxLayout()
        self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))
        self.RightDetect = QtGui.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setItalic(True)
        self.RightDetect.setFont(font)
        self.RightDetect.setObjectName(_fromUtf8("RightDetect"))
        self.horizontalLayout_2.addWidget(self.RightDetect)
        spacerItem3 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem3)
        self.craftButton = QtGui.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setItalic(True)
        self.craftButton.setFont(font)
        self.craftButton.setObjectName(_fromUtf8("craftButton"))
        self.horizontalLayout_2.addWidget(self.craftButton)
        self.verticalLayout_2.addLayout(self.horizontalLayout_2)
        self.horizontalLayout_4.addLayout(self.verticalLayout_2)
        self.verticalLayout_3.addLayout(self.horizontalLayout_4)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 31))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(MainWindow)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "AIDefender-demo", None))
        self.DataSetChooser.setItemText(0, _translate("MainWindow", "mnist", None))
        self.DataSetChooser.setItemText(1, _translate("MainWindow", "roadsign", None))
        self.NormalPic.setText(_translate("MainWindow", "TextLabel", None))
        self.NormalDetect.setText(_translate("MainWindow", "Detect", None))
        self.uploadButton.setText(_translate("MainWindow", "Upload", None))
        self.AdvPic.setText(_translate("MainWindow", "TextLabel", None))
        self.RightDetect.setText(_translate("MainWindow", "Detect", None))
        self.craftButton.setText(_translate("MainWindow", "craft", None))

