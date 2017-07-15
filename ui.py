# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'hugm.ui'
#
# Created: Fri Jul 14 03:01:33 2017
#      by: PyQt4 UI code generator 4.11.2
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
        MainWindow.resize(749, 599)
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.layoutWidget = QtGui.QWidget(self.centralwidget)
        self.layoutWidget.setGeometry(QtCore.QRect(10, 10, 731, 581))
        self.layoutWidget.setObjectName(_fromUtf8("layoutWidget"))
        self.verticalLayout = QtGui.QVBoxLayout(self.layoutWidget)
        self.verticalLayout.setMargin(0)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.buttonRowLayout = QtGui.QHBoxLayout()
        self.buttonRowLayout.setObjectName(_fromUtf8("buttonRowLayout"))
        self.performFFCButton = QtGui.QPushButton(self.layoutWidget)
        self.performFFCButton.setObjectName(_fromUtf8("performFFCButton"))
        self.buttonRowLayout.addWidget(self.performFFCButton)
        self.stepSpinBox = QtGui.QSpinBox(self.layoutWidget)
        self.stepSpinBox.setMinimum(1)
        self.stepSpinBox.setMaximum(512)
        self.stepSpinBox.setObjectName(_fromUtf8("stepSpinBox"))
        self.buttonRowLayout.addWidget(self.stepSpinBox)
        self.indexSpinBox = QtGui.QSpinBox(self.layoutWidget)
        self.indexSpinBox.setObjectName(_fromUtf8("indexSpinBox"))
        self.buttonRowLayout.addWidget(self.indexSpinBox)
        self.allOnPushButton = QtGui.QPushButton(self.layoutWidget)
        self.allOnPushButton.setObjectName(_fromUtf8("allOnPushButton"))
        self.buttonRowLayout.addWidget(self.allOnPushButton)
        self.allOffPushButton = QtGui.QPushButton(self.layoutWidget)
        self.allOffPushButton.setObjectName(_fromUtf8("allOffPushButton"))
        self.buttonRowLayout.addWidget(self.allOffPushButton)
        self.chasePushButton = QtGui.QPushButton(self.layoutWidget)
        self.chasePushButton.setObjectName(_fromUtf8("chasePushButton"))
        self.buttonRowLayout.addWidget(self.chasePushButton)
        spacerItem = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.buttonRowLayout.addItem(spacerItem)
        self.verticalLayout.addLayout(self.buttonRowLayout)
        self.splitter = QtGui.QSplitter(self.layoutWidget)
        self.splitter.setOrientation(QtCore.Qt.Vertical)
        self.splitter.setObjectName(_fromUtf8("splitter"))
        self.imageLabel = QtGui.QLabel(self.splitter)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(9)
        sizePolicy.setHeightForWidth(self.imageLabel.sizePolicy().hasHeightForWidth())
        self.imageLabel.setSizePolicy(sizePolicy)
        self.imageLabel.setStyleSheet(_fromUtf8("background: pink"))
        self.imageLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.imageLabel.setWordWrap(False)
        self.imageLabel.setObjectName(_fromUtf8("imageLabel"))
        self.logTextEdit = QtGui.QTextEdit(self.splitter)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.logTextEdit.sizePolicy().hasHeightForWidth())
        self.logTextEdit.setSizePolicy(sizePolicy)
        self.logTextEdit.setObjectName(_fromUtf8("logTextEdit"))
        self.verticalLayout.addWidget(self.splitter)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow", None))
        self.performFFCButton.setText(_translate("MainWindow", "FFC", None))
        self.allOnPushButton.setText(_translate("MainWindow", "All On", None))
        self.allOffPushButton.setText(_translate("MainWindow", "All Off", None))
        self.chasePushButton.setText(_translate("MainWindow", "Chase", None))
        self.imageLabel.setText(_translate("MainWindow", "TextLabel", None))

