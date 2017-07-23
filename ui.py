# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'hugm.ui'
#
# Created: Sun Jul 23 05:42:07 2017
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
        MainWindow.resize(1000, 700)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        self.centralwidget = QtGui.QWidget(MainWindow)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.centralwidget.sizePolicy().hasHeightForWidth())
        self.centralwidget.setSizePolicy(sizePolicy)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.layoutWidget = QtGui.QWidget(self.centralwidget)
        self.layoutWidget.setGeometry(QtCore.QRect(10, 10, 981, 681))
        self.layoutWidget.setObjectName(_fromUtf8("layoutWidget"))
        self.verticalLayout = QtGui.QVBoxLayout(self.layoutWidget)
        self.verticalLayout.setMargin(0)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.buttonRowLayout = QtGui.QHBoxLayout()
        self.buttonRowLayout.setObjectName(_fromUtf8("buttonRowLayout"))
        self.performFFCButton = QtGui.QPushButton(self.layoutWidget)
        self.performFFCButton.setObjectName(_fromUtf8("performFFCButton"))
        self.buttonRowLayout.addWidget(self.performFFCButton)
        self.pixelIndexStepSpinBox = QtGui.QSpinBox(self.layoutWidget)
        self.pixelIndexStepSpinBox.setMinimum(1)
        self.pixelIndexStepSpinBox.setMaximum(512)
        self.pixelIndexStepSpinBox.setObjectName(_fromUtf8("pixelIndexStepSpinBox"))
        self.buttonRowLayout.addWidget(self.pixelIndexStepSpinBox)
        self.pixelIndexSpinBox = QtGui.QSpinBox(self.layoutWidget)
        self.pixelIndexSpinBox.setWrapping(True)
        self.pixelIndexSpinBox.setMaximum(1023)
        self.pixelIndexSpinBox.setObjectName(_fromUtf8("pixelIndexSpinBox"))
        self.buttonRowLayout.addWidget(self.pixelIndexSpinBox)
        self.allOnPushButton = QtGui.QPushButton(self.layoutWidget)
        self.allOnPushButton.setObjectName(_fromUtf8("allOnPushButton"))
        self.buttonRowLayout.addWidget(self.allOnPushButton)
        self.allOffPushButton = QtGui.QPushButton(self.layoutWidget)
        self.allOffPushButton.setObjectName(_fromUtf8("allOffPushButton"))
        self.buttonRowLayout.addWidget(self.allOffPushButton)
        self.chaseSpeedSpinBox = QtGui.QSpinBox(self.layoutWidget)
        self.chaseSpeedSpinBox.setMaximum(9)
        self.chaseSpeedSpinBox.setProperty("value", 5)
        self.chaseSpeedSpinBox.setObjectName(_fromUtf8("chaseSpeedSpinBox"))
        self.buttonRowLayout.addWidget(self.chaseSpeedSpinBox)
        self.rSpinBox = QtGui.QSpinBox(self.layoutWidget)
        self.rSpinBox.setWrapping(True)
        self.rSpinBox.setMaximum(15)
        self.rSpinBox.setObjectName(_fromUtf8("rSpinBox"))
        self.buttonRowLayout.addWidget(self.rSpinBox)
        self.sSpinBox = QtGui.QSpinBox(self.layoutWidget)
        self.sSpinBox.setWrapping(True)
        self.sSpinBox.setMaximum(9)
        self.sSpinBox.setObjectName(_fromUtf8("sSpinBox"))
        self.buttonRowLayout.addWidget(self.sSpinBox)
        self.redSpinBox = QtGui.QSpinBox(self.layoutWidget)
        self.redSpinBox.setMinimum(-255)
        self.redSpinBox.setMaximum(255)
        self.redSpinBox.setObjectName(_fromUtf8("redSpinBox"))
        self.buttonRowLayout.addWidget(self.redSpinBox)
        self.greenSpinBox = QtGui.QSpinBox(self.layoutWidget)
        self.greenSpinBox.setMinimum(-255)
        self.greenSpinBox.setMaximum(255)
        self.greenSpinBox.setObjectName(_fromUtf8("greenSpinBox"))
        self.buttonRowLayout.addWidget(self.greenSpinBox)
        self.blueSpinBox = QtGui.QSpinBox(self.layoutWidget)
        self.blueSpinBox.setMinimum(-255)
        self.blueSpinBox.setMaximum(255)
        self.blueSpinBox.setObjectName(_fromUtf8("blueSpinBox"))
        self.buttonRowLayout.addWidget(self.blueSpinBox)
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
        self.pixelIndexStepSpinBox.setToolTip(_translate("MainWindow", "pixel index increment", None))
        self.pixelIndexSpinBox.setToolTip(_translate("MainWindow", "pixel index", None))
        self.allOnPushButton.setText(_translate("MainWindow", "All On", None))
        self.allOffPushButton.setText(_translate("MainWindow", "All Off", None))
        self.chaseSpeedSpinBox.setToolTip(_translate("MainWindow", "chase speed", None))
        self.rSpinBox.setToolTip(_translate("MainWindow", "r index", None))
        self.sSpinBox.setToolTip(_translate("MainWindow", "s index", None))
        self.imageLabel.setText(_translate("MainWindow", "TextLabel", None))

