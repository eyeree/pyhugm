# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'hugm.ui'
#
# Created: Thu Jul 27 05:59:09 2017
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
        self.gridLayout = QtGui.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.buttonRowLayout = QtGui.QHBoxLayout()
        self.buttonRowLayout.setSizeConstraint(QtGui.QLayout.SetNoConstraint)
        self.buttonRowLayout.setContentsMargins(0, -1, -1, -1)
        self.buttonRowLayout.setObjectName(_fromUtf8("buttonRowLayout"))
        self.performFFCButton = QtGui.QPushButton(self.centralwidget)
        self.performFFCButton.setObjectName(_fromUtf8("performFFCButton"))
        self.buttonRowLayout.addWidget(self.performFFCButton)
        spacerItem = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.buttonRowLayout.addItem(spacerItem)
        self.gridLayout.addLayout(self.buttonRowLayout, 0, 0, 1, 1)
        self.horizontalSplitter = QtGui.QSplitter(self.centralwidget)
        self.horizontalSplitter.setOrientation(QtCore.Qt.Vertical)
        self.horizontalSplitter.setObjectName(_fromUtf8("horizontalSplitter"))
        self.verticalSplitter = QtGui.QSplitter(self.horizontalSplitter)
        self.verticalSplitter.setOrientation(QtCore.Qt.Horizontal)
        self.verticalSplitter.setObjectName(_fromUtf8("verticalSplitter"))
        self.toolBox = QtGui.QToolBox(self.verticalSplitter)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.toolBox.sizePolicy().hasHeightForWidth())
        self.toolBox.setSizePolicy(sizePolicy)
        self.toolBox.setObjectName(_fromUtf8("toolBox"))
        self.sunPage = QtGui.QWidget()
        self.sunPage.setGeometry(QtCore.QRect(0, 0, 859, 292))
        self.sunPage.setObjectName(_fromUtf8("sunPage"))
        self.formLayout_5 = QtGui.QFormLayout(self.sunPage)
        self.formLayout_5.setFieldGrowthPolicy(QtGui.QFormLayout.FieldsStayAtSizeHint)
        self.formLayout_5.setObjectName(_fromUtf8("formLayout_5"))
        self.label_10 = QtGui.QLabel(self.sunPage)
        self.label_10.setObjectName(_fromUtf8("label_10"))
        self.formLayout_5.setWidget(0, QtGui.QFormLayout.LabelRole, self.label_10)
        self.sunSpeedSpinBox = QtGui.QSpinBox(self.sunPage)
        self.sunSpeedSpinBox.setMinimum(1)
        self.sunSpeedSpinBox.setMaximum(9)
        self.sunSpeedSpinBox.setProperty("value", 1)
        self.sunSpeedSpinBox.setObjectName(_fromUtf8("sunSpeedSpinBox"))
        self.formLayout_5.setWidget(0, QtGui.QFormLayout.FieldRole, self.sunSpeedSpinBox)
        self.toolBox.addItem(self.sunPage, _fromUtf8(""))
        self.colorAdjustPage = QtGui.QWidget()
        self.colorAdjustPage.setGeometry(QtCore.QRect(0, 0, 859, 292))
        self.colorAdjustPage.setObjectName(_fromUtf8("colorAdjustPage"))
        self.horizontalLayout_11 = QtGui.QHBoxLayout(self.colorAdjustPage)
        self.horizontalLayout_11.setObjectName(_fromUtf8("horizontalLayout_11"))
        self.groupBox = QtGui.QGroupBox(self.colorAdjustPage)
        self.groupBox.setObjectName(_fromUtf8("groupBox"))
        self.formLayout = QtGui.QFormLayout(self.groupBox)
        self.formLayout.setFieldGrowthPolicy(QtGui.QFormLayout.AllNonFixedFieldsGrow)
        self.formLayout.setObjectName(_fromUtf8("formLayout"))
        self.label_5 = QtGui.QLabel(self.groupBox)
        self.label_5.setObjectName(_fromUtf8("label_5"))
        self.formLayout.setWidget(0, QtGui.QFormLayout.LabelRole, self.label_5)
        self.horizontalLayout_2 = QtGui.QHBoxLayout()
        self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))
        self.redColorSlider = QtGui.QSlider(self.groupBox)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.redColorSlider.sizePolicy().hasHeightForWidth())
        self.redColorSlider.setSizePolicy(sizePolicy)
        self.redColorSlider.setMaximum(255)
        self.redColorSlider.setPageStep(16)
        self.redColorSlider.setSliderPosition(128)
        self.redColorSlider.setOrientation(QtCore.Qt.Horizontal)
        self.redColorSlider.setTickPosition(QtGui.QSlider.TicksAbove)
        self.redColorSlider.setTickInterval(64)
        self.redColorSlider.setObjectName(_fromUtf8("redColorSlider"))
        self.horizontalLayout_2.addWidget(self.redColorSlider)
        self.redColorSpinBox = QtGui.QSpinBox(self.groupBox)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.redColorSpinBox.sizePolicy().hasHeightForWidth())
        self.redColorSpinBox.setSizePolicy(sizePolicy)
        self.redColorSpinBox.setMaximum(255)
        self.redColorSpinBox.setProperty("value", 128)
        self.redColorSpinBox.setObjectName(_fromUtf8("redColorSpinBox"))
        self.horizontalLayout_2.addWidget(self.redColorSpinBox)
        self.horizontalLayout_2.setStretch(0, 3)
        self.formLayout.setLayout(0, QtGui.QFormLayout.FieldRole, self.horizontalLayout_2)
        self.label_6 = QtGui.QLabel(self.groupBox)
        self.label_6.setObjectName(_fromUtf8("label_6"))
        self.formLayout.setWidget(1, QtGui.QFormLayout.LabelRole, self.label_6)
        self.horizontalLayout_6 = QtGui.QHBoxLayout()
        self.horizontalLayout_6.setObjectName(_fromUtf8("horizontalLayout_6"))
        self.greenColorSlider = QtGui.QSlider(self.groupBox)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.greenColorSlider.sizePolicy().hasHeightForWidth())
        self.greenColorSlider.setSizePolicy(sizePolicy)
        self.greenColorSlider.setMaximum(255)
        self.greenColorSlider.setPageStep(16)
        self.greenColorSlider.setSliderPosition(128)
        self.greenColorSlider.setOrientation(QtCore.Qt.Horizontal)
        self.greenColorSlider.setTickPosition(QtGui.QSlider.TicksAbove)
        self.greenColorSlider.setTickInterval(64)
        self.greenColorSlider.setObjectName(_fromUtf8("greenColorSlider"))
        self.horizontalLayout_6.addWidget(self.greenColorSlider)
        self.greenColorSpinBox = QtGui.QSpinBox(self.groupBox)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.greenColorSpinBox.sizePolicy().hasHeightForWidth())
        self.greenColorSpinBox.setSizePolicy(sizePolicy)
        self.greenColorSpinBox.setMaximum(255)
        self.greenColorSpinBox.setProperty("value", 128)
        self.greenColorSpinBox.setObjectName(_fromUtf8("greenColorSpinBox"))
        self.horizontalLayout_6.addWidget(self.greenColorSpinBox)
        self.horizontalLayout_6.setStretch(0, 3)
        self.formLayout.setLayout(1, QtGui.QFormLayout.FieldRole, self.horizontalLayout_6)
        self.horizontalLayout_3 = QtGui.QHBoxLayout()
        self.horizontalLayout_3.setObjectName(_fromUtf8("horizontalLayout_3"))
        self.allOffPushButton = QtGui.QPushButton(self.groupBox)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.allOffPushButton.sizePolicy().hasHeightForWidth())
        self.allOffPushButton.setSizePolicy(sizePolicy)
        self.allOffPushButton.setMaximumSize(QtCore.QSize(50, 16777215))
        self.allOffPushButton.setObjectName(_fromUtf8("allOffPushButton"))
        self.horizontalLayout_3.addWidget(self.allOffPushButton)
        self.allHalfPushButton = QtGui.QPushButton(self.groupBox)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.allHalfPushButton.sizePolicy().hasHeightForWidth())
        self.allHalfPushButton.setSizePolicy(sizePolicy)
        self.allHalfPushButton.setMaximumSize(QtCore.QSize(50, 16777215))
        self.allHalfPushButton.setObjectName(_fromUtf8("allHalfPushButton"))
        self.horizontalLayout_3.addWidget(self.allHalfPushButton)
        self.allFullPushButton = QtGui.QPushButton(self.groupBox)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.allFullPushButton.sizePolicy().hasHeightForWidth())
        self.allFullPushButton.setSizePolicy(sizePolicy)
        self.allFullPushButton.setMaximumSize(QtCore.QSize(50, 16777215))
        self.allFullPushButton.setObjectName(_fromUtf8("allFullPushButton"))
        self.horizontalLayout_3.addWidget(self.allFullPushButton)
        self.formLayout.setLayout(3, QtGui.QFormLayout.FieldRole, self.horizontalLayout_3)
        self.horizontalLayout_5 = QtGui.QHBoxLayout()
        self.horizontalLayout_5.setObjectName(_fromUtf8("horizontalLayout_5"))
        self.blueColorSlider = QtGui.QSlider(self.groupBox)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.blueColorSlider.sizePolicy().hasHeightForWidth())
        self.blueColorSlider.setSizePolicy(sizePolicy)
        self.blueColorSlider.setMaximum(255)
        self.blueColorSlider.setPageStep(16)
        self.blueColorSlider.setSliderPosition(128)
        self.blueColorSlider.setOrientation(QtCore.Qt.Horizontal)
        self.blueColorSlider.setTickPosition(QtGui.QSlider.TicksAbove)
        self.blueColorSlider.setTickInterval(64)
        self.blueColorSlider.setObjectName(_fromUtf8("blueColorSlider"))
        self.horizontalLayout_5.addWidget(self.blueColorSlider)
        self.blueColorSpinBox = QtGui.QSpinBox(self.groupBox)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.blueColorSpinBox.sizePolicy().hasHeightForWidth())
        self.blueColorSpinBox.setSizePolicy(sizePolicy)
        self.blueColorSpinBox.setMaximum(255)
        self.blueColorSpinBox.setProperty("value", 128)
        self.blueColorSpinBox.setObjectName(_fromUtf8("blueColorSpinBox"))
        self.horizontalLayout_5.addWidget(self.blueColorSpinBox)
        self.horizontalLayout_5.setStretch(0, 3)
        self.formLayout.setLayout(2, QtGui.QFormLayout.FieldRole, self.horizontalLayout_5)
        self.label_7 = QtGui.QLabel(self.groupBox)
        self.label_7.setObjectName(_fromUtf8("label_7"))
        self.formLayout.setWidget(2, QtGui.QFormLayout.LabelRole, self.label_7)
        self.horizontalLayout_11.addWidget(self.groupBox)
        self.adjustmentGroupBox = QtGui.QGroupBox(self.colorAdjustPage)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.adjustmentGroupBox.sizePolicy().hasHeightForWidth())
        self.adjustmentGroupBox.setSizePolicy(sizePolicy)
        self.adjustmentGroupBox.setCheckable(True)
        self.adjustmentGroupBox.setObjectName(_fromUtf8("adjustmentGroupBox"))
        self.formLayout_4 = QtGui.QFormLayout(self.adjustmentGroupBox)
        self.formLayout_4.setFieldGrowthPolicy(QtGui.QFormLayout.AllNonFixedFieldsGrow)
        self.formLayout_4.setContentsMargins(0, -1, -1, -1)
        self.formLayout_4.setHorizontalSpacing(0)
        self.formLayout_4.setObjectName(_fromUtf8("formLayout_4"))
        self.horizontalLayout_7 = QtGui.QHBoxLayout()
        self.horizontalLayout_7.setObjectName(_fromUtf8("horizontalLayout_7"))
        self.redAdjustmentSlider = QtGui.QSlider(self.adjustmentGroupBox)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.redAdjustmentSlider.sizePolicy().hasHeightForWidth())
        self.redAdjustmentSlider.setSizePolicy(sizePolicy)
        self.redAdjustmentSlider.setMinimum(0)
        self.redAdjustmentSlider.setMaximum(11)
        self.redAdjustmentSlider.setPageStep(1)
        self.redAdjustmentSlider.setProperty("value", 5)
        self.redAdjustmentSlider.setSliderPosition(5)
        self.redAdjustmentSlider.setOrientation(QtCore.Qt.Horizontal)
        self.redAdjustmentSlider.setTickPosition(QtGui.QSlider.TicksAbove)
        self.redAdjustmentSlider.setTickInterval(5)
        self.redAdjustmentSlider.setObjectName(_fromUtf8("redAdjustmentSlider"))
        self.horizontalLayout_7.addWidget(self.redAdjustmentSlider)
        self.redAdjustmentDoubleSpinBox = QtGui.QDoubleSpinBox(self.adjustmentGroupBox)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.redAdjustmentDoubleSpinBox.sizePolicy().hasHeightForWidth())
        self.redAdjustmentDoubleSpinBox.setSizePolicy(sizePolicy)
        self.redAdjustmentDoubleSpinBox.setDecimals(2)
        self.redAdjustmentDoubleSpinBox.setMinimum(0.7)
        self.redAdjustmentDoubleSpinBox.setMaximum(1.3)
        self.redAdjustmentDoubleSpinBox.setSingleStep(0.05)
        self.redAdjustmentDoubleSpinBox.setProperty("value", 1.0)
        self.redAdjustmentDoubleSpinBox.setObjectName(_fromUtf8("redAdjustmentDoubleSpinBox"))
        self.horizontalLayout_7.addWidget(self.redAdjustmentDoubleSpinBox)
        self.formLayout_4.setLayout(0, QtGui.QFormLayout.FieldRole, self.horizontalLayout_7)
        self.horizontalLayout_8 = QtGui.QHBoxLayout()
        self.horizontalLayout_8.setObjectName(_fromUtf8("horizontalLayout_8"))
        self.greenAdjustmentSlider = QtGui.QSlider(self.adjustmentGroupBox)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.greenAdjustmentSlider.sizePolicy().hasHeightForWidth())
        self.greenAdjustmentSlider.setSizePolicy(sizePolicy)
        self.greenAdjustmentSlider.setMinimum(0)
        self.greenAdjustmentSlider.setMaximum(11)
        self.greenAdjustmentSlider.setPageStep(1)
        self.greenAdjustmentSlider.setProperty("value", 5)
        self.greenAdjustmentSlider.setSliderPosition(5)
        self.greenAdjustmentSlider.setOrientation(QtCore.Qt.Horizontal)
        self.greenAdjustmentSlider.setTickPosition(QtGui.QSlider.TicksAbove)
        self.greenAdjustmentSlider.setTickInterval(5)
        self.greenAdjustmentSlider.setObjectName(_fromUtf8("greenAdjustmentSlider"))
        self.horizontalLayout_8.addWidget(self.greenAdjustmentSlider)
        self.greenAdjustmentDoubleSpinBox = QtGui.QDoubleSpinBox(self.adjustmentGroupBox)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.greenAdjustmentDoubleSpinBox.sizePolicy().hasHeightForWidth())
        self.greenAdjustmentDoubleSpinBox.setSizePolicy(sizePolicy)
        self.greenAdjustmentDoubleSpinBox.setDecimals(2)
        self.greenAdjustmentDoubleSpinBox.setMinimum(0.7)
        self.greenAdjustmentDoubleSpinBox.setMaximum(1.3)
        self.greenAdjustmentDoubleSpinBox.setSingleStep(0.05)
        self.greenAdjustmentDoubleSpinBox.setProperty("value", 1.0)
        self.greenAdjustmentDoubleSpinBox.setObjectName(_fromUtf8("greenAdjustmentDoubleSpinBox"))
        self.horizontalLayout_8.addWidget(self.greenAdjustmentDoubleSpinBox)
        self.formLayout_4.setLayout(1, QtGui.QFormLayout.FieldRole, self.horizontalLayout_8)
        self.horizontalLayout_10 = QtGui.QHBoxLayout()
        self.horizontalLayout_10.setObjectName(_fromUtf8("horizontalLayout_10"))
        self.blueAdjustmentSlider = QtGui.QSlider(self.adjustmentGroupBox)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.blueAdjustmentSlider.sizePolicy().hasHeightForWidth())
        self.blueAdjustmentSlider.setSizePolicy(sizePolicy)
        self.blueAdjustmentSlider.setMinimum(0)
        self.blueAdjustmentSlider.setMaximum(11)
        self.blueAdjustmentSlider.setPageStep(1)
        self.blueAdjustmentSlider.setProperty("value", 5)
        self.blueAdjustmentSlider.setSliderPosition(5)
        self.blueAdjustmentSlider.setOrientation(QtCore.Qt.Horizontal)
        self.blueAdjustmentSlider.setTickPosition(QtGui.QSlider.TicksAbove)
        self.blueAdjustmentSlider.setTickInterval(5)
        self.blueAdjustmentSlider.setObjectName(_fromUtf8("blueAdjustmentSlider"))
        self.horizontalLayout_10.addWidget(self.blueAdjustmentSlider)
        self.blueAdjustmentDoubleSpinBox = QtGui.QDoubleSpinBox(self.adjustmentGroupBox)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.blueAdjustmentDoubleSpinBox.sizePolicy().hasHeightForWidth())
        self.blueAdjustmentDoubleSpinBox.setSizePolicy(sizePolicy)
        self.blueAdjustmentDoubleSpinBox.setDecimals(2)
        self.blueAdjustmentDoubleSpinBox.setMinimum(0.7)
        self.blueAdjustmentDoubleSpinBox.setMaximum(1.3)
        self.blueAdjustmentDoubleSpinBox.setSingleStep(0.05)
        self.blueAdjustmentDoubleSpinBox.setProperty("value", 1.0)
        self.blueAdjustmentDoubleSpinBox.setObjectName(_fromUtf8("blueAdjustmentDoubleSpinBox"))
        self.horizontalLayout_10.addWidget(self.blueAdjustmentDoubleSpinBox)
        self.formLayout_4.setLayout(2, QtGui.QFormLayout.FieldRole, self.horizontalLayout_10)
        self.horizontalLayout_9 = QtGui.QHBoxLayout()
        self.horizontalLayout_9.setObjectName(_fromUtf8("horizontalLayout_9"))
        self.label_8 = QtGui.QLabel(self.adjustmentGroupBox)
        self.label_8.setObjectName(_fromUtf8("label_8"))
        self.horizontalLayout_9.addWidget(self.label_8)
        self.rAdjustmentSpinBox = QtGui.QSpinBox(self.adjustmentGroupBox)
        self.rAdjustmentSpinBox.setWrapping(True)
        self.rAdjustmentSpinBox.setMaximum(15)
        self.rAdjustmentSpinBox.setObjectName(_fromUtf8("rAdjustmentSpinBox"))
        self.horizontalLayout_9.addWidget(self.rAdjustmentSpinBox)
        self.label_9 = QtGui.QLabel(self.adjustmentGroupBox)
        self.label_9.setObjectName(_fromUtf8("label_9"))
        self.horizontalLayout_9.addWidget(self.label_9)
        self.sAdjustmentSpinBox = QtGui.QSpinBox(self.adjustmentGroupBox)
        self.sAdjustmentSpinBox.setWrapping(True)
        self.sAdjustmentSpinBox.setMaximum(9)
        self.sAdjustmentSpinBox.setObjectName(_fromUtf8("sAdjustmentSpinBox"))
        self.horizontalLayout_9.addWidget(self.sAdjustmentSpinBox)
        self.formLayout_4.setLayout(3, QtGui.QFormLayout.FieldRole, self.horizontalLayout_9)
        self.horizontalLayout_11.addWidget(self.adjustmentGroupBox)
        spacerItem1 = QtGui.QSpacerItem(36, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_11.addItem(spacerItem1)
        self.horizontalLayout_11.setStretch(2, 1)
        self.toolBox.addItem(self.colorAdjustPage, _fromUtf8(""))
        self.indexPage = QtGui.QWidget()
        self.indexPage.setGeometry(QtCore.QRect(0, 0, 859, 292))
        self.indexPage.setObjectName(_fromUtf8("indexPage"))
        self.horizontalLayout = QtGui.QHBoxLayout(self.indexPage)
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.formLayout_2 = QtGui.QFormLayout()
        self.formLayout_2.setObjectName(_fromUtf8("formLayout_2"))
        self.label_2 = QtGui.QLabel(self.indexPage)
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.formLayout_2.setWidget(0, QtGui.QFormLayout.LabelRole, self.label_2)
        self.label_3 = QtGui.QLabel(self.indexPage)
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.formLayout_2.setWidget(1, QtGui.QFormLayout.LabelRole, self.label_3)
        self.pixelIndexStepSpinBox = QtGui.QSpinBox(self.indexPage)
        self.pixelIndexStepSpinBox.setMinimum(1)
        self.pixelIndexStepSpinBox.setMaximum(512)
        self.pixelIndexStepSpinBox.setObjectName(_fromUtf8("pixelIndexStepSpinBox"))
        self.formLayout_2.setWidget(0, QtGui.QFormLayout.FieldRole, self.pixelIndexStepSpinBox)
        self.pixelIndexSpinBox = QtGui.QSpinBox(self.indexPage)
        self.pixelIndexSpinBox.setWrapping(True)
        self.pixelIndexSpinBox.setMaximum(1023)
        self.pixelIndexSpinBox.setObjectName(_fromUtf8("pixelIndexSpinBox"))
        self.formLayout_2.setWidget(1, QtGui.QFormLayout.FieldRole, self.pixelIndexSpinBox)
        self.horizontalLayout.addLayout(self.formLayout_2)
        self.formLayout_3 = QtGui.QFormLayout()
        self.formLayout_3.setObjectName(_fromUtf8("formLayout_3"))
        self.label = QtGui.QLabel(self.indexPage)
        self.label.setObjectName(_fromUtf8("label"))
        self.formLayout_3.setWidget(0, QtGui.QFormLayout.LabelRole, self.label)
        self.label_4 = QtGui.QLabel(self.indexPage)
        self.label_4.setObjectName(_fromUtf8("label_4"))
        self.formLayout_3.setWidget(1, QtGui.QFormLayout.LabelRole, self.label_4)
        self.sIndexSpinBox = QtGui.QSpinBox(self.indexPage)
        self.sIndexSpinBox.setWrapping(True)
        self.sIndexSpinBox.setMaximum(9)
        self.sIndexSpinBox.setObjectName(_fromUtf8("sIndexSpinBox"))
        self.formLayout_3.setWidget(0, QtGui.QFormLayout.FieldRole, self.sIndexSpinBox)
        self.rIndexSpinBox = QtGui.QSpinBox(self.indexPage)
        self.rIndexSpinBox.setWrapping(True)
        self.rIndexSpinBox.setMaximum(15)
        self.rIndexSpinBox.setObjectName(_fromUtf8("rIndexSpinBox"))
        self.formLayout_3.setWidget(1, QtGui.QFormLayout.FieldRole, self.rIndexSpinBox)
        self.horizontalLayout.addLayout(self.formLayout_3)
        spacerItem2 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem2)
        self.toolBox.addItem(self.indexPage, _fromUtf8(""))
        self.imageLabel = QtGui.QLabel(self.verticalSplitter)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.imageLabel.sizePolicy().hasHeightForWidth())
        self.imageLabel.setSizePolicy(sizePolicy)
        self.imageLabel.setStyleSheet(_fromUtf8("background: pink"))
        self.imageLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.imageLabel.setWordWrap(False)
        self.imageLabel.setObjectName(_fromUtf8("imageLabel"))
        self.logTextEdit = QtGui.QTextEdit(self.horizontalSplitter)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.logTextEdit.sizePolicy().hasHeightForWidth())
        self.logTextEdit.setSizePolicy(sizePolicy)
        self.logTextEdit.setBaseSize(QtCore.QSize(0, 0))
        self.logTextEdit.setReadOnly(True)
        self.logTextEdit.setObjectName(_fromUtf8("logTextEdit"))
        self.gridLayout.addWidget(self.horizontalSplitter, 1, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        self.toolBox.setCurrentIndex(1)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow", None))
        self.performFFCButton.setText(_translate("MainWindow", "FFC", None))
        self.label_10.setText(_translate("MainWindow", "Speed:", None))
        self.sunSpeedSpinBox.setToolTip(_translate("MainWindow", "sun speed", None))
        self.toolBox.setItemText(self.toolBox.indexOf(self.sunPage), _translate("MainWindow", "Sun", None))
        self.groupBox.setTitle(_translate("MainWindow", "Color", None))
        self.label_5.setText(_translate("MainWindow", "Red:", None))
        self.label_6.setText(_translate("MainWindow", "Green:", None))
        self.allOffPushButton.setText(_translate("MainWindow", "0%", None))
        self.allHalfPushButton.setText(_translate("MainWindow", "50%", None))
        self.allFullPushButton.setText(_translate("MainWindow", "100%", None))
        self.label_7.setText(_translate("MainWindow", "Blue:", None))
        self.adjustmentGroupBox.setTitle(_translate("MainWindow", "Adjustment", None))
        self.label_8.setText(_translate("MainWindow", "r:", None))
        self.label_9.setText(_translate("MainWindow", "s:", None))
        self.toolBox.setItemText(self.toolBox.indexOf(self.colorAdjustPage), _translate("MainWindow", "Color Adjust", None))
        self.label_2.setText(_translate("MainWindow", "Increment:", None))
        self.label_3.setText(_translate("MainWindow", "Index:", None))
        self.pixelIndexStepSpinBox.setToolTip(_translate("MainWindow", "pixel index increment", None))
        self.pixelIndexSpinBox.setToolTip(_translate("MainWindow", "pixel index", None))
        self.label.setText(_translate("MainWindow", "S Index:", None))
        self.label_4.setText(_translate("MainWindow", "R Index:", None))
        self.sIndexSpinBox.setToolTip(_translate("MainWindow", "s index", None))
        self.rIndexSpinBox.setToolTip(_translate("MainWindow", "r index", None))
        self.toolBox.setItemText(self.toolBox.indexOf(self.indexPage), _translate("MainWindow", "Index", None))
        self.imageLabel.setText(_translate("MainWindow", "TextLabel", None))

