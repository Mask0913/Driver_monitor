# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'MainWindow.ui'
#
# Created by: PyQt5 UI code generator 5.15.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1400, 700)
        MainWindow.setContextMenuPolicy(QtCore.Qt.DefaultContextMenu)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.label = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(32)
        font.setBold(True)
        font.setUnderline(False)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.verticalLayout_6.addWidget(self.label)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.HLayout_0 = QtWidgets.QHBoxLayout()
        self.HLayout_0.setObjectName("HLayout_0")
        self.headpose_checkbtn = QtWidgets.QCheckBox(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        font.setUnderline(False)
        font.setWeight(75)
        self.headpose_checkbtn.setFont(font)
        self.headpose_checkbtn.setObjectName("headpose_checkbtn")
        self.HLayout_0.addWidget(self.headpose_checkbtn)
        self.smoke_checkbtn = QtWidgets.QCheckBox(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        font.setUnderline(False)
        font.setWeight(75)
        self.smoke_checkbtn.setFont(font)
        self.smoke_checkbtn.setObjectName("smoke_checkbtn")
        self.HLayout_0.addWidget(self.smoke_checkbtn)
        self.phone_checkbtn = QtWidgets.QCheckBox(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        font.setUnderline(False)
        font.setWeight(75)
        self.phone_checkbtn.setFont(font)
        self.phone_checkbtn.setObjectName("phone_checkbtn")
        self.HLayout_0.addWidget(self.phone_checkbtn)
        self.verticalLayout_4.addLayout(self.HLayout_0)
        self.HLayout_1 = QtWidgets.QHBoxLayout()
        self.HLayout_1.setObjectName("HLayout_1")
        self.facestatus_checkbtn = QtWidgets.QCheckBox(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        font.setUnderline(False)
        font.setWeight(75)
        self.facestatus_checkbtn.setFont(font)
        self.facestatus_checkbtn.setObjectName("facestatus_checkbtn")
        self.HLayout_1.addWidget(self.facestatus_checkbtn)
        self.write_checkbtn = QtWidgets.QCheckBox(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        font.setUnderline(False)
        font.setWeight(75)
        self.write_checkbtn.setFont(font)
        self.write_checkbtn.setObjectName("write_checkbtn")
        self.HLayout_1.addWidget(self.write_checkbtn)
        self.eat_checkbtn = QtWidgets.QCheckBox(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        font.setUnderline(False)
        font.setWeight(75)
        self.eat_checkbtn.setFont(font)
        self.eat_checkbtn.setObjectName("eat_checkbtn")
        self.HLayout_1.addWidget(self.eat_checkbtn)
        self.verticalLayout_4.addLayout(self.HLayout_1)
        self.HLayout_2 = QtWidgets.QHBoxLayout()
        self.HLayout_2.setObjectName("HLayout_2")
        self.mask_checkbtn_2 = QtWidgets.QCheckBox(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        font.setUnderline(False)
        font.setWeight(75)
        self.mask_checkbtn_2.setFont(font)
        self.mask_checkbtn_2.setObjectName("mask_checkbtn_2")
        self.HLayout_2.addWidget(self.mask_checkbtn_2)
        self.sunglasses_checkbtn_2 = QtWidgets.QCheckBox(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        font.setUnderline(False)
        font.setWeight(75)
        self.sunglasses_checkbtn_2.setFont(font)
        self.sunglasses_checkbtn_2.setObjectName("sunglasses_checkbtn_2")
        self.HLayout_2.addWidget(self.sunglasses_checkbtn_2)
        self.stuff_checkbtn_2 = QtWidgets.QCheckBox(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        font.setUnderline(False)
        font.setWeight(75)
        self.stuff_checkbtn_2.setFont(font)
        self.stuff_checkbtn_2.setObjectName("stuff_checkbtn_2")
        self.HLayout_2.addWidget(self.stuff_checkbtn_2)
        self.verticalLayout_4.addLayout(self.HLayout_2)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.file_edit = QtWidgets.QLineEdit(self.centralwidget)
        self.file_edit.setObjectName("file_edit")
        self.horizontalLayout_8.addWidget(self.file_edit)
        self.file_btn = QtWidgets.QPushButton(self.centralwidget)
        self.file_btn.setObjectName("file_btn")
        self.horizontalLayout_8.addWidget(self.file_btn)
        self.horizontalLayout_8.setStretch(0, 2)
        self.verticalLayout.addLayout(self.horizontalLayout_8)
        self.log_tabwidget = QtWidgets.QTabWidget(self.centralwidget)
        self.log_tabwidget.setMinimumSize(QtCore.QSize(0, 200))
        self.log_tabwidget.setTabShape(QtWidgets.QTabWidget.Rounded)
        self.log_tabwidget.setDocumentMode(False)
        self.log_tabwidget.setMovable(True)
        self.log_tabwidget.setTabBarAutoHide(True)
        self.log_tabwidget.setObjectName("log_tabwidget")
        self.tab_1 = QtWidgets.QWidget()
        self.tab_1.setAccessibleName("")
        self.tab_1.setObjectName("tab_1")
        self.log_tabwidget.addTab(self.tab_1, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.log_tabwidget.addTab(self.tab_2, "")
        self.verticalLayout.addWidget(self.log_tabwidget)
        self.verticalLayout_4.addLayout(self.verticalLayout)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        spacerItem = QtWidgets.QSpacerItem(158, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.search_button = QtWidgets.QPushButton(self.centralwidget)
        self.search_button.setObjectName("search_button")
        self.horizontalLayout.addWidget(self.search_button)
        self.refresh_button = QtWidgets.QPushButton(self.centralwidget)
        self.refresh_button.setObjectName("refresh_button")
        self.horizontalLayout.addWidget(self.refresh_button)
        self.clear_button = QtWidgets.QPushButton(self.centralwidget)
        self.clear_button.setObjectName("clear_button")
        self.horizontalLayout.addWidget(self.clear_button)
        self.verticalLayout_4.addLayout(self.horizontalLayout)
        self.horizontalLayout_2.addLayout(self.verticalLayout_4)
        self.verticalLayout_5 = QtWidgets.QVBoxLayout()
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.live_preview = QtWidgets.QLabel(self.centralwidget)
        self.live_preview.setMinimumSize(QtCore.QSize(300, 0))
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        font.setUnderline(False)
        font.setWeight(75)
        self.live_preview.setFont(font)
        self.live_preview.setAlignment(QtCore.Qt.AlignCenter)
        self.live_preview.setObjectName("live_preview")
        self.verticalLayout_5.addWidget(self.live_preview)
        self.live_preview_2 = QtWidgets.QLabel(self.centralwidget)
        self.live_preview_2.setMinimumSize(QtCore.QSize(300, 0))
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        font.setUnderline(False)
        font.setWeight(75)
        self.live_preview_2.setFont(font)
        self.live_preview_2.setAlignment(QtCore.Qt.AlignCenter)
        self.live_preview_2.setObjectName("live_preview_2")
        self.verticalLayout_5.addWidget(self.live_preview_2)
        self.horizontalLayout_2.addLayout(self.verticalLayout_5)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.verticalLayout_3 = QtWidgets.QHBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.live_preview_keypoints = QtWidgets.QLabel(self.centralwidget)
        self.live_preview_keypoints.setMinimumSize(QtCore.QSize(30, 30))
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        font.setUnderline(False)
        font.setWeight(75)
        self.live_preview_keypoints.setFont(font)
        self.live_preview_keypoints.setAlignment(QtCore.Qt.AlignCenter)
        self.live_preview_keypoints.setObjectName("live_preview_keypoints")
        self.verticalLayout_3.addWidget(self.live_preview_keypoints)
        self.verticalLayout_31 = QtWidgets.QVBoxLayout()
        self.verticalLayout_31.setObjectName("verticalLayout_31")
        self.label_X = QtWidgets.QLabel(self.centralwidget)
        self.label_X.setMinimumSize(QtCore.QSize(300, 0))
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        font.setUnderline(False)
        font.setWeight(75)
        self.label_X.setFont(font)
        self.label_X.setAlignment(QtCore.Qt.AlignCenter)
        self.label_X.setObjectName("label_X")
        self.verticalLayout_31.addWidget(self.label_X)
        self.label_Y = QtWidgets.QLabel(self.centralwidget)
        self.label_Y.setMinimumSize(QtCore.QSize(300, 0))
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        font.setUnderline(False)
        font.setWeight(75)
        self.label_Y.setFont(font)
        self.label_Y.setAlignment(QtCore.Qt.AlignCenter)
        self.label_Y.setObjectName("label_Y")
        self.verticalLayout_31.addWidget(self.label_Y)
        self.label_Z = QtWidgets.QLabel(self.centralwidget)
        self.label_Z.setMinimumSize(QtCore.QSize(300, 0))
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        font.setUnderline(False)
        font.setWeight(75)
        self.label_Z.setFont(font)
        self.label_Z.setAlignment(QtCore.Qt.AlignCenter)
        self.label_Z.setObjectName("label_Z")
        self.verticalLayout_31.addWidget(self.label_Z)
        self.verticalLayout_3.addLayout(self.verticalLayout_31)
        self.verticalLayout_2.addLayout(self.verticalLayout_3)
        self.hr_preview1 = QtWidgets.QLabel(self.centralwidget)
        self.hr_preview1.setMinimumSize(QtCore.QSize(500, 60))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.hr_preview1.setFont(font)
        self.hr_preview1.setAlignment(QtCore.Qt.AlignCenter)
        self.hr_preview1.setObjectName("hr_preview1")
        self.verticalLayout_2.addWidget(self.hr_preview1)
        self.hr_preview2 = QtWidgets.QLabel(self.centralwidget)
        self.hr_preview2.setMinimumSize(QtCore.QSize(500, 60))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.hr_preview2.setFont(font)
        self.hr_preview2.setAlignment(QtCore.Qt.AlignCenter)
        self.hr_preview2.setObjectName("hr_preview2")
        self.verticalLayout_2.addWidget(self.hr_preview2)
        self.hr_preview3 = QtWidgets.QLabel(self.centralwidget)
        self.hr_preview3.setMinimumSize(QtCore.QSize(500, 60))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.hr_preview3.setFont(font)
        self.hr_preview3.setAlignment(QtCore.Qt.AlignCenter)
        self.hr_preview3.setObjectName("hr_preview3")
        self.verticalLayout_2.addWidget(self.hr_preview3)
        self.horizontalLayout_2.addLayout(self.verticalLayout_2)
        self.verticalLayout_6.addLayout(self.horizontalLayout_2)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusBar = QtWidgets.QStatusBar(MainWindow)
        self.statusBar.setObjectName("statusBar")
        MainWindow.setStatusBar(self.statusBar)

        self.retranslateUi(MainWindow)
        self.log_tabwidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "AI??????????????????"))
        self.label.setText(_translate("MainWindow", "AI???????????????????????????????????????"))
        self.headpose_checkbtn.setText(_translate("MainWindow", "????????????"))
        self.smoke_checkbtn.setText(_translate("MainWindow", "????????????"))
        self.phone_checkbtn.setText(_translate("MainWindow", "????????????"))
        self.facestatus_checkbtn.setText(_translate("MainWindow", "????????????"))
        self.write_checkbtn.setText(_translate("MainWindow", "????????????"))
        self.eat_checkbtn.setText(_translate("MainWindow", "????????????"))
        self.mask_checkbtn_2.setText(_translate("MainWindow", "????????????"))
        self.sunglasses_checkbtn_2.setText(_translate("MainWindow", "????????????"))
        self.stuff_checkbtn_2.setText(_translate("MainWindow", "????????????"))
        self.file_btn.setText(_translate("MainWindow", "??????????????????"))
        self.log_tabwidget.setTabText(self.log_tabwidget.indexOf(self.tab_1), _translate("MainWindow", "Tab 1"))
        self.log_tabwidget.setTabText(self.log_tabwidget.indexOf(self.tab_2), _translate("MainWindow", "Tab 2"))
        self.search_button.setText(_translate("MainWindow", "??????"))
        self.refresh_button.setText(_translate("MainWindow", "??????"))
        self.clear_button.setText(_translate("MainWindow", "??????"))
        self.live_preview.setText(_translate("MainWindow", "?????????"))
        self.live_preview_2.setText(_translate("MainWindow", "?????????"))
        self.live_preview_keypoints.setText(_translate("MainWindow", "?????????"))
        self.label_X.setText(_translate("MainWindow", "??????3D??????-X: ??????"))
        self.label_Y.setText(_translate("MainWindow", "??????3D??????-Y: ??????"))
        self.label_Z.setText(_translate("MainWindow", "??????3D??????-Z: ??????"))
        self.hr_preview1.setText(_translate("MainWindow", "?????????"))
        self.hr_preview2.setText(_translate("MainWindow", "?????????"))
        self.hr_preview3.setText(_translate("MainWindow", "?????????"))
