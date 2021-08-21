# Copyright © 2020, Yingping Liang. All Rights Reserved.

# Copyright Notice
# Yingping Liang copyrights this specification.
# No part of this specification may be reproduced in any form or means,
# without the prior written consent of Yingping Liang.


# Disclaimer
# This specification is preliminary and is subject to change at any time without notice.
# Yingping Liang assumes no responsibility for any errors contained herein.


import cv2
import time
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import QTimer
import imutils
import pyqtgraph as pg
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtWidgets import QMainWindow, QStatusBar, QListWidget, QAction, qApp, QMenu, QVBoxLayout, QFileDialog
from PyQt5.uic import loadUi
from UILib.ViolationItem import ViolationItem
from processor.MainProcessor import MainProcessor
import csv


class MainWindowLayOut(QMainWindow):
    def __init__(self, opt):
        super(MainWindowLayOut, self).__init__()
        loadUi("./data/UI/MainWindow.ui", self)
        self.face_count = 0
        self.max_log_num = 5
        self.face_size = 140
        self.feed = None
        self.vs = None
        self.opt = opt
        self.updateCamInfo()

        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("欢迎")

        self.clear_button.clicked.connect(self.clear)
        self.refresh_button.clicked.connect(self.refresh)

        font = QFont()
        font.setPointSize(12)

        self.log_tabwidget.clear()
        self.log_tabwidget.setFont(font)
        self.face_list = QListWidget(self)
        self.illegal_list = QListWidget(self)
        self.log_tabwidget.addTab(self.face_list, "行为记录")

        self.initParams()
        self.initMenu(font)
        self.initGraph()

    def initParams(self):

        menubar = self.menuBar()
        settingsMenu = menubar.addMenu('&设置')
        self.add_setting_menu(settingsMenu)
        # self.file_btn.clicked.connect(
        #     lambda: self.getFile(self.file_edit))  # 文件选择槽函数绑定
        self.update_func_status()
        self.headpose_checkbtn.clicked.connect(self.update_func_status)
        self.smoke_checkbtn.clicked.connect(self.update_func_status)
        self.phone_checkbtn.clicked.connect(self.update_func_status)
        self.facestatus_checkbtn.clicked.connect(self.update_func_status)
        self.write_checkbtn.clicked.connect(self.update_func_status)
        self.eat_checkbtn.clicked.connect(self.update_func_status)
        self.mask_checkbtn_2.clicked.connect(self.update_func_status)

        self.graph_values = {
            'Blinks': [],
            'Yawning': [],
            'Nod': []
        }

        self.values_max_num = 30

    def update_func_status(self):

        self.func_status = {
            'headpose': self.headpose_checkbtn.isChecked(),
            'smoke': self.smoke_checkbtn.isChecked(),
            'phone': self.phone_checkbtn.isChecked(),
            'facestatus': self.facestatus_checkbtn.isChecked(),
            'write': self.write_checkbtn.isChecked(),
            'eat-drink': self.eat_checkbtn.isChecked(),
            'mask-detector': self.mask_checkbtn_2.isChecked()
        }

        print('-[INFO] Update function.')
        for k, v in self.func_status.items():
            print('   -[Func] {}: {}'.format(k, v))

        print('\n-[INFO] Reset tracker.')
        self.processor.processor.build_config()

    def add_setting_menu(self, settingsMenu):

        speed_menu = QMenu("更改模型", self)
        settingsMenu.addMenu(speed_menu)

        act = QAction('YOLOv3+DarkNet53', self)
        act.setStatusTip('YOLOv3+DarkNet53')
        act.triggered.connect(self.add_speed_limit)
        speed_menu.addAction(act)

        act = QAction('YOLOv3+MobileNetV3', self)
        act.setStatusTip('YOLOv3+MobileNetV3')
        act.triggered.connect(self.reduce_speed_limit)
        speed_menu.addAction(act)

        direct_menu = QMenu("更改阈值", self)
        settingsMenu.addMenu(direct_menu)

        act = QAction('阈值+0.05', self)
        act.setStatusTip('阈值+0.05')
        act.triggered.connect(self.add_speed_limit)
        direct_menu.addAction(act)

        act = QAction('阈值-0.05', self)
        act.setStatusTip('阈值-0.05')
        act.triggered.connect(self.reduce_speed_limit)
        direct_menu.addAction(act)

    def add_speed_limit(self):

        self.max_speed += 10
        self.max_speed_lbl.setText(str(self.max_speed) + " km/hr")

    def reduce_speed_limit(self):

        if self.max_speed > 0:
            self.max_speed -= 10
            self.max_speed_lbl.setText(str(self.max_speed) + " km/hr")

    def set_right_rear(self):

        self.right_direct_lbl.setText('正向↑')
        self.right_direction = 'Rear'

    def set_right_front(self):

        self.right_direct_lbl.setText('反向↓')
        self.right_direction = 'Front'

    def initMenu(self, font):

        menubar = self.menuBar()
        menubar.setFont(font)
        fileMenu = menubar.addMenu('&文件')

        addRec = QMenu("添加记录", self)

        act = QAction('添加摄像头', self)
        act.setStatusTip('Add Camera Manually')
        # act.triggered.connect(self.addCamera)
        addRec.addAction(act)

        act = QAction('添加记录', self)
        act.setStatusTip('Add Car Manually')
        # act.triggered.connect(self.addCar)
        addRec.addAction(act)

        fileMenu.addMenu(addRec)

        act = QAction('&存档', self)
        act.setStatusTip('Show Archived Records')
        fileMenu.addAction(act)

        # Add Exit
        fileMenu.addSeparator()
        act = QAction('&退出', self)
        act.setStatusTip('退出应用')
        act.triggered.connect(qApp.quit)
        fileMenu.addAction(act)

        self.headpose_checkbtn.setChecked(True)
        self.stuff_checkbtn_2.setChecked(True)

    def updateCamInfo(self, feed=0):
        self.feed = feed
        self.processor = MainProcessor(
            model_type=self.opt.model_type, tracker_type=self.opt.tracker)
        self.vs = cv2.VideoCapture(self.feed)

    def getFile(self, lineEdit):
        file_path = QFileDialog.getOpenFileName()[0]
        lineEdit.setText(file_path)  # 获取文件路径
        self.updateCamInfo(file_path)

    def updateLog(self, data=[]):
        for row in data:
            if self.face_count < self.max_log_num:
                self.face_count += 1
            else:
                self.face_count -= 1
                self.face_list.clear()
            listWidget = ViolationItem()
            listWidget.setData(row)
            listWidgetItem = QtWidgets.QListWidgetItem(self.face_list)
            listWidgetItem.setSizeHint(listWidget.sizeHint())
            self.face_list.addItem(listWidgetItem)
            self.face_list.setItemWidget(listWidgetItem, listWidget)

    def update_status_log(self, data=[]):
        # [{'CARID': 0, 'CARIMAGE': <PyQt5.QtGui.QPixmap object at 0x0000021EA4710D68>,
        # 'CARCOLOR': '2021-08-15 15:08:59.572038', 'FACE': 'Driver 1', 'LICENSEIMAGE': None,
        # 'LICENSENUMBER': 'Driver 1', 'LOCATION': '未知', 'RULENAME': '未佩戴口罩'}]
        for row in data:
            listWidget = ViolationItem()
            listWidget.setData(row)
            listWidgetItem = QtWidgets.QListWidgetItem(self.face_list)
            listWidgetItem.setSizeHint(listWidget.sizeHint())
            self.face_list.addItem(listWidgetItem)
            self.face_list.setItemWidget(listWidgetItem, listWidget)

    @QtCore.pyqtSlot()
    def refresh(self):
        self.updateCamInfo()

    @QtCore.pyqtSlot()
    def clear(self):
        qm = QtWidgets.QMessageBox
        prompt = qm.question(self, '', "是否重置所有记录?", qm.Yes | qm.No)
        if prompt == qm.Yes:
            self.face_list.clear()
        else:
            pass

    def initGraph(self):

        verticalLayout = QVBoxLayout(self.hr_preview1)
        win = pg.GraphicsLayoutWidget(self.hr_preview1)
        verticalLayout.addWidget(win)
        p = win.addPlot(title="动态波形图")
        p.showGrid(x=True, y=True)
        p.setLabel(axis="left", text="幅值 / V")
        p.setLabel(axis="bottom", text="时间 / s")
        p.setTitle("闭眼频率")
        p.addLegend()
        self.curve11 = p.plot(pen="g", name="y1")
        self.curve12 = p.plot(pen="r", name="y2")

        verticalLayout = QVBoxLayout(self.hr_preview2)
        win = pg.GraphicsLayoutWidget(self.hr_preview2)
        verticalLayout.addWidget(win)
        p = win.addPlot(title="动态波形图")
        p.showGrid(x=True, y=True)
        p.setLabel(axis="left", text="幅值 / V")
        p.setLabel(axis="bottom", text="时间 / s")
        p.setTitle("哈欠频率")
        p.addLegend()
        self.curve21 = p.plot(pen="g", name="y1")
        self.curve22 = p.plot(pen="r", name="y2")

        verticalLayout = QVBoxLayout(self.hr_preview3)
        win = pg.GraphicsLayoutWidget(self.hr_preview3)
        verticalLayout.addWidget(win)
        p = win.addPlot(title="动态波形图")
        p.showGrid(x=True, y=True)
        p.setLabel(axis="left", text="幅值 / V")
        p.setLabel(axis="bottom", text="时间 / s")
        p.setTitle("点头频率")
        p.addLegend()
        self.curve31 = p.plot(pen="g", name="y1")
        self.curve32 = p.plot(pen="r", name="y2")

    def toQImage(self, img, height=480):
        if height is not None:
            img = imutils.resize(img, height=height)
        qformat = QImage.Format_Indexed8
        if len(img.shape) == 3:
            if img.shape[2] == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888

        outImg = QImage(
            img.tobytes(), img.shape[1], img.shape[0], img.strides[0], qformat)
        outImg = outImg.rgbSwapped()

        return outImg

    def load_data(self, sp):
        for i in range(1, 11):  # 模拟主程序加载过程
            # time.sleep(0.5)                   # 加载数据
            sp.showMessage(
                "加载... {0}%".format(
                    i * 10),
                QtCore.Qt.AlignHCenter | QtCore.Qt.AlignBottom,
                QtCore.Qt.black)
            QtWidgets.qApp.processEvents()  # 允许主进程处理事件
