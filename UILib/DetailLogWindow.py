# Copyright © 2021, Yuzu Liu. All Rights Reserved.

# Copyright Notice
# Yuzu Liu copyrights this specification.
# No part of this specification may be reproduced in any form or means,
# without the prior written consent of Yuzu Liu.


# Disclaimer
# This specification is preliminary and is subject to change at any time without notice.
# Yuzu Liu assumes no responsibility for any errors contained herein.


import os

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMainWindow
from PyQt5.uic import loadUi

from UILib.Database import KEYS


class DetailLogWindow(QMainWindow):
    def __init__(self, data, parent=None):
        super(DetailLogWindow, self).__init__(parent)
        loadUi("./data/UI/DetailLog.ui", self)
        self.data = data
        self.face_image.setScaledContents(True)
        # self.license_image.setScaledContents(True)
        self.ticket_button.clicked.connect(self.ticket)
        self.initData()

    def ticket(self):
        self.destroy()

    def initData(self):

        self.cam_id.setText(str(self.data['CARID']))
        self.behavior.setText(self.data['CARCOLOR'])

        if self.data['CARIMAGE'] is not None:
            self.face_image.setPixmap(self.data['CARIMAGE'])
        if self.data['LICENSEIMAGE'] is not None:
            self.license_image.setPixmap(self.data['LICENSEIMAGE'])

        self.face_name.setText(self.data['FACE'])
        self.preson_id.setText(self.data['LICENSENUMBER'])
        self.rule.setText(self.data['RULENAME'])

        self.close_button.clicked.connect(self.close)
        self.delete_button.clicked.connect(self.deleteRecord)

    def close(self):
        self.destroy()

    def deleteRecord(self):
        qm = QtWidgets.QMessageBox
        prompt = qm.question(self, '', "确定要删除吗?", qm.Yes | qm.No)
        if prompt == qm.Yes:
            self.destroy()
        else:
            pass
