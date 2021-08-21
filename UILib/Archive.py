# Copyright © 2021, Yuzu Liu. All Rights Reserved.

# Copyright Notice
# Yuzu Liu copyrights this specification.
# No part of this specification may be reproduced in any form or means,
# without the prior written consent of Yuzu Liu.


# Disclaimer
# This specification is preliminary and is subject to change at any time without notice.
# Yuzu Liu assumes no responsibility for any errors contained herein.

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMainWindow, QListWidget
from PyQt5.uic import loadUi

from UILib.Database import Database
from UILib.ViolationItem import ViolationItem


class ArchiveWindow(QMainWindow):
    def __init__(self, parent=None):
        super(ArchiveWindow, self).__init__(parent)
        loadUi('./data/UI/Archive.ui', self)

        self.cancel.clicked.connect(self.close)

        self.log_tabwidget.clear()
        self.violation_list = QListWidget(self)
        self.log_tabwidget.addTab(self.violation_list, "Violations")
        self.violation_list.clear()
        rows = Database().get_violations_from_cam(None, cleared=True)
        for row in rows:
            listWidget = ViolationItem()
            listWidget.setData(row)
            listWidgetItem = QtWidgets.QListWidgetItem(self.violation_list)
            listWidgetItem.setSizeHint(listWidget.sizeHint())
            self.violation_list.addItem(listWidgetItem)
            self.violation_list.setItemWidget(listWidgetItem, listWidget)


    def close(self):
        self.destroy(True)
