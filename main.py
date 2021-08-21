# Copyright © 2021, Yuzu Liu. All Rights Reserved.

# Copyright Notice
# Yuzu Liu copyrights this specification.
# No part of this specification may be reproduced in any form or means,
# without the prior written consent of Yuzu Liu.


# Disclaimer
# This specification is preliminary and is subject to change at any time without notice.
# Yuzu Liu assumes no responsibility for any errors contained herein.

import sys
import argparse
import qdarkstyle
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtWidgets import QApplication
from UILib.MainWindow import MainWindow


def main(opt):
    '''
    启动PyQt5程序，打开GUI界面
    '''
    app = QApplication(sys.argv)
    splash = QtWidgets.QSplashScreen(QtGui.QPixmap("web/logo.png"))
    splash.showMessage("加载... 0%", QtCore.Qt.AlignHCenter, QtCore.Qt.black)
    splash.show()                           # 显示启动界面
    QtWidgets.qApp.processEvents()          # 处理主进程事件
    main_window = MainWindow(opt)
    main_window.load_data(splash)                # 加载数据2
    main_window.showFullScreen()
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    main_window.show()
    sys.exit(app.exec_())




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str,
                        default='torch', help='model framework.')
    parser.add_argument('--tracker', type=str,
                        default='deep_sort', help='tracker framework.')
    opt = parser.parse_args()

    main(opt)
