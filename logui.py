import sys
import torch
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QFont, QIcon
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QDesktopWidget, QHBoxLayout, QFormLayout, \
    QPushButton, QLineEdit, QMainWindow


class LoginForm(QMainWindow):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        """
        初始化UI
        :return:
        """
        self.setObjectName("loginWindow")
        self.setStyleSheet('#loginWindow{background-color:white}')
        self.setFixedSize(650, 300)
        self.setWindowTitle("登录")
        self.setWindowIcon(QIcon('web/logo.png'))

        # 登录表单内容部分
        login_widget = QWidget(self)
        login_widget.move(0, 0)
        login_widget.setGeometry(0, 0, 650, 260)

        hbox = QHBoxLayout()
        # 添加左侧logo
        logolb = QLabel(self)
        logopix = QPixmap("web/logo.png")
        logolb.setPixmap(logopix)
        logolb.setAlignment(Qt.AlignCenter)
        hbox.addWidget(logolb, 1)
        # 添加右侧表单
        fmlayout = QFormLayout()
        lbl_workerid = QLabel("用户名")
        lbl_workerid.setFont(QFont("Microsoft YaHei"))
        led_workerid = QLineEdit()
        led_workerid.setFixedWidth(270)
        led_workerid.setFixedHeight(38)

        lbl_pwd = QLabel("密码")
        lbl_pwd.setFont(QFont("Microsoft YaHei"))
        led_pwd = QLineEdit()
        led_pwd.setEchoMode(QLineEdit.Password)
        led_pwd.setFixedWidth(270)
        led_pwd.setFixedHeight(38)

        self.btn_login = QPushButton("登录")
        self.btn_login.setFixedWidth(270)
        self.btn_login.setFixedHeight(40)
        self.btn_login.setFont(QFont("Microsoft YaHei"))
        self.btn_login.setObjectName("login_btn")
        self.btn_login.setStyleSheet(
            "#login_btn{background-color:#2c7adf;color:#fff;border:none;border-radius:4px;}")
        self.btn_login.clicked.connect(self.close)

        fmlayout.addRow(lbl_workerid, led_workerid)
        fmlayout.addRow(lbl_pwd, led_pwd)
        fmlayout.addWidget(self.btn_login)
        hbox.setAlignment(Qt.AlignCenter)
        # 调整间距
        fmlayout.setHorizontalSpacing(20)
        fmlayout.setVerticalSpacing(12)

        hbox.addLayout(fmlayout, 2)

        login_widget.setLayout(hbox)

        self.center()
        self.show()

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = LoginForm()
    sys.exit(app.exec_())
