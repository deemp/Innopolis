import sys

from PyQt5.QtWidgets import QApplication

from View import App

app = QApplication(sys.argv)
ex = App()
sys.exit(app.exec_())
