import sys

from PyQt5.QtWidgets import QApplication

from Code.View import App

app = QApplication(sys.argv)
ex = App()
sys.exit(app.exec_())
