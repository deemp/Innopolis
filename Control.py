import sys

from PyQt5.QtWidgets import QApplication

from View import App

from View import Tabs
from Model import Grid

class Control:
    def __init__(self):
        self.view()

    def view(self):
        app = QApplication(sys.argv)
        ex = App()
        sys.exit(app.exec_())

    def recalculate(self):
        x0 = float(Tabs.x0.text())
        X = float(Tabs.X.text())
        y0 = float(Tabs.y0.text())
        N = int(Tabs.N.text())

        n0 = int(Tabs.n0.text())
        N0 = int(Tabs.N0.text())

        (tab1, tab2) = Grid.generate_data(x0, X, y0, N, n0, N0)
