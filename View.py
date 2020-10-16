import pyqtgraph as qtg
from PyQt5.QtWidgets import QPushButton, QMainWindow, QWidget, QTabWidget, QVBoxLayout, \
    QLineEdit, QHBoxLayout, QFormLayout, QCheckBox

from Controller import Controller
from Model import MyFunction, Grid


class App(QMainWindow):

    def __init__(self):
        super().__init__()
        self.title = 'Computational Practicum app'
        self.left = 0
        self.top = 0
        self.width = 800
        self.height = 600
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        self.table_widget = Tabs(self)
        self.setCentralWidget(self.table_widget)

        self.show()


class Tabs(QWidget):

    def __init__(self, parent):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)

        # create tabs
        self.tabs = QTabWidget()
        self.produce_tab1()
        self.produce_tab2()

        self.layout.addWidget(self.tabs)
        self.setLayout(self.layout)

        self.user_input()

    def produce_tab1(self):
        self.tab1 = QWidget()
        self.tabs.addTab(self.tab1, "Solutions and LTE graphs")

        panel = QHBoxLayout()
        self.tab1.setLayout(panel)

        # Column 1
        column1 = QFormLayout()

        # --- layout for input

        self.x0, self.X = QLineEdit(str(MyFunction.x0)), QLineEdit(str(MyFunction.X))
        self.y0, self.N = QLineEdit(str(MyFunction.y0)), QLineEdit(str(MyFunction.N))
        column1.addRow("x0", self.x0)
        column1.addRow("X", self.X)
        column1.addRow("y0", self.y0)
        column1.addRow("N", self.N)

        # --- layout for checkboxes for enabling/disabling graphs

        self.es_check1, self.em_check1 = QCheckBox(), QCheckBox()
        self.iem_check1, self.rk_check1 = QCheckBox(), QCheckBox()
        column1.addRow("Exact solution", self.es_check1)
        column1.addRow("Euler method", self.em_check1)
        column1.addRow("Improved Euler method", self.iem_check1)
        column1.addRow("Runge-Kutta method", self.rk_check1)

        # --- layout for "Calculate" button
        calc_button = QPushButton('Calculate!')
        calc_button.clicked.connect(self.user_input)
        column1.addWidget(calc_button)

        # Adding column 1 to panel
        inp_wd = QWidget()
        inp_wd.setLayout(column1)

        panel.addWidget(inp_wd)

        # Column 2

        # --- layout for graphs
        graphs = QVBoxLayout()

        self.g1 = qtg.PlotWidget()
        graphs.addWidget(self.g1)

        self.g2 = qtg.PlotWidget()
        graphs.addWidget(self.g2)

        graphs_wd = QWidget()
        graphs_wd.setLayout(graphs)

        # Adding column 2 to panel
        panel.addWidget(graphs_wd)

    def produce_tab2(self):
        self.tab2 = QWidget()
        self.tabs.addTab(self.tab2, "GTE graphs")

        panel = QHBoxLayout()
        self.tab2.setLayout(panel)

        # Column 1
        column1 = QFormLayout()

        # --- layout for input

        self.n0, self.N0 = QLineEdit('10'), QLineEdit('30')
        column1.addRow("n0", self.n0)
        column1.addRow("N", self.N0)

        # --- layout for checkboxes for enabling/disabling graphs

        self.em_check2, self.iem_check2, self.rk_check2 = QCheckBox(), QCheckBox(), QCheckBox()
        column1.addRow("Euler method", self.em_check2)
        column1.addRow("Improved Euler method", self.iem_check2)
        column1.addRow("Runge-Kutta method", self.rk_check2)

        # --- layout for "Calculate" button
        calc_button = QPushButton('Calculate!')
        calc_button.clicked.connect(self.user_input)
        column1.addWidget(calc_button)

        # Adding column 1 to panel
        inp_wd = QWidget()
        inp_wd.setLayout(column1)

        panel.addWidget(inp_wd)

        # Column 2

        # --- layout for graphs
        graphs = QVBoxLayout()

        self.g = qtg.PlotWidget()
        graphs.addWidget(self.g)

        graphs_wd = QWidget()
        graphs_wd.setLayout(graphs)

        # Adding column 2 to panel
        panel.addWidget(graphs_wd)

    def user_input(self):

        # sending user input to Controller
        tab1, tab2 = Controller.update_tabs(self.x0, self.X, self.y0, self.N, self.n0, self.N0)
        self.plot_tab1(tab1)
        self.plot_tab2(tab2)

    def plot_tab1(self, tab1):
        #   plotting tab 1
        xs = tab1['xs'].tolist()

        self.renew_graph(self.g1)
        self.renew_graph(self.g2)

        if self.es_check1.isChecked():
            self.plot(self.g1, xs, tab1['exact'], 'Exact solution', 'y')
        if self.em_check1.isChecked():
            self.plot(self.g1, xs, tab1['em_approx'], 'Euler Method', 'r')
            self.plot(self.g2, xs, tab1['em_lte'], 'Euler Method', 'r')
        if self.iem_check1.isChecked():
            self.plot(self.g1, xs, tab1['iem_approx'], 'Improved Euler Method', 'b')
            self.plot(self.g2, xs, tab1['iem_lte'], 'Improved Euler Method', 'b')
        if self.rk_check1.isChecked():
            self.plot(self.g1, xs, tab1['rk_approx'], 'Runge-Kutta Method', 'g')
            self.plot(self.g2, xs, tab1['rk_lte'], 'Runge-Kutta Method', 'g')

    def plot_tab2(self, tab2):
        #   plotting tab 2
        self.renew_graph(self.g)

        ns = tab2['ns']

        if self.em_check2.isChecked():
            self.plot(self.g, ns, tab2['em_gte'], 'Euler Method', 'r')
        if self.iem_check2.isChecked():
            self.plot(self.g, ns, tab2['iem_gte'], 'Improved Euler Method', 'b')
        if self.rk_check2.isChecked():
            self.plot(self.g, ns, tab2['rk_gte'], 'Runge-Kutta Method', 'g')

    def renew_graph(self, graph):
        graph.clear()
        graph.showGrid(x=True, y=True)

    def plot(self, graph_wd, x, y, plotname, color):
        pen = qtg.mkPen(color=color)
        graph_wd.addLegend()
        graph_wd.plot(x, y, name=plotname, pen=pen, symbol='+', symbolSize=10, symbolBrush=(color))
