import pyqtgraph as qtg

from PyQt5.QtWidgets import QPushButton, QMainWindow, QWidget, QTabWidget, QVBoxLayout, \
    QLineEdit, QHBoxLayout, QFormLayout, QCheckBox

from Code.Model import Exact

from Code.Controller import Controller


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


class Plot(qtg.PlotWidget):
    def __init__(self, title, label_left, label_bottom):
        super().__init__()
        self.showGrid(x=True, y=True)

        self.setTitle(title, color="w", size="14pt")

        styles = {"color": "white", "font-size": "15px"}
        self.setLabel("left", label_left, **styles)
        self.setLabel("bottom", label_bottom, **styles)

    def show_plot(self, x, y, plotname, color):
        pen = qtg.mkPen(color=color)
        self.addLegend()
        self.plot(x, y, name=plotname, pen=pen, symbol='+', symbolSize=10, symbolBrush=color)


class Tab(QWidget):
    method_names = ["Exact solution", "Euler", "Improved Euler", "Runge-Kutta"]
    colors = ["y", "r", "b", "g"]
    checkbox_wds = ["es_check", "em_check", "iem_check", "rk_check"]

    def __init__(self, parent):
        super().__init__()

        panel = QHBoxLayout()
        self.setLayout(panel)

        # Column 1
        column1 = QFormLayout()

        # --- layout for input

        for input_label in self.input_labels:
            setattr(self, input_label, QLineEdit(str(getattr(Exact, input_label))))
            column1.addRow(input_label, getattr(self, input_label))

        # --- layout for checkboxes for enabling/disabling graphs

        for checkbox_wd, method in zip(self.checkbox_wds, self.method_names):
            setattr(self, checkbox_wd, QCheckBox())
            checkbox = getattr(self, checkbox_wd)
            checkbox.toggle()
            column1.addRow(method, checkbox)

        # --- layout for "Calculate" button

        calc_button = QPushButton('Calculate!')
        calc_button.clicked.connect(parent.user_input)
        column1.addWidget(calc_button)

        # Adding column 1 to panel
        column1_wd = QWidget()
        column1_wd.setLayout(column1)

        panel.addWidget(column1_wd)

        # Column 2

        # --- layout for graphs

        column2 = QVBoxLayout()

        for plot_wd, title, label_left, label_bottom in self.plots:
            setattr(self, plot_wd, Plot(title, label_left, label_bottom))
            column2.addWidget(getattr(self, plot_wd))

        column2_wd = QWidget()
        column2_wd.setLayout(column2)

        # Adding column 2 to panel
        panel.addWidget(column2_wd)

    def draw_graphs(self, plot_wd, tab_data, xs, ys, checkbox_wds, methods, colors):
        plot_wd.clear()
        for y, checkbox, method, color in zip(ys, checkbox_wds, methods, colors):
            checkbox = getattr(self, checkbox)
            if checkbox.isChecked():
                plot_wd.show_plot(xs, tab_data[y], method, color)

    def update_plots(self, tab_data):
        pass


class Tab1(Tab):
    name = "Solutions and LTE graphs"

    input_labels = ["x0", "X", "y0", "N"]
    ys = ["exact", "em_approx", "iem_approx", "rk_approx"]
    lte_cols = ["em_lte", "iem_lte", "rk_lte"]
    plots = [['g1', 'Plots for different methods and exact solution', 'y', 'x'],
             ['g2', 'Plots of LTE-s for different methods', 'LTE', 'x']]

    def __init__(self, parent):
        super().__init__(parent)

    def update_plots(self, tab_data):
        self.draw_graphs(self.g1, tab_data, tab_data['xs'], self.ys, self.checkbox_wds, self.method_names, self.colors)
        self.draw_graphs(self.g2, tab_data, tab_data['xs'], self.lte_cols, self.checkbox_wds[1:],
                         self.method_names[1:], self.colors[1:])


class Tab2(Tab):
    name = "GTE graphs"

    method_names = Tab.method_names[1:]
    colors = Tab.colors[1:]
    checkbox_wds = Tab.checkbox_wds[1:]

    input_labels = ["n0", "N0"]
    ys = ["em_gte", "iem_gte", "rk_gte"]
    plots = [['g', 'Plot of max GTE-s for different grid steps', 'max GTE', 'n']]

    def __init__(self, parent):
        super().__init__(parent)

    def update_plots(self, tab_data):
        self.draw_graphs(self.g, tab_data, tab_data['ns'], self.ys, self.checkbox_wds, self.method_names, self.colors)


class Tabs(QWidget):

    def __init__(self, parent):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)

        # create tabs
        self.tabs = QTabWidget()

        self.tab1 = Tab1(self)
        self.tabs.addTab(self.tab1, self.tab1.name)

        self.tab2 = Tab2(self)
        self.tabs.addTab(self.tab2, self.tab2.name)

        self.layout.addWidget(self.tabs)
        self.setLayout(self.layout)

        self.user_input()

    def user_input(self):
        tab1_data, tab2_data = \
            Controller.get_model_state(self.tab1.x0, self.tab1.X, self.tab1.y0, self.tab1.N, self.tab2.n0, self.tab2.N0)
        self.tab1.update_plots(tab1_data)
        self.tab2.update_plots(tab2_data)
