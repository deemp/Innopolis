import pyqtgraph as qtg
from PyQt5.QtWidgets import QPushButton, QMainWindow, QWidget, QTabWidget, QVBoxLayout, \
    QLineEdit, QHBoxLayout, QFormLayout, QCheckBox

from Code.Controller import Controller
from Code.Model import MyFunction


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


# class Plot(qtg.PlotWidget):
#     def __init__(self, title, label_left, label_bottom):
#         super().__init__()
#         self.showGrid(x=True, y=True)
#
#         self.setTitle(title, color="w", size="14pt")
#
#         styles = {"color": "white", "font-size": "15px"}
#         self.setLabel("left", label_left, **styles)
#         self.setLabel("bottom", label_bottom, **styles)
#
#     def show_plot(self, x, y, plotname, color):
#         pen = qtg.mkPen(color=color)
#         self.addLegend()
#         self.plot(x, y, name=plotname, pen=pen, symbol='+', symbolSize=10, symbolBrush=color)
#
#
# class Tab(QWidget):
#     methods = ["Exact solution", "Euler", "Improved Euler", "Runge-Kutta"]
#
#     def __init__(self):
#         super().__init__()
#
#         panel = QHBoxLayout()
#         self.setLayout(panel)
#
#         # Column 1
#         column1 = QFormLayout()
#
#         # --- layout for input
#
#         for input_label in input_labels:
#             setattr(self, input_label, QLineEdit(str(getattr(MyFunction, input_label))))
#             column1.addRow(input_label, getattr(self, input_label))
#
#         # --- layout for checkboxes for enabling/disabling graphs
#
#         for checkbox_wd, method in zip(self.checkbox_names, self.methods):
#             setattr(self, checkbox_wd, QCheckBox())
#             checkbox = getattr(self, checkbox_wd)
#             checkbox.toggle()
#             column1.addRow(method, checkbox)
#
#         # --- layout for "Calculate" button
#
#         calc_button = QPushButton('Calculate!')
#         calc_button.clicked.connect(self.user_input())
#         column1.addWidget(calc_button)
#
#         # Adding column 1 to panel
#         column1_wd = QWidget()
#         column1_wd.setLayout(column1)
#
#         panel.addWidget(column1_wd)
#
#         # Column 2
#
#         # --- layout for graphs
#
#         column2 = QVBoxLayout()
#
#         for plot in plots:
#             column2.addWidget(plot)
#
#         column2_wd = QWidget()
#         column2_wd.setLayout(column2)
#
#         # Adding column 2 to panel
#         panel.addWidget(column2_wd)
#
#     def draw_graphs(self, plot_wd, tab_data, xs, ys, checkboxes, methods, colors):
#         for y, checkbox, method, color in zip(ys, checkboxes, methods, colors):
#             checkbox = getattr(self, checkbox)
#             if checkbox.isChecked():
#                 plot_wd.show_plot(xs, tab_data[y], method, color)
#
#     def update_plots(self, tab_data):
#         pass
#
#
# class Tab1(Tab):
#     name = "tab1"
#
#     title = "Solutions and LTE graphs"
#     input_labels = ["x0", "X", "y0", "N"]
#     checkbox_names = ["es_check1", "em_check1", "iem_check1", "rk_check1"]
#     ys = ["exact", "em_approx", "iem_approx", "rk_approx"]
#     lte_cols = ["em_lte", "iem_lte", "rk_lte"]
#     colors = ["y", "r", "b", "g"]
#
#     g1 = Plot('Plots for different methods and exact solution', 'y', 'x')
#     g2 = Plot('Plots of LTE-s for different methods', 'LTE', 'x')
#     plots = [g1, g2]
#
#     def __init__(self):
#         super().__init__()
#
#     def update_plots(self, tab_data):
#         self.draw_graphs(self.g1, tab_data, tab_data['xs'], ys, checkboxes, self.methods, colors)
#         self.draw_graphs(self.g2, tab_data, tab_data['xs'], lte_cols, checkboxes[1:], self.methods[1:], colors[1:])
#
#
# class Tab2(Tab):
#     name = "tab2"
#
#     methods = Tab.methods[1:]
#
#     title = "GTE graphs"
#     input_labels = ["n0", "N0"]
#     checkbox_names = ["em_check2", "iem_check2", "rk_check2"]
#     ys = ["em_approx", "iem_approx", "rk_approx"]
#     colors = ["r", "b", "g"]
#
#     g = Plot('Plot of max GTE-s for different grid steps', 'max GTE', 'n')
#
#     def __init__(self):
#         super().__init__()
#
#     def update_plots(self, tab_data):
#         self.draw_graphs(self.g, tab_data, tab_data['ns'], ys, checkbox_names, self.methods, colors)


class Tabs(QWidget):

    def __init__(self, parent):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)

        # create tabs
        self.tabs = QTabWidget()

        self.methods = ["Exact solution", "Euler", "Improved Euler", "Runge-Kutta"]

        # tab1 = Tab1(tabs)
        # self.tabs.addTab(tab1, tab1.name)

        # tab2 = Tab2()
        # self.tabs.addTab(tab2, tab2.name)

        tab_name = 'tab1'
        tab_title = "Solutions and LTE graphs"
        input_labels = ["x0", "X", "y0", "N"]
        checkboxes = ["es_check1", "em_check1", "iem_check1", "rk_check1"]
        methods = self.methods
        plots = [['g1', 'Plots for different methods and exact solution', 'y', 'x'],
                 ['g2', 'Plots of LTE-s for different methods', 'LTE', 'x']]

        self.produce_tab(tab_name, tab_title, input_labels, checkboxes, methods, plots)

        tab_name = 'tab2'
        tab_title = "GTE graphs"
        input_labels = ["n0", "N0"]
        checkboxes = ["em_check2", "iem_check2", "rk_check2"]
        methods = self.methods[1:]
        plots = [['g', 'Plot of max GTE-s for different grid steps', 'max GTE', 'n']]

        self.produce_tab(tab_name, tab_title, input_labels, checkboxes, methods, plots)

        self.layout.addWidget(self.tabs)
        self.setLayout(self.layout)

        self.user_input()

    def produce_tab(self, tab_name, tab_title, input_labels, checkboxes, methods, plots):
        setattr(self, tab_name, QWidget())
        tab = getattr(self, tab_name)
        self.tabs.addTab(tab, tab_title)

        panel = QHBoxLayout()
        tab.setLayout(panel)

        # Column 1
        column1 = QFormLayout()

        # --- layout for input

        for i in input_labels:
            setattr(self, i, QLineEdit(str(getattr(MyFunction, i))))
            column1.addRow(i, getattr(self, i))

        # --- layout for checkboxes for enabling/disabling graphs

        for i, j in zip(checkboxes, methods):
            setattr(self, i, QCheckBox())
            check_box = getattr(self, i)
            check_box.toggle()
            column1.addRow(j, check_box)

        # --- layout for "Calculate" button

        calc_button = QPushButton('Calculate!')
        calc_button.clicked.connect(self.user_input)
        column1.addWidget(calc_button)

        # Adding column 1 to panel
        column1_wd = QWidget()
        column1_wd.setLayout(column1)

        panel.addWidget(column1_wd)

        # Column 2

        # --- layout for graphs

        column2 = QVBoxLayout()

        for plot_widget, plot_name, plot_label_left, plot_label_bottom in plots:
            setattr(self, plot_widget, qtg.PlotWidget())
            plot = getattr(self, plot_widget)
            self.init_graph(plot, plot_name, plot_label_left, plot_label_bottom)
            column2.addWidget(plot)

        column2_wd = QWidget()
        column2_wd.setLayout(column2)

        # Adding column 2 to panel
        panel.addWidget(column2_wd)

    def user_input(self):
        # sending user input to Controller

        tab1_data, tab2_data = Controller.update_model(self.x0, self.X, self.y0, self.N, self.n0, self.N0)
        # self.tab1.update_plots(tab1_data)
        self.plot_tab1(tab1_data)
        self.plot_tab2(tab2_data)

    def plot_tab1(self, tab1):
        #   plotting tab 1
        xs = tab1['xs'].tolist()

        self.renew_graph(self.g1)
        self.renew_graph(self.g2)

        ys = ["exact", "em_approx", "iem_approx", "rk_approx"]
        lte_cols = ["em_lte", "iem_lte", "rk_lte"]
        colors = ["y", "r", "b", "g"]
        checkboxes = ["es_check1", "em_check1", "iem_check1", "rk_check1"]

        self.draw_graphs(self.g1, xs, tab1, ys, checkboxes, self.methods, colors)
        self.draw_graphs(self.g2, xs, tab1, lte_cols, checkboxes[1:], self.methods[1:], colors[1:])

    def plot_tab2(self, tab2):
        #   plotting tab 2
        self.renew_graph(self.g)

        ns = tab2['ns']

        ys = ["em_gte", "iem_gte", "rk_gte"]
        checkboxes = ["em_check2", "iem_check2", "rk_check2"]
        colors = ["r", "b", "g"]
        self.draw_graphs(self.g, ns, tab2, ys, checkboxes, self.methods[1:], colors)

    def draw_graphs(self, graph_wd, xs, tab, ys, checkboxes, methods, colors):
        for y, checkbox, method, color in zip(ys, checkboxes, methods, colors):
            checkbox = getattr(self, checkbox)
            if checkbox.isChecked():
                self.plot(graph_wd, xs, tab[y], method, color)

    def init_graph(self, graph_wd, title, label_left, label_bottom):
        graph_wd.showGrid(x=True, y=True)

        graph_wd.setTitle(title, color="w", size="14pt")

        styles = {"color": "white", "font-size": "15px"}
        graph_wd.setLabel("left", label_left, **styles)
        graph_wd.setLabel("bottom", label_bottom, **styles)

    def renew_graph(self, graph_wd):
        graph_wd.clear()

    def plot(self, graph_wd, x, y, plotname, color):
        pen = qtg.mkPen(color=color)
        graph_wd.addLegend()
        graph_wd.plot(x, y, name=plotname, pen=pen, symbol='+', symbolSize=10, symbolBrush=(color))
