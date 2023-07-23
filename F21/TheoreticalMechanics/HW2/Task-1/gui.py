from PyQt6 import QtCore, QtWidgets
from PyQt6.QtWidgets import QVBoxLayout

from vispy_canvas import canvas, visualUpdates, informationUpdates, timer

import sys
sys.path.append('../')

from UIDesign.gui import Ui_Dialog

class PointDataUpdater():
    def __init__(self, ui: Ui_Dialog):
        self.data_objects = {
            "A": ui.dataA,
            "B": ui.dataB,
            "C": ui.dataC,
            "D": ui.dataD,
            "E": ui.dataE,
            "F": ui.dataF
        }

        self.data_setters = {
            "Coordinates": self.set_coordinates,
            "Velocities": self.set_velocities,
            "Accelerations": self.set_accelerations
        }

        self.selected = ui.combo_point_information

        self.timer_connected_to = self.data_setters[self.selected.currentText()]
        timer.connect(self.timer_connected_to)

        self.selected.currentTextChanged.connect(self.select_setter)

    def select_setter(self):
        new_setter = self.data_setters[self.selected.currentText()]
        timer.disconnect(self.timer_connected_to)
        timer.connect(new_setter)
        self.timer_connected_to = new_setter
    
    def format_pair(self, pair):
        x, y = pair
        return f"({x:3.2f}; {y:3.2f})"

    def set_coordinates(self, ev):
        coordinates = informationUpdates.points
        for i in self.data_objects.keys():
            self.data_objects[i].setText(self.format_pair(coordinates[i]))
    
    def set_velocities(self, ev):
        velocities = informationUpdates.velocities
        for i in self.data_objects.keys():
            self.data_objects[i].setText(self.format_pair(velocities[i]))

    def set_accelerations(self, ev):
        accelerations = informationUpdates.accelerations
        for i in self.data_objects.keys():
            self.data_objects[i].setText(self.format_pair(accelerations[i]))

class LinkDataUpdater():
    def __init__(self, ui: Ui_Dialog):
        self.data_objects = {
            "O1_A": ui.data_O1_A,
            "A_B": ui.data_A_B,
            "C_D": ui.data_C_D,
            "E_F": ui.data_E_F,
            "O2_B": ui.data_O2_B,
            "O3_F": ui.data_O3_F
        }
        
        self.data_setters = {
            "Angular velocities": self.set_angular_velocities
        }

        self.selected = ui.combo_link_information

        self.timer_connected_to = self.data_setters[self.selected.currentText()]
        timer.connect(self.timer_connected_to)

        self.selected.currentTextChanged.connect(self.select_setter)

    def select_setter(self):
        new_setter = self.data_setters[self.selected.currentText()]
        timer.disconnect(self.timer_connected_to)
        timer.connect(new_setter)
        self.timer_connected_to = new_setter
    
    def format_number(self, number):
        number
        return f"{number:2.2f}"

    def set_angular_velocities(self, ev):
        angular_velocities = informationUpdates.angular_velocities
        for i in self.data_objects.keys():
            self.data_objects[i].setText(self.format_number(angular_velocities[i]))
    

class ElementsToggler():
    def __init__(self, ui: Ui_Dialog):
        # add vispy canvas

        self.checkBox_objects = {
            "Points": ui.checkPoints,
            "Velocities": ui.checkVelocities,
            "Traces": ui.checkTrajectories,
            "Accelerations": ui.checkAccelerations,
            "Links": ui.checkLinks
        }

        self.toggle_methods = {
            "Points": visualUpdates.toggle_enabled_points,
            "Velocities": visualUpdates.toggle_enabled_velocities,
            "Traces": visualUpdates.toggle_enabled_traces,
            "Accelerations": visualUpdates.toggle_enabled_accelerations,
            "Links": visualUpdates.toggle_enabled_links,
        }

        self.connect_visuals()

    def connect_visuals(self):
        for k in self.checkBox_objects.keys():
            self.checkBox_objects[k].stateChanged.connect(self.toggle_methods[k])
    

class Functionality():
    def __init__(self, ui: Ui_Dialog):
        self.add_canvas(ui)
        self.toggler = ElementsToggler(ui)
        self.point_updater = PointDataUpdater(ui)
        self.link_updater = LinkDataUpdater(ui)

    def add_canvas(self, ui: Ui_Dialog):
        ui.widgetVispyCanvas.setLayout(QVBoxLayout())
        ui.widgetVispyCanvas.layout().addWidget(canvas.native)


def run():
    import sys
    app = QtWidgets.QApplication(sys.argv)
    ui = Ui_Dialog()
    Dialog = QtWidgets.QDialog()
    ui.setupUi(Dialog)
    
    f = Functionality(ui)

    Dialog.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    run()