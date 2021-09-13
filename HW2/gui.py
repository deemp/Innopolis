from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QVBoxLayout

from vispy_canvas import canvas, visualUpdates, informationUpdates, timer


# Substitute this 
# class Ui_Dialog 
# for new generated interface


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(693, 495)
        self.widgetVispyCanvas = QtWidgets.QWidget(Dialog)
        self.widgetVispyCanvas.setGeometry(QtCore.QRect(240, 50, 410, 410))
        self.widgetVispyCanvas.setObjectName("widgetVispyCanvas")
        self.labelSimulation = QtWidgets.QLabel(Dialog)
        self.labelSimulation.setGeometry(QtCore.QRect(405, 20, 81, 16))
        self.labelSimulation.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.labelSimulation.setFrameShadow(QtWidgets.QFrame.Raised)
        self.labelSimulation.setTextFormat(QtCore.Qt.RichText)
        self.labelSimulation.setAlignment(QtCore.Qt.AlignBottom|QtCore.Qt.AlignHCenter)
        self.labelSimulation.setTextInteractionFlags(QtCore.Qt.NoTextInteraction)
        self.labelSimulation.setObjectName("labelSimulation")
        self.layoutWidget = QtWidgets.QWidget(Dialog)
        self.layoutWidget.setGeometry(QtCore.QRect(20, 20, 170, 161))
        self.layoutWidget.setObjectName("layoutWidget")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.layoutWidget)
        self.verticalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.labelDisplayedElements = QtWidgets.QLabel(self.layoutWidget)
        self.labelDisplayedElements.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.labelDisplayedElements.setFrameShadow(QtWidgets.QFrame.Raised)
        self.labelDisplayedElements.setTextFormat(QtCore.Qt.RichText)
        self.labelDisplayedElements.setAlignment(QtCore.Qt.AlignBottom|QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft)
        self.labelDisplayedElements.setTextInteractionFlags(QtCore.Qt.NoTextInteraction)
        self.labelDisplayedElements.setObjectName("labelDisplayedElements")
        self.verticalLayout_4.addWidget(self.labelDisplayedElements)
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.checkPoints = QtWidgets.QCheckBox(self.layoutWidget)
        self.checkPoints.setChecked(True)
        self.checkPoints.setObjectName("checkPoints")
        self.verticalLayout_3.addWidget(self.checkPoints)
        self.checkVelocities = QtWidgets.QCheckBox(self.layoutWidget)
        self.checkVelocities.setChecked(True)
        self.checkVelocities.setObjectName("checkVelocities")
        self.verticalLayout_3.addWidget(self.checkVelocities)
        self.checkAccelerations = QtWidgets.QCheckBox(self.layoutWidget)
        self.checkAccelerations.setChecked(True)
        self.checkAccelerations.setObjectName("checkAccelerations")
        self.verticalLayout_3.addWidget(self.checkAccelerations)
        self.checkTrajectories = QtWidgets.QCheckBox(self.layoutWidget)
        self.checkTrajectories.setChecked(True)
        self.checkTrajectories.setObjectName("checkTrajectories")
        self.verticalLayout_3.addWidget(self.checkTrajectories)
        self.checkLinks = QtWidgets.QCheckBox(self.layoutWidget)
        self.checkLinks.setChecked(True)
        self.checkLinks.setObjectName("checkLinks")
        self.verticalLayout_3.addWidget(self.checkLinks)
        self.verticalLayout_4.addLayout(self.verticalLayout_3)
        self.labelCurrentInfo = QtWidgets.QLabel(Dialog)
        self.labelCurrentInfo.setGeometry(QtCore.QRect(20, 200, 169, 15))
        self.labelCurrentInfo.setObjectName("labelCurrentInfo")
        self.tabWidget = QtWidgets.QTabWidget(Dialog)
        self.tabWidget.setGeometry(QtCore.QRect(20, 230, 181, 251))
        self.tabWidget.setObjectName("tabWidget")
        self.tabPoints = QtWidgets.QWidget()
        self.tabPoints.setObjectName("tabPoints")
        self.layoutWidget1 = QtWidgets.QWidget(self.tabPoints)
        self.layoutWidget1.setGeometry(QtCore.QRect(0, 0, 171, 220))
        self.layoutWidget1.setObjectName("layoutWidget1")
        self.gridLayout = QtWidgets.QGridLayout(self.layoutWidget1)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.combo_point_information = QtWidgets.QComboBox(self.layoutWidget1)
        self.combo_point_information.setObjectName("combo_point_information")
        self.combo_point_information.addItem("")
        self.combo_point_information.addItem("")
        self.combo_point_information.addItem("")
        self.gridLayout.addWidget(self.combo_point_information, 0, 0, 1, 2)
        self.labelA = QtWidgets.QLabel(self.layoutWidget1)
        self.labelA.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.labelA.setObjectName("labelA")
        self.gridLayout.addWidget(self.labelA, 1, 0, 1, 1)
        self.dataA = QtWidgets.QLineEdit(self.layoutWidget1)
        self.dataA.setReadOnly(True)
        self.dataA.setPlaceholderText("")
        self.dataA.setObjectName("dataA")
        self.gridLayout.addWidget(self.dataA, 1, 1, 1, 1)
        self.labelB = QtWidgets.QLabel(self.layoutWidget1)
        self.labelB.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.labelB.setObjectName("labelB")
        self.gridLayout.addWidget(self.labelB, 2, 0, 1, 1)
        self.dataB = QtWidgets.QLineEdit(self.layoutWidget1)
        self.dataB.setReadOnly(True)
        self.dataB.setPlaceholderText("")
        self.dataB.setObjectName("dataB")
        self.gridLayout.addWidget(self.dataB, 2, 1, 1, 1)
        self.labelC = QtWidgets.QLabel(self.layoutWidget1)
        self.labelC.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.labelC.setObjectName("labelC")
        self.gridLayout.addWidget(self.labelC, 3, 0, 1, 1)
        self.dataC = QtWidgets.QLineEdit(self.layoutWidget1)
        self.dataC.setReadOnly(True)
        self.dataC.setPlaceholderText("")
        self.dataC.setObjectName("dataC")
        self.gridLayout.addWidget(self.dataC, 3, 1, 1, 1)
        self.labelD = QtWidgets.QLabel(self.layoutWidget1)
        self.labelD.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.labelD.setObjectName("labelD")
        self.gridLayout.addWidget(self.labelD, 4, 0, 1, 1)
        self.dataD = QtWidgets.QLineEdit(self.layoutWidget1)
        self.dataD.setReadOnly(True)
        self.dataD.setPlaceholderText("")
        self.dataD.setObjectName("dataD")
        self.gridLayout.addWidget(self.dataD, 4, 1, 1, 1)
        self.labelE = QtWidgets.QLabel(self.layoutWidget1)
        self.labelE.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.labelE.setObjectName("labelE")
        self.gridLayout.addWidget(self.labelE, 5, 0, 1, 1)
        self.dataE = QtWidgets.QLineEdit(self.layoutWidget1)
        self.dataE.setReadOnly(True)
        self.dataE.setPlaceholderText("")
        self.dataE.setObjectName("dataE")
        self.gridLayout.addWidget(self.dataE, 5, 1, 1, 1)
        self.labelF = QtWidgets.QLabel(self.layoutWidget1)
        self.labelF.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.labelF.setObjectName("labelF")
        self.gridLayout.addWidget(self.labelF, 6, 0, 1, 1)
        self.dataF = QtWidgets.QLineEdit(self.layoutWidget1)
        self.dataF.setReadOnly(True)
        self.dataF.setPlaceholderText("")
        self.dataF.setObjectName("dataF")
        self.gridLayout.addWidget(self.dataF, 6, 1, 1, 1)
        self.tabWidget.addTab(self.tabPoints, "")
        self.tabLinks = QtWidgets.QWidget()
        self.tabLinks.setObjectName("tabLinks")
        self.combo_link_information = QtWidgets.QComboBox(self.tabLinks)
        self.combo_link_information.setGeometry(QtCore.QRect(1, 3, 169, 23))
        self.combo_link_information.setObjectName("combo_link_information")
        self.combo_link_information.addItem("")
        self.layoutWidget2 = QtWidgets.QWidget(self.tabLinks)
        self.layoutWidget2.setGeometry(QtCore.QRect(0, 30, 164, 191))
        self.layoutWidget2.setObjectName("layoutWidget2")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.layoutWidget2)
        self.gridLayout_2.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.labelA_2 = QtWidgets.QLabel(self.layoutWidget2)
        self.labelA_2.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.labelA_2.setObjectName("labelA_2")
        self.gridLayout_2.addWidget(self.labelA_2, 0, 0, 1, 1)
        self.labelA_7 = QtWidgets.QLabel(self.layoutWidget2)
        self.labelA_7.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.labelA_7.setObjectName("labelA_7")
        self.gridLayout_2.addWidget(self.labelA_7, 2, 0, 1, 1)
        self.data_O2_B = QtWidgets.QLineEdit(self.layoutWidget2)
        self.data_O2_B.setReadOnly(True)
        self.data_O2_B.setPlaceholderText("")
        self.data_O2_B.setObjectName("data_O2_B")
        self.gridLayout_2.addWidget(self.data_O2_B, 2, 1, 1, 1)
        self.data_A_B = QtWidgets.QLineEdit(self.layoutWidget2)
        self.data_A_B.setReadOnly(True)
        self.data_A_B.setPlaceholderText("")
        self.data_A_B.setObjectName("data_A_B")
        self.gridLayout_2.addWidget(self.data_A_B, 1, 1, 1, 1)
        self.data_O1_A = QtWidgets.QLineEdit(self.layoutWidget2)
        self.data_O1_A.setReadOnly(True)
        self.data_O1_A.setPlaceholderText("")
        self.data_O1_A.setObjectName("data_O1_A")
        self.gridLayout_2.addWidget(self.data_O1_A, 0, 1, 1, 1)
        self.labelA_3 = QtWidgets.QLabel(self.layoutWidget2)
        self.labelA_3.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.labelA_3.setObjectName("labelA_3")
        self.gridLayout_2.addWidget(self.labelA_3, 1, 0, 1, 1)
        self.data_C_D = QtWidgets.QLineEdit(self.layoutWidget2)
        self.data_C_D.setReadOnly(True)
        self.data_C_D.setPlaceholderText("")
        self.data_C_D.setObjectName("data_C_D")
        self.gridLayout_2.addWidget(self.data_C_D, 3, 1, 1, 1)
        self.labelA_5 = QtWidgets.QLabel(self.layoutWidget2)
        self.labelA_5.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.labelA_5.setObjectName("labelA_5")
        self.gridLayout_2.addWidget(self.labelA_5, 4, 0, 1, 1)
        self.data_E_F = QtWidgets.QLineEdit(self.layoutWidget2)
        self.data_E_F.setReadOnly(True)
        self.data_E_F.setPlaceholderText("")
        self.data_E_F.setObjectName("data_E_F")
        self.gridLayout_2.addWidget(self.data_E_F, 4, 1, 1, 1)
        self.labelA_6 = QtWidgets.QLabel(self.layoutWidget2)
        self.labelA_6.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.labelA_6.setObjectName("labelA_6")
        self.gridLayout_2.addWidget(self.labelA_6, 5, 0, 1, 1)
        self.labelA_4 = QtWidgets.QLabel(self.layoutWidget2)
        self.labelA_4.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.labelA_4.setObjectName("labelA_4")
        self.gridLayout_2.addWidget(self.labelA_4, 3, 0, 1, 1)
        self.data_O3_F = QtWidgets.QLineEdit(self.layoutWidget2)
        self.data_O3_F.setReadOnly(True)
        self.data_O3_F.setPlaceholderText("")
        self.data_O3_F.setObjectName("data_O3_F")
        self.gridLayout_2.addWidget(self.data_O3_F, 5, 1, 1, 1)
        self.tabWidget.addTab(self.tabLinks, "")

        self.retranslateUi(Dialog)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.labelSimulation.setText(_translate("Dialog", "Simulation"))
        self.labelDisplayedElements.setText(_translate("Dialog", "Select elements to display"))
        self.checkPoints.setText(_translate("Dialog", "Points"))
        self.checkVelocities.setText(_translate("Dialog", "Velocities"))
        self.checkAccelerations.setText(_translate("Dialog", "Accelerations"))
        self.checkTrajectories.setText(_translate("Dialog", "Trajectories"))
        self.checkLinks.setText(_translate("Dialog", "Links"))
        self.labelCurrentInfo.setText(_translate("Dialog", "Current information"))
        self.combo_point_information.setItemText(0, _translate("Dialog", "Coordinates"))
        self.combo_point_information.setItemText(1, _translate("Dialog", "Velocities"))
        self.combo_point_information.setItemText(2, _translate("Dialog", "Accelerations"))
        self.labelA.setText(_translate("Dialog", "A"))
        self.labelB.setText(_translate("Dialog", "B"))
        self.labelC.setText(_translate("Dialog", "C"))
        self.labelD.setText(_translate("Dialog", "D"))
        self.labelE.setText(_translate("Dialog", "E"))
        self.labelF.setText(_translate("Dialog", "F"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tabPoints), _translate("Dialog", "Points"))
        self.combo_link_information.setItemText(0, _translate("Dialog", "Angular velocities"))
        self.labelA_2.setText(_translate("Dialog", "O1-A"))
        self.labelA_7.setText(_translate("Dialog", "O2-B"))
        self.labelA_3.setText(_translate("Dialog", "A-B"))
        self.labelA_5.setText(_translate("Dialog", "E-F"))
        self.labelA_6.setText(_translate("Dialog", "O3-F"))
        self.labelA_4.setText(_translate("Dialog", "C-D"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tabLinks), _translate("Dialog", "Links"))

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


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    ui = Ui_Dialog()
    Dialog = QtWidgets.QDialog()
    ui.setupUi(Dialog)
    
    f = Functionality(ui)

    Dialog.show()
    sys.exit(app.exec_())