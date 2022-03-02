import numpy as np

# 35 turns motor
milli = 10**(-3)
centi = 10**(-2)
motor_properties = {
    "J_m, rotor inertia (g*cm^2)": 268 * milli * centi**2, 
    "K_m, torque constant (N*M/A)": 0.23,
    "B_m, nominal friction": 0.3,
    "t_l, torque load (N*M)": 0
}

J = motor_properties['J_m, rotor inertia (g*cm^2)']
K_m = motor_properties['K_m, torque constant (N*M/A)']
B_m = motor_properties['B_m, nominal friction']
tau_l = motor_properties['t_l, torque load (N*M)']

A = np.array([
    [0, 1],
    [0,-B_m/J]
    ])
B = (1/J) * np.array([[0],[K_m]])

