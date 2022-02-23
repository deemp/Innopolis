from matplotlib.pyplot import contour
from libs.can import CANSocket
from libs.myactuator import MyActuator
from time import perf_counter
import numpy as np

# the serial port of device
# you may find one by examing /dev/ folder,
# this is usually devices ttyACM
serial_device = "ttyACM0"

# Initiate the can bus socket
can_bus = CANSocket(serial_port=serial_device)

# Initiate motor
pendulum = MyActuator(can_bus=can_bus)
pendulum.torque_constant = 1e-3

# ////////////////////////////////
# USE THIS TO SET MOTOR ZERO
# the motor must be rebooted after
# ////////////////////////////////
# pendulum.set_zero(persistant=True)


# Set the control loop timings
frequency = 500
sampling_time = 1 / frequency
# length of a single iteration in seconds
T = 4
T1 = 2

dtheta_desired = 0.

p_gain = 0.3
d_gain = 0.03
i_max = 0.1
i_gain = 0.4


full_turn = 2 * np.pi
N = T * frequency
pendulum.set_torque(0.001)
theta_home = pendulum.state["angle"]
theta_desireds = np.array([1/4, 1/3, 1/2, 1]) * np.pi

k = 0.95
m = 0.14
g = 9.81
l = 0.1

# PD+/PID
modes = 2

is_mass_attached = True

if not is_mass_attached:
    m = 0.04

thetas = np.zeros((modes,len(theta_desireds),N))

# prompt in console to attach/remove mass
# measure thetas for angles with mass
# go home after each measurement
# repeat without mass

def get_control(position_error, velocity_error, term=0):
    return -(p_gain * position_error + d_gain * velocity_error + term)


def go_home_pid(theta_home=theta_home):
    initial_time = perf_counter()
    i_term = 0
    last_execution = 0
    control = 0
    theta_desired = 0
    d_theta_desired = 0
    while True:
        time = perf_counter() - initial_time  # get actual time in secs

        if (time - last_execution) >= sampling_time: 
            
            if time >= T1:
                break

            state = pendulum.state
            theta = state["angle"] - theta_home
            dtheta = state["speed"]

            last_execution = time
            # YOUR CONTROLLER GOES HERE

            position_error = theta - theta_desired
            velocity_error = dtheta - dtheta_desired
            i_term += position_error * sampling_time
            i_term = min(i_max, i_term)
            
            control = get_control(position_error, velocity_error, i_gain * i_term)

        pendulum.set_torque(control)
    
    for j in range(100):
        pendulum.set_torque(0)

txt = "attach mass to" if is_mass_attached else "detach mass from"
input(f"\r\nPlease, {txt} the pendulum and home it vertically down. Then press 'Enter'\r\n")

labels = ["PD+", "PID"]

try:
    for mode in range(modes):
        print(f"\r\nStarting experiments for {labels[mode]} controller\r\n")
        for i, theta_desired in enumerate(theta_desireds):
            print("\r\n\r\nHoming...\r")
            go_home_pid(theta_home)
            # find the global time before intering control loop
            initial_time = perf_counter()
            i_term = 0
            last_execution = 0
            control = 0
            j = 0
            while True:
                time = perf_counter() - initial_time  # get actual time in secs

                if (time - last_execution) >= sampling_time: 
                    j += 1
                    if j >= N:
                        break
                    
                    state = pendulum.state
                    theta = state["angle"] - theta_home
                    dtheta = state["speed"]

                    thetas[mode,i,j] = theta

                    last_execution = time
                    # YOUR CONTROLLER GOES HERE

                    position_error = theta - theta_desired
                    velocity_error = dtheta - dtheta_desired
                    i_term += position_error * sampling_time
                    i_term = min(i_max, i_term)

                    grav = -m * g * l * np.sin(theta) * k
                    
                    term = i_gain * i_term if mode == 1 else grav
                    control = get_control(position_error, velocity_error, term)

                    print(f"Motor angle data: {theta:.5f}, position error: {position_error:.5f}", end="    \r", flush=True)

                pendulum.set_torque(control)
            
            go_home_pid(theta_home)

except KeyboardInterrupt:
    print("Disabled by interrupt")
except Exception as e:
    print(f"\n!!!! EXCEPTION !!!!\n {type(e)} \n {e} \n!!!! EXCEPTION !!!!\n")
finally:
    for j in range(100):
        pendulum.set_torque(0)

ts = np.linspace(0,T,N)

import matplotlib.pyplot as plt


fig, ax = plt.subplots(1,2,figsize=(20,9))

for mode in range(modes):
    for i,theta in enumerate(theta_desireds):
        ax[mode]
        ax[mode].plot(ts, thetas[mode,i], label=f'$\\theta_{{desired}} = {theta:.3f}$, SSE = {thetas[mode,i,-1] - theta:.3f}')
        ax[mode].plot(ts, np.full((N),theta), label=f'$\\theta_{{desired}}$ = {theta:.3f}', linestyle="-.", linewidth=3)
        ax[mode].set_title(f"{labels[mode]} controller")
        ax[mode].grid()
        ax[mode].set_xlabel("$t$ [seconds]")
        ax[mode].set_ylabel("$\\theta$ [rad]")
        ax[mode].legend(loc="best")

txt = "attached" if is_mass_attached else "detached"
fig.suptitle(f"Comparison of PD+ and PID controllers with mass {txt}\np_gain={p_gain:.3f}, d_gain={d_gain:.3f}", fontsize=16)

plt.tight_layout()
name = "+" if is_mass_attached else "-"
plt.savefig(f"./images/task3.1{name}mass.png")
plt.show()