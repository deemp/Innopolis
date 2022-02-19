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
T = 15

dtheta_desired = 0.

p_gain = 0.2
d_gain = 0.03
i_max = 0.1
i_gain = 0.4


full_turn = 2 * np.pi
N = T * frequency
# thetas = np.linspace(0, full_turn, N)
# thetas_desired = thetas + theta_home
pendulum.set_torque(0.001)
theta_home = pendulum.state["angle"]
print(theta_home)
theta_desired = np.pi + theta_home

thetas = np.zeros(N)
ts = np.zeros(N)
errs = np.zeros(N)
m = 0.14
g = 9.81
l = 0.1

try:
    # find the global time before intering control loop
    initial_time = perf_counter()
    i_term = 0
    last_execution = 0
    control = 0
    i = 0
    while True:
        time = perf_counter() - initial_time  # get actual time in secs
        state = pendulum.state
        theta = state["angle"]
        dtheta = state["speed"]
        torque = state["torque"]
        current = state["current"]

        # if time >= T:
            # Kms[i] = np.abs(m * g * l * np.sin(theta) / current)
            # Kms[i] = torque / current

            # theta_reals[i] = theta
            # break
        if (time - last_execution) >= sampling_time: 
            i += 1
            if i >= N:
                break
            
            thetas[i] = theta - theta_home
            ts[i] = time

            last_execution = time
            # YOUR CONTROLLER GOES HERE

            position_error = theta - theta_desired
            velocity_error = dtheta - dtheta_desired
            i_term = i_term + position_error * sampling_time
            i_term = min(i_max, i_term)

            control = -(
                p_gain * position_error + d_gain * velocity_error 
                + i_gain * i_term
            )

            errs[i] = position_error

            print(f"Motor angle data: {theta:.5f}, position error: {position_error:.5f}", end="    \r", flush=True)

        pendulum.set_torque(control)
    
except KeyboardInterrupt:
    print("Disabled by interrupt")
except Exception as e:
    print(f"\n!!!! EXCEPTION !!!!\n {type(e)} \n {e} \n!!!! EXCEPTION !!!!\n")
finally:
    for i in range(100):
        pendulum.set_torque(0)

import matplotlib.pyplot as plt


fig, ax = plt.subplots(1,2,figsize=(14,6))

theta_desireds = np.full(N, theta_desired - theta_home)

ax[0].plot(ts, thetas, label=f'$\\theta$')
ax[0].plot(ts, theta_desireds, label=f'desired $\\theta^\\ast$ = {theta_desired-theta_home:.4f}', linestyle="-.", linewidth=3)
ax[0].grid()
ax[0].set_title(f"Pendulum's angle with inverted gravity under manual disturbances")
ax[0].set_xlabel("time [s]")
ax[0].set_ylabel("$\\theta$ [rad]")
ax[0].legend()

ax[1].plot(ts, errs, label=f'$\\hat{{\\theta}}$')
ax[1].grid()
ax[1].set_title(f"Pendulum's position error")
ax[1].set_xlabel("time [s]")
ax[1].set_ylabel("$\\hat{{\\theta}}$ [rad]")
ax[1].legend()

plt.tight_layout()
plt.savefig("./images/task3.2.png")
plt.show()

