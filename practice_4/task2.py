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

dtheta_desired = 0.

p_gain = 0.1
d_gain = 0.03
i_max = 0.3
i_gain = 0.3


full_turn = 2 * np.pi
N = 10
theta_home = pendulum.state["angle"]
print(theta_home)
thetas = np.linspace(0, full_turn, N)
thetas_desired = thetas + theta_home
print(thetas_desired)

Kms = np.zeros(N)
theta_reals = np.zeros(N)
m = 0.14
g = 9.81
l = 0.1

try:
    # find the global time before intering control loop
    # initial_time = perf_counter()
    for i,theta_desired in enumerate(thetas_desired):
        initial_time = perf_counter()
        i_term = 0
        last_execution = 0
        control = 0
        pendulum.set_zero()
        while True:
            time = perf_counter() - initial_time  # get actual time in secs
            state = pendulum.state
            theta = state["angle"]
            dtheta = state["speed"]
            torque = state["torque"]
            current = state["current"]

            if time >= T:
                Kms[i] = np.abs(m * g * l * np.sin(theta) / current)
                # Kms[i] = torque / current

                theta_reals[i] = theta
                break
            if (time - last_execution) >= sampling_time:                
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


m = np.mean(Kms)

plt.plot(theta_reals, Kms, label=f'$K_{{m}}$')
plt.grid()
plt.title(f"Motor constant $K_{{m}}$ calculated at different angles $\\theta$\nMean={m:.5f}")
plt.xlabel("$\\theta, [rad]$")
plt.ylabel("$K_m$, $[\\frac{{N \\cdot m}}{{A}}]$")
plt.legend()
plt.savefig("./images/task2.png")
plt.show()

