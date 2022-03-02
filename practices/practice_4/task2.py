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
T = 5

dtheta_desired = 0.

p_gain = 0.2
d_gain = 0.03
i_max = 0.1
i_gain = 0.4


full_turn = 2 * np.pi
N = 10

theta_desired = full_turn/N

Kms = np.zeros(N)
theta_reals = np.zeros(N)
m = 0.14
g = 9.81
l = 0.1

pendulum.set_torque(0.001)
theta_home = pendulum.state["angle"]

try:
    # find the global time before intering control loop
    # initial_time = perf_counter()
    for i in range(N):
        initial_time = perf_counter()
        i_term = 0
        last_execution = 0
        control = 0
        theta_desired_current = theta_desired * i
        while True:
            time = perf_counter() - initial_time  # get actual time in secs


            if (time - last_execution) >= sampling_time:                
                state = pendulum.state
                theta = state["angle"] - theta_home
                dtheta = state["speed"]

                if time >= T:
                    current = state["current"]
                    theta_reals[i] = theta
                    Kms[i] = np.abs(m * g * l * np.sin(theta) / current)
                    break

                last_execution = time

                position_error = theta - theta_desired_current
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

fig, ax = plt.subplots(1,1,figsize=(10,7))

m = np.mean(Kms)

ax.plot(theta_reals, Kms, label=f'$K_{{m}}$')
ax.grid()
ax.set_title(f"Motor constant $K_{{m}}$ calculated at different angles $\\theta$\nMean={m:.5f}")
ax.set_xlabel("$\\theta, [rad]$")
ax.set_ylabel("$K_m$, $[\\frac{{N \\cdot m}}{{A}}]$")
plt.legend()
plt.savefig("./images/task2.png")
plt.show()

