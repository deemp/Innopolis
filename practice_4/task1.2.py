from libs.can import CANSocket
from libs.myactuator import MyActuator
from time import perf_counter
from math import pi
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

theta_desired = np.pi / 2
dtheta_desired = 0
p_gain = 0.3
d_gain = 0.03
i_max = 0.1
i_gain = 0.4

seconds = 4
N = seconds * frequency

thetas = np.zeros((2,N))

def get_control(position_error, velocity_error, i_term=0):
    return -(p_gain * position_error + d_gain * velocity_error + i_gain * i_term)

try:
    # find the global time before intering control loop
    # initial_time = perf_counter()
    for i in range(2):
        j = 0
        initial_time = perf_counter()
        i_term = 0.
        last_execution = 0
        control = 0
        pendulum.set_torque(0.001)
        theta_home = pendulum.state["angle"]
        while True:
            time = perf_counter() - initial_time  # get actual time in secs
            # print("ok")
            # break

            # /////////////////////////
            # Get and parse motor state
            # /////////////////////////

            # ///////////////////////////////////////////
            # Update the control only on specific timings
            # ///////////////////////////////////////////
            if (time - last_execution) >= sampling_time:
                j += 1
                if j >= N:
                    break
                
                last_execution = time
                
                state = pendulum.state
                theta = state["angle"] - theta_home
                dtheta = state["speed"]
                torque = state["torque"]

                thetas[i,j] = theta

                # YOUR CONTROLLER GOES HERE

                position_error = theta - theta_desired
                velocity_error = dtheta - dtheta_desired
                i_term += position_error * sampling_time
                i_term = min(i_max, i_term)
                
                control = get_control(position_error, velocity_error, 0 if i == 0 else i_term)

                print(f"Motor angle data: {round(theta, 5)}", end="    \r", flush=True)

            pendulum.set_torque(control)
    
except KeyboardInterrupt:
    print("Disabled by interrupt")
except Exception as e:
    print(f"\n!!!! EXCEPTION !!!!\n {type(e)} \n {e} \n!!!! EXCEPTION !!!!\n")
finally:
    for i in range(100):
        pendulum.set_torque(0)

# print ("not ok")

import matplotlib.pyplot as plt


ts = np.linspace(0, time, N)

fig, ax = plt.subplots(1,1,figsize=(10,7))

labels = ["PD", "PID"]

for i,label in enumerate(labels):
    ax.plot(ts, thetas[i], label=f'{label}, SSE: {(thetas[i,-1]-theta_desired):.4f}')

ax.grid()
ax.plot(ts, np.full(N, theta_desired), label=f'$\\theta_{{desired}}$', linestyle="-.", linewidth=3)
ax.set_title(f"$\\theta_{{desired}}: {theta_desired:.2f}$, p_gain: {p_gain}, d_gain: {d_gain}")
ax.legend()
plt.tight_layout()
plt.savefig("./images/task1.2.png")
plt.show()

