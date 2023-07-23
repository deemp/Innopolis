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
p_gain = 0.2
d_gain = 0.02
i_max = 0.2
i_max = 0.2
# i_gain = 0.2

seconds = 3
N = seconds * frequency

ps = [0.1, 0.3]
ds = [0.01, 0.03]

thetas = np.zeros((len(ps), len(ds), N))
try:
    # find the global time before intering control loop
    # initial_time = perf_counter()
    for i, p_gain in enumerate(ps):
        for j, d_gain in enumerate(ds):
            iterations = 0
            initial_time = perf_counter()
            # i_term = 0.
            last_execution = 0
            control = 0
            pendulum.set_torque(0.0001)
            theta_home = pendulum.state["angle"]
            while True:
                time = perf_counter() - initial_time  # get actual time in secs
                # print("ok")
                # break


                # ///////////////////////////////////////////
                # Update the control only on specific timings
                # ///////////////////////////////////////////
                if (time - last_execution) >= sampling_time:
                    # /////////////////////////
                    # Get and parse motor state
                    # /////////////////////////
                    state = pendulum.state
                    theta = state["angle"] - theta_home
                    dtheta = state["speed"]
                    # torque = state["torque"]
                    iterations += 1
                    if iterations >= N:
                        break
                    
                    thetas[i,j,iterations] = theta

                    last_execution = time
                    # YOUR CONTROLLER GOES HERE

                    position_error = theta - theta_desired
                    velocity_error = dtheta - dtheta_desired
                    
                    # grav = 0.14 * 9.81 * 0.1 * np.sin(theta)
                    control = -(
                        p_gain * position_error + d_gain * velocity_error 
                        # + grav
                        # + i_gain * i_term
                    )
                    # control = 0.2

                    print(f"Motor angle data: {theta:.5f}, position error: {position_error:.5f}", end="    \r", flush=True)

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

for i, p in enumerate(ps):
    for j,d in enumerate(ds):
        ax.plot(ts, thetas[i,j], label=f'p_gain: {p}, d_gain: {d}, SSE: {thetas[i,j,-1]-theta_desired:.4f}')

ax.grid()
ax.plot(ts, np.full(N, theta_desired), label=f'$\\theta_{{desired}}$', linestyle="-.", linewidth=3)
ax.set_title(f"$\\theta_{{desired}}: {theta_desired:.2f}$")
ax.legend()
plt.savefig("./images/task1.1.png")
plt.show()

