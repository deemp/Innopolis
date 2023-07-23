#%%
from libs.can import CANSocket
from libs.myactuator import MyActuator
from time import perf_counter

import numpy as np

# import getpass
# password = getpass.getpass()

# the serial port of device
# you may find one by examing /dev/ folder,
# this is usually devices ttyACM
# os.system(f"sudo slcand -o -c -s8 /dev/serial/by-id/usb-Protofusion_Labs_CANable_8c005eb_https\:__github.com_normaldotcom_cantact-fw.git_001D00335734570920343135-if00 can0")

serial_device = "ttyACM1"

# Initiate the can bus socket
can_bus = CANSocket(serial_port=serial_device)

# Initiate motor
motor = MyActuator(can_bus=can_bus)

# Set the control loop timings
frequency = 500
sampling_time = 1 / frequency


def stop_motor(motor):
    for i in range(100):
        motor.set_current(0)


# total working time
T = 3
N = T * frequency + 100

gains = [20, 50, 100]
g_n = len(gains)

angles = np.zeros((g_n, N))
velocities = np.zeros((g_n, N))
times = np.zeros(N)

angle_desired = -3
velocity_desired = 0

try:
    for k in range(g_n):
        val = input(f"Start iteration {k}? [y/n]")
        i = 0
        last_execution = 0
        control = 0
        motor.set_angle(0)
        # find the global time before intering control loop
        initial_time = perf_counter()
        while True:
            time = perf_counter() - initial_time  # get actual time in secs
            if time >= T:
                break
            # /////////////////////////
            # Get and parse motor state
            # /////////////////////////
            state = motor.state
            theta = state["angle"]
            dtheta = state["speed"]
            current = state["current"]

            # ///////////////////////////////////////////
            # Update the control only on specific timings
            # ///////////////////////////////////////////

            # P-control
            if (time - last_execution) >= sampling_time:
                angles[k, i] = theta
                velocities[k, i] = dtheta
                times[i] = time
                i += 1

                last_execution = time
                # YOUR CONTROLLER GOES HERE
                control = -gains[k] * (theta - angle_desired)

            motor.set_current(control)

            # print(f'Motor angle data: {theta}', end='    \r', flush=True)

        stop_motor(motor)

except KeyboardInterrupt:
    stop_motor(motor)
    print("Disabled by interrupt")

#%%
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2, figsize=(20, 10))

print(angles[0][:10])

last = i - 10
angles = angles[:, :last]
velocities = velocities[:, :last]
times = times[:last]

ax[0].set_xlabel("t [s]")
ax[0].set_ylabel("$\\theta$ [$rad$]")

ax[1].set_xlabel("t [s]")
ax[1].set_ylabel("$\\dot{\\theta}$ [$\\frac{rad}{s}$]")

bound = 0.05
last_n = 10

def get_settling_time(x, ts):
    theta_f = x[-1]
    is_setttled = np.all(np.abs(x[-last_n:] - theta_f) <= bound)
    if not is_setttled:
        return -1
    diffs = (np.abs(x - theta_f) <= bound)[::-1]
    j = 0
    n = len(diffs)
    for i in range(len(diffs)):
        if diffs[i] == False:
            j = i
            break
    return ts[n - j - 1]

def get_overshoot(x):
    final_value = np.abs(x[-1])
    max_diff = np.max(np.abs(x - final_value))
    return max_diff / final_value

def get_steady_state_error(x, x_d):
    return x[-1] - x_d

def get_characteristics(x, x_d, ts):
    settling_time = get_settling_time(x, ts)
    if ts == -1:
        return "not reached", "undefined", "undefined"
    overshoot = get_overshoot(x)
    steady_state_error = get_steady_state_error(x, x_d)
    return settling_time, overshoot, steady_state_error
    
def add_plot(ax, x, x_d, ts, gain):
    settling_time, overshoot, steady_state_error = get_characteristics(x, x_d, ts)
    ax.plot(ts, x, 
        label=f"gain: {gains}, settling time: {settling_time}, overshoot: {overshoot}, steady state error: {steady_state_error}",
    )

for i in range(g_n):
    add_plot(ax=ax[0], x=angles[i], x_d=angle_desired, ts=times, gain=gains[i])
    add_plot(ax=ax[1], x=velocities[i], x_d=velocity_desired, ts=times, gain=gains[i])

ax[0].legend()
ax[1].legend()
fig.suptitle(f"control loop frequency = {frequency} Hz", fontsize="13")
fig.tight_layout(pad=3.0)

plt.savefig("./plots/2.1.png")
plt.show()
# %%
