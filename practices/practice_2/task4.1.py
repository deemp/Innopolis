from libs.can import CANSocket
from libs.myactuator import MyActuator
from time import perf_counter, sleep

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
    for _ in range(100):
        motor.set_current(0)


# total working time
T = 6
N = T * frequency

gains = [20, 50, 100]
g_n = len(gains)

angles = np.zeros((g_n, N))
velocities = np.zeros((g_n, N))
times = np.zeros(N)

angle_initial = 2
velocity_desired = 0

period = 2

import random

left, right = -4*np.pi, 4*np.pi 

def get_desired(period):
    previous = 0
    angle_desired = np.zeros(N)
    angle = random.uniform(left, right)
    for i in range(N):
        current = int(i / (period * frequency))
        if current > previous:
            angle = random.uniform(left, right)
            previous = current

        angle_desired[i] = angle

    return angle_desired


angle_desired = get_desired(period)

motor.set_current(0)

try:
    # for k in range(0):
    for k in range(g_n):
        i = 0
        last_execution = 0
        control = 0
        # find the global time before entering control loop
        initial_time = perf_counter()
        # motor.set_zero()
        initial_angle = motor.state["angle"] + angle_initial
        while True:
            time = perf_counter() - initial_time  # get actual time in secs
            # /////////////////////////
            # Get and parse motor state
            # /////////////////////////
            state = motor.state
            theta = state["angle"] - initial_angle
            dtheta = state["speed"]
            current = state["current"]

            # ///////////////////////////////////////////
            # Update the control only on specific timings
            # ///////////////////////////////////////////

            # P-control
            if (time - last_execution) >= sampling_time:
                if i < N:
                    angles[k, i] = theta
                    velocities[k, i] = dtheta
                    times[i] = time
                else:
                    break

                control = -gains[k] * (theta - angle_desired[i])

                i += 1

                last_execution = time
                # YOUR CONTROLLER GOES HERE

            motor.set_current(control)

        stop_motor(motor)
        sleep(1)

except KeyboardInterrupt:

    stop_motor(motor)
    print("Disabled by interrupt")

motor = None

import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2, figsize=(18, 9))


ax[0].set_xlabel("t [s]")
ax[0].set_ylabel("$\\theta$ [$rad$]")

ax[1].set_xlabel("t [s]")
ax[1].set_ylabel("$\\dot{\\theta}$ [$\\frac{rad}{s}$]")

bound = 1
last_n = 10


def get_settling_time(x, ts):
    x_f = x[-1]
    is_setttled = np.all(np.abs(x[-last_n:] - x_f) <= bound)
    if not is_setttled:
        return -1
    diffs = (np.abs(x - x_f) <= bound)[::-1]
    j = 0
    n = len(diffs)
    for i in range(len(diffs)):
        if diffs[i] == False:
            j = i
            break
    return ts[n - j - 1]


def get_overshoot(x):
    final_value = np.abs(x[-1])
    max_diff = np.max(x[x - final_value >= 0])
    return max_diff


def get_steady_state_error(x, x_d):
    return np.abs(x[-1] - x_d)


def get_characteristics(x, x_d, ts):
    settling_time = get_settling_time(x, ts)
    if settling_time == -1:
        return [-100., -100., -100.]
    overshoot = get_overshoot(x)
    steady_state_error = get_steady_state_error(x, x_d)
    return [settling_time, overshoot, steady_state_error]


def add_plot(ax, x, x_d, ts, gain, for_angle):
    ax.plot(
        ts,
        x,
        label=f"gain: {gain}",
    )


for i in range(g_n):
    add_plot(
        ax=ax[0],
        x=angles[i],
        x_d=angle_desired[i],
        ts=times,
        gain=gains[i],
        for_angle=True,
    )
    add_plot(
        ax=ax[1],
        x=velocities[i],
        x_d=velocity_desired,
        ts=times,
        gain=gains[i],
        for_angle=False,
    )


ax[0].plot(times, angle_desired, label="$\\theta_{{desired}}$")

ax[0].legend()
ax[1].legend()
fig.suptitle(f"control loop frequency = {frequency} Hz", fontsize="13")
fig.tight_layout(pad=3.0)

plt.savefig("./plots/4.1.png")
plt.show()
