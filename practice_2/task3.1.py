from libs.can import CANSocket
from libs.myactuator import MyActuator
from time import perf_counter

import numpy as np
import matplotlib.pyplot as plt

serial_device = "ttyACM1"

# Initiate the can bus socket
can_bus = CANSocket(serial_port=serial_device)

# Initiate motor
motor = MyActuator(can_bus=can_bus)

# Set the control loop timings
frequencies = np.array([100, 250, 500])
f_n = len(frequencies)
sampling_time = 1 / frequencies

def stop_motor(motor):
    for _ in range(100):
        motor.set_current(0)


# total working time
T = 2
Ns = T * frequencies

# for each p_gain, we have a plot for d_gains
p_gain = 120
d_gain = 25

angles = [np.zeros(N) for N in Ns]
velocities = [np.zeros(N) for N in Ns]
times = [np.zeros(N) for N in Ns]

angle_initial = 2
angle_desired = 0

velocity_desired = 0

motor.set_current(0)

try:
    for k in range(f_n):
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
            if (time - last_execution) >= sampling_time[k]:
                if i < Ns[k]:
                    angles[k][i] = theta
                    velocities[k][i] = dtheta
                    times[k][i] = time
                else:
                    break

                i += 1

                last_execution = time
                # YOUR CONTROLLER GOES HERE
                control = -p_gain * (theta - angle_desired) - d_gain * (
                    dtheta - velocity_desired
                )

            motor.set_current(control)

        stop_motor(motor)

except KeyboardInterrupt:

    stop_motor(motor)
    print("Disabled by interrupt")

motor = None

angle_bound = 0.05
velocity_bound = 1
last_n = 10


def get_settling_time(x, ts, for_angle=True):
    x_f = x[-1]
    bound = angle_bound if for_angle else velocity_bound
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
    over = x[x - final_value >= 0]
    max_diff = 0. if over.size == 0 else np.max(over)
    return max_diff

def get_steady_state_error(x, x_d):
    return np.abs(x[-1] - x_d)


def get_characteristics(x, x_d, ts, for_angle=True):
    settling_time = get_settling_time(x, ts, for_angle)
    if settling_time == -1:
        return [-100, -100, -100]
    overshoot = get_overshoot(x)
    steady_state_error = get_steady_state_error(x, x_d)
    return [settling_time, overshoot, steady_state_error]


def add_plot(ax, x, x_d, ts, frequency, for_angle=True):
    settling_time, overshoot, steady_state_error = get_characteristics(x, x_d, ts, for_angle)
    overshoot_unit = "rad" if for_angle else "$\\frac{rad}{sec}$"
    ax.plot(
        ts,
        x,
        label=f"control loop frequency = {frequency} Hz, settling time: {settling_time:.2f}, overshoot: {overshoot:.2f} {overshoot_unit}, steady state error: {steady_state_error:.2f}",
    )


fig, ax = plt.subplots(1, 2, figsize=(18, 9))

ax0, ax1 = ax[0], ax[1]
ax0.set_xlabel("t [s]")
ax0.set_ylabel("$\\theta$ [$rad$]")

ax1.set_xlabel("t [s]")
ax1.set_ylabel("$\\dot{\\theta}$ [$\\frac{rad}{s}$]")

for i in range(f_n):
    add_plot(
        ax=ax0,
        x=angles[i],
        x_d=angle_desired,
        ts=times[i],
        for_angle=True,
        frequency=frequencies[i]
    )
    add_plot(
        ax=ax1,
        x=velocities[i],
        x_d=velocity_desired,
        ts=times[i],
        for_angle=False,
        frequency=frequencies[i]
    )

des = np.zeros(Ns[0])
ax0.plot(times[0], des, label="$\\theta_{{desired}}$")
ax0.legend()
ax1.legend()

fig.suptitle(
    f"P-gain: {p_gain:.2f}, D-gain: {d_gain:.2f}, $\\theta_{{desired}}$ = {angle_desired}",
    fontsize="13",
)

fig.tight_layout(pad=3.0)

plt.savefig("./plots/2.2.png")
plt.show()
