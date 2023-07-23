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
frequency = 1000
sampling_time = 1 / frequency


def stop_motor(motor):
    for _ in range(100):
        motor.set_current(0)


# total working time
T = 3
N = T * frequency

# gains = [20]
gains = [20,50,70]
g_n = len(gains)


# sin_amplitudes = [2]
sin_amplitudes = [2, 6]

# sin_frequencies = [10]
sin_frequencies = [1, 7]

amp_n = len(sin_amplitudes)
freq_n = len(sin_frequencies)

angles = np.zeros((amp_n, freq_n, g_n, N))
velocities = np.zeros((amp_n, freq_n, g_n, N))
times = np.zeros(N)

angle_initial = 2
velocity_desired = 0

angle_desired = np.zeros((amp_n, freq_n, N))

motor.set_current(0)

try:
    for amp in range(amp_n):
        for freq in range(freq_n):
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
                        if i >= N:
                            break

                        angles[amp, freq, k, i] = theta
                        velocities[amp, freq, k, i] = dtheta
                        times[i] = time
                        
                        current_desired = sin_amplitudes[amp] * np.sin(sin_frequencies[freq] * time)
                        angle_desired[amp, freq, i] = current_desired

                        control = -gains[k] * (theta - current_desired)

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

fig, ax = plt.subplots(amp_n * freq_n, 2, figsize=(16, amp_n*freq_n*5))
bound = 1
last_n = 10


def add_plot(ax, x, ts, gain):
    ax.plot(
        ts,
        x,
        label=f"gain: {gain}",
    )

for amp in range(amp_n):
    for freq in range(freq_n):
        ax0 = ax[amp * freq_n + freq, 0]
        ax1 = ax[amp * freq_n + freq, 1]
        ax0.set_xlabel("t [s]")
        ax0.set_ylabel("$\\theta$ [$rad$]")

        ax1.set_xlabel("t [s]")
        ax1.set_ylabel("$\\dot{\\theta}$ [$\\frac{rad}{s}$]")

        for i in range(g_n):
            add_plot(
                ax=ax0,
                x=angles[amp, freq, i],
                ts=times,
                gain=gains[i],
            )
            add_plot(
                ax=ax1,
                x=velocities[amp, freq, i],
                ts=times,
                gain=gains[i],
            )

        ax0.plot(times, angle_desired[amp, freq], label=f"$\\theta_{{desired}}$, amplitude: {sin_amplitudes[amp]}, frequency: {sin_frequencies[freq]} Hz")
        ax0.legend()
        ax1.legend()

fig.suptitle(f"control loop frequency = {frequency} Hz", fontsize="13")
fig.tight_layout(pad=3.0)

plt.savefig("./plots/4.2.png")
plt.show()
