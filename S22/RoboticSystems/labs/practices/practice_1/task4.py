#%%
from time import sleep

from can import CANDevice, CANSocket
import numpy as np

can_bus = CANSocket()
encoder = CANDevice(can_bus = can_bus, device_id = 0x141)

encoder.command = b"\x9C"+ 7 * b"\x00"

raw_state = {}

def parse_sensor_data(reply):
    """parse the raw sensor data from the CAN frame"""
    raw_state["temp"] = reply[1]
    raw_state["current"] = encoder.from_bytes(reply[2:4])
    raw_state["speed"] = encoder.from_bytes(reply[4:6])
    raw_state["encoder"] = encoder.from_bytes(reply[6:])
    
    return raw_state

import matplotlib.pyplot as plt

t = 2
n = 1000
dt = t/n
a = np.zeros(n)
N = 2**14
threshold = 16000

cutoff_frequencies = [5,20,50]

try:
    counts = 0
    prev = 0

    for i in range(n):
        encoder.execute()
        state = parse_sensor_data(encoder.reply)
        current = state["encoder"]

            # limit in between
        if prev - current >= threshold:
            counts += N - prev + current
        # limit not in between
        elif 0 <= current - prev <= threshold:
            counts += current - prev
        # limit in between
        elif 0 <= prev - current <= threshold:
            counts -= prev - current
        elif current - prev >= threshold:
            counts -= N - current + prev

        a[i] = counts/N * 2 * np.pi
        prev = current

        sleep(dt)

except KeyboardInterrupt:
    print('Disabled by interrupt')


#%%

fig, axs = plt.subplots(1,3, figsize=(18,6))

for i, ax in enumerate(axs):
    
    k = cutoff_frequencies[i]
    rc = 1 / (2 * np.pi * k)
    alpha = dt / (rc + dt)

    vs = np.gradient(a, dt)
    
    for j in range(1,n):
        vs[j] = alpha * vs[j] + (1 - alpha) * vs[j-1]

    ts = np.linspace(0, t, n)

    ax.plot(ts, vs)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("$\\dot{\\theta}$ [$\\frac{rad}{s}$]")   
    ax.set_title(f"cutoff frequency: {k} Hz")

fig.tight_layout(pad=3.0)

plt.savefig("practice_1/plots/4.1.png")
plt.show()