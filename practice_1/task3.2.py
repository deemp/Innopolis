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

t = 1
n = 1000
dt = t/n
dts = [dt, 2 * dt, 3 * dt]
a = np.zeros((3,n))
real_speed = np.zeros((3,n))

# milliseconds per count
count_time = 90

N = 2**14
threshold = 16000


try:
    for j,k in enumerate(dts):
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

            a[j,i] = counts / N * 2 * np.pi
            real_speed[j,i] = state["speed"] / N * 2 * np.pi * count_time

            prev = current

            sleep(k)

except KeyboardInterrupt:
    print('Disabled by interrupt')

# np.savetxt("./data/3.1.txt", a, delimiter=",")

#%%

fig, axs = plt.subplots(1,3, figsize=(18,6))

for i,ax in enumerate(axs):
    d = dts[i]
    vs = np.gradient(a[i], d)
    ts = np.linspace(0, n * d, n)
    ax.plot(ts, vs, label="estimated")
    ax.plot(ts, real_speed[i], label="motor data")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("$\\dot{\\theta}$ [$\\frac{rad}{s}$]")   
    ax.set_title(f"sampling time = {int(d*1000)} ms")
    ax.legend()

fig.tight_layout(pad=3.0)

plt.savefig("./plots/3.2.png")
plt.show()