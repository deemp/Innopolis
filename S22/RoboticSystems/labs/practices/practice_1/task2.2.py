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

direction = "mixed"

try:
    t = 3
    n = 1000
    dt = t/n
    N = 2**14
    counts = 0
    prev = 0
    threshold = 16000
    a = np.zeros(n)
    
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
    
    t = np.linspace(0,t,n)
    plt.plot(t, a)
    plt.xlabel("t [s]")
    plt.ylabel("$\\theta$ [rad]")
    plt.title(f"{direction} rotation")
    plt.savefig(f"practice_1/plots/2.2/{direction}.png")
    plt.show()

except KeyboardInterrupt:
    print('Disabled by interrupt')


