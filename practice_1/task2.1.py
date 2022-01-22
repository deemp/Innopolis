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

try:
    t = 10
    n = 1000
    dt = t/n
    a = np.zeros(n)
    for i in range(n):
        encoder.execute()
        state = parse_sensor_data(encoder.reply)
        a[i] = state["encoder"]
        sleep(dt)
    
    t = np.linspace(0,t,n)
    plt.plot(t, a)
    plt.savefig("practice_1/plots/2.1.png")
    plt.show()

except KeyboardInterrupt:
    print('Disabled by interrupt')


