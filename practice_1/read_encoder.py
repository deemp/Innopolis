from can import CANDevice, CANSocket

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

try:
    while True:
        encoder.execute()
        state = parse_sensor_data(encoder.reply)
        print(f'Encoder data: {state["encoder"]}', end='    \r', flush=True)

except KeyboardInterrupt:
    print('Disabled by interrupt')
