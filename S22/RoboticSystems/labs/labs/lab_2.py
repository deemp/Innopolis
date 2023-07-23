from libs.can import CANSocket
from libs.myactuator import MyActuator
from time import perf_counter

# the serial port of device
# you may find one by examing /dev/ folder,
# this is usually devices ttyACM
serial_device = 'ttyACM4'

# Initiate the can bus socket
can_bus = CANSocket(serial_port=serial_device)

# Initiate motor
motor = MyActuator(can_bus=can_bus)

# Set the control loop timings
frequency = 200
sampling_time = 1/frequency


try:
    last_execution = 0
    control = 0

    # find the global time before intering control loop
    initial_time = perf_counter()
    while True:
        time = perf_counter() - initial_time  # get actual time in secs

        # /////////////////////////
        # Get and parse motor state
        # /////////////////////////
        state = motor.state
        theta = state['angle']
        dtheta = state['speed']
        current = state['current']

        # ///////////////////////////////////////////
        # Update the control only on specific timings
        # ///////////////////////////////////////////
        if (time - last_execution) >= sampling_time:
            last_execution = time
            # YOUR CONTROLLER GOES HERE
            control = -100*theta - 10*dtheta
            print(f'Motor angle data: {theta}', end='    \r', flush=True)

        motor.set_current(control)



except KeyboardInterrupt:
    print('Disabled by interrupt')
except Exception as e:
    print(f'\n!!!! EXCEPTION !!!!\n {e} \n!!!! EXCEPTION !!!!\n')
finally:
    for i in range(100):
        motor.set_current(0)
