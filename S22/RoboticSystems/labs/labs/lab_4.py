from libs.can import CANSocket
from libs.myactuator import MyActuator
from time import perf_counter
from math import pi

# the serial port of device 
# you may find one by examing /dev/ folder,
# this is usually devices ttyACM
serial_device = 'ttyACM0'

# Initiate the can bus socket 
can_bus = CANSocket(serial_port=serial_device)

# Initiate motor 
pendulum = MyActuator(can_bus=can_bus)
pendulum.torque_constant = 1e-3

# ////////////////////////////////
# USE THIS TO SET MOTOR ZERO
# the motor must be rebooted after 
# ////////////////////////////////
# pendulum.set_zero(persistant=True)


# Set the control loop timings
frequency  = 500
sampling_time = 1/frequency

try:
    last_execution = 0
    control = 0 
    
    # find the global time before intering control loop 
    initial_time = perf_counter()
    while True:
        time = perf_counter() - initial_time # get actual time in secs
        
        # /////////////////////////
        # Get and parse motor state
        # /////////////////////////
        state = pendulum.state
        theta = state['angle']
        dtheta = state['speed']
        torque = state['torque']

        # ///////////////////////////////////////////
        # Update the control only on specific timings
        # /////////////////////////////////////////// 
        if (time - last_execution) >= sampling_time:
            last_execution = time
            # YOUR CONTROLLER GOES HERE
            control = 0
            print(f'Motor angle data: {round(theta, 5)}', end='    \r', flush=True)

        pendulum.set_torque(control)

except KeyboardInterrupt:
    print('Disabled by interrupt')
except Exception as e:
    print(f'\n!!!! EXCEPTION !!!!\n {e} \n!!!! EXCEPTION !!!!\n')
finally:
    for i in range(100):
        pendulum.set_torque(0)
