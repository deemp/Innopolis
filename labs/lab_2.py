from libs.can import CANSocket
from libs.myactuator import MyActuator
from time import perf_counter

# the serial port of device 
# you may find one by examing /dev/ folder,
# this is usually devices ttyACM
serial_device = 'ttyACM1'

# Initiate the can bus socket 
can_bus = CANSocket(serial_port=serial_device)

# Initiate motor 
motor = MyActuator(can_bus=can_bus)

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
            control = -100*theta - 5*dtheta


        motor.set_current(control)
        
        print(f'Motor angle data: {theta}', end='    \r', flush=True)
            

except KeyboardInterrupt:
    motor.set_current(0)
    print('Disabled by interrupt')
