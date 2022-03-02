from libs.can import CANSocket
from libs.myactuator import MyActuator
from time import perf_counter
from math import pi, sin
# the serial port of device 
# you may find one by examing /dev/ folder,
# this is usually devices ttyACM
serial_device = 'ttyACM0'

# Initiate the can bus socket 
can_bus = CANSocket(serial_port=serial_device)

# Initiate motor 
pendulum = MyActuator(can_bus=can_bus)
pendulum.torque_constant = 1.0
# motor.set_zero(persistant=True)
# Set the control loop timings
frequency  = 500
sampling_time = 1/frequency

kp, kd = 100, 20
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
            control = kp*(-pi/2 - theta) - kd*dtheta + 1000*(0.1)*9.81*0.1*sin(theta)


        # motor.set_current(control)
        pendulum.set_torque(control)
        
        print(f'Motor angle data: {theta}', end='    \r', flush=True)
        # print(f'Motor angle data: {torque} {control}')

# m*g*l*sin(theta) = 0.1*9.81*0.1*0.95 = km*I
# 0.9430844374660935 100 
# -0.9562698664006581 -101 

except KeyboardInterrupt:
    for i in range(100):
        pendulum.set_torque(0)
    # motor.disable()
    print('Disabled by interrupt')
