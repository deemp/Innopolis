# Material:
# Link to the Colab notebook:
# https://colab.research.google.com/drive/1s5F1uFqCK1MOG2GHzCWqSWqECDmC3O5h#scrollTo=zPmrTNlSBW-R

# You may find the IPYNB/RISE notebook in: 
# robotic_systems_lab/notebooks/_modeling_and_control_of_manipulators.ipynb

# Link to the practice report: 
# https://docs.google.com/document/d/1pclpAymcGvW-LY-nnHapWQWj-ISFJGZDbQDDOkX6a38/edit#


from time import perf_counter, sleep
from multiprocessing import Process, Manager
from numpy import pi, array, sin, cos, concatenate, zeros



def sysode(x, t, control, params):
    q, dq = x[:2], x[2:4]

    # /////////////////////// 
    # put your dynamics here
    # ddq = INVERSE DYNAMICS
    # ///////////////////////
    
    dx1 = dq
    dx2 = ddq
    dx = dx1, dx2

    return concatenate(dx)


def simulator(system):
    # //////////////////////
    
    # System parameters
    l = 0.3, 0.3 # lengths of links
    m = 0.5, 1.0 # masses of links 
    J = 0.01, 0.01 # Inertia of motors 
    b = 0.1, 0.1 # damping coefficients 
    g = 9.81 # gravitational acceleration 
    params = l, m, J, b, g

    # ////////////////////
    
    try:
        last_execution = 0
        initial_time = perf_counter()
        while True:
            # ///////////////////////////////////////////
            time = perf_counter() - initial_time  # get actual time in secs
            dt = time - last_execution
            if dt >= sampling_time/sim_ratio:
                last_execution = time
                control = system.control

                # DO SIMULATION
                # IMPLEMENT YOUR SIMULATOR HERE
                system.state = system.state


    except KeyboardInterrupt:
        print('\nSimulator is terminated')

# Set the control loop timings
frequency = 500
sampling_time = 1/frequency
sim_ratio = 5

manipulator = Manager().Namespace()

# SET INITIAL STATE
manipulator.state = array([0, 0, 0 ,0])
manipulator.control = zeros(2)


simulator_proc = Process(target=simulator, args=(manipulator,))
simulator_proc.start()


try:
    last_execution = 0
    control = 0
    # find the global time before intering control loop
    initial_time = perf_counter()
    while True:
        time = perf_counter() - initial_time  # get actual time in secs

        theta_1, theta_2, dtheta_1, dtheta_2 = manipulator.state
        # ///////////////////////////////////////////
        # Update the control only on specific timings
        # ///////////////////////////////////////////
        if (time - last_execution) >= sampling_time:
            last_execution = time
            control = zeros(2)

        manipulator.control = control

        print(f'State: {manipulator.state}', end='    \r', flush=True)


except KeyboardInterrupt:

    print('Disabled by interrupt')
# except Exception as e:
#     print(f'\n!!!! EXCEPTION !!!!\n {e} \n!!!! EXCEPTION !!!!\n')


finally:
    sleep(0.5)
    simulator_proc.join()
