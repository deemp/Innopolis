# Material:
# Link to the Colab notebook:
# https://colab.research.google.com/drive/1s5F1uFqCK1MOG2GHzCWqSWqECDmC3O5h#scrollTo=zPmrTNlSBW-R

# You may find the IPYNB/RISE notebook in:
# robotic_systems_lab/notebooks/_modeling_and_control_of_manipulators.ipynb

# Link to the practice report:
# https://docs.google.com/document/d/1pclpAymcGvW-LY-nnHapWQWj-ISFJGZDbQDDOkX6a38/edit#


from time import perf_counter, sleep
from multiprocessing import Process, Manager
from numpy import pi, array, sin, cos, concatenate, zeros, dot
import numpy as np
from numpy.linalg import inv

# import numpy as np

# System parameters
l = 0.3, 0.3  # lengths of links
m = 0.5, 1.0  # masses of links
J = 0.01, 0.01  # Inertia of motors
b = 0.1, 0.1  # damping coefficients
g = 9.81  # gravitational acceleration
params = l, m, J, b, g

b1, b2 = b
l1, l2 = l
m1, m2 = m
J1, J2 = J


def D(a1, a2):

    return array(
        [
            [l1 ** 2 * (m1 + m2) + J1, l1 * l2 * m2 * cos(a1 - a2)],
            [l1 * l2 * m2 * cos(a1 - a2), l2 ** 2 * m2 + J2],
        ]
    )


def Qd(da1, da2):
    return array([b1 * da1, b2 * da2])


def h(a1, a2, da1, da2):
    # print("h",a1,a2,da1,da2)
    c = array(
        [
            l1 * l2 * m2 * sin(a1 - a2) * da2 ** 2,
            -l1 * l2 * m2 * sin(a1 - a2) * da1 ** 2,
        ]
    )

    G = array([l1 * m1 * g * cos(a1) + l1 * m2 * g * cos(a1), l2 * m2 * g * cos(a2)])

    return c + G


def sysode(x, t, control, params=[]):
    x1, x2 = x[:2], x[2:4]
    # Q == u?

    # ///////////////////////
    # put your dynamics here
    # ddq = INVERSE DYNAMICS
    # ///////////////////////
    u = control
    dx1 = x2
    # print("calling h", *x1, *x2)
    dx2 = inv(D(*x1)).dot(u - Qd(*x2) - h(*x1, *x2))
    dx = dx1, dx2

    return concatenate(dx)


T = 1
N = 2000


def simulator(system):
    try:
        last_execution = 0
        initial_time = perf_counter()
        iteration = 0
        system.T = 0
        arr = np.zeros((4, N))
        while True:
            # ///////////////////////////////////////////
            time = perf_counter() - initial_time  # get actual time in secs
            dt = time - last_execution
            if dt * sim_ratio >= sampling_time:
                # if time >= T:
                # break
                if iteration >= N:
                    system.T = time
                    break

                last_execution = time
                control = system.control

                arr[:,iteration] = system.state
                # DO SIMULATION
                # IMPLEMENT YOUR SIMULATOR HERE
                system.state = system.state + sysode(system.state, time, control) * dt

                # print(
                #     f"iteration: {iteration}, State: {system.state}",
                #     end="    \r",
                #     flush=True,
                # )
                iteration += 1

    except KeyboardInterrupt:
        print("\nSimulator is terminated")
    finally:
        system.array = arr


# Set the control loop timings
frequency = 500
sampling_time = 1 / frequency
sim_ratio = 5

manipulator = Manager().Namespace()

# SET INITIAL STATE
manipulator.state = array([0, 0, 0, 0])
manipulator.control = zeros(2)
manipulator.array = np.array([])
manipulator.T = 1

simulator_proc = Process(target=simulator, args=(manipulator,))
simulator_proc.start()

data = zeros((4, N))
try:
    last_execution = 0
    control = 0
    # find the global time before intering control loop
    initial_time = perf_counter()
    iteration = 0
    while True:
        # break
        time = perf_counter() - initial_time  # get actual time in secs

        # ///////////////////////////////////////////
        # Update the control only on specific timings
        # ///////////////////////////////////////////
        if (time - last_execution) >= sampling_time:
            # if time >= T:
            #     break
            if iteration >= N:
                break
            theta_1, theta_2, dtheta_1, dtheta_2 = manipulator.state
            data[:, iteration] = manipulator.state

            last_execution = time
            control = zeros(2)

            manipulator.control = control

            iteration += 1

        # print(f'State: {manipulator.state}', end='    \r', flush=True)
        # print(f'State: {manipulator.state}', end='    \r', flush=False)


except KeyboardInterrupt:

    print("Disabled by interrupt")
except Exception as e:
    print(f"\n!!!! EXCEPTION !!!!\n {e} \n!!!! EXCEPTION !!!!\n")


finally:
    sleep(0.5)
    simulator_proc.join()
    arr = manipulator.array
    T = manipulator.T * sim_ratio

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(10, 6))

    ts = np.linspace(0, T, N)
    # theta_desireds = np.full(N, theta_desired)

    ax[0].plot(ts, arr[0, :], label=f"$\\alpha_{{1}}$")
    ax[0].plot(ts, arr[1, :], label=f"$\\alpha_{{2}}$")
    ax[0].grid()
    ax[0].set_title(f"Joint angles")
    ax[0].set_xlabel("simulated time [s]")
    ax[0].set_ylabel("$\\alpha$ [rad]")
    ax[0].legend()

    ax[1].plot(ts, arr[2, :], label=f"$\\dot{{\\alpha}}_{{1}}$")
    ax[1].plot(ts, arr[3, :], label=f"$\\dot{{\\alpha}}_{{2}}$")
    ax[1].grid()
    ax[1].set_title(f"Joint velocities")
    ax[1].set_xlabel("simulated time [s]")
    ax[1].set_ylabel("$\\dot{\\alpha}$ [$\\frac{rad}{s}$]")
    ax[1].legend()

    plt.tight_layout()
    plt.savefig("./images/task1.png")
    plt.show()
