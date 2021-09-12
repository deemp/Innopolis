
#%%
import numpy as np
from sympy import sin, diff, integrate, symbols, sqrt

"""
Given
"""

a_t_max = 10
a_n_max = 6
v_max = 1.5
O_m = 3
theta_0 = 0.2
A = 1
x_min = 0
x_max = 4
x_ = symbols('x')
y = A * sin(O_m * x_ + theta_0)

"""
Calculations
"""

# for each point, we want to choose the velocity with which we can move to the next point

time_steps=1000
time = np.linspace(start=0., stop=20., num=time_steps,dtype=np.float128)
time_step = time[1] - time[0]

def num_derivative_1(vector, time=time):
    return np.gradient(vector, time[:vector.shape[0]])

def num_derivative_2(point, time=time):
    dxdt = num_derivative_1(point.T[0])
    dydt = num_derivative_1(point.T[1])
    return np.array([dxdt, dydt]).T

velocities = np.zeros((time_steps,2))

def sigma_ddot_at(i):
    if i == 0:
        return 0.3
    sigma = np.cumsum(np.linalg.norm(velocities[:i+1],axis=1) * time_step)
    sigma_ddot = num_derivative_1(num_derivative_1(sigma))
    return sigma_ddot[i]

#%%

last_window = 10
def last_ind(i):
    return np.arange(start=max(0,i-last_window+1),stop=i+1)

def is_satisfying(i):
    # given last n velocities and last coordinate,
    # check if a_t, a_n, v satisfy the constraints

    ind = last_ind(i)
    v = velocities[ind]

    # take norms of all velocities
    v_abs = np.linalg.norm(v,axis=1)
    
    # get last taus
    tau_i = v/v_abs[:,None]
    
    # get acceleration
    a = num_derivative_2(v)
    
    # get last indices for natural acceleration
    sigma_ddots = np.array([sigma_ddot_at(i) for i in ind])
    a_t = tau_i * sigma_ddots[:,None]
    # Maybe TODO: this may change previous accelerations
    # so check previus (need to remember tau-s)
    a_n = a - a_t

    a_t_abs = np.linalg.norm(a_t, axis=1)
    a_n_abs = np.linalg.norm(a_n, axis=1)


    is_good = (a_t_abs <= a_t_max).all() and (a_n_abs <= a_n_max).all()

    return is_good


"""
we start at x = 0 and try to find velocities for each moment of time
"""

def numeric(expr, s, t):
    return float(expr.subs(s, t).evalf())

points = np.zeros((time_steps, 2))
points[0] = np.array([x_min, numeric(y, x_, 0)])

h_max = v_max * time_step
hs = np.linspace(start=0., stop=h_max, num=10)
h_step = hs[1] - hs[0]
hs = hs[::-1]


# we need to find velocity for point 0
# it'll be the average velocity to the next point
def calculate_velocity_for_first():
    P = points[0]
    x = P[0]
    for h in hs:
        # try next point
        Q = np.array([x+h,numeric(y, x_, x+h)])
        # dy/dt
        v = (Q-P)/time_step
        if np.linalg.norm(v) > np.linalg.norm(np.array([a_n_max, a_t_max])*time_step)/2:
            pass
        
        # velocity at current point
        velocities[0] = v
        points[1] = Q
        break
    

def next_point(i):
    # choose i-th point
    P = points[i]
    x = P[0]
    for h in hs:
        # try next point
        Q = np.array([x+h, numeric(y, x_, x+h)])
        # dy/dt
        v = (Q-P)/time_step
        if np.linalg.norm(v) > v_max:
            pass
            
        # velocity at current point
        velocities[i] = v
        # check if everything is still ok
        if is_satisfying(i):
            # Q is at max possible h from current x
            points[i+1] = Q
            break

m = 5
def calculate_velocities():
    for i in range(1,m):
        next_point(i)


calculate_velocity_for_first()
calculate_velocities()

# velocities
print(velocities[:m])
# %%
