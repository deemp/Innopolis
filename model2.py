import numpy as np

from sympy import sin, integrate, diff, symbols, sqrt

a_t_max = 10
a_n_max = 6
v_max = 1.5
O_m = 3
theta_0 = 0.2
A = 1
x_min = 0
x_max = 4
num_x = 100000

xs = np.linspace(start=0., stop=4., num=num_x)
ys = np.sin(3*xs + theta_0)
y_dots = np.gradient(ys,xs)

dx = xs[1] - xs[0]
sigma = np.cumsum(np.sqrt(1+np.square(y_dots))*dx)

def sigma_from_x(x):
    return sigma[int(x/dx)]

# need to design sigma(t) such that 
# sigma_dot <= v_max
# sigma_ddot <= a_t_max
# sigma_dot^2 
# print(x_from_sigma(3))

# we take segments between critical trajectory points
# and let particle reach given speeds

period = np.pi/3
peak = (np.pi/2-0.2)/3
crit_x = np.array([0]+[peak + period * i for i in range(4)] + [4])

n = 6

segments = 2

v_goal = np.array([
    [0., 1.4, 1.02],
    [1.02, 1.4, 1.02],
    [1.02, 1.4, 1.02],
    [1.02, 1.4, 1.02],
    [1.02, 1.02, 0.]
])

t_ratios = np.array([
    [0.5, 0.5],
    [0.5, 0.5],
    [0.5, 0.5],
    [0.5, 0.5],
    [0.5, 0.5]
])

# calculate total time

ls = np.array([sigma_from_x(crit_x[i+1])-sigma_from_x(crit_x[i]) for i in range(n-1)])
def get_t_1(i):
    v = v_goal[i]
    r = t_ratios[i]
    l = ls[i]

    k = 0
    for j in range(segments):
        k += (v[j]+v[j+1])/2 * r[j]/r[0]
    
    t_1 = l/k
    return t_1

# t_1 -s only
t_1_i = np.array([get_t_1(i) for i in range(n-1)])

# time segments between acceleration changes
def get_t_i_s():
    # express t_i-s in terms of t_1-s
    t_T = np.divide(t_ratios,t_ratios.T[0][:,None])
    # multiply by corresponding t_1-s
    t_T = np.multiply(t_T,t_1_i[:,None])
    return t_T

t_i_s = get_t_i_s()

# times of start of accceleration change
t_i = np.cumsum(t_i_s.flatten())

def get_v_i():
    v = [v_goal[i] if i == 0 else v_goal[i][1:] for i in range(n-1)]
    return np.concatenate(v)
    
v_i = get_v_i()

print(v_i)

# num_t = 100000
num_t = 100

time = np.linspace(start=0, stop=t_i[-1], num=num_t)
dt = time[1] - time[0]

def get_velocities():
    v = np.zeros(num_t)

    # we use pointer to keep track of segment 
    # where velocity changes
    t_i_cur_idx = 0
    
    for i in range(num_t):
        t = time[i]
        # go to next segment
        if t > t_i[t_i_cur_idx]:
            t_i_cur_idx += 1
        
        # where the current segment ends
        t_i_cur = t_i[t_i_cur_idx]
        # where the previous segment ends
        t_i_prev = t_i[t_i_cur_idx-1] if t_i_cur_idx > 0 else 0.

        # velocity changes from v_1 to v_2 
        v_1 = v_i[t_i_cur_idx]
        v_2 = v_i[t_i_cur_idx+1]

        v[i] = v_1 + (v_2 - v_1) * (t - t_i_prev)/(t_i_cur - t_i_prev)
    
    return v

vs = get_velocities()

sigma_i = np.cumsum(vs * dt)

def x_from_sigma(s):
    return np.searchsorted(a=sigma, v=s, side='left')

x = xs[x_from_sigma(sigma_i)]
y = np.sin(3*x + theta_0)

p = np.vstack((x, y)).T



# print(p[:100])