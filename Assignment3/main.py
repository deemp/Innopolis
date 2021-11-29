#%% 
import numpy as np
from sympy.solvers.solvers import solve

# Assumptions:
#   t_params: [t0, tau, T, t_f] 
#     tau is the time of acceleration end
#     T is the start of deceleration start


#%%

# q_params = [q0, qf, dq_max, ddq_max]
# t0=   #initial trajectory time

# q_params_list = [] #list of q_params

#%%
def trajectory_time(q_params, t0): 
  '''
  q_params: [q0, qf, dq, ddq]
  t0: initial time of motion

  Output: 
  t_params: [t0, tau, T, t_f]
  '''
  
  q0, qf, dq, ddq = q_params
  delta_q = qf-q0
  tau, T = 0, 0

  if np.sqrt(np.abs(delta_q) * ddq) <= dq:
    # triangular profile
    tau = np.sqrt(np.abs(delta_q) / ddq)
    T = tau
  
  else:
    # trapezoidal profile
    tau = dq / ddq
    T = delta_q / dq

  t_params = t0 + np.array([0, tau, T, T + tau])

  return t_params

#%%

# check trajectory_time
q_params=[10,45,8,4]
t0 = 0

print(trajectory_time(q_params, t0))

#%%

N = 10000

#%%

def trapezoidal_trajectory(q_params, t_params, sync=False):
  '''
  t_params: [t0, tau, T, t_f]

  Output: 
  t: list of timesteps from t0 till tf
  q: list of positions from q0 to qf
  v: list of velocities from beginning to end of motion
  a: list of accelerations from beginning to end of motion
  '''
  
  t0, tau, T, t_f = t_params

  t = np.linspace(start=t0, stop=t_f, num=N)
  
  q0, qf, dq_max, ddq_max = q_params
  
  dq, ddq = 0,0
  
  if sync:
    delta_q = qf-q0
    dq = delta_q / (T - t0)
    ddq = dq / (tau - t0)

  else: 
    dq = dq_max
    ddq = ddq_max

  qs = np.zeros(N)
  dqs = np.zeros(N)
  ddqs = np.zeros(N)

  for i in range(N):
    if t[i] <= tau:
      # acceleration
      qs[i] = q0 + 1/2 * ddq * (t[i] - t0)**2
      dqs[i] = ddq * (t[i] - t0)
      ddqs[i] = ddq

    elif t[i] <= T:
      # steady velocity
      qs[i] = q0 - 1/2 * ddq * (tau - t0)**2 + dq * (t[i] - t0)
      dqs[i] = dq
      ddqs[i] = 0

    else:
      # deceleration
      qs[i] = qf - 1/2 * ddq * (t_f - t[i])**2
      dqs[i] = ddq * (t_f - t[i])
      ddqs[i] = -ddq

  q = qs
  v = dqs
  a = ddqs

  return t, q, v, a


#%%

# check trapezoidal_trajectory, trajectory_time

traj = trapezoidal_trajectory(q_params, trajectory_time(q_params, t0))

from utils import plot_list
plot_list([
  {
    'x': traj[0],
    'xlabel': "$t$ [s]",
    'ylabel': "$q$ [$rad$]",
    'graphs': [{"y": traj[1]}]
  },
  {
    'x': traj[0],
    'xlabel': "$t$ [s]",
    'ylabel': "$\dot{q}$ [$\\frac{rad}{sec}$]",
    'graphs': [{"y": traj[2]}]
  },
  {
    'x': traj[0],
    'xlabel': "$t$ [s]",
    'ylabel': "$\ddot{q}$ [$\\frac{rad}{sec^2}$]",
    'graphs': [{"y": traj[3]}]
  },
  ])

#%%

def time_synch(t_params_list): 
  '''
  t_params_list: a list of lists of time params of each joint motion [t_params_1, t_params_2, .....t_params_n]

  Output: 
  t_params_sync: list of synchronized time params [t_0, tau, T, t_f]
  '''

  t_params = np.array(t_params_list)
  
  # select max start time
  t0 = np.amax(t_params[:,0])
  
  # make start times the same for all joints
  for t in t_params:
    t += t0 - t[0]

  # select max tau
  tau = np.amax(t_params[:,1])

  # select max steady velocity time
  tau_T = np.amax(t_params[:,2] - t_params[:,1])

  T = tau + tau_T

  t_f = T + tau - t0

  t_params_sync = np.array([t0, tau, T, t_f])

  return t_params_sync


#%% 

# check time_synch

q_params_list = np.array(
  [
    [0, 35, 8, 4],
    [10, 50, 8, 4]
  ]
)

t0_list = np.array(
  [
    0,
    3
  ]
)
t_params_list = np.array([trajectory_time(i, j) for i,j in zip(q_params_list, t0_list)])

t_synced = time_synch(t_params_list)

traj_list = np.array([trapezoidal_trajectory(i, t_synced, sync=True) for i in q_params_list])

ts = traj_list[0,0]

plot_list([
  {
    'x': ts,
    'xlabel': "$t$ [s]",
    'ylabel': "$q$ [$rad$]",
    'graphs': [{"y": i[1]} for i in traj_list]
  },
  {
    'x': ts,
    'xlabel': "$t$ [s]",
    'ylabel': "$\dot{q}$ [$\\frac{rad}{sec}$]",
    'graphs': [{"y": i[2]} for i in traj_list]
  },
  {
    'x': ts,
    'xlabel': "$t$ [s]",
    'ylabel': "$\ddot{q}$ [$\\frac{rad}{sec^2}$]",
    'graphs': [{"y": i[3]} for i in traj_list]
  },
  ])

#%%
def trapezoid_p2p(q_params_list, t0_list):
  '''
  q_params_list: list of lists of actuator parameters 
  t0_list: list of moments at which joints can start motion

  Output: 
  t: list of time steps
  q: list of joint positions after synchronization [[q1], [q2] ...[qn]]
  dq, ddq: list of lists of joint velocities and accelerations in the same format as q
  '''
  # There was no description of t_params_sync, so I removed it

  t_params_list = np.array([trajectory_time(i, j) for i,j in zip(q_params_list, t0_list)])
  t_params_sync = time_synch(t_params_list)
  traj_list = np.array([trapezoidal_trajectory(i, t_params_sync, sync=True) for i in q_params_list])
  
  t = traj_list[0,0]
  q = traj_list[:,1]
  dq = traj_list[:,2]
  ddq = traj_list[:,3]

  return t, q, dq, ddq

# %%

# Main tasks

#%%

# 1 & 2

# Suppose each joint has q_params and t0
#%%

q_params_list = np.array(
  [
    [0, 6, 2, 3],
    [3, -2, 2, 3],
    [2, 1, 2, 3],
    [1, 3, 2, 3],
    [3, 0.5, 2, 3],
    [5, 1, 2, 3],
  ]
)

t0_list = np.array(
  [
    0,
    3,
    6,
    8,
    4,
    5
  ]
)

p2p = trapezoid_p2p(q_params_list, t0_list)

labels = [f'{i}' for i in range(1,7)]

plot_list([
  {
    'x': p2p[0],
    'xlabel': "$t$ [s]",
    'ylabel': "$q$ [$rad$]",
    'title': "Joint angles",
    'graphs': [{"y": i, "label": j} for i,j in zip(p2p[1], labels)]
  },
  {
    'x': p2p[0],
    'xlabel': "$t$ [s]",
    'ylabel': "$\dot{q}$ [$\\frac{rad}{sec}$]",
    'title': "Joint velocities",
    'graphs': [{"y": i, "label": j} for i,j in zip(p2p[2], labels)]
  },
  {
    'x': p2p[0],
    'xlabel': "$t$ [s]",
    'ylabel': "$\ddot{q}$ [$\\frac{rad}{sec^2}$]",
    'title': "Joint accelerations",
    'graphs': [{"y": i, "label": j} for i,j in zip(p2p[3], labels)]
  },
  ])

# %%

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

def solve_3(q_list, t_list, dqs, ddqs):
  """
  q_list: list of 3 angles that are needed to reach
  t_list: list of 3 moments at which given angles should be reached

  Output: 
  t: list of time steps
  q: list of joint positions after synchronization [[q1], [q2] ...[qn]]
  dq, ddq: list of lists of joint velocities and accelerations in the same format as q
  """
  # use https://drive.google.com/open?id=1poC8hXvsWiGc-ll9PadjtGfdDJCgkZLJ&disco=AAAASPiglvc
  
  a13, a12, a11, a10 = sp.symbols("a13 a12 a11 a10")
  a23, a22, a21, a20 = sp.symbols("a23 a22 a21 a20")
  a33, a32, a31, a30 = sp.symbols("a33 a32 a31 a30")
  a43, a42, a41, a40 = sp.symbols("a43 a42 a41 a40")
  
  t1, t2, t3 = [sp.Matrix([j**i for i in range(3,-1,-1)]) for j in t_list]
  q1, q2, q3 = q_list

  m1 = sp.Matrix([
    [a13, a12,         a11,         a10        ],
    [0,   3*a13,       2*a12,       a11        ],
    [0,   0,           6*a13,       2*a12      ],
    [0,   3*(a13-a23), 2*(a12-a22), a11-a21    ],
    [0,   0,           6*(a13-a23), 2*(a12-a22)],
    ])  
  c1 = sp.Matrix([q1, dqs[0], ddqs[0], 0, 0])
  
  m2 = sp.Matrix([t1.T, t2.T])
  a2 = sp.Matrix([a23, a22, a21, a20])
  c2 = sp.Matrix([q1, q2])

  m3 = sp.Matrix([t2.T, t3.T])
  a3 = sp.Matrix([a33, a32, a31, a30])
  c3 = sp.Matrix([q2, q3])

  m4 = sp.Matrix([
    [a43, a42,         a41,         a40        ],
    [0,   3*a43,       2*a42,       a41        ],
    [0,   0,           6*a43,       2*a42      ],
    [0,   3*(a33-a43), 2*(a32-a42), a31-a41    ],
    [0,   0,           6*(a33-a43), 2*(a32-a42)],
    ])
  c4 = sp.Matrix([q3, dqs[1], ddqs[1], 0, 0])
  
  m5 = sp.Matrix([
    [0  , 3*(a23-a33), 2*(a22-a32), a21-a31      ],
    [0  , 0,           6*(a23-a33), 2*(a22 - a32)],
    ])

  eq1 = m1 * t1-c1
  eq2 = m2 * a2-c2
  eq3 = m3 * a3-c3
  eq4 = m4 * t3-c4
  eq5 = m5 * t2

  sol = sp.solve([eq1, eq2, eq3, eq4, eq5])
  print(sol)

v1,v2 = sp.symbols("v1 v2")
a1,a2 = sp.symbols("a1 a2")
# v1,v2 = 1,-1
# a1,a2 = 0,0
q_list = [2.,3.,2.]
t_list = [1.,2.,3.]

# fig, ax = plt.subplots()
# ax.scatter(t_list, q_list)
# ax.plot([t_list[0],t_list[0]+0.1],[q_list[0],q_list[0]+a1*0.1])
# ax.plot([t_list[2],t_list[2]+0.1],[q_list[2],q_list[2]+a2*0.1])

# for i in np.arange(0,3,0.02):
solve_3(q_list,t_list,[v1,v2],[a1,a2])

  

# %%
