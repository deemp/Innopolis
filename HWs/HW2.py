#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import sympy as sp
from sympy.utilities.lambdify import lambdify
from numpy import array

plt.rcParams['font.size'] = '16'

#%%
T_START = 0
T_END = 1000000
N_LIM = 10000
T_LINSPACE = np.linspace(0,T_END,N_LIM)
PHI_0 = [5, -5, 0]
K = 0.01

def sign(x):
    return np.tanh(100 * x)

# https://stackoverflow.com/a/43843544
# using tanh instead of sign function

def sys_1(state, t):
    phi, dphi = state
    ddphi = -sign(K * phi + dphi)
    return np.array([dphi, ddphi])

def f (phi_0):
    y0 = np.array([phi_0,0.])
    ys = odeint(func=sys_1,y0=y0,t=T_LINSPACE)
    axs.plot(ys[:,0], ys[:,1],label=f"{phi_0:.3f}")
    axs.scatter([ys[0,0]],[ys[0,1]],label="start",marker='o')
    axs.legend()
    axs.grid()
    axs.set_xlabel("$\\varphi$")
    axs.set_ylabel("$\\dot{\\varphi}$")

fig, axs = plt.subplots(1,1,figsize=(10,7))
fig.suptitle(
    f"Phase plots for different initial conditions$\\varphi_{{0}}$\n"+\
        f"of the system $\\ddot{{\\varphi}}=-sign(k\\varphi+\\dot{{\\varphi}})$, k={K}")


for x in PHI_0:
    f(x)


plt.tight_layout()
plt.plot()

#%%
CHARACTERISTICS = {
    "Stable node": [-3,-2],
    "Unstable node": [2,3],
    "Saddle point": [-3,3],
    "Stable focus": [-3+2j,-3-2j],
    "Unstable focus": [3+2j,3-2j],
    "Center point": [3j,-3j]
}
HEIGHT = 2
WIDTH = 3
X_RANGE = [4,5]
N = 30
K = 1.1
SCALE = 4

X1_MAX, X2_MAX = X_RANGE
X1_SPAN = np.linspace(-K*X1_MAX, K*X1_MAX, N)
X2_SPAN = np.linspace(-K*X2_MAX, K*X2_MAX, N)
X1_GRID, X2_GRID = np.meshgrid(X1_SPAN, X2_SPAN)


def g(l1,l2):
    a = np.real(-(l1 + l2))
    b = np.real(l1 * l2)
    
    def sys(x):
        return x[1], - b * x[0] - a * x[1]

    dx1, dx2 = sys([X1_GRID, X2_GRID])
    return dx1, dx2


def f (l1,l2,name,ax):
    dx1, dx2 = g(l1,l2)
    ax.streamplot(x=X1_SPAN, y=X2_SPAN, u=dx1, v=dx2,
            arrowsize=1.2, # size of the arrows 
           density=0.9, # density of the vectorfield
           color='k', # color for the trajectories
           linewidth=1, # the width of the lines for trajectories 
           arrowstyle='->') # you can change type of arrows
    ax.set_title(name)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$\\dot{x}$")

fig.suptitle(f"Trajectories near different types of equilibrium points")
fig,axs = plt.subplots(nrows=HEIGHT,ncols=WIDTH,figsize=(WIDTH * SCALE, HEIGHT * SCALE))

def plot_systems():
    for (k,t) in enumerate(CHARACTERISTICS.items()):
        x,[y,z] = t
        i = k//WIDTH
        j = k%WIDTH
        # print(i,j)
        f(name = x,l1=y,l2=z, ax=axs[i,j])

plot_systems()

plt.tight_layout()
plt.plot()

#%%

HEIGHT = 1
WIDTH = 3
X_RANGE = [4,5]
N = 30
K = 1.1
SCALE = 4

X1_MAX, X2_MAX = X_RANGE
X1_SPAN = np.linspace(-K*X1_MAX, K*X1_MAX, N)
X2_SPAN = np.linspace(-K*X2_MAX, K*X2_MAX, N)
X1_GRID, X2_GRID = np.meshgrid(X1_SPAN, X2_SPAN)

def sys_a (x):
    dx1 = x[1] - x[0] * (x[0] ** 2 + x[1] ** 2 - 1)
    dx2 = -x[0] - x[1] * (x[0] ** 2 + x[1] ** 2 - 1)
    return dx1, dx2

def sys_b (x):
    dx1 = x[1] + x[0] * (x[0] ** 2 + x[1] ** 2 - 1)
    dx2 = -x[0] + x[1] * (x[0] ** 2 + x[1] ** 2 - 1)
    return dx1, dx2

def sys_c (x):
    dx1 = x[1] - x[0] * (x[0] ** 2 + x[1] ** 2 - 1)
    dx2 = -x[0] - x[1] * (x[0] ** 2 + x[1] ** 2 - 1)
    return dx1, dx2


SYSTEMS = {
    "a": sys_a,
    "b": sys_b,
    "c": sys_c
}

def f (sys,name,ax):
    dx1, dx2 = sys([X1_GRID, X2_GRID])
    ax.streamplot(x=X1_SPAN, y=X2_SPAN, u=dx1, v=dx2,
            arrowsize=1.2, # size of the arrows 
           density=0.9, # density of the vectorfield
           color='k', # color for the trajectories
           linewidth=1, # the width of the lines for trajectories 
           arrowstyle='->') # you can change type of arrows
    ax.set_title(name)
    ax.set_xlabel("$x_{1}$")
    ax.set_ylabel("$x_{2}$")

fig.suptitle(f"Trajectories of different systems")
fig,axs = plt.subplots(nrows=HEIGHT,ncols=WIDTH,figsize=(WIDTH * SCALE, HEIGHT * SCALE))

def plot_systems():
    for (i,t) in enumerate(SYSTEMS.items()):
        name,sys = t
        f(name = name,sys = sys, ax=axs[i])

plot_systems()

plt.tight_layout()
plt.plot()
# %%
STABLE = "asymptotically stable"
UNSTABLE = "unstable"
UNDEFINED ="undefined"

NEG_1 = -1
ZERO = 0
POS_1 = 1

def f(x):
    return x[1], -x[1] - np.sin(x[0])

def classify(x):
    # strictly in the left-hand plane
    if np.real(x) < 0.:
        return NEG_1
    if np.real(x) == 0.:
        return ZERO
    if np.real(x) > 0.:
        return POS_1


def g():
    x = sp.symbols(r'x_1, x_2')
    f_sym = sp.Matrix([f(x)]).T
    # equlibriums = sp.solve(f_sym, x)
    # jacobian = f_sym.jacobian(x)
    # jacobian_num = lambdify([x], jacobian)

    # for equlibrium in equlibriums:
    #     x_e = array(equlibrium, dtype='double')
    #     A = array(jacobian_num(x_e), dtype='double')
    #     ls = np.apply_along_axis(classify, np.linalg.eigvals(A))
    #     p = None
    #     if np.all(ls < 0):
    #         p = STABLE
    #     elif np.all(ls <= 0):
    #         p = UNDEFINED
    #     else:
    #         p = UNSTABLE
        
    #     print(f"The type of equilibrium {x_e} is: {p}")

g()

# print point, in legend put type
# %%

X_RANGE = [4,5]
N = 30
K = 1.1
SCALE = 4

X1_MAX, X2_MAX = X_RANGE
X1_SPAN = np.linspace(-K*X1_MAX, K*X1_MAX, N)
X2_SPAN = np.linspace(-K*X2_MAX, K*X2_MAX, N)
X1_GRID, X2_GRID = np.meshgrid(X1_SPAN, X2_SPAN)

