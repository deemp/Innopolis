# %%

from cProfile import label
from sympy import *
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = '20'

t, a, l, k, m, g, nu = symbols('t a l k m g \\nu')
phi_s, dphi_s, ddphi_s, nu_s = symbols('\\varphi \\dot{\\varphi} \\ddot{\\varphi} \\nu',real=True,cls=Function)
phi = phi_s(t)
dphi = dphi_s(t)
nu = nu_s(t)


#%%

Sy = a * cos(nu * t)

#%%
Ft = -m * Sy.diff(t).diff(t)
dnu, ddnu = symbols('\\dot{\\nu} \\ddot{\\nu}')
# Ft_nu = Ft
Ft_nu = Ft.subs({Derivative(nu, t): dnu, Derivative(nu, t): ddnu})
#%%

Ft_nu
print(Ft_nu)
#%%
ddphi_nu = ddphi_s(t)
ks = solve(-(Ft_nu + m * g) * l * sin(phi) - k * dphi - m * l ** 2 * ddphi_nu, ddphi_nu)

#%%
ks[0]
#%%

# ddphi = - ((a * nu ** 2 * cos(nu * t) + g) * sin(phi)) / l - k * dphi / (m * l**2)
ddphi = ks[0]
#%%
ddphi
#%%
print(latex(ddphi))

#%%

x_sym = phi, dphi
# %%
f_sym = Matrix([dphi, ddphi])
f_sym
# %%

Jx_sym = f_sym.jacobian(x_sym)
Jx_sym

#%%
print(latex(Jx_sym))
# %%

Ju_sym = f_sym.jacobian([nu])
Ju_sym

#%%

print(latex(Ju_sym))

#%% 
u_d = {a: 0.1, nu: 120}
x_d = {phi: pi, dphi: 0}
subs_d = u_d | x_d
#%%

A_sym = Jx_sym.subs(subs_d)
A_sym
#%%
print(latex(A_sym))
# %%

B_sym = Ju_sym.subs(subs_d)
B_sym
print(latex(B_sym))

#%%

m_v = 1
l_v = 1
params = {l: l_v, g: 9.8, m: m_v, k: m_v * l **2}
subs = u_d | x_d | params

A_num_linear = lambdify([t, ddnu],A_sym.subs(subs), 'numpy')
#%%
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


def pendulum_linearized(x, t, A_num):
    return A_num(t).dot(x)

def calc(sys, A_num, label):
    t = np.linspace(0, 10, 100000)
    y0 = np.array([3 * np.pi / 4, 0])
    print(A_num(3))
    ys = odeint(func=sys, y0=y0, t=t, args=(A_num,))
    pi2 = np.pi * 2
    md = lambda x: np.fmod(np.fmod(x, pi2) + pi2, pi2)
    md1 = lambda x: np.fmod(x, pi2)
    id = lambda x : x
    fig, ax = plt.subplots(figsize=(20,10))
    fig.suptitle(f"$\\nu$ = {label}")
    f = id
    # ax.scatter(t, id(ys[:,0]), label="$\\varphi(t)$", s=3)
    ax.plot(t, f(ys[:,0]), label="$\\varphi(t)$")
    # plt.plot(t, ys[:,1], label="$\\dot{\\varphi}(t)$")
    ax.legend()
    plt.plot()

#%%
nu_sin = sin(t).diff(t).diff(t)
nu_sin

#%%
print(latex(A_sym.subs(subs| {ddnu: nu_sin})))
#%%
A_sin = lambdify(t,A_sym.subs(subs | {ddnu: nu_sin}), 'numpy')
calc(sys=pendulum_linearized, A_num=A_sin, label=f"${latex(sin(t))}$")

#%%
nu_constant = Rational(120).diff(t).diff(t)
nu_constant
#%%
A_const = lambdify(t,A_sym.subs(subs | {ddnu: nu_constant}), 'numpy')
calc(sys=pendulum_linearized, A_num=A_const, label=f"${latex(120)}$")

#%%
nu_inv = ((t**2.5+3)).diff(t).diff(t)
nu_inv

#%%
print(latex(A_sym.subs(subs | {ddnu: nu_inv})))

#%%
A_inv = lambdify(t, A_sym.subs(subs | {ddnu: nu_inv}), 'numpy')
calc(sys=pendulum_linearized, A_num=A_inv, label=f"${latex((t**2.5+3))}$")



# %%
ddphi1 = ddphi.subs(u_d | params | {ddnu: 0, nu: 120})
#%%
ddphi1
# print(latex(ddphi1))
A_num_true = lambdify([t,phi,dphi], ddphi1, 'numpy')
#%%

def pendulum_true(x, t, A_num=A_num_true):
    phi, dphi = x
    return dphi, A_num(t, phi, dphi)

def calc1(label):
    t = np.linspace(0, 10, 100000)
    y0 = np.array([3 * np.pi / 4, 0])
    ys = odeint(func=pendulum_true, y0=y0, t=t)
    pi2 = np.pi * 2
    md = lambda x: np.fmod(np.fmod(x, pi2) + pi2, pi2)
    md1 = lambda x: np.fmod(x, pi2)
    id = lambda x : x
    fig, ax = plt.subplots(figsize=(20,10))
    fig.suptitle(f"$\\nu=120$ in the {label}")
    f = id
    # ax.scatter(t, id(ys[:,0]), label="$\\varphi(t)$", s=3)
    ax.plot(t, f(ys[:,0]), label="$\\varphi(t)$")
    # plt.plot(t, ys[:,1], label="$\\dot{\\varphi}(t)$")
    ax.legend()
    plt.plot()

calc1(label="original nonlinear system")

# %%

# print(get_nu(0.1, 9.8, 0.01))
# %%
