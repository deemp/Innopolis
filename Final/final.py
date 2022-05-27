# %%

from cProfile import label
from sympy import *

t, a, l1, l2, w, M1, M2, g = symbols('t a l_{1} l_{2} \\omega M_{1} M_{2} g')
phi_s, dphi_s, ddphi_s = symbols('\\varphi \\dot{\\varphi} \\ddot{\\varphi}',real=True,cls=Function)
phi = phi_s(t)
dphi = dphi_s(t)
ddphi = ddphi_s(t)


x1 = Rational(0)
y1 = - a * cos(w * t) - l1

x2 = l2 * sin(phi)
y2 = - a * cos(w * t) - l2 * cos(phi)

dx1 = x1.diff(t)
dy1 = y1.diff(t)

dx2 = x2.diff(t)
dy2 = y2.diff(t)

Epot1 = M1 * y1 * g
Epot2 = M2 * y2 * g

Ekin1 = Rational(1, 2) * M1 * (dx1 ** 2 + dy1** 2)
Ekin2 = Rational(1, 2) * M2 * (dx2 ** 2 + dy2 ** 2)

L = Ekin1 + Ekin2 - (Epot1 + Epot2)
L = L.subs([(Derivative(phi,t), dphi)]).expand().simplify()
L
# print(latex(L))

# %%
e1 = L.diff(dphi)
print(latex(e1))
# %%
e2 = L.diff(phi)
print(latex(e2))
# %%
e3 = e1.diff(t).subs({Derivative(phi,t): dphi,Derivative(dphi,t): ddphi}).expand().simplify()
print(latex(e3))
# %%

e4 = solve((e3 - e2).simplify(), ddphi)[0].simplify().expand()
ddphi_sym = e4
ddphi_sym
# print(latex(e4))
#%%

x_sym = phi, dphi
# %%
f_sym = Matrix([dphi, e4])
f_sym
# %%

Jx_sym = f_sym.jacobian(x_sym)
Jx_sym
# %%

Ju_sym = f_sym.jacobian([w])
Ju_sym
# %%

# nu, amplitude
u_d = {w: 100, a: 0.01}
x_d = {phi: pi, dphi: 0}
subs = u_d | x_d

#%%

A_sym = Jx_sym.subs(subs)
A_sym
# %%

B_sym = Ju_sym.subs(subs)
B_sym


#%%


import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def pendulum_linearized(x, t, A_num):
    return A_num(t).dot(x)

def calc(sys, A_num):
    t = np.linspace(0, 10, 100000)
    y0 = np.array([np.pi, 0])
    ys = odeint(func=sys, y0=y0, t=t, args=(A_num,))
    pi2 = np.pi * 2
    md = lambda x: np.fmod(np.fmod(x, pi2) + pi2, pi2)
    md1 = lambda x: np.fmod(x, pi2)
    fig, ax = plt.subplots(figsize=(20,10))
    ax.scatter(t, md(ys[:,0]), label="$\\varphi(t)$", s=3)
    # ax.plot(t, md(ys[:,0]), label="$\\varphi(t)$")
    # plt.plot(t, ys[:,1], label="$\\dot{\\varphi}(t)$")
    ax.legend()
    plt.plot()



l2g = {l2: 0.01, g: 9.8}
#%%
A_num = lambdify(t, A_sym.subs(l2g), 'numpy')
# calc(sys=pendulum_linearized, A_num=A_num)

# %%
ddphi_sym.subs(u_d)

#%%

def pendulum(x, t, A_num):
    phi, dphi = x
    return dphi, A_num(t, phi) - dphi

# def calc1(l2_, g_):
#     t = np.linspace(0, 5, 300)
#     y0 = np.array([3 * np.pi / 2, 0.1])
#     ddphi_num = lambdify(phi, ddphi_sym.subs(u_d | {l2:l2_, g:g_}), 'numpy')
#     ys = odeint(func=sys, y0=y0, t=t, args=(A_num,))
#     plt.plot(t, ys)

def get_nu(l, g, a):
    return (sqrt(2 * g * l / a ** 2) + 1)

def get_a(l, g, nu):
    return sqrt(2 * g * l / nu ** 2) + 0.001

u_d = {a:0.1, w:get_nu(l=l2g[l2], g=l2g[g], a=0.01),}
ddphi_num = lambdify([t,phi], ddphi_sym.subs(u_d | l2g), 'numpy')
# ddphi_num(3)
ddphi_sym.subs(u_d | l2g)
calc(sys=pendulum, A_num=ddphi_num)

# %%

print(get_nu(0.1, 9.8, 0.01))
# %%
