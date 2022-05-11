
#%%

# Task 1

from sympy import *

t = symbols('t')
x1, x2 = symbols('{x}_{1}, {x}_{2}', real=True, cls=Function)
x_1 = x1(t)
x_2 = x2(t)
dx1t, dx2t = symbols('\\dot{x}_{1}, \\dot{x}_{2}', real = True)


V = x_1 ** 2 / (1 + x_1 ** 2) ** 2 + x_2 ** 2
dV = V.diff(t)
dv = dV.subs([(Derivative(x_1, t), dx1t), (Derivative(x_2, t), dx2t)])
dv
# print(latex(dV))

#%%

# Task 2

t, c = symbols('t c')
x1, x2 = symbols('x_{1}, x_{2}', real=True, cls=Function)

x_1 = x1(t)
x_2 = x2(t)
dx1 = x_1 * (x_1 ** 2 + x_2 ** 2 - c) - 4 * x_1 * x_2 ** 2
dx2 = 4 * x_1 ** 2 * x_2 + x_2 * (x_1 ** 2 + x_2 ** 2 - c)
x1t, x2t = symbols('x_{1}, x_{2}')

V = x1(t) ** 2  + x2(t) ** 2
dV = V.diff(t)
dv = simplify(dV.subs([(Derivative(x_1,t), dx1), (Derivative(x_2,t), dx2)]))
dv = factor(dv.subs([(x_1, x1t), (x_2, x2t)]))
dv
print(latex(dv))
# %%

t, Hd, du, k = symbols('t H_{d} \\dot{u} k')
theta, H, u, m = symbols('\\theta H u m', cls=Function)
dtheta = Derivative(theta(t))
ddtheta = Derivative(dtheta,t)
dtheta = u(t) - sin(theta(t)) - ddtheta
Htilde = Hd - H(t)
V = Rational(1,2) * Htilde ** 2
Hsub = Rational(1,2) * dtheta ** 2 + 1 - cos(theta(t))
dv = V.diff(t)
# dv
dv = dv.subs([(Derivative(H(t), t), Derivative(Hsub, t)),(H(t),Hsub)])
dv = simplify(dv)
dv

th = sqrt(2 * Hd) * tanh(k * theta(t))

usub = ddtheta + sin(theta(t)) - th
dv = simplify(dv.subs([(u(t), usub)]))
factor(dv)

dthetasub = - th
dv = dv.subs([(Derivative(theta(t), t), dthetasub)])
dv
thetasub = symbols("\\theta")
dv = factor(simplify(dv.subs([(theta(t), thetasub)])))
print(latex(dv))
dv
# print(latex(dv))
# dv = factor(simplify(dv))

# print(latex(dv))
# dv
# %%
# %%
