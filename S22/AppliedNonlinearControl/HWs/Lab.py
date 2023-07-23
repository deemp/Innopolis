#%%
import sympy as sp

t = sp.symbols('t')
x1 = sp.Function('x_1')
x2 = sp.Function('x_2')
def dx2(t): return -x1(t) + 1 / 3 * x1(t) ** 3 - x2(t)
def dx1(t): return dx2(t)
def V(t) : return 3/2 * x1(t) ** 2 - x1(t)**4/6 + x1(t) * x2(t) + x2(t) ** 2
Vd = V(t).subs([
    (sp.Derivative(x1(t), t), dx1(t)), 
    (sp.Derivative(x2(t), t), dx2(t))
    ])
print (Vd)
# %%
