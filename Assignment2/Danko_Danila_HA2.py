import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp

from sympy.utilities.lambdify import lambdify

np.set_printoptions(precision=2, suppress=True)

# dimension of homogeneous matrices
dim_h = 4
n = 6
link_lengths = np.array([1,1,1,1,1,1])
q_final = np.array([2.5,1.8,3.14,4.6,2.1,1.1])
q_initial = np.array([0.,0.,0.,0.,0.,0.])

q = sp.Symbol('q')

def get_rot_symb():
    """
    produce dicts of rotation matrices
    """
    c = sp.cos(q)
    s = sp.sin(q)
    R = {
        "z": sp.Matrix([
                [c, -s, 0],
                [s, c, 0],
                [0, 0, 1]
            ]),
        "y": sp.Matrix([
                [c, 0, s],
                [0, 1, 0],
                [-s, 0, c]
            ]),
        "x": sp.Matrix([
                [1, 0, 0],
                [0, c, -s],
                [0, s, c]
            ]),
        }

    RH = {}

    # insert 3x3 rotation matrix into homogeneous 4x4
    for axis, rot33 in R.items():
        t = sp.Matrix(sp.Identity(4))
        t[:3,:3] = rot33
        RH[axis] = t
    
    return RH

rot_symb = get_rot_symb()

d = sp.Symbol('d')

def get_trans_symb():
    """
    produce dicts of translation matrices
    """
    T = {
        "z": sp.Matrix([
                [0],
                [0],
                [d]
            ]),
        "y": sp.Matrix([
                [0],
                [d],
                [0]
            ]),
        "x": sp.Matrix([
                [d],
                [0],
                [0]
            ]),
        }

    TH = {}
    # insert 3x1 translation matrix into homogeneous 4x4
    for axis, trans31 in T.items():
        t = sp.Matrix(sp.Identity(4))
        t[:3,3] = trans31
        TH[axis] = t
    
    return TH

trans_symb = get_trans_symb()

qs_symb = sp.symbols('q1 q2 q3 q4 q5 q6')
q1, q2, q3, q4, q5, q6 = qs_symb
l1, l2, l3, l4, l5, l6 = link_lengths

def get_T(args):
    """
    get symbolic transformation matrix depending on the type of transformation
    """
    if args["T"] == "R":
        return rot_symb[args["axis"]].subs(q, args["angle"])
    elif args["T"] == "T":
        return trans_symb[args["axis"]].subs(d, args["distance"])

def compose_T_symb(args):
    """
    compose a symbolic transformation from a list of transformations
    """
    t = [get_T(arg) for arg in args]
    for i in range(1, len(t)):
        t[i] = t[i-1] * t[i]
    return sp.simplify(t[-1])

def get_dh(use_different_qs=True):
    """
    produce a list of T_i_i+1 for DH parameterization
    """
    T12 = compose_T_symb([
        {"T": "R", "axis": "z", "angle": q1},
        {"T": "T", "axis": "z", "distance": l1},
        {"T": "T", "axis": "x", "distance": 0},
        {"T": "R", "axis": "x", "angle": sp.pi/2},
    ])
    T23 = compose_T_symb([
        {"T": "R", "axis": "z", "angle": q2},
        {"T": "T", "axis": "z", "distance": 0},
        {"T": "T", "axis": "x", "distance": l2},
        {"T": "R", "axis": "x", "angle": 0},
    ])
    T34 = compose_T_symb([
        {"T": "R", "axis": "z", "angle": q3},
        {"T": "T", "axis": "z", "distance": 0},
        {"T": "T", "axis": "x", "distance": l3},
        {"T": "R", "axis": "x", "angle": -sp.pi/2},
    ])
    T45 = compose_T_symb([
        {"T": "R", "axis": "z", "angle": q4},
        {"T": "T", "axis": "z", "distance": l4},
        {"T": "T", "axis": "x", "distance": 0},
        {"T": "R", "axis": "x", "angle": -sp.pi/2},
    ])
    T56 = compose_T_symb([
        {"T": "R", "axis": "z", "angle": q5 - sp.pi/2},
        {"T": "T", "axis": "z", "distance": 0},
        {"T": "T", "axis": "x", "distance": 0},
        {"T": "R", "axis": "x", "angle": - sp.pi/2},
    ])
    T6e = compose_T_symb([
        {"T": "R", "axis": "z", "angle": q6},
        {"T": "T", "axis": "z", "distance": l5+l6},
        {"T": "T", "axis": "x", "distance": 0},
        {"T": "R", "axis": "x", "angle": 0},
    ])
    ret = [T12, T23, T34, T45, T56, T6e]
    if not use_different_qs:
        ret = [i.subs([(j, q)]) for i,j in zip(ret, qs_symb)]
    return ret


# list of numeric functions for transitions T_i_i+1
ts_num = [lambdify(q, i, 'numpy') for i in get_dh(use_different_qs=False)]

def get_fk_symb():
    """
    symbolic forward kinematics solution
    """
    ts = get_dh(use_different_qs=True)
    fk_symb = [ts[0]]
    for i in range(n-1):
        fk_symb += [fk_symb[i] * ts[i+1]]
    return fk_symb

fk_symb = get_fk_symb()

def get_fk_solution(qs=q_final, flag="ee"):
    """
    numeric forward kinematics solution
    """
    ts = [t(q) for t,q in zip(ts_num, qs)]
    for i in range(n-1):
        ts[i+1] = ts[i].dot(ts[i+1])

    if flag == "ee":
        return ts[-1]
    elif flag == "full":
        return ts

eps = 0.00000000001

def eq(a,b):
    return np.abs(a-b) < eps

fk_full_default = get_fk_solution(qs=q_initial, flag="full")

base_default = np.eye(dim_h)
ee_default = fk_full_default[-1]

def jacobian(frames=fk_full_default):
    fks = [base_default] + frames
    ee = fks[-1]
    
    J = np.zeros((n,n))

    for i in range(6):
        z = fks[i][:3,2]
        o_diff = ee[:3,3] - fks[i][:3,3]
        J[:3,i] = np.cross(z, o_diff)
        J[3:,i] = z

    return J

def check_singular(jacobian):
    return eq(np.linalg.det(jacobian), 0.)

def cartesian_velocity(jacobian, q_dot):
    return jacobian.dot(q_dot)


from scipy.spatial.transform import Rotation as R
def decompose_transformation(W): 
    r = R.from_matrix(W[:3,:3]).as_euler('xyz')
    t = W[:3,3]
    x = np.zeros(6)
    x[:3] = t
    x[3:] = r
    return x
    

t0 = 0
tf = 500
k = 0.01

# print(decompose_transformation(get_fk_solution(q_initial)))
x_target_default = decompose_transformation(get_fk_solution(q_final))
# print(x_target_default)
# print("fk_final\n",get_fk_solution(q_final))

def solve_motion(y0=q_initial,t=(t0,tf), x1=x_target_default):
    print("x_targ:\n", x1)
    def state_space(t, y):
        fk = get_fk_solution(y, flag="full")
        J = jacobian(frames=fk)
        xi = decompose_transformation(fk[-1])
        dx = (x1-xi)*k
        return np.linalg.inv(J).dot(dx)
    return solve_ivp(fun=state_space, y0=y0, t_span=t)

sol = solve_motion()

t_len = sol.t.shape[0]
x_current = np.zeros((t_len,6))
for i in range(t_len):
    x_current[i,:] = decompose_transformation(get_fk_solution(sol.y.T[i]))
x_target = np.full(shape=(t_len,6), fill_value=x_target_default)


from utils import plot_sol
labels = ["x", "y", "z", "\\theta_x", "\\theta_y", "\\theta_z"]
target_labels = ["\\hat{" + f"{i}" + "}" for i in labels]

plot_sol([
    {
        "x": sol.t,
        "xlabel": "Time (s)",
        "ylabel": "Joint\\ angles (rad)",
        "title" : "Change\\ of\\ joint\\ angles\\ between\\ configurations",
        "graphs": [{"y": sol.y[i], "label": f"q_{i}"} for i in range(n)]
    },
    {
        "x": sol.t,
        "xlabel": "Time (s)",
        "ylabel": "Pose",
        "title" : f"Change\\ of\\ EE\\ pose\\ between\\ configurations $ \n target: {x_target_default} $\\ ",
        "graphs": 
            [{"y": x_current.T[i], "label": labels[i]} for i in range(n)] + []
            # [{"y": x_target.T[i], "label": target_labels[i]} for i in range(n)]
    }
])