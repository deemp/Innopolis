import numpy as np
import sympy as sp
from sympy.core.symbol import symbols

link_lengths = [1,1,1,1,1,1]
fk_input = [0.1,0.1,0.1,0.1,0.1,0.1]

def transoform_base(base_frame, q, flag):
    sol = FK_solve(q, flag)

    if flag == "ee":
        return np.dot(base_frame, sol)
    
    elif flag == "full":
        return [np.dot(base_frame, i) for i in sol]


# rotation matrices
q = sp.Symbol('q')
c = sp.cos(q)
s = sp.sin(q)
rot_symb = {
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

# insert 3x3 rotation matrix into homogeneous 4x4
for axis, rot33 in rot_symb.items():
    T = sp.Matrix(sp.Identity(4))
    T[:3,:3] = rot_symb[axis]
    rot_symb[axis] = T

# translation matrices
d = sp.Symbol('d')
trans_symb = {
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

# insert 3x1 translation matrix into homogeneous 4x4
for axis, trans31 in trans_symb.items():
    T = sp.Matrix(sp.Identity(4))
    T[:3,3] = trans_symb[axis]
    trans_symb[axis] = T

qs = sp.symbols('q1 q2 q3 q4 q5 q6')
q1, q2, q3, q4, q5, q6 = qs
l1, l2, l3, l4, l5, l6 = link_lengths

def transform_base(base_frame, ts):
    return [base_frame] + ts

def rotate_symbolic(axis, angle):
    return rot_symb[axis].subs(q, angle)

def translate_symbolic(axis, distance):
    return trans_symb[axis].subs(d, distance)

def get_T(args):
    if args["T"] == "R":
        args.pop("T", None)
        return rotate_symbolic(**args)
    elif args["T"] == "T":
        args.pop("T", None)
        return translate_symbolic(**args)

def compose_T(args):
    t = [get_T(arg) for arg in args]
    for i in range(1, len(t)):
        t[i] = t[i-1] * t[i]
    return t[-1]

def fk_symb():
    T12 = compose_T([
        {"T": "R", "axis": "z", "angle": q1 + sp.pi/2},
        {"T": "R", "axis": "x", "angle": sp.pi/2},
        {"T": "T", "axis": "y", "distance": l1}
    ])
    T23 = compose_T([
        {"T": "R", "axis": "z", "angle": q2},
        {"T": "T", "axis": "x", "distance": l2}
    ])
    T34 = compose_T([
        {"T": "R", "axis": "z", "angle": q3 - sp.pi/2},
        {"T": "R", "axis": "y", "angle": -sp.pi/2},
        {"T": "T", "axis": "y", "distance": l3}
    ])
    T45 = compose_T([
        {"T": "R", "axis": "z", "angle": q4 + sp.pi/2},
        {"T": "R", "axis": "x", "angle": sp.pi/2},
        {"T": "T", "axis": "y", "distance": l4}
    ])
    T56 = compose_T([
        {"T": "R", "axis": "z", "angle": q5 + sp.pi/2},
        {"T": "R", "axis": "x", "angle": sp.pi/2},
        {"T": "T", "axis": "z", "distance": l5}
    ])
    T6e = compose_T([
        {"T": "R", "axis": "z", "angle": q6},
        {"T": "T", "axis": "z", "distance": l6}
    ])
    return [T12, T23, T34, T45, T56, T6e]

ts = fk_symb()

def fk_dh_symb():
    T12 = compose_T([
        {"T": "R", "axis": "z", "angle": q1 + sp.pi/2},
        {"T": "T", "axis": "z", "distance": l1},
        {"T": "R", "axis": "x", "angle": sp.pi/2}
    ])
    T23 = compose_T([
        {"T": "R", "axis": "z", "angle": q2},
        {"T": "T", "axis": "x", "distance": l2}
    ])
    T34 = compose_T([
        {"T": "R", "axis": "z", "angle": q3},
        {"T": "T", "axis": "x", "distance": l3},
        {"T": "R", "axis": "x", "angle": -sp.pi/2}
    ])
    T45 = compose_T([
        {"T": "R", "axis": "z", "angle": q4},
        {"T": "T", "axis": "z", "distance": l4},
        {"T": "R", "axis": "x", "angle": sp.pi/2}
    ])
    T56 = compose_T([
        {"T": "R", "axis": "z", "angle": q5 + sp.pi/2},
        {"T": "R", "axis": "x", "angle": sp.pi/2},
    ])
    T6e = compose_T([
        {"T": "R", "axis": "z", "angle": q6},
        {"T": "T", "axis": "z", "distance": l5+l6}
    ])
    return [T12, T23, T34, T45, T56, T6e]

ts_dh = fk_dh_symb()

def transform_ij_sym(l, i, j):
    x = sp.Identity(4)
    for k in range(i,j+1):
        x = x * l[k]
    return x

def FK_solve(q=fk_input, flag="ee"):
    ts_num = [np.array(ts[i].subs(qs[i], q[i]).evalf()).astype(np.float64) for i in range(len(fk_input))]

    for i in range(1, len(ts_num)):
        ts_num[i] = np.dot(ts_num[i-1], ts_num[i])
    if flag == "ee":
        return ts_num[-1]
    elif flag == "full":
        return ts_num

base_default = sp.Identity(4)
ee_default = FK_solve()

# sp.pprint(transform_ij_sym(l=ts, i=0, j=2), num_columns = 180)
from scipy.optimize import fsolve

# def f14(x):
#     return [
#         -np.sin(x[0])*np.sin(x[1])*np.sin(x[2]) + np.sin(x[0])*np.cos(x[1])*np.cos(x[2]) + np.sin(x[0])*np.cos(x[1]) + 0.339421405717406,
#         np.sin(x[1])*np.sin(x[2])*np.cos(x[0]) - np.cos(x[0])*np.cos(x[1])*np.cos(x[2]) - np.cos(x[0])*np.cos(x[1]) - 0.728718482386846,
#         -np.sin(x[1])*np.cos(x[2]) - np.sin(x[1]) - np.sin(x[2])*np.cos(x[1]) + 0.492903383442055
#     ]
# root = fsolve(f14, [0.2,0.3,0.9])
# print(root)

def ik(ee = ee_default, base_frame = base_default):
    # ee = np.linalg.inv(base_default).dot(ee)
    # T14
    x,y,z = sp.symbols('x y z')
    t14 = transform_ij_sym(l=ts, i=0, j=2)
    # .subs([(q1,x),(q2,y),(q3,z)])
    # t4e = transform_ij_sym(l=ts, i=3, j=5)
    pc = ee[:3,3] - np.linalg.norm(ee[:3,3])*ee[:3,2]
    t14_transl = sp.transpose(t14[:3,3])
    eqs = [(pc[i]-t14_transl[i]) for i in range(3)]
    # print(eqs)
    q123 = sp.solve(eqs, [q2, q3])
    print(q123)

    # print(q123)

# for i in range(4):
#     for j in range(4):
#         print(f"T_{i}_{j}: {t13[i,j]}")

ik()

# def IK_solve(base_frame=base_default, ee_frame=ee_default):
    # """Inverse kinematics solver

    # ----------
    # ### Parameters
    #     `base_frame` : `np.array`
    #         4x4 homogeneous transformation matrix representing robot base
    #     `ee_frame` : `np.array`
    #         4x4 homogeneous transformation matrix representing end effector pose

    # -------
    # ### Returns
    #     `[np.array]`
    #         list of all possible solutions
    # """
    # T04 = fk_symb_ij(0, 3)[-1]
    # T4e = fk_symb_ij(4, 6)[-1]
    # T0e = T04 * T4e
    
    # N = CoordSys3D('N')
    # v1 = T0e[0,3] * N.i + T0e[1,3] * N.j + T0e[2,3] * N.k
    # v2 = T0e[0,2] * N.i + T0e[1,2] * N.j + T0e[2,2] * N.k
    # Pc = v1 - v1.cross(v2)

    # system = [
    #     Pc.dot(N.i) - T04[0,3], 
    #     Pc.dot(N.j) - T04[1,3], 
    #     Pc.dot(N.k) - T04[2,3], 
    # ]

    # print(system)

    # kinem_sol = nonlinsolve([
    #     Pc.dot(N.i) - T04[0,3], 
    #     Pc.dot(N.j) - T04[1,3], 
    #     Pc.dot(N.k) - T04[2,3], 
    # ],
    # [
    #     q1, q2, q3
    # ]
    # )
    
    # print(kinem_sol)
    # wrist_sol = nonlinsolve([

    # ])


# # IK_solve()

# print(fk_symb_ij(0,3)[-1])


    
    

# # def select_similar():
# #     # selects the ik solution that is the most similar to given input
# #     pass

# # def check_solvers(base_frame, ee_frame):
# #     # solve IK
# #     # solve FK with base frame for each IK solution
# #     pass