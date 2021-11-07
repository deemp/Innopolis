import numpy as np
import sympy as sp
from sympy.core.symbol import symbols
from sympy.printing.rcode import print_rcode

link_lengths = [1,1,1,1,1,1]
fk_input = [0.1,0.1,0.1,0.1,0.1,0.1]

# def transoform_base(base_frame, q, flag):
#     sol = FK_solve(q, flag)

#     if flag == "ee":
#         return np.dot(base_frame, sol)
    
#     elif flag == "full":
#         return [np.dot(base_frame, i) for i in sol]


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

# def transform_base(base_frame, ts):
#     return [base_frame] + ts

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

def transform_ij_sym(l, i, j):
    x = sp.Identity(4)
    for k in range(i,j+1):
        x = x * l[k]
    return x

def FK_solve(q=fk_input, flag="ee"):
    # print(f"input:\n{q}")
    ts_num = [np.array(ts[i].subs(qs[i], q[i]).evalf()).astype(np.float64) for i in range(len(q))]

    for i in range(1, len(ts_num)):
        ts_num[i] = np.dot(ts_num[i-1], ts_num[i])

    if flag == "ee":
        return ts_num[-1]
    elif flag == "full":
        return ts_num

base_default = np.eye(4)
ee_default = FK_solve()

from sympy.printing import print_rcode, latex

def get_q123(a, b, c):
    def go(m23):
        q1 = np.arctan2(-a * m23, b * m23)
        s1 = np.sin(q1)
        if s1 != 0.:
            return [[q1] + i for i in go_m1(s1)]
        else:
            return [[q1] + i for i in go_m1_0()]

    def go_m1(s1):
        return go_m1_m3(s1, -1) + go_m1_m3(s1, 1)

    def go_m1_0():
        return go_m1_0_m3(-1) + go_m1_0_m3(1)
    
    def go_m1_m3(s1, m3):
        p = np.arctan2((c-1) * m3, -a / s1 * m3)
        q3 = np.arccos(-a/(2 * s1 * np.cos(p)))
        if np.cos(q3/2) * m3 > 0:
            q2 = (2 * p - q3)/2
            return [[q2, q3]]
        return []

    def go_m1_0_m3(m3):
        p = np.arctan2((c-1) * m3, b * m3)
        q3 = np.arccos(b/(2 * np.cos(p)))
        if np.cos(q3/2) * m3 > 0:
            q2 = (2 * p - q3)/2
            return [[q2, q3]]
        return []
    
    return go(1) + go(-1)

# 1T4 and 4Te symbolical representation
t14 = sp.simplify(transform_ij_sym(l=ts, i=0, j=2))
# print(latex(t14[:3,3]))
t4e = sp.simplify(transform_ij_sym(l=ts, i=3, j=5))
# print(latex(sp.simplify(t4e)))

# print(f"T14:\n{FK_solve(fk_input[:3])}")

def get_T4e_s(angles, ee):
    global q1, q2, q3
    T4e_s = [None for i in angles]
    for i, (a1, a2, a3) in enumerate(angles):
        subs = [(q1, a1), (q2, a2), (q3, a3)]
        T14 = np.array(sp.simplify(t14.subs(subs)).evalf()).astype(np.float64)
        T4e_s[i] = np.linalg.inv(T14).dot(ee)

    return T4e_s

def get_q(q123, W):
    def go_m5(m5):
        q6 = np.arctan2(-W[2,1] * m5, W[2,0] * m5)
        c6 = np.cos(q6)
        q4, q5 = None, None
        if c6 != 0:
            q4 = np.arctan2(-W[0,2] * m5, W[1,2] * m5)
            q5 = np.arctan2(W[2,2] * m5, W[2,0]/c6*m5)
        else:
            s6 = np.sin(q6)
            q4 = np.arctan2(W[0,2]/(-np.cos(q5)), W[1,2]/np.cos(q5))
            q5 = np.arctan2(W[2,2],W[2,1]/(-s6))
        return [q4, q5, q6]
    
    def go_m5_0():
        q4 = 0.5
        if W[2,2] == 1:
            q46 = np.arctan2(W[0,0], W[0,1])
            return [q4, np.pi/2, q46 - q4]
        if W[2,2] == -1:
            q64 = np.arctan2(W[0,0], W[0,1])
            return [q4, np.pi/2, q64 + q4]

    def go():
        ret = None
        if np.abs(W[2,2]) == 1:
            ret = [go_m5_0()]
        else:
            ret = [go_m5(-1), go_m5(1)]
        return [q123 + i for i in ret]

    return go()

def check_q(q1, q2, q3):
    a = -np.sin(q1)*(np.cos(q2)+np.cos(q2+q3))
    b = np.cos(q1)*(np.cos(q2)+np.cos(q2+q3))
    c = np.sin(q2) + np.sin(q2+q3) + 1
    print(a,b,c)

def ik(ee = ee_default, base_frame = base_default):
    # Step 1
    # move base
    ee = np.linalg.inv(base_frame).dot(ee)
    
    # print(f"FK:\n{ee}")

    # Step 2
    t14
    t4e
    
    # Step 3
    pc = ee[:3,3] - np.linalg.norm(ee[:3,3])*ee[:3,2]

    # Step 4
    print(f"Pc:{pc}")
    q123 = get_q123(*pc)
    
    for i in q123:
        check_q(*i)
    # Step 5
    T4e_s = get_T4e_s(q123, ee)
    
    # Step 6
    # qs = [k for i, j in zip(q123,T4e_s) for k in get_q(i, j)]
    qs = q123
    return qs
    
ik()

def check():
    qs = ik()
    for i,j in enumerate(qs):
        pass
        # print(f"{i}:\n{FK_solve(j[:3])[:3,3]}")
        # print(f"{i:}\n{FK_solve(j)}")
    
# check()