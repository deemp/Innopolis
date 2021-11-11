import re
import numpy as np
import sympy as sp
from sympy.core.symbol import symbols
from sympy.printing.rcode import print_rcode

link_lengths = [1,1,1,1,1,1]
fk_input = [0.1,0.1,0.1,0.1,0.1,0.1]
fk_input_init = np.zeros(6)
q = sp.Symbol('q')
c = sp.cos(q)
s = sp.sin(q)

def get_rot_symb():
    """
    produce dicts of rotation matrices
    """
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

symb_qs = sp.symbols('q1 q2 q3 q4 q5 q6')
q1, q2, q3, q4, q5, q6 = symb_qs
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

def get_fk_dh():
    """
    produce a list of Tii+1 for DH parameterization
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

    return [T12, T23, T34, T45, T56, T6e]

ts_dh = get_fk_dh()

def transform_ij_symb(i=0, j=5, l=ts_dh):
    x = sp.Identity(4)
    for k in range(i,j+1):
        x = x * l[k]
    return x

def compose_T_num(T):
    return np.array(compose_T_symb(T).evalf()).astype(np.float64)

def FK_solve(qs=fk_input, flag="ee", ts=ts_dh, symb_qs=symb_qs):
    ts_num = [np.array(ts[i].subs(symb_qs[i], qs[i]).evalf()).astype(np.float64) for i in range(len(qs))]

    for i in range(1, len(ts_num)):
        ts_num[i] = np.dot(ts_num[i-1], ts_num[i])

    if flag == "ee":
        return ts_num[-1]

    elif flag == "full":
        return ts_num

base_default = np.eye(4)
ee_default = FK_solve()

from sympy.printing import latex

eps = 0.000000001

# T14 symbolic representation
t14 = sp.simplify(transform_ij_symb(l=ts_dh, i=0, j=2))
# T4e symbolic representation
t4e = sp.simplify(transform_ij_symb(l=ts_dh, i=3, j=5))

def eq(a,b):
    return np.abs(a-b) < eps

def get_R4e_s(angles, ee):
    global q1, q2, q3
    T4e_s = [None for _ in angles]
    for i, (a1, a2, a3) in enumerate(angles):
        subs = [(q1, a1), (q2, a2), (q3, a3)]
        T14 = np.array(t14.subs(subs).evalf()).astype(np.float64)
        T4e_s[i] = np.linalg.inv(T14).dot(ee)[:3,:3]

    return T4e_s

def get_q(q123, W):
    def go_m5(m5):
        q6 = np.arctan2(-W[2,1] * m5, W[2,0] * m5)
        c6 = np.cos(q6)
        q4, q5 = None, None
        if not eq(c6,0):
            q5 = np.arctan2(-W[2,2], W[2,0]/c6)
            c5 = np.cos(q5)
            q4 = np.arctan2(W[1,2]/c5, W[0,2]/c5)
        else:
            s6 = np.sin(q6)
            q5 = np.arctan2(-W[2,2],-W[2,1]/s6)
            c5 = np.cos(q5)
            q4 = np.arctan2(W[1,2]/c5, W[0,2]/c5)
        return [q4, q5, q6]

    def go():
        ret = [go_m5(-1), go_m5(1)]
        return [q123 + i for i in ret]

    return go()

def transform_base(trans=base_default,q=fk_input,flag="ee"):
    return trans.dot(FK_solve(q,flag))

def get_q123(a,b,c):
    
    def get_q23(a,b):
        # https://www.wolframalpha.com/input/?i=cos%28x%29%2Bcos%28x%2By%29-sin%28x%2By%29%3Da%2Csin%28x%29%2Bsin%28x%2By%29%2Bcos%28x%2By%29%3Db
        t = []
        t1 = 1-a**2
        if t1 >= 0:
            if not eq(a,0) and eq(b, -np.sqrt(t1)):
                x = 2 * np.arctan((b-1)/a)
                y = np.pi/2
                t += [[x,y]]
            if not eq(a,0) and eq(b, -np.sqrt(t1)):
                x = 2 * np.arctan((1-b)/a)
                y = np.pi
                t += [[x,y]]
            if not eq(a,0) and eq(b, np.sqrt(t1)):
                x = 2 * np.arctan((b-1)/a)
                y = np.pi/2
                t += [[x,y]]
            if not eq(a,0) and eq(b, np.sqrt(t1)):
                x = 2 * np.arctan((b+1)/a)
                y = np.pi
                t += [[x,y]]
        
        t1 = -a**2-2*a+1
        if t1 >= 0:
            if not eq(a,0) and eq(b, -np.sqrt(t1)):
                x = np.pi
                y = 2 * np.arctan((1+b)/a)
                t += [[x,y]]
            if not eq(a,0) and eq(b, np.sqrt(t1)):
                x = np.pi
                y = 2 * np.arctan((1+b)/a)
                t += [[x,y]]
        
        t5 = -a**4-2*a**2*b**2+6*a**2-b**4+6*b**2-1
        if t5 >= 0:
            t1 = np.sqrt(t5)
            t2 = a**3+2*a**2+a*b**2-a+2*b**2
            t3 = a**2+2*a+b**2-1
            t4 = a**2 + b**2 -1
            if not eq(t3,0) and not eq(t4,0) and not eq(t2-b*t1,0):
                x = 2 * np.arctan((2*b-t1)/t3)
                y = 2 * np.arctan((t1-2)/t4)
                t += [[x,y]]
            
            if not eq(t3,0) and not eq(t4, 0) and not eq(t2+b*t1,0):
                x = 2 * np.arctan((2*b+t1)/t3)
                y = 2 * np.arctan((-t1-2)/t4)
                t += [[x,y]]


        t3 = a**2+2*a-1
        if -t3 >= 0:
            t1 = 2*a**3+3*a**2-4*a+1
            t2 = np.sqrt(-t3)
            if not eq(t3,0) and not eq(a,0) \
                and not eq(t1,0) and eq(b,-t2):
                x = -2 * np.arctan((a*b)/t3)
                y = 2 * np.arctan((1-b)/a)
                t += [[x,y]]

            if not eq(t3,0) and not eq(a,0) \
                and not eq(t1,0) and eq(b,t2):
                x = 2 * np.arctan((a*b)/t3)
                y = 2 * np.arctan((1-b)/a)
                t += [[x,y]]

        return t


    def go_q1(m1, a, b, c):
        q1 = np.arctan2(b * m1, a * m1)
        c1 = np.cos(q1)
        s1 = np.sin(q1)
        t = []
        if not eq(c1,0):
            t = get_q23(a/c1, c-1)
        else:
            t = get_q23(b/s1, c-1)
        return [[q1]+i for i in t]

    t = go_q1(-1,a,b,c) + go_q1(1,a,b,c)
    return t

def filter_sols(qs=[fk_input],ts=ts_dh, ee=ee_default):
    t = []
    for q in qs:
        if eq_matrix(FK_solve(qs=q, ts=ts), ee):
            t += [q]
    return t


def IK_solve(ee = ee_default, base_frame = base_default, ts=ts_dh):
    """
    get a list of solutions to IK problem for a given base frame and end effector frame
    """
    # Step 1
    # move base to origin
    ee = np.linalg.inv(base_frame).dot(ee)
    
    # Step 2
    
    # Step 3
    pc = ee[:3,3] - (l5+l6)*ee[:3,2]

    # Step 4
    q123 = get_q123(*pc)
    
    # Step 5
    R4e_s = get_R4e_s(q123, ee)

    # Step 6
    qs = [k for i, j in zip(q123,R4e_s) for k in get_q(i, j)]
    qs = filter_sols(qs=qs, ee=ee, ts=ts)

    return qs


import matplotlib.pyplot as plt
def plot_manipulator(qs, ts=ts_dh):
    """
    plot a manipulator with given joint angles
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlabel('$X$')
    ax.set_ylabel('$Y$')
    ax.set_xlim([-4,4])
    ax.set_ylim([-4,4])
    ax.set_zlim([0,5])

    for q in qs:
        t = np.array([np.zeros(3)] + [i[:3,3] for i in FK_solve(qs=q, flag="full",ts=ts)])
        ax.plot3D(t[:,0], t[:,1], t[:,2])

    plt.show()

def latex_r_ij(l=ts_dh,i=0,j=3):
    """
    print Latex representation of symbolic Tij
    """
    print(latex(sp.simplify(transform_ij_symb(l=l, i=i, j=j)[:3,:3])))

def num_ij(l=ts_dh, i=0, j=3, qs=symb_qs, q=fk_input):
    """
    print numeric representation of Tij
    """
    subs = [(i,j) for i,j in zip(qs, q)]
    print(np.array(transform_ij_symb(l=ts_dh,i=i,j=j).subs(subs)).astype(np.float64))

def eq_matrix(a, b):
    """
    compare if matrices are equal with eps precision
    """
    return np.abs(np.sum((a-b).flatten())) < eps

def check_IK(q=fk_input, r=len(fk_input), ts=ts_dh):
    """
    print IK solutions
    """
    ee = FK_solve(qs=q, ts=ts)
    qs = IK_solve(ee=ee)

    fk = FK_solve(qs=q[:r])
    print(f"FK:\n{fk}")
    for i,j in enumerate(qs):
        print(f"{i}:{j[:r]}\n{transform_base(q=j[:r])}")

def get_sols(ts=ts_dh,qs=fk_input):
    """
    returns a list of IK solutions for a given set of joint angles
    """
    t = IK_solve(ee=FK_solve(qs=qs,ts=ts),ts=ts)
    return t

def plot_umbrella(rng = 2*np.pi, n=10):
    """
    make funny umbrellas via rotating the first joint
    """
    ps = [0. for _ in range(5)]
    sols = []
    for i in range(n):
        step = rng / n
        sols += get_sols(qs=[i * step] + ps)
    plot_manipulator(sols)


def get_transformation_parts(W):
    x, y, z = sp.symbols("x y z")
    Rx = compose_T_symb([{"T": "R", "axis": "x", "angle": q1},])
    Ry = compose_T_symb([{"T": "R", "axis": "y", "angle": q2},])
    Rz = compose_T_symb([{"T": "R", "axis": "z", "angle": q3},])

    Rot = Rx * Ry * Rz

    Trans = compose_T_symb([
        {"T": "T", "axis": "x", "distance": x},
        {"T": "T", "axis": "y", "distance": y},
        {"T": "T", "axis": "z", "distance": z},
    ])

    T = Trans * Rot

    # print(latex(T[:3,:3]))

    def go(m2):
        if eq(np.abs(W[0,2]),0):
            return [[]]
        q3 = np.arctan2(-W[0,1]*m2, W[0,0]*m2)
        c3 = np.cos(q3)
        if not eq(c3, 0):
            q2 = np.arctan2(W[0,2], W[0,0]/c3)
            c2 = np.cos(q2)
            q1 = np.arctan2(-W[1,2]/c2, W[2,2]/c2)
            return [[q1, q2, q3]]
        else:
            s3 = np.sin(q3)
            q2 = np.arctan2(W[0,2], W[0,1]/-s3)
            c2 = np.cos(q2)
            q1 = np.arctan2(-W[1,2]/c2, W[2,2]/c2)
            return [[q1, q2, q3]]

    q123 = go(-1) + go(1)

    def num_matrix(T,q,p):
        return np.array(T.subs(q,p).evalf()).astype(np.float64)

    ret = []
    for p1,p2,p3 in q123:
        subs = [(q1, p1), (q2, p2), (q3, p3)]
        R = np.array(Rot.subs(subs).evalf()).astype(np.float64)
        ret += [(W[0,3], W[1,3], W[2,3], num_matrix(Rx, q1, p1), num_matrix(Ry,q2,p2), num_matrix(Rz,q3,p3))]
    
    for x,y,z,rx,ry,rz in ret:
        r = np.linalg.multi_dot([rx,ry,rz])
        r[:3,3] = np.array([x,y,z])
        print(r)
    
    return ret

T = compose_T_num([
    {"T": "T", "axis": "x", "distance": 1},
    {"T": "T", "axis": "y", "distance": 2},
    {"T": "T", "axis": "z", "distance": 3},
    {"T": "R", "axis": "x", "angle": 0.7},
    {"T": "R", "axis": "y", "angle": 1.},
    {"T": "R", "axis": "z", "angle": 0.5},
])
# print(T)

# check_IK()
# get_transformation_parts(T)
# plot_umbrella(n=30)