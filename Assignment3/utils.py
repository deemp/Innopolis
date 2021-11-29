import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sympy as sp
from scipy.spatial.transform import Rotation as R
from sympy.utilities.lambdify import lambdify




#############################################
# Plotting
#############################################





font = {'family' : 'DejaVu Sans',
        'weight' : 'bold',
        'size'   : 15}

matplotlib.rc('font', **font)

def add_graph(ax=None, x=None, y=None, color=None, label=None):
    """add a graph to axis"""
    ax.plot(x, y, color=color, label=(f"{label}" if label else None))

def plot_graphs(ax=None, x=None, xlabel=None, ylabel=None, title=None, graphs=None):
    """
    plot several graphs in a grid
    """
    ax.grid()
    # enable latex
    ax.set(
        xlabel=(f"{xlabel}" if xlabel else None),
        ylabel=(f"{ylabel}" if ylabel else None),
    )

    # plot graphs
    for graph in graphs:
        add_graph(ax=ax, x=x, **graph)

    # there might be no label
    _, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(loc="best")

    if title:
        ax.set_title(f"{title}")

def plot_list(args):
    """plot separately graphs in a list

    -------
    ### Parameters
        `args` : `[dict]`
            `x` : `np.array`
            `xlabel` : `str`
                latex string
            `ylabel` : `str`
                latex string
            `title` : `str`
                latex string
            `graphs` : `[dict]`
                `y` : `np.array`
                `color` : `str`
                    see [options](https://matplotlib.org/stable/gallery/color/named_colors.html)
                `label` : `str`
                    latex string
    -------
    ### Returns
      `None`

    -------
    ### Examples
    """

    # plots will be aligned into a line
    fig, ax = plt.subplots(1, len(args), figsize=(6 * len(args), 6))

    # convert to iterable
    ax = np.array(ax).reshape(-1)

    # set padding between plots
    fig.tight_layout(pad=4.0)

    for i, arg in enumerate(args):
        plot_graphs(ax[i], **arg)

    plt.show()





#############################################
# Manipulator
#############################################





dim_h = 4 
"""dimensions of a homogeneous matrix"""

n = 6
"""number of joints"""

q = sp.Symbol('q')

def get_rot_symb():
    """
    produce dict of rotation matrices
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
rot_num = {i: lambdify(q, rot_symb[i], 'numpy') for i in rot_symb.keys()}

d = sp.Symbol('d')

def get_trans_symb():
    """
    produce dict of translation matrices
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
trans_num = {i: lambdify(d, trans_symb[i], 'numpy') for i in trans_symb.keys()}

qs_symb = sp.symbols('q1 q2 q3 q4 q5 q6')
q1, q2, q3, q4, q5, q6 = qs_symb

link_lengths = np.array([1,1,1,1,1,1])
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


ts_num = [lambdify(q, i, 'numpy') for i in get_dh(use_different_qs=False)]
"""list of numeric functions for transitions T_i_i+1""" 

def get_fk_solution(qs, flag="ee"):
    """
    numeric forward kinematics solution
    
    considers |qs| first links
    """
    ts = [t(q) for t,q in zip(ts_num, qs)]
    for i in range(len(qs)-1):
        ts[i+1] = ts[i].dot(ts[i+1])

    if flag == "ee":
        return ts[-1]
    elif flag == "full":
        return ts


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
"""list of symbolic transitions T_0_i for links i"""

eps = 0.00000000001
"""epsilon used in comparisons"""


def eq(a,b):
    """check equality within eps"""
    return np.abs(a-b) < eps


def decompose_transformation(W): 
    """decompose a rotation matrix into XYZ euler angles"""
    r = R.from_matrix(W[:3,:3]).as_euler('xyz')
    t = W[:3,3]
    x = np.zeros(6)
    x[:3] = t
    x[3:] = r
    return x

t14 = fk_symb[3]

base_default = np.eye(4)
fk_input_init = np.zeros(6)
ee_default = get_fk_solution(fk_input_init)

def eq(a,b):
    return np.abs(a-b) < eps

def get_R4e_s(angles, ee):
    T4e_s = [None for _ in angles]
    for i, qs in enumerate(angles):
        T14 = get_fk_solution(qs)
        T4e_s[i] = np.linalg.inv(T14).dot(ee)[:3,:3]

    return T4e_s

def get_q(q123, W):
    """
    get all full sets of angles given the angles q1, q2, q3 and matrix W
    """
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

def transform_base(trans=base_default,q=fk_input_init,flag="ee"):
    return trans.dot(get_fk_solution(q,flag))

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


def eq_matrix(a, b):
    """
    compare if matrices are equal with eps precision
    """
    return np.abs(np.sum((a-b).flatten())) < eps

ts_dh = get_dh()

def filter_sols(qs=[fk_input_init],ts=ts_dh, ee=ee_default):
    """
    select only solutions that lead to the actual fk
    """
    t = []
    for q in qs:
        if eq_matrix(get_fk_solution(qs=q), ee):
            t += [q]
    return t

def IK_solve(ee=ee_default, base_frame = base_default, ts=ts_dh):
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


def check_IK(q=fk_input_init):
    """
    print IK solutions
    """
    ee = get_fk_solution(qs=q)
    qs = IK_solve(ee=ee)

    print(len(qs))


def decompose_transformation(W): 
    """decompose a rotation matrix into translation xyz and XYZ euler angles"""
    r = R.from_matrix(W[:3,:3]).as_euler('xyz')
    t = W[:3,3]
    x = np.zeros(6)
    x[:3] = t
    x[3:] = r
    return x
    
def t_from_pose(pose):
    """
    construct homogeneous transformation matrix for a given pose
    """
    x,y,z,a,b,c = pose

    W = np.linalg.multi_dot([
        trans_num['x'](x),
        trans_num['y'](y),
        trans_num['z'](z),

        rot_num['x'](a),
        rot_num['y'](b),
        rot_num['z'](c),
        ])
    
    return W

def get_ik_from_pose(pose):
    return IK_solve(t_from_pose(pose))

# print(t_from_pose([3,4,3,0,0,0]))