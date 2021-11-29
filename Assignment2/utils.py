import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from scipy.spatial.transform import Rotation as R
from sympy.utilities.lambdify import lambdify

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
        xlabel=(f"${xlabel}$" if xlabel else None),
        ylabel=(f"${ylabel}$" if ylabel else None),
    )

    # plot graphs
    for graph in graphs:
        add_graph(ax=ax, x=x, **graph)

    # there might be no label
    _, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(loc="best")

    if title:
        ax.set_title(f"${title}$")

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
    """
    ts = [t(q) for t,q in zip(ts_num, qs)]
    for i in range(n-1):
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
    """decompose a rotation matrix into translation xyz and XYZ euler angles"""
    r = R.from_matrix(W[:3,:3]).as_euler('xyz')
    t = W[:3,3]
    x = np.zeros(6)
    x[:3] = t
    x[3:] = r
    return x
    