import numpy as np
from sympy.core.symbol import Symbol

link_lengths = [1,1,1,1,1,1]
fk_input = [0,0,0,0,0,0]

def transoform_base(base_frame, q, flag):
    sol = FK_solve(q, flag)

    if flag == "ee":
        return np.dot(base_frame, sol)
    
    elif flag == "full":
        return [np.dot(base_frame, i) for i in sol]

import sympy as sp

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

def rot_symb_q(axis: str, theta: sp.Symbol):
    return rot_symb[axis].subs(q, theta)

def trans_symb_d(axis: str, dist: sp.Symbol):
    return trans_symb[axis].subs(d, dist)

def rot_num(axis: str, theta: np.float64):
    return np.array(rot_symb[axis].subs(q, theta).evalf()).astype(np.float64)

def trans_num(axis: str, dist: np.float64):
    return np.array(trans_symb[axis].subs(d, dist).evalf()).astype(np.float64)

q1, q2, q3, q4, q5, q6 = sp.symbols('q1 q2 q3 q4 q5 q6')

def fk_solve_symb():
    """Forward kinematics symbolic solver

    ----------
    ### Parameters
        `q` : `[q_i]`
            6 joint angles
        `flag` : `str`
            defines output format

    -------
    ### Returns
            list of 4x4 homogeneous symbolic T_{0}^{i}
    """

    
    l1, l2, l3, l4, l5, l6 = link_lengths

    # kinematic part
    T12 = \
        rot_symb_q("z", q1 + np.pi/2) * \
        rot_symb_q("x", np.pi/2) * \
        trans_num("y", l1)

    T23 = \
        rot_symb_q("z", q2) * \
        trans_symb_d("x", l2)

    T34 = \
        rot_symb_q("z", q3 - np.pi/2) * \
        rot_symb_q("y", - np.pi/2) * \
        trans_symb_d("y", l3)
    
    # wrist part
    T45 = \
        rot_symb_q("z", q4 + np.pi/2) * \
        rot_symb_q("y", - np.pi/2) * \
        trans_symb_d("y", l4)

    T56 = \
        rot_symb_q("z", q5 + np.pi/2) * \
        rot_symb_q("x", np.pi/2) * \
        trans_symb_d("z", l5)

    T6e = \
        rot_symb_q("z", q6) * \
        trans_symb_d("z", l6)

    Ts = [T12, T23, T34, T45, T56, T6e]

    return Ts

transformations = fk_solve_symb()

def transform_base(base_frame):
    return [base_frame] + transformations

def fk_symb_ij(l: int, r: int, base_frame=sp.Identity(4)):
    ts = transform_base(base_frame)[l:r+1]
    for i in range(1,len(ts)):
        ts[i] = ts[i-1] * ts[i]
    return ts

def FK_solve(q=fk_input, flag="ee"):
    """Forward kinematics solver

    ----------
    ### Parameters
        `q` : `[q_i]`
            6 joint coordinates
        `flag` : `str`
            defines output format


    -------
    ### Returns
        if `flag == "ee"` then `np.array`
            4x4 homogeneous T_{0}^{e}
        if `flag == "full"` then `[np.array]`
            list of 4x4 homogeneous T_{0}^{i}
    """
    ts = fk_symb_ij(0, 6)
    a1, a2, a3, a4, a5, a6 = fk_input
    subs = [(q1,a1),(q2,a2),(q3,a3),(q4,a4),(q5,a5),(q6,a6)]
    
    if flag == "ee":
        return np.array(ts[-1].subs(subs).evalf()).astype(np.float64)
    
    elif flag == "full":
        return np.array(ts.subs(subs).evalf()).astype(np.float64)

print(FK_solve())

# sp.pprint(fk_symb_ij(4, 6), num_columns = 100)
# print(fk_symb_ij(4, 6))

# def IK_solve(base_frame, ee_frame):
#     """Inverse kinematics solver

#     ----------
#     ### Parameters
#         `base_frame` : `np.array`
#             4x4 homogeneous transformation matrix representing robot base
#         `ee_frame` : `np.array`
#             4x4 homogeneous transformation matrix representing end effector pose


#     -------
#     ### Returns
#         `[np.array]`
#             list of all possible solutions
#     """
    
    

# def select_similar():
#     # selects the ik solution that is the most similar to given input
#     pass

# def check_solvers(base_frame, ee_frame):
#     # solve IK
#     # solve FK with base frame for each IK solution
#     pass