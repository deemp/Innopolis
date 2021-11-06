import numpy as np
from sympy.core.symbol import Symbol

link_lengths = [1,1,1,1,1,1]
fk_input = [1,1,1,1,1,1]

def rot33(axis, theta):
    """produce homogeneous 4x4 rotation matrix around given axis by given angle

    ----------
    ### Parameters
        `axis` : `str`
            "x" | "y" | "z" - axis to rotate about
        `theta` : `np.float`
            angle of rotation

    -------
    ### Returns
        `np.array`
            rotation matrix
    """
    c = np.cos(theta)
    s = np.sin(theta)
    r = {
        "z": np.array([
                [c, -s, 0],
                [s, c, 0],
                [0, 0, 1]
            ]),
        "y": np.array([
                [c, 0, s],
                [0, 1, 0],
                [-s, 0, c]
            ]),
        "x": np.array([
                [1, 0, 0],
                [0, c, -s],
                [0, s, c]
            ]),
    }
    T = np.eye(4)
    T[:3,:3] = r[axis]
    return T

def trans(axis, d):
    """produce homogeneous 4x4 translation matrix along given axis by given distance

    ----------
    ### Parameters
        `axis` : `str`
            "x" | "y" | "z" - axis to rotate about
        `distance` : `np.float`
            distance by which to move

    -------
    ### Returns
        `np.array`
            translation matrix
    """
    r = {
        "z": np.array([
                [0],
                [0],
                [d]
            ]),
        "y": np.array([
                [0],
                [d],
                [0]
            ]),
        "x": np.array([
                [d],
                [0],
                [0]
            ]),
    }
    T = np.eye(4)
    T[:3,3] = r[axis]
    return T

def FK_solve(q, flag):
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

    q1, q2, q3, q4, q5, q6 = q
    l1, l2, l3, l4, l5, l6 = link_lengths
    
    # kinematic part
    T12 = np.linalg.multi_dot([
        rot_symb("z", q1 + np.pi/2),
        rot_symb("x", np.pi/2),
        trans_symb("y", l1)
    ])
    T23 = np.linalg.multi_dot([
        rot_symb("z", q2),
        trans_symb("x", l2)
    ])
    T34 = np.linalg.multi_dot([
        rot_symb("z", q3 - np.pi/2),
        rot_symb("y", - np.pi/2),
        trans_symb("y", l3)
    ])
    
    # wrist part
    T45 = np.linalg.multi_dot([
        rot_symb("z", q4 + np.pi/2),
        rot_symb("y", - np.pi/2),
        trans_symb("y", l3)
    ])
    T56 = np.linalg.multi_dot([
        rot_symb("z", q5 + np.pi/2),
        rot_symb("x", np.pi/2),
        trans_symb("z", l5)
    ])
    T6e = np.linalg.multi_dot([
        rot_symb("z", q5),
        trans_symb("z", l6)
    ])

    Ts = [T12, T23, T34, T45, T56, T6e]

    if flag == "ee":
        return np.linalg.multi_dot(Ts)
    
    elif flag == "full":
        return Ts

def transoform_base(base_frame, q, flag):
    sol = FK_solve(q, flag)

    if flag == "ee":
        return np.dot(base_frame, sol)
    
    elif flag == "full":
        return [np.dot(base_frame, i) for i in sol]

import sympy as sp
joint_angles_symbolic = sp.symbols('q1 q2 q3 q4 q5 q6')

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

def trans_symb_q(axis: str, dist: sp.Symbol):
    return trans_symb[axis].subs(d, dist)

def rot_num(axis: str, theta: np.float32):
    return np.array(rot_symb[axis].subs(q, theta)).astype(np.float32)

def trans_num(axis: str, dist: np.float32):
    return np.array(trans_symb[axis].subs(d, dist)).astype(np.float32)

qs = sp.symbols('q1 q2 q3 q4 q5 q6')

def FK_solve_symb(q):
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

    q1, q2, q3, q4, q5, q6 = qs
    l1, l2, l3, l4, l5, l6 = link_lengths
    
    # kinematic part
    T12 = rot_symb_q("z", q1 + np.pi/2) * \
        rot_symb_q("x", np.pi/2) * \
        trans_num("y", l1)

    T23 = rot_symb("z", q2) * \
        trans_symb("x", l2)

    T34 = np.linalg.multi_dot([
        rot_symb("z", q3 - np.pi/2),
        rot_symb("y", - np.pi/2),
        trans_symb("y", l3)
    ])
    
    # wrist part
    T45 = np.linalg.multi_dot([
        rot_symb("z", q4 + np.pi/2),
        rot_symb("y", - np.pi/2),
        trans_symb("y", l3)
    ])
    T56 = np.linalg.multi_dot([
        rot_symb("z", q5 + np.pi/2),
        rot_symb("x", np.pi/2),
        trans_symb("z", l5)
    ])
    T6e = np.linalg.multi_dot([
        rot_symb("z", q6),
        trans_symb("z", l6)
    ])

    Ts = [T12, T23, T34, T45, T56, T6e]

    # need prefix product
    if flag == "ee":
        return np.linalg.multi_dot(Ts)
    
    elif flag == "full":
        return Ts

print(rot_num(axis="x", theta=0.4))

def IK_solve(base_frame, ee_frame):
    """Inverse kinematics solver

    ----------
    ### Parameters
        `base_frame` : `np.array`
            4x4 homogeneous transformation matrix representing robot base
        `ee_frame` : `np.array`
            4x4 homogeneous transformation matrix representing end effector pose


    -------
    ### Returns
        `[np.array]`
            list of all possible solutions
    """
    
    

def select_similar():
    # selects the ik solution that is the most similar to given input
    pass

def check_solvers(base_frame, ee_frame):
    # solve IK
    # solve FK with base frame for each IK solution
    pass