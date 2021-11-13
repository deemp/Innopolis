import numpy as np
from scipy.integrate import solve_ivp
from utils import get_fk_solution, n, dim_h, eq, plot_list, decompose_transformation

np.set_printoptions(precision=2, suppress=True)

# dimension of homogeneous matrices

q_final = np.array([2.5,1.8,3.14,4.6,2.1,1.1])
"""final joint configuration"""
q_initial = np.array([0.,0.,0.,0.,0.,0.])
"""initial joint configuration"""

fk_full_default = get_fk_solution(qs=q_initial, flag="full")
base_default = np.eye(dim_h)
ee_default = fk_full_default[-1]

def jacobian(frames=fk_full_default):
    """get numeric geometric jacobian from forward kinematics"""
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
    """check if a given jacobian is singular"""
    return eq(np.linalg.det(jacobian), 0.)

def cartesian_velocity(jacobian, q_dot):
    """get a vector of cartesian velocities given joint velocities"""
    return jacobian.dot(q_dot)


t0 = 0
"""start of simulation time"""
tf = 500
"""end of simulation time"""
k = 0.01
"""difference scale coefficient"""

pose_target = decompose_transformation(get_fk_solution(q_final))
"""default target pose: [x,y,z,theta_x,theta_y,theta_z]^T"""

def solve_motion(y0=q_initial, t=(t0,tf), x1=pose_target):
    """solve differential equation for joint angles"""
    print("x_targ:\n", x1)
    def state_space(t, y):
        fk = get_fk_solution(y, flag="full")
        J = jacobian(frames=fk)
        xi = decompose_transformation(fk[-1])
        dx = (x1-xi)*k
        return np.linalg.inv(J).dot(dx)
    return solve_ivp(fun=state_space, y0=y0, t_span=t)

sol = solve_motion()
"""solution to differential equation for joint angles"""

x_current = np.array([decompose_transformation(get_fk_solution(sol.y.T[i])) for i in range(sol.t.shape[0])])
"""current pose: [x,y,z,theta_x,theta_y,theta_z]^T"""

labels = ["$x$ (m)", "$y$ (m)", "$z$ (m)", "$\\theta_x$ (rad)", "$\\theta_y$ (rad)", "$\\theta_z$ (rad)"]

plot_list([
    {
        "x": sol.t,
        "xlabel": "Time (s)",
        "ylabel": "Joint\\ angles (rad)",
        "title" : "Change\\ of\\ joint\\ angles\\ between\\ poses",
        "graphs": [{"y": sol.y[i], "label": f"$q_{i+1}$"} for i in range(n)]
    },
    {
        "x": sol.t,
        "xlabel": "Time (s)",
        "ylabel": "Pose",
        "title" : f"Change\\ of\\ EE\\ pose $ \n target: {pose_target} $\\ ",
        "graphs": 
            [{"y": x_current.T[i], "label": labels[i]} for i in range(n)]
    }
])