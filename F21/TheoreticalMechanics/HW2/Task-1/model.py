# %%

"""
Given
"""

omega_O1_A=2.
a=56.
b=10.
c=26.
d=16.
e=25.
O1_A=21.
O2_B=25.
O3_F=20.
AB=54.
BC=52.
CD=69.
CE=35.
EF=32.
AC=1/3*CD
N = 400

# %%

"""
Maf
"""

import numpy as np
from vispy.scene.widgets import axis

def num_derivative(point):
    dxdt = np.gradient(point.T[0], time)
    dydt = np.gradient(point.T[1], time)
    return np.array([dxdt, dydt]).T

def angleT(TL, TR, LR):
    return np.arccos((TL**2 + TR**2 - LR**2)/(2 * TL * TR))

def Rot(angle):
    return np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]])

#%%

"""
Constraints
"""

O1 = np.array([0.,0.])
O2 = np.array([a,-c])
O1_O2 = np.linalg.norm(O2-O1)

theta = np.arctan(c/a)

# domain of correct solutions
# take larger angle of the triangle O1_A_O2 with largest side
ang=angleT(O1_A, O1_O2, O2_B+AB)
phi_min=-ang-theta
phi_max=ang-theta

# full circle
# phi_min=0
# phi_max=2*np.pi

t_min=phi_min/omega_O1_A
t_max=phi_max/omega_O1_A
time = np.linspace(t_min,t_max,num=N,endpoint=False)

ind = np.arange(N)


# %%

"""
Find trajectory of A
"""

phiA = omega_O1_A * time
A = O1_A * np.array([np.cos(phiA), np.sin(phiA)]).T
vA = num_derivative(A)
aA = num_derivative(vA)
# A

# %%

"""
Find trajectory of B
"""

theta1 = 2*np.pi + phiA + theta

def coordB(idx):
    O2_A=np.sqrt(O1_O2**2 + O1_A**2 - 2 * O1_O2 * O1_A * np.cos(theta1[idx]))
    beta1 = angleT(O1_O2, O2_A, O1_A)
    beta1[(theta1 > np.pi) & (theta1 < 2*np.pi)] *= -1
    delta1= angleT(O2_A, O2_B, AB)
    phiB = np.pi - delta1 - beta1 - theta
    return O2 + O2_B * np.array([np.cos(phiB), np.sin(phiB)]).T

B = coordB(ind)
# B

# %%

"""
Find trajectory of C
"""

def coordC(idx):
    angCAB = angleT(AC,AB,BC)
    vAB = B[idx]-A[idx]
    # rotate rows-vectors individually
    # https://stackoverflow.com/a/38372534
    return A[idx] + Rot(angCAB).dot(vAB.T).T * AC / AB

C = coordC(ind)
# C


# %%

"""
Find trajectory of D
"""

def xD(idx):
    return C[idx,0] + CD * np.cos(np.arcsin((C[idx,1]-d)/CD))
xD(ind)
np.full(N,d)
D = np.vstack((xD(ind), np.full(N,d))).T

# D


# %%

"""
Find trajectory of E
"""

def coordE(idx):
    vCD = D[idx] - C[idx]
    return C[idx] + vCD * CE/CD

E = coordE(ind)
# E


# %%

"""
Find trajectory of F
"""

O3 = np.array([a+b,d+e])

F = np.empty((N,2))

for idx in range(N):
    vE_O3 = O3 - E[idx]
    E_O3 = np.linalg.norm(vE_O3)

    angleO2_E_F = angleT(E_O3, EF,O3_F)
    F[idx] = E[idx] + Rot(-angleO2_E_F).dot(vE_O3)* EF / E_O3

# F
#%%

"""
Calculating values
"""

O1 = np.full((N,2),O1)
O2 = np.full((N,2),O2)
O3 = np.full((N,2),O3)

point_names = ["A","B","C","D","E","F","O1","O2","O3"]

points_list = [A,B,C,D,E,F,O1,O2,O3]

#%%

velocities_list = [num_derivative(i) for i in points_list]

free_velocities = {
    k : num_derivative(i)
    for k,i in zip(point_names,points_list)
    }

# attached to points
velocities = {
    k : np.hstack((points_list[i], points_list[i]+velocities_list[i])) 
    for i,k in enumerate(point_names)
}

#%%

accelerations_list = [num_derivative(i) for i in velocities_list]

free_accelerations = {
    k : num_derivative(i)
    for k,i in zip(point_names,velocities_list)
}

# attached to points
accelerations = {
    k : np.hstack((points_list[i], points_list[i]+accelerations_list[i]))
    for i,k in enumerate(point_names)
}

#%%

points = {
    k : points_list[i]
    for i,k in enumerate(point_names)
}

#%%

def get_IC(X, Y, v1, v2):
    n_v1 = v1[::-1]
    n_v1[0] = -n_v1[0]
    n_v2 = v2[::-1]
    n_v2[0] = -n_v2[0]
    N_v = np.array([n_v1,n_v2]).T
    k, l = np.linalg.solve(N_v, Y-X)
    l = -l
    IC = X + k*n_v1
    return IC

def get_angular_velocity_at(X, Y, v1, v2):
    IC = get_IC(X,Y,v1,v2)
    IC_X = np.linalg.norm(IC - X)
    omega = np.cross(IC - X,v1)/(IC_X**2)
    return omega

def get_angular_velocities(X, Y, v1, v2):
    w = np.zeros(len(X))
    for i in range(len(X)):
        w[i] = get_angular_velocity_at(X[i], Y[i], v1[i], v2[i])
    return w

def get_angular_velocity_around_fixed(IC, X, v1):
    D = np.linalg.norm(X-IC,axis=1)
    # r cross v / r^2
    omega = np.cross(X-IC,v1)/(D**2)
    return omega

non_fixed_links = [["A","B"],["C","D"],["E","F"],["A","C"],["B","C"]]
fixed_links = [["O1","A"],["O2","B"],["O3","F"]]
fixed_radii = [O1_A, O2_B, O3_F]

links = non_fixed_links + fixed_links

angular_velocities = {
    f"{a}_{b}" :

    get_angular_velocities(
        points[a],
        points[b],
        free_velocities[a],
        free_velocities[b]
    )

    for a,b in non_fixed_links
}

for (i,j),k in zip(fixed_links, fixed_radii):
    angular_velocities[f"{i}_{j}"] = get_angular_velocity_around_fixed(
        points[i], 
        points[j],
        free_velocities[j],
        )

# print({i:j.shape for i,j in angular_velocities.items()})
# print(angular_velocities["O1_A"][0:100])

# %%
