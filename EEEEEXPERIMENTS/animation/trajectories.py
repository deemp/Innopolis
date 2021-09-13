#!/usr/bin/env python
# coding: utf-8

# %%

import numpy as np

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

# math
theta = np.arctan(c/a)
# phi_min=-theta
# phi_max=np.pi/2
phi_min=0
phi_max=2*np.pi

t_min=phi_min/omega_O1_A
t_max=phi_max/omega_O1_A
time = np.linspace(t_min,t_max,num=N,endpoint=False)

ind = np.arange(N)

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


# %%

phiA = omega_O1_A * time
A = O1_A * np.array([np.cos(phiA), np.sin(phiA)]).T
vA = num_derivative(A)
aA = num_derivative(vA)
# A


# %%

theta1 = phiA + theta
O1 = np.array([0.,0.])
O2 = np.array([a,-c])
O1_O2 = np.linalg.norm(O2-O1)

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

def coordC(idx):
    angCAB = angleT(AC,AB,BC)
    vAB = B[idx]-A[idx]
    # rotate rows-vectors individually
    # https://stackoverflow.com/a/38372534
    return A[idx] + Rot(angCAB).dot(vAB.T).T * AC / AB

C = coordC(ind)
# C


# %%


def xD(idx):
    return C[idx,0] + CD * np.cos(np.arcsin((C[idx,1]-d)/CD))
xD(ind)
np.full(N,d)
D = np.vstack((xD(ind), np.full(N,d))).T

# D


# %%

def coordE(idx):
    vCD = D[idx] - C[idx]
    return C[idx] + vCD * CE/CD

E = coordE(ind)
# E


# %%

O3 = np.array([a+b,d+e])

F = np.empty((N,2))

for idx in range(N):
    vE_O3 = O3 - E[idx]
    E_O3 = np.linalg.norm(vE_O3)

    angleO2_E_F = angleT(E_O3, EF,O3_F)
    F[idx] = E[idx] + Rot(-angleO2_E_F).dot(vE_O3)* EF / E_O3

# F
#%%
O1 = np.full((N,2),O1)
O2 = np.full((N,2),O2)
O3 = np.full((N,2),O3)

# %%
point_names = ["A","B","C","D","E","F"]

points = [A,B,C,D,E,F,O1,O2,O3]
velocities = [num_derivative(i) for i in points[:-3]]
accelerations = [num_derivative(i) for i in velocities]

velocities = {
    k : np.hstack((points[i], points[i]+velocities[i])) 
    for i,k in enumerate(point_names)
}

accelerations = {
    k : np.hstack((points[i], points[i]+accelerations[i]))
    for i,k in enumerate(point_names)
}

points = {
    k : points[i] 
    for i,k in enumerate(point_names+["O1","O2","O3"])
}
#%%
class Class():
    def __init__(self) -> None:
        self.cnt = 3
        self.change_cnt()

    def change_cnt(self):
        l = self.cnt
        l += 3
        print(self.cnt)

t = Class()
# %%