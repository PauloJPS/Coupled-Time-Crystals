import qutip as qt
import numpy as np 
import scipy as sp
import pickle
import matplotlib.pyplot as plt 
import matplotlib.ticker as mtick

import matplotlib.ticker as ticker

from scipy.integrate import solve_ivp
from scipy.linalg import sqrtm, eigvals
from scipy.integrate import simpson


 


######
###### The dynamics of the covariance matrix and the mean-field quantities 
######

def func(t, y, ω1x, ω1z, ω2x, ω2z, gx, gy, gz, k1, k2, n1, n2):

    print(t)

    G = y[0:36].reshape((6, 6))
    m1x, m1y, m1z, m2x, m2y, m2z = y[36:36+6]

    Dl = -np.array([[0.0,  ω1z, 0.0,  0.0,  0.0,  0.0],
                    [-ω1z, 0.0, ω1x,  0.0,  0.0,  0.0],
                    [0.0, -ω1x, 0.0,  0.0,  0.0,  0.0],
                    [0.0,  0.0, 0.0,  0.0,  ω2z,  0.0],
                    [0.0,  0.0, 0.0, -ω2z,  0.0,  ω2x],
                    [0.0,  0.0, 0.0,  0.0, -ω2x,  0.0]])

    Db = -np.sqrt(2) * np.array([[0.0,    0.0,   -k1*m1x, 0.0,    0.0,    0.0],
                                 [0.0,    0.0,   -k1*m1y, 0.0,    0.0,    0.0],
                                 [k1*m1x, k1*m1y, 0.0,    0.0,    0.0,    0.0],
                                 [0.0,    0.0,    0.0,    0.0,    0.0,   -k2*m2x],
                                 [0.0,    0.0,    0.0,    0.0,    0.0,   -k2*m2y],
                                 [0.0,    0.0,    0.0,    k2*m2x, k2*m2y, 0.0]])
    
    Dc = -np.sqrt(2) * np.array([[ 0.0,         m2z*gz, -m2y*gy, 0.0,     0.0,     0.0],
                                 [-m2z*gz,      0.0,     m2x*gx, 0.0,     0.0,     0.0],
                                 [ m2y*gy,     -m2x*gx,  0.0,    0.0,     0.0,     0.0],
                                 [ 0.0,         0.0,     0.0,    0.0,     m1z*gz, -m1y*gy],
                                 [ 0.0,         0.0,     0.0,   -m1z*gz,  0.0,     m1x*gx],
                                 [ 0.0,         0.0,     0.0,    m1y*gy, -m1x*gx,  0.0]])

    Q = np.sqrt(2.0) * np.array([[ 0.0,     0.0,     0.0,    0.0,     m1z*gy, -m1y*gz],
                                 [ 0.0,     0.0,     0.0,   -m1z*gx,  0.0,     m1x*gz],
                                 [ 0.0,     0.0,     0.0,    m1y*gx, -m1x*gy,  0.0],
                                 [ 0.0,     m2z*gy, -m2y*gz, 0.0,     0.0,     0.0],
                                 [-m2z*gx,  0.0,     m2x*gz, 0.0,     0.0,     0.0],
                                 [ m2y*gx, -m2x*gy,  0.0,    0.0,     0.0,     0.0]])

    s = np.sqrt(2.0) * np.array([[0.0,  m1z, -m1y,  0.0,  0.0,  0.0],
                                 [-m1z, 0.0,  m1x,  0.0,  0.0,  0.0],
                                 [m1y, -m1x,  0.0,  0.0,  0.0,  0.0],
                                 [0.0,  0.0,  0.0,  0.0,  m2z, -m2y],
                                 [0.0,  0.0,  0.0, -m2z,  0.0,  m2x],
                                 [0.0,  0.0,  0.0,  m2y, -m2x,  0.0]])

    B = np.array([[0.0, -k1,   0.0,  0.0,  0.0,  0.0],
                  [k1,   0.0,  0.0,  0.0,  0.0,  0.0],
                  [0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
                  [0.0,  0.0,  0.0,  0.0, -k2,   0.0],
                  [0.0,  0.0,  0.0,  k2,   0.0,  0.0],
                  [0.0,  0.0,  0.0,  0.0,  0.0,  0.0]])

    A = np.array([[k1*(2.0*n1 + 1.0), 0.0,               0.0, 0.0,               0.0,               0.0],
                  [0.0,               k1*(2.0*n1 + 1.0), 0.0, 0.0,               0.0,               0.0],
                  [0.0,               0.0,               0.0, 0.0,               0.0,               0.0],
                  [0.0,               0.0,               0.0, k2*(2.0*n2 + 1.0), 0.0,               0.0],
                  [0.0,               0.0,               0.0, 0.0,               k2*(2.0*n2 + 1.0), 0.0],
                  [0.0,               0.0,               0.0, 0.0,               0.0,               0.0]])

    #### Covariance dynamics

    W = (Q + s @ B)
    dG = (W @ G + G @ W.T - s @ A @ s)

    #### Mean-field dynamics for system 1
    dm1x = -np.sqrt(2.0) * (gz * m1y * m2z) + np.sqrt(2.0) * (gy * m1z * m2y) - (ω1z * m1y) + np.sqrt(2.0) * (k1 * m1x * m1z)

    dm1y = -np.sqrt(2.0) * (gx * m1z * m2x) + np.sqrt(2.0) * (gz * m1x * m2z) + (ω1z * m1x) - (ω1x * m1z) + np.sqrt(2.0) * (k1 * m1y * m1z)

    dm1z = -np.sqrt(2.0) * (gy * m1x * m2y) + np.sqrt(2.0) * (gx * m1y * m2x) + (ω1x * m1y) - np.sqrt(2.0) * k1 * (m1x**2.0 + m1y**2.0)

    #### Mean-field dynamics for system 2
    dm2x = -np.sqrt(2.0) * (gz * m2y * m1z) + np.sqrt(2.0) * (gy * m2z * m1y) - (ω2z * m2y) + np.sqrt(2.0) *  (k2 * m2x * m2z)

    dm2y = -np.sqrt(2.0) * (gx * m2z * m1x) + np.sqrt(2.0) * (gz * m2x * m1z) + (ω2z * m2x) - (ω2x * m2z) + np.sqrt(2.0) * (k2 * m2y * m2z)

    dm2z = -np.sqrt(2.0) * (gy * m2x * m1y) + np.sqrt(2.0) * (gx * m2y * m1x) + (ω2x * m2y)  - np.sqrt(2.0) * k2 * (m2x**2.0 + m2y**2.0)

    #### Updating

    return dG.flatten().tolist() +  [dm1x, dm1y, dm1z, dm2x, dm2y, dm2z]


#########
#########
#########

def get_traj(tspan, dt, ω1x, ω1z, ω2x, ω2z, gx, gy, gz, k1, k2, n1, n2,
             m1x0, m1y0, m1z0, m2x0, m2y0, m2z0, G0):

    initial = np.zeros(36 +  6)

    initial[0:36] = G0.flatten()

    initial[36:36+6] = [m1x0, m1y0, m1z0, m2x0, m2y0, m2z0]

    p = (ω1x, ω1z, ω2x, ω2z, gx, gy, gz, k1, k2, n1, n2)

    sol = solve_ivp(func, tspan, initial, args=p, method='DOP853', t_eval=np.arange(tspan[0], tspan[1], dt), rtol=1e-12, atol=1e-12)
    print("Status = {}\n".format(sol.status))

    return sol.t, sol.y.T


def Simul_Traj(parameters):

    t, sol = get_traj(**parameters)
    
    return t, sol

#########
######### the physical quantities from the object returned by the solver 
#########

def MeanField(sol):
    num_rows, num_columns = sol.shape

    m1x = [sol[i,36+0] for i in range(num_rows)]
    m1y = [sol[i,36+1] for i in range(num_rows)]
    m1z = [sol[i,36+2] for i in range(num_rows)]
    m2x = [sol[i,36+3] for i in range(num_rows)]
    m2y = [sol[i,36+4] for i in range(num_rows)]
    m2z = [sol[i,36+5] for i in range(num_rows)]

    return m1x, m1y, m1z, m2x, m2y, m2z

def CovarianceMatrix(sol):
    num_rows, num_columns = sol.shape
    covariance_matrix = [sol[i,0:36].reshape((6, 6)) for i in range(num_rows)]
    return np.array(covariance_matrix)

def SymplecticMatrix(sol):

    num_rows, num_columns = sol.shape
    symplectic = [np.sqrt(2.0) * np.array([[0, sol[i, 36+2], -sol[i, 36+1], 0, 0, 0],
                                           [-sol[i, 36+2], 0, sol[i, 36+0], 0, 0, 0],
                                           [sol[i, 36+1], -sol[i, 36+0], 0, 0, 0, 0],
                                           [0, 0, 0, 0, sol[i, 36+5], -sol[i, 36+4]],
                                           [0, 0, 0, -sol[i, 36+5], 0, sol[i, 36+3]],
                                           [0, 0, 0, sol[i, 36+4], -sol[i, 36+3], 0]]) for i in range(num_rows)]

    return np.array(symplectic) 

####
#### Simulation 
####
    
def simulate():
    ω1x = 2.0
    ω1z = 0.0
    
    ω2x = 2.0
    ω2z = 0.0
    
    gx, gy = 2, 2
    gz = 2.0
    
    k1 = 1.0
    k2 = 1.0
    
    n1 = 0.
    n2 = 0.
    
    ### Initial state in the ground-state of the bare Hamiltonian
    
    parameters = {"tspan": (0, 50),
                      "dt": 0.01,
                      "ω1x": ω1x,
                      "ω1z": ω1z,
                      "ω2x": ω2x,
                      "ω2z": ω2z,
                      "gx": gx,
                      "gy": gy,
                      "gz": gz,
                      "k1": k1,
                      "k2": k2,
                      "n1": n1, 
                      "n2": n2, 
                      "m1x0": 0.0,
                      "m1y0": 0.0,
                      "m1z0": -np.sqrt(1.0/2.0),
                      "m2x0": 0.0,
                      "m2y0": 0.0,
                      "m2z0": -np.sqrt(1.0/2.0),
                      "G0": np.array([[0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
                                      [0.0, 0.5, 0.0, 0.0, 0.0, 0.0],
                                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                      [0.0, 0.0, 0.0, 0.5, 0.0, 0.0],
                                      [0.0, 0.0, 0.0, 0.0, 0.5, 0.0],
                                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
                 }
    
    time, sol = Simul_Traj(parameters)
    information = QuantumClassicalThermodynamics(sol)

    return time, sol, information

