import numpy as np
import pickle
from multiprocessing import Pool
from scipy.integrate import simpson


import qutip as qt
import numpy as np 
import scipy as sp
import matplotlib.pyplot as plt 
import matplotlib.ticker as mtick
import matplotlib.ticker as ticker
import pickle
import sys

from scipy.integrate import solve_ivp
from scipy.linalg import sqrtm, eigvals
from scipy.integrate import simpson

from scipy.fft import fft, fftfreq

# adding Folder_2/subfolder to the system path
sys.path.insert(0, '../../')

from MeanField import Simul_Traj

def initial_state():
    ω1x = 0.0
    ω1z = 0.0
    
    ω2x = 0.0
    ω2z = 0.0
    
    gx, gy = 0.0, 0.0
    gz = 0.0
    
    k1 = 1.0
    k2 = 1.0
   
    ### Initial state in the ground-state of the bare Hamiltonian
    
    parameters = {"tspan": (0, 1e3),
                      "dt": 0.1,
                      "ω1x": ω1x,
                      "ω1z": ω1z,
                      "ω2x": ω2x,
                      "ω2z": ω2z,
                      "gx": gx,
                      "gy": gy,
                      "gz": gz,
                      "k1": k1,
                      "k2": k2,
                      "m1x0": 0.0,
                      "m1y0": 0.0,
                      "m1z0": -np.sqrt(1.0/2.0),
                      "m2x0": 0.0,
                      "m2y0": 0.0,
                      "m2z0": -np.sqrt(1.0/2.0)}
    
    return parameters

def compute(param):
    i, j, Omega, J, parameters = param
    local_params = parameters.copy()
    local_params["ω2x"] = Omega
    local_params["gx"], local_params["gy"] = J, J

    t, sol = Simul_Traj(local_params)
    
    mz1e2 = simpson(sol[:, 2]**2, t) / t[-1]
    mz2e2 = simpson(sol[:, 5]**2, t) / t[-1]
    
    print(i,j, "\n")

    return (i, j, mz1e2, mz2e2)

def MeanField_PD_Work():
    parameters = initial_state()
    parameters["tspan"] = (0, 1000)
    parameters["dt"] = 0.1

    ### Parameter list 
    N = 10
    J_list = np.linspace(0, 3, N)
    Omega_list = np.linspace(0, 3, N)

    mz1e2_mat = np.zeros((N, N))
    mz2e2_mat = np.zeros((N, N))

    # Prepare parameter list for parallel computation
    param_list = [(i, j, Omega, J, parameters) for i, Omega in enumerate(Omega_list) for j, J in enumerate(J_list)]

    # Perform parallel computation using multiprocessing
    with Pool() as pool:
        results = pool.map(compute, param_list)

    # Populate the matrices with the results
    for i, j, mz1e2, mz2e2 in results:
        mz1e2_mat[i][j] = mz1e2
        mz2e2_mat[i][j] = mz2e2

    # Saving data
    data = parameters.copy()
    data.update({"J_list": J_list})
    data.update({"Omega_list": Omega_list})
    data.update({"mz1e2_mat": mz1e2_mat})
    data.update({"mz2e2_mat": mz2e2_mat})

    with open('Data_seeding_PhaseDiagram_SP_TC_test.pickle', 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return Omega_list, J_list, mz1e2_mat, mz2e2_mat


if __name__ == '__main__':
    Omega_list, J_list, mz1e2_mat, mz2e2_mat = MeanField_PD_Work()
