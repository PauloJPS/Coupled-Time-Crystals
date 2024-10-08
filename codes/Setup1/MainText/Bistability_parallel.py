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
from multiprocessing import Pool, Manager
from scipy.integrate import simpson

# adding Folder_2/subfolder to the system path
sys.path.insert(0, '../../')
from MeanField import Simul_Traj

#
#
# Evaluate the phase diagram in setup 1 
#
#

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

# Move total_tasks definition here
total_tasks = 0

def compute(param, counter, lock):
    i, J, parameters = param
    local_params = parameters.copy()
    local_params["gx"], local_params["gy"] = J, J

    t, sol = Simul_Traj(local_params)
    internal_energy = simpson(1/np.sqrt(2)*(sol[:,2] + sol[:,5] + 2/np.sqrt(2)), x=t)/t[-1]

    global total_tasks

    with lock:  
        counter.value += 1
        if counter.value % 10 == 0:
            print(f"Progress : {counter.value}/{total_tasks}")

    return (i, internal_energy)

def MeanField_Bistability_InternalEnergy():

    global total_tasks  # ensure we are modifying the global total_tasks

    parameters = initial_state()
    parameters["tspan"] = (0, 1000)
    parameters["dt"] = 0.1
    parameters["ω1x"] = 2.0
    parameters["ω2x"] = 2.0
    parameters["gz"] = 1.0

    ### Parameter list 
    N = 1000
    J_list = np.linspace(2, 4, N)
    internal_energy_list = np.zeros(N)

    total_tasks = int(N)  # assign total_tasks here

    # Prepare parameter list for parallel computation
    param_list = [(i, J, parameters) for i, J in enumerate(J_list)]

    # Perform parallel computation using multiprocessing
    manager = Manager()
    counter = manager.Value("i", 0)
    lock = manager.Lock()

    with Pool(processes=7) as pool:
        results = pool.starmap(compute, [(param, counter, lock) for param in param_list])

    # Populate the matrices with the results
    for i, internal_energy in results:
        internal_energy_list[i] = internal_energy

    # Saving data
    data = parameters.copy()
    data.update({"J_list": J_list})
    data.update({"internal_energy_list": internal_energy_list})

    with open('Data/Data_Bistability_InternalEnergy.pickle', 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return J_list, internal_energy_list

if __name__ == '__main__':
    J_list, internal_energy_list = MeanField_Bistability_InternalEnergy()

