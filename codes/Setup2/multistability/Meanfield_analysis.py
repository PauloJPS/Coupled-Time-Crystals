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

def simulate(theta_1, theta_2):

    ### Simulate one mean-field trajectory 
    parameters = initial_state()
    parameters["tspan"] = (0, 100)
    parameters["dt"] = 0.1

    parameters["ω1x"] = 0.0
    parameters["ω2x"] = 2.5

    
    parameters["gx"] = 2.1
    parameters["gy"] = 2.1
    parameters["gz"] = 0.0

    #### theta {0, pi}, phi {0, 2pi}

    #theta_1  = np.random.rand()*(np.pi)
    #phi_1 = np.random.rand()*(2*np.pi)

    #theta_1 = theta
    #theta_2  = theta_1

    phi_1 = 0
    phi_2 = np.pi/2

    parameters["m1x0"] =  np.sqrt(1/2)*(np.sin(theta_1)*np.cos(phi_1))
    parameters["m1y0"] =  np.sqrt(1/2)*(np.sin(theta_1)*np.sin(phi_1))
    parameters["m1z0"] =  np.sqrt(1/2)*(np.cos(theta_1))

    parameters["m2x0"] =  np.sqrt(1/2)*(np.sin(theta_2)*np.cos(phi_2))
    parameters["m2y0"] =  np.sqrt(1/2)*(np.sin(theta_2)*np.sin(phi_2))
    parameters["m2z0"] =  np.sqrt(1/2)*(np.cos(theta_2))

    times, sol = Simul_Traj(parameters)
 
    return times, sol, parameters


def Multistability_trajectories():
    
    times, sol1, parameters = simulate(0, 0)
    times, sol2, parameters = simulate(0, np.pi/6)


    data = parameters.copy()
    data.update({"times":times})
    data.update({"sol1":sol1})
    data.update({"sol2":sol2})
    with open('Data/Data_bistable_trajectories.pickle', 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return times, sol1, sol2



