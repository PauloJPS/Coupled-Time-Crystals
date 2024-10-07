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

from  Correlations_non_rotated import *
 
def initial_state():

    ω1x = 2.0
    ω1z = 0.0
    
    ω2x = 2.0
    ω2z = 0.0
    
    gx, gy = 0, 0
    gz = 0.0
    
    k1 = 1.0
    k2 = 1.0
    
    n1 = 0.0
    n2 = 0.0
    
    ### Initial state in the ground-state of the bare Hamiltonian
    
    parameters = {"tspan": (0, 100),
                      "dt": 0.05,
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
   
    return parameters


def Subheat_trajectories():

    #### Fig Appendix (subdominant heat and entropy)

    parameters = initial_state()
    parameters["tspan"] = (0, 20)
    parameters["dt"] = 0.1

    parameters["n1"] = 1.0
    parameters["n2"] = 1.0

    parameters["ω1x"] = 2.0
    parameters["ω2x"] = 2.0

    parameters_tc = parameters.copy()
    parameters_sp = parameters.copy()

    
    ####
    parameters_tc["gx"] = 2.0
    parameters_tc["gy"] = 2.0
    parameters_tc["gz"] = 2.0

    parameters_sp["gx"] = 3.5
    parameters_sp["gy"] = 3.5
    parameters_sp["gz"] = 0.0


    ### Time interval to avarege
    tlen = parameters["tspan"][1]
    Nint = int(tlen/parameters["dt"])

    ### Phase-diagram maps heat matrix 
    SHeat_tc = []
    SHeat_sp = [] 
 
    time, sol_tc = Simul_Traj(parameters_tc)
    time, sol_sp = Simul_Traj(parameters_sp)

    ### Total sub_heat 

    SHeat_tc = -(sol_tc[:,0] + sol_tc[:,0+7] + np.sqrt(2)*sol_tc[:,36+2]*(2*parameters["n1"] + 1))
    SHeat_tc += -(sol_tc[:,21] + sol_tc[:,21+7] + np.sqrt(2)*sol_tc[:,36+5]*(2*parameters["n2"] + 1))

    SHeat_sp = -(sol_sp[:,0] + sol_sp[:,0+7] + np.sqrt(2)*sol_sp[:,36+2]*(2*parameters["n1"] + 1))
    SHeat_sp += -(sol_sp[:,21] + sol_sp[:,21+7] + np.sqrt(2)*sol_sp[:,36+5]*(2*parameters["n2"] + 1))

    ### Saving 
    data = parameters.copy()
    data.update({"sub_heat_tc":SHeat_tc})
    data.update({"sub_heat_sp":SHeat_sp})
    data.update({"times":time})


    with open('Data/Data_SubHeat_Entropy.pickle', 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    ###  
    return time, SHeat_tc, SHeat_sp
 

