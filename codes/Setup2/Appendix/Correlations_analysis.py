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

from Correlations import *

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
                      "L0": np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                      [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                                      [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                                      [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                                      [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                                      [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
                  }
   
    return parameters


def Trajectory_Correlations():
    
    parameters = initial_state()

    parameters["tspan"] = (0, 100)
    parameters["dt"] = 0.1

    parameters["ω1x"] = 0
    parameters["ω2x"] = 3.5

    ### Time interval to avarege
    tlen = parameters["tspan"][1]
    Nint = int(tlen/parameters["dt"])
 
    parameters["gz"] = 0
    parameters["gy"] = 2
    parameters["gx"] = 2
    ### getting the solution 
    time, sol = Simul_Traj(parameters)

    ## getting the information theory quantities
    information = QuantumClassicalThermodynamics(sol)

    Ccor = information[0]
    Disc = information[1]
    Nega = information[2]
    Entr = information[4]

    return time, Ccor, Disc, Nega, Entr 


def Negativities_trajectories():
    
    parameters = initial_state()

    parameters["tspan"] = (0, 100)
    parameters["dt"] = 0.05

    parameters["ω1x"] = 0
    parameters["ω2x"] = 2.5
    parameters["gz"] = 0

    ### Time interval to avarege
    tlen = parameters["tspan"][1]
    Nint = int(tlen/parameters["dt"])
 
    J_list = [0.5, 1, 1.5] 

    Nega_list = []
    for J in J_list:
        parameters["gy"] = J
        parameters["gx"] = J

        ### getting the solution 
        time, sol = Simul_Traj(parameters)

        ## getting the information theory quantities
        information = QuantumClassicalThermodynamics(sol)

        Nega_list.append(information[2])
    data = parameters.copy()
    data.update({"time":time})
    data.update({"J_list":J_list})
    data.update({"Nega_list":Nega_list})
    with open('Data/Data_Nega_trajectories.pickle', 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    return time, Nega_list









def Discord_Entropy_trajectories():
    
    parameters = initial_state()

    parameters["tspan"] = (0, 100)
    parameters["dt"] = 0.1

    parameters["ω1x"] = 2.0
    parameters["ω2x"] = 2.0

    ### Time interval to avarege
    tlen = parameters["tspan"][1]
    Nint = int(tlen/parameters["dt"])
 
    gs_list = [(1, 2), (2,2)]

    Discord_list = []
    Entropy_list = []

    for gs in gs_list:
        gz, g = gs

        parameters["gz"] = gz 
        parameters["gy"] = g
        parameters["gx"] = g
    
        ### getting the solution 
        time, sol = Simul_Traj(parameters)

        ## getting the information theory quantities
        information = QuantumClassicalThermodynamics(sol)

        Ccor = information[0]
        Disc = information[1]
        Nega = information[2]
        Entr = information[4]

        Discord_list.append(Disc)
        Entropy_list.append(Entr)

    data = parameters.copy()
    data.update({"time":time})
    data.update({"Discord_list":Discord_list})
    data.update({"Entropy_list":Entropy_list})
    data.update({"gs_list":gs_list})
    with open('Data/Data_Discord_trajectories.pickle', 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return time, Discord_list, Entropy_list

















def HeatEntropy_tcEsp():

    #### Fig Appendix (subdominant heat and entropy)

    parameters = initial_state()

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

    parameters_sp["gx"] = 3.0
    parameters_sp["gy"] = 3.0
    parameters_sp["gz"] = 0.5


    ### Time interval to avarege
    tlen = parameters["tspan"][1]
    Nint = int(tlen/parameters["dt"])

    ### Phase-diagram maps heat matrix 
    SHeat_tc = []
    SHeat_sp = [] 
 
    Entropy_tc = []
    Entropy_sp = [] 
 
    time, sol_tc = Simul_Traj(parameters_tc)
    time, sol_sp = Simul_Traj(parameters_sp)

    information_tc = QuantumClassicalThermodynamics(sol_tc)
    information_sp = QuantumClassicalThermodynamics(sol_sp)

    ### Total sub_heat 

    SHeat_tc = -(sol_tc[:,0] + sol_tc[:,0+7] + 2*np.sqrt(2)*sol_tc[:,36+36+2]*(2*parameters["n1"] + 1))
    SHeat_tc += -(sol_tc[:,21] + sol_tc[:,21+7] + 2*np.sqrt(2)*sol_tc[:,36+36+5]*(2*parameters["n2"] + 1))

    SHeat_sp = -(sol_sp[:,0] + sol_sp[:,0+7] + 2*np.sqrt(2)*sol_sp[:,36+36+2]*(2*parameters["n1"] + 1))
    SHeat_sp += -(sol_sp[:,21] + sol_sp[:,21+7] + 2*np.sqrt(2)*sol_sp[:,36+36+5]*(2*parameters["n2"] + 1))

    Entropy_tc = information_tc[4]
    Entropy_sp = information_sp[4]

    ### Saving 
    data = parameters.copy()
    data.update({"sub_heat_tc":SHeat_tc})
    data.update({"sub_heat_sp":SHeat_sp})
    data.update({"Entropy_tc":Entropy_tc})
    data.update({"Entropy_sp":Entropy_sp})
    data.update({"times":time})


    with open('Data_SubHeat_Entropy.pickle', 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    ###  
    return time, SHeat_tc, SHeat_sp, Entropy_tc, Entropy_sp
 

