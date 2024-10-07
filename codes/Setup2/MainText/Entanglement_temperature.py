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
sys.path.insert(0, '../')

from Correlations import *

def initial_state():

    ω1x = 0.0
    ω1z = 0.0
    
    ω2x = 2.5
    ω2z = 0.0
    
    gx, gy = 1, 1
    gz = 0.0
    
    k1 = 1.0
    k2 = 1.0
    
    n1 = 0.0
    n2 = 0.0
    
    ### Initial state in the ground-state of the bare Hamiltonian
    
    parameters = {"tspan": (0, 200),
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


def Entanglement_Temp_SP():
    
    parameters = initial_state()
    parameters["tspan"] = (0, 200)
    parameters["ω1x"] = 0.0
    parameters["ω2x"] = 2.5
    parameters["gx"] = 2.5
    parameters["gy"] = 2.5
    parameters["gz"] = 0.0

    ### Time interval to avarege
    tlen = parameters["tspan"][1]
    Nint = int(tlen/parameters["dt"])

    ### List of couplings 
    N_n = 20
    ocuppation_list = np.linspace(0, 0.5, N_n)

    ### Phase-diagram maps heat matrix 
    Nega_mat = np.zeros(N_n)

    ### counter 
    counter = 0  
    ### Looping 
    for i,n in enumerate(ocuppation_list):
        print("Interaction [{}] out of [{}], n={}".format(counter, N_n, n))
        ### setting parameter 
        parameters["n1"] = n
        parameters["n2"] = n
    
        ### getting the solution 
        time, sol = Simul_Traj(parameters)

        ## getting the information theory quantities
        information = QuantumClassicalThermodynamics(sol)

        ### getting heat  
        Nega = information[2]

        ### getting time interval (last quarter) 
        Nint_init = int((3/4)*Nint)

        av_nega = 1/(time[-1] - time[Nint_init]) * simpson(Nega[Nint_init:], dx=parameters["dt"])
        
        ####
        Nega_mat[i] = av_nega

        counter += 1   

    ### Saving 
    data = parameters.copy()
    data.update({"PhaseMap_nega":Nega_mat})

    data.update({"ocuppation_list":ocuppation_list})
    with open('Data_Entanglement_Temperature_sp.pickle', 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    ###  
    return ocuppation_list, Nega_mat
    




def Entanglement_Temp_tc():
    
    parameters = initial_state()
    parameters["ω1x"] = 0.0
    parameters["ω2x"] = 2.5
    parameters["gx"] = 1.0
    parameters["gy"] = 1.0
    parameters["gz"] = 0.0

    ### Time interval to avarege
    tlen = parameters["tspan"][1]
    Nint = int(tlen/parameters["dt"])

    ### List of couplings 
    N_n = 200
    ocuppation_list = np.linspace(0, 0.5, N_n)

    ### Phase-diagram maps heat matrix 
    Nega_mat = np.zeros(N_n)

    ### counter 
    counter = 0  
    ### Looping 
    for i,n in enumerate(ocuppation_list):
        print("Interaction [{}] out of [{}], n={}".format(counter, N_n, n))
        ### setting parameter 
        parameters["n1"] = n
        parameters["n2"] = n
    
        ### getting the solution 
        time, sol = Simul_Traj(parameters)

        ## getting the information theory quantities
        information = QuantumClassicalThermodynamics(sol)

        ### getting heat  
        Nega = information[2]

        ### getting time interval (last quarter) 
        Nint_init = int((3/4)*Nint)

        av_nega = 1/(time[-1] - time[Nint_init]) * simpson(Nega[Nint_init:], dx=parameters["dt"])
        
        ####
        Nega_mat[i] = av_nega

        counter += 1   

    ### Saving 
    data = parameters.copy()
    data.update({"PhaseMap_nega":Nega_mat})

    data.update({"ocuppation_list":ocuppation_list})
    with open('Data_Entanglement_Temperature.pickle', 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    ###  
    return ocuppation_list, Nega_mat
    



def Asymmetry_correlations():
    
    parameters_tc = initial_state()
    parameters_tc["ω1x"] = 2.0
    parameters_tc["ω2x"] = 2.0

    parameters_sp = initial_state()
    parameters_sp["ω1x"] = 0.5
    parameters_sp["ω2x"] = 0.5


    ### Time interval to avarege
    tlen = parameters_tc["tspan"][1]
    Nint = int(tlen/parameters_tc["dt"])

    ### List of couplings 
    N_g = 40
    g_list = np.linspace(0, 4, N_g)
    #N_g = 7
    #g_list = (4, 3.5, 3, 2, 1, 0.5, 0) 

    ### Phase-diagram maps heat matrix 
    Entr_mat_tc = np.zeros(N_g)
    Ccor_mat_tc = np.zeros(N_g)
    Disc_mat_tc = np.zeros(N_g)
    Nega_mat_tc = np.zeros(N_g)

    Entr_mat_sp = np.zeros(N_g)
    Ccor_mat_sp = np.zeros(N_g)
    Disc_mat_sp = np.zeros(N_g)
    Nega_mat_sp = np.zeros(N_g)

    ### 
    ### counter 
    counter = 0  
    ### Looping 
    for i,g in enumerate(g_list):
        gz = 4-g
        print("Interaction [{}] out of [{}], gz={}, g={} ".format(counter,N_g, gz, g))
        ### setting parameter 
        parameters_tc["gz"] = gz 
        parameters_tc["gy"] = g
        parameters_tc["gx"] = g

        parameters_sp["gz"] = gz 
        parameters_sp["gy"] = g
        parameters_sp["gx"] = g
    

    
        ### getting the solution 
        time_tc, sol_tc = Simul_Traj(parameters_tc)
        time_sp, sol_sp = Simul_Traj(parameters_sp)

        ## getting the information theory quantities
        information_tc = QuantumClassicalThermodynamics(sol_tc)
        information_sp = QuantumClassicalThermodynamics(sol_sp)

        ### getting heat  
        Ccor_tc = information_tc[0]
        Disc_tc = information_tc[1]
        Nega_tc = information_tc[2]
        Entr_tc = information_tc[4]

        Ccor_sp = information_sp[0]
        Disc_sp = information_sp[1]
        Nega_sp = information_sp[2]
        Entr_sp = information_sp[4]

        ### getting time interval (last quarter) 
        Nint_init = int((3/4)*Nint)
        
        av_ccor_tc = 1/(time_tc[-1] - time_tc[Nint_init]) * simpson(Ccor_tc[Nint_init:], dx=parameters_tc["dt"])
        av_disc_tc = 1/(time_tc[-1] - time_tc[Nint_init]) * simpson(Disc_tc[Nint_init:], dx=parameters_tc["dt"])
        av_nega_tc = 1/(time_tc[-1] - time_tc[Nint_init]) * simpson(Nega_tc[Nint_init:], dx=parameters_tc["dt"])
        av_entr_tc = 1/(time_tc[-1] - time_tc[Nint_init]) * simpson(Entr_tc[Nint_init:], dx=parameters_tc["dt"])

        av_ccor_sp = 1/(time_sp[-1] - time_sp[Nint_init]) * simpson(Ccor_sp[Nint_init:], dx=parameters_sp["dt"])
        av_disc_sp = 1/(time_sp[-1] - time_sp[Nint_init]) * simpson(Disc_sp[Nint_init:], dx=parameters_sp["dt"])
        av_nega_sp = 1/(time_sp[-1] - time_sp[Nint_init]) * simpson(Nega_sp[Nint_init:], dx=parameters_sp["dt"])
        av_entr_sp = 1/(time_sp[-1] - time_sp[Nint_init]) * simpson(Entr_sp[Nint_init:], dx=parameters_sp["dt"])
        
        ####
        
        Ccor_mat_tc[i] = av_ccor_tc
        Disc_mat_tc[i] = av_disc_tc
        Nega_mat_tc[i] = av_nega_tc
        Entr_mat_tc[i] = av_entr_tc
 
        Ccor_mat_sp[i] = av_ccor_sp
        Disc_mat_sp[i] = av_disc_sp
        Nega_mat_sp[i] = av_nega_sp
        Entr_mat_sp[i] = av_entr_sp

        counter += 1   

    ### Saving 
    data = parameters_tc.copy()
    data.update({"PhaseMap_ccor_tc":Ccor_mat_tc})
    data.update({"PhaseMap_disc_tc":Disc_mat_tc})
    data.update({"PhaseMap_nega_tc":Nega_mat_tc})
    data.update({"PhaseMap_entr_tc":Entr_mat_tc})

    data.update({"PhaseMap_ccor_sp":Ccor_mat_sp})
    data.update({"PhaseMap_disc_sp":Disc_mat_sp})
    data.update({"PhaseMap_nega_sp":Nega_mat_sp})
    data.update({"PhaseMap_entr_sp":Entr_mat_sp})

    data.update({"g_list":g_list})
    with open('Data_asymmetry_correlations.pickle', 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    ###  
    return g_list, Ccor_mat_tc, Disc_mat_tc, Nega_mat_tc, Entr_mat_tc, Ccor_mat_sp, Disc_mat_sp, Nega_mat_sp, Entr_mat_sp
    




