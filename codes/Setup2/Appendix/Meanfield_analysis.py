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
    ω1x = 2.0
    ω1z = 0.0
    
    ω2x = 2.0
    ω2z = 0.0
    
    gx, gy = 1.0, 1.0
    gz = 1.0
    
    k1 = 1.0
    k2 = 1.0
   
    ### Initial state in the ground-state of the bare Hamiltonian
    
    parameters = {"tspan": (0, 1e3),
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
                      "m1x0": 0.0,
                      "m1y0": 0.0,
                      "m1z0": -np.sqrt(1.0/2.0),
                      "m2x0": 0.0,
                      "m2y0": 0.0,
                      "m2z0": -np.sqrt(1.0/2.0)}
    
    return parameters

def simulate():

    ### Simulate one mean-field trajectory 
    parameters = initial_state()
    parameters["ω1x"] = 0  
    parameters["ω2x"] = 2.5

    
    parameters["gx"] = 3
    parameters["gy"] = 3
    parameters["gz"] = 0.0

    #### theta {0, pi}, phi {0, 2pi}

    theta_1 = np.pi#np.random.rand()*(np.pi)
    phi_1 = 0 
    
    theta_2  = theta_1
    phi_2 =  phi_1

    parameters["m1x0"] =  np.sqrt(1/2)*(np.sin(theta_1)*np.cos(phi_1))
    parameters["m1y0"] =  np.sqrt(1/2)*(np.sin(theta_1)*np.sin(phi_1))
    parameters["m1z0"] =  np.sqrt(1/2)*(np.cos(theta_1))

    parameters["m2x0"] =  np.sqrt(1/2)*(np.sin(theta_2)*np.cos(phi_2))
    parameters["m2y0"] =  np.sqrt(1/2)*(np.sin(theta_2)*np.sin(phi_2))
    parameters["m2z0"] =  np.sqrt(1/2)*(np.cos(theta_2))

    times, sol = Simul_Traj(parameters)
 
    return times, sol, parameters

def Charger_battery_trajs():

    ### Simulate one mean-field trajectory 

    parameters = initial_state()
    parameters["tspan"] = (0, 20)
    parameters["dt"] = 0.01

    parameters["ω2x"] = 2.5
    parameters["ω1x"] = 0
    parameters["gz"] = 0

    parameters["m1x0"], parameters["m1z0"], parameters["m1z0"] = 0, 0, -1/np.sqrt(2)
    parameters["m2x0"], parameters["m2y0"], parameters["m2z0"] = 0, 0, -1/np.sqrt(2)

    mz_battery = []
    mz_charger = []

    J_list = [1, 1.5, 2.5]

    for J in J_list:
        parameters["gx"] = J
        parameters["gy"] = J
        
        times, sol = Simul_Traj(parameters)

        mz_battery.append(sol[:,2])
        mz_charger.append(sol[:,2+3])

    data = parameters.copy()
    data.update({"J_list":J_list})
    data.update({"mz_battery":mz_battery})
    data.update({"mz_charger":mz_charger})
    with open('Data/Data_ChargerBattery_trajectories.pickle', 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return times, mz_battery, mz_charger

def Fourier_PD():

    #### Function Appendix (fourier spectrum in fuction of J)

    parameters = initial_state()

    parameters["tspan"] = (0, 10000)
    parameters["dt"] = 0.1

    parameters["ω2x"] = 2
    parameters["ω1x"] = 0
    parameters["gz"] = 0

    parameters["m1x0"], parameters["m1z0"], parameters["m1z0"] = 0, 0, -1/np.sqrt(2)
    parameters["m2x0"], parameters["m2y0"], parameters["m2z0"] = 0, 0, -1/np.sqrt(2)

    g_list = np.linspace(0, 2.0, 301)
    modes_1 = []
    modes_2 = []

    for i, g in enumerate(g_list):
        print("Iteration {} ".format(i))
        parameters["gx"] = g
        parameters["gy"] = g

        times, sol = Simul_Traj(parameters)

        aux_times = times[len(times)//2:]
        aux_mz1 = sol[:,2][len(times)//2:]
        aux_mz2 = sol[:,5][len(times)//2:]

        y1 = fft(aux_mz1 - np.mean(aux_mz1))
        y2 = fft(aux_mz2 - np.mean(aux_mz2))

        xf = fftfreq(len(aux_times), parameters["dt"])[:len(aux_times)//2]
        aux_m1 = 2.0/len(aux_times) * np.abs(y1[0:len(aux_times)//2])
        aux_m2 = 2.0/len(aux_times) * np.abs(y2[0:len(aux_times)//2])

        modes_1.append(aux_m1/max(aux_m1))
        modes_2.append(aux_m2/max(aux_m2))

    #### Saving

    data = parameters.copy()

    data.update({"xf":xf})
    data.update({"g_list":g_list})
    data.update({"modes_m1":modes_1})
    data.update({"modes_m2":modes_2})

    with open('Data/Data_seeding_fourier.pickle', 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return g_list, xf, modes_1, modes_2

def Fourier_traj():

    #### Function Appendix (fourier spectrum in fuction of J)

    parameters = initial_state()

    parameters["tspan"] = (0, 10000)
    parameters["dt"] = 0.1

    parameters["ω2x"] = 2.5
    parameters["ω1x"] = 0
    parameters["gz"] = 0
    parameters["gx"] = 1.95
    parameters["gy"] = 1.95

    parameters["m1x0"], parameters["m1z0"], parameters["m1z0"] = 0, 0, -1/np.sqrt(2)
    parameters["m2x0"], parameters["m2y0"], parameters["m2z0"] = 0, 0, -1/np.sqrt(2)

    times, sol = Simul_Traj(parameters)

    aux_times = times[len(times)//2:]
    aux_mz1 = sol[:,2][len(times)//2:]
    aux_mz2 = sol[:,5][len(times)//2:]

    y1 = fft(aux_mz1 - np.mean(aux_mz1))
    y2 = fft(aux_mz2 - np.mean(aux_mz2))

    xf = fftfreq(len(aux_times), parameters["dt"])[:len(aux_times)//2]
    aux_m2 = 2.0/len(aux_times) * np.abs(y2[0:len(aux_times)//2])

    modes_2 = aux_m2/max(aux_m2)

    #### Saving

    data = parameters.copy()

    data.update({"xf":xf})
    data.update({"mode2_m2":modes_2})

    #with open('Data/Data_seeding_fourier_traj.pickle', 'wb') as handle:
    #    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return xf, modes_2


