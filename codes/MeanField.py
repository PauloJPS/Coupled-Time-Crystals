import qutip as qt
import numpy as np 
import scipy as sp
import matplotlib.pyplot as plt 
import matplotlib.ticker as mtick
import matplotlib.ticker as ticker
import pickle

from scipy.integrate import solve_ivp
from scipy.linalg import sqrtm, eigvals
from scipy.integrate import simpson

from pandas import DataFrame

from scipy.fft import fft, fftfreq

######
###### The dynamics of the covariance matrix and the mean-field quantities 
######

def func(t, y, ω1x, ω1z, ω2x, ω2z, gx, gy, gz, k1, k2):

    m1x, m1y, m1z, m2x, m2y, m2z = y
    
    #### Mean-field dynamics for system 1
    dm1x = -np.sqrt(2.0) * (gz * m1y * m2z) + np.sqrt(2.0) * (gy * m1z * m2y) - (ω1z * m1y) + np.sqrt(2.0) * (k1 * m1x * m1z)

    dm1y = -np.sqrt(2.0) * (gx * m1z * m2x) + np.sqrt(2.0) * (gz * m1x * m2z) + (ω1z * m1x) - (ω1x * m1z) + np.sqrt(2.0) * (k1 * m1y * m1z)

    dm1z = -np.sqrt(2.0) * (gy * m2y * m1x) + np.sqrt(2.0) * (gx * m1y * m2x) + (ω1x * m1y) - np.sqrt(2.0) * k1 * (m1x**2.0 + m1y**2.0)

    #### Mean-field dynamics for system 2
    dm2x = -np.sqrt(2.0) * (gz * m2y * m1z) + np.sqrt(2.0) * (gy * m2z * m1y) - (ω2z * m2y) + np.sqrt(2.0) *  (k2 * m2x * m2z)

    dm2y = -np.sqrt(2.0) * (gx * m2z * m1x) + np.sqrt(2.0) * (gz * m2x * m1z) + (ω2z * m2x) - (ω2x * m2z) + np.sqrt(2.0) * (k2 * m2y * m2z)

    dm2z = -np.sqrt(2.0) * (gy * m1y * m2x) + np.sqrt(2.0) * (gx * m2y * m1x) + (ω2x * m2y)  - np.sqrt(2.0) * k2 * (m2x**2.0 + m2y**2.0)

    #### Updating

    return dm1x, dm1y, dm1z, dm2x, dm2y, dm2z


#########
#########
#########

def get_traj(tspan, dt, ω1x, ω1z, ω2x, ω2z, gx, gy, gz, k1, k2,
             m1x0, m1y0, m1z0, m2x0, m2y0, m2z0):
    
    
    initial = m1x0, m1y0, m1z0, m2x0, m2y0, m2z0

    p = (ω1x, ω1z, ω2x, ω2z, gx, gy, gz, k1, k2)

    sol = solve_ivp(func, tspan, initial, args=p, method='DOP853', t_eval=np.arange(tspan[0], tspan[1], dt), rtol=1e-10, atol=1e-10)
    print("Status = {}\n".format(sol.status))

    return sol.t, sol.y.T

def Simul_Traj(parameters):

    t, sol = get_traj(**parameters)
    
    return t, sol

####
#### Simulation 
####

def Fourier():
    ω1x = 0.0
    ω1z = 0.0
    
    ω2x = 2.5
    ω2z = 0.0
    
    gx, gy = 2.0, 2.0
    gz = 0.0 
    
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
    
    
    g_list = np.linspace(0, 3, 200)
    modes_1 = []
    modes_2 = []
    
    for i, g in enumerate(g_list):
        print("Iteration {} ".format(i))
        parameters["gx"] = g
        parameters["gy"] = g
        parameters["gz"] = g
        
        times, sol = Simul_Traj(parameters)

        aux_times = times[len(times)//2:]
        aux_mz1 = sol[:,2][len(times)//2:]
        aux_mz2 = sol[:,5][len(times)//2:]

        y1 = fft(aux_mz1 - np.mean(aux_mz1))
        y2 = fft(aux_mz2 - np.mean(aux_mz2))

        xf = fftfreq(len(aux_times), parameters["dt"])[:len(aux_times)//2]
        aux_m1 = 2.0/len(aux_times) * np.abs(y1[0:len(aux_times)//2])
        aux_m2 = 2.0/len(aux_times) * np.abs(y2[0:len(aux_times)//2])

        if max(aux_m1) < 0.00001:
            modes_1.append(aux_m1)
            modes_2.append(aux_m2)
        else: 
            modes_1.append(aux_m1/max(aux_m1))
            modes_2.append(aux_m2/max(aux_m2))


    np.save("test_xf", xf)
    np.save("test_g_list", g_list)
    np.save("test_m1", modes_1)
    np.save("test_m2", modes_2)

    return g_list, xf, modes_1, modes_2


    #data = parameters.copy()
    #data.update({"PhaseMap_result":Heat})
    #data.update({"gz_list": gz_list})
    #data.update({"g_list":g_list})


    #with open('data_phase_maps/Data_heat_gz_g.pickle', 'wb') as handle:
    #    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    #return Heat, parameters

def simulate():
    ω1x = 0.0
    ω1z = 0.0
    
    ω2x = 2.5
    ω2z = 0.0
    
    gx, gy = 1.5, 1.5
    gz = 1.5
    
    k1 = 1.0
    k2 = 1.0
   
    ### Initial state in the ground-state of the bare Hamiltonian
    
    parameters = {"tspan": (0, 5e3),
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
    
    time, sol = Simul_Traj(parameters)
    return time, sol

