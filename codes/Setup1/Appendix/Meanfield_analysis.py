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
    parameters["ω1x"] = 1.0001
    parameters["ω2x"] = 1.0001

    
    parameters["gx"] = 0.0
    parameters["gy"] = 0.0
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

def TimeCrystal_Familiy():


    ### FIGURE ? For now at the appendix


    ### Simulate one mean-field trajectory 
    parameters = initial_state()

    parameters["tspan"] = (0, 10)
    parameters["dt"] = 0.001

    parameters["ω1x"] = 2.0
    parameters["ω2x"] = 2.0

    
    parameters["gx"] = 2.0
    parameters["gy"] = 2.0
    parameters["gz"] = 1.0

    #### theta {0, pi}, phi {0, 2pi}

    theta_1  = np.pi/2#*(2*np.random.rand()-1)/2
    theta_2  = theta_1

    phi_list = np.linspace(0, np.pi/6, 10)

    mx_list = []
    my_list = []
    mz_list = []
    for phi_1 in phi_list: 
        phi_2 =  phi_1

        parameters["m1x0"] =  np.sqrt(1/2)*(np.sin(theta_1)*np.cos(phi_1))
        parameters["m1y0"] =  np.sqrt(1/2)*(np.sin(theta_1)*np.sin(phi_1))
        parameters["m1z0"] =  np.sqrt(1/2)*(np.cos(theta_1))

        parameters["m2x0"] =  np.sqrt(1/2)*(np.sin(theta_2)*np.cos(phi_2))
        parameters["m2y0"] =  np.sqrt(1/2)*(np.sin(theta_2)*np.sin(phi_2))
        parameters["m2z0"] =  np.sqrt(1/2)*(np.cos(theta_2))

        times, sol = Simul_Traj(parameters)
        mx_list.append(sol[:,0])
        my_list.append(sol[:,1])
        mz_list.append(sol[:,2])

    data = parameters.copy()
    data.update({"phi_list":phi_list})
    data.update({"mx_list":mx_list})
    data.update({"my_list":my_list})
    data.update({"mz_list":mz_list})
    data.update({"times":times})
    with open('Data/Data_TC_family.pickle', 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return mx_list, my_list, mz_list, phi_list, times


def Efficiency_SP_trajs():

    ### Simulate one mean-field trajectory 
    parameters = initial_state()

    parameters["tspan"] = (0,100)
    parameters["dt"] = 0.01

    parameters["ω1x"] = 2.0
    parameters["ω2x"] = 2.0

    
    parameters["gx"] = 0.0
    parameters["gy"] = 0.0
    parameters["gz"] = 0.0

    #### theta {0, pi}, phi {0, 2pi}

    theta_1  = np.pi
    theta_2  = theta_1

    phi_1 = 0
    phi_2 = 0

    parameters["m1x0"] = np.sqrt(1/2)*(np.sin(theta_1)*np.cos(phi_1))
    parameters["m1y0"] = np.sqrt(1/2)*(np.sin(theta_1)*np.sin(phi_1))
    parameters["m1z0"] = np.sqrt(1/2)*(np.cos(theta_1))

    parameters["m2x0"] = parameters["m1x0"]
    parameters["m2y0"] = parameters["m1y0"]
    parameters["m2z0"] = parameters["m1z0"]

    efficiency_list = []
    J_list = [(2,0), (0,0)]

    for J in J_list:

        parameters["gx"] = J[0] 
        parameters["gy"] = J[0]
        parameters["gz"] = J[1]

        times, sol = Simul_Traj(parameters)
    
        mx, my, mz = sol[:,0], sol[:,1], sol[:,2]

        energy_time = 2*np.sqrt(1/2)*(mz -mz[0])

        heat_time = np.array([2*simpson(mx[0:j+1]**2 + my[0:j+1]**2, times[0:j+1]) for j in range(len(times)-1)])

        efficiency_list.append(energy_time[0:-1]/(energy_time[:-1] + heat_time))


    data = parameters.copy()
    data.update({"J_list":J_list})
    data.update({"efficiency_list":efficiency_list})
    data.update({"times":times})
    with open('Data/Data_Efficiency_trajs_J.pickle', 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return times, efficiency_list

