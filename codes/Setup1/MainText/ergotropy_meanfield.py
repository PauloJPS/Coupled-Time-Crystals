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
    parameters["ω1x"] = 2.0
    parameters["ω2x"] = 2.0

    
    parameters["gx"] = 2.0
    parameters["gy"] = 2.0
    parameters["gz"] = 1.0

    #### theta {0, pi}, phi {0, 2pi}

    theta_1  = np.pi/2#np.random.rand()*(np.pi)
    phi_1 = np.random.rand()*(2*np.pi)
    phi_1 = np.pi/6
    
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


def MeanField_PD():

    #### FIGURE 2(a)

    parameters_tc = initial_state()
    parameters_sp = initial_state()

    #### theta {0, pi}, phi {0, 2pi}

    theta_1  = np.pi
    phi_1 = 2*np.pi

    theta_2  = theta_1
    phi_2 =  phi_1

    #####

    parameters_tc["m1x0"] =  np.sqrt(1/2)*(np.sin(theta_1)*np.cos(phi_1))
    parameters_tc["m1y0"] =  np.sqrt(1/2)*(np.sin(theta_1)*np.sin(phi_1))
    parameters_tc["m1z0"] =  np.sqrt(1/2)*(np.cos(theta_1))

    parameters_tc["m2x0"] =  np.sqrt(1/2)*(np.sin(theta_2)*np.cos(phi_2))
    parameters_tc["m2y0"] =  np.sqrt(1/2)*(np.sin(theta_2)*np.sin(phi_2))
    parameters_tc["m2z0"] =  np.sqrt(1/2)*(np.cos(theta_2))


    parameters_sp["m1x0"] =  np.sqrt(1/2)*(np.sin(theta_1)*np.cos(phi_1))
    parameters_sp["m1y0"] =  np.sqrt(1/2)*(np.sin(theta_1)*np.sin(phi_1))
    parameters_sp["m1z0"] =  np.sqrt(1/2)*(np.cos(theta_1))

    parameters_sp["m2x0"] =  np.sqrt(1/2)*(np.sin(theta_2)*np.cos(phi_2))
    parameters_sp["m2y0"] =  np.sqrt(1/2)*(np.sin(theta_2)*np.sin(phi_2))
    parameters_sp["m2z0"] =  np.sqrt(1/2)*(np.cos(theta_2))
    
    parameters_tc["ω1x"] = 2.0
    parameters_tc["ω2x"] = 2.0

    parameters_sp["ω1x"] = 0.5
    parameters_sp["ω2x"] = 0.5

    ### Time interval to avarege
    tlen = parameters_tc["tspan"][1]
    Nint = int(tlen/parameters_tc["dt"])

    ### Parameter list 
    N = 20
    g_list = np.linspace(0, 4, N)
    gz_list = np.linspace(0, 4, N)

    work_tc = np.zeros((N, N))
    work_sp = np.zeros((N, N))


    for i, g in enumerate(g_list):
        for j, gz in enumerate(gz_list):
            print("Iteration {}-{} ".format(i,j))

            parameters_tc["gx"] = g
            parameters_tc["gy"] = g
            parameters_tc["gz"] = gz

            parameters_sp["gx"] = g
            parameters_sp["gy"] = g
            parameters_sp["gz"] = gz
            
            #### Initial conditions 
            J_z, J, Omega, k = gz, g, parameters_tc["ω1x"], parameters_tc["k1"]

            mx = + np.sqrt(0.5)*(Omega * (J_z - J)/((J_z-J)**2 + k**2))
            my = + np.sqrt(0.5)*(Omega * k/((J_z-J)**2 + k**2))
            mz = - np.sqrt(0.5)*np.sqrt(1 - Omega**2/((J_z-J)**2 + k**2))

            if 1 < Omega**2/((J_z-J)**2 + k**2):
                print("oi - {} {}".format(g, gz))
                mx, my, mz = 0, 0, -1/np.sqrt(2)

            parameters_tc["m1x0"] = mx
            parameters_tc["m1y0"] = my
            parameters_tc["m1z0"] = mz
                                    
            parameters_tc["m2x0"] = mx
            parameters_tc["m2y0"] = my
            parameters_tc["m2z0"] = mz
                                    
                                    
            parameters_sp["m1x0"] = mx
            parameters_sp["m1y0"] = my
            parameters_sp["m1z0"] = mz
                                    
            parameters_sp["m2x0"] = mx
            parameters_sp["m2y0"] = my
            parameters_sp["m2z0"] = mz           

            times_tc, sol_tc = Simul_Traj(parameters_tc)
            times_sp, sol_sp = Simul_Traj(parameters_sp)

            ### Sub dominant heat current 

            total_work_tc = (sol_tc[:,0]**2 + sol_tc[:,1]**2 + sol_tc[:,3]**2 + sol_tc[:,4]**2)
            total_work_sp = (sol_sp[:,0]**2 + sol_sp[:,1]**2 + sol_sp[:,3]**2 + sol_sp[:,4]**2)

            ### getting time interval (last half)
            Nint_init = int((1/2)*Nint)

            total_work_tc = (1/(times_tc[-1] - times_tc[Nint_init]) 
                             * simpson(total_work_tc[Nint_init:], dx=parameters_tc["dt"]))
            total_work_sp = (1/(times_sp[-1] - times_sp[Nint_init]) 
                             * simpson(total_work_sp[Nint_init:], dx=parameters_sp["dt"]))
    
            work_tc[i][j] = total_work_tc
            work_sp[i][j] = total_work_sp

    #### Saving  

    data = parameters_tc.copy()

    data.update({"g_list":g_list})
    data.update({"gz_list":gz_list})
    data.update({"work_tc":work_tc})
    data.update({"work_sp":work_sp})

    with open('Data_mean_field_test.pickle', 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return g_list, gz_list, work_tc, work_sp


def Evaluating_Bistability():

    ### Simulate one mean-field trajectory 
    parameters = initial_state()

    parameters["tspan"] = (0, 1000)
    parameters["dt"] = 0.1

    Omega = 2.0
    J = 4.0
    Jz = 1.0
    k = 1.0

    parameters["ω1x"] =  Omega
    parameters["ω2x"] = Omega 

    parameters["k1"] = k
    parameters["k2"] = k

    parameters_tc2sp = parameters.copy()
    parameters_sp2tc = parameters.copy()

    N = 100
    g_list_tc2sp = np.linspace(2.0,4.0, N) 
    g_list_sp2tc = np.linspace(4,2.0, N) 

    ergotropy_tc2sp = []
    ergotropy_sp2tc = []
    ergotropy_pred = []

    ### Time interval to avarege
    tlen = parameters["tspan"][1]
    Nint = int(tlen/parameters["dt"])

    for i in range(N):
        print(i) 

        J_tc = g_list_tc2sp[i]
        J_sp = g_list_sp2tc[i]

        parameters_tc2sp["gx"] = J_tc 
        parameters_tc2sp["gy"] = J_tc 

        parameters_sp2tc["gx"] = J_sp
        parameters_sp2tc["gy"] = J_sp
        
        #####
        times, sol_tc2sp = Simul_Traj(parameters_tc2sp)
        times, sol_sp2tc = Simul_Traj(parameters_sp2tc)

        E_tc = 1/np.sqrt(2)*(sol_tc2sp[:,2][Nint//2:] + 1/np.sqrt(2))
        E_sp = 1/np.sqrt(2)*(sol_sp2tc[:,2][Nint//2:] + 1/np.sqrt(2))

        ergotropy_tc2sp.append(simpson(E_tc, times[Nint//2:])/(times[-1]-times[Nint//2]))
        ergotropy_sp2tc.append(simpson(E_sp, times[Nint//2:])/(times[-1]-times[Nint//2]))
        ####

    data = parameters.copy()

    data.update({"g_list_tc2sp":g_list_tc2sp})
    data.update({"g_list_sp2tc":g_list_sp2tc})
    data.update({"erg_tc":ergotropy_tc2sp})
    data.update({"erg_sp":ergotropy_sp2tc})

    with open('Data_erg_bistable.pickle', 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return ergotropy_tc2sp, ergotropy_sp2tc, g_list_tc2sp, g_list_sp2tc


def AdiabaticFollowing():

    ### Simulate one mean-field trajectory 
    parameters = initial_state()

    Omega = 2.0
    J = 4.0
    Jz = 1.0
    k = 1.0

    parameters["ω1x"] =  Omega
    parameters["ω2x"] = Omega 

    parameters["k1"] = k
    parameters["k2"] = k

    parameters["m1x0"] = np.sqrt(1/2) * ( Omega * (Jz-J) / ((Jz-J)**2 + k**2))
    parameters["m1y0"] = np.sqrt(1/2) * ( Omega * k / ((Jz-J)**2 + k**2))
    parameters["m1z0"] = -np.sqrt(1/2) * np.sqrt( 1-  Omega**2 / ((Jz-J)**2 + k**2))
                        
    parameters["m2x0"] = np.sqrt(1/2) * ( Omega * (Jz-J) / ((Jz-J)**2 + k**2))
    parameters["m2y0"] = np.sqrt(1/2) * ( Omega * k / ((Jz-J)**2 + k**2))
    parameters["m2z0"] =-np.sqrt(1/2) * np.sqrt( 1-  Omega**2 / ((Jz-J)**2 + k**2))

    parameters_tc2sp = parameters.copy()
    parameters_sp2tc = parameters.copy()

    N = 1000
    g_list_tc2sp = np.linspace(2.0,4.0, N) 
    g_list_sp2tc = np.linspace(4,2.0, N) 

    ergotropy_tc2sp = []
    ergotropy_sp2tc = []
    ergotropy_pred = []

    ### Time interval to avarege
    tlen = parameters["tspan"][1]
    Nint = int(tlen/parameters["dt"])

    for i in range(N):
        print(i) 

        J_tc = g_list_tc2sp[i]
        J_sp = g_list_sp2tc[i]

        parameters_tc2sp["gx"] = J_tc 
        parameters_tc2sp["gy"] = J_tc 

        parameters_sp2tc["gx"] = J_sp
        parameters_sp2tc["gy"] = J_sp

        
        #####
        times, sol_tc2sp = Simul_Traj(parameters_tc2sp)
        times, sol_sp2tc = Simul_Traj(parameters_sp2tc)

        E_tc = (sol_tc2sp[:,2][Nint//2:] + 1/np.sqrt(2))
        E_sp = (sol_sp2tc[:,2][Nint//2:] + 1/np.sqrt(2))

        ergotropy_tc2sp.append(simpson(E_tc, times)/times[-1])
        ergotropy_sp2tc.append(simpson(E_sp, times)/times[-1])
        ####
        parameters_tc2sp["m1x0"] = sol_tc2sp[:,0][-1]
        parameters_tc2sp["m1y0"] = sol_tc2sp[:,1][-1]
        parameters_tc2sp["m1z0"] = sol_tc2sp[:,2][-1]

        parameters_tc2sp["m2x0"] = sol_tc2sp[:,3][-1]
        parameters_tc2sp["m2y0"] = sol_tc2sp[:,4][-1]
        parameters_tc2sp["m2z0"] = sol_tc2sp[:,5][-1]
        ####
        
        parameters_sp2tc["m1x0"] = sol_sp2tc[:,0][-1]
        parameters_sp2tc["m1y0"] = sol_sp2tc[:,1][-1]
        parameters_sp2tc["m1z0"] = sol_sp2tc[:,2][-1]

        parameters_sp2tc["m2x0"] = sol_sp2tc[:,3][-1]
        parameters_sp2tc["m2y0"] = sol_sp2tc[:,4][-1]
        parameters_sp2tc["m2z0"] = sol_sp2tc[:,5][-1]

    
    data = parameters.copy()

    data.update({"g_list_tc2sp":g_list_tc2sp})
    data.update({"g_list_sp2tc":g_list_sp2tc})
    data.update({"erg_tc":ergotropy_tc2sp})
    data.update({"erg_sp":ergotropy_sp2tc})

    with open('Data_erg.pickle', 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return ergotropy_tc2sp, ergotropy_sp2tc, g_list_tc2sp, g_list_sp2tc

def Comparison_SS_TC_efficiency():

    ### FIGURE 3 (b)-(c)

    ### Simulate one mean-field trajectory 
    parameters_tc = initial_state()

    parameters_tc["tspan"] = (0, 100)
    parameters_tc["dt"] = 0.1

    parameters_ss = parameters_tc.copy()

    parameters_tc["ω1x"] = 2 
    parameters_tc["ω2x"] = 2 

    parameters_ss["ω1x"] = 2 
    parameters_ss["ω2x"] = 2 
    
    
    parameters_tc["gx"] = 3.4
    parameters_tc["gy"] = 3.4
    parameters_tc["gz"] = 1.0 

    parameters_ss["gx"] = 3.41
    parameters_ss["gy"] = 3.41
    parameters_ss["gz"] = 1.0

    ########     
    times, sol = Simul_Traj(parameters_tc)

    mx_tc = sol[:,0]
    my_tc = sol[:,1]
    mz_tc = sol[:,2]

    ########     

    times, sol = Simul_Traj(parameters_ss)

    mx_ss = sol[:,0]
    my_ss = sol[:,1]
    mz_ss = sol[:,2]

    data = dict()
    data.update({"dic_tc":parameters_tc})
    data.update({"dic_sp":parameters_ss})

    data.update({"mx_tc":mx_tc})
    data.update({"my_tc":my_tc})
    data.update({"mz_tc":mz_tc})

    data.update({"mx_ss":mx_ss})
    data.update({"my_ss":my_ss})
    data.update({"mz_ss":mz_ss})

    data.update({"times":times})
    with open('Data_efficiency_sp_tp.pickle', 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


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

    theta_1  = np.pi#*(2*np.random.rand()-1)/2
    theta_2  = theta_1

    #mid_value = np.pi/6  
    #start = 0  
    #end = 2*np.pi/6  
    #
    #first_half = np.linspace(start, mid_value, num=5, endpoint=True)
    #second_half = np.linspace(mid_value, end, num=6, endpoint=True)[1:]
    #
    #phi_list = np.concatenate([first_half, second_half])

    random = np.random.default_rng(1)

    phi_list = 2*np.pi*random.random(10)

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



