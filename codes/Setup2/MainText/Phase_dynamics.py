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

from MeanField import Simul_Traj

def initial_state_seeding_maps():
    ω1x = 0.0
    ω1z = 0.0
    
    ω2x = 2.5
    ω2z = 0.0
    
    gx, gy = 0.8, 0.8
    gz = 0.0 
    
    k1 = 1.0
    k2 = 1.0
   
    ### Initial state in the ground-state of the bare Hamiltonian
    
    parameters = {"tspan": (0, 5e3),
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

########################
########################
########################
########################

def initial_state_phase():
    Omega = 2.5
    J = 2.5 
    k = 1.0
   
    ### Initial state in the ground-state of the bare Hamiltonian
    
    parameters = {"tspan": (1, 1e2),
                      "dt": 0.01,
                      "Omega": Omega,
                      "J": J,
                      "k": k,
                      "f0": 0.0,
                      "g0": 0.0}
    
    return parameters


def func_phase(t, y, J, k, Omega):

    f, g = y
    
    #### Mean-field dynamics 
    df = -J*np.sin(g) - k*np.sin(f)
    dg = +J*np.sin(f) - k*np.sin(g) - Omega

    return df, dg

def get_traj_phase(tspan, dt, J, k, Omega, f0, g0):
    
    initial = f0, g0

    p = (J, k, Omega)

    sol = solve_ivp(func_phase, tspan, initial, args=p, method='DOP853', t_eval=np.arange(tspan[0], tspan[1], dt), rtol=1e-10, atol=1e-10)
    #print("Status = {}\n".format(sol.status))

    return sol.t, sol.y.T

def Simul_Traj_phase():

    parameters = initial_state_phase()
    parameters["f0"] = 0
    parameters["g0"] = 0

    Omega = 2.5

    parameters["J"] = 1.5
    parameters["Omega"] = Omega

    t, sol = get_traj_phase(**parameters)

    mx1 = -np.sqrt(1/2)*np.sin(sol[:,0])
    mz1 = -np.sqrt(1/2)*np.cos(sol[:,0])

    my2 = -np.sqrt(1/2)*np.sin(sol[:,1])
    mz2 = -np.sqrt(1/2)*np.cos(sol[:,1])

    return t, mx1, mz1, my2, mz2, sol[:,0], sol[:,1]

def Traj_Efficiency():

    parameters = initial_state_phase()

    parameters["tspan"] = (0,100)
    parameters["dt"] = 0.01

    parameters["f0"] = 0
    parameters["g0"] = 0

    Omega = 2.5
    parameters["Omega"] = Omega

    
    J_list = [1.0, 2, 2.1, 3]

    mx1_list = []
    my2_list = []
    mz1_list = []
    mz2_list = []
    
    for J in J_list:
        parameters["J"] = J 
        t, sol = get_traj_phase(**parameters)

        mx1 = -np.sqrt(1/2)*np.sin(sol[:,0])
        mz1 = -np.sqrt(1/2)*np.cos(sol[:,0])

        my2 = -np.sqrt(1/2)*np.sin(sol[:,1])
        mz2 = -np.sqrt(1/2)*np.cos(sol[:,1])

        mx1_list.append(mx1)
        my2_list.append(my2)
        mz1_list.append(mz1)
        mz2_list.append(mz2)

    data = parameters.copy()
    data.update({"J_list": J_list})
    data.update({"time": t})
    data.update({"mx1_list":np.array(mx1_list)})
    data.update({"my2_list":np.array(my2_list)})
    data.update({"mz1_list":np.array(mz1_list)})
    data.update({"mz2_list":np.array(mz2_list)})

    with open('Data_seeding_traj_eff.pickle', 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
 
    return t, mx1_list, my2_list, mz1_list, mz2_list


def PhaseDiagram_J_Omega():

    parameters = initial_state_phase()
    t, sol = get_traj_phase(**parameters)

    N = 100

    Omega_list = np.linspace(2.5,0, N)
    J_list = np.linspace(2.5, 0, N)

    mz1_mat = np.zeros((N,N))
    mz2_mat = np.zeros((N,N))

    for i, Omega in enumerate(Omega_list):
        parameters["f0"] = -Omega/(J_list[0]**2 + 1)
        parameters["g0"] = J_list[0]*Omega/(J_list[0]**2 + 1)
        for j, J in enumerate(J_list):
            print(i,j)
            parameters["Omega"] = Omega
            parameters["J"] = J

            t, sol = get_traj_phase(**parameters)

            mz1 = -np.sqrt(1/2)*np.cos(sol[:,0])
            mz2 = -np.sqrt(1/2)*np.cos(sol[:,1])
            
            mz1_mat[i][j] = simpson(mz1**2, t)/t[-1]
            mz2_mat[i][j] = simpson(mz2**2, t)/t[-1]

            parameters["f0"] = sol[:,0][-1]
            parameters["g0"] = sol[:,1][-1]

    data = dict()
    data.update({"Omega_list": Omega_list})
    data.update({"J_list":J_list})
    data.update({"mz1e2_mat":mz1_mat})
    data.update({"mz2e2_mat":mz2_mat})
    with open('Data_seeding_PhaseDiagram_SP_TC.pickle', 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    return Omega_list, J_list, mz1_mat, mz2_mat


def Multistability():

    N = 1000

    J_tc_sp = np.linspace(2.5, 2.0, N)
    J_sp_tc = np.linspace(2.5, 2.0, N)

    ######### 

    parameters_tc_sp = initial_state_phase()
    parameters_sp_tc = initial_state_phase()

    parameters_tc_sp["tspan"] = (0, 1000)
    parameters_sp_tc["tspan"] = (0, 1000)

    parameters_tc_sp["dt"] = 0.01
    parameters_sp_tc["dt"] = 0.01


    parameters_tc_sp["f0"] = 1011.8549960918995
    parameters_tc_sp["g0"] = -2019.982816513612

    parameters_sp_tc["f0"] = 0
    parameters_sp_tc["g0"] = 0


    int_mz_tc_sp = []
    int_mz_sp_tc = []
   
    for n in range(N):

        print("iteration={}".format(n))
        parameters_tc_sp["J"] = J_tc_sp[n]
        parameters_sp_tc["J"] = J_sp_tc[n]
        ###

        t_tc_sp, sol_tc_sp = get_traj_phase(**parameters_tc_sp)
        t_sp_tc, sol_sp_tc = get_traj_phase(**parameters_sp_tc)

        #########

        int_mz_tc_sp.append(simpson(-np.sqrt(1/2)*np.cos(sol_tc_sp[:,0]), t_tc_sp)/np.max(t_tc_sp))
        int_mz_sp_tc.append(simpson(-np.sqrt(1/2)*np.cos(sol_sp_tc[:,0]), t_sp_tc)/np.max(t_sp_tc))

        ##########

        parameters_tc_sp["f0"] = sol_tc_sp[:,0][-1]
        parameters_tc_sp["g0"] = sol_tc_sp[:,1][-1]

        parameters_sp_tc["f0"] = sol_sp_tc[:,0][-1]
        parameters_sp_tc["g0"] = sol_sp_tc[:,1][-1]

        #########

    data = parameters_tc_sp.copy()
    data.update({"tc_sp":[J_tc_sp, int_mz_tc_sp]})
    data.update({"sp_tc":[J_sp_tc, int_mz_sp_tc]})

    #with open('Data_seeding_Bistability.pickle', 'wb') as handle:
    #    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
 
    return J_tc_sp, J_sp_tc, int_mz_tc_sp, int_mz_sp_tc



def Bistability_sp_tc():
    J = np.linspace(2.5, 1.9, 0)
   
    parameters = initial_state_phase()
    t, sol = get_traj_phase(**parameters)

    f = [sol[:,0][-1]] 
    g = [sol[:,1][-1]] 

    pred = []
    int_mz = []
    for j in J:
        parameters["f0"] = f[-1]
        parameters["g0"] = g[-1]
        parameters["J"] = j
       
        t, sol = get_traj_phase(**parameters)
        f.append(sol[:,0][-1])
        g.append(sol[:,1][-1])


        int_mz.append(simpson(-np.sqrt(1/2)*np.cos(np.array(sol[:,0])), t)/np.max(t))
        pred.append(-parameters["k"]*parameters["Omega"]/(parameters["J"]**2 + parameters["k"]**2))
       
    return J, -np.sqrt(1/2)*np.sin(np.array(f[1:])), -np.sqrt(1/2)*np.sin(np.array(g[1:])), -np.sqrt(1/2)* np.array(pred), int_mz

def Bistability_tc_sp():
    J = np.linspace(1.9, 2.5, 30)
   
    parameters = initial_state_phase()
    t, sol = get_traj_phase(**parameters)

    f = [sol[:,0][-1]] 
    g = [sol[:,1][-1]] 

    int_mz = []

    pred = []

    for j in J:
        #parameters["f0"] = f[-1]
        #parameters["g0"] = g[-1]
        parameters["J"] = j
       
        t, sol = get_traj_phase(**parameters)
        f.append(sol[:,0][-1])
        g.append(sol[:,1][-1])

        int_mz.append(simpson(-np.sqrt(1/2)*np.cos(np.array(sol[:,0])), t)/np.max(t))
        pred.append(-parameters["k"]*parameters["Omega"]/(parameters["J"]**2 + parameters["k"]**2))

       
    return J, -np.sqrt(1/2)*np.sin(np.array(f[1:])), -np.sqrt(1/2)*np.sin(np.array(g[1:])), -np.sqrt(1/2)* np.array(pred), int_mz



########################
########################
########################
########################



def simulate():
    ω1x = 0.0
    ω1z = 0.0

    ω2x = 2.5
    ω2z = 0.0

    gx, gy = 2.0, 2.0
    gz = 0

    k1 = 1.0
    k2 = 1.0

    ### Initial state in the ground-state of the bare Hamiltonian

    parameters = {"tspan": (0, 1e2),
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

    time, sol = Simul_Traj(parameters)
    return time, sol

###########
###########
###########
###########
###########




def One_mode_fourier():
    
    parameters = initial_state_seeding_maps()

    g = 1.1
    modes_1 = []
    modes_2 = []
    
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

    return aux_m1, aux_m2



def Fourier():
    
    parameters = initial_state_seeding_maps()

    g_list = np.linspace(0, 4, 100)
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

        if max(aux_m1) < 0.00001:
            modes_1.append(aux_m1)
            modes_2.append(aux_m2)
        else: 
            modes_1.append(aux_m1/max(aux_m1))
            modes_2.append(aux_m2/max(aux_m2))


    #### Saving  

    #data = parameters.copy()

    #data.update({"xf":xf})
    #data.update({"g_list":g_list})
    #data.update({"modes_m1":modes_1})
    #data.update({"modes_m2":modes_2})

    with open('Data_seeding_fourier.pickle', 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return g_list, xf, modes_1, modes_2


