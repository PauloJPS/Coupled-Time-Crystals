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

def simulate():

    ### Simulate one mean-field trajectory 
    parameters = initial_state()
    parameters["tspan"] = (0, 500)
    parameters["dt"] = 0.01

    parameters["ω1x"] = 0.0
    parameters["ω2x"] = 3.0

    
    parameters["gx"] = 1.75
    parameters["gy"] = 1.75
    parameters["gz"] = 0.0

    #### theta {0, pi}, phi {0, 2pi}

    theta_1  = 0*(np.pi)
    phi_1 = (2*np.pi)

    theta_2  = theta_1
    phi_2 =  phi_1

    #theta_2  = np.random.rand()*(np.pi)
    #phi_2 = np.random.rand()*(2*np.pi)

    parameters["m1x0"] =  np.sqrt(1/2)*(np.sin(theta_1)*np.cos(phi_1))
    parameters["m1y0"] =  np.sqrt(1/2)*(np.sin(theta_1)*np.sin(phi_1))
    parameters["m1z0"] =  np.sqrt(1/2)*(np.cos(theta_1))

    parameters["m2x0"] =  np.sqrt(1/2)*(np.sin(theta_2)*np.cos(phi_2))
    parameters["m2y0"] =  np.sqrt(1/2)*(np.sin(theta_2)*np.sin(phi_2))
    parameters["m2z0"] =  np.sqrt(1/2)*(np.cos(theta_2))

    times, sol = Simul_Traj(parameters)
 
    return times, sol, parameters



def MeanField_PD_Work():

    ### Fig 4(a)

    parameters = initial_state()
    parameters["tspan"] = (0, 1000)
    parameters["dt"] = 0.1

    ### Parameter list 
    N = 100
    J_list = np.linspace(0, 3, N)
    Omega_list = np.linspace(0, 3, N)

    mz1e2_mat = np.zeros((N,N))
    mz2e2_mat = np.zeros((N,N))


    for i, Omega in enumerate(Omega_list):
        for j, J in enumerate(J_list):
            print(i,j)
            parameters["ω2x"] = Omega
            parameters["gx"], parameters["gy"]  = J, J 
            t, sol =  Simul_Traj(parameters)
            
            ######## 
            mz1e2_mat[i][j] = simpson(sol[:,2]**2, t)/t[-1]
                                            
            mz2e2_mat[i][j] = simpson(sol[:,5]**2, t)/t[-1]

            ######## 
       
    ## saving 
    data = dict()
    data.update({"Omega_list": Omega_list})
    data.update({"J_list":J_list})
    data.update({"mz1e2_mat":mz1e2_mat})
    data.update({"mz2e2_mat":mz2e2_mat})
    with open('Data_seeding_PhaseDiagram_SP_TC_test.pickle', 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
       
    return Omega_list, J_list, mz1e2_mat, mz2e2_mat


def Traj_Efficiency():

    ### Fig 4(c)

    parameters = initial_state()

    parameters["tspan"] = (0,100)
    parameters["dt"] = 0.01
    parameters["ω1x"] = 0
    parameters["ω2x"] = 2.5
    
    J_list = [1.0, 1.9, 2.1, 3]

    mx1_list = []
    my2_list = []
    mz1_list = []
    mz2_list = []
    
    for J in J_list:
        parameters["gx"], parameters["gy"] = J, J
        t, sol = Simul_Traj(parameters)

        mx1 = sol[:,0]
        mz1 = sol[:,2]

        my2 = sol[:,4]
        mz2 = sol[:,5]

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

    with open('Data/Data_seeding_traj_eff.pickle', 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
 
    return t, mx1_list, my2_list, mz1_list, mz2_list


def Sampling_Tcs():

    ### Simulate one mean-field trajectory 
    parameters = initial_state()
    parameters["ω1x"] = 2
    parameters["ω2x"] = 2

    parameters["gx"] = 1
    parameters["gy"] = 1
    parameters["gz"] = 0.5

    N_conditions = 100

    solutions_x = []
    solutions_y = []
    solutions_z = []
    
    for n in range(N_conditions):

        #### theta {0, pi}, phi {0, 2pi}
        theta_1  = np.random.rand()*(np.pi)
        phi_1 = np.random.rand()*(2*np.pi)

        theta_2  = theta_1
        phi_2 =  phi_1

        parameters["m1x0"] =  np.sqrt(1/2)*(np.sin(theta_1)*np.cos(phi_1))
        parameters["m1y0"] =  np.sqrt(1/2)*(np.sin(theta_1)*np.sin(phi_1))
        parameters["m1z0"] =  np.sqrt(1/2)*(np.cos(theta_1))

        parameters["m2x0"] =  np.sqrt(1/2)*(np.sin(theta_2)*np.cos(phi_2))
        parameters["m2y0"] =  np.sqrt(1/2)*(np.sin(theta_2)*np.sin(phi_2))
        parameters["m2z0"] =  np.sqrt(1/2)*(np.cos(theta_2))

        times, sol = Simul_Traj(parameters)

        solutions_x.append(simpson(sol[:,0], times))
        solutions_y.append(simpson(sol[:,1], times))
        solutions_z.append(simpson(sol[:,2], times))
 
    return solutions_x, solutions_y, solutions_z



def Sampling_SP():

    ### Simulate one mean-field trajectory 
    parameters = initial_state()
    parameters["ω1x"] = 2
    parameters["ω2x"] = 2

    parameters["gx"] = 4
    parameters["gy"] = 4
    parameters["gz"] = 0.1

    N_conditions = 10

    solutions_x1 = []
    solutions_y1 = []
    solutions_z1 = []

    solutions_x2 = []
    solutions_y2 = []
    solutions_z2 = []
 
    
    for n in range(N_conditions):

        #### theta {0, pi}, phi {0, 2pi}
        theta_1  = np.random.rand()*(np.pi)
        phi_1 = np.random.rand()*(2*np.pi)

        theta_2  = np.random.rand()*(np.pi)
        phi_2 = np.random.rand()*(2*np.pi)

        parameters["m1x0"] =  np.sqrt(1/2)*(np.sin(theta_1)*np.cos(phi_1))
        parameters["m1y0"] =  np.sqrt(1/2)*(np.sin(theta_1)*np.sin(phi_1))
        parameters["m1z0"] =  np.sqrt(1/2)*(np.cos(theta_1))

        parameters["m2x0"] =  np.sqrt(1/2)*(np.sin(theta_2)*np.cos(phi_2))
        parameters["m2y0"] =  np.sqrt(1/2)*(np.sin(theta_2)*np.sin(phi_2))
        parameters["m2z0"] =  np.sqrt(1/2)*(np.cos(theta_2))

        times, sol = Simul_Traj(parameters)

        aux_dmx = np.gradient(sol[:,0], times)
        aux_dmy = np.gradient(sol[:,1], times) 
        aux_dmz = np.gradient(sol[:,2], times) 

        if np.abs(aux_dmx[-1]) < 0.001 and np.abs(aux_dmz[-1]) < 0.001 and np.abs(aux_dmz[-1]) < 0.001 : 
            print("OI")
            solutions_x1.append(sol[:,0])
            solutions_y1.append(sol[:,1])
            solutions_z1.append(sol[:,2])

            solutions_x2.append(sol[:,3])
            solutions_y2.append(sol[:,4])
            solutions_z2.append(sol[:,5])

    my_pred = np.sqrt(1/2) * parameters["k1"] * parameters["ω1x"] / ((parameters["gx"]-parameters["gz"])**2 + parameters["k1"]**2)
    mz_pred = -np.sqrt(1/2) * np.sqrt(1 - parameters["ω1x"]**2 / ((parameters["gx"]-parameters["gz"])**2 + parameters["k1"]**2))
 
    return solutions_x1, solutions_y1, solutions_z1, solutions_x2, solutions_y2, solutions_z2, my_pred, mz_pred


def Conserved_Quantity(mx, my, parameters):

    
    ### Calculate the conserved quantity for setup 1 defined bellow Eq(11) in the main text 

    k, J, Jz, Omega = parameters["k1"], parameters["gx"], parameters["gz"], parameters["ω1x"]
    print(J, Jz)
    lambda_p = np.sqrt(2)*(k + 1.j*(J-Jz))
    lambda_m = np.sqrt(2)*(k - 1.j*(J-Jz))

    eta_p = (my + 1.j*mx)/np.sqrt(2)
    eta_m = (my - 1.j*mx)/np.sqrt(2)
    
    M = (lambda_m*np.log(lambda_p*eta_p - Omega/np.sqrt(2)) - lambda_p*np.log(lambda_m*eta_m - Omega/np.sqrt(2)))

    return M 


def AdiabaticFollowing():

    ### Changing parameter g adiabatically and observing the solution to check for bistability
    parameters = initial_state()
    parameters["ω1x"] = 2.0
    parameters["ω2x"] = 2.0

    
    parameters["gx"] = 4
    parameters["gy"] = 4
    parameters["gz"] = 0.0

    parameters["m1x0"] =-0.332756132305987
    parameters["m1y0"] = 0.0831890330728998
    parameters["m1z0"] =-0.6183469423478889
                        
    parameters["m2x0"] =-0.332756132305987
    parameters["m2y0"] = 0.0831890330728998
    parameters["m2z0"] =-0.6183469423478889   

    times, sol = Simul_Traj(parameters)

    g_list = np.linspace(4,1.01, 100) 

    sol_list = []
    for i, g in enumerate(g_list): 
        print(i) 
        parameters["gx"] = g
        parameters["gy"] = g

        times, sol = Simul_Traj(parameters)
        sol_list.append([sol[:,0][-1], sol[:,1][-1], sol[:,2][-1]])
         

        parameters["m1x0"] = sol[:,0][-1]
        parameters["m1y0"] = sol[:,1][-1]
        parameters["m1z0"] = sol[:,2][-1]
                             
        parameters["m2x0"] = sol[:,3][-1]
        parameters["m2y0"] = sol[:,4][-1]
        parameters["m2z0"] = sol[:,5][-1]

    return sol_list


def One_mode_fourier():

    ### Fourier Analysis 

    parameters = initial_state()

    modes_1 = []
    modes_2 = []
    
    parameters["ω1x"] = 0.0
    parameters["ω2x"] = 2.5

    parameters["gx"] = 2.1
    parameters["gy"] = 2.1
    parameters["gz"] = 0.0

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



