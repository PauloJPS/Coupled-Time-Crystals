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

plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage[T1]{fontenc}\usepackage{amsmath}\usepackage{amsfonts}\usepackage{bbm}",
    #
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,

    "ytick.right": True,
    "ytick.left": True,

    "xtick.top": True,
    "xtick.bottom": True,

    #
    "xtick.direction": "in",
    "ytick.direction": "in",
    #
    "xtick.major.width": 1.5,     # major tick width in points
    "ytick.major.width": 1.5,     # major tick width in points
    #
    "xtick.minor.width": 1.5,     # minor tick width in points
    "ytick.minor.width": 1.5,     # minor tick width in points
    #
    "xtick.major.pad": 3.0,     # distance to major tick label in points
    "ytick.major.pad": 3.0,     # distance to major tick label in points
    #
    "xtick.minor.pad": 1.4,     # distance to the minor tick label in points
    "ytick.minor.pad": 1.4,     # distance to the minor tick label in points
    #
    "xtick.major.size": 5.5,
    "ytick.major.size": 5.5,

    "xtick.minor.size": 3.5,
    "ytick.minor.size": 3.5,
    #
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    #
    "legend.frameon": True,
    "legend.fontsize": 15,
    "legend.edgecolor": "white",
    "axes.titlesize": 15,
    "axes.titleweight": "bold",
    "axes.labelsize":15
})



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

def Correlations_and_MeanField():

    parameters = initial_state()

    parameters["tspan"] = (0, 100)
    parameters["dt"] = 0.01

    parameters["ω1x"] = 0
    parameters["ω2x"] = 3.0
    parameters["gz"] = 0
    parameters["gy"] = 1
    parameters["gx"] = 1
 
    ### Time interval to avarege
    tlen = parameters["tspan"][1]
    Nint = int(tlen/parameters["dt"])
 
    ### getting the solution 
    time, sol = Simul_Traj(parameters)

    magnetizations = MeanField(sol)

    information = QuantumClassicalThermodynamics(sol)
   
    return magnetizations, information

def Fourier_traj_MF_Cor():

    #### Function Appendix (fourier spectrum in fuction of J)

    parameters = initial_state()

    parameters["tspan"] = (0, 2000)
    parameters["dt"] = 0.1

    parameters["ω2x"] = 3
    parameters["ω1x"] = 0
    parameters["gz"] = 0
    parameters["gx"] = 1
    parameters["gy"] = 1

    parameters["m1x0"], parameters["m1z0"], parameters["m1z0"] = 0, 0, -1/np.sqrt(2)
    parameters["m2x0"], parameters["m2y0"], parameters["m2z0"] = 0, 0, -1/np.sqrt(2)

    times, sol = Simul_Traj(parameters)
    magnetizations = MeanField(sol)
    information = QuantumClassicalThermodynamics(sol)

    aux_times = times[len(times)//2:]
    aux_mx1 = sol[:,0][len(times)//2:]
    aux_mz1 = sol[:,2][len(times)//2:]

    aux_CC = information[0][len(times)//2:]
    aux_QD = information[1][len(times)//2:]

    mx1 = fft(aux_mx1 - np.mean(aux_mx1))
    mz1 = fft(aux_mz1 - np.mean(aux_mz1))
    CC = fft(aux_CC - np.mean(aux_CC))
    QD = fft(aux_QD - np.mean(aux_QD))

    aux_mx1 = 2.0/len(aux_times) * np.abs(mx1[0:len(aux_times)//2])
    aux_mz1 = 2.0/len(aux_times) * np.abs(mz1[0:len(aux_times)//2])
    aux_CC = 2.0/len(aux_times) * np.abs(CC[0:len(aux_times)//2])
    aux_QD = 2.0/len(aux_times) * np.abs(QD[0:len(aux_times)//2])

    modes_mx1 = aux_mx1/max(aux_mx1)
    modes_mz1 = aux_mz1/max(aux_mz1)
    modes_CC = aux_CC/max(aux_CC)
    modes_QD = aux_QD/max(aux_QD)

    xf = fftfreq(len(aux_times), parameters["dt"])[:len(aux_times)//2]

    #### Saving

    data = parameters.copy()

    data.update({"xf":xf})
    data.update({"modes_mx1":modes_mx1})
    data.update({"modes_mz1":modes_mz1})
    data.update({"modes_CC":modes_CC})
    data.update({"modes_QD":modes_QD})

    with open('Data/Data_seeding_modes_mf_cor.pickle', 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return xf, modes_mx1, modes_mz1, modes_CC, modes_QD



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

    parameters["tspan"] = (0, 30)
    parameters["dt"] = 0.1

    parameters["ω1x"] = 0
    parameters["ω2x"] = 3
    parameters["gz"] = 0

    ### Time interval to avarege
    tlen = parameters["tspan"][1]
    Nint = int(tlen/parameters["dt"])
 
    J_list = [1, 2, 3] 

    Nega_list = []
    Clas_list = []
    Quan_list = []
    for J in J_list:
        parameters["gy"] = J
        parameters["gx"] = J

        ### getting the solution 
        time, sol = Simul_Traj(parameters)

        ## getting the information theory quantities
        information = QuantumClassicalThermodynamics(sol)

        Clas_list.append(information[0])
        Quan_list.append(information[1])
        Nega_list.append(information[2])
    data = parameters.copy()
    data.update({"time":time})
    data.update({"J_list":J_list})
    data.update({"Clas_list":Clas_list})
    data.update({"Quan_list":Quan_list})
    data.update({"Nega_list":Nega_list})
    with open('Data/Data_Nega_trajectories.pickle', 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    return time, Clas_list, Quan_list, Nega_list

def plot():
   
    with open("Data/Data_Nega_trajectories.pickle", "rb") as handle:
        data = pickle.load(handle)


    time = data["time"]
    Clas_list = data["Clas_list"] 
    Quan_list = data["Quan_list"] 
    Nega_list = data["Nega_list"] 

    fig, ax = plt.subplots(1,3, figsize=(8, 2))#, layout="constrained")

    ax[0].plot(time, Clas_list[0], color="#e41a1c", linestyle="-", label=r"$J = \kappa$")
    ax[0].plot(time, Clas_list[1], color="#377eb8", linestyle="-", label=r"$J = 2\kappa$")
    ax[0].plot(time, Clas_list[2], color="#4daf4a", linestyle="-", label=r"$J = 3\kappa$")

    ax[0].set_xlim((0, 20))
    ax[0].set_ylim(bottom=0)
    ax[0].set_xlabel(r"$t \kappa$")
    ax[0].set_ylabel(r"$\mathcal{J}$")
    #### 
    ax[1].plot(time, Quan_list[0], color="#e41a1c", linestyle="-", label=r"$J = \kappa$")
    ax[1].plot(time, Quan_list[1], color="#377eb8", linestyle="-", label=r"$J = 2\kappa$")
    ax[1].plot(time, Quan_list[2], color="#4daf4a", linestyle="-", label=r"$J = 3\kappa$")

    ax[1].set_xlim((0, 20))
    ax[1].set_ylim(bottom=0)
    ax[1].set_xlabel(r"$t \kappa$")
    ax[1].set_ylabel(r"$\mathcal{D}$")


    #### 
    ax[2].plot(time, Nega_list[0], color="#e41a1c", linestyle="-", label=r"$J = \kappa$")
    ax[2].plot(time, Nega_list[1], color="#377eb8", linestyle="-", label=r"$J = 2\kappa$")
    ax[2].plot(time, Nega_list[2], color="#4daf4a", linestyle="-", label=r"$J = 3\kappa$")

    ax[2].set_xlim((0, 20))
    ax[2].set_ylim(bottom=0)
    ax[2].set_xlabel(r"$t \kappa$")
    ax[2].set_ylabel(r"$\mathcal{N}$")

    ax[2].legend(edgecolor="black", framealpha=1,handlelength=1.3, 
                 borderpad=0.3, fontsize=15, loc=1, labelspacing=0.1, handletextpad=0.4, ncol=1, columnspacing=0.3)

    plt.tight_layout()
    plt.savefig("Figures/Trajectories.pdf", bbox_inches="tight")
