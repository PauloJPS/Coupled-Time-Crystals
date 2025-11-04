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




def func(t, y, k):

    mz = y
    
    #### Mean-field dynamics for system 1
    dmz = - k * np.sqrt(2.0) * (1/2 - mz**2 )

    return dmz


#########
#########
#########

def get_traj(tspan, dt, k, mz0):
    
    initial = [mz0]

    p = (k,)

    sol = solve_ivp(func, tspan, initial, args=p, 
                    method='DOP853', t_eval=np.arange(tspan[0], tspan[1], dt), rtol=1e-10, atol=1e-10)
    print("Status = {}\n".format(sol.status))

    return sol.t, sol.y.T

def Simul_Traj(parameters):

    t, sol = get_traj(**parameters)
    
    return t, sol

####
#### Simulation 
####

def initial_state():
    k = 1.0
   
    ### Initial state in the ground-state of the bare Hamiltonian
    
    parameters = {"tspan": (0, 1e2),
                      "dt": 0.01,
                      "k": k,
                      "mz0": -np.sqrt(1.0/2.0)}
    
    return parameters

def simulate():

    ### Simulate one mean-field trajectory 
    parameters = initial_state()

    parameters["tspan"] = (0, 20)

    #### theta {0, pi}, phi {0, 2pi}

    theta_1 = 0 + 0.0001
    phi_1 = 0
    
    theta_2  = theta_1
    phi_2 =  phi_1

    parameters["mz0"] =  np.sqrt(1/2)*(np.cos(theta_1))

    times, sol = Simul_Traj(parameters)
 
    return times, sol, parameters

def SelfDischargin_Comparison():

    parameters = initial_state()
    parameters = initial_state()

    parameters["tspan"] = (0, 20)

    parameters_SP = parameters.copy()
    parameters_TC = parameters.copy()

    #### Initial state for stationary phase

    Omega_0 = 0.999
    J_0 = 0
    mz0 = -np.sqrt(1/2) * np.sqrt(1 - Omega_0**2/(1 + J_0**2))
   
    parameters_SP["mz0"] =  mz0  
    #### Initial state for the time crystal phase
    delta = 1 - Omega_0 
    theta_TC = np.sqrt(1/2) - delta 
    
    parameters_TC["mz0"] = np.sqrt(1/2) - np.abs(delta) 
    
    t, sol_SP = Simul_Traj(parameters_SP)
    t, sol_TC = Simul_Traj(parameters_TC)


    return t, sol_SP, sol_TC


def Plotting():

    t, sol_SP, sol_TC = SelfDischargin_Comparison()

    fig, ax = plt.subplots( figsize=(4, 2))#, layout="constrained")
    
    ax.plot(t, sol_TC.T[0], color="#e41a1c", linestyle="-", label="$\mathrm{Time Crystal}$")
    ax.plot(t, sol_SP.T[0], color="#377eb8", linestyle="--", label="$\mathrm{Stationary}$")

    ax.legend(edgecolor="black", framealpha=1,handlelength=1.3, borderpad=0.3, fontsize=15, loc=0, 
                    labelspacing=0.1, handletextpad=0.4, ncol=1, columnspacing=0.3) 


    ax.set_xlim((0, 10))
    ax.set_xlabel(r"$t\kappa$")
    ax.set_ylabel(r"$m_z(t)$")

    #plt.savefig("Figures/self_discharging.pdf", bbox_inches="tight")
