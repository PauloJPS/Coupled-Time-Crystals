####### execute using "python m.py"
####### 21/09/2024 #########

import time
import numpy as np
import math as mt
import scipy as sp
from scipy import linalg as LA
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rc
import matplotlib as mpl

import matplotlib.pyplot as plt 
import matplotlib.ticker as mtick
import matplotlib.gridspec as gridspec

import matplotlib.ticker as ticker
import matplotlib.pyplot as plt

import matplotlib.cm as cm

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import pickle

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



###########################################################
###########################################################
############# MF equations

res=np.loadtxt("Data/phase_diagram_setup2_var_method.dat")


with open("Data/Data_bistable_trajectories.pickle", "rb") as handle:
    data_traj = pickle.load(handle)


fig, ax = plt.subplots(1,2, figsize=(8, 2))

########

pos=ax[0].scatter(res[:,1],res[:,0],c=np.log(np.abs(res[:,3])+1e-8),marker='s', cmap=plt.cm.viridis_r)
cbar = fig.colorbar(pos)

cbar.ax.set_title(r"$\log(\sigma)$", y=-0.2, pad=-0, loc="left")

ax[0].set_xlabel(r"$\Omega/\kappa$")
ax[0].set_ylabel(r"$J/ \kappa$")
ax[0].set_xlim(0.1,3)
ax[0].set_ylim(0.1,3)

########

ax[1].plot(data_traj["times"], data_traj["sol1"][0:,5], color="#e41a1c") 
ax[1].plot(data_traj["times"], data_traj["sol2"][0:,5], color="#377eb8", linestyle="-") 

ax[1].set_ylabel(r"$m_z(t)$")
ax[1].set_xlabel(r"$t\kappa$")

ax[1].set_xlim((0, 30))

plt.subplots_adjust(wspace=0.4, bottom=0.1, top=0.9)

plt.savefig("multistability_var_2.pdf", bbox_inches="tight")




