import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.cm as cm
import matplotlib.ticker as mtick

import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from qutip import Bloch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import pickle

from scipy.integrate import simpson


plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage[T1]{fontenc} \usepackage{amsmath}\usepackage{amsfonts}\usepackage{bbm}",# times, txfonts}",
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
    "axes.labelsize": 15
})

with open('Data/Data_seeding_PhaseDiagram_SP_TC_test_parallel.pickle', 'rb') as handle:
    data_PD_work = pickle.load(handle)

with open('Data/Data_Bistability_InternalEnergy.pickle', 'rb') as handle:
    data_Bi = pickle.load(handle)

with open('Data/Data_Entanglement_Temperature.pickle', 'rb') as handle:
    data_Ent = pickle.load(handle)

with open('Data/Data_seeding_traj_eff.pickle', 'rb') as handle:
    data_eff = pickle.load(handle)


##
X_work, Y_work = np.meshgrid(data_PD_work["Omega_list"], data_PD_work["J_list"])


##
negativity = data_Ent["PhaseMap_nega"]
ocupp_numb = data_Ent["ocuppation_list"]
##

#######
#######
#######
#######
#######


fig, ax = plt.subplots(2,2, figsize=(8, 5))#, layout="constrained")

#################### Plot 1 
cbar_et = ax[0][0].pcolormesh(X_work, Y_work, (1 - data_PD_work["mz1e2_mat"] - data_PD_work["mz2e2_mat"]).T.real
                              , cmap="PuBu_r",rasterized=True)
cbar2 = fig.colorbar(cbar_et, ax=ax[0][0], pad=.01, fraction=0.05,  orientation='vertical')
#cbar2.ax.set_title(r"$\bar{\dot{w}}/\kappa\nu$", y=-0.2, pad=-0, loc="left")
cbar2.ax.set_title(r"$\bar{\dot{w}}/\kappa\nu$",  loc="left")
cbar2.set_ticks([0.0, 1.0])


ax[0][0].set_xlabel(r"$\Omega/\kappa$")#, labelpad=-10)
ax[0][0].set_ylabel(r"$J/\kappa$")

ax[0][0].set_ylim((0, 3.0))

ax[0][0].set_xlim((0, 3.0))


Omega_list1 = np.linspace(0, 2, 1000)
Omega_list2 = np.linspace(2, 3.0, 1000)

ax[0][0].plot(Omega_list2, np.real((Omega_list2 + np.sqrt(Omega_list2**2  - 4))/2), color="black")
ax[0][0].plot(Omega_list1, np.real(np.sqrt(Omega_list1-1)), color="black")

ax[0][0].plot(Omega_list2, np.real(np.sqrt(Omega_list2-1)), color="black", linestyle="--")
ax[0][0].plot(Omega_list2, np.real((Omega_list2 - np.sqrt(Omega_list2**2  - 4))/2), color="black", linestyle="-.")


##################### Plot 2
J_list = data_Bi["J_list"]
internal_energy = data_Bi["internal_energy_list"]

ax[0][1].plot(J_list, internal_energy, color="#e41a1c", linestyle="-")
ax[0][1].plot(J_list, 1/2*(-np.sqrt(1 - (data_Bi["ω2x"]*J_list /(J_list**2 + 1))**2) + 1), color="#377eb8", linestyle="--")

ax[0][1].set_xlabel(r"$J/\kappa$")#, labelpad=-10)
ax[0][1].set_ylabel(r"$\bar{\mathcal{E}}$")

ax[0][1].set_xlim((1.8, 2.3))
ax[0][1].set_ylim((0.2, 0.6))
#ax[0][1].set_xticks([1.8, 2.3])
#ax[0][1].set_yticks([0.3, 0.5])

#ax[0][1].legend(edgecolor="black", framealpha=1,handlelength=1.3, borderpad=0.3, fontsize=15, loc=0, labelspacing=0.1, handletextpad=0.4, ncol=1, columnspacing=0.3)


#################### Plot 3

eff_list = []
for i, J in enumerate(data_eff["J_list"]):
    print(J)
    energy_time = np.sqrt(1/2)*(data_eff["mz1_list"][i] + np.sqrt(1/2))
    total_energy_time = np.sqrt(1/2)*(data_eff["mz1_list"][i] + data_eff["mz2_list"][i] + 2*np.sqrt(1/2))
    heat_time = [simpson(data_eff["mx1_list"][i][:j+1]**2 + data_eff["my2_list"][i][:j+1]**2, data_eff["time"][:j+1]) for j in range(len(data_eff["time"])-1)]
    eff_list.append(energy_time[:-1]/(total_energy_time[:-1] + heat_time))

ax[1][0].plot(data_eff["time"][0:-1], eff_list[0], color="#4daf4a", label=r"$J=1.0\kappa$", linestyle = (0, (5, 1)))
ax[1][0].plot(data_eff["time"][0:-1], eff_list[1], color="#e41a1c", label=r"$J=1.9\kappa$", linestyle = "-")
ax[1][0].plot(data_eff["time"][0:-1], eff_list[2], color="#377eb8", label=r"$J=2.1\kappa$", linestyle = (0,(3, 1, 1, 1)))
ax[1][0].plot(data_eff["time"][0:-1], eff_list[3], color="#984ea3", label=r"$J=3.0\kappa$", linestyle = (0, (1, 1)))

ax[1][0].set_xlabel(r"$t\kappa$")#, labelpad=-10)
ax[1][0].set_ylabel(r"$\eta(t)$")
       
ax[1][0].set_xlim((0, 30))
ax[1][0].set_ylim((0.0, 0.5))
#ax[1][0].set_yticks([0, 0.5])
#ax[1][0].set_xticks([0, 30.0])
        
ax[1][0].legend(edgecolor="black", framealpha=1,handlelength=1.3, borderpad=0.3, fontsize=15, loc=0, labelspacing=0.1, handletextpad=0.4, ncol=1, columnspacing=0.3)

#################### Plot 4 

ax[1][1].plot(ocupp_numb, negativity, color="black")
ax[1][1].set_xlabel(r"$n$")#, labelpad=-10)
ax[1][1].set_ylabel(r"$\bar{\mathcal{N}}$")


ax[1][1].set_xscale("log")
#ax[1][1].set_yscale("log")

ax[1][1].text(0.0025, 0.005, r"$J=1\kappa$", fontsize=20)
ax[1][1].text(0.0025, 0.015, r"$\Omega=2.5\kappa$", fontsize=20)
ax[1][1].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
#ax[1][1].set_yticks([0., 0.045])


###################

plt.tight_layout()
plt.subplots_adjust(wspace=0.2, hspace=0.3)

plt.savefig("Figures/Setup2_panels.pdf", bbox_inches="tight" ,dpi=400)

