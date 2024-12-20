import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.ticker as mtick
import matplotlib.gridspec as gridspec


import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from qutip import Bloch
from mpl_toolkits.mplot3d import Axes3D

from scipy.integrate import simpson


import pickle

plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage[T1]{fontenc} \usepackage{amsmath} \usepackage{amsfonts}",
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


#with open("Data/Data_Bistability_InternalEnergy.pickle", 'rb') as handle:
#    data_erg = pickle.load(handle)

with open("Data/Data_erg_bistable.pickle", "rb") as handle:
    data_erg_bis = pickle.load(handle)

with open("Data/Data_erg_adiabatic_following.pickle", "rb") as handle:
    data_erg_bis_ad = pickle.load(handle)

with open('Data/Data_efficiency_sp_tp.pickle', 'rb') as handle:
    data_eff = pickle.load(handle)

J_list =  data_erg_bis["g_list"]
internal_energy = data_erg_bis["erg"]
internal_energy_ad = data_erg_bis_ad["erg"]

stationary_internal_energy = -1*np.sqrt((1 - data_erg_bis["ω2x"]**2/((J_list-1)**2 + 1))) + 1

#mx_list  = data_TC["mx_list"] 
#my_list  = data_TC["my_list"] 
#mz_list  = data_TC["mz_list"] 
#phi_list = data_TC["phi_list"] 
#times    = data_TC["times"] 

##################

fig = plt.figure(figsize=(8, 3))

# Add the large square subplot (2x2 grid) in the left part
ax0 = fig.add_subplot(1, 2, 1)  # 1 row, 2 columns, 1st position

ax0.plot(J_list, np.array(internal_energy), color="#e41a1c", linestyle="-")
ax0.plot(J_list, np.array(internal_energy_ad), color="#377eb8", linestyle="--")
ax0.plot(J_list, np.real(stationary_internal_energy), color="#4daf4a", linestyle=":")
ax0.set_xlabel(r"$J/\kappa$")
ax0.set_ylabel(r"$\bar{\mathcal{E}}$")

ax0.set_xlim((2, 5))
ax0.set_ylim((0, 2*0.55))

ax0.set_yticks([0, 2*0.5])
ax0.set_xticks([2, 3, 4]) ############
ax1 = fig.add_subplot(2, 2, 2)  # 2 rows, 2 columns, 2nd position

ax1.plot(data_eff["times"], 2*np.sqrt(1/2)*(data_eff["mz_tc"] - data_eff["mz_tc"][0]), color="#e41a1c", linestyle="-")
ax1.plot(data_eff["times"], 2*np.sqrt(1/2)*(data_eff["mz_ss"] - data_eff["mz_ss"][0]), color="#377eb8", linestyle="--")

ax1.set_ylabel(r"$\mathcal{E}(t)$")

ax1.set_xlim((0.0, 20))
ax1.set_ylim((0, 2.1))

#############
ax2 = fig.add_subplot(2, 2, 4)  # 2 rows, 2 columns, 4th position

energy_time_tc = 2*np.sqrt(1/2)*(data_eff["mz_tc"] - data_eff["mz_tc"][0])
energy_time_ss = 2*np.sqrt(1/2)*(data_eff["mz_ss"] - data_eff["mz_ss"][0])

heat_time_tc = np.array([2*simpson(data_eff["mx_tc"][0:j+1]**2 + data_eff["my_tc"][0:j+1]**2, data_eff["times"][0:j+1]) 
                         for j in range(len(data_eff["times"])-1)])
heat_time_ss = np.array([2*simpson(data_eff["mx_ss"][0:j+1]**2 + data_eff["my_ss"][0:j+1]**2, data_eff["times"][0:j+1]) 
                         for j in range(len(data_eff["times"])-1)])

ax2.plot(data_eff["times"][0:-1], energy_time_tc[0:-1]/(energy_time_tc[0:-1] + heat_time_tc), 
         color="#e41a1c", linestyle="-", label=r"$\mathrm{Time\;crystal}$")
ax2.plot(data_eff["times"][0:-1], energy_time_ss[0:-1]/(energy_time_ss[0:-1] + heat_time_ss),
         color="#377eb8", linestyle="--", label=r"$\mathrm{Stationary}$")

ax2.set_xlim((0.0, 20))
ax2.set_ylim((0, 1.01))
#ax2.set_xticks([0, 20])

ax2.set_xlabel(r"$t\kappa$")#, labelpad=-10)
ax2.set_ylabel(r"$\eta(t)$")

ax2.legend(edgecolor="black", framealpha=1,handlelength=1.3, borderpad=0.3, fontsize=15, loc=1, labelspacing=0.1, handletextpad=0.4, ncol=1, columnspacing=0.3)


##############
plt.tight_layout()

plt.setp(ax1.get_xticklabels(), visible=False)
plt.subplots_adjust(bottom=0.2, top=0.9)

#############
plt.savefig("Figures/Setup1_batteries.pdf", bbox_inches="tight")


