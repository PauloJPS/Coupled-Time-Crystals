import numpy as np 
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
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    #
    "legend.frameon": True,
    "legend.fontsize": 20,
    "legend.edgecolor": "white",
    "axes.titlesize": 20,
    "axes.titleweight": "bold",
    "axes.labelsize":20
})


with open("Data/Data_ChargerBattery_trajectories.pickle", "rb") as handle:
    data_ChargerBatery = pickle.load(handle)

with open("Data/Data_Nega_trajectories.pickle", "rb") as handle: 
    data_Nega = pickle.load(handle)

with open("Data/Data_seeding_fourier.pickle", "rb") as handle: 
    data_seed = pickle.load(handle)

##################

fig, ax = plt.subplots(2,2, figsize=(8, 4))#, layout="constrained")

#################### Plot 1 

X, Y = np.meshgrid(data_seed["g_list"], data_seed["xf"])
cbar_et = ax[0][0].pcolormesh(X, Y, (np.array(data_seed["modes_m2"])).T.real, cmap="gist_heat_r",rasterized=True)

cbar1 = fig.colorbar(cbar_et, ax=ax[0][0], pad=.01, fraction=0.05,  orientation='vertical')

ax[0][0].set_ylim((0, 0.6))

ax[0][0].set_xlabel(r"$J/\kappa$")
ax[0][0].set_ylabel(r"$\tilde{\omega}_2$")

ax[0][0].set_xticks([0, 2])
################

ax[0][1].plot(data_seed["xf"], data_seed["modes_m2"][240], color="#e41a1c", linestyle="-", label=r"$J=2\kappa$")
ax[0][1].plot(data_seed["xf"], data_seed["modes_m1"][240], color="#4daf4a", linestyle="-.", label=r"$J=2\kappa$")

ax[0][1].set_xlim((0, 0.5))
ax[0][1].set_ylim((0,1.01))
##
ax[0][1].set_xticks([0, 0.5])
ax[0][1].set_yticks([0, 1])
##
#
################

times = np.arange(data_ChargerBatery["tspan"][0], data_ChargerBatery["tspan"][1], data_ChargerBatery["dt"])

ax[1][0].plot(times, data_ChargerBatery["mz_charger"][0], color="#e41a1c", linestyle="-.", label=r"$J_z=1\kappa$")
ax[1][0].plot(times, data_ChargerBatery["mz_charger"][1], color="#e41a1c", linestyle="-", label=r"$J_z=1\kappa$")

ax[1][0].plot(times, data_ChargerBatery["mz_battery"][0], color="#4daf4a", linestyle="-.", label=r"$J_z=1\kappa$")
ax[1][0].plot(times, data_ChargerBatery["mz_battery"][1], color="#4daf4a", linestyle="-", label=r"$J_z=1\kappa$")


ax[1][0].set_xlim((0, 15))
ax[1][0].set_ylim((-np.sqrt(1/2), np.sqrt(1/2)))


ax[1][0].set_yticks([-0.5, 0.5])
ax[1][0].set_xticks([0, 15])

ax[1][0].set_xlabel(r"$t \kappa$")

############

ax[1][1].plot(data_Nega["time"], data_Nega["Nega_list"][0], color="#4daf4a", linestyle="-", label=r"$J_z=0.5\kappa$")
ax[1][1].plot(data_Nega["time"], data_Nega["Nega_list"][1], color="#377eb8", linestyle="-.", label=r"$J_z=1.0\kappa$")
ax[1][1].plot(data_Nega["time"], data_Nega["Nega_list"][2], color="#e41a1c", linestyle="--", label=r"$J_z=1.5\kappa$")

ax[1][1].set_xlim((0, 20))
ax[1][1].set_ylim((0, 0.5))


ax[1][1].set_yticks([0, 0.5])
ax[1][1].set_xticks([0, 20])

ax[1][1].set_xlabel(r"$t \kappa$")
ax[1][1].set_ylabel(r"$\mathcal{N}$")

ax[1][1].legend(edgecolor="black", framealpha=1,handlelength=1.3, borderpad=0.3, fontsize=15, loc=1, labelspacing=0.1, handletextpad=0.4, ncol=1, columnspacing=0.3)

###############
plt.tight_layout()
plt.subplots_adjust(wspace=0.25, hspace=0.6)
plt.savefig("Setup2_Appendix.pdf", bbox_inches="tight",dpi=400)


