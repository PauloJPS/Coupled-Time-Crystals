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



with open("Data/Data_Discord_trajectories.pickle", "rb") as handle:
    data_discord = pickle.load(handle)

with open("Data/Data_Efficiency_trajs_J.pickle", "rb") as handle: 
    data_eff = pickle.load(handle)

with open("Data/Data_TC_family.pickle", "rb") as handle: 
    data_TC = pickle.load(handle)

#### Mean-field data 


##################


fig, ax = plt.subplots(2,2, figsize=(8, 5))#, layout="constrained")


######## 
n_curves = 10
colormap = cm.cividis

mx_list  = data_TC["mx_list"] 
my_list  = data_TC["my_list"] 
mz_list  = data_TC["mz_list"] 
phi_list = data_TC["phi_list"] 
times    = data_TC["times"] 

for i in range(n_curves):
    color = colormap(i / n_curves)  # Normalize i to get colors from the colormap
    ax[0][0].plot(mx_list[i], mz_list[i], color=color, label=f'Curve {i+1}', linewidth=2)


ax[0][0].set_xlim((0.38, 0.72))
ax[0][0].set_ylim((-0.3,0.3))
#
ax[0][0].set_xticks([0.4, 0.7])
ax[0][0].set_yticks([-0.3, 0.3])
#
ax[0][0].set_xlabel(r"$m_x(t)$")
ax[0][0].set_ylabel(r"$m_z(t)$")
#
#

################

ax[0][1].plot(data_eff["times"][0:-1], data_eff["efficiency_list"][0], color="#e41a1c", linestyle="-", label=r"$J=2\kappa$")
ax[0][1].plot(data_eff["times"][0:-1], data_eff["efficiency_list"][1], color="#4daf4a", linestyle="-.", label=r"$J=0\kappa$")



ax[0][1].set_xlim((0, 20))
ax[0][1].set_ylim((0,1.01))
#
ax[0][1].set_xticks([0, 20])
ax[0][1].set_yticks([0, 1])
#
ax[0][1].set_xlabel(r"$t \kappa$")
ax[0][1].set_ylabel(r"$\eta(t)$")

ax[0][1].legend(edgecolor="black", framealpha=1,handlelength=1.3, borderpad=0.3, fontsize=15, loc=1, labelspacing=0.1, handletextpad=0.4, ncol=1, columnspacing=0.3)

################

ax[1][0].plot(data_discord["time"], data_discord["Discord_list"][0], color="#e41a1c", linestyle="-", label=r"$J_z=1\kappa$")
ax[1][0].plot(data_discord["time"], data_discord["Discord_list"][1], color="#4daf4a", linestyle="--", label=r"$J_z=2\kappa$")

ax[1][0].legend(edgecolor="black", framealpha=1,handlelength=1.3, borderpad=0.3, fontsize=15, loc=0, labelspacing=0.1, handletextpad=0.4, ncol=1, columnspacing=0.3)
#
ax[1][0].set_xscale("log")
ax[1][0].set_yscale("log")
#
ax[1][0].set_xlim((0.5, 100))
ax[1][0].set_ylim((0.00001,1))
#
ax[1][1].set_xticks([0.1, 100])
#
ax[1][0].set_xlabel(r"$t\kappa$")
ax[1][0].set_ylabel(r"$\mathcal{D}(t)$")

#############

ax[1][1].plot(data_discord["time"], data_discord["Entropy_list"][0], color="#e41a1c", linestyle="-", label=r"$J_z=1\kappa$")
ax[1][1].plot(data_discord["time"], data_discord["Entropy_list"][1], color="#4daf4a", linestyle="--", label=r"$J_z=2\kappa$")

ax[1][1].legend(edgecolor="black", framealpha=1,handlelength=1.3, borderpad=0.3, fontsize=15, loc=0, labelspacing=0.1, handletextpad=0.4, ncol=1, columnspacing=0.3)

ax[1][1].set_xlim((0.5, 100))

ax[1][1].set_xticks([0, 100])

ax[1][1].set_xlabel(r"$t\kappa$")
ax[1][1].set_ylabel(r"$S(t)$")

#################

plt.tight_layout()
plt.subplots_adjust(wspace=0.25, hspace=0.3)

plt.savefig("Figures/Setup1_Appendix_Trajectories.pdf", bbox_inches="tight")

