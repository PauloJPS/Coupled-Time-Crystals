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
#ax[1][1].set_ylim((0.0,10))

ax[1][1].set_xticks([0, 100])
#ax[1][1].set_yticks([0, 1])
#
#
#
ax[1][1].set_xlabel(r"$t\kappa$")
ax[1][1].set_ylabel(r"$S(t)$")




#################

plt.tight_layout()
plt.subplots_adjust(wspace=0.25, hspace=0.4)
plt.savefig("Setup1_Appendix.pdf", bbox_inches="tight")







#########
#fig = plt.figure(figsize=(8, 3.5))
#
## Add the large square subplot (2x2 grid) in the left part
#ax0 = fig.add_subplot(1, 2, 1)  # 1 row, 2 columns, 1st position
#
#cbar_et_1 = ax0.pcolormesh(GZ, G, np.array(work_tc), cmap="PuBu_r", rasterized=True)
#cbar1 = fig.colorbar(cbar_et_1, ax=ax0, pad=.02, orientation='vertical')
#cbar1.ax.set_title(r"$\bar{\dot{w}}/(\kappa\nu)$", y=-0.15)
#
#ax0.set_xlabel(r"$J_z/\kappa$",labelpad=-0.5)
#ax0.set_ylabel(r"$J/\kappa$",labelpad=-0.5)
#
#ax0.set_xticks([0, 3])
#ax0.set_yticks([0, 4])
#
#ax0.set_xlim((0, 4))
#ax0.set_ylim((0, 4))
#
#
#ax0.plot([0,4], [4,0], color="black", linewidth=1)
#
### Inset 
#
#ax01 = inset_axes(ax0, width="35%", height="40%", loc="upper right")  # Adjust size and position
#
#ax01.plot(g_list, work_tc[0], linestyle="-", color="black", marker="s", markersize=0.5, linewidth=0)
#
#ax01.plot(g_list, 2**2/(np.array(g_list)**2 + 1) )
#
#ax01.set_ylabel(r"$ \bar{\dot{w}}/(\kappa\nu)$",labelpad=-0.5)
#ax01.set_xlabel(r"$J_z/\kappa$",labelpad=-0.5)
#
#ax01.set_xticks([0, 4])
#ax01.set_yticks([0, 1])
#
#ax01.set_xlim((0, 4))
#ax01.set_ylim((0, 1))
#
#ax01.text(0.4, 0.1, r"$J=0$", fontsize=20)
#
#
## Add the first small subplot (1x1 grid) to the right
#ax1 = fig.add_subplot(2, 2, 2)  # 2 rows, 2 columns, 2nd position
#
#ax1.plot(g_list_cor, PhaseMap_ccor_tc, color="#e41a1c", marker="o", markersize=5)
#                                                                   
#ax1.set_ylabel(r"$\mathcal{J}$")
#
#ax1.set_xlim((0,4))
#ax1.set_ylim((0,7))
#
## Add the second small subplot (1x1 grid) below the first small one
#ax2 = fig.add_subplot(2, 2, 4, sharex=ax1)  # 2 rows, 2 columns, 4th position
#
#ax2.plot(g_list_cor, PhaseMap_disc_tc,color="#e41a1c", marker="o", markersize=5, linestyle="-",  label=r"$\mathcal{D}$")
#ax2.plot(g_list_cor, PhaseMap_nega_tc,color="#377eb8", marker="s", markersize=5, linestyle="--", label=r"$\mathcal{N}$")
#
#ax2.set_xlim((0,4))
#ax2.set_ylim((0,0.22))
#ax2.set_xlabel(r"$J + J_z=4\kappa$")
#
#ax2.legend(edgecolor="black", framealpha=1,handlelength=1.3, borderpad=0.3, fontsize=15, loc=1, labelspacing=0.1, handletextpad=0.4, ncol=1, columnspacing=0.3)
#
#ax21 = inset_axes(ax2, width="40%", height="35%", loc="center")  # Adjust size and position
#ax21.plot(g_list_cor, PhaseMap_disc_tc, linestyle="-", color="#e41a1c", marker="o")
#ax21.plot(g_list_cor, PhaseMap_nega_tc, linestyle="--", color="#377eb8", marker="s")
#
#ax21.set_xlim((1.5,2.5))
#ax21.set_ylim((0,0.01))
#ax21.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
#
## Hide x-axis labels of the first small subplot to avoid overlap
#plt.setp(ax1.get_xticklabels(), visible=False)
#
## Adjust layout to prevent overlapping
#plt.tight_layout()
#
#
##plt.savefig("Setup1_MF_QF.pdf", bbox_inches="tight")
#
