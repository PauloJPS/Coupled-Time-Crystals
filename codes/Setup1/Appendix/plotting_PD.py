import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.ticker as mtick
import matplotlib.gridspec as gridspec

import matplotlib.ticker as ticker
import matplotlib.pyplot as plt


from qutip import Bloch
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import pickle

plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage[T1]{fontenc} \usepackage{amsmath} \usepackage{amsfonts} \usepackage{bbm}",
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


with open("Data/Data_mean_field.pickle", 'rb') as handle:
    data_mef = pickle.load(handle)

with open("Data/Data_asymmetry_correlations.pickle", 'rb') as handle:
    data_cor = pickle.load(handle)


#### Mean-field data 

work_tc = data_mef["work_tc"]
work_sp = data_mef["work_sp"]

g_list = data_mef["g_list"]
gz_list = data_mef["gz_list"]

GZ, G = np.meshgrid(gz_list, g_list)


#### Correlations_data 

PhaseMap_ccor_tc = data_cor["PhaseMap_ccor_tc"]
PhaseMap_disc_tc = data_cor["PhaseMap_disc_tc"]
PhaseMap_nega_tc = data_cor["PhaseMap_nega_tc"]
PhaseMap_entr_tc = data_cor["PhaseMap_entr_tc"]
PhaseMap_ccor_sp = data_cor["PhaseMap_ccor_sp"]
PhaseMap_disc_sp = data_cor["PhaseMap_disc_sp"]
PhaseMap_nega_sp = data_cor["PhaseMap_nega_sp"]
PhaseMap_entr_sp = data_cor["PhaseMap_entr_sp"]

g_list_cor = data_cor["g_list"]


##################

fig = plt.figure(figsize=(8, 3))

# Add the large square subplot (2x2 grid) in the left part
ax0 = fig.add_subplot(1, 2, 1)  # 1 row, 2 columns, 1st position
ax1 = fig.add_subplot(2, 2, 2)  # 2 rows, 2 columns, 2nd position
ax2 = fig.add_subplot(2, 2, 4, sharex=ax1)  # 2 rows, 2 columns, 4th position


plt.tight_layout()

cbar_et_1 = ax0.pcolormesh(GZ, G, np.array(work_sp), cmap="PuBu_r", rasterized=True)
cbar1 = fig.colorbar(cbar_et_1, ax=ax0, pad=.02, orientation='vertical')
cbar1.ax.set_title(r"$\bar{\dot{w}}/(\kappa\nu)$")

ax0.set_xlabel(r"$J_z/\kappa$")
ax0.set_ylabel(r"$J/\kappa$")

ax0.set_xlim((0, 4))
ax0.set_ylim((0, 4))

# Add the first small subplot (1x1 grid) to the right

ax1.plot(g_list_cor, PhaseMap_ccor_sp, color="#e41a1c", marker="o", markersize=5)
                                                                   
ax1.set_ylabel(r"$\bar{\mathcal{J}}$")

ax1.set_xlim((0,4))
ax1.set_ylim((0,0.01))


ax1.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

# Add the second small subplot (1x1 grid) below the first small one

ax2.plot(g_list_cor, PhaseMap_disc_sp,color="#e41a1c", marker="o", markersize=5, linestyle="-",  label=r"$\bar{\mathcal{D}}$")
ax2.plot(g_list_cor, PhaseMap_nega_sp,color="#377eb8", marker="s", markersize=5, linestyle="--", label=r"$\bar{\mathcal{N}}$")

ax2.set_xlabel(r"$J + J_z=4\kappa$")

ax2.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

ax2.set_xlim((0,4))
ax2.set_ylim((0,0.07))

ax2.legend(edgecolor="black", framealpha=1,handlelength=1.3, borderpad=0.3, fontsize=15, loc=1, labelspacing=0.1, handletextpad=0.4, ncol=1, columnspacing=0.3)


# Hide x-axis labels of the first small subplot to avoid overlap
plt.setp(ax1.get_xticklabels(), visible=False)
plt.subplots_adjust(bottom=0.2, top=0.9)
# Adjust layout to prevent overlapping


plt.savefig("Figures/Setup1_PD_Appendix.pdf")#, bbox_inches="tight")
