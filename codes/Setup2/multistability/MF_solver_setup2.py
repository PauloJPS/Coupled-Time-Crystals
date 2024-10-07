####### execute using "python m.py"
####### 21/09/2024 #########

import time
import numpy as np
import math as mt
import scipy as sp
from scipy import linalg as LA
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rc
from matplotlib import gridspec
import matplotlib as mpl
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = "STIX"
mpl.rcParams["mathtext.fontset"] = "stix"


###########################################################
###########################################################
############# MF equations
def MF_setup2(t,v, ww,JJ):
	ang1_0,ang2_0=v
	ang1_f=-JJ*np.sin(ang2_0)-np.sin(ang1_0)
	ang2_f=JJ*np.sin(ang1_0)-np.sin(ang2_0)-ww
	return[ang1_f,ang2_f]


###########################################################
###########################################################

J=2.4
w=2.5

f1_0=np.random.rand()*2*np.pi#0.0
f2_0=np.random.rand()*2*np.pi#0.0

time0=time.time()
sol=solve_ivp(MF_setup2, [0, 200], [f1_0,f2_0],t_eval=np.linspace(0, 200, 100), args=(w,J),atol=1e-12,rtol=1e-12)
time1=time.time()
print("time solution: ", time1-time0)

resS=sol.y
times=sol.t

mz1=-(1/np.sqrt(2))*np.cos(resS[0,:])
mx1=-(1/np.sqrt(2))*np.sin(resS[0,:])
mz2=-(1/np.sqrt(2))*np.cos(resS[1,:])
my2=-(1/np.sqrt(2))*np.sin(resS[1,:])




plt.subplot(121)
plt.plot(times,mz1,label="z")
plt.plot(times,mx1,label="x")
plt.legend()

plt.subplot(122)
plt.plot(times,mz2)
plt.plot(times,my2)

plt.show()

