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

J=3
w=0.0

num_reps=20


f1_0=np.random.rand()*2*np.pi
f2_0=np.random.rand()*2*np.pi

time0=time.time()
sol=solve_ivp(MF_setup2, [0, 50], [f1_0,f2_0],t_eval=np.linspace(0, 50, 100), args=(w,J),atol=1e-12,rtol=1e-12)


resS=sol.y

my1=-(1/np.sqrt(2))*np.sin(resS[0,50:])
mz2=-(1/np.sqrt(2))*np.cos(resS[1,50:])

av1=np.average(my1)
av2=np.average(mz2)
s_av1=np.average(my1)**2
s_av2=np.average(mz2)**2

for j in range(1,num_reps):
	f1_0=np.random.rand()*2*np.pi
	f2_0=np.random.rand()*2*np.pi

	sol=solve_ivp(MF_setup2, [0, 50], [f1_0,f2_0],t_eval=np.linspace(0, 50, 100), args=(w,J),atol=1e-12,rtol=1e-12)


	resS=sol.y

	my1=-(1/np.sqrt(2))*np.sin(resS[0,50:])
	mz2=-(1/np.sqrt(2))*np.cos(resS[1,50:])

	av1+=np.average(my1)
	av2+=np.average(mz2)
	s_av1+=np.average(my1)**2
	s_av2+=np.average(mz2)**2



time1=time.time()
print("time solution: ", time1-time0)

var1=s_av1/(num_reps)-(av1/num_reps)**2
var2=s_av2/(num_reps)-(av2/num_reps)**2

print(J,w,var1,var2)


res=np.array([[J,w,var1,var2]])

for w in np.arange(0.1,1.0,0.05):
	for J in np.arange(3.5,0.05,-0.05):
		av1=0
		av2=0
		s_av1=0
		s_av2=0
		time0=time.time()
		for j in range(0,num_reps):
			f1_0=np.random.rand()*2*np.pi
			f2_0=np.random.rand()*2*np.pi
			sol=solve_ivp(MF_setup2, [0, 50], [f1_0,f2_0],t_eval=np.linspace(0, 50, 100), args=(w,J),atol=1e-12,rtol=1e-12)


			resS=sol.y

			my1=-(1/np.sqrt(2))*np.sin(resS[0,50:])
			mz2=-(1/np.sqrt(2))*np.cos(resS[1,50:])

			av1+=np.average(my1)
			av2+=np.average(mz2)
			s_av1+=np.average(my1)**2
			s_av2+=np.average(mz2)**2


		time1=time.time()
		print("time solution: ", time1-time0)

		var1=s_av1/(num_reps)-(av1/num_reps)**2
		var2=s_av2/(num_reps)-(av2/num_reps)**2

		print(J,w,var1,var2)


		res=np.append(res,np.array([[J,w,var1,var2]]),axis=0)



for w in np.arange(1.0,2.0,0.05):
	for J in np.arange(3.5,0.05,-0.05):
		av1=0
		av2=0
		s_av1=0
		s_av2=0
		time0=time.time()
		if J > np.sqrt(w-1):
			for j in range(0,num_reps):
				f1_0=np.random.rand()*2*np.pi
				f2_0=np.random.rand()*2*np.pi
				sol=solve_ivp(MF_setup2, [0, 200], [f1_0,f2_0],t_eval=np.linspace(0, 200, 1000), args=(w,J),atol=1e-12,rtol=1e-12)


				resS=sol.y

				my1=-(1/np.sqrt(2))*np.sin(resS[0,900:])
				mz2=-(1/np.sqrt(2))*np.cos(resS[1,900:])

				av1+=np.average(my1)
				av2+=np.average(mz2)
				s_av1+=np.average(my1)**2
				s_av2+=np.average(mz2)**2


			time1=time.time()
			print("time solution: ", time1-time0)

			var1=s_av1/(num_reps)-(av1/num_reps)**2
			var2=s_av2/(num_reps)-(av2/num_reps)**2

			print(J,w,var1,var2)


			res=np.append(res,np.array([[J,w,var1,var2]]),axis=0)


for w in np.arange(2.0,3.5,0.05):
	for J in np.arange(3,0.05,-0.05):
		av1=0
		av2=0
		s_av1=0
		s_av2=0
		time0=time.time()
		if J > 0.5*(w+np.sqrt(w**2-4)):
			for j in range(0,num_reps):
				f1_0=np.random.rand()*2*np.pi
				f2_0=np.random.rand()*2*np.pi
				sol=solve_ivp(MF_setup2, [0, 200], [f1_0,f2_0],t_eval=np.linspace(0, 200, 1000), args=(w,J),atol=1e-12,rtol=1e-12)


				resS=sol.y

				my1=-(1/np.sqrt(2))*np.sin(resS[0,900:])
				mz2=-(1/np.sqrt(2))*np.cos(resS[1,900:])

				av1+=np.average(my1)
				av2+=np.average(mz2)
				s_av1+=np.average(my1)**2
				s_av2+=np.average(mz2)**2


			time1=time.time()
			print("time solution: ", time1-time0)

			var1=s_av1/(num_reps)-(av1/num_reps)**2
			var2=s_av2/(num_reps)-(av2/num_reps)**2

			print(J,w,var1,var2)


			res=np.append(res,np.array([[J,w,var1,var2]]),axis=0)


np.savetxt("phase_diagram_setup2_var_method.dat",res)






