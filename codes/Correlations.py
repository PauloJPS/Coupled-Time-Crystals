import qutip as qt
import numpy as np 
import scipy as sp
import pickle
import matplotlib.pyplot as plt 
import matplotlib.ticker as mtick

import matplotlib.ticker as ticker

from scipy.integrate import solve_ivp
from scipy.linalg import sqrtm, eigvals
from scipy.integrate import simpson


 


######
###### The dynamics of the covariance matrix and the mean-field quantities 
######

def func(t, y, ω1x, ω1z, ω2x, ω2z, gx, gy, gz, k1, k2, n1, n2):

    print(t)

    G = y[0:36].reshape((6, 6))
    L = y[0+36:36+36].reshape((6, 6))
    m1x, m1y, m1z, m2x, m2y, m2z = y[36+36:36+36+6]

    Dl = -np.array([[0.0,  ω1z, 0.0,  0.0,  0.0,  0.0],
                    [-ω1z, 0.0, ω1x,  0.0,  0.0,  0.0],
                    [0.0, -ω1x, 0.0,  0.0,  0.0,  0.0],
                    [0.0,  0.0, 0.0,  0.0,  ω2z,  0.0],
                    [0.0,  0.0, 0.0, -ω2z,  0.0,  ω2x],
                    [0.0,  0.0, 0.0,  0.0, -ω2x,  0.0]])

    Db = -np.sqrt(2) * np.array([[0.0,    0.0,   -k1*m1x, 0.0,    0.0,    0.0],
                                 [0.0,    0.0,   -k1*m1y, 0.0,    0.0,    0.0],
                                 [k1*m1x, k1*m1y, 0.0,    0.0,    0.0,    0.0],
                                 [0.0,    0.0,    0.0,    0.0,    0.0,   -k2*m2x],
                                 [0.0,    0.0,    0.0,    0.0,    0.0,   -k2*m2y],
                                 [0.0,    0.0,    0.0,    k2*m2x, k2*m2y, 0.0]])
    
    Dc = -np.sqrt(2) * np.array([[ 0.0,         m2z*gz, -m2y*gy, 0.0,     0.0,     0.0],
                                 [-m2z*gz,      0.0,     m2x*gx, 0.0,     0.0,     0.0],
                                 [ m2y*gy,     -m2x*gx,  0.0,    0.0,     0.0,     0.0],
                                 [ 0.0,         0.0,     0.0,    0.0,     m1z*gz, -m1y*gy],
                                 [ 0.0,         0.0,     0.0,   -m1z*gz,  0.0,     m1x*gx],
                                 [ 0.0,         0.0,     0.0,    m1y*gy, -m1x*gx,  0.0]])

    Q = np.sqrt(2.0) * np.array([[ 0.0,     0.0,     0.0,    0.0,     m1z*gy, -m1y*gz],
                                 [ 0.0,     0.0,     0.0,   -m1z*gx,  0.0,     m1x*gz],
                                 [ 0.0,     0.0,     0.0,    m1y*gx, -m1x*gy,  0.0],
                                 [ 0.0,     m2z*gy, -m2y*gz, 0.0,     0.0,     0.0],
                                 [-m2z*gx,  0.0,     m2x*gz, 0.0,     0.0,     0.0],
                                 [ m2y*gx, -m2x*gy,  0.0,    0.0,     0.0,     0.0]])

    s = np.sqrt(2.0) * np.array([[0.0,  m1z, -m1y,  0.0,  0.0,  0.0],
                                 [-m1z, 0.0,  m1x,  0.0,  0.0,  0.0],
                                 [m1y, -m1x,  0.0,  0.0,  0.0,  0.0],
                                 [0.0,  0.0,  0.0,  0.0,  m2z, -m2y],
                                 [0.0,  0.0,  0.0, -m2z,  0.0,  m2x],
                                 [0.0,  0.0,  0.0,  m2y, -m2x,  0.0]])

    B = np.array([[0.0, -k1,   0.0,  0.0,  0.0,  0.0],
                  [k1,   0.0,  0.0,  0.0,  0.0,  0.0],
                  [0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
                  [0.0,  0.0,  0.0,  0.0, -k2,   0.0],
                  [0.0,  0.0,  0.0,  k2,   0.0,  0.0],
                  [0.0,  0.0,  0.0,  0.0,  0.0,  0.0]])

    A = np.array([[k1*(2.0*n1 + 1.0), 0.0,               0.0, 0.0,               0.0,               0.0],
                  [0.0,               k1*(2.0*n1 + 1.0), 0.0, 0.0,               0.0,               0.0],
                  [0.0,               0.0,               0.0, 0.0,               0.0,               0.0],
                  [0.0,               0.0,               0.0, k2*(2.0*n2 + 1.0), 0.0,               0.0],
                  [0.0,               0.0,               0.0, 0.0,               k2*(2.0*n2 + 1.0), 0.0],
                  [0.0,               0.0,               0.0, 0.0,               0.0,               0.0]])
    #### Rotation dynamics
    dL = (Dl + Db + Dc) @ L

    #### Covariance dynamics

    W = L.T @ (Q + s @ B) @ L
    dG = (W @ G + G @ W.T - L.T @ s @ A @ s @ L)

    #### Mean-field dynamics for system 1
    dm1x = -np.sqrt(2.0) * (gz * m1y * m2z) + np.sqrt(2.0) * (gy * m1z * m2y) - (ω1z * m1y) + np.sqrt(2.0) * (k1 * m1x * m1z)

    dm1y = -np.sqrt(2.0) * (gx * m1z * m2x) + np.sqrt(2.0) * (gz * m1x * m2z) + (ω1z * m1x) - (ω1x * m1z) + np.sqrt(2.0) * (k1 * m1y * m1z)

    dm1z = -np.sqrt(2.0) * (gy * m1x * m2y) + np.sqrt(2.0) * (gx * m1y * m2x) + (ω1x * m1y) - np.sqrt(2.0) * k1 * (m1x**2.0 + m1y**2.0)

    #### Mean-field dynamics for system 2
    dm2x = -np.sqrt(2.0) * (gz * m2y * m1z) + np.sqrt(2.0) * (gy * m2z * m1y) - (ω2z * m2y) + np.sqrt(2.0) *  (k2 * m2x * m2z)

    dm2y = -np.sqrt(2.0) * (gx * m2z * m1x) + np.sqrt(2.0) * (gz * m2x * m1z) + (ω2z * m2x) - (ω2x * m2z) + np.sqrt(2.0) * (k2 * m2y * m2z)

    dm2z = -np.sqrt(2.0) * (gy * m2x * m1y) + np.sqrt(2.0) * (gx * m2y * m1x) + (ω2x * m2y)  - np.sqrt(2.0) * k2 * (m2x**2.0 + m2y**2.0)

    #### Updating

    return dG.flatten().tolist() + dL.flatten().tolist() +  [dm1x, dm1y, dm1z, dm2x, dm2y, dm2z]


#########
#########
#########

def get_traj(tspan, dt, ω1x, ω1z, ω2x, ω2z, gx, gy, gz, k1, k2, n1, n2,
             m1x0, m1y0, m1z0, m2x0, m2y0, m2z0, G0, L0):

    initial = np.zeros(36 + 36 + 6)

    initial[0:36] = G0.flatten()

    initial[36+0:36+36] = L0.flatten()

    initial[36+36:36+36+6] = [m1x0, m1y0, m1z0, m2x0, m2y0, m2z0]

    p = (ω1x, ω1z, ω2x, ω2z, gx, gy, gz, k1, k2, n1, n2)

    sol = solve_ivp(func, tspan, initial, args=p, method='DOP853', t_eval=np.arange(tspan[0], tspan[1], dt), rtol=1e-12, atol=1e-12)
    print("Status = {}\n".format(sol.status))

    return sol.t, sol.y.T


def Simul_Traj(parameters):

    t, sol = get_traj(**parameters)
    
    return t, sol

#########
######### the physical quantities from the object returned by the solver 
#########

def MeanField(sol):
    num_rows, num_columns = sol.shape

    m1x = [sol[i,36+36+0] for i in range(num_rows)]
    m1y = [sol[i,36+36+1] for i in range(num_rows)]
    m1z = [sol[i,36+36+2] for i in range(num_rows)]
    m2x = [sol[i,36+36+3] for i in range(num_rows)]
    m2y = [sol[i,36+36+4] for i in range(num_rows)]
    m2z = [sol[i,36+36+5] for i in range(num_rows)]

    return m1x, m1y, m1z, m2x, m2y, m2z

def CovarianceMatrix(sol):
    num_rows, num_columns = sol.shape
    covariance_matrix = [sol[i,0:36].reshape((6, 6)) for i in range(num_rows)]
    return np.array(covariance_matrix)

def SymplecticMatrix(sol):

    num_rows, num_columns = sol.shape
    symplectic = [np.sqrt(2.0) * np.array([[0, sol[i, 36+2], -sol[i, 36+1], 0, 0, 0],
                                           [-sol[i, 36+2], 0, sol[i, 36+0], 0, 0, 0],
                                           [sol[i, 36+1], -sol[i, 36+0], 0, 0, 0, 0],
                                           [0, 0, 0, 0, sol[i, 36+5], -sol[i, 36+4]],
                                           [0, 0, 0, -sol[i, 36+5], 0, sol[i, 36+3]],
                                           [0, 0, 0, sol[i, 36+4], -sol[i, 36+3], 0]]) for i in range(num_rows)]

    return np.array(symplectic) 

#########
######### Computing information theory quantities 
#########

def EvaluateEmin(c_alpha, c_beta, c_gamma, c_delta):
    if (c_delta - c_alpha*c_beta)**2 <= (1 + c_beta)*(c_gamma**2)*(c_alpha + c_delta):
        Emin = (1/(c_beta-1)**2) * (2*c_gamma**2 + (c_beta-1)*(c_delta - c_alpha) 
                                    + 2*np.abs(c_gamma)*np.sqrt(c_gamma**2 + (c_beta-1)*(c_delta - c_alpha)))
    else:
        Emin = (1/(2*c_beta))*(c_alpha*c_beta - c_gamma**2 + c_delta 
                               - np.sqrt(c_gamma**4 + (c_delta-c_alpha*c_beta)**2 - 2*c_gamma**2*(c_alpha*c_beta + c_delta) ))

    return np.nan_to_num(Emin) 

#### function defined for calculating discord (just time/spacing saving)
def func_log(x):
    ##### removing log with negative values in the argument
    if np.isclose(x, 1.0)==True:
        x = 1
    if x == 1: 
        #print("got a log problem\n") 
        return (x+1)/2 * np.log((x+1)/2)
    else: 
        return (x+1)/2 * np.log((x+1)/2) - (x-1)/2 * np.log((x-1)/2)
  
def QuantumClassicalThermodynamics(sol):
    """
        Calculate quantum discord, one-way classical correlation, mutual information, according to the definition in the paper PRL 105, 030501 (2010)
    """
    num_rows, num_columns = sol.shape

    ##### this part will need further optimization, basically, I dont need to create another array of covariance matrices 
    covariance = CovarianceMatrix(sol)

    bosonic_sy = np.array([[0,1],[-1,0]])

    #### Vector for quantum discord 
    informational_quantities = []

    #### compute quantum discord for each time 
    for i in range(num_rows): 
        #### Get mean field at time t 
        mean_field =  sol[i,36 + 36], sol[i,36 + 36 + 1], sol[i,36 + 36 + 2], sol[i,36 + 36 + 3], sol[i,36 + 36 +4], sol[i,36 + 36 + 5]
        cov = 2*covariance[i]

        #### Get sub matrices 
        alpha = cov[0:2, 0:2]
        beta  = cov[3:5, 3:5]
        gamma = cov[0:2, 3:5]
        reducedCov = np.delete(np.delete(cov, [2,5], axis=0), [2,5], axis=1)

        #### upper diagonal block
        c_alpha = np.linalg.det(alpha)
        #print("c_alpha = {:.4f}".format(c_alpha))

        c_beta = np.linalg.det(beta)
        #print("c_beta = {:.4f}".format(c_beta))

        c_gamma = np.linalg.det(gamma)
        #print("c_gamma = {:.4f}".format(c_gamma))

        #### covariance matrix 
        c_delta = np.linalg.det(reducedCov)
        #print("c_delta = {:.4f}".format(c_delta))

        #### Symplectic eigenvalues 
        Delta = c_alpha + c_beta + 2*c_gamma
        TR_Delta = c_alpha + c_beta - 2*c_gamma

        #print("Delta = {:.4f}".format(Delta))
        #print("Delta**2 - 4*c_delta = {:.4f}".format(Delta**2 - 4*c_delta)) 
    
        v_p = np.sqrt(0.5*Delta + 0.5*np.sqrt(np.abs(Delta**2 - 4*c_delta) ))#### the abs prevents numeric negative zeros
        v_m = np.sqrt(0.5*Delta - 0.5*np.sqrt(np.abs(Delta**2 - 4*c_delta) ))#### the abs prevents numeric negative zeros
                                                                                      
        #### E min 
        Emin = EvaluateEmin(c_alpha, c_beta, c_gamma, c_delta)

        #### Thermodynamic quantities 
        J_one_way = func_log(np.sqrt(c_alpha)) - func_log(np.sqrt(Emin))

        Q_Discord = func_log(np.sqrt(c_beta)) - func_log(v_m) - func_log(v_p) + func_log(np.sqrt(Emin))

        Log_neg = np.max([0, -np.log(np.sqrt(0.5*TR_Delta - 0.5*np.sqrt(np.abs(TR_Delta**2 - 4*c_delta))))])

        Mutual_informaiton = Q_Discord + J_one_way

        Entropy_1 = func_log(np.abs(np.linalg.eigvals(1.j*(bosonic_sy @ alpha))[0]))
        Entropy_2 = func_log(np.abs(np.linalg.eigvals(1.j*(bosonic_sy @ beta))[0]))

        Total_entropy = func_log(v_p) + func_log(v_m)

        #### Ergotropy - test 

        Erg = (2*(reducedCov[0][2] + reducedCov[1][3]) + ((v_p+v_m) - (v_p-v_m))/8)

        print(reducedCov[2][0], v_p)
       
        #### Appending everything in a vector 
        informational_quantities.append([J_one_way, Q_Discord, Log_neg, Mutual_informaiton, Total_entropy, Entropy_1, Entropy_2, Erg])

    #### returning the quantities 
    return np.array(informational_quantities).T

####
#### Simulation 
####


    
def simulate():
    ω1x = 2.0
    ω1z = 0.0
    
    ω2x = 2.0
    ω2z = 0.0
    
    gx, gy = 2, 2
    gz = 2.0
    
    k1 = 1.0
    k2 = 1.0
    
    n1 = 0.
    n2 = 0.
    
    ### Initial state in the ground-state of the bare Hamiltonian
    
    parameters = {"tspan": (0, 50),
                      "dt": 0.01,
                      "ω1x": ω1x,
                      "ω1z": ω1z,
                      "ω2x": ω2x,
                      "ω2z": ω2z,
                      "gx": gx,
                      "gy": gy,
                      "gz": gz,
                      "k1": k1,
                      "k2": k2,
                      "n1": n1, 
                      "n2": n2, 
                      "m1x0": 0.0,
                      "m1y0": 0.0,
                      "m1z0": -np.sqrt(1.0/2.0),
                      "m2x0": 0.0,
                      "m2y0": 0.0,
                      "m2z0": -np.sqrt(1.0/2.0),
                      "G0": np.array([[0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
                                      [0.0, 0.5, 0.0, 0.0, 0.0, 0.0],
                                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                      [0.0, 0.0, 0.0, 0.5, 0.0, 0.0],
                                      [0.0, 0.0, 0.0, 0.0, 0.5, 0.0],
                                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
                      "L0": np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                      [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                                      [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                                      [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                                      [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                                      [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
                  }
    
    time, sol = Simul_Traj(parameters)
    information = QuantumClassicalThermodynamics(sol)

    return time, sol, information

#def PhaseMap_n_g():
#    ω1x = 2.0
#    ω1z = 0.0
#    
#    ω2x = 2.0
#    ω2z = 0.0
#    
#    gx, gy = 2.0, 2.0
#    gz = 2.0 
#    
#    k1 = 1.0
#    k2 = 1.0
#
#    n1 = 1 
#    n2 = 0.5
#   
#    ### Initial state in the ground-state of the bare Hamiltonian
#    
#    parameters = {"tspan": (0, 50),
#                      "dt": 0.1,
#                      "ω1x": ω1x,
#                      "ω1z": ω1z,
#                      "ω2x": ω2x,
#                      "ω2z": ω2z,
#                      "gx": gx,
#                      "gy": gy,
#                      "gz": gz,
#                      "k1": k1,
#                      "k2": k2,
#                      "n1": n1, 
#                      "n2": n2, 
#                      "m1x0": 0.0,
#                      "m1y0": 0.0,
#                      "m1z0": -np.sqrt(1.0/2.0),
#                      "m2x0": 0.0,
#                      "m2y0": 0.0,
#                      "m2z0": -np.sqrt(1.0/2.0),
#                      "G0": np.array([[0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
#                                      [0.0, 0.5, 0.0, 0.0, 0.0, 0.0],
#                                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#                                      [0.0, 0.0, 0.0, 0.5, 0.0, 0.0],
#                                      [0.0, 0.0, 0.0, 0.0, 0.5, 0.0],
#                                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
#
#                      "L0": np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#                                      [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
#                                      [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
#                                      [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
#                                      [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
#                                      [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
#                  }
# 
#   
#    ### Parameter lists 
#    N_n2 = 20
#    N_g = 20
#
#    n2_list = np.linspace(0.5, 1.5, N_n2)
#    g_list = np.linspace(0.5, 1.5, N_g)
#
#    
#    ### Time interval to avarege 
#    tlen = parameters["tspan"][1]
#    Nint = int(tlen/parameters["dt"])
#
#    ### Phase-diagram maps heat matrix 
#    Heat_mat = np.zeros((N_n2, N_g))
#    SubH_mat = np.zeros((N_n2, N_g))
#
#    Entr_mat = np.zeros((N_n2, N_g))
#    Ccor_mat = np.zeros((N_n2, N_g))
#    Disc_mat = np.zeros((N_n2, N_g))
#    Nega_mat = np.zeros((N_n2, N_g))
#
#    ### counter 
#    counter = 0  
#
#    ### Looping 
#    for i, n2 in enumerate(n2_list):
#        for j, g in enumerate(g_list):
#
#            print("Interaction [{}] out of [{}], gz={}, g={} ".format(counter,N_n2 * N_g, n2, g))
#
#            ### setting parameter 
#            parameters["gz"] = g 
#            parameters["gy"] = g
#            parameters["gx"] = g
#            parameters["n2"] = n2
#    
#            ### getting the solution 
#            time, sol = Simul_Traj(parameters)
#
#            ## getting the information theory quantities
#            information = QuantumClassicalThermodynamics(sol)
#
#            ### getting heat  
#            total_heat = -(sol[:,-6]**2 + sol[:,-5]**2) - (sol[:,-3]**2 + sol[:,-2]**2)
#
#            ### Sub dominant heat current 
#
#            q_sub_1 = sol[:,0] + sol[:,0+7] + 2*np.sqrt(2)*sol[:,36+36+2]*(2*parameters["n1"] + 1)
#            q_sub_2 = sol[:,21] + sol[:,21+7] + 2*np.sqrt(2)*sol[:,36+36+5]*(2*parameters["n2"] + 1)
#
#            q_sub = q_sub_1 - q_sub_2
#
#            ### Information quantities 
#
#            Ccor = information[0]
#            Disc = information[1]
#            Nega = information[2]
#            Entr = information[4]
#
#            ### getting time interval (last quarter) 
#            Nint_init = int((3/4)*Nint)
#            
#            av_heat = 1/(time[-1] - time[Nint_init]) * simpson(total_heat[Nint_init:], dx=parameters["dt"])
#
#            av_subh = 1/(time[-1] - time[Nint_init]) * simpson(q_sub[Nint_init:], dx=parameters["dt"])
#
#            av_entr = 1/(time[-1] - time[Nint_init]) * simpson(Entr[Nint_init:], dx=parameters["dt"])
#
#            av_ccor = 1/(time[-1] - time[Nint_init]) * simpson(Ccor[Nint_init:], dx=parameters["dt"])
#
#            av_disc = 1/(time[-1] - time[Nint_init]) * simpson(Disc[Nint_init:], dx=parameters["dt"])
#
#            av_nega = 1/(time[-1] - time[Nint_init]) * simpson(Nega[Nint_init:], dx=parameters["dt"])
#            
#            Heat_mat[i][j] = av_heat
#            SubH_mat[i][j] = av_subh
#            Entr_mat[i][j] = av_entr
#            Ccor_mat[i][j] = av_ccor
#            Disc_mat[i][j] = av_disc
#            Nega_mat[i][j] = av_nega
#
#            counter += 1   
#
#    data = parameters.copy()
#    data.update({"PhaseMap_heat":Heat_mat})
#    data.update({"PhaseMap_subh":SubH_mat})
#    data.update({"PhaseMap_entr":Entr_mat})
#    data.update({"PhaseMap_ccor":Ccor_mat})
#    data.update({"PhaseMap_disc":Disc_mat})
#    data.update({"PhaseMap_nega":Nega_mat})
#    data.update({"n2_list": n2_list})
#    data.update({"g_list":g_list})
#
#
#    with open('data_phase_maps/Data_heat_gz_g.pickle', 'wb') as handle:
#        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
#
#    return Heat_mat, SubH_mat, Entr_mat, Ccor_mat, Disc_mat, Nega_mat
#
#def PhaseMap_g():
#    ω1x = 2.0
#    ω1z = 0.0
#    
#    ω2x = 2.0
#    ω2z = 0.0
#    
#    gx, gy = 2.0, 2.0
#    gz = 2.0 
#    
#    k1 = 1.0
#    k2 = 1.0
#
#    n1 = 0
#    n2 = 0
#   
#    ### Initial state in the ground-state of the bare Hamiltonian
#    
#    parameters = {"tspan": (0, 1000),
#                      "dt": 0.1,
#                      "ω1x": ω1x,
#                      "ω1z": ω1z,
#                      "ω2x": ω2x,
#                      "ω2z": ω2z,
#                      "gx": gx,
#                      "gy": gy,
#                      "gz": gz,
#                      "k1": k1,
#                      "k2": k2,
#                      "n1": n1, 
#                      "n2": n2, 
#                      "m1x0": 0.0,
#                      "m1y0": 0.0,
#                      "m1z0": -np.sqrt(1.0/2.0),
#                      "m2x0": 0.0,
#                      "m2y0": 0.0,
#                      "m2z0": -np.sqrt(1.0/2.0),
#                      "G0": np.array([[0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
#                                      [0.0, 0.5, 0.0, 0.0, 0.0, 0.0],
#                                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#                                      [0.0, 0.0, 0.0, 0.5, 0.0, 0.0],
#                                      [0.0, 0.0, 0.0, 0.0, 0.5, 0.0],
#                                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
#
#                      "L0": np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#                                      [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
#                                      [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
#                                      [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
#                                      [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
#                                      [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
#                  }
# 
#   
#    ### Parameter lists 
#    N_g = 40
#
#    g_list = np.linspace(0, 4, N_g)
#
#    
#    ### Time interval to avarege 
#    tlen = parameters["tspan"][1]
#    Nint = int(tlen/parameters["dt"])
#
#    ### Phase-diagram maps heat matrix 
#    Heat_mat = np.zeros(N_g)
#    Entr_mat = np.zeros(N_g)
#    Ccor_mat = np.zeros(N_g)
#    Disc_mat = np.zeros(N_g)
#    Nega_mat = np.zeros(N_g)
#
#    ### counter 
#    counter = 0  
#
#    ### Looping 
#    for i,g in enumerate(g_list):
#
#        gz = 4-g
#
#        print("Interaction [{}] out of [{}], gz={}, g={} ".format(counter,N_g, gz, g))
#
#        ### setting parameter 
#        parameters["gz"] = gz 
#        parameters["gy"] = g
#        parameters["gx"] = g
#    
#        ### getting the solution 
#        time, sol = Simul_Traj(parameters)
#
#        ## getting the information theory quantities
#        information = QuantumClassicalThermodynamics(sol)
#
#        ### getting heat  
#        total_heat = -(sol[:,-6]**2 + sol[:,-5]**2) - (sol[:,-3]**2 + sol[:,-2]**2)
#
#        Entr = information[4]
#        Ccor = information[0]
#        Disc = information[1]
#        Nega = information[2]
#
#        ### getting time interval (last quarter) 
#        Nint_init = int((3/4)*Nint)
#        
#        av_heat = 1/(time[-1] - time[Nint_init]) * simpson(total_heat[Nint_init:], dx=parameters["dt"])
#
#        av_entr = 1/(time[-1] - time[Nint_init]) * simpson(Entr[Nint_init:], dx=parameters["dt"])
#
#        av_ccor = 1/(time[-1] - time[Nint_init]) * simpson(Ccor[Nint_init:], dx=parameters["dt"])
#
#        av_disc = 1/(time[-1] - time[Nint_init]) * simpson(Disc[Nint_init:], dx=parameters["dt"])
#
#        av_nega = 1/(time[-1] - time[Nint_init]) * simpson(Nega[Nint_init:], dx=parameters["dt"])
#        
#        Heat_mat[i] = av_heat
#        Entr_mat[i] = av_entr
#        Ccor_mat[i] = av_ccor
#        Disc_mat[i] = av_disc
#        Nega_mat[i] = av_nega
#
#        counter += 1   
#
#    data = parameters.copy()
#    data.update({"PhaseMap_heat":Heat_mat})
#    data.update({"PhaseMap_entr":Entr_mat})
#    data.update({"PhaseMap_ccor":Ccor_mat})
#    data.update({"PhaseMap_disc":Disc_mat})
#    data.update({"PhaseMap_nega":Nega_mat})
#    data.update({"g_list":g_list})
#
#
#    with open('data_phase_maps/Data_heat_g.pickle', 'wb') as handle:
#        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
#
#    return Heat_mat, Entr_mat, Ccor_mat, Disc_mat, Nega_mat
#
#
#
