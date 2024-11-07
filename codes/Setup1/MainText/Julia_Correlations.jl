using PyCall 
using LinearAlgebra
using DifferentialEquations
using Printf

np = pyimport("numpy")

######
###### The dynamics of the covariance matrix and the mean-field quantities 
######


function func!(dydt, y, p, t)
    # Extract parameters
    ω1x, ω1z, ω2x, ω2z, gx, gy, gz, k1, k2, n1, n2 = p

    # Extract variables
    G = reshape(y[1:36], 6, 6)  # G matrix
    L = reshape(y[36+1:36+36], 6, 6)  # L matrix
    m1x, m1y, m1z = y[36+36+1], y[36+36+2], y[36+36+3]
    m2x, m2y, m2z = y[36+36+4], y[36+36+5], y[36+36+6]


    Dl = -[0.0  ω1z  0.0  0.0  0.0  0.0;
           -ω1z  0.0  ω1x  0.0  0.0  0.0;
           0.0 -ω1x  0.0  0.0  0.0  0.0;
           0.0  0.0  0.0  0.0  ω2z  0.0;
           0.0  0.0  0.0 -ω2z  0.0  ω2x;
           0.0  0.0  0.0  0.0 -ω2x  0.0]

    Db = -sqrt(2) * [0.0    0.0   -k1*m1x  0.0    0.0    0.0;
                     0.0    0.0   -k1*m1y  0.0    0.0    0.0;
                     k1*m1x k1*m1y  0.0    0.0    0.0    0.0;
                     0.0    0.0    0.0    0.0   -k2*m2x  0.0;
                     0.0    0.0    0.0    0.0    0.0   -k2*m2y;
                     0.0    0.0    0.0    0.0    k2*m2x  0.0]

    Dc = -sqrt(2) * [ 0.0         m2z*gz  -m2y*gy  0.0     0.0     0.0;
                     -m2z*gz      0.0     m2x*gx   0.0     0.0     0.0;
                      m2y*gy     -m2x*gx   0.0     0.0     0.0     0.0;
                      0.0         0.0      0.0     0.0     m1z*gz -m1y*gy;
                      0.0         0.0      0.0    -m1z*gz  0.0     m1x*gx;
                      0.0         0.0      0.0     m1y*gy -m1x*gx  0.0]

    Q = sqrt(2) * [ 0.0     0.0     0.0    0.0     m1z*gy  -m1y*gz;
                    0.0     0.0     0.0   -m1z*gx  0.0     m1x*gz;
                    0.0     0.0     0.0    m1y*gx -m1x*gy  0.0;
                    0.0     m2z*gy -m2y*gz 0.0     0.0     0.0;
                   -m2z*gx  0.0     m2x*gz 0.0     0.0     0.0;
                    m2y*gx -m2x*gy  0.0    0.0     0.0     0.0]

    s = sqrt(2) * [ 0.0  m1z -m1y  0.0  0.0  0.0;
                   -m1z  0.0  m1x  0.0  0.0  0.0;
                    m1y -m1x  0.0  0.0  0.0  0.0;
                    0.0  0.0  0.0  0.0  m2z -m2y;
                    0.0  0.0  0.0 -m2z  0.0  m2x;
                    0.0  0.0  0.0  m2y -m2x  0.0]

    B = [0.0 -k1  0.0  0.0  0.0  0.0;
         k1   0.0  0.0  0.0  0.0  0.0;
         0.0  0.0  0.0  0.0  0.0  0.0;
         0.0  0.0  0.0  0.0 -k2   0.0;
         0.0  0.0  0.0  k2   0.0  0.0;
         0.0  0.0  0.0  0.0  0.0  0.0]

    A = [k1*(2*n1+1) 0.0         0.0  0.0         0.0         0.0;
         0.0         k1*(2*n1+1) 0.0  0.0         0.0         0.0;
         0.0         0.0         0.0  0.0         0.0         0.0;
         0.0         0.0         0.0  k2*(2*n2+1) 0.0         0.0;
         0.0         0.0         0.0  0.0         k2*(2*n2+1) 0.0;
         0.0         0.0         0.0  0.0         0.0         0.0]

    #### Rotation dynamics
    dL = (Dl + Db + Dc) * L

    #### Covariance dynamics
    W = L' * (Q + s * B) * L
    dG = W * G + G * W' - L' * s * A * s * L	

    # Derivatives for m1 and m2 (replace these with the correct expressions)
    #
   
    dm1x = -sqrt(2.0)*(gz * m1y * m2z) + sqrt(2.0)*(gy*m1z*m2y) - (ω1z*m1y) + sqrt(2.0)*(k1 * m1x * m1z)
    dm1y = -sqrt(2.0)*(gx * m1z * m2x) + sqrt(2.0)*(gz*m1x*m2z) + (ω1z*m1x) - (ω1x * m1z) + sqrt(2.0)*(k1 * m1y * m1z)
    dm1z = -sqrt(2.0)*(gy * m1x * m2y) + sqrt(2.0)*(gx*m1y*m2x) + (ω1x*m1y) - sqrt(2.0)*k1*(m1x^2.0 + m1y^2.0)

    #### Mean-field dynamics for system 2
    dm2x = -sqrt(2.0)*(gz * m2y * m1z) + sqrt(2.0)*(gy*m2z*m1y) - (ω2z*m2y) + sqrt(2.0)*(k2 * m2x * m2z)
    dm2y = -sqrt(2.0)*(gx * m2z * m1x) + sqrt(2.0)*(gz*m2x*m1z) + (ω2z*m2x) - (ω2x * m2z) + sqrt(2.0)*(k2 * m2y * m2z)
    dm2z = -sqrt(2.0)*(gy * m2x * m1y) + sqrt(2.0)*(gx*m2y*m1x) + (ω2x*m2y) - sqrt(2.0)*k2*(m2x^2.0 + m2y^2.0)

    # Combine results
    dydt[1:36] .= reshape(dG, 36)
    dydt[36+1:36+36] .= reshape(dL, 36)
    dydt[36+36+1] = dm1x
    dydt[36+36+2] = dm1y
    dydt[36+36+3] = dm1z
    dydt[36+36+4] = dm2x
    dydt[36+36+5] = dm2y
    dydt[36+36+6] = dm2z
end


function get_params_and_initial_condition(g, gz)
    # Define parameters
    ω1x, ω1z = 2.0, 0.0
    ω2x, ω2z = 2.0, 0.0
    gx, gy, gz = g, g, gz 
    k1, k2 = 1.0, 1.0
    n1, n2 = 0.0, 0.0

    # Combine parameters into a vector
    params = [ω1x, ω1z, ω2x, ω2z, gx, gy, gz, k1, k2, n1, n2]

    # Initial condition matrices and vectors
   
    G0 = [0.5  0.0  0.0  0.0  0.0  0.0;
          0.0  0.5  0.0  0.0  0.0  0.0;
          0.0  0.0  0.0  0.0  0.0  0.0;
          0.0  0.0  0.0  0.5  0.0  0.0;
          0.0  0.0  0.0  0.0  0.5  0.0;
          0.0  0.0  0.0  0.0  0.0  0.0]
    L0 = Matrix(1.0I, 6, 6)  # Initial L matrix (6x6 identity matrix)
    m1x0, m1y0, m1z0 = 0.0, 0.0, -sqrt(0.5)  # Initial m1 vector components
    m2x0, m2y0, m2z0 = 0.0, 0.0, -sqrt(0.5)  # Initial m2 vector components

    # Combine initial conditions into a vector
    initial_conditions = [reshape(G0, 36); reshape(L0, 36); m1x0; m1y0; m1z0; m2x0; m2y0; m2z0]

    return params, initial_conditions
end

#########
######### Computing information theory quantities 
#########

function EvaluateEmin(c_alpha, c_beta, c_gamma, c_delta)
    if (c_delta - c_alpha * c_beta)^2 <= (1 + c_beta) * (c_gamma^2) * (c_alpha + c_delta) && c_beta > 1
        Emin = (2 * c_gamma^2 + (c_beta - 1) * (c_delta - c_alpha) +
                 2 * abs(c_gamma) * sqrt(c_gamma^2 + (c_beta - 1) * (c_delta - c_alpha))
                )
        Emin /= (c_beta - 1)^2
    elseif c_gamma^4 + (c_delta - c_alpha * c_beta)^2 - 2 * c_gamma^2 * (c_alpha * c_beta + c_delta) > 0
        Emin = (c_alpha * c_beta - c_gamma^2 + c_delta -
                sqrt(c_gamma^4 + (c_delta - c_alpha * c_beta)^2 - 2 * c_gamma^2 * (c_alpha * c_beta + c_delta))
               )
        Emin /= 2 * c_beta
    end
    return Emin
end

#### function defined for calculating discord (just time/spacing saving)
function func_log(x)
    ##### removing log with negative values in the argument
    if x <= 1
        return 0.0
    else
        return (x + 1) / 2 * log((x + 1) / 2) - (x - 1) / 2 * log((x - 1) / 2)
    end
end

function QuantumClassicalThermodynamics(sol)
    """
    Calculate quantum discord, one-way classical correlation, mutual information,
    according to the definition in the paper PRL 105, 030501 (2010)
    """
    num_rows, num_columns = size(sol)

    ##### This part will need further optimization, basically, I don't need to create another array of covariance matrices 
    covariance = CovarianceMatrix(sol)  # Define or import this function

    #### Vector for quantum discord 
    informational_quantities = Float64[]

    #### Compute quantum discord for each time 
    for i in 1:num_rows
        #### Get mean field at time t 
        mean_field = [sol[i, 36 + 36 + j] for j in 0:5]  # Get relevant columns
        cov = 2 * reshape(sol[1:36], 6, 6)  # G matrix 

        #### Get sub matrices 
        alpha = cov[1:3, 1:3]
        beta  = cov[4:6, 4:6]
        gamma = cov[1:3, 4:6]
	reducedCov = cov[[1,2, 4, 5], [1,2, 4, 5]]  # Remove specific rows and columns

        #### Upper diagonal block
        c_alpha = det(alpha)
        c_beta = det(beta)
        c_delta = det(reducedCov)
        c_gamma = det(gamma)

        #### Symplectic eigenvalues 
        Delta = c_alpha + c_beta + 2 * c_gamma
        TR_Delta = c_alpha + c_beta - 2 * c_gamma

        v_p = sqrt(0.5 * Delta + 0.5 * sqrt(Delta^2.0 - 4 * c_delta))  # The abs prevents numeric negative zeros
        v_m = sqrt(0.5 * Delta - 0.5 * sqrt(Delta^2.0 - 4 * c_delta))  # The abs prevents numeric negative zeros

        #### E min 
        Emin = EvaluateEmin(c_alpha, c_beta, c_gamma, c_delta)

        #### Thermodynamic quantities 
        J_one_way = func_log(sqrt(c_alpha)) - func_log(sqrt(Emin))
        Q_Discord = func_log(sqrt(c_beta)) - func_log(v_m) - func_log(v_p) + func_log(sqrt(Emin))

        Log_neg = max(0, -log(sqrt(0.5 * TR_Delta - 0.5 * sqrt(abs(TR_Delta^2 - 4 * c_delta)))))

        Total_entropy = func_log(v_p) + func_log(v_m)

        #### Appending everything in a vector 
        push!(informational_quantities, [J_one_way, Q_Discord, Log_neg, Total_entropy])
    end

    #### Returning the quantities 
    return hcat(informational_quantities...)  # Combine into a matrix
end

function simulate(tf)
	p, y0 = get_params_and_initial_condition(1.0, 1.0)

	tspan = (0.0, tf)
	prob = ODEProblem(func!, y0, tspan, p)
	sol = solve(prob, DP8(), abstol=1e-12, reltol=1e-12, maxiters=1e8, saveat=0.1,
		    save_everystep=false, progress=true)

	np.savetxt("sol.txt", sol.u)
	np.savetxt("time.txt", sol.t)

	return sol
end


