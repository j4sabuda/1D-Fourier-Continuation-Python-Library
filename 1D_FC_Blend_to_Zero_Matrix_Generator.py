# -*- coding: utf-8 -*-
"""
This code implements the 1D-FC "blend to zero" matrix generation process as
described by Bruno et al. in the following papers:
    
    https://arxiv.org/abs/2010.03901
    https://www.sciencedirect.com/science/article/abs/pii/S0021999115008086?via%3Dihub

It takes as inputs:
    d: The number of function values to sample; typically 1 <= d <= 12
    C: The number of continuation points to add.
    Z: The number of points in the "zero" region.
    E: The number of points following the zero region. Typically E = C
    n_over: The oversampling factor; typically n_over = 20
    modes_to_reduce: The number of Fourier modes to subtract from the least squares problem.
    num_digits: The number of digits to carry out the calculations with; default is 256.

And returns a single C x d matrix A, which can be used to compute the periodic
continuation of a given sufficiently smooth, non-periodic data set. An example
of how A is used is provided at the end of the code below.

Note: While this computation can take a significant amount of time, the 
returned matrix A is not problem-dependent; once calculated and assuming the 
underlying calculation parameters (typically just d) work for a given problem, 
it can be saved and re-used as needed.

@author: Josef M Sabuda

Note 2: While Josef M Sabuda authored this Python code, he is neither a 
co-author or inventor of FC. Please direct any theoretical questions about the 
underlying algorithm to the contacts specified in the above referenced papers.
"""

#Import libraries
import mpmath as mp
import sympy as sp
import numpy as np

def Generate_1D_FC_matrix_A(d,n_over=20,C=27,Z=12,E=27,num_digits=256,modes_to_reduce=0):
    #Define various parameters and initialize various grids
    
    #the total number of points in the coarse domain
    coarse_N = d+C+Z+E
    
    #the bandwidth
    bandwidth = coarse_N // 2 - modes_to_reduce
    
    #the total number of oversampled points in the sampled interval
    num_oversampled_original = n_over*(d-1)+1
    
    #the total number of oversampled points in the zero interval
    num_oversampled_zero = n_over*(Z-1)+1
    
    #the total number of points in the oversampled domain
    oversampled_N = coarse_N*n_over
    
    #the coarse grid for the long interval [0,1)
    coarse_full_grid = sp.Array(range(coarse_N))/coarse_N
    
    #the coarse grid for the sampled interval
    coarse_short_grid = sp.Array(range(d))/coarse_N
    
    #the fine grid for the sampled interval
    fine_short_grid = sp.Array(range(num_oversampled_original))/oversampled_N
    
    # The fine grid for the zero interval
    fine_zero_grid = sp.Array([(sp.Integer(x) + sp.Integer(d + C) * sp.Integer(n_over)) / sp.Integer(oversampled_N) for x in range(num_oversampled_zero)])
    
    # The Fourier modes
    k = sp.Matrix([x - bandwidth for x in range(2 * bandwidth + 1)]).transpose()
    
    #the number of Fourier modes
    num_modes = 2*bandwidth+1
            
    # Generate the polynomial basis for the coarse and fine grids
    #these variables will hold the polynomial basis on the coarse and fine grids respectively
    coarse_p = sp.Matrix([[coarse_short_grid[i]**j for j in range(d)] for i in range(d)])
    fine_p = sp.Matrix([[fine_short_grid[i]**j for j in range(d)] for i in range(num_oversampled_original)])
    
    # use QR to construct coarse basis
    coarse_Q,coarse_R = coarse_p.QRdecomposition()
    
    #use R to construct fine basis
    fine_Q_sym = fine_p*coarse_R.inv()
    
    # Evaluate fine_Q_sym to num_digits
    fine_Q = mp.matrix(fine_Q_sym.evalf(num_digits).tolist())
    
    ## generate the continuations
    # Generate the reconstruction matrix on the fine mesh
    product = sp.Matrix.vstack(sp.Matrix(fine_short_grid), sp.Matrix(fine_zero_grid)) * k
    A1 = product.applyfunc(lambda x: sp.cos(2 * sp.pi * x))  # Apply cos element-wise
    A2 = product.applyfunc(lambda x: sp.sin(2 * sp.pi * x))  # Apply sin element-wise
    A_p_sym = A1.row_join(A2)  # Join the matrices
    
    # Generate the reconstruction matrix on the total coarse mesh
    product = sp.Matrix(coarse_full_grid) * k
    C1 = product.applyfunc(lambda x: sp.cos(2 * sp.pi * x))  # Apply cos element-wise
    C2 = product.applyfunc(lambda x: sp.sin(2 * sp.pi * x))  # Apply sin element-wise
    C_m_sym = C1.row_join(C2)  # Join the matrices
    
    # Evaluate A_p_sym and C_m_sym to num_digits precision
    mp.mp.dps = num_digits
    
    # Convert A_p_sym and C_m_sym to mpmath matrices with num_digits precision
    A_p = mp.matrix(A_p_sym.evalf(num_digits).tolist()) 
    C_m = mp.matrix(C_m_sym.evalf(num_digits).tolist()) 
    
    # Calculate the SVD of A_p; this is the most computationally intensive step
    U,S,V = mp.svd_r(A_p)
    
    # Adjust V from SVD
    V = V.T #mp.svd_r returns V.T, we want V
    
    # Compute z and determine ind; ind determines how many singular values to keep
    z = [S[i] / S[i + 1] for i in range(len(S) - 1)]
    ind = next((i for i in range(len(z)) if z[i] > 1e16), num_modes)
    ind = min(ind, num_modes)
    ind+=1
    
    # Construct diagonal matrix t
    t = mp.diag([1 / mp.mpf(S[i]) for i in range(ind)])
    
    # Update U and V
    V = V[:, :ind]
    U = U[:, :ind]
    
    # Initialize cont_data
    cont_data = mp.matrix(coarse_N, d)
    
    # Loop through each point
    for kk in range(d):
        # Build b by concatenating fine_Q and a zero vector
        b = mp.matrix(list(fine_Q[:, kk]) + [mp.mpf(0)] * num_oversampled_zero)
    
        # Compute a
        a = V * (t * (U.T * b))
    
        # Update cont_data
        for j in range(coarse_N):
            cont_data[j, kk] = mp.fsum(C_m[j, k] * a[k] for k in range(2 * num_modes))
    
    # Calculate Q,A as numpy arrays
    Q = np.array(coarse_Q.evalf(num_digits).tolist(), dtype=float)
    A = np.array([[float(cont_data[i, j]) for j in range(cont_data.cols)] for i in range(d, d + C)])
    
    # Calculate final A as a numpy array
    A = A@Q.T
    
    return(A)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Define x_data via the step size and range, for non-periodic y_data
    step_size = 0.005
    x_data = np.arange(0, 1 + step_size, step_size)

    # Define example non-periodic y_data 
    y_data = np.exp(np.sin(5.4 * np.pi * x_data - 2.7 * np.pi) - np.cos(2 * np.pi * x_data))
    
    # Select number of edge points to use in FC by setting d
    d = 6
    
    #Take d points from the beginning and end of y_data
    f_left = y_data[:d][::-1] #Reverse order
    f_right = y_data[-d:]
    
    #Calculate A
    A = Generate_1D_FC_matrix_A(d,n_over=20,C=27,Z=12,E=27,num_digits=256,modes_to_reduce=0)
    
    # Calculate Af_left and Af_right
    Af_left = (A @ f_left)[::-1] #Reverse order
    Af_right = A @ f_right.T
    
    #Calculate the full continuation, Af
    Af = Af_left+Af_right
    
    #Plot output
    n = len(Af)
    continued_x_data = np.arange(x_data[-1] + step_size, x_data[-1] + n * step_size, step_size)
    
    plt.figure(0)
    plt.plot(x_data,y_data,'k',label='Input Data')
    plt.plot(continued_x_data,Af,'r',label='FC Data')
    plt.legend()
    plt.show()
