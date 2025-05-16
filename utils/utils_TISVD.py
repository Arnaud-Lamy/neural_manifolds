import numpy as np
import matplotlib.pyplot as plt
import time
import random
import sys
from tqdm import tqdm



def check_for_numerical_stability(x_perp, x_perp_norm, p_x, S_exp):
    '''
        This function checks for potential numerical stability issues. 
        If x_perp_norm is very small, it simply attaches a 0 instead of
        the small number that could cause numerical issues.
    '''
    if x_perp_norm <= 1e-6:
        # print('MUST AMEND X_PERP_NORM')
        Ma = np.vstack((p_x, 0)) # attach a clean 0 instead of x_perp_norm
        K = S_exp + np.dot(Ma, Ma.T)
        x_perp_new = x_perp * 1e6
        x_perp_norm_new = np.linalg.norm(x_perp_new)
        unit_norm_x_perp = x_perp_new / x_perp_norm_new
        # # print('Amended x_perp_norm = {}'.format(x_perp_norm_new))
        # # print('Amended unit_norm_x_perp = {}'.format(unit_norm_x_perp))
    else:
        # # print('NO NEED TO AMEND X_PERP_NORM')
        unit_norm_x_perp = (x_perp / x_perp_norm).reshape(-1,1)

        p_x_reshaped = p_x.reshape(-1, 1)

        # Reshape x_perp_norm to be (1,1)
        x_perp_norm_reshaped = np.array([x_perp_norm]).reshape(-1, 1)


        # # print('These two should have the same shape:')
        # # print('x_perp.shape = ', x_perp.shape)
        # print('unit_norm_x_perp.shape = ', unit_norm_x_perp.shape)
        # # print('x_perp_norm_reshaped.shape = ', x_perp_norm_reshaped.shape)

        Ma = np.vstack((p_x_reshaped, x_perp_norm_reshaped))
        # print('MA.shape = ', Ma)
        K = S_exp + np.dot(Ma, Ma.T)

    return unit_norm_x_perp, K

def TISVD(cur_x, U_old, S_old, i, d, E_list):
    '''
        This function is to be called when computing TISVD on data.
        This function takes one point cur_x, and efficiently computes TISVD using previous
        U and S matrices:

        U_old*S_old*U_old^T + cur_x * cur_x^T
        
        It outputs U_new and S_new_diag of the new matrix.
        Inputs:
            cur_x ------ ndarray of size (D, 1), current point being processed
            U_old ------ orthonormal matrix
            S_old ------ diagonal matrix
            i ---------- current iteration
            d ---------- intrinsic dimension
        
        Outputs:
            U_new ------------ D x d orthonormal matrix
            S_new_diag ------- a diagonal matrix of singular values, d x d
            E_list ----------- list of energy in the wrong coordinates of matrix U (i.e. d+1:D) coordinates

    '''

    p_x = np.dot(U_old.T, cur_x)
    x_perp = cur_x - np.dot(U_old, p_x)
    x_perp_norm = np.linalg.norm(x_perp)

    # expand S into an i+1 x i+1 matrix. Expan by adding a column of zeros
    S_exp_ = np.column_stack((S_old, np.zeros(S_old.shape[0])))
    S_exp = np.vstack((S_exp_, np.zeros(S_old.shape[1]+1))) # then add a row of zeros
    # # print('S_exp = ', S_exp.shape)

    unit_norm_x_perp, K = check_for_numerical_stability(x_perp, x_perp_norm, p_x, S_exp)

    U_K, S_K, _ = np.linalg.svd(K)

    U_new_  = np.dot(np.hstack((U_old, unit_norm_x_perp)), U_K)
    # # print('U_new_ shape = ', U_new_.shape)
    S_new_  = S_K.copy()

    if i >= d:
        # Take the first d columns - slicing creates a new array already
        U_new      = (U_new_[:, :d])
        # # print('U_new shape = ', U_new.shape)
        S_new_diag = (np.diag(S_new_[:d]))
    else:
        # Use the full array
        U_new = U_new_ # no need for copy since U_new_ is already a new array
        # # print('U_new shape = ', U_new.shape)
        S_new_diag = (np.diag(S_new_))

    # square and sum the values of U starting from row d for all columns
    E = np.sum(U_new[d:, :]**2)
    # E_list.append(E) # creates a side effect

    # # print('S_new_diag = ', S_new_diag.shape)
    
    # Return new list instead of modifying in place
    return U_new, S_new_diag, E_list + [E]




def TISVD_gw(cur_x, U_old, S_old, i, d):
    '''
        This function is to be called when computing TISVD on data.
        This function takes one point cur_x, and efficiently computes TISVD using previous
        U and S matrices:

        U_old*S_old*U_old^T + cur_x * cur_x^T
        
        It outputs U_new and S_new_diag of the new matrix.
        Inputs:
            cur_x ------ ndarray of size (D, 1), current point being processed
            U_old ------ orthonormal matrix of size (D, d)
            S_old ------ diagonal matrix
            i ---------- current iteration
            d ---------- intrinsic dimension
        
        Outputs:
            U_new ------------ D x d orthonormal matrix
            S_new_diag ------- a diagonal matrix of singular values, d x d
    '''

    # print('U_old.T = ', (U_old.T).shape)
    # print('cur_x.shape = ', cur_x.shape)

    p_x = (np.dot(U_old.T, cur_x))

    # print('p_x = np.dot(U_old.T, cur_x) = ', p_x.shape)

    x_perp = cur_x - np.dot(U_old, p_x)
    x_perp_norm = (np.linalg.norm(x_perp))

    # print('x_perp_norm.shape = ', x_perp_norm.shape)

    # expand S into an i+1 x i+1 matrix. Expan by adding a column of zeros
    # print('S_old.shape = ', S_old.shape)

    S_exp_ = np.column_stack((S_old, np.zeros(S_old.shape[0])))
    S_exp = np.vstack((S_exp_, np.zeros(S_old.shape[1]+1))) # then add a row of zeros

    # print('S_exp = ', S_exp.shape)

    unit_norm_x_perp, K = check_for_numerical_stability(x_perp, x_perp_norm, p_x, S_exp)
    # print('K.shape = ', K)

    U_K, S_K, _ = np.linalg.svd(K)

    # print('U_K.shape = ', U_K.shape)
    # print('S_K.shape = ', S_K.shape)

    U_new_  = np.dot(np.hstack((U_old, unit_norm_x_perp)), U_K)
    # # print('U_new_ shape = ', U_new_.shape)
    S_new_  = S_K.copy()

    # if i >= d:
    #     # Take the first d columns - slicing creates a new array already
    #     U_new      = (U_new_[:, :d])
    #     # # print('U_new shape = ', U_new.shape)
    #     S_new_diag = (np.diag(S_new_[:d]))
    # else:
    #     # Use the full array
    #     U_new = U_new_ # no need for copy since U_new_ is already a new array
    #     # # print('U_new shape = ', U_new.shape)
    #     S_new_diag = (np.diag(S_new_))

    U_new      = (U_new_[:, :d])
    S_new_diag = (np.diag(S_new_[:d]))
    # print('U_new shape = ', U_new.shape)
    # print('S_new_diag shape = ', S_new_diag.shape)

    # square and sum the values of U starting from row d for all columns
    E = np.sum(U_new[d:, :]**2)
    # E_list.append(E) # creates a side effect

    # # print('S_new_diag = ', S_new_diag.shape)
    
    # Return new list instead of modifying in place
    return U_new, S_new_diag









def TISVD_on_cov_matrix(X, d):
    '''
         Compute energy in the wrong coordinates.

        Inputs:
            X -------- data matrix of size D x n, where D is the extrinsic dimension. Should be centered about the landmark q.
            d -------- intrinsic dimension
    '''
    D = X.shape[0]
    n = X.shape[1]

    # center the dataset around landmark q
    # X_centered = X - q
    X_centered = X
    
    # extract the current point x to be the first data point from X
    x_i = X_centered[:, 0].reshape(-1,1) # make it into a column shape
    x_i_norm = np.linalg.norm(x_i)
    
    U_old = x_i/x_i_norm
    S     = np.array([x_i_norm**2])
    S_old = np.diag(S) # since S is a 1D array, np.diag makes it a diagonal matrix
    # # print(S_old.shape)

    # initializa an empty list for energy in the wrong coordinates
    E_list = []

    for i in tqdm(range(1, n)):
        # # print('_______________ i = {}_________________________'.format(i))
        cur_x = X_centered[:, i].reshape(-1,1)
        U_new, S_new_diag, E_list = TISVD(cur_x, U_old, S_old, i, d, E_list)

        U_old = U_new
        S_old = S_new_diag
        
    # frame_num += 1

    return U_new, S_new_diag, E_list


def TISVD_minibatch(cur_X, U_old, S_old, d, E_list):
    '''
        This function is to be called when computing TISVD via minibatches on a covariance matrix.
        This function takes the current batch of points cur_X, and efficiently computes TISVD of a new matrix
        U_old*S_old*U_old^T + cur_X * cur_X^T. It outputs U_new and S_new_diag of the new matrix.
        Inputs:
            cur_X ------ ndarray of size (D, d), current batch of points being processed
            U_old ------ orthonormal matrix
            S_old ------ diagonal matrix
            i ---------- current iteration
            d ---------- intrinsic dimension
        
        Outputs:
            U_new ------------ D x d orthonormal matrix
            S_new_diag ------- a diagonal matrix of singular values, d x d
            E_list ----------- list of energy in the wrong coordinates of matrix U (i.e. d+1:D) coordinates

    '''

    # # print('U_old = ', U_old.shape)
    # # print('cur_X = ', cur_X.shape)
    p_x = np.dot(U_old.T, cur_X)
    # x_perp = cur_X - np.dot(U_old, p_x)
    U_i_cur_X = np.concatenate((U_old, cur_X), axis=1)
    # # print('U_i_cur_X = ', U_i_cur_X.shape)

    Q, R = np.linalg.qr(U_i_cur_X)
    # # print('Q = ', Q.shape)
    # # print('R = ', R.shape)

    S_exp = np.zeros((2*d, 2*d))
    S_exp[0:d, 0:d] = S_old

    Ma = R[:, d:]
    # # print('Ma = ', Ma.shape)
    K = S_exp + np.dot(Ma, Ma.T)
    U_K, S_K, _ = np.linalg.svd(K)

    U_new_  = np.dot(Q, U_K)
    # # print('U_new_ shape = ', U_new_.shape)
    S_new_  = S_K.copy()

    # Take the first d columns - slicing creates a new array already
    U_new      = (U_new_[:, :d])
    S_new_diag = (np.diag(S_new_[:d]))

    # # print('U_new shape = ', U_new.shape)
    # # print('S_new_diag = ', S_new_diag.shape)
    
    # square and sum the values of U starting from row d for all columns
    E = np.sum(U_new[d:, :]**2)
    # E_list.append(E) # creates a side effect

    
    # Return new list instead of modifying in place
    return U_new, S_new_diag, E_list + [E]



def TISVD_on_cov_matrix_minibatch(X, d):
    '''
         Compute energy in the wrong coordinates.

        Inputs:
            q -------- current landmark or center point
            X -------- data matrix of size D x n, where D is the extrinsic dimension. Should be centered about the landmark q.
            d -------- intrinsic dimension
            R_loc ---- points inside the R_loc ball will be assigned to landmark q
            elev ----- elevation for viewing the 3d plots
            azim ----- aizmuth for viewing the 3d plots
            plot_circle ---- True only if we are plotting a 2D circle
    '''
    D = X.shape[0]
    n = X.shape[1]

    # center the dataset around landmark q
    # X_centered = X - q
    X_centered = X
    
    # extract the current point x to be the first data point from X
    X_i = X_centered[:, :d].reshape(-1,d) # make it into a column shape
    Cov_X_i = X_i @ X_i.T

    U, S, E_list = TISVD_on_cov_matrix(X_i, d)
    U_old = U[:, :d]

    # # print('X_i =     ', X_i.shape)
    # # print('Cov_X_i = ', Cov_X_i.shape)
    # # print('U_old =   ', U_old.shape)
    # # print('S =   ', S.shape)

    S_old = np.diag(S[:d]) # since S is a 1D array, np.diag makes it a diagonal matrix
    # # print('S_old = ', S_old.shape)
    
    # Process remaining points in batches of size d
    num_complete_batches = n // d
    # # print('num_complete_batches = ', num_complete_batches)

    for i in tqdm(range(1, num_complete_batches)):
        start_idx = (i-1)*d
        end_idx = i*d
        # # print(f'_______________ batch {i} (points {start_idx}:{end_idx})_________________________')
        
        cur_X = X_centered[:, start_idx:end_idx].reshape(-1,d)
        U_new, S_new_diag, E_list = TISVD_minibatch(cur_X, U_old, S_old, d, E_list)

        U_old = U_new
        S_old = S_new_diag

    return U_new, S_new_diag, E_list