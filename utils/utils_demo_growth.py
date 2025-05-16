# math
import numpy as np
import math

# plotting 
import matplotlib.pyplot as plt 
import matplotlib.colors as mcolors
all_colors = list(mcolors.CSS4_COLORS)

from utils.utils_TISVD import TISVD_gw

from tqdm import tqdm


from sklearn.decomposition import TruncatedSVD
from scipy.linalg import svd

# Gamma = []
N0 = []
N1 = []


def MT_perform_traversal(x, Q, T, N1, W1, Xi, N0, W0, calc_mults = False):

    '''
        Inputs:
            x ------------ the given point to be denoised
            Q ------------ landmarks
            T ------------ a matrix of tangent space basis vectors
            N1 ----------- a list of lists indicating first-order edges?
            W1 ----------- weight matrix for 1st-order edges
            Xi ----------- a matrix of the psi edge embeddings in the tangent space of vertex i ?
            N0 ----------- number of zero-order edges?
            W0 ----------- weight matrix for zero-order edges ?


        Outputs:
            i ------------ the final vertex
            phi ---------- the final objective value
            trajectory --- the list of vertices visited
            edge_orders -- list of edge orders used (0 or 1) for each step in the trajectory   
            mults -------- the number of multiplications performed (0 if calc_mults = False)
    '''
    mults = 0
    if calc_mults == True:
        D = len(x)
        d = len(T[0][0])

    i = 0  # starting vertex 

    converged = False
    iter = 0

    trajectory = [i]
    edge_orders = []

    while not converged: 
        # calculate and # print current objective 
        phi = np.sum( (Q[i] - x) ** 2 )

        if calc_mults:
            mults += D 

        # compute Riemannian gradient in coordinates
        # this is the gradient of .5 || q - x ||_2^2 with respect to q
        grad_phi = T[i].transpose() @ ( Q[i] - x )

        if calc_mults:
            mults += D*d

        # 1st order outdegree of vertex i
        # ie how many 1st degree neighbors does vertex i have
        deg_1_i = len( N1[i] )

        # find the most correlated edge embedding -- this is the speculated next vertex
        next_i = 0 
        best_corr = math.inf 

        # check the correlation for each 1st order edge of vertex i with the gradient
        for j in range( 0, deg_1_i ): 
            corr = np.dot( Xi[i][j], grad_phi )

            if (corr < best_corr):
                best_corr = corr
                next_i = N1[i][j]
        
        
        if calc_mults:
            mults += d * deg_1_i

        # compute objective value at speculated next vertex 
        next_phi = np.sum( (Q[next_i] - x) ** 2 ) 

        if calc_mults:
            mults += D

        if (next_phi >= phi):
            # If the first-order step doesn't improve the objective
            # Try a zero-order step by checking all zero-order neighbors
            # Move to the neighbor with the best (lowest) objective value
            # set order = 0

            # first order step failed, try zero order step 
            best_i = 0 
            best_phi = math.inf 

            # zero-order out-degree of this vertex
            deg_0_i = len( W0[i] )

            # compute the objective at each of the neighbors, record the best objective 
            if calc_mults:
                mults += D * deg_0_i

            for j in range( 0, deg_0_i ):
                
                cur_nbr_phi = np.sum( ( Q[ N0[i][j] ] - x ) ** 2 ) 

                if (cur_nbr_phi < best_phi):
                    best_phi = cur_nbr_phi
                    best_i = N0[i][j]

            order = 0
            next_i = best_i
            next_phi = best_phi
        else:
            # If moving to the selected neighbor improves the objective
            # Move to that neighbor (first-order step)
            # and set order = 1
            order = 1

        if (next_i == i):
            # If no improvement is found (i.e., next_i == i)
            # # declare convergence and exit the loop.
            # # print('   MT converged') 
            converged = True 
        else:
            # Otherwise, update i to the new vertex, append to the trajectory, and continue the loop.
            i = next_i 
            phi = next_phi 

            trajectory.append(i) 
            edge_orders.append(order) 

            iter += 1 
            
    return i, phi, trajectory, edge_orders, mults

def MT_denoise_local(x,Q,T,i):

    # denoise by projection -- this finds the closest point in Q[i] + span(T[i]) to x:
    x_hat = Q[i] + T[i] @ ( T[i].transpose() @ ( x - Q[i] ) )

    return x_hat 

def MT_initialize_new_landmark(x,Q,T,P,N1,W1,Xi,N0,W0, D, d, S_collection, distances_sq, R_nbrs): 

    M=len(Q)
    if (M == 0): 
        
        "initialize landmark and tangent space" 
        Q.append( x.copy() )                          # initialize the new landmark as the current sample 

        # SCIPY
        random_matrix = np.random.randn(D, d)
        U, s, Vt = svd(random_matrix, full_matrices=False)  # Economy SVD
        U_new = U[:, :d]  # Take first d columns
        s_new = s[:d]     # Take first d singular values
        S_new_diag = np.diag(s_new)  # If you need the diagonal matrix


        T.append( U_new )                                 # make this our initial tangent space estimate
        S_collection.append(S_new_diag)                        # save our initial matrix of singular values 
    
        P.append( 1 )                                 # number of points contributing to this model 
    
        " initialize first order graph info " 
        N1.append( [ 0   ] )                          # initially, only first order neighbor is the vertex itself  
        W1.append( [ 1.0 ] )                          # give the self-edge unit weight
        Xi.append( [ np.zeros((d,)) ] )               # initially, only a zero edge embedding for the self-edge
        # # print('LITTLE d = ', d)
    
        " initialize zero order graph info " 
        N0.append( [ 0   ] )                          # initially, only zero order neighbor is the vertex itself 
        W0.append( [ 1.0 ] )                          # give the self-edge unit weight 

    else: # if we have seen more than one point

        "initialize landmark and tangent space" 
        Q.append( x.copy() )                          # initialize the new landmark as the current sample 
        
        P.append( 1 )                                 # this landmark represents 1 data point so far

        " initialize zero order graph info " 
        N0.append( [ M   ] )                          # initially, only zero order neighbor is the vertex itself 
        W0.append( [ 1.0 ] )                          # give the self-edge unit weight 
        
        "initialize first order graph info " 
        N1.append( [] )
        W1.append( [] )
        Xi.append( [] ) 
        
        # Find 1st order neighbors within radius R_1st_order_nbhd

        # This is gross and annoying. 
        # I am fighting numpy but the payoff is that it should be faster than for loops (previous thing) 
        # and it should only do exhaustive search once
        temp1 = np.argwhere(distances_sq <= R_nbrs**2)
        if len(temp1)==0:
            temp3 = temp1
        else:
            temp2=np.stack(temp1,axis=1)
            temp3 = temp2[0]
            
        idces = temp3

        for idx in idces:
            # this is not a self-edge ... need to also make M a neighbor of l
            N1[idx].append( M   )
            W1[idx].append( 1.0 )
            N1[M].append(idx)
            W1[M].append(1.0)
            # # print('N1 = ', N1)
        
        N1[M].append(M)
        W1[M].append(1.0)

        # JW: random tangent space could be replaced with average of neighbors 
        # 
        # can use both secant directions and tangent directions at neighbors ... 

        # check if the new point has more than one neighbor
        # if it does, update its tangent space
        if ( len(N1[M]) > 1 ): 
            # len(N1[M])-1 is number of neighbors (excluding self)

            "Form H with difference vectors AND with estimated tangent spaces"
            # # iterate over all 1st-order neighbors of point M
                
                # This is what H looks like
                # H = [  Direction1 | TangentSpace1 | Direction2 | TangentSpace2 | ... ]
                #        <1 column>   <d columns>     <1 column>   <d columns>

            "Form H with difference vectors only"
            H = np.zeros( ( D, (d+1) * (len(N1[M])-1) ) )
            # iterate over all 1st-order neighbors of point M
            for l in range(0,len(N1[M])-1):
                H[ :, l ] = ( Q[N1[M][l]] - Q[M] ) / np.linalg.norm( Q[N1[M][l]] - Q[M] )

            # SCIPY
            U, s, Vt = svd(H, full_matrices=False)  # Economy SVD
            U_new = U[:, :d]  # Take first d columns
            s_new = s[:d]     # Take first d singular values
            S_new_diag = np.diag(s_new)  # If you need the diagonal matrix

            # update the tangent space
            T.append( U_new ) # make this our initial tangent space estimate
            S_collection.append(S_new_diag)    


        else: # if the point has no neighbors, just itself
            # no neighbors, just use a random subspace ... 
            # SCIPY
            random_matrix = np.random.randn(D, d)
            U, s, _ = svd(random_matrix, full_matrices=False)  # Economy SVD
            U_new = U[:, :d]  # Take first d columns
            s_new = s[:d]     # Take first d singular values
            S_new_diag = np.diag(s_new)  # If you need the diagonal matrix

            T.append( U_new )                                 # make this our initial tangent space estimate
            S_collection.append(S_new_diag)                        # save our initial matrix of singular values 
        


        # JW: can condense these loops to just use the neighbor information, rather than recompute locality 
        for l in range(0,len(N1[M])):
            Xi[M].append( T[M].transpose() @ ( Q[N1[M][l]] - Q[M] ) )
            Xi[N1[M][l]].append( T[l].transpose() @ ( Q[M] - Q[N1[M][l]] ) )

def MT_update_local_SGD_TISVD(x, Q, T, P, Xi, i, d, N1, S_collection): 
    '''
        Update the local parameters using truncated incremental SVD per sample (not minibatch).

        Inputs:
            x -------------------- current point
            Q -------------------- collecgtion of landmarks
            T -------------------- a collection of tangent space basis vectors
            P -------------------- counter for the number of updates at each point (keeps count of # of pts assigned per landmark)
            Xi ------------------- collection of edge embeddings
            i -------------------- index of the current point
            d -------------------- intrinsic dimension
        Outputs:
            U[:,0:d] ------------- the orthonormal matrix U with d columns
    
    '''

    P[i] += 1

    Q[i] = (( P[i] - 1.0 ) / P[i] ) * Q[i] + ( 1.0 / P[i] ) * x

    U_old = T[i-1]
    S_old = S_collection[i-1]
    U_new, S_new_diag = TISVD_gw(x, U_old, S_old, i, d)

    T[i] = U_new.copy()

    S_collection[i] = S_new_diag.copy()
    
    # recompute edge embeddings 
    for j in range(0,len(N1[i])):
        i_pr = N1[i][j]
        Xi[i][j] = T[i].transpose() @ ( Q[i_pr] - Q[i] )
        for l in range(0,len(N1[i_pr])):
            if ( N1[i_pr][l] == i ):
                Xi[i_pr][l] = T[i_pr].transpose() @ ( Q[i] - Q[i_pr] )

def main_fn(X, N, 
            d = 2, D = 3, N_cur = 0,
            R_den = lambda P_i: 0.65,
            R_nbrs = 0.7,
            all_MT_SE = [],
            all_data_SE    = [], 
            mean_MT_SE     = [], 
            mean_data_SE   = [],
            frame_num = 0,
            M = 0, Q = [], T = [], S_collection = [], P = [],
            N1 = [], W1 = [], Xi = [], N0 = [], W0 = [], enforce_symmetry=False, ZOE_on_init=False, calc_error=True):
    '''
        This function runs the online algorithm.
    '''



    while (N_cur < N): #TODO: add tqdm... was bugged

        # grab the current sample 
        x = X[:,N_cur]

        if (M == 0):
            # current traversal network is empty -- let's use this sample to initialize a landmark
            # JW: this should eventually be cut out into a 'new landmark' function, which also handles the case of a non-empty graph
            MT_initialize_new_landmark(x,Q,T,P,N1,W1,Xi,N0,W0, D, d, S_collection, None, R_nbrs)
            M += 1 
            x_hat = x


        else:

            " Current Network Structure " 
            i, phi, trajectory, order, _ = MT_perform_traversal(x,Q,T,N1,W1,Xi,N0,W0)

            " Evaluate the result of traversal and update network parameters " 
            if ( phi <= (R_den(P[i]))**2): 

                # this sample is an inlier ... perform denoising and update model parameters  

                # denoise by projection onto local model at the i-th landmark 
                x_hat = MT_denoise_local(x,Q,T,i) 

                MT_update_local_SGD_TISVD(x,Q,T,P,Xi,i,d, N1, S_collection)
                        
            else:
                # perform exhaustive search 
                i_best = 0 
                # compute the objective at each of the neighbors, record the best objective 
                if M==1:
                    distances_sq=np.array([np.sum((np.array(Q[0]).reshape(1,D)-x)**2)]) # fighting dimensions stuff
                else:
                    temp_test = np.array(Q)
                    distances_sq = np.sum((temp_test-x).T**2, axis=0)

                i_best = np.argmin(distances_sq)
                if ( distances_sq[i_best] <= (R_den(P[i_best]))**2): 
                    # add the neighbor, with weight 1 
                    N0[i].append(i_best)
                    W0[i].append( 1.0 )
                    if enforce_symmetry:
                        W0[i_best].append(1.0)
                        N0[i_best].append(i)

                    # denoise by projection onto local model at the i-th landmark 
                    x_hat = MT_denoise_local(x,Q,T,i_best) 
        
                    # update local model around the i-th landmark
                    MT_update_local_SGD_TISVD(x,Q,T,P,Xi,i,d, N1, S_collection)

                else:                     
                    MT_initialize_new_landmark(x,Q,T,P,N1,W1,Xi,N0,W0, D, d, S_collection, distances_sq, R_nbrs)
                    # no local model, so just make the trivial prediction
                    x_hat = x 
                    
                    if ZOE_on_init:
                        N0[i].append(M)
                        W0[i].append(1.0)

                        if enforce_symmetry:
                            N0[M].append(i)
                            W0[M].append(1.0)

                    M += 1 
            frame_num += 1

        
        N_cur += 1

    return (mean_MT_SE, mean_data_SE, all_MT_SE), (Q,T,S_collection,P,Xi), (N1,W1,N0,W0), (D, d, M, P)
