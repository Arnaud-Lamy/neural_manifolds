# math
import numpy as np
import math
from sklearn.decomposition import PCA

# plotting 
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
all_colors = list(mcolors.CSS4_COLORS)


# manifold traversal
from utils.utils_demo_growth import *
from utils.utils_TISVD import *

# helping
import pickle
import time
import subprocess
import os
import sys

np.random.seed(1234)

def train_network(N_start, N_end, errors, cur_network_params, R_nbrs, R_den, X):
    """
    Trains on some samples
    Input: 
        - N_start = starting index of training samples for this batch
        - N_end = ending index of training samples for this batch
        - cur_network_params = [local_params, nbrs_info, misc]
            - local_params = [Q, T, s_collection, P, Xi]
            - nbrs_info = [N1, W1, N0, W0]
            - misc = [tangent_colors, D, d, M, P]

    Output:
        - total_time = the time it took to train with this batch of data
        - errors
        - new_network_params = [new_local_params, new_nbrs_info, new_misc] (after output)
            - new_local_params = [Q, T, P, Xi]
            - new_nbrs_info = [N1, W1, N0, W0]
            - new_misc = [tangent_colors, D, d, M, P]

    Assumes the following params are floating around in the Jupyter Notebook:
    trunc_waves, manifold, N=N, manifold_type, output_path, R_denoising, R_1st_order_nbhd,d,D,sigma, X, X_natural

    NOTE: go into method to change if symmetry is enforced and errors are calculated yada yada
    """
    # load params
    [mean_MT_SE, mean_data_SE, all_MT_SE] = errors
    [local_params, nbrs_info, misc] = cur_network_params
    [Q, T, S_collection, P, Xi] = local_params
    [N1, W1, N0, W0] = nbrs_info
    [D, d, M, P] = misc


    start_time = time.time()
    
    # NOTE: X_natural is only used to calculate the MFD and data errors
    errors, new_local_params, new_nbrs_info, new_misc = main_fn(X, N=N_end,
                                                    d = d,
                                                    D = D,
                                                    N_cur = N_start,
                                                    R_den = R_den,
                                                    R_nbrs=R_nbrs,
                                                    all_MT_SE = all_MT_SE,
                                                    mean_MT_SE     = mean_MT_SE,
                                                    mean_data_SE   = mean_data_SE,
                                                    M = M,
                                                    Q = Q,
                                                    T = T,
                                                    S_collection = S_collection,
                                                    P = P,
                                                    N1 = N1,
                                                    W1 = W1,
                                                    Xi = Xi,
                                                    N0 = N0,
                                                    W0 = W0,
                                                    enforce_symmetry=False, ZOE_on_init=False, calc_error=False)

    end_time = time.time()
    total_time = end_time - start_time

    print('TOTAL TIME = ', total_time)
    new_network_params = [new_local_params, new_nbrs_info, new_misc]
    return [total_time, errors, new_network_params]


def save_data(errors, network_params, time_array, save_dir, prefix,suffix):
    # Save these into a pickle file
    [local_params, nbrs_info, misc]                       = network_params
    (mean_MT_SE, mean_data_SE, all_MT_SE)                 = errors
    (Q, T, S_collection, P, Xi)                           = local_params
    (N1, W1, N0, W0)                                      = nbrs_info
    (D, d, M, P)                                          = misc

    os.makedirs(save_dir, exist_ok=True)

    # Save each group in a separate pickle file
    with open(os.path.join(save_dir, f'{prefix}_errors_{suffix}.pkl'), 'wb') as f:
        pickle.dump((mean_MT_SE, mean_data_SE, all_MT_SE), f)

    with open(os.path.join(save_dir, f'{prefix}_local_params_{suffix}.pkl'), 'wb') as f:
        pickle.dump((Q, T, S_collection, P, Xi), f)

    with open(os.path.join(save_dir, f'{prefix}_nbrs_info_{suffix}.pkl'), 'wb') as f:
        pickle.dump((N1, W1, N0, W0), f)

    with open(os.path.join(save_dir, f'{prefix}_time_array_{suffix}.pkl'), 'wb') as f:
        pickle.dump(time_array, f)

    with open(os.path.join(save_dir, f'{prefix}_misc_{suffix}.pkl'), 'wb') as f:
        pickle.dump((D, d, M, P), f)

def train_denoiser_batches(R_den, R_nbrs, X_train, name, D, d, save_dir, batch_size=None):
    N_train = X_train.shape[1]
    # assume X_train is in R^{D x N}
    if batch_size == None:
        batch_size = N_train
    # initialize an MTN object 
    M = 0 # number of landmarks

    "local approximation info"
    Q = [] # list of landmarks w
    T = [] # list of basis matrices
    S_collection = [] # a list of sigma matrices obtained after each TISVD

    P = [] # number of points in each approximation 

    "first order graph info"
    N1 = [] # list of lists of first order neighbors
    W1 = [] # list of lists of weights of first order neighbors 
    Xi = [] # list of lists of edge embeddings 

    "zero order graph info" 
    N0 = [] # list of lists of zero order neighbors
    W0 = [] # list of lists of weights of zero order neighbors 


    local_params = [Q, T, S_collection, P, Xi]
    nbrs_info = [N1, W1, N0, W0]
    misc = [D, d, M, P]
    network_params = [local_params, nbrs_info, misc]
    N_cur = 0

    all_MT_SE      = []
    mean_MT_SE     = [] 
    mean_data_SE   = []

    errors = [mean_MT_SE, mean_data_SE, all_MT_SE]

    ## train online method in batches:
    num_batches = N_train // batch_size
    time_array = []
    errors_array = []
    for i in range(num_batches):
        start_index = i * batch_size
        end_index = start_index + batch_size
        cur_time, errors, network_params = train_network(start_index, end_index, errors, network_params, R_nbrs, R_den, X_train)
        time_array.append(cur_time)
        errors_array.append(errors)
        #save_data(errors, network_params, time_array, save_dir, name, {end_index}pts")
        print(f"{end_index} samples processed...")

    print(f"DONE... TOTAL TIME = {np.sum(np.array(time_array))}")

    new_save_dir = f"{save_dir}/{name}"
    os.makedirs(new_save_dir, exist_ok=True)
    save_data(errors, network_params, time_array, new_save_dir, name, f"{end_index}pts")
    return network_params, errors

