import h5py
import numpy as np
from sklearn.decomposition import PCA
import math
from sklearn.decomposition import FactorAnalysis
import matplotlib.pyplot as  plt
import os

import plotly.graph_objects as go
import numpy as np

def batch_ar_band_features(data, fs=1000, order=20, win_size=1000, step_size=100, n_fft=512,frequency_bands = [
    (0.3, 5),    # δ
    (5, 15),     # θ–α
    (15, 30),    # β
    (30, 50),    # γ1
    (50, 100),   # γ2
    (100, 200),  # γ3
    (200, 400),  # hfECoG
]):
    """
    Compute AR-based band power features for each electrode and time window, looping over channels first.

    Parameters:
    - data: (T, C) array, where T is time samples and C is number of electrodes
    - fs: Sampling frequency (Hz)
    - order: AR model order
    - win_size: Window size in samples (e.g., 1000 for 1s)
    - step_size: Step size in samples (e.g., 100 for 100ms)
    - n_fft: FFT resolution

    Returns:
    - band_feature_arrays: list of 7 arrays, each of shape (num_windows, num_channels)
      One array per frequency band (δ, θ–α, etc.)
    """
    num_samples, num_channels = data.shape
    num_windows = (num_samples - win_size) // step_size + 1

    # Prepare storage for each band
    num_bands = len(frequency_bands)
    band_feature_arrays = [np.zeros((num_windows, num_channels)) for _ in range(num_bands)]

    for ch in range(num_channels):
        for win_idx in range(num_windows):
            start = win_idx * step_size
            end = start + win_size
            signal = data[start:end, ch]

            try:
                freqs, psd = modified_covariance_ar_psd(signal, fs=fs, order=order, n_fft=n_fft)
                band_powers = average_psd_in_bands(freqs, psd, frequency_bands)
                for band_idx, power in enumerate(band_powers):
                    band_feature_arrays[band_idx][win_idx, ch] = np.log(power + 1e-10)
            except Exception as e:
                print(f"AR model failed at channel {ch}, window {win_idx}: {e}")
                for band_idx in range(num_bands):
                    band_feature_arrays[band_idx][win_idx, ch] = np.nan

        print(f"Finished processing channel {ch}")

    return band_feature_arrays

def modified_covariance_ar_psd(x, fs=1000, order=20, n_fft=512):
    """
    Estimate the power spectral density (PSD) of a signal using the modified covariance AR method.

    Parameters:
    - x: 1D numpy array, the signal
    - fs: Sampling frequency in Hz
    - order: Order of the AR model
    - n_fft: Number of FFT points to evaluate the PSD

    Returns:
    - freqs: Frequencies (Hz)
    - psd: Power spectral density estimate
    """
    x = np.asarray(x)
    N = len(x)
    p = order

    # Only use valid points
    T = N - 2 * p
    if T <= 0:
        raise ValueError("Signal too short for given AR order.")

    X_f = np.array([x[n - np.arange(1, p + 1)] for n in range(p, N - p)])
    y_f = x[p:N - p]

    X_b = np.array([x[n + np.arange(1, p + 1)] for n in range(p, N - p)])
    y_b = x[p:N - p]

    X = np.vstack([X_f, X_b])
    y = np.concatenate([y_f, y_b])

    # Solve normal equations
    a = np.linalg.inv(X.T @ X) @ X.T @ y

    # Estimate noise variance
    sigma2 = np.mean((y - X @ a)**2)

    # Evaluate PSD from AR coefficients
    freqs = np.linspace(0, fs / 2, n_fft)
    omega = 2 * np.pi * freqs / fs

    denom = np.abs(1 - np.sum([a[k] * np.exp(-1j * omega * (k + 1)) for k in range(p)], axis=0))**2
    psd = sigma2 / denom

    return freqs, psd

def average_psd_in_bands(freqs, psd, bands):
    """
    Average PSD values within given frequency bands.

    Parameters:
    - freqs: Array of frequency values
    - psd: PSD values corresponding to freqs
    - bands: List of (low, high) tuples for frequency bands

    Returns:
    - band_means: List of average PSDs in each band
    """
    band_means = []
    for low, high in bands:
        idx = np.where((freqs >= low) & (freqs < high))[0]
        if len(idx) > 0:
            band_means.append(np.mean(psd[idx]))
        else:
            band_means.append(0.0)
    return band_means

def perform_PCA(data, n_components=3):
    pca = PCA(n_components=n_components).fit(data)
    transformed = pca.transform(data)
    U = pca.components_.T
    centering_mean = pca.mean_
    
    return transformed, U, centering_mean, pca #, explained_variance


def perform_factor_analysis(data, n_components=10):
    """
    Perform factor analysis on neural data.
    
    Parameters:
        data: array of shape (n_samples, n_features) -- e.g., (T, q)
              where each row is a time point, and each column is a sensor/electrode.
        n_components: number of latent dimensions (e.g., 10)
    
    Returns:
        z: latent variables (T x n_components)
        Lambda: loading matrix (q x n_components)
        mu: mean vector (q,)
        Psi: diagonal noise variance (q,)
    """
    fa = FactorAnalysis(n_components=n_components)
    z = fa.fit_transform(data)               # latent variables z_t
    Lambda = fa.components_.T                # loading matrix (q x d)
    mu = fa.mean_                            # mean vector (q,)
    Psi = fa.noise_variance_                 # diagonal of Psi (q,)
    
    return z, Lambda, mu, Psi


def procrustes(Lambda_1, Lambda_2):
    M = Lambda_2.T @ Lambda_1
    U,S,Vh = np.linalg.svd(M)
    O_star = U @ Vh
    return Lambda_1, Lambda_2 @ O_star

def plot_3d(data, figName, figs_folder):
    elev = [0, 45, 90, 135, 180, 270]
    azim = [0, 45, 90, 135, 180, 270]

    cmap_type = 'bwr'
    # Make the figures and save them
    for elev_i in elev:
        for azim_i in azim:
            # Create a new figure
            fig = plt.figure(figsize=(12, 10))

            # 3D plot
            ax = fig.add_subplot(111, projection='3d')
            scatter = ax.scatter(data[0, :], data[1, :], data[2, :],
                                    alpha=1)
            ax.view_init(elev=elev_i, azim=azim_i)

            # Set labels and title
            ax.set_xlabel('PCA feature 1')
            ax.set_ylabel('PCA feature 2')
            ax.set_zlabel('PCA feature 3')
            ax.set_title(f'PCA visualization of naive neural data')

            # Adjust the layout
            plt.tight_layout()
            filename = f'{figName}_elev_{elev_i}_azim_{azim_i}.png'
            # Create the full file path
            file_path = os.path.join(figs_folder, filename)

            # Save the figure
            plt.savefig(file_path, dpi=100, bbox_inches='tight')
            plt.close()

def scree_plot(data, num_components, figName, figs_folder):
    centered = data - np.mean(data, axis=1, keepdims=True)
    print(np.mean(data, axis=1, keepdims=True).shape)
    normalized = centered / np.std(centered, axis=1, keepdims=True)
    print(np.std(centered, axis=1, keepdims=True).shape)
    pca_result, PCA_U, centering_mean, PCA_emb = perform_PCA(normalized.T, n_components=num_components) 
    plt.plot(np.arange(1,num_components+1), PCA_emb.explained_variance_, marker='o')

    file_path = os.path.join(figs_folder, figName)

    # Save the figure
    plt.savefig(file_path, dpi=100, bbox_inches='tight')
    plt.close()
    return PCA_emb.explained_variance_, pca_result

def dimensionality(explained_variances, tol=0.95):
    total = np.sum(explained_variances)
    threshold = total * tol
    running_total = 0
    for i in range(len(explained_variances)):
        running_total += explained_variances[i]
        if running_total > threshold:
            return i+1
        

def make_plotly_fig(pca_result,
                    filename,
                    opacity=1, marker_size = 0.5, cmap_type = 'RdBu', output_path = "./figs", pca_feature_nums=[0,1,2], xrange=[-20,20], yrange=[-20,20], zrange=[-20,20]):
    # cmap_type = 'YlOrRd' # sequential

    x = pca_result[:,pca_feature_nums[0]]
    y = pca_result[:, pca_feature_nums[1]]
    z = pca_result[:, pca_feature_nums[2]]
    color_array = list(range(pca_result.shape[1]))

    # Split the data into thirds
    n_points = pca_result.shape[1]
    
    traces = []
    traces.append(go.Scatter3d(
            x=x[:],
            y=y[:],
            z=z[:],
            mode='markers',
            opacity=opacity,
            marker=dict(
                size=marker_size,
                showscale=False
            ),
            showlegend=False,
            visible=True  # Only show first third initially
        ))

        # Create figure with all traces
    fig = go.Figure(data=traces)

    # Update layout with adjusted positions and add buttons
    fig.update_layout(
        title=dict(
            text=f'Visualization of Motor Cortex Data (mc-1)',
            y=0.95,
            x=0.5,
            xanchor='center',
            yanchor='top'
        ),
        scene=dict(
            xaxis_title=f'PCA feature {pca_feature_nums[0]+1}',
            yaxis_title=f'PCA feature {pca_feature_nums[1]+1}',
            zaxis_title=f'PCA feature {pca_feature_nums[2]+1}',
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=1.5, z=1.5)  # This controls the zoom level - smaller numbers = more zoomed in
            ),
            xaxis = dict(nticks=4, range=xrange,),
            yaxis = dict(nticks=4, range=yrange,),
            zaxis = dict(nticks=4, range=zrange,),
        ),
        width=1600,
        height=1200,
        margin=dict(r=100, t=100)
    )

    # Save and show the plot
    fig_name= f"{filename}.html"
    file_path = os.path.join(output_path, fig_name)
    fig.write_html(file_path)
    fig.show()