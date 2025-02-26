import numpy as np
from skimage import io, color
from scipy.ndimage import gaussian_filter
from lib.algorithms import *

def compute_transport_map(mu, v, w, image_shape, sigma, gamma=0.1):
    """
    Computes a transport map T(x) from a source histogram mu on the chrominance
    grid to the target (barycenter) using the scaling factors v and w.
    
    The grid is assumed to cover the domain [-128, 128]².
    """
    num_bins = image_shape[0]
    N = num_bins * num_bins
    bins = np.linspace(-128, 128, num_bins)
    X, Y = np.meshgrid(bins, bins)
    grid = np.stack([X.flatten(), Y.flatten()], axis=1)  # (N, 2)
    
    # Construct full Gaussian kernel H (for small histograms)
    diff = grid[:, None, :] - grid[None, :, :]  # shape (N, N, 2)
    dist2 = np.sum(diff**2, axis=2)
    H = np.exp(-dist2 / (2 * sigma**2))
    
    pi = (v[:, None] * H) * w[None, :]
    
    T = np.zeros_like(grid)
    for i in range(N):
        if mu[i] > 1e-10:
            T[i] = (pi[i, :].dot(grid)) / mu[i]
        else:
            T[i] = grid[i]
    T_map = T.reshape(num_bins, num_bins, 2)
    return T_map

def apply_transport_map_to_ab(ab_channel, T_map, bins):
    """
    Remaps a chrominance channel using the transport map.
    Here we use a simple nearest-neighbor approach.
    """
    H_img, W_img = ab_channel.shape
    num_bins = len(bins)
    idx = np.clip(((ab_channel + 128) / 256 * num_bins).astype(int), 0, num_bins-1)
    new_ab = np.zeros_like(ab_channel)
    for i in range(H_img):
        for j in range(W_img):
            bin_idx = idx[i, j]
            new_ab[i, j] = T_map[bin_idx, bin_idx, 0]  # Using channel 0 as an example
    return new_ab

def compute_histogram(ab, bins):
    """
    Computes a joint 2D histogram for the ab channels.
    """
    a_vals = ab[..., 0].flatten()
    b_vals = ab[..., 1].flatten()
    H_hist, _, _ = np.histogram2d(a_vals, b_vals, bins=[bins, bins])
    H_hist = H_hist / np.sum(H_hist)
    return H_hist.flatten()

def color_transfer(imageA_path, imageB_path, num_bins=64, sigma=2.0, tol=1e-6, max_iter=200):
    """
    Performs color transfer between images A and B.
    
    Steps:
      1. Convert images to Lab and extract ab channels.
      2. Compute joint histograms (as probability distributions) for ab.
      3. Compute the Wasserstein barycenter of the two histograms.
      4. Compute transport maps from each input histogram to the barycenter.
      5. Remap the ab channels of image A (for example) using its transport map.
      6. Recombine with the original L channel and convert back to RGB.
    
    Returns:
      Original image A, image B, and the color-transferred version of image A.
    """
    imageA = io.imread(imageA_path) / 255.0
    imageB = io.imread(imageB_path) / 255.0
    
    labA = color.rgb2lab(imageA)
    labB = color.rgb2lab(imageB)
    
    abA = labA[..., 1:]
    abB = labB[..., 1:]
    
    H_img, W_img, _ = abA.shape
    bins = np.linspace(-128, 128, num_bins)
    
    muA = compute_histogram(abA, bins)
    muB = compute_histogram(abB, bins)
    
    # Compute barycenter histogram with equal weights.
    mu_list = [muA, muB]
    alpha_list = [0.5, 0.5]
    image_shape_hist = (num_bins, num_bins)
    mu_bar, v_list, w_list = wasserstein_barycenter(mu_list, alpha_list, image_shape_hist, sigma, tol, max_iter, verbose=True)
    
    # Compute transport map for image A.
    T_mapA = compute_transport_map(muA, v_list[0], w_list[0], image_shape_hist, sigma)
    
    # Remap ab channels of image A using the transport map.
    new_abA_channel0 = apply_transport_map_to_ab(abA[..., 0], T_mapA, bins)
    new_abA_channel1 = apply_transport_map_to_ab(abA[..., 1], T_mapA, bins)
    
    new_labA = labA.copy()
    new_labA[..., 1] = new_abA_channel0
    new_labA[..., 2] = new_abA_channel1
    
    new_imageA = color.lab2rgb(new_labA)
    return imageA, imageB, new_imageA