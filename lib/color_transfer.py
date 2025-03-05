import numpy as np
from skimage import io, color
from scipy.spatial import cKDTree
# from lib.algorithms import wasserstein_barycenter, wasserstein_barycenter_stable
from lib.algo import convolutional_wasserstein_barycenter
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import ot

def plot_2d_histogram(hist):
    
    plt.imshow(hist, interpolation='nearest', cmap='viridis', origin='lower')
    plt.colorbar()
    plt.title('2D Histogram')
    plt.xlabel('A axis')
    plt.ylabel('B axis')
    plt.show()

def compute_gamma(mu_list, percentile=0.1):
    """
    Computes gamma as a small percentage of the median transport cost
    by estimating pairwise squared distances between distributions.

    Parameters:
      mu_list   : List of probability distributions (each flattened 1D).
      percentile: Percentage of the median transport cost (default 1%).

    Returns:
      gamma     : Optimal entropy regularization parameter.
    """
    # Get bin positions in a 2D grid
    n_bins = int(np.sqrt(len(mu_list[0])))  # Assume square grid
    x, y = np.meshgrid(np.arange(n_bins), np.arange(n_bins))
    positions = np.vstack([x.ravel(), y.ravel()]).T  # Flattened 2D positions

    # Compute pairwise squared distances
    pairwise_dists = cdist(positions, positions, metric='sqeuclidean')

    # Compute transport cost using mass-weighted distances
    transport_costs = []
    for mu in mu_list:
        for mu_target in mu_list:
            cost = np.sum(mu[:, None] * pairwise_dists * mu_target[None, :])  # Transport cost
            transport_costs.append(cost)

    # Compute the median cost
    median_cost = np.median(transport_costs)

    # Set gamma as 1% of the median cost
    gamma = (percentile / 100) * median_cost
    return gamma

def match_l_channel(L_src, L_tgt):
    """
    Simple 1D histogram matching for the L channel in Lab space.
    L_src, L_tgt: arrays of shape (H,W) or flattened, each in [0..100].
    Returns L_out with the same shape as L_src, matched to L_tgt distribution.
    """
    orig_shape = L_src.shape
    Ls = L_src.ravel()
    Lt = L_tgt.ravel()
    
    # Sort
    Ls_sorted = np.sort(Ls)
    Lt_sorted = np.sort(Lt)
    
    n = len(Ls)
    # Ranks
    ranks = np.searchsorted(Ls_sorted, Ls, side='left') / float(n)
    # Map fraction => L_tgt
    idx = np.clip((ranks*(n-1)).astype(int), 0, n-1)
    L_out = Lt_sorted[idx]
    return L_out.reshape(orig_shape)

def make_ab_histogram(img_lab, n_bins=64):
    """
    Discretize the ab-plane in [-128..128]^2 into n_bins x n_bins bins
    and build a normalized 2D histogram => flatten to shape (n_bins^2).
    
    Returns:
      hist_flat:  1D array of length (n_bins^2) summing to 1
      bin_centers: array of shape (n_bins^2, 2) listing ab-centers
      shape2D:     (n_bins, n_bins) for easy reshaping
    """
    a_vals = img_lab[...,1].ravel()
    b_vals = img_lab[...,2].ravel()
    
    a_min,a_max = -128,128
    b_min,b_max = -128,128
    a_bins = np.linspace(a_min, a_max, n_bins+1)
    b_bins = np.linspace(b_min, b_max, n_bins+1)
    
    H2d, _, _ = np.histogram2d(a_vals, b_vals, bins=[a_bins,b_bins])
    H2d = H2d / np.sum(H2d)
    
    # Build bin-centers
    a_centers = 0.5*(a_bins[:-1] + a_bins[1:])
    b_centers = 0.5*(b_bins[:-1] + b_bins[1:])
    Agrid,Bgrid = np.meshgrid(a_centers, b_centers, indexing='ij')
    bin_centers = np.stack([Agrid,Bgrid], axis=-1).reshape(-1,2)  # shape(n_bins^2,2)
    
    return H2d.ravel(), bin_centers, (n_bins,n_bins)


def compute_transport_map(source_bins_grid, target_bins_grid, V, W, gamma):
    """
    Compute the transport map using entropic optimal transport.

    Parameters:
    -----------
    source_bins_grid : np.ndarray
        Grid of (a, b) values for the source histogram.
    target_bins_grid : np.ndarray
        Grid of (a, b) values for the target histogram.
    V : np.ndarray
        Left scaling factor from Sinkhorn iterations.
    W : np.ndarray
        Right scaling factor from Sinkhorn iterations.
    gamma : float
        Entropic regularization parameter.

    Returns:
    --------
    transport_map : np.ndarray
        The computed transport mapping from source to target.
    """
    # Compute cost matrix M (Squared Euclidean Distance between histogram bins)
    M = cdist(source_bins_grid, target_bins_grid, metric='sqeuclidean')
    M = M / M.max()  # Normalize

    # Compute transport kernel K using entropic scaling
    K = np.exp(-M / gamma)
    V = V.ravel()
    W = W.ravel()
    
    print("K shape:", K.shape)
    print("V shape:", V.shape)
    print("W shape:", W.shape)

    # Compute transport plan
    transport_plan = np.einsum("ij,i,j->ij", K, V, W)

    # Normalize transport plan row-wise
    row_sums = transport_plan.sum(axis=1, keepdims=True)
    normalized_plan = transport_plan / (row_sums + 1e-10)

    # Compute the transport map via barycentric projection
    transport_map = normalized_plan @ target_bins_grid  

    return transport_map

# def apply_transport_map(source_lab, transport_map, source_bins, bins):
#     transformed_lab = source_lab.copy().astype(np.float32)
#     for i in range(transformed_lab.shape[0]):
#         for j in range(transformed_lab.shape[1]):
#             a_idx = np.argmin(np.abs(source_bins[0] - transformed_lab[i, j, 1]))
#             b_idx = np.argmin(np.abs(source_bins[1] - transformed_lab[i, j, 2]))
#             index = a_idx * bins + b_idx  
#             transformed_lab[i, j, 1:] = transport_map[index]
#     return transformed_lab

def apply_transport_map_knn(img_ab, bin_centers, Tmap, k=4, eps=1e-8):
    """
    Applies the transport map using k-nearest neighbor interpolation.
    Computes an inverse-distance weighted average of Tmap values.

    Inputs:
    - img_ab: (H, W, 2) ab color channels of input image.
    - bin_centers: (M, 2) grid of color bins in the ab space.
    - Tmap: (M, 2) new color mapping for each bin.
    - k: Number of nearest neighbors to consider.
    - eps: Small constant to prevent division by zero.

    Returns:
    - out_ab: (H, W, 2) transformed image in ab space.
    """
    H, W, _ = img_ab.shape
    N = H * W  # Total number of pixels

    # Flatten the image to shape (N, 2)
    ab_flat = img_ab.reshape(N, 2)

    # Build KD-Tree and find k-nearest neighbors for each pixel
    tree = cKDTree(bin_centers)
    dist, idx = tree.query(ab_flat, k=k)  # dist: (N, k), idx: (N, k)

    # Compute inverse-distance weights
    w = 1.0 / (dist + eps)  # shape (N, k)
    w /= w.sum(axis=1, keepdims=True)  # Normalize weights (sum_j w_j = 1)

    # Retrieve colors from Tmap using nearest neighbor indices
    tcolors = Tmap[idx]  # shape (N, k, 2)

    # Compute weighted sum: sum_j [ w_j * Tmap[idx_j] ]
    out_flat = np.einsum("nk,nkj->nj", w, tcolors)  # Efficient weighted sum

    # Reshape back to (H, W, 2)
    out_ab = out_flat.reshape(H, W, 2)
    return out_ab


def apply_transport_map(img_ab, bin_centers, Tmap):
    """
    For each pixel's ab, find the nearest bin in bin_centers, replace ab by Tmap[bin].
    Simple nearest-neighbor. One could do bilinear interpolation for smoother results.
    """
    H,W,_ = img_ab.shape
    out_ab = np.zeros_like(img_ab)
    for i in range(10):  # Print first 10 assigned values
        print(f"Pixel {i}: Original AB {img_ab.reshape(-1,2)[i]}, Mapped AB {Tmap[i]}")
    
    # Build a KD-Tree over bin_centers
    tree = cKDTree(bin_centers)
    
    ab_flat = img_ab.reshape(-1, 2)
    _, idx = tree.query(ab_flat)  # nearest bin
    out_flat = Tmap[idx]
    out_ab = out_flat.reshape(H,W,2)
    # out_ab = apply_transport_map_knn(img_ab, bin_centers, Tmap, k=4)
    
    return out_ab

def color_transfer(
    src_rgb,
    tgt_rgb,
    t=0.5,
    n_bins=64,
    gamma=None,
    match_l=False,
    sharpen_entropy=None,
    max_iter=200,
    tol=1e-6
):
    """
    Color-transfer style barycenter approach, reusing 'wasserstein_barycenter'
    from algorithms.py. Also optionally:
      - match_l => do 1D histogram matching for L
      - sharpen_entropy => if set, do a final post-hoc entropic sharpening 
        (not done inside each iteration, just once after computing the barycenter).
    
    Returns: (out_src_rgb, out_tgt_rgb, mu_bar)
    """
    # Convert images to float if needed
    if src_rgb.dtype.kind in ['u','i']:
        src_rgb = src_rgb.astype(np.float32)/255.
    if tgt_rgb.dtype.kind in ['u','i']:
        tgt_rgb = tgt_rgb.astype(np.float32)/255.
    
    # Convert to Lab
    lab_src = color.rgb2lab(src_rgb)
    lab_tgt = color.rgb2lab(tgt_rgb)
    
    # Possibly match the L channel from src to tgt
    if match_l:
        L_src_matched = match_l_channel(lab_src[...,0], lab_tgt[...,0])
        lab_src[...,0] = L_src_matched
    
    # Build ab histograms
    hist_src, bc_src, shape2D = make_ab_histogram(lab_src, n_bins=n_bins)
    hist_tgt, bc_tgt, _       = make_ab_histogram(lab_tgt, n_bins=n_bins)
    mu_list = np.array([hist_src, hist_tgt])
    # Reshape mu_list to (2, n_bins, n_bins)
    mu_list = mu_list.reshape(2, n_bins, n_bins)
    
    print("Source AB range:",
      lab_src[...,1].min(), lab_src[...,1].max(),
      lab_src[...,2].min(), lab_src[...,2].max())

    print("Target AB range:",
      lab_tgt[...,1].min(), lab_tgt[...,1].max(),
      lab_tgt[...,2].min(), lab_tgt[...,2].max())
    
    print("Nonzero bins (source):", np.count_nonzero(hist_src))
    print("Nonzero bins (target):", np.count_nonzero(hist_tgt))
    
    # Barycenter with alpha=[(1-t), t]
    alpha_list = np.array([1.0 - t, t])
    
    # gamma = compute_gamma(mu_list, 1)
    # print(f"gamma is {gamma}")
    
    # Use your existing 'wasserstein_barycenter' function from algorithms.py
    mu_bar, v_list, w_list, K = convolutional_wasserstein_barycenter(
        distributions=mu_list,
        weights=alpha_list,
        gamma=gamma,
        verbose=True,
        max_iterations=max_iter,
        stop_threshold=tol,
    )
    
    plot_2d_histogram(hist_src.reshape(shape2D))
    plot_2d_histogram(hist_tgt.reshape(shape2D))
    plot_2d_histogram(mu_bar.reshape(shape2D))
    
    print("Sum of mu_bar:", np.sum(mu_bar))
    
    # v_list[0], w_list[0] => plan for distribution #0 => the barycenter
    # v_list[1], w_list[1] => plan for distribution #1 => the barycenter
    # We'll build Tmap for each distribution => barycenter
    #   T_i(x) = sum_y pi_i(x,y)*y / sum_y pi_i(x,y)
    # where pi_i = diag(v_i)*K*diag(w_i). 
    
    # T_src = build_transport_map(v_list[0], w_list[0], bc_src, sigma)
    # T_tgt = build_transport_map(v_list[1], w_list[1], bc_tgt, sigma)
    # T_src = build_transport_map(hist_src, v_list[0].ravel(), w_list[0].ravel(), bc_src, gamma, K)
    # T_tgt = build_transport_map(hist_tgt, v_list[1].ravel(), w_list[1].ravel(), bc_tgt, gamma, K)
    T_src = compute_transport_map(bc_src, bc_tgt, v_list[0], w_list[0], gamma)
    T_tgt = compute_transport_map(bc_tgt, bc_src, v_list[1], w_list[1], gamma)
    hist_src += 1e-8  # Add a tiny mass to avoid zero bins
    hist_tgt += 1e-8
    mu_bar += 1e-8  # Ensure barycenter also has mass
    hist_src /= np.sum(hist_src)
    hist_tgt /= np.sum(hist_tgt)

    cost_matrix = np.clip(cdist(bc_src, bc_tgt, metric="sqeuclidean"), 1e-6, 100)  # Limit extreme values
    # transport_src = ot.sinkhorn(hist_src, mu_bar.flatten(), cost_matrix, reg=0.1)
    # transport_tgt = ot.sinkhorn(hist_tgt, mu_bar.flatten(), cost_matrix, reg=0.1)
    
    # T_src = transport_src @ bc_tgt
    # T_tgt = transport_tgt @ bc_src
    
    print("Finished Building Transport Maps")
    
    # Apply Tmaps
    ab_src = lab_src[...,1:3]
    out_src_ab = apply_transport_map(ab_src, bc_src, T_src)
    print("Applied Tmap on source")
    out_src_lab = np.dstack([lab_src[...,0], out_src_ab])
    print("Min/Max of final Lab image: L in [", out_src_lab[...,0].min(), 
    ",", out_src_lab[...,0].max(), "]",
    " a in [", out_src_lab[...,1].min(), ",", out_src_lab[...,1].max(), "]",
    " b in [", out_src_lab[...,2].min(), ",", out_src_lab[...,2].max(), "]")
    out_src_rgb = color.lab2rgb(out_src_lab)
    
    ab_tgt = lab_tgt[...,1:3]
    out_tgt_ab = apply_transport_map(ab_tgt, bc_tgt, T_tgt)
    print("Applied Tmap on target")
    out_tgt_lab = np.dstack([lab_tgt[...,0], out_tgt_ab])
    print("Min/Max of final Lab image: L in [", out_tgt_lab[...,0].min(), 
    ",", out_src_lab[...,0].max(), "]",
    " a in [", out_tgt_lab[...,1].min(), ",", out_tgt_lab[...,1].max(), "]",
    " b in [", out_tgt_lab[...,2].min(), ",", out_tgt_lab[...,2].max(), "]")
    out_tgt_rgb = color.lab2rgb(out_tgt_lab)
    
    return out_src_rgb, out_tgt_rgb, mu_bar
