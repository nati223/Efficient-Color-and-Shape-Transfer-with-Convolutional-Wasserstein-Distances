import numpy as np
from skimage import io, color
from scipy.spatial import cKDTree
from lib.algorithms import wasserstein_barycente


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

def build_transport_map(v, w, bin_centers, sigma):
    """
    Recovers T(x) = sum_y pi(x,y)*y / sum_y pi(x,y),
    where pi = diag(v)*K*diag(w), K ~ exp(-||x-y||^2/(2sigma^2)).
    
    For each x, we do a naive O(n^2) approach.  
    If n_bins is large, multi-scale or convolution-based approaches are recommended.
    """
    n = bin_centers.shape[0]
    Tmap = np.zeros((n,2), dtype=np.float32)
    
    def kernel_weight(i, j):
        diff = bin_centers[i] - bin_centers[j]
        return np.exp(-0.5 * np.dot(diff,diff) / (sigma*sigma))
    
    for x in range(n):
        denom = 1e-16
        num   = np.zeros(2, dtype=np.float32)
        for y in range(n):
            val = v[x] * kernel_weight(x,y) * w[y]
            denom += val
            num   += val * bin_centers[y]
        Tmap[x] = num / denom
    return Tmap


def apply_transport_map_knn(
    img_ab,      # shape (H,W,2) in ab space
    bin_centers, # shape (M,2) for M=n_bins^2
    Tmap,        # shape (M,2), the new color for each bin
    k=4,         # how many neighbors to interpolate
    eps=1e-8
):
    """
    For each pixel's ab, find k nearest bins, then do an inverse-distance
    weighted average of Tmap for those bins.
    """
    H,W,_ = img_ab.shape
    out_ab = np.zeros_like(img_ab)

    tree = cKDTree(bin_centers)
    
    ab_flat = img_ab.reshape(-1, 2)  # shape (N,2), N=H*W
    dist, idx = tree.query(ab_flat, k=k)  
    # => dist.shape (N,k), idx.shape (N,k)
    
    # Weighted average
    # w_j = 1/(dist_j + eps)
    # newColor = sum_j [ w_j * Tmap[idx_j] ] / sum_j w_j
    N = ab_flat.shape[0]
    out_flat = np.zeros((N, 2), dtype=np.float32)
    
    for i in range(N):
        d = dist[i]  # shape (k,)
        neighbors = idx[i]  # shape (k,)
        w = 1.0 / (d + eps)  # inverse-dist weighting
        wsum = np.sum(w)
        
        # Weighted sum of Tmap
        tcolors = Tmap[neighbors]  # shape (k,2)
        out_flat[i] = np.sum(tcolors * w[:,None], axis=0) / wsum
    
    out_ab = out_flat.reshape(H,W,2)
    return out_ab


def apply_transport_map(img_ab, bin_centers, Tmap):
    """
    For each pixel's ab, find the nearest bin in bin_centers, replace ab by Tmap[bin].
    Simple nearest-neighbor. One could do bilinear interpolation for smoother results.
    """
    H,W,_ = img_ab.shape
    out_ab = np.zeros_like(img_ab)
    
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
    sigma=5.0,
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
    
    print("Source AB range:",
      lab_src[...,1].min(), lab_src[...,1].max(),
      lab_src[...,2].min(), lab_src[...,2].max())

    print("Target AB range:",
      lab_tgt[...,1].min(), lab_tgt[...,1].max(),
      lab_tgt[...,2].min(), lab_tgt[...,2].max())
    
    print("Nonzero bins (source):", np.count_nonzero(hist_src))
    print("Nonzero bins (target):", np.count_nonzero(hist_tgt))
    
    # Barycenter with alpha=[(1-t), t]
    alpha_list = [1.0 - t, t]
    
    # Use your existing 'wasserstein_barycenter' function from algorithms.py
    mu_bar, v_list, w_list = wasserstein_barycenter(
        mu_list=[hist_src, hist_tgt],
        alpha_list=alpha_list,
        image_shape=shape2D,
        sigma=sigma,
        max_iter=max_iter,
        tol=tol,
        sharpen_entropy=sharpen_entropy,  # <= pass it here
        verbose=True
    )
    
    # print("Nonzero bins in barycenter:", np.count_nonzero(mu_bar))
    nz_indices = np.where(mu_bar > 0)[0]   # the indices of nonzero bins
    nz_values = mu_bar[nz_indices]         # the corresponding nonzero values

    # print("Nonzero indices:", nz_indices)
    print("Nonzero values:", nz_values)
    print("Sum of mu_bar:", np.sum(mu_bar))
    
    # v_list[0], w_list[0] => plan for distribution #0 => the barycenter
    # v_list[1], w_list[1] => plan for distribution #1 => the barycenter
    # We'll build Tmap for each distribution => barycenter
    #   T_i(x) = sum_y pi_i(x,y)*y / sum_y pi_i(x,y)
    # where pi_i = diag(v_i)*K*diag(w_i). 
    # 
    # By default, 'wasserstein_barycenter' sets the row dimension to the barycenter and the col dimension to mu_i.
    # So if you want row=mu_i, col=barycenter, you might need to 'swap' v_i, w_i. 
    # Or just keep it consistent with the code that produced them. 
    #
    # We'll try the same pattern as your custom sinkhorn code: 
    # "build_transport_map(v_i, w_i, bc_src, sigma)" => T from src to barycenter. 
    # If the barycenter code is reversed, you might do build_transport_map(w_i, v_i, bc_src, sigma).
    #
    # Quick test: if the result looks reversed, just swap arguments. :-)
    
    # T_src = build_transport_map(v_list[0], w_list[0], bc_src, sigma)
    # T_tgt = build_transport_map(v_list[1], w_list[1], bc_tgt, sigma)
    T_src = build_transport_map(w_list[0], v_list[0], bc_src, sigma)
    T_tgt = build_transport_map(w_list[1], v_list[1], bc_tgt, sigma)
    
    # Apply Tmaps
    ab_src = lab_src[...,1:3]
    out_src_ab = apply_transport_map(ab_src, bc_src, T_src)
    out_src_lab = np.dstack([lab_src[...,0], out_src_ab])
    print("Min/Max of final Lab image: L in [", out_src_lab[...,0].min(), 
    ",", out_src_lab[...,0].max(), "]",
    " a in [", out_src_lab[...,1].min(), ",", out_src_lab[...,1].max(), "]",
    " b in [", out_src_lab[...,2].min(), ",", out_src_lab[...,2].max(), "]")
    out_src_rgb = color.lab2rgb(out_src_lab)
    
    ab_tgt = lab_tgt[...,1:3]
    out_tgt_ab = apply_transport_map(ab_tgt, bc_tgt, T_tgt)
    out_tgt_lab = np.dstack([lab_tgt[...,0], out_tgt_ab])
    print("Min/Max of final Lab image: L in [", out_tgt_lab[...,0].min(), 
    ",", out_src_lab[...,0].max(), "]",
    " a in [", out_tgt_lab[...,1].min(), ",", out_tgt_lab[...,1].max(), "]",
    " b in [", out_tgt_lab[...,2].min(), ",", out_tgt_lab[...,2].max(), "]")
    out_tgt_rgb = color.lab2rgb(out_tgt_lab)
    
    return out_src_rgb, out_tgt_rgb, mu_bar
