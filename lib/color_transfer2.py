import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from skimage import io, color

##################################
# 1. AUX FUNCTIONS
##################################

def match_l_histogram(L_src, L_tgt):
    """
    Matches the distribution of L_src to that of L_tgt via a simple 1D histogram-matching.
    
    Parameters:
      L_src, L_tgt : (H×W) or flattened arrays of L-channel in [0..100] (typical for Lab).
    
    Returns:
      L_out : L_src matched to L_tgt's distribution, same shape as L_src.
    """
    shape = L_src.shape
    Ls = L_src.flatten()
    Lt = L_tgt.flatten()
    
    n = len(Ls)
    # Sort them
    Ls_sorted = np.sort(Ls)
    Lt_sorted = np.sort(Lt)
    # Get fraction rank for each L_src pixel
    ranks = np.searchsorted(Ls_sorted, Ls, side='left') / float(n)
    # Map fraction => L_tgt using Lt_sorted
    idx = np.clip((ranks*(n-1)).astype(int), 0, n-1)
    L_out = Lt_sorted[idx]
    return L_out.reshape(shape)

def entropy_of(mu, area_weights):
    """
    Computes the discrete 'entropy' of mu with area weights a:
      H(mu) = -∑ a_i mu[i] log(mu[i])
    We assume ∑ a_i mu[i] = 1, i.e. mu is a probability distribution.
    """
    eps = 1e-16
    return -np.sum(area_weights * mu * np.log(np.maximum(mu, eps)))

def entropic_sharpen(mu, a, H0, tol=1e-6, max_iter=50):
    """
    Sharpens distribution mu^β so that its entropy does not exceed H0.
    
    We solve for β >= 1 s.t. Entropy(normalized(mu^β)) = H0 (or less if already <=H0).
    This bisection approach is from the 'Entropic Sharpening' section in the article.
    """
    current_H = entropy_of(mu, a)
    if current_H <= H0:
        # Already below or equal => no action
        return mu
    
    # We'll do a bisection search for β in [1..some large].
    def F(beta):
        # evaluate entropy of normalized( mu^beta )
        m = mu**beta
        m /= (a * m).sum()  # normalize => sum_i a[i]*m[i] = 1
        return entropy_of(m, a) - H0
    
    beta_low = 1.0
    beta_high = 2.0
    
    # Expand beta_high until F(beta_high) < 0 or we hit limit
    while beta_high < 1e7:
        val_high = F(beta_high)
        if val_high < 0:
            break
        beta_high *= 2.0
    
    # Bisection
    for _ in range(max_iter):
        beta_mid = 0.5*(beta_low + beta_high)
        val_mid = F(beta_mid)
        if abs(val_mid) < tol:
            break
        if val_mid > 0:
            # we need bigger beta
            beta_low = beta_mid
        else:
            beta_high = beta_mid
    
    # Final sharpen
    beta = 0.5*(beta_low + beta_high)
    m = mu**beta
    m /= (a * m).sum()
    return m


##################################
# 2. COLOR-HISTOGRAM PIPELINE
##################################

def make_ab_histogram(img_lab, n_bins=64):
    """
    Build 2D histogram of ab-values in [-128,128]^2, returning
      (hist, bin_centers, (n_bins,n_bins))
    The histogram is flattened (n_bins^2) and normalized to sum=1.
    """
    a_vals = img_lab[...,1].ravel()
    b_vals = img_lab[...,2].ravel()
    
    a_min,a_max = -128,128
    b_min,b_max = -128,128
    
    a_bins = np.linspace(a_min, a_max, n_bins+1)
    b_bins = np.linspace(b_min, b_max, n_bins+1)
    
    hist2d, _, _ = np.histogram2d(a_vals, b_vals, bins=[a_bins,b_bins])
    hist2d = hist2d / np.sum(hist2d)  # normalize
    
    # Bin centers
    a_centers = 0.5*(a_bins[:-1]+a_bins[1:])
    b_centers = 0.5*(b_bins[:-1]+b_bins[1:])
    Agrid,Bgrid = np.meshgrid(a_centers,b_centers,indexing='ij')
    bin_centers = np.stack([Agrid,Bgrid],axis=-1).reshape(-1,2)
    
    return hist2d.ravel(), bin_centers, (n_bins,n_bins)

def gaussian_conv(vec, shape2D, sigma):
    """
    Apply 2D Gaussian convolution to a flattened (n_bins^2) array.
    """
    arr_2d = vec.reshape(shape2D)
    arr_filt = gaussian_filter(arr_2d, sigma=sigma, mode='constant')
    return arr_filt.ravel()

def sinkhorn_barycenter_2d(hist1, hist2, alpha1, alpha2,
                           shape2D, sigma, max_iter=200, tol=1e-7,
                           sharpen_entropy=None):
    """
    Special case of the article's Algorithm 2 for k=2 distributions
    with weights alpha1, alpha2. We store only v1,w1,v2,w2,
    and the barycenter mu that their row marginals share.

    If sharpen_entropy is not None, we apply an entropy constraint each iteration.
    """
    n = hist1.size
    # area weights for each bin (uniform in color histogram)
    a = np.full(n, 1.0/n)
    
    # Initialize v1,w1,v2,w2
    v1 = np.ones(n)
    w1 = np.ones(n)
    v2 = np.ones(n)
    w2 = np.ones(n)
    
    # Barycenter mu => uniform start
    mu = np.full(n, 1.0/n)
    
    def Hconv(x):
        # heat kernel ~ Gaussian conv
        return gaussian_conv(x, shape2D, sigma)
    
    for it in range(max_iter):
        mu_old = mu.copy()
        
        # Project onto constraints fixing the column marginals = hist1, hist2
        Hv1 = Hconv(v1*mu)
        Hv1 = np.maximum(Hv1,1e-16)
        w1  = hist1 / Hv1
        
        Hv2 = Hconv(v2*mu)
        Hv2 = np.maximum(Hv2,1e-16)
        w2  = hist2 / Hv2
        
        # d1, d2 for next steps
        d1 = v1 * Hconv(w1*mu)
        d2 = v2 * Hconv(w2*mu)
        
        # Weighted geometric mean => barycenter
        mu = (d1**alpha1)*(d2**alpha2)
        s  = mu.sum()
        if s < 1e-16:
            mu = np.full(n, 1.0/n)
        else:
            mu /= s
        
        # Entropic sharpening step (optional)
        if sharpen_entropy is not None:
            mu = entropic_sharpen(mu, a, sharpen_entropy)
        
        # Project onto constraints that row marginals match => pi_1, pi_2 share same row=mu
        d1 = np.maximum(d1,1e-16)
        v1 = v1*(mu/d1)
        d2 = np.maximum(d2,1e-16)
        v2 = v2*(mu/d2)
        
        err = np.linalg.norm(mu - mu_old,1)
        if err < tol:
            break
    
    return mu, (v1,w1), (v2,w2)

def build_transport_map_2d(v, w, bin_centers, sigma):
    """
    pi=diag(v)*H*diag(w). T(x) = sum_y pi(x,y)*y / sum_y pi(x,y).
    We'll do naive O(n^2).
    """
    n = bin_centers.shape[0]
    Tmap = np.zeros((n,2))
    
    def gauss_weight(i,j):
        diff = bin_centers[i] - bin_centers[j]
        return np.exp(-0.5*(diff[0]**2 + diff[1]**2)/(sigma*sigma))
    
    for x in range(n):
        denom = 1e-16
        num = np.zeros(2)
        for y in range(n):
            val = v[x]*gauss_weight(x,y)*w[y]
            denom += val
            num   += val * bin_centers[y]
        Tmap[x] = num / denom
    return Tmap

def apply_transport_map_2d(img_ab, bin_centers, Tmap):
    """
    For each pixel's ab, find nearest bin, replace by Tmap[that bin].
    """
    H,W,_ = img_ab.shape
    out_ab = np.zeros_like(img_ab)
    tree = cKDTree(bin_centers)
    
    ab_flat = img_ab.reshape(-1,2)
    _, idx = tree.query(ab_flat)
    out_flat = Tmap[idx]
    
    out_ab = out_flat.reshape(H,W,2)
    return out_ab

##################################
# 3. WRAPPER FOR COLOR TRANSFER
##################################

def color_transfer_barycenter(
    src_rgb, tgt_rgb, t=0.5, n_bins=64, sigma=5.0,
    match_l=False,  # if True => 1D histogram match L from src->tgt
    sharpen_entropy=None
    ):
    """
    Creates a barycenter for ab-channels (as in the paper) between src and tgt.
    If t=0 => the barycenter = src's distribution; if t=1 => = tgt's distribution.
    Intermediate t => partial blending.
    
    - match_l: if True, do 1D histogram matching for L from src to tgt.
    - sharpen_entropy: if set (e.g. 4.0), we enforce that final barycenter's
      entropy doesn't exceed this value at each iteration (entropic sharpening).
    
    Returns:
      out_src_rgb : corrected version of src => barycenter
      out_tgt_rgb : corrected version of tgt => barycenter
      mu_bar      : the barycenter histogram (1D array)
    """
    # Convert to float
    if src_rgb.dtype.kind in ['u','i']:
        src_rgb = src_rgb.astype(np.float32)/255.
    if tgt_rgb.dtype.kind in ['u','i']:
        tgt_rgb = tgt_rgb.astype(np.float32)/255.
    
    lab_src = color.rgb2lab(src_rgb)
    lab_tgt = color.rgb2lab(tgt_rgb)
    
    # Optionally match L from src to tgt
    if match_l:
        # Replace the L channel of src by histogram-matched version
        L_matched = match_l_histogram(lab_src[...,0], lab_tgt[...,0])
        lab_src[...,0] = L_matched
    
    # 2) Build ab-hist for each
    hist_src, bc_src, shape2D = make_ab_histogram(lab_src, n_bins=n_bins)
    hist_tgt, bc_tgt, _       = make_ab_histogram(lab_tgt, n_bins=n_bins)
    
    # 3) Solve barycenter problem with alpha=(1-t,t)
    alpha1, alpha2 = (1.0 - t), t
    mu_bar, (v1,w1), (v2,w2) = sinkhorn_barycenter_2d(
        hist_src, hist_tgt, alpha1, alpha2,
        shape2D=shape2D, sigma=sigma,
        max_iter=200, tol=1e-7,
        sharpen_entropy=sharpen_entropy
    )
    
    # 4) Build transport maps: T1 for src->mu_bar, T2 for tgt->mu_bar
    T1 = build_transport_map_2d(v1, w1, bc_src, sigma)
    T2 = build_transport_map_2d(v2, w2, bc_tgt, sigma)
    
    # 5) Apply T1 => out_src
    ab_src = lab_src[...,1:3]
    out_src_ab = apply_transport_map_2d(ab_src, bc_src, T1)
    out_src_lab = np.dstack([lab_src[...,0], out_src_ab])
    out_src_rgb = color.lab2rgb(out_src_lab)
    
    # 6) Apply T2 => out_tgt
    ab_tgt = lab_tgt[...,1:3]
    out_tgt_ab = apply_transport_map_2d(ab_tgt, bc_tgt, T2)
    out_tgt_lab = np.dstack([lab_tgt[...,0], out_tgt_ab])
    out_tgt_rgb = color.lab2rgb(out_tgt_lab)
    
    return out_src_rgb, out_tgt_rgb, mu_bar
