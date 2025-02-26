import numpy as np
from scipy.ndimage import gaussian_filter

def heat_kernel_convolution(f, sigma, image_shape):
    """
    Applies the heat kernel (approximated by a Gaussian filter) to vector f.
    
    Parameters:
      f           : 1D numpy array representing a function on the domain.
      sigma       : Standard deviation for the Gaussian (controls diffusion).
      image_shape : Tuple indicating the dimensions of the image (e.g., (128, 128)).
    
    Returns:
      A 1D numpy array (flattened) representing the convolution result.
    """
    f_img = f.reshape(image_shape)
    conv_img = gaussian_filter(f_img, sigma=sigma)
    return conv_img.flatten()

def sinkhorn(mu0, mu1, image_shape, sigma, tol=1e-6, max_iter=1000):
    """
    Implements the Sinkhorn iterations for the convolutional Wasserstein distance.
    
    Parameters:
      mu0, mu1    : 1D numpy arrays representing the source and target probability distributions.
      image_shape : Tuple of ints (e.g., (128, 128)).
      sigma       : Standard deviation for the Gaussian convolution (heat kernel approximation).
      tol         : Convergence tolerance.
      max_iter    : Maximum number of iterations.
    
    Returns:
      v, w        : Scaling vectors such that the transport plan is given by diag(v) @ H_t @ diag(w).
    """
    n = mu0.size
    # For images, area weights are typically uniform.
    a = np.full_like(mu0, 1.0 / n)
    
    # Initialize scaling vectors to ones.
    v = np.ones_like(mu0)
    w = np.ones_like(mu1)
    
    for i in range(max_iter):
        v_prev = v.copy()
        
        # Update v: divide mu0 by the convolution of (w * a)
        Hv_w = heat_kernel_convolution(w * a, sigma, image_shape)
        # Avoid division by zero.
        Hv_w[Hv_w == 0] = 1e-16
        v = mu0 / Hv_w
        
        # Update w: divide mu1 by the convolution of (v * a)
        Hv_v = heat_kernel_convolution(v * a, sigma, image_shape)
        Hv_v[Hv_v == 0] = 1e-16
        w = mu1 / Hv_v
        
        # Check for convergence (using L1 norm on the change in v)
        if np.linalg.norm(v - v_prev, 1) < tol:
            print(f"Sinkhorn converged in {i+1} iterations.")
            break
    else:
        print("Maximum iterations reached without full convergence.")
    
    return v, w, a

def compute_convolutional_distance(mu0, mu1, v, w, a, gamma):
    """
    Computes the regularized Wasserstein distance (squared) based on the scaling factors.
    
    Uses the formula:
      W² ≈ gamma * aᵀ[(mu0 * ln(v)) + (mu1 * ln(w))]
    
    Parameters:
      mu0, mu1 : Source and target distributions (1D numpy arrays).
      v, w     : Scaling vectors from the Sinkhorn iterations.
      a        : Area weights.
      gamma    : Regularization parameter.
      
    Returns:
      The approximate regularized Wasserstein distance (squared).
    """
    # To avoid log(0), use a small constant.
    log_v = np.log(np.maximum(v, 1e-16))
    log_w = np.log(np.maximum(w, 1e-16))
    return gamma * np.dot(a, mu0 * log_v + mu1 * log_w)

def heat_kernel_convolution(f, sigma, image_shape):
    """
    Applies the heat kernel (approximated by a Gaussian filter)
    to vector f, reshaping it to an image of shape image_shape.
    
    Parameters:
      f           : 1D numpy array.
      sigma       : Standard deviation for the Gaussian filter.
      image_shape : Tuple (height, width) of the image.
    
    Returns:
      Flattened array of the convolution result.
    """
    f_img = f.reshape(image_shape)
    conv_img = gaussian_filter(f_img, sigma=sigma)
    return conv_img.flatten()

def wasserstein_barycenter(mu_list, alpha_list, image_shape, sigma,
                           tol=1e-6, max_iter=200, 
                           sharpen_entropy=None,  # <= new param
                           verbose=True):
    """
    Computes the entropic Wasserstein barycenter of the distributions in mu_list,
    with weights alpha_list (which should sum to 1).  We approximate the cost
    using a heat-kernel (Gaussian) with std= sigma, discretized as image_shape.
    
    If sharpen_entropy is not None (a float), we clamp the barycenter's entropy
    at each iteration so that H(mu) ≤ sharpen_entropy.
    """
    # We assume all mu_list[i] have the same length n = image_shape[0]*image_shape[1]
    n = mu_list[0].size
    k = len(mu_list)
    # area weights a for each "pixel/bin" => uniform
    a = np.full(n, 1.0/n)

    # Build v_i,w_i = 1, for each i
    v_list = [np.ones(n) for _ in range(k)]
    w_list = [np.ones(n) for _ in range(k)]
    
    # Start barycenter mu_bar as uniform
    mu_bar = np.full(n, 1.0/n)
 
    # main iteration
    for it in range(max_iter):
        mu_bar_old = mu_bar.copy()
        
        # Projection onto constraints fixing the 'column marginal' = mu_list[i]
        # => update w_i
        d_list = []
        for i in range(k):
            # w_i <- mu_list[i] / heat_conv(v_i * mu_bar)
            Hv = heat_kernel_convolution(v_list[i] * mu_bar, sigma, image_shape)
            Hv = np.maximum(Hv, 1e-16)
            w_list[i] = mu_list[i] / Hv
        
        # Now gather d_i = v_i * heat_conv(w_i * mu_bar)
        for i in range(k):
            conv_i = heat_kernel_convolution(w_list[i] * mu_bar, sigma, image_shape)
            d_i = v_list[i] * conv_i
            d_list.append(d_i)
        
        # Weighted geometric mean => new mu_bar
        # mu_bar = product_{i=1..k} ( d_i ^ alpha_i )
        mu_bar[:] = 1.0
        for i in range(k):
            alpha_i = alpha_list[i]
            # d_i^alpha_i
            mu_bar *= d_list[i] ** alpha_i
        sum_mb = mu_bar.sum()
        if sum_mb < 1e-16:
            # fallback to uniform if degenerate
            mu_bar[:] = 1.0/n
        else:
            mu_bar /= sum_mb
        
        # -------------- Entropy Sharpening Step --------------
        if sharpen_entropy is not None:
            # clamp mu_bar's entropy to sharpen_entropy
            mu_bar = entropic_sharpening(mu_bar, a, sharpen_entropy, tol=1e-7, max_iter=100)
        # -----------------------------------------------------
        
        # Projection onto constraints that all row marginals = mu_bar
        # => update v_i
        for i in range(k):
            d_i = np.maximum(d_list[i], 1e-16)
            v_list[i] = v_list[i] * (mu_bar / d_i)
        
        # Check for convergence
        diff = np.linalg.norm(mu_bar - mu_bar_old, 1)
        if verbose and (it%10==0 or diff<tol):
            print(f"[Iter {it}] barycenter L1 change={diff:.3e}")
        if diff < tol:
            break
    
    return mu_bar, v_list, w_list


def displacement_interpolation(mu0, mu1, t, image_shape, sigma, tol=1e-6, max_iter=200, verbose=False):
    """
    Computes the displacement interpolation (Wasserstein interpolation) between two distributions
    mu0 and mu1 for a given t in [0,1] using the barycenter approach.
    
    Parameters:
      mu0, mu1   : 1D numpy arrays (flattened) representing the two input distributions.
      t          : Interpolation parameter between 0 and 1.
      image_shape: Tuple representing the image dimensions, e.g., (128, 128).
      sigma      : Standard deviation for the Gaussian convolution (heat kernel approximation).
      tol        : Tolerance for the barycenter iteration convergence.
      max_iter   : Maximum number of iterations for barycenter computation.
      verbose    : If True, print convergence details.
      
    Returns:
      mu_t       : The interpolated distribution as a 1D numpy array (flattened).
    """
    # For two distributions, set barycenter weights [1-t, t]
    alpha_list = [1 - t, t]
    mu_list = [mu0, mu1]
    
    # Call the wasserstein_barycenter function (from our barycenter module)
    mu_bar, _, _ = wasserstein_barycenter(mu_list, alpha_list, image_shape, sigma, tol, max_iter, verbose)
    return mu_bar

def wasserstein_propagation(vertices, edges, fixed, mu_fixed, image_shape, sigma, tol=1e-6, max_iter=100, verbose=True):
    """
    Propagates distributions on a graph using Wasserstein propagation via iterated Bregman projections.
    
    Parameters:
      vertices   : List of vertex indices.
      edges      : List of tuples (v, w, weight) representing directed edges from v to w.
      fixed      : List or set of vertices that have fixed distributions.
      mu_fixed   : Dictionary mapping each fixed vertex to its distribution (flattened, normalized).
      image_shape: Tuple (height, width) specifying the grid dimensions.
      sigma      : Standard deviation for the Gaussian (heat kernel approximation).
      tol        : Convergence tolerance (based on L1 change on unknown vertices).
      max_iter   : Maximum number of iterations.
      verbose    : If True, prints diagnostic messages.
      
    Returns:
      mu         : Dictionary mapping each vertex to its propagated distribution.
    """
    n = np.prod(image_shape)
    # Uniform area weights (for images, each pixel has weight 1/n)
    a = np.full(n, 1.0 / n)
    
    # Initialize distributions for all vertices.
    mu = {}
    for v in vertices:
        if v in fixed:
            mu[v] = mu_fixed[v]
        else:
            # Start with a uniform distribution.
            temp = np.ones(n)
            mu[v] = temp / np.dot(a, temp)
    
    # Initialize scaling factors for each edge.
    # For each edge (v, w, weight), we store scaling factors in a dictionary.
    edge_scaling = {}
    for (v, w, weight) in edges:
        edge_scaling[(v, w)] = {"v": np.ones(n), "w": np.ones(n)}
    
    # Build a neighbor mapping: for each vertex, store incident edges.
    # For each vertex, store a list of tuples (neighbor, edge, direction, weight)
    # 'out' means edge (v, nbr) leaving v; 'in' means edge (nbr, v) entering v.
    neighbors = {v: [] for v in vertices}
    for (v, w, weight) in edges:
        neighbors[v].append((w, (v, w), 'out', weight))
        neighbors[w].append((v, (v, w), 'in', weight))
    
    # Main propagation loop.
    for it in range(max_iter):
        # Save previous distributions for unknown vertices to monitor convergence.
        mu_prev = {v: mu[v].copy() for v in vertices if v not in fixed}
        
        for v in vertices:
            if v in fixed:
                # For fixed vertices, enforce the given distribution.
                mu[v] = mu_fixed[v]
                # Update scaling factors on incident edges.
                for (nbr, edge, direction, weight) in neighbors[v]:
                    if direction == 'in':  # edge (nbr, v)
                        v_factor = edge_scaling[edge]["v"]
                        conv_val = heat_kernel_convolution(v_factor * a, sigma, image_shape)
                        conv_val[conv_val == 0] = 1e-16
                        edge_scaling[edge]["w"] = mu[v] / conv_val
                    elif direction == 'out':  # edge (v, nbr)
                        w_factor = edge_scaling[edge]["w"]
                        conv_val = heat_kernel_convolution(w_factor * a, sigma, image_shape)
                        conv_val[conv_val == 0] = 1e-16
                        edge_scaling[edge]["v"] = mu[v] / conv_val
            else:
                # For an unknown vertex, combine information from all incident edges.
                d_list = []    # list to store d_e for each incident edge
                weight_list = []  # corresponding edge weights
                for (nbr, edge, direction, weight) in neighbors[v]:
                    if direction == 'in':  # edge (nbr, v)
                        # d_e = (w scaling) * H_t(a * (v scaling))
                        v_factor = edge_scaling[edge]["v"]
                        conv_val = heat_kernel_convolution(v_factor * a, sigma, image_shape)
                        d_e = edge_scaling[edge]["w"] * conv_val
                    elif direction == 'out':  # edge (v, nbr)
                        w_factor = edge_scaling[edge]["w"]
                        conv_val = heat_kernel_convolution(w_factor * a, sigma, image_shape)
                        d_e = edge_scaling[edge]["v"] * conv_val
                    d_list.append(np.maximum(d_e, 1e-16))  # avoid zeros
                    weight_list.append(weight)
                # Compute the weighted geometric mean:
                omega = np.sum(weight_list)
                mu_v_new = np.ones(n)
                for d_e, w_e in zip(d_list, weight_list):
                    mu_v_new *= d_e ** (w_e / omega)
                # Normalize the new distribution so that a^T mu = 1.
                mu[v] = mu_v_new / np.dot(a, mu_v_new)
                
                # Update scaling factors on all incident edges.
                for (nbr, edge, direction, weight) in neighbors[v]:
                    if direction == 'in':  # edge (nbr, v)
                        v_factor = edge_scaling[edge]["v"]
                        conv_val = heat_kernel_convolution(v_factor * a, sigma, image_shape)
                        d_e = edge_scaling[edge]["w"] * np.maximum(conv_val, 1e-16)
                        edge_scaling[edge]["w"] = edge_scaling[edge]["w"] * (mu[v] / d_e)
                    elif direction == 'out':  # edge (v, nbr)
                        w_factor = edge_scaling[edge]["w"]
                        conv_val = heat_kernel_convolution(w_factor * a, sigma, image_shape)
                        d_e = edge_scaling[edge]["v"] * np.maximum(conv_val, 1e-16)
                        edge_scaling[edge]["v"] = edge_scaling[edge]["v"] * (mu[v] / d_e)
        
        # Compute convergence error for unknown vertices.
        err = 0
        count = 0
        for v in vertices:
            if v not in fixed:
                err += np.linalg.norm(mu[v] - mu_prev[v], 1)
                count += 1
        avg_err = err / count if count > 0 else 0
        if verbose:
            print(f"Iteration {it+1}: Average L1 change on unknown vertices = {avg_err:.2e}")
        if count > 0 and avg_err < tol:
            if verbose:
                print("Propagation converged.")
            break
            
    return mu

def modified_heat_kernel_convolution(f, sigma, image_shape, compatibility, tau, gamma, N_v):
    """
    Applies a modified heat kernel convolution to f.
    
    The modification uses a diagonal scaling:
      D = exp(-tau * compatibility / (gamma * N_v))
    and computes the result as: 
      result = D * H_t( f * D )
      
    Parameters:
      f            : 1D numpy array.
      sigma        : Gaussian std parameter.
      image_shape  : Shape tuple for target grid.
      compatibility: 1D numpy array of compatibility values at vertex v.
      tau          : Parameter controlling the strength of the compatibility penalty.
      gamma        : Regularization parameter.
      N_v          : Valence (number of neighbors) of vertex v.
      
    Returns:
      Modified convolution result (flattened).
    """
    D = np.exp(-tau * compatibility / (gamma * N_v))
    f_mod = f * D
    conv = gaussian_filter(f_mod.reshape(image_shape), sigma=sigma).flatten()
    return conv * D

def soft_maps_propagation(vertices, edges, fixed, mu_fixed, 
                          compatibility, tau, gamma, image_shape, sigma, 
                          tol=1e-6, max_iter=100, verbose=True):
    """
    Computes a measure-valued map (soft map) from source vertices (domain M0)
    to a target domain M by propagating distributions using a modified
    Wasserstein propagation algorithm.
    
    Parameters:
      vertices    : List of vertex indices (source graph vertices).
      edges       : List of tuples (v, w, weight) representing directed edges.
      fixed       : List/set of vertices in M0 with fixed distributions.
      mu_fixed    : Dictionary mapping each fixed vertex to its distribution 
                    (flattened, normalized).
      compatibility: Dictionary mapping each vertex v (in M0) to a 1D numpy array 
                     of compatibility values on the target domain.
      tau         : Parameter for the compatibility penalty.
      gamma       : Regularization parameter.
      image_shape : Tuple (height, width) of the target domain grid.
      sigma       : Gaussian standard deviation for the heat kernel approximation.
      tol         : Tolerance for convergence (L1 change on unknown vertices).
      max_iter    : Maximum number of iterations.
      verbose     : If True, prints progress messages.
      
    Returns:
      mu          : Dictionary mapping each vertex v in M0 to its propagated 
                    distribution (flattened, normalized).
    """
    n = np.prod(image_shape)
    a = np.full(n, 1.0/n)  # uniform area weights on target domain
    
    # Initialize distributions on each vertex.
    mu = {}
    for v in vertices:
        if v in fixed:
            mu[v] = mu_fixed[v]
        else:
            temp = np.ones(n)
            mu[v] = temp / np.dot(a, temp)
    
    # Initialize scaling factors for each edge.
    edge_scaling = {}
    for (v, w, weight) in edges:
        edge_scaling[(v, w)] = {"v": np.ones(n), "w": np.ones(n)}
    
    # Build neighbor mapping.
    neighbors = {v: [] for v in vertices}
    for (v, w, weight) in edges:
        neighbors[v].append((w, (v, w), 'out', weight))
        neighbors[w].append((v, (v, w), 'in', weight))
    
    # Compute valence for each vertex (number of incident edges).
    valence = {v: len(neighbors[v]) for v in vertices}
    
    # Main propagation loop.
    for it in range(max_iter):
        mu_prev = {v: mu[v].copy() for v in vertices if v not in fixed}
        
        for v in vertices:
            if v in fixed:
                mu[v] = mu_fixed[v]
                # For fixed vertices, update scaling factors on incident edges (using standard kernel)
                for (nbr, edge, direction, weight) in neighbors[v]:
                    if direction == 'in':
                        v_factor = edge_scaling[edge]["v"]
                        conv_val = heat_kernel_convolution(v_factor * a, sigma, image_shape)
                        conv_val[conv_val == 0] = 1e-16
                        edge_scaling[edge]["w"] = mu[v] / conv_val
                    elif direction == 'out':
                        w_factor = edge_scaling[edge]["w"]
                        conv_val = heat_kernel_convolution(w_factor * a, sigma, image_shape)
                        conv_val[conv_val == 0] = 1e-16
                        edge_scaling[edge]["v"] = mu[v] / conv_val
            else:
                # For an unknown vertex, use the compatibility function to modify updates.
                d_list = []     # store modified d_e values for each incident edge
                weight_list = []  # corresponding edge weights
                # Use the compatibility vector for vertex v and its valence.
                comp_v = compatibility[v]
                N_v = valence[v] if valence[v] > 0 else 1
                for (nbr, edge, direction, weight) in neighbors[v]:
                    if direction == 'in':  # edge (nbr, v)
                        # Compute d_e = (w scaling) * modified convolution of (v scaling * a)
                        v_factor = edge_scaling[edge]["v"]
                        conv_val = modified_heat_kernel_convolution(v_factor * a, sigma, image_shape, comp_v, tau, gamma, N_v)
                        d_e = edge_scaling[edge]["w"] * np.maximum(conv_val, 1e-16)
                    elif direction == 'out':  # edge (v, nbr)
                        w_factor = edge_scaling[edge]["w"]
                        conv_val = modified_heat_kernel_convolution(w_factor * a, sigma, image_shape, comp_v, tau, gamma, N_v)
                        d_e = edge_scaling[edge]["v"] * np.maximum(conv_val, 1e-16)
                    d_list.append(d_e)
                    weight_list.append(weight)
                # Compute the weighted geometric mean at vertex v.
                omega = np.sum(weight_list)
                mu_v_new = np.ones(n)
                for d_e, w_e in zip(d_list, weight_list):
                    mu_v_new *= d_e ** (w_e / omega)
                mu[v] = mu_v_new / np.dot(a, mu_v_new)
                
                # Update scaling factors on incident edges.
                for (nbr, edge, direction, weight) in neighbors[v]:
                    if direction == 'in':
                        v_factor = edge_scaling[edge]["v"]
                        conv_val = modified_heat_kernel_convolution(v_factor * a, sigma, image_shape, comp_v, tau, gamma, N_v)
                        d_e = edge_scaling[edge]["w"] * np.maximum(conv_val, 1e-16)
                        edge_scaling[edge]["w"] = edge_scaling[edge]["w"] * (mu[v] / d_e)
                    elif direction == 'out':
                        w_factor = edge_scaling[edge]["w"]
                        conv_val = modified_heat_kernel_convolution(w_factor * a, sigma, image_shape, comp_v, tau, gamma, N_v)
                        d_e = edge_scaling[edge]["v"] * np.maximum(conv_val, 1e-16)
                        edge_scaling[edge]["v"] = edge_scaling[edge]["v"] * (mu[v] / d_e)
                        
        # Convergence check on unknown vertices.
        err = 0
        count = 0
        for v in vertices:
            if v not in fixed:
                err += np.linalg.norm(mu[v] - mu_prev[v], 1)
                count += 1
        avg_err = err / count if count > 0 else 0
        if verbose:
            print(f"Iteration {it+1}: Average L1 change on unknown vertices = {avg_err:.2e}")
        if count > 0 and avg_err < tol:
            if verbose:
                print("Soft maps propagation converged.")
            break
            
    return mu

def entropy(mu, a):
    """
    Computes the differential entropy of a probability distribution mu,
    with area weights a, using the formula
       H(mu) = -∑_i a_i * mu_i * log(mu_i)
    A small constant is used to avoid log(0).
    
    Parameters:
      mu : 1D numpy array, assumed normalized (a^T mu = 1).
      a  : 1D numpy array of area weights.
    
    Returns:
      Entropy (a scalar).
    """
    eps = 1e-16
    return -np.sum(a * mu * np.log(np.maximum(mu, eps)))

def entropic_sharpening(mu, a, H0, tol=1e-6, max_iter=50):
    """
    Sharpens a distribution mu by raising it to a power beta, so that
    the resulting distribution (after re-normalization) has entropy equal to H0.
    If mu already has entropy less than or equal to H0, no sharpening is applied.
    
    Parameters:
      mu      : 1D numpy array, the input distribution (assumed normalized: a^T mu = 1).
      a       : 1D numpy array of area weights.
      H0      : Desired entropy bound.
      tol     : Tolerance for the bisection solver.
      max_iter: Maximum iterations for the bisection method.
      
    Returns:
      mu_sharp: The sharpened (and re-normalized) distribution.
    """
    current_entropy = entropy(mu, a)
    # If the current entropy is already below or equal to H0, do nothing.
    if current_entropy <= H0:
        return mu
    
    # We want to find beta >= 1 such that the entropy of the re-normalized mu^beta is H0.
    # Let mu_beta = (mu ** beta) normalized so that a^T mu_beta = 1.
    def F(beta):
        mu_beta = mu ** beta
        mu_beta = mu_beta / np.dot(a, mu_beta)  # re-normalize
        return entropy(mu_beta, a) - H0

    # Since mu is smooth, F(1) = entropy(mu)-H0 > 0.
    beta_low = 1.0
    F_low = F(beta_low)
    
    # Find an upper bound beta_high so that F(beta_high) < 0.
    beta_high = 2.0
    while F(beta_high) > 0:
        beta_high *= 2.0
        if beta_high > 1e6:
            # In case we cannot find a suitable beta_high, break.
            break
    F_high = F(beta_high)
    
    # Bisection to solve F(beta) = 0
    for it in range(max_iter):
        beta_mid = (beta_low + beta_high) / 2.0
        F_mid = F(beta_mid)
        if np.abs(F_mid) < tol:
            beta = beta_mid
            break
        if F_mid > 0:
            beta_low = beta_mid
        else:
            beta_high = beta_mid
        beta = (beta_low + beta_high) / 2.0
    else:
        print("Warning: entropic sharpening did not converge within the maximum iterations.")
    
    # Compute and return the sharpened distribution:
    mu_sharp = mu ** beta
    mu_sharp = mu_sharp / np.dot(a, mu_sharp)
    return mu_sharp
