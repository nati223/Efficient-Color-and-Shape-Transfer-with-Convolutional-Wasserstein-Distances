import numpy as np
from scipy.optimize import root_scalar


def compute_entropy_and_mass(b, stability_threshold=1e-30):
    """
    Compute the entropy H(b) = - sum( b * ln(b) )
    and total mass sum(b) for a 2D array b.
    """
    clipped = np.maximum(b, stability_threshold)
    entropy = -np.sum(clipped * np.log(clipped))
    total_mass = np.sum(b)
    return entropy, total_mass

def compute_H0(distributions):
    entropies = []
    for i in range(len(distributions)):
        Hi, mi = compute_entropy_and_mass(distributions[i])
        entropies.append(Hi)

    return max(entropies)

def sharpening_find_beta(
    b,
    H0,
    stability_threshold=1e-30,
    bracket=(0.0, 50.0),
    max_bracket=50.0,
    method='bisect',
    x0=1.0,
    max_iter=50,
    tol=1e-8
):
    """
    Returns beta >= 0 such that H(b^beta) + sum(b^beta) = H0 + 1, if possible.
    Otherwise returns 1.0 if no suitable beta.

    Parameters
    ----------
    b : 2D array
        The current barycenter or distribution we want to sharpen.
    H0 : float
        Entropy bound: We want H(b^beta)+sum(b^beta) <= H0 + 1.
    stability_threshold : float
        For numerical stability in logs and divisions.
    bracket : tuple (low, high)
        Initial bracket for bracket-based methods.
    max_bracket : float
        We'll expand 'high' if needed, up to this limit.
    method : str
        One of {'bisect', 'brentq', 'ridder', 'newton', 'secant'}, etc.
    x0 : float
        Initial guess for derivative-based methods like 'newton', 'secant'.
    max_iter : int
        Maximum iterations in the root-finder.
    tol : float
        Tolerance on the solution.

    Returns
    -------
    beta : float
        If the function doesn't find a root, returns 1.0.
        If it does, returns the solution in [0,inf).
    """

    def f(beta):
        # b^beta
        safe_b = np.maximum(b, stability_threshold)
        bbeta = np.power(safe_b, beta)
        # sum(b^beta)
        mass = bbeta.sum()
        # H(b^beta) = - beta sum( b^beta ln(b) )
        ln_b = np.log(safe_b)
        entropy = -beta * np.sum(bbeta * ln_b)
        return entropy + mass - (H0 + 1)

    def fprime(beta):
        # derivative of f w.r.t beta
        # f'(beta) = derivative of [entropy + mass] wrt beta
        # mass' = sum( b^beta ln b )
        # entropy' -> final closed form = - beta sum(b^beta (ln b)^2)
        safe_b = np.maximum(b, stability_threshold)
        bbeta = np.power(safe_b, beta)
        ln_b = np.log(safe_b)
        sum_bbeta_ln2 = np.sum(bbeta * (ln_b**2))
        return -beta * sum_bbeta_ln2

    # Quick check: if f(1) <= 0 => no sharpening needed
    val_at_1 = f(1.0)
    if val_at_1 <= 0.0:
        return 1.0

    # For bracket-based methods, do the bracket expansion logic
    if method in ('bisect', 'brentq', 'ridder'):
        low, high = bracket
        val_low, val_high = f(low), f(high)

        # If f(low) > 0 => no solution in [low,1], fallback
        if val_low > 0:
            return 1.0

        # expand if needed
        while val_high < 0 and high < max_bracket:
            high *= 2.0
            val_high = f(high)
        if val_high < 0:
            # no sign change in [low, max_bracket]
            return 1.0

        # Now we can solve in [low, high]
        sol = root_scalar(f, bracket=(low, high), method=method, maxiter=max_iter, rtol=tol)
        if sol.converged:
            return max(sol.root, 0.0)
        else:
            return 1.0

    elif method in ('newton', 'secant'):
        # derivative-based method => we pass fprime if 'newton'
        kwargs = dict(x0=x0, maxiter=max_iter, rtol=tol)
        if method == 'newton':
            sol = root_scalar(f, fprime=fprime, method='newton', **kwargs)
        else:  # 'secant'
            sol = root_scalar(f, x1=x0*1.1, method='secant', **kwargs)  # you can adapt x1

        if sol.converged and sol.root >= 0.0:
            return sol.root
        else:
            # fallback
            return 1.0

    else:
        raise ValueError(f"Unknown method '{method}'; must be bracket-based or derivative-based.")


def entropic_sharpening(b, H0, stability_threshold=1e-30):
    """
    If H(b) + sum(b) > H0 + 1, exponentiate b by some beta < 1 to reduce entropy.

    Implements the logic from Algorithm 3 in the Solomon et al. paper,
    but with a SciPy-based root find for the exponent beta.
    """
    curr_entropy, curr_mass = compute_entropy_and_mass(b, stability_threshold=stability_threshold)
    diff = (curr_entropy + curr_mass) - (H0 + 1.0)
    
    # Print how far above the threshold we are
    # print(f"[Sharpening]  H(b)+sum(b)={curr_entropy+curr_mass:.4f}, "
    #       f"H0+1={H0+1:.4f}, diff={diff:.4e}")
    
    # Compute current H(b)+sum(b)
    curr_entropy, curr_mass = compute_entropy_and_mass(b, stability_threshold=stability_threshold)
    if diff < 0:
        # Already satisfies the constraint => no change
        # print("[Sharpening]  No sharpening needed.")
        return b

    # Otherwise solve for beta
    beta = sharpening_find_beta(b, H0, stability_threshold=stability_threshold, method='newton')
    if abs(beta - 1.0) < 1e-12:
        # print(f"[Sharpening]  Beta ~ 1 => no major exponent change.")
        # means no solution or no need to sharpen
        return b

    # Exponentiate
    safe_b = np.maximum(b, stability_threshold)
    b_sharp = np.power(safe_b, beta)
    # print(f"[Sharpening]  Sharpening with beta={beta:.6f}")
    return b_sharp


def convolutional_wasserstein_barycenter(
    distributions,
    gamma,
    weights=None,
    max_iterations=10000,
    stop_threshold=1e-9,
    stability_threshold=1e-30,
    verbose=False,
    log_output=False,
    warn=True,
    H0 = None,
    entropy_sharpening=False, # To be implemented
):
    """
    Compute the entropic regularized Wasserstein barycenter of a collection of 2D distributions.

    Parameters:
    ----------
    distributions : list or np.ndarray
        A collection of 2D probability distributions (images).
    gamma : float
        Entropic regularization parameter.
    weights : np.ndarray, optional
        Weights for each distribution, default is uniform.
    max_iterations : int, optional
        Maximum number of iterations, default is 10000.
    stop_threshold : float, optional
        Convergence threshold, default is 1e-9.
    stability_threshold : float, optional
        Small stability value to prevent numerical issues, default is 1e-30.
    verbose : bool, optional
        If True, print progress during iterations.
    log_output : bool, optional
        If True, return additional log details.
    warn : bool, optional
        If True, warns if the algorithm does not converge.

    Returns:
    -------
    barycenter : np.ndarray
        The computed Wasserstein barycenter.
    v : np.ndarray
        Scaling factors corresponding to the input distributions.
    w : np.ndarray
        Scaling factors corresponding to the convolution operation.
    log : dict (optional)
        Contains iteration errors if log_output is True.
    """

    distributions = np.array(distributions)
    num_distributions, height, width = distributions.shape

    if weights is None:
        weights = np.full(num_distributions, 1 / num_distributions)
    else:
        assert len(weights) == num_distributions, "Weights must match number of distributions."

    if log_output:
        log = {"errors": []}

    barycenter = np.ones((height, width)) / (height * width)
    v = np.ones_like(distributions)  # Scaling factor v in Algorithm 2
    w = np.ones_like(distributions)  # Scaling factor w in Algorithm 2
    error = float("inf")
    prev_error = float("inf")
    
    # Initialize entropy bound H0
    H0 = compute_H0(distributions) if H0 is None else H0

    # Construct the convolution kernel using Gaussian approximation
    x = np.linspace(0, 1, height)
    X, Y = np.meshgrid(x, x)
    heat_kernel = np.exp(-((X - Y) ** 2) / gamma)

    def convolve_images(images):
        """Applies separable convolution using kernel_x and kernel_y."""
        intermediate = np.einsum("ij,bjk->bik", heat_kernel, images)
        return np.einsum("ij,bkj->bki", heat_kernel, intermediate)

    Kv = convolve_images(v)

    for iteration in range(max_iterations):
        w = barycenter[None, :, :] / Kv
        Kw = convolve_images(w)
        v = distributions / Kw
        Kv = convolve_images(v)

        # Compute barycenter update
        log_Kv = np.log(np.maximum(Kv, stability_threshold))
        barycenter = np.exp(np.sum(weights[:, None, None] * log_Kv, axis=0))
        
        # Apply entropy sharpening if enabled
        if entropy_sharpening:
            prev_barycenter = barycenter
            barycenter = entropic_sharpening(barycenter, H0)
        
        prev_error = error
        error = np.sum(np.std(w * Kv, axis=0))
        
        if (error > 1.1 * prev_error):
            return prev_barycenter, v, w, heat_kernel
            
        # Check for convergence
        if iteration % 10 == 9:

            if log_output:
                log["errors"].append(error)

            if verbose:
                if iteration % 200 == 0:
                    print(f"{'Iter':<5} | {'Error':<12}\n" + "-" * 19)
                print(f"{(1 + iteration):<5} | {error:.8e}")

            if error < stop_threshold:
                break
    else:
        if warn:
            print("Warning: Convolutional Sinkhorn did not converge. Consider increasing max_iterations or regularization.")

    if log_output:
        log["iterations"] = iteration
        return barycenter, v, w, log
    
    return barycenter, v, w, heat_kernel