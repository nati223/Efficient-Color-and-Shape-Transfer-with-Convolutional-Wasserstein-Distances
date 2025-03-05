import numpy as np
import cv2
import cv2.ximgproc
from cv2.ximgproc import jointBilateralFilter

def cross_bilateral_batch_opencv(batch, guidance, d=5, sigmaColor=25, sigmaSpace=10):
    """
    Applies OpenCV's jointBilateralFilter to each distribution in 'batch',
    using 'guidance' as the separate image that controls edge-aware range differences.

    Parameters
    ----------
    batch : np.ndarray, shape (B, H, W)
        The data to be filtered, for each distribution b=0..B-1.
    guidance : np.ndarray, shape (H, W)
        Single-channel guidance image for cross-bilateral. Must be same size as each dist.
    d : int
        Diameter of each pixel neighborhood.
    sigmaColor : float
        Filter sigma in the color (range) space.
    sigmaSpace : float
        Filter sigma in the coordinate (spatial) space.

    Returns
    -------
    filtered : np.ndarray, shape (B, H, W)
        The filtered batch.
    """

    B, H, W = batch.shape
    # OpenCV typically expects single-channel float 32 or 8-bit for the guidance
    guide_f32 = guidance.astype(np.float32)

    out = np.zeros_like(batch, dtype=np.float64)

    for i in range(B):
        src_f32 = batch[i].astype(np.float32)
        # Perform joint bilateral
        # The 'dst' is returned as float32
        filtered_f32 = cv2.ximgproc.jointBilateralFilter(
            joint=guide_f32,
            src=src_f32,
            d=d,
            sigmaColor=sigmaColor,
            sigmaSpace=sigmaSpace
        )
        out[i] = filtered_f32  # convert to float64 if needed
    return out

def cross_bilateral_batch_opencv(batch, guidance, d=5, sigmaColor=25, sigmaSpace=10):
    """
    Applies OpenCV's jointBilateralFilter to each distribution in 'batch',
    using 'guidance' as the separate image that controls edge-aware range differences.

    Parameters
    ----------
    batch : np.ndarray, shape (B, H, W)
        The data to be filtered, for each distribution b=0..B-1.
    guidance : np.ndarray, shape (H, W)
        Single-channel guidance image for cross-bilateral. Must be same size as each dist.
    d : int
        Diameter of each pixel neighborhood.
    sigmaColor : float
        Filter sigma in the color (range) space.
    sigmaSpace : float
        Filter sigma in the coordinate (spatial) space.

    Returns
    -------
    filtered : np.ndarray, shape (B, H, W)
        The filtered batch.
    """

    B, H, W = batch.shape
    # OpenCV typically expects single-channel float 32 or 8-bit for the guidance
    guide_f32 = guidance.astype(np.float32)

    out = np.zeros_like(batch, dtype=np.float64)

    for i in range(B):
        src_f32 = batch[i].astype(np.float32)
        # Perform joint bilateral
        # The 'dst' is returned as float32
        filtered_f32 = cv2.ximgproc.jointBilateralFilter(
            joint=guide_f32,
            src=src_f32,
            d=d,
            sigmaColor=sigmaColor,
            sigmaSpace=sigmaSpace
        )
        out[i] = filtered_f32  # convert to float64 if needed
    return out

def cross_bilateral_wasserstein_barycenter_opencv(
    distributions,
    weights=None,
    max_iterations=100,
    d=5,
    sigmaColor=25.0,
    sigmaSpace=10.0,
    stop_threshold=1e-7,
    verbose=False
):
    """
    Replaces the isotropic heat-kernel convolution with OpenCV's jointBilateralFilter,
    using the evolving barycenter as the guidance image each iteration.
    """

    # Convert to float64 for stable math
    distributions = np.array(distributions, dtype=np.float64)
    B, H, W = distributions.shape

    # Weights
    if weights is None:
        weights = np.ones(B, dtype=np.float64) / B

    # Initialize barycenter as uniform or average
    barycenter = np.ones((H, W), dtype=np.float64) / (H*W)

    # Initialize scaling factors v, w
    v = np.ones_like(distributions)
    w = np.ones_like(distributions)

    for iteration in range(max_iterations):
        # 1) Kv = cross_bilateral_batch_opencv(v, barycenter, ...)
        Kv = cross_bilateral_batch_opencv(
            batch=v,
            guidance=barycenter,  # using the barycenter as guidance
            d=d,
            sigmaColor=sigmaColor,
            sigmaSpace=sigmaSpace
        )

        # 2) w_i = bary / Kv_i
        for i in range(B):
            w[i] = barycenter / np.maximum(Kv[i], 1e-30)

        # 3) Kw = cross_bilateral_batch_opencv(w, barycenter, ...)
        Kw = cross_bilateral_batch_opencv(
            batch=w,
            guidance=barycenter,
            d=d,
            sigmaColor=sigmaColor,
            sigmaSpace=sigmaSpace
        )

        # 4) v_i = dist_i / Kw_i
        for i in range(B):
            v[i] = distributions[i] / np.maximum(Kw[i], 1e-30)

        # 5) Kv again, for updated barycenter
        Kv = cross_bilateral_batch_opencv(
            batch=v,
            guidance=barycenter,
            d=d,
            sigmaColor=sigmaColor,
            sigmaSpace=sigmaSpace
        )

        # 6) barycenter = exp( sum_i weights_i log(Kv_i) )
        logKv = np.zeros((H, W), dtype=np.float64)
        for i in range(B):
            logKv += weights[i] * np.log(np.maximum(Kv[i], 1e-30))
        new_bary = np.exp(logKv)

        # 7) Check for convergence
        diff = np.linalg.norm(new_bary - barycenter, 1)
        barycenter = new_bary

        if verbose:
            print(f"[Iteration {iteration}] diff={diff:.3e}")

        if diff < stop_threshold:
            if verbose:
                print("Converged!")
            break

    return barycenter